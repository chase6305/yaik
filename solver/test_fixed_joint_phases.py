"""Branch-aligned phase recovery keeps original fixed coordinates and FK gates."""
from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.fixed_joint_phases import fixed_joint_branch_centers, joint_is_periodic
from codegen.skeleton.skeleton_tree import SkeletonTree
from fk.robots import DHEntry, RobotDescription, RobotAuxiliaryData
from python_run_import import fixed_joint_phase_aliases
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionDegenerateRecord, SolutionMethod, VariableSolution
from fk.fk_equations import Pz

ROOT = Path(__file__).resolve().parents[1]


def phase_tree():
    angle, displacement = sp.symbols('angle displacement')
    robot = RobotDescription('phase_endpoint')
    robot.unknowns = [Unknown(angle, UnknownType.Revolute.name),
                      Unknown(displacement, UnknownType.Translational.name)]
    robot.unknown_as_parameter_more_dof = {angle}
    robot.dh_params = [DHEntry(0, 0, 0, angle), DHEntry(0, 0, displacement, 0)]
    record = SolutionDegenerateRecord.record_variable_value({angle: [sp.pi]})
    regular = VariableSolution.make_explicit_solution(displacement, [Pz], SolutionMethod.OneVariableAlgebra.name,
        argument_valid_checkers=[sp.Abs(sp.sin(angle)) > sp.Float(1e-6)], degenerate_record=record)
    tree = SkeletonTree([regular], robot)
    branch = VariableSolution.make_explicit_solution(displacement, [Pz], SolutionMethod.OneVariableAlgebra.name)
    tree.append_branched_solution([branch], tree.root, 1)
    return tree


class TestFixedJointPhases(TestCase):
    def test_aliases_preserve_phase_and_add_a_combined_fixed_parameter_case(self):
        for angle in (-math.pi, -math.pi-1e-8, -math.pi+1e-8, 3*math.pi):
            aliases = fixed_joint_phase_aliases((angle,), [(0, math.pi)])
            self.assertEqual(len(aliases), 1)
            self.assertLess(abs(math.sin(aliases[0][0])-math.sin(angle)), 1e-14)
            self.assertLess(abs(math.cos(aliases[0][0])-math.cos(angle)), 1e-14)
        self.assertEqual(fixed_joint_phase_aliases((math.pi,), [(0, math.pi)]), [])
        self.assertEqual(fixed_joint_phase_aliases((.2,), [(0, math.pi)]), [])
        self.assertEqual(len(fixed_joint_phase_aliases((-math.pi,-math.pi), [(0,math.pi),(1,math.pi)])), 3)
        huge = 1e308
        center = math.atan2(math.sin(huge), math.cos(huge))
        alias = fixed_joint_phase_aliases((huge,), [(0,center)])[0][0]
        self.assertLess(abs(math.sin(alias)-math.sin(huge)), 1e-14)

    def test_only_periodic_models_and_implemented_centers_are_enabled(self):
        tree = phase_tree(); symbol = tree.robot.unknowns[0].symbol
        self.assertTrue(joint_is_periodic(tree.robot, symbol))
        self.assertEqual(fixed_joint_branch_centers(tree), [(symbol, float(sp.pi))])
        tree.robot.dh_params[0].d = symbol
        self.assertFalse(joint_is_periodic(tree.robot, symbol))
        self.assertEqual(fixed_joint_branch_centers(tree), [])
        tree.robot.dh_params[0].d = 0
        ratio = sp.Symbol('ratio', integer=True)
        tree.robot.dh_params[0].theta = ratio*symbol
        self.assertFalse(joint_is_periodic(tree.robot, symbol))
        self.assertGreater(abs(math.cos(.3*1.2)-math.cos((.3+2*math.pi)*1.2)), .9)
        tree.robot.dh_params[0].theta = 2*symbol
        self.assertTrue(joint_is_periodic(tree.robot, symbol))
        tree.robot.dh_params[0].theta = symbol
        tree.robot.auxiliary_data = RobotAuxiliaryData()
        angle = ratio*symbol
        tree.robot.auxiliary_data.pre_transform_sp = sp.Matrix([
            [sp.cos(angle),-sp.sin(angle),0,0],[sp.sin(angle),sp.cos(angle),0,0],
            [0,0,1,0],[0,0,0,1]])
        self.assertFalse(joint_is_periodic(tree.robot, symbol))
        tree.robot.auxiliary_data.pre_transform_sp = sp.eye(4)
        tree.robot.auxiliary_data.pre_transform_sp[0,3] = symbol
        self.assertFalse(joint_is_periodic(tree.robot, symbol))

    def test_python_uses_original_fixed_joint_in_every_checked_candidate(self):
        tree = phase_tree(); scope = {'__name__': 'phase_regression'}
        with redirect_stdout(StringIO()):
            exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
        for angle in (-math.pi, -math.pi-1e-8, -math.pi+1e-8, 3*math.pi, math.pi):
            q = np.array([angle, .4]); target = np.eye(4)
            target[:2,:2] = [[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]
            target[2,3] = .4
            np.testing.assert_allclose(scope['phase_endpoint_fk'](q), target, atol=1e-15)
            output = scope['phase_endpoint_ik_solve'](target, angle)
            self.assertTrue(output)
            for candidate in output:
                self.assertEqual(candidate[0,0], angle)
                np.testing.assert_allclose(scope['phase_endpoint_fk'](candidate), target, atol=1e-14)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(), 'Requires g++ and Eigen3')
    def test_cpp_recovers_phase_without_extra_random_draws_and_preserves_fixed_values(self):
        with redirect_stdout(StringIO()):
            code = '\n'.join(CodeGenerationCpp(phase_tree(), use_safe_operator=True).generate_code()).split('// Code below for debug')[0]
        main = r'''#include <cassert>
#include <cstdlib>
int main() {
    for(double angle : {-M_PI,-M_PI-1e-8,-M_PI+1e-8,3*M_PI,M_PI}) {
        const auto target=phase_endpoint_ik::computeFK({{angle,.4}});
        const auto output=phase_endpoint_ik::computeIK(target,angle);
        assert(!output.empty());
        for(const auto& q:output) {
            assert(q[0]==angle);
            assert((phase_endpoint_ik::computeFK(q)-target).cwiseAbs().maxCoeff()<1e-12);
        }
    }
    std::srand(123);const int expected=std::rand();std::srand(123);
    const auto aliases=yaik_cpp::fixedJointPhaseAliases(std::array<double,2>{{-M_PI,-M_PI}},
        std::array<std::pair<std::size_t,double>,2>{{{0,M_PI},{1,M_PI}}});
    assert(aliases.size()==3 && std::rand()==expected);
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-phase-') as temporary:
            temporary = Path(temporary)
            source,binary = temporary/'phase.cpp', temporary/'phase'
            source.write_text(code+main)
            result = subprocess.run(['g++','-std=c++14','-O2','-I',str(ROOT),'-I','/usr/include/eigen3',
                                     str(source),'-o',str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run([str(binary)],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
