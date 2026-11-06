"""Unassigned joints must never be returned as numeric IK candidates."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.visitor_numerical_solve import SkeletonNumericalSolution, VisitorNumericalSolution
from fk.fk_equations import Pz
from fk.robots import DHEntry, RobotDescription
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionDegenerateRecord
from solver.test_equation_branches import explicit

ROOT = Path(__file__).resolve().parents[1]


def incomplete_tree(non_root=False):
    x, y, auxiliary, parameter = sp.symbols('x y auxiliary parameter')
    robot = RobotDescription('assigned_prefix' if non_root else 'assigned_root')
    robot.unknowns = [Unknown(symbol, UnknownType.Translational.name, lower_bound=-2e6, upper_bound=2e6)
                      for symbol in (x, y)]
    robot.dh_params = [DHEntry(0, 0, symbol, 0) for symbol in (x, y)]
    robot.symbolic_parameters = {parameter}
    robot.parameters_bound[parameter] = (0., 1.)
    record = SolutionDegenerateRecord.record_all_equations([parameter])
    tree = SkeletonTree([explicit(x, [Pz/2], None if non_root else record),
                         explicit(y, [Pz/2], record if non_root else None)], robot)
    dispatcher = tree.node_list[2] if non_root else tree.root
    tree.append_branched_solution([explicit(auxiliary, [sp.S.One]) if non_root else explicit(x, [Pz])],
                                  dispatcher, 1)
    return tree


class TestCandidateAssignment(TestCase):
    def test_python_rejects_incomplete_paths_and_preserves_large_assigned_values(self):
        for non_root in (False, True):
            tree = incomplete_tree(non_root)
            scope = {'__name__': 'assignment_regression'}
            exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
            for position in (0., .4, 2e5, 2e6):
                target = np.eye(4); target[2, 3] = position
                self.assertEqual(scope[tree.robot.name+'_ik_solve_raw'](target, 0.), [])
                output = scope[tree.robot.name+'_ik_solve_raw'](target, 1.)
                self.assertEqual(len(output), 1)
                np.testing.assert_array_equal(output[0].ravel(), [position/2]*2)
                self.assertTrue(scope[tree.robot.name+'_ik_solve'](target, 1.))

    def test_offline_substitutions_use_assignment_indices_instead_of_magic_values(self):
        x, y = sp.symbols('x y')
        visitor = VisitorNumericalSolution(1)
        visitor.initialize({}, {x, y})
        for legitimate in (0., 1e5, 1e6, -1e6):
            state = SkeletonNumericalSolution([legitimate, 0.], [0, -1])
            self.assertEqual(visitor.value_to_substitute_map(state), {x: legitimate})
        self.assertEqual(visitor.value_to_substitute_map(SkeletonNumericalSolution([np.nan, np.inf], [0, 0])), {})

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(), 'Requires g++ and Eigen3')
    def test_cpp_rejects_incomplete_paths_without_treating_zero_as_assigned(self):
        with redirect_stdout(StringIO()):
            sections = ['\n'.join(CodeGenerationCpp(incomplete_tree(non_root), use_safe_operator=True).generate_code())
                        .split('// Code below for debug')[0] for non_root in (False, True)]
        main = '''#include <cassert>
int main() {
    for (double position : {0., .4, 2e5, 2e6}) {
        Eigen::Matrix4d target=Eigen::Matrix4d::Identity();target(2,3)=position;
        assigned_root_ik::RawIKWorksace root;
        assigned_prefix_ik::RawIKWorksace prefix;
        assigned_root_ik::computeRawIK(target,0.,root);
        assigned_prefix_ik::computeRawIK(target,0.,prefix);
        assert(root.raw_ik_out.empty() && prefix.raw_ik_out.empty());
        assigned_root_ik::computeRawIK(target,1.,root);
        assigned_prefix_ik::computeRawIK(target,1.,prefix);
        assert(root.raw_ik_out.size()==1 && prefix.raw_ik_out.size()==1);
        for (const auto& q : {root.raw_ik_out[0],prefix.raw_ik_out[0]})
            assert(q[0]==position/2 && q[1]==position/2);
        assert(!assigned_root_ik::computeIK(target,1.).empty());
        assert(!assigned_prefix_ik::computeIK(target,1.).empty());
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-assignment-') as temporary:
            temporary = Path(temporary)
            (temporary/'test.cpp').write_text('\n'.join(sections)+main)
            result = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                     str(temporary/'test.cpp'), '-o', str(temporary/'test')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(temporary/'test')], check=True)
