"""Roundoff at inverse-trig endpoints must not remove a reachable pose."""

from contextlib import redirect_stdout
import copy
from io import StringIO
import json
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
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_serialize import read_skeleton_tree
from codegen.skeleton.visitor_numerical_solve import (
    SkeletonNumericalSolution, step_from_explicit_solution, step_from_polynomial_solution)
from fk.robots import DHEntry, RobotDescription
from solver.argument_guards import UNIT_ROUNDOFF_MARGIN, runtime_argument_guards
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod, VariableSolution


ROOT = Path(__file__).resolve().parents[1]
r_11, r_21 = sp.symbols('r_11 r_21')


def endpoint_tree(sine=False, prefix=False, overlap=False, name=None):
    x, z = sp.symbols('x z')
    robot = RobotDescription(name or ('endpoint_'+str(int(sine))+str(int(prefix))+str(int(overlap))))
    robot.unknowns = ([Unknown(z, UnknownType.Translational.name)] if prefix else [])+[Unknown(x)]
    robot.dh_params = ([DHEntry(0, 0, z, 0)] if prefix else [])+[DHEntry(0, 0, 0, x)]
    argument = r_21 if sine else r_11
    first = sp.asin(argument) if sine else sp.acos(argument)
    expressions = [first, sp.pi-first if sine else -first]
    guards = [sp.Abs(argument) <= 1]*2
    if overlap:
        expressions.append(sp.S.Zero)
        guards.append(sp.Abs(argument) > 1)
    solutions = [VariableSolution.make_explicit_solution(x, expressions,
                 SolutionMethod.ArcSin.name if sine else SolutionMethod.ArcCos.name,
                 argument_valid_checkers=guards)]
    if prefix:
        solutions.insert(0, VariableSolution.make_explicit_solution(
            z, [sp.S.Zero], SolutionMethod.OneVariableAlgebra.name))
    return SkeletonTree(solutions, robot)


class TestArgumentGuards(TestCase):
    def test_guard_rounding_only_applies_to_the_inverse_argument(self):
        x, y, denominator = sp.symbols('x y denominator')
        checker = (sp.Abs(x/denominator) <= 1) & (sp.Abs(denominator) > sp.Rational(1, 10**6))
        rounded = runtime_argument_guards([sp.acos(x/denominator)], [checker])[0]
        self.assertTrue(rounded.subs({x: np.nextafter(1., math.inf), denominator: 1}))
        self.assertFalse(rounded.subs({x: 1+1e-6, denominator: 1}))
        self.assertFalse(rounded.subs({x: 0, denominator: 0}))
        self.assertEqual(runtime_argument_guards([sp.acos(y)], [sp.Abs(x) <= 1]), [sp.Abs(x) <= 1])
        for original in (sp.Abs(x) <= sp.Float(1), sp.Ge(sp.Float(1), sp.Abs(x), evaluate=False)):
            rounded = runtime_argument_guards([sp.acos(x)], [original])[0]
            self.assertTrue(rounded.subs(x, np.nextafter(1., math.inf)))
        for original in (sp.Abs(x) < 1, sp.Abs(x) <= sp.Rational(9, 10)):
            self.assertEqual(runtime_argument_guards([sp.acos(x)], [original]), [original])

    def test_python_root_and_prefix_keep_both_endpoint_candidates(self):
        for sine in (False, True):
            for prefix in (False, True):
                tree = endpoint_tree(sine, prefix)
                scope = {'__name__': 'endpoint_test'}
                exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
                for sign in (1., -1.):
                    target = np.eye(4)
                    if sine:
                        target[:2, :2] = [[0., -sign], [sign, 0.]]
                        target[1, 0] = sign*np.nextafter(1., math.inf)
                    else:
                        target[:2, :2] = [[sign, 0.], [0., sign]]
                        target[0, 0] = sign*np.nextafter(1., math.inf)
                    with self.subTest(sine=sine, prefix=prefix, sign=sign):
                        output = scope[tree.robot.name+'_ik_solve'](target)
                        self.assertEqual(len(output), 2)
                        for candidate in output:
                            np.testing.assert_allclose(scope[tree.robot.name+'_fk'](candidate), target,
                                                       atol=1e-14, rtol=0)
                        # Inputs appreciably outside the domain remain rejected.
                        target[1 if sine else 0, 0] = sign*(1+1e-6)
                        self.assertEqual(scope[tree.robot.name+'_ik_solve_raw'](target), [])

    def test_numerical_visitor_matches_endpoint_candidate_generation(self):
        for sine in (False, True):
            tree = endpoint_tree(sine)
            solution = next(n.solution for n in tree.node_list if n.is_solution_node)
            for sign in (-1., 1.):
                output = step_from_explicit_solution(SkeletonNumericalSolution([0.], [-1]),
                    {r_21 if sine else r_11: sign*np.nextafter(1., math.inf)}, solution, 0, {})
                self.assertEqual(len(output), 2)
                expected = sign*math.pi/2 if sine else (0. if sign > 0 else math.pi)
                self.assertLess(abs(math.remainder(output[0].variable_values[0]-expected, 2*math.pi)), 1e-14)
                family = copy.deepcopy(solution)
                family.internal_solution.set_family_critical_equations([(0, 0, 1)], (-math.pi, math.pi))
                family_output = step_from_explicit_solution(SkeletonNumericalSolution([0.], [-1]),
                    {r_21 if sine else r_11: sign*np.nextafter(1., math.inf)}, family, 0, {})
                self.assertTrue(any(abs(math.remainder(candidate.variable_values[0]-expected, 2*math.pi))<1e-14
                                    for candidate in family_output))

    def test_numerical_polynomial_endpoints_and_coefficient_scaling(self):
        x, p = sp.symbols('x p')
        current = SkeletonNumericalSolution([0.], [-1])
        for method in (SolutionMethod.PolynomialSin.name, SolutionMethod.PolynomialCos.name):
            for sign in (-1, 1):
                solution = VariableSolution.make_polynomial_solution(
                    x, method, {0: (-sign*p, p), 1: (p, p)})
                for denominator in (1., 1e-300, np.nextafter(0., 1.)):
                    with self.subTest(method=method, sign=sign, denominator=denominator):
                        output = step_from_polynomial_solution(current, {}, solution, 0, {p: denominator})
                        self.assertEqual(len(output), 2)
                        trig = math.sin if method == SolutionMethod.PolynomialSin.name else math.cos
                        for candidate in output:
                            self.assertAlmostEqual(trig(candidate.variable_values[0]), sign)
                for invalid in (0., sp.oo, sp.nan):
                    self.assertEqual(step_from_polynomial_solution(current, {}, solution, 0, {p: invalid}), [])
        large = sp.Integer(10)**400
        solution = VariableSolution.make_polynomial_solution(x, SolutionMethod.PolynomialDirect.name,
                   {0: (-2*large, sp.S.One), 1: (large, sp.S.One)})
        output = step_from_polynomial_solution(current, {}, solution, 0, {})
        self.assertEqual([candidate.variable_values[0] for candidate in output], [2.])

    def test_numerical_explicit_unknown_guards_do_not_admit_candidates(self):
        x, p = sp.symbols('x p')
        solution = VariableSolution.make_explicit_solution(x, [sp.S.Zero],
                   SolutionMethod.OneVariableAlgebra.name, argument_valid_checkers=[p > 0])
        self.assertEqual(step_from_explicit_solution(SkeletonNumericalSolution([0.], [-1]),
                         {}, solution, 0, {}), [])

    def test_saved_models_recover_recorded_reachable_endpoints(self):
        for case in json.loads((ROOT/'solver/fixtures/argument_domains.json').read_text()):
            tree = read_skeleton_tree(str(ROOT/case['tree']))
            scope = {'__name__': 'saved_endpoint_test'}
            with redirect_stdout(StringIO()):
                exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
            arguments = [case['parameters'][str(s)] for s in tree.get_free_symbols()]
            geometry = [case['parameters'][str(s)] for s in tree.get_free_symbols(False)]
            target = np.asarray(case['target'])
            with self.subTest(robot=case['robot'], index=case['index']):
                output = scope[case['robot']+'_ik_solve'](target, *arguments)
                self.assertTrue(output)
                self.assertTrue(any(np.max(np.abs(scope[case['robot']+'_fk'](q, *geometry)-target))<1e-4
                                    for q in output))

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_safe_unsafe_and_overlapping_guard_capacity(self):
        sections = []
        for sine, prefix in ((False, False), (False, True), (True, False), (True, True)):
            tree = endpoint_tree(sine, prefix)
            sections.append('\n'.join(CodeGenerationCpp(tree, use_safe_operator=True).generate_code())
                            .split('// Code below for debug')[0])
        for safe in (False, True):
            tree = endpoint_tree(overlap=True, name='overlap_'+str(int(safe)))
            sections.append('\n'.join(CodeGenerationCpp(tree, use_safe_operator=safe).generate_code())
                            .split('// Code below for debug')[0])
        main = '''#include <cassert>
int main() {
    Eigen::Matrix4d target=Eigen::Matrix4d::Identity();
    target(0,0)=std::nextafter(1.,2.);
    assert(!endpoint_000_ik::computeIK(target).empty());
    assert(!endpoint_010_ik::computeIK(target).empty());
    target(0,0)=0.; target(0,1)=-1.; target(1,0)=std::nextafter(1.,2.); target(1,1)=0.;
    assert(!endpoint_100_ik::computeIK(target).empty());
    assert(!endpoint_110_ik::computeIK(target).empty());
    target=Eigen::Matrix4d::Identity(); target(0,0)=std::nextafter(1.,2.);
    static_assert(overlap_1_ik::max_n_solutions>=3,"rounded guards can overlap");
    overlap_1_ik::RawIKWorksace safe;
    overlap_1_ik::computeRawIK(target,safe);
    assert(safe.raw_ik_out.size()==3);
    overlap_0_ik::RawIKWorksace unsafe;
    overlap_0_ik::computeRawIK(target,unsafe);
    assert(unsafe.raw_ik_out.size()==1);
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-domain-') as temporary:
            temporary = Path(temporary)
            source = temporary/'test.cpp'; source.write_text('\n'.join(sections)+main)
            compiled = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                       str(source), '-o', str(temporary/'test')], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(temporary/'test')], check=True)
