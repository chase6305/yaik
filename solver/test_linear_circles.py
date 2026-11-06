"""Rank-two coupled circles can have four isolated angles, not just a family."""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import generate_code_cpp
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_serialize import read_skeleton_tree
from codegen.skeleton.visitor_numerical_solve import step_from_linear_type2_solution, SkeletonNumericalSolution
from fk.robots import DHEntry, RobotDescription
from python_run_import import solve_linear_type2_all
from solver.equation_types import Unknown
from solver.solved_variable import VariableSolution
from solver.solved_variable_impl import LinearSinCosType_2_SolutionImpl
from solver.test_equation_branches import explicit, scope_for


MATRIX = np.array([[1., 0., -2., 0.], [0., 1., 0., -.5], [1., 1., -2., -.5]])


def linear_tree(non_root=False, impossible=False):
    x, y, z = sp.symbols('x y z')
    name = 'linear_circles_' + ('prefix' if non_root else 'root') + ('_empty' if impossible else '')
    model = RobotDescription(name)
    model.unknowns = [Unknown(symbol) for symbol in (x, y, z)]
    model.dh_params = [DHEntry(0, 0, 0, symbol) for symbol in (x, y, z)]
    matrix = MATRIX.copy()
    if impossible:
        matrix[1, 3] = matrix[2, 3] = -2.
    numerical = VariableSolution(LinearSinCosType_2_SolutionImpl(y, sp.Matrix(matrix), [(0, 1, 2)]))
    first = explicit(x, [sp.atan2(2*sp.sin(y), sp.cos(y)/2)])
    prefix = explicit(z, [sp.S.Zero])
    return SkeletonTree([prefix, numerical, first] if non_root else [numerical, first, prefix], model)


class TestLinearCircles(TestCase):
    def test_four_isolated_roots_with_rank_two_and_uniform_extreme_scales(self):
        expected = [math.atan(.5), -math.atan(.5), math.pi-math.atan(.5), -math.pi+math.atan(.5)]
        for scale in (1., 1e-200, 1e200):
            roots = solve_linear_type2_all(MATRIX*scale)
            self.assertEqual(len(roots), 4)
            for angle in expected:
                self.assertLess(min(abs(math.remainder(root-angle, 2*math.pi)) for root in roots), 1e-12)
            for y in roots:
                x = math.atan2(2*math.sin(y), .5*math.cos(y))
                vector = [math.sin(x), math.cos(x), math.sin(y), math.cos(y)]
                np.testing.assert_allclose(MATRIX@vector, 0., atol=2e-15)

    def test_numerical_visitor_and_offline_selection_keep_all_four_directions(self):
        from solver.linear_solver.linear_sin_cos_type2 import verify_linear_coefficient_as_solution
        tree = linear_tree()
        node = next(node for node in tree.node_list if node.is_solution_node and
                    node.solution.impl_class_key == 'LinearSinCosType_2')
        current = SkeletonNumericalSolution([0., 0., 0.], [-1, -1, -1])
        output = step_from_linear_type2_solution(current, {}, node.solution, 1, {})
        self.assertEqual(len(output), 4)
        self.assertEqual({value.variable_value_in_solution_idx[1] for value in output}, set(range(4)))
        x, y = tree.robot.unknowns[0].symbol, tree.robot.unknowns[1].symbol
        cases = [{x: math.atan2(2*math.sin(angle), .5*math.cos(angle)), y: angle}
                 for angle in (math.atan(.5), -math.atan(.5), math.pi-math.atan(.5), -math.pi+math.atan(.5))]
        self.assertTrue(verify_linear_coefficient_as_solution(sp.Matrix(MATRIX), x, y, cases))

    def test_rank_three_no_real_circle_and_invalid_matrices(self):
        angle = .431
        rank_three = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.],
                               [0., 0., math.cos(angle), -math.sin(angle)]])
        roots = solve_linear_type2_all(rank_three)
        self.assertEqual(len(roots), 2)
        for y in roots:
            np.testing.assert_allclose(rank_three@[math.sin(y), math.cos(y), math.sin(y), math.cos(y)],
                                       0., atol=1e-14)
        impossible = MATRIX.copy()
        impossible[1, 3] = impossible[2, 3] = -2.
        self.assertEqual(solve_linear_type2_all(impossible), [])
        self.assertEqual(solve_linear_type2_all(np.zeros((4, 4))), [])
        for bad in (math.nan, math.inf):
            invalid = MATRIX.copy(); invalid[0, 0] = bad
            self.assertEqual(solve_linear_type2_all(invalid), [])

    def test_python_root_nonroot_and_failed_prefix_lifecycle(self):
        for non_root in (False, True):
            for impossible in (False, True):
                tree = linear_tree(non_root, impossible)
                scope = scope_for(tree)
                raw = scope[tree.robot.name+'_ik_solve_raw'](np.eye(4))
                self.assertEqual(len(raw), 0 if impossible else 4)
                if not impossible:
                    q = np.array([math.atan(2.), math.atan(.5), 0.])
                    pose = scope[tree.robot.name+'_fk'](q)
                    candidates = scope[tree.robot.name+'_ik_solve'](pose)
                    self.assertEqual(len(candidates), 2)
                    for candidate in candidates:
                        np.testing.assert_allclose(scope[tree.robot.name+'_fk'](candidate), pose, atol=1e-14)

    def test_saved_franka_model_keeps_the_approximate_rotation_fallback(self):
        root = Path(__file__).resolve().parents[1]
        tree = read_skeleton_tree(str(root/'gallery/franka_panda/franka_panda_ik.yaml'))
        scope = scope_for(tree)
        # The saved model uses 0.707107 for sqrt(1/2). Its transformed targets
        # need the established tolerance fallback even when the matrix is rank 4.
        for q in ([-1.1806897001852492, 1.6406747015325394, 3.1062806166176156,
                   -.7177022474750263, -math.pi/2, 1.8515763549567943, 2.393016639925662],
                  [-1.1645896866134173, -.21854204650755893, -2.792449340496721,
                   -.11016080954544183, 2.3927567444420577, 1.1875905033437357, 2.100460831409918]):
            target = scope['franka_panda_fk'](np.asarray(q))
            output = scope['franka_panda_ik_solve'](target, q[3])
            self.assertTrue(output)
            for value in output:
                self.assertEqual(float(value[3, 0]), q[3])
                self.assertLess(np.max(np.abs(scope['franka_panda_fk'](value)-target)), 1e-4)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_root_nonroot_rank_two_and_empty_prefix(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='yaik-circles-') as temporary:
            temporary = Path(temporary)
            includes = []
            for non_root, impossible in ((False, False), (True, False), (False, True), (True, True)):
                tree = linear_tree(non_root, impossible)
                filename = tree.robot.name+'.cpp'
                generate_code_cpp(tree, str(temporary/filename), use_safe_operator=True)
                includes.append('#define main '+tree.robot.name+'_demo\n#include "'+filename+'"\n#undef main\n')
            franka = read_skeleton_tree(str(root/'gallery/franka_panda/franka_panda_ik.yaml'))
            generate_code_cpp(franka, str(temporary/'franka.cpp'), use_safe_operator=True)
            includes.append('#define main franka_demo\n#include "franka.cpp"\n#undef main\n')
            source = temporary/'test.cpp'
            source.write_text(''.join(includes)+'''#include <cassert>
int main() {
    Eigen::Matrix<double, 3, 4> A;
    A << 1,0,-2,0, 0,1,0,-.5, 1,1,-2,-.5;
    for (double scale : {1.0, 1e-200, 1e200}) {
        const Eigen::Matrix<double, 3, 4> scaled = A*scale;
        const auto roots = yaik_cpp::linear_solver::solveLinearType2All<3>(scaled);
        assert(roots.size() == 4);
        for (double y : roots) {
            const double x = std::atan2(2*std::sin(y), .5*std::cos(y));
            Eigen::Vector4d v; v << std::sin(x), std::cos(x), std::sin(y), std::cos(y);
            assert((A*v).cwiseAbs().maxCoeff() < 1e-13);
        }
    }
    const std::array<double, 3> q{{std::atan(2.0), std::atan(.5), 0}};
    const auto target = linear_circles_root_ik::computeFK(q);
    assert(linear_circles_root_ik::computeIK(target).size() == 2);
    assert(linear_circles_prefix_ik::computeIK(target).size() == 2);
    assert(linear_circles_root_empty_ik::computeIK(target).empty());
    assert(linear_circles_prefix_empty_ik::computeIK(target).empty());
    const std::array<double, 7> panda{{-1.1806897001852492, 1.6406747015325394, 3.1062806166176156,
        -.7177022474750263, -M_PI/2, 1.8515763549567943, 2.393016639925662}};
    const auto panda_target = franka_panda_ik::computeFK(panda);
    const auto panda_output = franka_panda_ik::computeIK(panda_target, panda[3]);
    assert(!panda_output.empty());
    for (const auto& value : panda_output) {
        assert(value[3] == panda[3]);
        assert((franka_panda_ik::computeFK(value)-panda_target).squaredNorm() < 1e-6);
    }
}
''')
            completed = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', '/usr/include/eigen3',
                                        str(source), '-o', str(temporary/'test')], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            subprocess.run([str(temporary/'test')], check=True)
