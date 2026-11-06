"""Exercise singular branch generation on small, independent robot models."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import generate_code_cpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.build_equation_branch import build_equation_zero_branches
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode
from codegen.skeleton.tree_serialize import TreeSerializer
from fk.robots import DHEntry, RobotDescription
from solver.branch_conditions import vanishes_on_branch
from solver.equation_types import Unknown
from solver.solved_variable import SolutionDegenerateRecord, SolutionMethod, VariableSolution


def explicit(symbol, expressions, record=None):
    return VariableSolution.make_explicit_solution(
        symbol, expressions, SolutionMethod.OneVariableAlgebra.name, degenerate_record=record)


def coaxial_tree():
    x, y, z, Px, Py, r11, r21 = sp.symbols('x y z Px Py r_11 r_21')
    model = RobotDescription('coaxial_family')
    model.unknowns = [Unknown(symbol) for symbol in (x, y, z)]
    model.dh_params = [DHEntry(0, 0, 0, symbol) for symbol in (x, y, z)]
    return SkeletonTree([
        explicit(x, [sp.atan2(Py, Px)], SolutionDegenerateRecord.record_all_equations([Px, Py])),
        explicit(y, [sp.S.Zero]), explicit(z, [sp.atan2(r21, r11)-x-y])], model)


def scope_for(tree, refine_seeds=True):
    scope = {}
    exec('\n'.join(CodeGenerationPython(tree, refine_seeds=refine_seeds).generate_code()), scope)
    return scope


def nested_root_tree():
    x, Px, Py = sp.symbols('x Px Py')
    model = RobotDescription('nested_root')
    model.unknowns = [Unknown(x)]
    model.dh_params = [DHEntry(0, 0, 0, x)]
    tree = SkeletonTree([explicit(x, [sp.S.One],
        SolutionDegenerateRecord.record_all_equations([Px]))], model)
    tree.append_branched_solution([explicit(x, [sp.S(2)],
        SolutionDegenerateRecord.record_all_equations([Py]))], tree.root, 1,
        preserve_first_dispatcher=True)
    tree.append_branched_solution([explicit(x, [sp.S(3)])], tree.root.degenerate_child, 1)
    return tree


class TestEquationBranches(TestCase):
    def test_branch_zero_checks_keep_nonzero_constants(self):
        x, y = sp.symbols('x y')
        self.assertTrue(vanishes_on_branch(2*x + 3*y, [x, -y]))
        self.assertFalse(vanishes_on_branch(sp.S.One, [2*x, sp.S.One]))
        self.assertFalse(vanishes_on_branch(sp.cos(x), [sp.sin(x)]))

    def test_branch_zero_checks_use_affine_prefix_identities_without_fixing_free_angles(self):
        x, y, auxiliary = sp.symbols('x y auxiliary')
        for value in (sp.S.Zero, sp.pi, -sp.pi):
            constraints = [y-auxiliary+x, auxiliary-value]
            self.assertTrue(vanishes_on_branch(sp.sin(x+y), constraints))
            self.assertTrue(vanishes_on_branch(sp.sin(x)*sp.cos(y)+sp.sin(y)*sp.cos(x), constraints))
            self.assertFalse(vanishes_on_branch(sp.sin(x), constraints))
            self.assertFalse(vanishes_on_branch(sp.cos(x), constraints))
        self.assertFalse(vanishes_on_branch(sp.sin(x+y), [x-y]))
        self.assertFalse(vanishes_on_branch(sp.sin(x), [x*y]))
        self.assertFalse(vanishes_on_branch(x-sp.Rational(1, 3)*y, [x-sp.Float(1/3)*y]))
        self.assertFalse(vanishes_on_branch(sp.S.One, [2*x, 1]))

    def test_deferred_joint_remains_available_when_it_is_the_only_regular_first_step(self):
        from solver.build_equations import build_ik_equations
        from solver.solver_snapshot import SolverSnapshot
        from solver.solve_equations import SolveEquationOption, solve_from_snapshot
        x, y = sp.symbols('x y')
        model = RobotDescription('deferred_planar')
        model.unknowns = [Unknown(x), Unknown(y)]
        model.dh_params = [DHEntry(0, 0, 0, x), DHEntry(0, 1, 0, y)]
        equations = build_ik_equations(model)
        snapshot = SolverSnapshot(equations.scalar_equations, equations.sos_hint, equations.used_unknowns,
                                  equations.all_symbol_set)
        option = SolveEquationOption(initially_deferred_symbols={x}, direct_solutions_only=True)
        option.on_main_branch = False
        option.disable_polynomial_solution = True
        option.stop_on_first_unsolved = True
        with redirect_stdout(StringIO()):
            solutions = solve_from_snapshot(snapshot, option)
        self.assertEqual({solution.solved_variable for solution in solutions}, {x, y})
        scope = scope_for(SkeletonTree(solutions, model))
        target = scope['deferred_planar_fk'](np.array([.7, -.3]))
        output = scope['deferred_planar_ik_solve'](target)
        self.assertTrue(output)
        for candidate in output:
            np.testing.assert_allclose(scope['deferred_planar_fk'](candidate), target, atol=1e-12)

    def test_recovers_folded_planar_arm_from_orientation(self):
        x, y, z, Px, Py = sp.symbols('x y z Px Py')
        model = RobotDescription('folded_planar')
        model.unknowns = [Unknown(symbol) for symbol in (x, y, z)]
        model.dh_params = [DHEntry(0, 0, 0, x), DHEntry(0, 1, 0, y), DHEntry(0, 1, 0, z)]
        model.unknown_as_parameter_more_dof = [z]
        bad = explicit(x, [sp.atan2(Py, Px)-sp.atan2(sp.sin(y), 1+sp.cos(y))],
                       SolutionDegenerateRecord.record_all_equations([Px, Py, 1+sp.cos(y), sp.sin(y)]))
        tree = SkeletonTree([explicit(y, [sp.pi]), bad], model)
        # Check the symbolic branch itself without the optional runtime recovery.
        before = scope_for(tree, refine_seeds=False)
        target = before['folded_planar_fk'](np.array([0.7, np.pi, 0.0]))
        self.assertEqual(before['folded_planar_ik_solve'](target, 0.0), [])
        numerical = scope_for(tree)
        self.assertTrue(any(np.max(np.abs(numerical['folded_planar_fk'](q)-target)) < 1e-7
                            for q in numerical['folded_planar_ik_solve'](target, 0.0)))
        with redirect_stdout(StringIO()):
            build_equation_zero_branches(tree, representative_count=0)
        recovered = scope_for(tree, refine_seeds=False)
        for angle in (0.0, 0.7, -1.2, np.pi/2):
            target = recovered['folded_planar_fk'](np.array([angle, np.pi, 0.0]))
            output = recovered['folded_planar_ik_solve'](target, 0.0)
            self.assertTrue(output)
            for candidate in output:
                np.testing.assert_allclose(recovered['folded_planar_fk'](candidate), target, atol=1e-12)

    def test_family_representatives_survive_serialization_and_fk_filtering(self):
        tree = coaxial_tree()
        build_equation_zero_branches(tree)
        data = TreeSerializer(tree.num_nodes, tree.robot)
        tree.preorder_visit(data)
        tree = TreeSerializer.load_skeleton_tree(data.serialize_dict())
        scope = scope_for(tree)
        target = scope['coaxial_family_fk'](np.array([0.2, -0.3, 0.6]))
        output = scope['coaxial_family_ik_solve'](target)
        self.assertEqual(len(output), 33)
        self.assertEqual({round(float(q[0, 0]), 10) for q in output},
                         {round(float(v), 10) for v in np.linspace(-np.pi, np.pi, 33)})
        for candidate in output:
            np.testing.assert_allclose(scope['coaxial_family_fk'](candidate), target, atol=1e-12)
        target[0, 3] = 0.3
        self.assertEqual(scope['coaxial_family_ik_solve'](target), [])

    def test_nested_root_guards_do_not_run_inactive_branches(self):
        tree = nested_root_tree()
        nested = tree.root.degenerate_child
        self.assertIsInstance(nested, EquationAllZeroDispatcherNode)
        solve = scope_for(tree)['nested_root_ik_solve_raw']
        for a, b, expected in ((1, 0, 1), (0, 1, 2), (0, 0, 3)):
            target = np.eye(4)
            target[:2, 3] = a, b
            np.testing.assert_allclose(solve(target), [[[expected]]])

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_retains_more_than_sixteen_family_candidates(self):
        tree = coaxial_tree()
        build_equation_zero_branches(tree)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='yaik-family-') as temporary:
            temporary = Path(temporary)
            generate_code_cpp(tree, str(temporary/'solver.cpp'), use_safe_operator=True)
            generate_code_cpp(nested_root_tree(), str(temporary/'nested.cpp'), use_safe_operator=True)
            (temporary/'main.cpp').write_text('''#define main generated_demo
#include "solver.cpp"
#undef main
#define main nested_demo
#include "nested.cpp"
#undef main
#include <cassert>
int main() {
    using Solver = coaxial_family_ik;
    static_assert(Solver::max_n_solutions >= 33, "representatives must fit");
    std::array<double, 3> q{{0.2, -0.3, 0.6}};
    auto target = Solver::computeFK(q);
    auto solutions = Solver::computeIK(target);
    assert(solutions.size() == 33);
    for (const auto& candidate : solutions)
        assert((Solver::computeFK(candidate)-target).cwiseAbs().maxCoeff() < 1e-12);
    for (const auto& test : {std::array<double, 3>{{1, 0, 1}}, {{0, 1, 2}}, {{0, 0, 3}}}) {
        auto pose = Eigen::Matrix4d::Identity().eval();
        pose(0, 3) = test[0]; pose(1, 3) = test[1];
        nested_root_ik::RawIKWorksace workspace;
        nested_root_ik::computeRawIK(pose, workspace);
        assert(workspace.raw_ik_out.size() == 1);
        assert(workspace.raw_ik_out[0][0] == test[2]);
    }
}
''')
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), '-I', '/usr/include/eigen3',
                            str(temporary/'main.cpp'), '-o', str(temporary/'check')],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(temporary/'check')], check=True)
