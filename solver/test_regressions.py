"""Regression coverage for model isolation, numerical validation, and code generation."""
from pathlib import Path
from unittest import TestCase, mock
import inspect
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import sympy as sp

from fk.robot_models import mini_dd_robot, puma_robot
from fk.chain_transform import ChainTransformElement
from fk.robots import DHEntry, RobotDescription
from python_run_import import (compute_polynomial_real_roots, try_solve_linear_type2,
                               try_solve_linear_type2_specific_rows)
from solver.build_equations import IkEquationCollection
from solver.degenerate_analyse.analyser_step import (
    CheckConstantAnalyserStep, DegenerateAnalyserContext, DegenerateAnalyserOption)
from solver.degenerate_analyse.numerical_analyser import NumericalCheckDegenerateAnalyzer
from solver.degenerate_analyse.numerical_check_data import generate_test_cases
from solver.equation_types import EquationInput, NumericalAnalyseContext, ScalarEquation, Unknown, UnknownType
from solver.equation_utils import scalar_equation_by_unknowns
from solver.linear_solver.linear_sin_cos_type2_numerical_solve import try_solve_linear_type2_combined
from solver.polynomial_solver.verify_polynomial_solution import verify_polynomial_solution
from solver.select_solution import select_solution_in_type
from solver.solved_variable import NumericalAnalysedResultType, SolutionDegenerateRecord, SolutionMethod, VariableSolution
from solver.solve_branched import construct_variable_branched_equations
from solver.solve_equations import solve_equations, SolveEquationOption
from solver.solver_snapshot import SolverSnapshot
from solver.substitute_transform import substitute_transform_parallel
from solver.unary_solvers.unary_solver_step import UnarySolverStep
from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_serialize import TreeSerializer


def polynomial_tree():
    x = sp.Symbol('x')
    robot = RobotDescription('regression')
    robot.unknowns = [Unknown(x, UnknownType.Translational.name)]
    robot.dh_params = [DHEntry(0, 0, x, 0)]
    solution = VariableSolution.make_polynomial_solution(
        x, SolutionMethod.PolynomialDirect.name,
        {0: (-sp.S.One, sp.S.One), 2: (sp.S.One, sp.S.One)})
    return SkeletonTree([solution], robot)


def parameterized_tree():
    tree = polynomial_tree()
    tree.robot.name = 'parameterized'
    length, y = sp.symbols('z_length y')
    tree.robot.symbolic_parameters.add(length)
    tree.robot.parameters_bound[length] = (0.1, 0.9)
    tree.robot.dh_params[0].a = length
    tree.robot.unknowns.append(Unknown(y, UnknownType.Translational.name))
    tree.robot.dh_params.append(DHEntry(0, 0, y, 0))
    tree.robot.unknown_as_parameter_more_dof = [y]
    return tree


class TestRegressions(TestCase):
    def test_gallery_solvers_round_trip_and_reject_unreachable_targets(self):
        rng = np.random.default_rng(20261103)
        gallery = Path(__file__).resolve().parents[1] / 'gallery'
        for path in sorted(gallery.glob('*/*_ik_generated.py')):
            name = path.name.removesuffix('_ik_generated.py')
            with self.subTest(robot=name), warnings.catch_warnings():
                warnings.simplefilter('error')
                scope = {}
                exec(compile(path.read_text(), str(path), 'exec'), scope)
                fk, ik = scope[name + '_fk'], scope[name + '_ik_solve']
                free_parameters = list(inspect.signature(ik).parameters)[1:]
                indices = [int(parameter.removeprefix('th_')) for parameter in free_parameters]
                for q in rng.uniform(0.2, 1.1, size=(20, scope['robot_nq'])):
                    target = fk(q)
                    output = ik(target, *(float(q[index]) for index in indices))
                    self.assertTrue(output)
                    for solution in output:
                        self.assertLess(float(np.max(np.abs(fk(solution) - target))), scope['pose_tolerance'])
                target = np.eye(4)
                target[:3, 3] = 100
                self.assertFalse(ik(target, *(float(q[index]) for index in indices)))
                for invalid in (np.nan, np.inf, -np.inf):
                    for argument in range(len(indices)):
                        parameters = [float(q[index]) for index in indices]
                        parameters[argument] = invalid
                        self.assertEqual(ik(fk(q), *parameters), [])
                for row in range(4):
                    for col in range(4):
                        for invalid in (np.nan, np.inf, -np.inf):
                            target = fk(q)
                            target[row, col] = invalid
                            self.assertFalse(ik(target, *(float(q[index]) for index in indices)))

    def test_loading_robots_does_not_change_previous_model(self):
        first_data = puma_robot().to_dict()
        second_data = mini_dd_robot().to_dict()
        first = RobotDescription.load_from_dict(first_data)
        second = RobotDescription.load_from_dict(second_data)
        self.assertEqual(first.to_dict(), first_data)
        self.assertEqual(second.to_dict(), second_data)
        self.assertIsNot(first.unknowns, second.unknowns)

    def test_parameter_bounds_round_trip(self):
        robot = puma_robot()
        length = sp.Symbol('length')
        robot.symbolic_parameters.add(length)
        robot.parameters_bound[length] = (0.1, 0.9)
        loaded = RobotDescription.load_from_dict(robot.to_dict())
        self.assertEqual(loaded.parameters_bound[length], (0.1, 0.9))

    def test_contexts_have_independent_mutable_defaults(self):
        for make in (lambda: IkEquationCollection(sp.eye(4)), NumericalAnalyseContext,
                     DegenerateAnalyserContext, lambda: SolverSnapshot([], [], [], set())):
            first, second = make(), make()
            for name, value in vars(first).items():
                if isinstance(value, (list, dict, set)):
                    self.assertIsNot(value, getattr(second, name), name)

    def test_unknown_and_solution_records_are_independent(self):
        x, y = sp.symbols('x y')
        first, second = Unknown(x), Unknown(y)
        first.degenerate_check_value.append(sp.Rational(1, 3))
        self.assertNotIn(sp.Rational(1, 3), second.degenerate_check_value)
        first, second = [VariableSolution.make_explicit_solution(
            symbol, [sp.S.Zero], SolutionMethod.OneVariableAlgebra.name) for symbol in (x, y)]
        first.update_numerical_analysis_result(NumericalAnalysedResultType.NumericalDegenerate.name)
        self.assertEqual(second.degenerate_record.numerical_analysed_result,
                         NumericalAnalysedResultType.NotAnalyzedYet.name)
        first, second = ChainTransformElement(np.eye(4)), ChainTransformElement(np.eye(4))
        first.joint_axis[0] = 1
        self.assertEqual(second.joint_axis, [0, 0, 1])

    def test_mixed_solution_types_keep_explicit_preference(self):
        tree = polynomial_tree()
        polynomial = tree.node_list[1].solution
        explicit = VariableSolution.make_explicit_solution(
            polynomial.solved_variable, [sp.S.One], SolutionMethod.OneVariableAlgebra.name)
        for solutions in ([explicit, polynomial], [polynomial, explicit]):
            self.assertIs(select_solution_in_type(solutions, None), explicit)

    def test_polynomial_candidate_must_satisfy_all_equations(self):
        x = sp.Symbol('x')
        wrong_polynomial = {0: (sp.S.Zero, sp.S.One), 1: (sp.S.One, sp.S.One)}
        output = (wrong_polynomial, x, Unknown(x), SolutionMethod.PolynomialDirect.name)
        self.assertEqual(verify_polynomial_solution([x * (x - 1), x - 1], output, [{x: 1}]),
                         (False, 0.0))
        self.assertEqual(verify_polynomial_solution([x], output, [{x: 0}]), (True, 1.0))

    def test_polynomial_verification_handles_boundaries_and_missing_values(self):
        x, parameter = sp.symbols('x parameter')
        polynomial = {0: (-sp.S.One, sp.S.One), 1: (sp.S.One, sp.S.One)}
        output = (polynomial, sp.sin(x), Unknown(x), SolutionMethod.PolynomialSin.name)
        self.assertEqual(verify_polynomial_solution([sp.sin(x) - 1], output, [{x: sp.pi / 2}]),
                         (True, 1.0))
        self.assertEqual(verify_polynomial_solution([x + parameter], output, [{x: 0}]),
                         (False, 0.0))
        self.assertEqual(verify_polynomial_solution([x], output, []), (False, 0))

    def test_polynomial_verification_preserves_zero_denominators(self):
        x, parameter = sp.symbols('x parameter')
        polynomial = {0: (-parameter, parameter), 1: (parameter, parameter)}
        output = (polynomial, x, Unknown(x), SolutionMethod.PolynomialDirect.name)
        self.assertEqual(verify_polynomial_solution([x - 1], output, [{x: 1, parameter: 0}]), (False, 0.0))
        self.assertEqual(verify_polynomial_solution([x - 1], output, [{x: 1, parameter: 2}]), (True, 1.0))

    def test_real_roots_handle_roundoff_and_degree_reduction(self):
        roots = compute_polynomial_real_roots([1, -0.6, 0.09])
        np.testing.assert_allclose(sorted(roots), [0.3, 0.3], atol=1e-6)
        self.assertEqual(compute_polynomial_real_roots([0, 0, 2, -4]), [2.0])
        self.assertEqual(compute_polynomial_real_roots([0, 0]), [])
        self.assertEqual(compute_polynomial_real_roots([np.nan, 1]), [])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(compute_polynomial_real_roots([1e-320, 1, 1]), [])

    def test_linear_type2_recovers_unit_circle_families(self):
        angle = 0.73
        B = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        A = np.vstack((np.column_stack((np.eye(2), -B)), np.zeros(4)))
        solutions = try_solve_linear_type2(A)
        np.testing.assert_allclose(solutions, [0, np.pi])
        for y in solutions:
            self.assertAlmostEqual(np.linalg.norm(B @ [np.sin(y), np.cos(y)]), 1)
        A[:2, 2:] = -2 * np.eye(2)
        self.assertIsNone(try_solve_linear_type2(A))

    def test_linear_type2_falls_back_to_independent_rows(self):
        A = np.array([[0, 0, 0, 0], [1, 0, 0, -1], [0, 1, -1, 0], [1, 0, -1, 0]], dtype=float)
        solutions = try_solve_linear_type2_specific_rows(A, 0, 1, 2)
        self.assertIsNotNone(solutions)
        for y in solutions:
            values = [np.cos(y), np.sin(y), np.sin(y), np.cos(y)]
            np.testing.assert_allclose(A @ values, 0, atol=1e-14)
        self.assertIsNone(try_solve_linear_type2_specific_rows(np.zeros((4, 4)), 0, 1, 2))

    def test_linear_type2_checks_remaining_rows_before_choosing_family(self):
        angle = 1.0
        A = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [0.7, -0.2, -0.7, 0.2],
                      [0, 0, np.cos(angle), -np.sin(angle)]])
        solutions = try_solve_linear_type2_specific_rows(A, 0, 1, 2)
        self.assertIsNotNone(solutions)
        for y in solutions:
            np.testing.assert_allclose(A @ [np.sin(y), np.cos(y), np.sin(y), np.cos(y)], 0, atol=1e-14)
        np.testing.assert_allclose(try_solve_linear_type2_combined(A, [(0, 1, 2)]), solutions)

    def test_linear_type2_returns_empty_when_row_selection_is_numerically_singular(self):
        A = np.zeros((4, 4))
        top = np.array([[1100, 1300], [770, 909.9999999999999]])
        A[1:3, :2], A[1:3, 2:] = top, -top
        self.assertIsNone(try_solve_linear_type2_specific_rows(A, 0, 1, 2))

    def test_equation_dedup_preserves_layout_needed_by_rule_matching(self):
        a, x, y = sp.symbols('a x y')
        equations = EquationInput([ScalarEquation(a + y, x + y), ScalarEquation(a, x)], [])
        unknowns = [Unknown(x, UnknownType.Translational.name), Unknown(y, UnknownType.Translational.name)]
        options = SolveEquationOption()
        options.disable_polynomial_solution = True
        options.stop_on_first_unsolved = True
        options.stop_on_bad_solution = True
        solutions = solve_equations(equations, unknowns, {a}, NumericalAnalyseContext(), options)
        self.assertTrue(any(solution.solved_variable == x and solution.explicit_solutions == [a]
                            for solution in solutions))

    def test_compiled_degeneration_checks_keep_unknown_samples_conservative(self):
        x = sp.Symbol('x')
        record = SolutionDegenerateRecord.record_all_equations([sp.sin(x)])
        analyser = NumericalCheckDegenerateAnalyzer([{x: 0}, {x: 1}, {}])
        with mock.patch('multiprocessing.Pool', side_effect=AssertionError('unexpected process')):
            kind, ratio = analyser.perform_analyse(record, [], {x}, {}, {})
        self.assertEqual(kind, NumericalAnalysedResultType.NumericalDegenerate.name)
        self.assertAlmostEqual(ratio, 2 / 3)

    def test_constant_degeneration_checks_do_not_certify_nonfinite_values(self):
        analyser = CheckConstantAnalyserStep()
        context, option = DegenerateAnalyserContext(), DegenerateAnalyserOption()
        for value in (sp.oo, -sp.oo, sp.nan, sp.S.Zero, sp.Float(0.0), sp.Float(-0.0)):
            record = SolutionDegenerateRecord.record_all_equations([value])
            self.assertIsNone(analyser.perform_analyse(record, context, option))
        record = SolutionDegenerateRecord.record_all_equations([sp.S.One])
        self.assertEqual(analyser.perform_analyse(record, context, option).type, 'AlwaysNonDegenerate')

    def test_degenerate_branch_can_solve_a_previously_unknown_variable(self):
        x, y = sp.symbols('x y')
        equations = IkEquationCollection(sp.eye(4))
        equations.used_unknowns = [Unknown(x), Unknown(y)]
        equations.scalar_equations = [ScalarEquation(x + y, sp.S.Zero)]
        solution = VariableSolution.make_explicit_solution(
            y, [sp.S.Zero], SolutionMethod.OneVariableAlgebra.name,
            degenerate_record=SolutionDegenerateRecord.record_variable_value({y: [sp.S.Zero]}))
        snapshot, fixed = construct_variable_branched_equations(equations, [solution], [(y, sp.S.Zero)])
        self.assertEqual([unknown.symbol for unknown in snapshot.unknowns], [x])
        self.assertEqual(fixed[0].solved_variable, y)
        self.assertEqual(fixed[0].explicit_solutions, [sp.S.Zero])

    def test_compiled_samples_match_symbolic_fk(self):
        x, length = sp.symbols('x length')
        pose = sp.eye(4)
        pose[0, 3] = length * sp.cos(x)
        pose[1, 3] = length * sp.sin(x)
        equations = IkEquationCollection(pose)
        equations.all_symbol_set = {x, length}
        equations.used_unknowns = [Unknown(x)]
        equations.parameter_value_dict = {length: 0.7}
        equations.parameter_bound_dict = {x: (-1.0, 1.0)}
        for sample in generate_test_cases(equations, {x: sp.pi / 2}, n_test_cases=4):
            expected = np.asarray(pose.subs(sample), dtype=float)
            from fk.fk_equations import ik_target_4x4
            actual = np.asarray(ik_target_4x4().subs(sample), dtype=float)
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_simple_rules_avoid_process_startup(self):
        x, a, b = sp.symbols('x a b')
        equations = scalar_equation_by_unknowns(
            [ScalarEquation(a, sp.sin(x)), ScalarEquation(b, sp.cos(x))], [x])
        with mock.patch('multiprocessing.Manager', side_effect=AssertionError('unexpected process')), \
                mock.patch('multiprocessing.Pool', side_effect=AssertionError('unexpected process')):
            solutions = UnarySolverStep.make_default().solve_step_parallel(equations, [x], with_timeout=False)
        self.assertTrue(any(solution.solution_method == SolutionMethod.Tangent.name for solution in solutions))

    def test_explicit_timeout_keeps_worker_isolation(self):
        x, a, b = sp.symbols('x a b')
        equations = scalar_equation_by_unknowns(
            [ScalarEquation(a, sp.sin(x)), ScalarEquation(b, sp.cos(x))], [x])
        step = UnarySolverStep.make_default()
        with mock.patch.object(step, 'solve_step', side_effect=AssertionError('timeout bypassed')), \
                mock.patch.object(step, 'parallel_run', return_value=[]) as workers:
            step.solve_step_parallel(equations, [x], with_timeout=True)
        workers.assert_called_once()

    def test_rule_workers_share_one_batch_deadline(self):
        clock = [0.0]

        class Worker:
            def __init__(self, **kwargs):
                self.alive = False

            def start(self):
                self.alive = True

            def join(self, timeout=None):
                if timeout is not None:
                    clock[0] += timeout

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.alive = False

        with mock.patch('multiprocessing.Manager') as manager, \
                mock.patch('multiprocessing.Lock'), \
                mock.patch('multiprocessing.Process', Worker), \
                mock.patch('solver.unary_solvers.unary_solver_step.time.monotonic', lambda: clock[0]):
            manager.return_value.list.return_value = []
            UnarySolverStep.parallel_run([{}] * 3, timeout_seconds=1)
        self.assertAlmostEqual(clock[0], 1.0)

    def test_substitution_pair_budget_and_no_self_pairs(self):
        x, y = sp.symbols('x y')
        equations = [ScalarEquation(sp.Integer(i), x + i * y) for i in range(50)]
        with mock.patch('multiprocessing.Pool') as pool:
            pool.return_value.__enter__.return_value.map.side_effect = lambda _, args: [None] * len(args)
            substitute_transform_parallel(equations, [x, y], max_candidate_pair=7)
            pairs = pool.return_value.__enter__.return_value.map.call_args.args[1]
        self.assertEqual(len(pairs), 7)
        self.assertTrue(all(pair['i'] != pair['j'] for pair in pairs))

    def test_codegen_loads_tree_without_rebuilding_equations(self):
        tree = polynomial_tree()
        serializer = TreeSerializer(tree.num_nodes, tree.robot)
        tree.preorder_visit(serializer)
        with mock.patch('solver.build_equations.build_ik_equations', side_effect=AssertionError('unexpected rebuild')):
            loaded = TreeSerializer.load_skeleton_tree(serializer.serialize_dict())
            scope = {}
            exec('\n'.join(CodeGenerationPython(loaded).generate_code()), scope)
        target = np.eye(4)
        target[2, 3] = 1.0
        raw = scope['regression_ik_solve_raw'](target)
        np.testing.assert_allclose(sorted(float(q[0, 0]) for q in raw), [-1, 1])
        checked = scope['regression_ik_solve'](target)
        self.assertEqual(len(checked), 1)
        np.testing.assert_allclose(checked[0], [[1]])

    def test_codegen_rejects_incomplete_branches(self):
        tree = polynomial_tree()
        y = sp.Symbol('y')
        tree.robot.unknowns.append(Unknown(y))
        tree.robot.dh_params.append(DHEntry(0, 0, 0, y))
        for generator in (CodeGenerationPython, CodeGenerationCpp):
            with self.assertRaisesRegex(ValueError, 'missing y'):
                generator(tree)

    def test_runtime_parameters_and_redundant_joints_without_offsets(self):
        scope = {}
        exec('\n'.join(CodeGenerationPython(parameterized_tree()).generate_code()), scope)
        self.assertEqual(list(inspect.signature(scope['parameterized_ik_solve']).parameters),
                         ['T_ee', 'z_length', 'y'])
        target = scope['parameterized_fk'](np.array([1, 0.25]), 0.4)
        output = scope['parameterized_ik_solve'](target, 0.4, 0.25)
        self.assertEqual(len(output), 1)
        np.testing.assert_allclose(output[0].reshape(-1), [1, 0.25])

    def test_free_joints_keep_robot_order_after_geometry_parameters(self):
        tree = parameterized_tree()
        th_2, th_10 = sp.symbols('th_2 th_10')
        tree.robot.unknowns[1] = Unknown(th_2, UnknownType.Translational.name)
        tree.robot.dh_params[1].d = th_2
        tree.robot.unknowns.append(Unknown(th_10, UnknownType.Translational.name))
        tree.robot.dh_params.append(DHEntry(0, 0, th_10, 0))
        tree.robot.unknown_as_parameter_more_dof = [th_2, th_10]
        self.assertEqual(tree.get_free_symbols(), [sp.Symbol('z_length'), th_2, th_10])
        scope = {}
        exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
        target = scope['parameterized_fk']([1, 0.25, 0.5], 0.4)
        solutions = scope['parameterized_ik_solve'](target, 0.4, 0.25, 0.5)
        self.assertEqual(len(solutions), 1)
        np.testing.assert_allclose(solutions[0].reshape(-1), [1, 0.25, 0.5])

    def test_generated_ik_rejects_nonfinite_targets(self):
        scope = {}
        exec('\n'.join(CodeGenerationPython(polynomial_tree()).generate_code()), scope)
        for row in range(4):
            for col in range(4):
                for invalid in (np.nan, np.inf, -np.inf):
                    target = np.eye(4)
                    target[2, 3] = 1
                    target[row, col] = invalid
                    self.assertEqual(scope['regression_ik_solve'](target), [])

    def test_cpp_sparse_polynomial_codegen_and_runtime(self):
        eigen = Path('/usr/include/eigen3')
        if not shutil.which('g++') or not (eigen / 'Eigen/Core').exists():
            self.skipTest('g++ and Eigen are needed for the C++ integration test')
        code = '\n'.join(CodeGenerationCpp(polynomial_tree(), use_safe_operator=True).generate_code())
        code += '\n' + '\n'.join(CodeGenerationCpp(parameterized_tree(), use_safe_operator=True).generate_code())
        main = r'''
int main() {
    const auto roots = yaik_cpp::computePolynomialRealRoots<3>({0, 0, 2, -4});
    assert(roots[0].is_valid && std::abs(roots[0].value - 2) < 1e-12);
    assert(!roots[1].is_valid && !roots[2].is_valid);
    const auto underflow_roots = yaik_cpp::computePolynomialRealRoots<2>({1e-300, 1e300, 1});
    assert(!underflow_roots[0].is_valid && !underflow_roots[1].is_valid);
    Eigen::Matrix<double, 4, 4> linear;
    linear << 1, 0, -1, 0,
              0, 1, 0, -1,
              0.7, -0.2, -0.7, 0.2,
              0, 0, std::cos(1.0), -std::sin(1.0);
    double y0 = 0, y1 = 0;
    assert(yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<4>(linear, 0, 1, 2, y0, y1));
    for (double y : {y0, y1}) {
        Eigen::Vector4d values;
        values << std::sin(y), std::cos(y), std::sin(y), std::cos(y);
        assert((linear * values).norm() < 1e-14);
    }
    linear << 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 1, 0, -1, 0;
    assert(yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<4>(linear, 0, 1, 2, y0, y1));
    for (double y : {y0, y1}) {
        Eigen::Vector4d values;
        values << std::cos(y), std::sin(y), std::sin(y), std::cos(y);
        assert((linear * values).norm() < 1e-14);
    }
    linear.row(0) = linear.row(3);
    assert(yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<4>(linear, 1, 2, 0, y0, y1));
    linear.setZero();
    linear(0, 0) = linear(1, 1) = 1;
    linear(0, 2) = linear(1, 3) = -1;
    assert(yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<4>(linear, 0, 1, 2, y0, y1));
    assert(y0 == 0 && std::abs(y1 - std::acos(-1.0)) < 1e-15);
    linear(0, 2) = linear(1, 3) = -2;
    assert(!yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<4>(linear, 0, 1, 2, y0, y1));
    std::array<double, 2> q{{0, 0}};
    auto fk = [](const std::array<double, 2>& value) -> Eigen::Matrix4d {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(2, 3) = value[0] + value[1];
        return pose;
    };
    auto jacobian = [](const std::array<double, 2>&, Eigen::Matrix<double, 6, 2>& value) {
        value.setZero();
        value(5, 0) = value(5, 1) = 1;
    };
    Eigen::Matrix4d refinement_target = Eigen::Matrix4d::Identity();
    refinement_target(2, 3) = 1;
    yaik_cpp::numericalRefinement<2>(fk, jacobian, refinement_target, q, 1e-4, {{true, false}});
    assert(q[0] == 0 && std::abs(q[1] - 1) < 1e-8);
    Eigen::Matrix4d target = Eigen::Matrix4d::Identity();
    target(2, 3) = 1;
    const auto output = regression_ik::computeIK(target);
    assert(output.size() == 1 && std::abs(output[0][0] - 1) < 1e-12);
    const auto parameterized_target = parameterized_ik::computeFK({{1, 0.25}}, 0.4);
    const auto parameterized_output = parameterized_ik::computeIK(parameterized_target, 0.4, 0.25);
    assert(parameterized_output.size() == 1);
    assert(std::abs(parameterized_output[0][0] - 1) < 1e-12 && parameterized_output[0][1] == 0.25);
    assert(parameterized_ik::computeIK(parameterized_target, std::numeric_limits<double>::quiet_NaN(), 0.25).empty());
    regression_ik::RawIKWorksace workspace;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()}) {
                Eigen::Matrix4d invalid_target = target;
                invalid_target(row, col) = invalid;
                assert(regression_ik::computeIK(invalid_target).empty());
                auto reused_output = output;
                regression_ik::computeIK(invalid_target, workspace, reused_output);
                assert(reused_output.empty());
            }
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'regression.cpp'
            executable = Path(directory) / 'regression'
            source.write_text('#define main generated_main\n' + code + '\n#undef main\n' + main)
            root = Path(__file__).resolve().parents[1]
            compiled = subprocess.run(['g++', '-std=c++14', '-O0', '-I' + str(root), '-I' + str(eigen),
                                       str(source), '-o', str(executable)], capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
