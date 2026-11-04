"""Bounded symbolic searches preserve candidates without quadratic scheduling costs."""
import contextlib
import io
import itertools
import multiprocessing
import os
import random
import time
import tracemalloc
from unittest import TestCase, mock, skipUnless

import numpy as np
import sympy as sp

from solver.equation_types import CollectedEquations, ScalarEquation, Unknown
from solver.equation_utils import scalar_equation_by_unknowns
from solver.polynomial_solver.find_candidate import collect_equation_by_variables
from solver.solved_variable import DegenerateType, SolutionDegenerateRecord, SolutionMethod, VariableSolution
from solver.sos_transform import sum_of_square_transform_pairwise
from solver.unary_solvers.unary_solver_step import UnarySolverStep
from solver.degenerate_analyse.detect_soa_solution import (
    _check_soa_equations_numerically, check_zero_processor, detect_soa_solution)
from solver.degenerate_analyse.analyser_step import (
    CheckConstantAnalyserStep, DegenerateAnalyserContext, DegenerateAnalyserOption)
from solver.soa_transform import SumOfAngleAccumulator
import solver.sos_transform as sos
import solver.general_6dof.numerical_reduce_solver as general
import solver.polynomial_solver.polynomial_solve as polynomial


class BlockingAlgebra:
    def __init__(self):
        self.parent_pid = os.getpid()

    def try_solve(self, *args):
        if os.getpid() == self.parent_pid:
            raise AssertionError('algebra expansion must be isolated')
        time.sleep(5)
        return []


class DelayedRule:
    def __init__(self, value, delay):
        self.value, self.delay = value, delay

    def try_solve(self, equations, variable, unknowns):
        time.sleep(self.delay)
        return [VariableSolution.make_explicit_solution(
            variable, [sp.Integer(self.value)], SolutionMethod.OneVariableAlgebra.name)]


def delayed_sos(task):
    time.sleep(task['delay'])
    return ScalarEquation(sp.Symbol('value'), sp.Integer(task['index']))


def delayed_general(value):
    time.sleep(0.08 if value == 0 else 0)
    return value


def delayed_polynomial(equations, unknowns, parameters, reduce_to_cos):
    variable = unknowns[0]
    time.sleep(0.08 if variable.symbol.name == 'x' else 0)
    return {0: (sp.S.Zero, sp.S.One), 1: (sp.S.One, sp.S.One)}, variable.symbol, variable, \
        SolutionMethod.PolynomialDirect.name


class TestSearchEfficiency(TestCase):
    def test_constant_analysis_cache_keeps_new_records_and_tracks_parameters(self):
        parameter = sp.Symbol('parameter')
        record = SolutionDegenerateRecord.record_all_equations([parameter])
        context = DegenerateAnalyserContext(parameter_value={parameter: 1})
        option = DegenerateAnalyserOption()
        analyser = CheckConstantAnalyserStep()
        with mock.patch('solver.degenerate_analyse.analyser_step.cast_expr_to_float',
                        wraps=float) as evaluate:
            first = analyser.perform_analyse(record, context, option)
            second = analyser.perform_analyse(record, context, option)
            self.assertIsNot(first, second)
            self.assertEqual(first.type, DegenerateType.AlwaysNonDegenerate.name)
            self.assertEqual(second.type, first.type)
            self.assertEqual(evaluate.call_count, 1)
            context.parameter_value[parameter] = 0
            self.assertIsNone(analyser.perform_analyse(record, context, option))
            self.assertIsNone(analyser.perform_analyse(record, context, option))
            self.assertEqual(evaluate.call_count, 2)
            context.parameter_value[parameter] = 2
            self.assertIsNotNone(analyser.perform_analyse(record, context, option))
            self.assertEqual(evaluate.call_count, 3)
            fresh_analyser = CheckConstantAnalyserStep()
            self.assertIsNotNone(fresh_analyser.perform_analyse(record, context, option))
            self.assertEqual(evaluate.call_count, 4)
        for zero in (sp.Float(0.0), -0.0):
            with self.subTest(zero=zero):
                context.parameter_value[parameter] = zero
                fresh_analyser = CheckConstantAnalyserStep()
                self.assertIsNone(fresh_analyser.perform_analyse(record, context, option))
                self.assertIsNone(fresh_analyser.perform_analyse(record, context, option))

    def test_compiled_soa_checks_match_scalar_reference(self):
        x, y, total = sp.symbols('x y total')
        equations = [ScalarEquation(total, x + y), ScalarEquation(sp.cos(total), sp.cos(x + y))]
        samples = [{x: 0.0, y: value, total: value} for value in (0.0, 0.2, -1.0, 2.7)]
        self.assertTrue(all(check_zero_processor(dict(scalar_equation=equation, test_map=sample))
                            for equation in equations for sample in samples))
        with mock.patch('sympy.lambdify', wraps=sp.lambdify) as compile_equations:
            self.assertTrue(_check_soa_equations_numerically(equations, samples))
        compile_equations.assert_called_once()
        self.assertFalse(_check_soa_equations_numerically([ScalarEquation(sp.S.Zero, sp.Float('1e-6'))], [{}]))
        self.assertTrue(_check_soa_equations_numerically([ScalarEquation(sp.S.Zero, sp.Float('0.999e-6'))], [{}]))

    def test_compiled_soa_checks_reject_invalid_or_unevaluable_sides(self):
        p = sp.Symbol('p')
        unknown_function = sp.Function('unknown_function')(p)
        cases = [([ScalarEquation(1 / p, 1 / p)], {p: 0}),
                 ([ScalarEquation(p, p)], {}),
                 ([ScalarEquation(p, p)], {p: float('nan')}),
                 ([ScalarEquation(sp.atan(p), sp.atan(p))], {p: float('inf')}),
                 ([ScalarEquation(sp.oo, sp.oo)], {}),
                 ([ScalarEquation(sp.I, sp.I)], {}),
                 ([ScalarEquation(unknown_function, unknown_function)], {p: 1}),
                 ([ScalarEquation(sp.diff(unknown_function, p), sp.S.Zero)], {p: 1})]
        for equations, sample in cases:
            with self.subTest(equations=equations, sample=sample):
                self.assertFalse(_check_soa_equations_numerically(equations, [sample]))

    def test_batched_soa_checks_constant_sides_and_every_sample(self):
        x, y, total = sp.symbols('x y total')
        equations = [ScalarEquation(total, x + y),
                     ScalarEquation(sp.cos(total), sp.cos(x + y)),
                     ScalarEquation(sp.S.One, sp.S.One)]
        rng = np.random.default_rng(20261104)
        samples = [{x: a, y: b, total: a + b} for a, b in rng.uniform(-3, 3, (500, 2))]
        self.assertTrue(_check_soa_equations_numerically(equations, samples))
        samples[-1] = {**samples[-1], total: samples[-1][total] + 1e-4}
        self.assertFalse(_check_soa_equations_numerically(equations, samples))
        samples[-1][total] = np.nan
        self.assertFalse(_check_soa_equations_numerically(equations, samples))
        for residual, accepted in ((np.nextafter(1e-6, 0), True), (1e-6, False),
                                   (np.nextafter(1e-6, np.inf), False)):
            self.assertEqual(_check_soa_equations_numerically(
                [ScalarEquation(sp.S.Zero, x)], [{x: 0.0}, {x: residual}]), accepted)

    def test_batched_soa_retains_scalar_only_expression_support(self):
        x = sp.Symbol('x')
        positive_part = sp.Piecewise((0, x < 0), (x, True))
        equation = ScalarEquation(sp.Max(0, x), positive_part)
        self.assertTrue(_check_soa_equations_numerically(
            [equation], [{x: value} for value in (-2.0, 0.0, 0.5, 3.0)]))

    def test_batched_soa_keeps_custom_callbacks_scalar(self):
        from sympy.utilities.lambdify import implemented_function
        x = sp.Symbol('x')
        scalar_sum = implemented_function('scalar_sum', np.sum)
        # f(x)=x for scalar inputs; summing the batch would incorrectly cancel.
        self.assertFalse(_check_soa_equations_numerically(
            [ScalarEquation(sp.S.Zero, scalar_sum(x))], [{x: 1.0}, {x: -1.0}]))

    def test_batched_soa_preserves_scalar_exception_semantics(self):
        x = sp.Symbol('x')
        reciprocal_square = sp.Pow(sp.Pow(x, 2, evaluate=False), -1, evaluate=False)
        # NumPy can otherwise hide an overflow behind a finite reciprocal.
        self.assertFalse(_check_soa_equations_numerically(
            [ScalarEquation(reciprocal_square, sp.S.Zero)], [{x: 1e200}, {x: -1e200}]))
        piecewise = sp.Piecewise((0, sp.Eq(x, 0)), (1 / x, True))
        # Scalar evaluation rejects the undefined inactive branch at x=0.
        self.assertFalse(_check_soa_equations_numerically(
            [ScalarEquation(piecewise, piecewise)], [{x: 0.0}, {x: 1.0}]))
        # Scalar NumPy evaluation already permits this underflow to zero.
        self.assertTrue(_check_soa_equations_numerically(
            [ScalarEquation(sp.exp(-x), sp.S.Zero)], [{x: 1000.0}, {x: 1100.0}]))

    def test_soa_detection_keeps_zero_representative_without_processes(self):
        x, y, total = sp.symbols('x y total')
        accumulator = SumOfAngleAccumulator([total], [], {total: x + y}, {x + y: total})
        context = DegenerateAnalyserContext()
        context.solved_variables = [VariableSolution.make_explicit_solution(
            total, [sp.S.Zero], SolutionMethod.DefaultMethod.name)]
        context.numerical_test_cases = [{x: 0.3, y: 0.6, total: 0.9}]
        equations = CollectedEquations([], [ScalarEquation(sp.cos(total), sp.cos(x + y))], [])
        with mock.patch('multiprocessing.Pool', side_effect=AssertionError('unexpected process')), \
                contextlib.redirect_stdout(io.StringIO()):
            output = detect_soa_solution(equations, [Unknown(x), Unknown(y)], accumulator,
                                         context, DegenerateAnalyserOption())
        self.assertEqual(output.solved_variable, x)
        self.assertEqual(output.explicit_solutions, [sp.S.Zero])

    @staticmethod
    def candidates(n, budget):
        x, y = sp.symbols('x y')
        equations = [ScalarEquation(sp.Integer(i), x + i * y) for i in range(1, n + 1)]
        with mock.patch.object(sos, 'parallel_run', return_value=[]) as workers, \
                contextlib.redirect_stdout(io.StringIO()):
            sum_of_square_transform_pairwise(
                equations, [x, y], None, only_include_position_expr=False,
                max_pairwise_sos_candidates=budget)
        return [(int(task['equ_1'].lhs) - 1, int(task['equ_2'].lhs) - 1)
                for task in workers.call_args.args[0]]

    def test_small_sos_search_keeps_original_pair_order(self):
        self.assertEqual(self.candidates(8, 100), list(itertools.combinations(range(8), 2)))

    def test_sampled_sos_search_preserves_seeded_pairs(self):
        state = random.getstate()
        try:
            random.seed(12345)
            actual = self.candidates(31, 7)
        finally:
            random.setstate(state)
        expected = random.Random(12345).sample(list(itertools.combinations(range(31), 2)), 7)
        self.assertEqual(actual, expected)

    def test_sos_candidate_budget_also_bounds_scheduling_memory(self):
        # The old implementation allocated 79,800 dictionaries before keeping 1,000.
        tracemalloc.start()
        try:
            candidates = self.candidates(400, 1000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(candidates), 1000)
        self.assertLess(peak, 8 * 1024 * 1024)

    def test_small_unary_search_batches_all_rules_in_one_isolated_worker(self):
        x, a, b = sp.symbols('x a b')
        equations = scalar_equation_by_unknowns(
            [ScalarEquation(a, sp.sin(x)), ScalarEquation(b, sp.cos(x))], [x])
        step = UnarySolverStep.make_default()
        expected = step.solve_step(equations, [x])
        with mock.patch('multiprocessing.Manager', side_effect=AssertionError('unnecessary manager')), \
                mock.patch('multiprocessing.Process', wraps=multiprocessing.Process) as workers:
            actual = step.solve_step_parallel(equations, [x], with_timeout=True)
        workers.assert_called_once()
        signature = lambda solutions: sorted((s.solution_method, tuple(map(str, s.explicit_solutions)))
                                             for s in solutions)
        self.assertEqual(signature(actual), signature(expected))

    def test_algebra_is_isolated_and_terminated_for_small_and_complex_searches(self):
        x, y = sp.symbols('x y')
        before = {child.pid for child in multiprocessing.active_children()}
        for unknowns in ([x], [x, y]):
            with self.subTest(unknowns=unknowns):
                step = UnarySolverStep.make_default()
                step._unary_algebra_solver = BlockingAlgebra()
                with contextlib.redirect_stdout(io.StringIO()):
                    output = step.solve_step_parallel(
                        CollectedEquations([], [], []), unknowns, with_timeout=True, timeout_seconds=0.05)
                self.assertEqual(output, [])
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_rule_results_follow_task_order_instead_of_completion_order(self):
        x = sp.Symbol('x')
        tasks = [dict(solver=DelayedRule(value, 0.08 if i == 0 else 0),
                      var_to_try=x, unknowns=[x], collected_equations=CollectedEquations([], [], []))
                 for i, value in enumerate([3, 1, 2])]
        with contextlib.redirect_stdout(io.StringIO()):
            output = UnarySolverStep.parallel_run(tasks, timeout_seconds=2)
        self.assertEqual([int(group[0].explicit_solutions[0]) for group in output], [3, 1, 2])

    def test_worker_startup_failures_still_shutdown_the_manager(self):
        for search in (UnarySolverStep.parallel_run, sos.parallel_run):
            with self.subTest(search=search.__module__), \
                    mock.patch('multiprocessing.Manager') as manager, \
                    mock.patch('multiprocessing.Process', side_effect=RuntimeError('cannot start worker')):
                with self.assertRaisesRegex(RuntimeError, 'cannot start worker'):
                    search([{}, {}])
                manager.return_value.shutdown.assert_called_once()

    @skipUnless(multiprocessing.get_start_method() == 'fork', 'requires inherited worker stubs')
    def test_sos_and_general_results_follow_task_order(self):
        tasks = [dict(index=i, delay=0.08 if i == 0 else 0) for i in range(3)]
        with mock.patch.object(sos, 'parallel_processor_dict', delayed_sos):
            output = sos.parallel_run(tasks, timeout_seconds=2)
        self.assertEqual([int(equation.rhs) for equation in output], [0, 1, 2])
        with mock.patch.object(general, 'build_solver_input', return_value=list(range(3))), \
                mock.patch.object(general, 'parallel_solve_processor', delayed_general), \
                contextlib.redirect_stdout(io.StringIO()):
            output = general.try_solve_parallel_v2(None)
        self.assertEqual(output, [0, 1, 2])

    def test_polynomial_candidates_have_stable_variable_order(self):
        a, b, c, d = sp.symbols('a b c d')
        equations = [ScalarEquation(sp.S.Zero, expr) for expr in [a + b, a - b, c + d, c - d, a + c, a - c]]
        output = collect_equation_by_variables(
            CollectedEquations([], equations, []), [Unknown(symbol) for symbol in (a, b, c, d)])
        self.assertEqual(list(output), [('a', 'b'), ('a', 'c'), ('c', 'd')])

    @skipUnless(multiprocessing.get_start_method() == 'fork', 'requires inherited worker stubs')
    def test_polynomial_results_follow_task_order(self):
        x, y, z = sp.symbols('x y z')
        with mock.patch.object(polynomial, 'collect_equation_by_variables',
                               return_value={(str(s),): [s] for s in (x, y, z)}), \
                mock.patch.object(polynomial, 'groebner_reduce', delayed_polynomial), \
                contextlib.redirect_stdout(io.StringIO()):
            output = polynomial.polynomial_try_solve(
                CollectedEquations([], [], []), [Unknown(s) for s in (x, y, z)], set(), use_permutation=False)
        self.assertEqual([solution.solved_variable for solution in output], [x, y, z])
