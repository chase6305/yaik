from solver.solved_variable import VariableSolution
from solver.equation_types import ScalarEquationType
from solver.equation_utils import CollectedEquations
from solver.unary_solvers.tangent_solver import UnaryVariableSolver, UnaryTangentSolver
from solver.unary_solvers.one_var_algebra import UnaryOneVariableAlgebraSolver
from solver.unary_solvers.sin_and_cos import UnarySinAndCosSolver
from solver.unary_solvers.linear_sin_cos_type_1 import UnaryLinearSolverType_1
from solver.unary_solvers.asin_or_acos import UnaryArcSinSolver, UnaryArcCosSolver
from typing import List, Dict
import sympy as sp
import copy
import multiprocessing
import time


def parallel_solve_processor(dict_in) -> List[VariableSolution]:
    """
    Just a parallel processor that encapsulate the solver.try_solve
    """
    collected_equations = dict_in['collected_equations']
    if 'solver_step' in dict_in:
        return dict_in['solver_step'].solve_step(collected_equations, dict_in['unknowns'])
    var_to_try = dict_in['var_to_try']
    solver_in: UnaryVariableSolver = dict_in['solver']
    unknowns = dict_in['unknowns']
    solver_output = solver_in.try_solve(collected_equations, var_to_try, unknowns)
    return solver_output


def parallel_solve_processor_list_output(dict_in, output_list, lock=None, task_index: int = 0):
    this_output = parallel_solve_processor(dict_in)

    # Append a result
    if lock is not None:
        lock.acquire()
    output_list.append((task_index, this_output))
    if lock is not None:
        lock.release()


def single_solve_processor(dict_in, connection):
    try:
        connection.send(parallel_solve_processor(dict_in))
    finally:
        connection.close()


class UnarySolverStep(object):

    def __init__(self):
        self._unary_algebra_solver = UnaryOneVariableAlgebraSolver()
        self._other_solvers: List[UnaryVariableSolver] = list()

    def append_solver(self, solver: UnaryVariableSolver):
        self._other_solvers.append(solver)

    @staticmethod
    def make_default():
        default_solver = UnarySolverStep()
        default_solver.append_solver(UnarySinAndCosSolver())
        default_solver.append_solver(UnaryLinearSolverType_1())
        default_solver.append_solver(UnaryTangentSolver())
        default_solver.append_solver(UnaryArcSinSolver())
        default_solver.append_solver(UnaryArcCosSolver())
        return default_solver

    def _try_algebra_solution(self,
                              collected_equations: CollectedEquations,
                              unknowns: List[sp.Symbol]) -> (List[VariableSolution], bool):
        """
        Try solving the equations using algebra solver.
        :return: the first tuple element is the variable solution,
                 the second tuple element is a bool indicating whether we found sum-of-angle solution
        """
        # The final output
        solutions = list()

        # First try algebra solver, record if we have solutions from sum-of-angle equations
        has_soa_solution: bool = False
        for var_to_try in unknowns:
            algebra_sol = self._unary_algebra_solver.try_solve(
                collected_equations, var_to_try, unknowns)
            for this_sol in algebra_sol:
                scalar_equs = this_sol.solved_from_scalar_equations
                assert len(scalar_equs) == 1
                if scalar_equs[0].equation_type == ScalarEquationType.SumOfAngle.name:
                    has_soa_solution = True
            solutions.extend(algebra_sol)

        # Just return (because SOA equations very good ones)
        return solutions, has_soa_solution

    def solve_step(self, collected_equations: CollectedEquations,
                   unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        # The final output
        solutions, has_soa_solution = self._try_algebra_solution(
            collected_equations, unknowns)

        # Just return (because SOA equations very good ones)
        if has_soa_solution:
            print('We have solutions from sum-of-angle equations, thus do not continue')
            return solutions

        # Get as many solutions as possible
        for var_to_try in unknowns:
            for solver in self._other_solvers:
                sol = solver.try_solve(collected_equations, var_to_try, unknowns)
                if len(sol) > 0:
                    solutions.extend(sol)
        return solutions

    def solve_step_parallel(self,
                            collected_equations: CollectedEquations,
                            unknowns: List[sp.Symbol],
                            with_timeout: bool = False,
                            timeout_seconds: float = 300):
        # Process startup costs more than these small rule searches.
        small_search = len(unknowns) == 1 and sum(
                sp.count_ops(equ.lhs) + sp.count_ops(equ.rhs)
                for equ in collected_equations.one_unknown_equations) <= 100
        if small_search:
            if not with_timeout:
                return self.solve_step(collected_equations, unknowns)
            output = self.parallel_run([dict(solver_step=self, collected_equations=collected_equations,
                                             unknowns=unknowns)], timeout_seconds=timeout_seconds)
            return output[0] if output else []

        # The final output
        deadline = time.monotonic() + timeout_seconds
        if with_timeout:
            algebra_args = [dict(solver=self._unary_algebra_solver, collected_equations=collected_equations,
                                 var_to_try=unknown, unknowns=unknowns) for unknown in unknowns]
            output = self.parallel_run(algebra_args, timeout_seconds=timeout_seconds)
            solutions = [solution for group in output for solution in group]
            has_soa_solution = any(solution.solved_from_scalar_equations[0].is_sum_of_angle
                                   for solution in solutions)
        else:
            solutions, has_soa_solution = self._try_algebra_solution(collected_equations, unknowns)

        # Just return (because SOA equations very good ones)
        if has_soa_solution:
            print('We have solutions from sum-of-angle equations, thus do not continue')
            return solutions

        # Make the parallel input args
        input_args = list()
        for var_to_try in unknowns:
            for this_solver in self._other_solvers:
                dict_in = dict()
                dict_in['collected_equations'] = collected_equations
                dict_in['var_to_try'] = var_to_try
                dict_in['solver'] = this_solver
                dict_in['unknowns'] = unknowns
                input_args.append(dict_in)

        # OK, map it
        if with_timeout:
            output = self.parallel_run(input_args, timeout_seconds=max(0.0, deadline - time.monotonic()))
        else:
            n_process = min(32, len(input_args))
            n_process = max(n_process, 1)
            print('Try solving the equations using rule solver. The candidate number: ', len(input_args))
            output: List[List[VariableSolution]] = list()
            with multiprocessing.Pool(n_process) as pool:
                output = pool.map(parallel_solve_processor, input_args)

        # Collect the result
        for this_sol in output:
            if len(this_sol) > 0:
                solutions.extend(this_sol)
        return solutions

    def get_solvers(self):
        solver_list = list()
        solver_list.append(self._unary_algebra_solver)
        for solver in self._other_solvers:
            solver_list.append(solver)
        return solver_list

    @staticmethod
    def parallel_run(input_args: List[Dict], timeout_seconds: float = 300) -> List[List[VariableSolution]]:
        """Return completed tasks in input order within one wall-clock budget."""
        if not input_args or timeout_seconds <= 0:
            return []
        deadline = time.monotonic() + timeout_seconds
        if len(input_args) == 1:
            receive, send = multiprocessing.Pipe(duplex=False)
            processor = None
            try:
                processor = multiprocessing.Process(target=single_solve_processor, args=(input_args[0], send))
                processor.start()
                send.close()
                if receive.poll(max(0.0, deadline - time.monotonic())):
                    try:
                        return [receive.recv()]
                    except EOFError:
                        pass
                return []
            finally:
                send.close()
                receive.close()
                if processor is not None and processor.pid is not None:
                    processor.join(timeout=max(0.0, deadline - time.monotonic()))
                    if processor.is_alive():
                        processor.terminate()
                    processor.join()
        n_processor = min(32, int(len(input_args)))
        n_processor = max(n_processor, 1)
        print('Try solving the equations using rule solver. The candidate number: ', len(input_args))
        lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        shared_result_list = manager.list()
        processed_offset = 0
        processor_list = []
        try:
            while processed_offset < len(input_args):
                processor_list = []
                for i in range(n_processor):
                    offset_i = processed_offset + i
                    if offset_i < len(input_args):
                        processor = multiprocessing.Process(
                            target=parallel_solve_processor_list_output,
                            args=(input_args[offset_i], shared_result_list, lock, offset_i))
                        processor_list.append(processor)

                for p in processor_list:
                    p.start()
                for p in processor_list:
                    p.join(timeout=max(0.0, deadline - time.monotonic()))
                    if p.is_alive():
                        p.terminate()
                    p.join()

                processed_offset += n_processor
                if time.monotonic() >= deadline:
                    break
            return [copy.deepcopy(elem) for _, elem in sorted(shared_result_list, key=lambda item: item[0])]
        finally:
            for p in processor_list:
                if getattr(p, 'pid', None) is not None:
                    if p.is_alive():
                        p.terminate()
                    p.join()
            manager.shutdown()
