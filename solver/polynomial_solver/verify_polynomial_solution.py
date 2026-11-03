from solver.solved_variable import SolutionMethod
from python_run_import import compute_polynomial_real_roots
from typing import List, Dict, Tuple
import sympy as sp
import numpy as np


def verify_polynomial_solution(
        polynomials: List[sp.Expr],
        poly_reduce_output,
        parameter_values_to_test: List[Dict[sp.Symbol, float]],
        log_if_not_match: bool = False,
        max_n_test_cases: int = 100) -> Tuple[bool, float]:
    """
    Given a polynomial solution reduction input/output and a set of test data, check whether the reduced
    polynomial is a solution.
    @param polynomials
    @param poly_reduce_output
    @param parameter_values_to_test
    @param log_if_not_match
    @param max_n_test_cases
    @return a tuple, first is a bool indicating whether all value match
                     the second is the ration of passed case (which <= len(parameter_values_to_test))
    """
    # Extract the output
    if poly_reduce_output is None or not polynomials or not parameter_values_to_test or max_n_test_cases <= 0:
        return False, 0
    poly_dict, solved_term, solved_unknown, solution_method = poly_reduce_output

    # Make the coefficient list
    poly_order = -1
    for order in poly_dict:
        if order >= poly_order:
            poly_order = order

    coefficients = [sp.S.Zero] * (poly_order + 1)
    for order, (numerator, denominator) in poly_dict.items():
        coefficients[poly_order - order] = sp.sympify(numerator) / denominator
    coefficient_symbols = sorted(set().union(*(expr.free_symbols for expr in coefficients)), key=str)
    equation_symbols = sorted(set().union(*(expr.free_symbols for expr in polynomials)), key=str)
    evaluate_coefficients = sp.lambdify(coefficient_symbols, coefficients, modules='numpy', cse=True)
    evaluate_equations = sp.lambdify(equation_symbols, polynomials, modules='numpy', cse=True)

    # Compile once, then evaluate all samples and candidate roots numerically.
    n_matched = 0
    all_value_match = True
    n_test_case = min(len(parameter_values_to_test), max_n_test_cases)
    index2test = np.random.choice(np.arange(len(parameter_values_to_test)), size=n_test_case, replace=False)
    for i in range(n_test_case):
        idx = index2test[i]
        value_map_i = parameter_values_to_test[idx]
        try:
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                p_coefficients = evaluate_coefficients(*(float(value_map_i[s]) for s in coefficient_symbols))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            all_value_match = False
            continue

        # Find the root
        poly_roots = compute_polynomial_real_roots(p_coefficients)
        solutions = list()
        for this_root in poly_roots:
            if solution_method == SolutionMethod.PolynomialDirect.name:
                solutions.append(this_root)
            if abs(this_root) > 1 + 1e-6:
                continue
            this_root = float(np.clip(this_root, -1, 1))
            if solution_method == SolutionMethod.PolynomialSin.name:
                first_angle = np.arcsin(this_root)
                solutions.append(first_angle)
                solutions.append(np.pi - first_angle)
            if solution_method == SolutionMethod.PolynomialCos.name:
                first_angle = np.arccos(this_root)
                solutions.append(first_angle)
                solutions.append(-first_angle)

        # Try the root
        contain_solution = False
        diff_values = list()
        for j in range(len(solutions)):
            try:
                values = [solutions[j] if symbol == solved_unknown.symbol else float(value_map_i[symbol])
                          for symbol in equation_symbols]
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    residuals = np.asarray(evaluate_equations(*values), dtype=complex)
            except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
                continue
            if np.all(np.isfinite(residuals)) and np.all(np.abs(residuals) < 1e-10):
                contain_solution = True
                break
            diff_values.extend(np.abs(residuals).tolist())

        # Update the statistic
        all_value_match = all_value_match and contain_solution
        if log_if_not_match and (not contain_solution):
            print(diff_values)
        if contain_solution:
            n_matched += 1
    return all_value_match, float(n_matched) / float(n_test_case)
