"""Conservative symbolic checks for equations assumed zero on a branch."""

import sympy as sp
from functools import lru_cache

from solver.solved_variable import DegenerateType


@lru_cache(maxsize=4096)
def _vanishes_on_branch(expression, zero_equations):
    replacements = {}
    for equation in zero_equations:
        equation = sp.factor_terms(equation)
        if not equation.free_symbols:
            continue
        _, body = equation.as_coeff_Mul()
        replacements[body] = sp.S.Zero
        replacements[-body] = sp.S.Zero
    reduced = sp.factor_terms(expression).subs(replacements, simultaneous=True)
    # Exact substitution and polynomial expansion suffice for a conservative
    # proof. General trigonometric simplification here can dominate generation
    # time for formulas that do not vanish on this branch.
    return reduced == 0 or sp.expand(reduced) == 0


def vanishes_on_branch(expression, zero_equations):
    return _vanishes_on_branch(sp.sympify(expression), tuple(zero_equations))


def formula_degenerates_on_branch(solution, zero_equations):
    if not zero_equations:
        return False
    record = solution.degenerate_record
    if record.type == DegenerateType.DegenerateIfAllEquationsZero.name:
        return bool(record.equations) and all(vanishes_on_branch(equation, zero_equations)
                                              for equation in record.equations)
    if record.is_degenerate_on_variable_values:
        for index in range(record.count_number_variable_solutions()):
            if all(vanishes_on_branch(variable-value, zero_equations)
                   for variable, value in record.get_variable_solution(index)):
                return True
    return False
