"""Conservative symbolic checks for equations assumed zero on a branch."""

import sympy as sp
from functools import lru_cache

from solver.solved_variable import DegenerateType


@lru_cache(maxsize=256)
def _linear_branch_substitutions(zero_equations):
    """Use exact affine identities, retaining all unconstrained variables."""
    linear = []
    for equation in zero_equations:
        equation = equation.xreplace({value: sp.Rational(value) for value in equation.atoms(sp.Float)})
        symbols = sorted(equation.free_symbols, key=str)
        if not symbols:
            continue
        try:
            polynomial = sp.Poly(equation, *symbols)
        except sp.PolynomialError:
            continue
        if polynomial.total_degree() <= 1:
            linear.append(equation)
    if not linear:
        return {}
    symbols = sorted(set().union(*(equation.free_symbols for equation in linear)), key=str)
    solutions = sp.solve(linear, symbols, dict=True, simplify=False, check=False)
    if len(solutions) != 1:
        return {}
    replacements = solutions[0]
    if not all(sp.expand(equation.subs(replacements, simultaneous=True)) == 0 for equation in linear):
        return {}
    return replacements


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
    if reduced == 0 or sp.expand(reduced) == 0:
        return True
    # Prefix assignments and selected variable-value branches can imply, for
    # example, y = pi-x. Substitution then proves sin(x+y) == 0 without treating
    # either free angle as zero or invoking an unrestricted trig simplifier.
    bindings = _linear_branch_substitutions(zero_equations)
    if bindings:
        reduced = reduced.xreplace({value: sp.Rational(value) for value in reduced.atoms(sp.Float)})
        reduced = reduced.subs(bindings, simultaneous=True)
    return reduced == 0 or sp.expand(reduced) == 0


def vanishes_on_branch(expression, zero_equations):
    return _vanishes_on_branch(sp.sympify(expression), tuple(map(sp.sympify, zero_equations)))


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
