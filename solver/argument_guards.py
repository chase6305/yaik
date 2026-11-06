"""Inverse-trig domain checks with a small double-roundoff allowance.

Only a guard for the actual asin/acos argument is adjusted. Other inequalities
and denominator guards stay unchanged; returned candidates still need FK.
"""

import math
import sympy as sp


UNIT_ROUNDOFF_MARGIN = sp.Rational(1, 2**46)  # 64 double epsilons.


def runtime_argument_guards(expressions, checkers):
    result = []
    for expression, checker in zip(expressions, checkers):
        replacements = {}
        arguments = {sp.Abs(inverse.args[0]) for inverse in expression.atoms(sp.asin, sp.acos)}
        for original in checker.atoms(sp.LessThan, sp.GreaterThan):
            argument, bound = ((original.lhs, original.rhs) if isinstance(original, sp.LessThan)
                               else (original.rhs, original.lhs))
            if argument in arguments and bound.is_number and (bound-1).is_zero:
                replacements[original] = argument <= 1+UNIT_ROUNDOFF_MARGIN
        result.append(checker.xreplace(replacements))
    return result


def evaluate_inverse_trig_boundary(expression, values):
    """Clamp only tiny out-of-domain arguments before SymPy evaluates them."""
    replacements = {}
    for inverse in expression.atoms(sp.asin, sp.acos):
        argument = inverse.args[0].subs(values)
        try:
            value = float(argument)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and 1 < abs(value) <= 1+float(UNIT_ROUNDOFF_MARGIN):
            replacements[inverse] = inverse.func(sp.S.One if value > 0 else -sp.S.One)
    return expression.xreplace(replacements).subs(values)
