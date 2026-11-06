"""Print scalar predicates without evaluating a zero denominator first."""

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction


def _denominator_checks(expression):
    checks = []
    # Inner denominators must be checked before expressions that depend on them.
    for node in sp.postorder_traversal(expression):
        if node.is_Pow and node.exp.is_negative:
            check = sp.Ne(node.base, 0)
            if check != sp.S.true and check not in checks:
                checks.append(check)
    return checks


def predicate_code(expression, printer, language):
    """Return predicate code, or None for an ordinary arithmetic expression."""
    if not isinstance(expression, (Relational, BooleanFunction)):
        return None
    conjunction, disjunction = ('and', 'or') if language == 'python' else ('&&', '||')

    def protect(value, checks):
        if not checks:
            return value
        return '('+f' {conjunction} '.join([*(f'({printer(check)})' for check in checks), '('+value+')'])+')'

    def render(value, negated=False):
        if value.func == sp.Not:
            return render(value.args[0], not negated)
        if value.func in (sp.And, sp.Or):
            arguments = value.args
            is_conjunction = (value.func == sp.And) != negated
            if is_conjunction:
                # Cheap coefficient guards also reject tiny denominators before
                # a range guard can overflow. Boolean operands are side-effect free.
                arguments = sorted(arguments, key=lambda item: (len(_denominator_checks(item)),
                                                               sp.count_ops(item), sp.default_sort_key(item)))
            operator = conjunction if is_conjunction else disjunction
            return '('+f' {operator} '.join('('+render(item, negated)+')' for item in arguments)+')'
        return protect(printer(sp.Not(value) if negated else value), _denominator_checks(value))

    return render(expression)
