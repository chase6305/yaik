"""Find implemented fixed-angle branch centers under exact model periodicity."""
import math

import sympy as sp

from codegen.skeleton.tree_node import SolvedVariableDispatcherNode


def joint_is_periodic(robot, symbol):
    """Prove invariance under a 2*pi shift, including base/tool transforms."""
    def unchanged(expression, angular=False):
        expression = sp.sympify(expression)
        if symbol not in expression.free_symbols:
            return True
        # Runtime parameters are finite reals; symbolic integer/positive
        # assumptions are not enforced by the generated argument guards.
        replacements = {item: sp.Dummy(real=True) for item in expression.free_symbols}
        phase = replacements[symbol]
        expression = expression.xreplace(replacements)
        difference = sp.trigsimp(expression.subs(phase, phase+2*sp.pi)-expression)
        if angular:
            turns = sp.simplify(difference/(2*sp.pi))
            return not turns.free_symbols and turns.is_integer is True
        return difference.is_zero is True

    for dh in robot.dh_params:
        if not (unchanged(dh.alpha, True) and unchanged(dh.theta, True)
                and unchanged(dh.a) and unchanged(dh.d)):
            return False
    for frame in (robot.get_pre_transform_sp(), robot.get_post_transform_sp()):
        if frame is not None and not all(unchanged(entry) for entry in frame):
            return False
    return True


def fixed_joint_branch_centers(tree):
    fixed = {unknown.symbol for unknown in tree.robot.unknowns
             if unknown.is_revolute and unknown.symbol in tree.robot.unknown_as_parameter_more_dof}
    candidates = set()
    for node in tree.node_list:
        if not isinstance(node, SolvedVariableDispatcherNode):
            continue
        for index, child in enumerate(node.children[1:]):
            if child is None:
                continue
            for symbol, value in node.get_branch_variable_value(index).items():
                value = sp.sympify(value)
                if symbol in fixed and not value.free_symbols and value.is_real is True and value.is_finite is True:
                    center = float(value)
                    if math.isfinite(center):
                        candidates.add((symbol, center))
    periodic = {symbol for symbol, _ in candidates if joint_is_periodic(tree.robot, symbol)}
    return sorted((item for item in candidates if item[0] in periodic), key=lambda item: (str(item[0]), item[1]))
