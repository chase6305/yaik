"""Small, recomputed certificates for unreachable exact-kinematics branches.

These checks deliberately do not certify floating-point evaluation or the IK
algorithm as a whole. They only prove that a particular serialized path has no
solution under the real modified-DH model.
"""

import sympy as sp

from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode
from fk.fk_equations import inv_Px, inv_Py, inv_Pz


def _valued_expression(expression, parameters):
    expression = sp.sympify(expression)
    expression = expression.xreplace({value: sp.Rational(value)
                                     for value in expression.atoms(sp.Float)
                                     if value.is_finite})
    return expression.subs(parameters)


def _exact_constant(expression, parameters):
    expression = _valued_expression(expression, parameters)
    return expression if expression.is_Rational else None


def _rigid_auxiliary_frames(robot, parameters):
    # A transpose is only the inverse rotation for a rigid frame. Approximate
    # or runtime-valued auxiliary matrices need their own numerical proof.
    for transform in (robot.get_pre_transform_sp(), robot.get_post_transform_sp()):
        if transform is None:
            continue
        transform = transform.applyfunc(lambda value: _valued_expression(value, parameters))
        if transform.shape != (4, 4) or any(value.free_symbols or value.is_real is not True or
                                           value.is_finite is not True for value in transform):
            return False
        if transform[3, :] != sp.Matrix([[0, 0, 0, 1]]):
            return False
        rotation = transform[:3, :3]
        gram = rotation.T*rotation-sp.eye(3)
        if any(sp.simplify(value) != 0 for value in gram) or sp.simplify(rotation.det()-1) != 0:
            return False
    return True


def _path_conditions(node):
    """Conditions that must hold when taking this coefficient-zero branch."""
    conditions = [(node.flatten_idx, node.degenerate_checker)]
    child, parent = node, node.parent
    while parent is not None:
        if isinstance(parent, EquationAllZeroDispatcherNode) and parent.degenerate_child is child:
            conditions.append((parent.flatten_idx, parent.degenerate_checker))
        child, parent = parent, parent.parent
    return conditions


def certify_unreachable_branch(tree, node):
    """Prove a small inverse-position box lies inside the arm's empty annulus.

    The norm of a modified-DH link translation is sqrt(a*a+d*d). Rotations
    preserve that norm. The reverse triangle inequality therefore gives
    |p| >= max(|a_i|, |d_i|) - sum(j != i)(|a_j|+|d_j|).
    For a rigid target |inverse_p|=|p|. All arithmetic below is rational and
    every position bound comes from an actual conjunct on the serialized path.
    Missing/runtime-valued geometry, disjunctions and unsupported predicates
    simply leave the branch unresolved.
    """
    if not isinstance(node, EquationAllZeroDispatcherNode) or not tree.is_node_in_tree(node):
        return None
    parameters = {}
    for symbol, value in tree.robot.parameters_value.items():
        exact = _exact_constant(value, {})
        if exact is not None:
            parameters[symbol] = exact
    conditions = _path_conditions(node)
    axes = (inv_Px, inv_Py, inv_Pz)
    bounds = {}
    for _, condition in conditions:
        conjuncts = condition.args if condition.func is sp.And else (condition,)
        for term in conjuncts:
            if not isinstance(term, (sp.LessThan, sp.StrictLessThan)):
                continue
            if term.lhs.func is not sp.Abs or term.lhs.args[0] not in axes:
                continue
            bound = _exact_constant(term.rhs, parameters)
            if bound is None or bound < 0:
                continue
            axis = term.lhs.args[0]
            bounds[axis] = min(bounds.get(axis, bound), bound)
    if len(bounds) != 3:
        return None
    if not _rigid_auxiliary_frames(tree.robot, parameters):
        return None
    real_symbols = ({unknown.symbol for unknown in tree.robot.unknowns} |
                    tree.robot.symbolic_parameters)
    real_substitution = {symbol: sp.Dummy(real=True) for symbol in real_symbols}
    translations = []
    for dh in tree.robot.dh_params:
        for angle in (dh.alpha, dh.theta):
            angle = sp.sympify(angle).subs(parameters)
            if (angle.free_symbols-real_symbols or
                    angle.xreplace(real_substitution).is_real is not True):
                return None
        a, d = _exact_constant(dh.a, parameters), _exact_constant(dh.d, parameters)
        if a is None or d is None:
            return None
        translations.append((a, d))
    if not translations:
        return None
    upper_lengths = [abs(a)+abs(d) for a, d in translations]
    lower_lengths = [max(abs(a), abs(d)) for a, d in translations]
    candidates = [lower+upper-sum(upper_lengths)
                  for lower, upper in zip(lower_lengths, upper_lengths)]
    radius_lower = max(candidates)
    radius_squared_upper = sum(bounds[axis]**2 for axis in axes)
    if radius_lower <= 0 or radius_lower**2 <= radius_squared_upper:
        return None
    return {
        'kind': 'serial_chain_annulus', 'arithmetic': 'exact_rational',
        'scope': 'real modified-DH kinematics and the serialized path predicates',
        'floating_point_evaluation_certified': False,
        'auxiliary_frames_verified_rigid': True,
        'path_conditions': [{'node': index, 'condition': str(condition)}
                            for index, condition in conditions],
        'link_translations': [{'a': str(a), 'd': str(d)} for a, d in translations],
        'dominant_link': candidates.index(radius_lower),
        'radius_lower_bound': str(radius_lower),
        'radius_lower_bound_m': float(radius_lower),
        'inverse_translation_upper_bounds': {str(axis): str(bounds[axis]) for axis in axes},
        'radius_squared_upper_bound': str(radius_squared_upper),
        'strict_squared_margin': str(radius_lower**2-radius_squared_upper),
        'proof': 'Reverse triangle inequality: the DH chain radius is at least the '
                 'positive lower bound, whereas every point on this path is strictly inside it.'}
