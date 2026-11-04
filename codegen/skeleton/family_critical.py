"""Derive joint-limit cuts for a spherical-shoulder angular family.

The supported rotation chart is Rz(t0) Rx(-pi/2) Rz(t1) Rx(-pi/2)
Rz(t2). Multiplying by Rx(pi) gives a ZYZ chart. Its joint-limit and
chart-singularity boundaries are linear trigonometric equations in the last
joint. This derives coefficients from the model, without Marvin dimensions.
"""

import math

import sympy as sp

from fk.fk_equations import ik_target_rotation, inv_Px, inv_Py, inv_Pz
from fk.kinematics_dh import modified_dh_transform
from solver.solved_variable import SolutionMethod
from solver.solved_variable import VariableSolution, SolutionDegenerateRecord


def add_spherical_family_cuts(tree, dispatcher, family, suffix, prefix, supplement_condition=sp.S.true):
    robot = tree.robot
    if len(robot.unknowns) < 4 or not all(unknown.is_revolute for unknown in robot.unknowns):
        return False
    if family.solved_variable != robot.unknowns[-1].symbol:
        return False
    if any(sp.simplify(dh.alpha-alpha) != 0 for dh, alpha in
           zip(robot.dh_params[:3], (0, -sp.pi/2, -sp.pi/2))):
        return False
    if any(dh.theta != unknown.symbol for dh, unknown in zip(robot.dh_params, robot.unknowns)):
        return False
    lower, upper = next((float(unknown.lower_bound), float(unknown.upper_bound))
                        for unknown in robot.unknowns if unknown.symbol == family.solved_variable)
    if upper-lower > 2*math.pi:
        return False

    shoulder = {unknown.symbol for unknown in robot.unknowns[:3]}
    known = {solution.solved_variable for solution in prefix}
    remaining = {unknown.symbol for unknown in tree.solved_unknowns} - known
    if not shoulder <= remaining:
        return False
    # A symbol itself being zero can be substituted. A zero product cannot
    # justify setting each factor to zero.
    zeros = {}
    for equation in dispatcher.degenerate_record.equations:
        _, body = sp.sympify(equation).as_coeff_Mul()
        if body.is_Symbol:
            zeros[body] = sp.S.Zero
    estimates = {}
    condition = supplement_condition
    for solution in suffix:
        if solution.solved_variable in shoulder or not solution.is_explicit_solution:
            continue
        if len(solution.explicit_solutions) != 1:
            return False
        expression = solution.explicit_solutions[0].subs(zeros).subs(estimates)
        if expression.free_symbols & remaining:
            return False
        estimates[solution.solved_variable] = expression
        condition &= solution.argument_validity_checkers[0].subs(zeros).subs(estimates)
    if remaining - shoulder - {family.solved_variable} - set(estimates):
        return False

    def euler_matrix(angle):
        rotation = sp.eye(3)
        for dh in robot.dh_params[3:]:
            piece = modified_dh_transform(dh)[:3, :3]
            piece = piece.subs(family.solved_variable, angle).subs(estimates)
            rotation = rotation*piece
        return ik_target_rotation()*rotation.T*sp.diag(1, -1, -1)

    zero, plus, minus = euler_matrix(0), euler_matrix(sp.pi/2), euler_matrix(-sp.pi/2)
    constant = (plus+minus)/2
    cosine, sine = zero-constant, (plus-minus)/2
    matrices = (cosine, sine, constant)
    rows = []
    for joint in (0, 2):
        unknown = robot.unknowns[joint]
        for bound in (unknown.lower_bound, unknown.upper_bound):
            c, s = sp.cos(sp.Float(bound)), sp.sin(sp.Float(bound))
            if joint == 0:
                row = tuple(matrix[1, 2]*c-matrix[0, 2]*s for matrix in matrices)
            else:
                row = tuple(-matrix[2, 1]*c+matrix[2, 0]*s for matrix in matrices)
            rows.append(row)
    middle = robot.unknowns[1]
    for bound in (middle.lower_bound, middle.upper_bound, 0, sp.pi):
        rows.append((cosine[2, 2], sine[2, 2], constant[2, 2]-sp.cos(bound)))
    rows = [tuple(sp.expand(value) for value in row) for row in rows]
    if any(value.free_symbols & remaining for row in rows for value in row):
        return False
    family.internal_solution.set_family_critical_equations(rows, (lower, upper), condition)
    return True


def enrich_spherical_families(tree):
    """Upgrade recorded family nodes while retaining their search provenance."""
    count = 0
    for node in tree.node_list:
        if not node.is_solution_node:
            continue
        solution = node.solution
        if solution.family_critical_equations is not None:
            continue
        if solution.solution_method not in (SolutionMethod.FamilyRepresentative.name, SolutionMethod.SinAndCos.name):
            continue
        dispatcher = node.parent
        if not hasattr(dispatcher, 'degenerate_record'):
            continue
        suffix = []
        child = node.dispatcher_child
        while child is not None:
            if child.is_solution_node:
                suffix.append(child.solution)
                child = child.dispatcher_child
            else:
                child = child.non_degenerate_child
        prefix = tree.variable_solutions_from_root(node, include_current_node=False)
        condition = sp.S.true
        if solution.solution_method == SolutionMethod.SinAndCos.name:
            record = dispatcher.degenerate_record
            if len(record.equations) != 3:
                continue
            a, b, c = record.equations
            # Supplements are needed at a repeated wrist root. The original
            # two analytic roots and their argument guards remain intact.
            condition = sp.Abs(a*a+b*b-c*c) <= sp.Float(128*2.220446049250313e-16)*(inv_Px**2+inv_Py**2+inv_Pz**2)
        count += add_spherical_family_cuts(tree, dispatcher, node.solution, suffix, prefix, condition)
    return count


def enrich_spherical_shoulders(tree):
    """Use all rotation columns and intersect aligned Euler families with limits."""
    robot = tree.robot
    if (len(robot.unknowns) < 3 or not all(unknown.is_revolute for unknown in robot.unknowns[:3]) or
            any(dh.theta != unknown.symbol for dh, unknown in zip(robot.dh_params[:3], robot.unknowns[:3])) or
            any(sp.simplify(dh.alpha-alpha) != 0
            for dh, alpha in zip(robot.dh_params[:3], (0, -sp.pi/2, -sp.pi/2)))):
        return 0
    # Matching rotation twists alone does not make the axes concurrent. An
    # offset between these axes lets translation constrain the aligned angle;
    # retain the original position-aware decomposition in that geometry.
    if any(sp.simplify(value) != 0 for value in
           (robot.dh_params[1].a, robot.dh_params[1].d, robot.dh_params[2].a)):
        return 0
    if any(not (-2*math.pi <= float(unknown.lower_bound) <= float(unknown.upper_bound) <= 2*math.pi)
           for unknown in (robot.unknowns[0], robot.unknowns[2])):
        return 0
    count = 0
    for node in list(tree.node_list):
        if (not node.is_solution_node or node.solution.impl_class_key != 'LinearSinCosType_2' or
                node.solved_variable != robot.unknowns[2].symbol):
            continue
        solved = {solution.solved_variable for solution in tree.variable_solutions_from_root(node, False)}
        remaining = {unknown.symbol for unknown in tree.solved_unknowns}-solved
        if remaining != {unknown.symbol for unknown in robot.unknowns[:3]}:
            continue
        distal = sp.eye(3)
        for dh in robot.dh_params[3:]:
            distal = distal*modified_dh_transform(dh)[:3, :3]
        x = ik_target_rotation()*distal.T*sp.diag(1, -1, -1)
        sine_squared = x[2, 0]**2+x[2, 1]**2
        expressions = [-sp.atan2(x[2, 1], -x[2, 0]), -sp.atan2(-x[2, 1], x[2, 0])]
        checkers = [sine_squared > sp.Float(1e-24)]*2
        xlo, xhi = map(float, (robot.unknowns[0].lower_bound, robot.unknowns[0].upper_bound))
        ylo, yhi = map(float, (robot.unknowns[2].lower_bound, robot.unknowns[2].upper_bound))
        for inverted in (True, False):
            delta = sp.atan2(-x[0, 1], x[1, 1]) if inverted else sp.atan2(x[1, 0], x[0, 0])
            alignment = (sine_squared <= sp.Float(1e-14)) & ((x[2, 2] < 0) if inverted else (x[2, 2] >= 0))
            previous = sp.S.false
            for turns in range(-3, 4):
                total = delta+turns*2*sp.pi
                low = sp.Max(sp.Float(ylo), total-sp.Float(xhi) if inverted else sp.Float(xlo)-total)
                high = sp.Min(sp.Float(yhi), total-sp.Float(xlo) if inverted else sp.Float(xhi)-total)
                valid = low <= high
                expressions.append((low+high)/2)
                checkers.append(alignment & valid & ~previous)
                previous |= valid
        solution = VariableSolution.make_explicit_solution(
            node.solved_variable, expressions, SolutionMethod.SphericalShoulder.name,
            argument_valid_checkers=checkers,
            degenerate_record=SolutionDegenerateRecord.record_always_non_degenerate())
        replacement = tree.new_solution_node(solution)
        replacement.set_flatten_idx(node.flatten_idx)
        replacement.set_parent(node.parent)
        node.parent.set_child_by_index(replacement, node.parent.children.index(node))
        if node.dispatcher_child is not None:
            replacement.set_child(node.dispatcher_child)
        tree.node_list[node.flatten_idx] = replacement
        count += 1
    return count


def correlate_spherical_tail(tree):
    """Recover all shoulder angles from one full-rotation decomposition."""
    enrich_spherical_shoulders(tree)
    robot = tree.robot
    node = tree.root
    while node is not None:
        if node.is_dispatcher_node:
            node = node.non_degenerate_child
            continue
        if node.solution.solution_method == SolutionMethod.SphericalShoulder.name:
            break
        node = node.dispatcher_child
    if node is None or node.solved_variable != robot.unknowns[2].symbol:
        return False
    prefix = tree.variable_solutions_from_root(node, False)
    remaining = {unknown.symbol for unknown in tree.solved_unknowns}-{solution.solved_variable for solution in prefix}
    if remaining != {unknown.symbol for unknown in robot.unknowns[:3]}:
        return False
    distal = sp.eye(3)
    for dh in robot.dh_params[3:]:
        distal = distal*modified_dh_transform(dh)[:3, :3]
    shoulder = ik_target_rotation()*distal.T
    x = shoulder*sp.diag(1, -1, -1)
    t0, t1, t2 = [unknown.symbol for unknown in robot.unknowns[:3]]
    middle = sp.atan2(-x[2, 0]*sp.cos(t2)-x[2, 1]*sp.sin(t2), x[2, 2])
    zero = sp.eye(3)
    for dh in robot.dh_params[:3]:
        zero = zero*modified_dh_transform(dh)[:3, :3].subs(t0, 0)
    delta = shoulder*zero.T
    first = sp.atan2(delta[1, 0]-delta[0, 1], delta[0, 0]+delta[1, 1])
    final = [node.solution]
    for variable, expression in ((t1, middle), (t0, first)):
        final.append(VariableSolution.make_explicit_solution(
            variable, [expression], SolutionMethod.SphericalShoulder.name,
            degenerate_record=SolutionDegenerateRecord.record_always_non_degenerate()))
    from codegen.skeleton.skeleton_tree import SkeletonTree
    replacement = SkeletonTree(prefix+final, robot, tree._solved_equations)
    tree._node_list = replacement.node_list
    tree._root_dispatcher = replacement.root
    return True
