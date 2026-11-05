"""Complete supported shoulder leaves without replacing implemented branches."""
import copy
import math

import sympy as sp

from codegen.skeleton.family_critical import spherical_shoulder_solutions, spherical_shoulder_supported
from codegen.skeleton.tree_node import SolvedVariableDispatcherNode
from fk.kinematics_dh import modified_dh_transform
from fk.robots import DHEntry
from solver.solved_variable import SolutionMethod, VariableSolution


def _exact(expression):
    expression = sp.sympify(expression)
    return expression.xreplace({value: sp.Rational(value) for value in expression.atoms(sp.Float)})


def branch_fixed_values(node):
    """Use selected constant assignments, without splitting zero products."""
    values = {}
    child, parent = node, node.parent
    while parent is not None:
        if isinstance(parent, SolvedVariableDispatcherNode):
            for index in range(parent.num_degenerate_branches):
                if parent.degenerate_branch_child(index) is child:
                    for symbol, value in parent.get_branch_variable_value(index).items():
                        value = _exact(value)
                        if not value.free_symbols and value.is_real is True and value.is_finite is True:
                            if symbol in values and sp.simplify(values[symbol]-value).is_zero is not True:
                                return None
                            values[symbol] = value
        child, parent = parent, parent.parent
    return values


def joints_are_coaxial(robot, first, second, fixed_values):
    """Prove coincident revolute axes for the exact selected path geometry.

    Floats become their exact rational values before geometric operations.
    No tolerance or sampled configurations establish axis coincidence here.
    Runtime branch guards and final floating-point FK remain uncertified.
    """
    if not 0 <= first < second < len(robot.unknowns):
        return False
    if not robot.unknowns[first].is_revolute or not robot.unknowns[second].is_revolute:
        return False
    values = {symbol: _exact(value) for symbol, value in robot.parameters_value.items()}
    values.update(fixed_values)
    middle = sp.eye(4)
    middle[2, 3] = _exact(robot.dh_params[first].d).subs(values)
    for index in range(first+1, second+1):
        dh = DHEntry(*(_exact(value) for value in robot.dh_params[index].flatten()))
        if index == second:
            dh.theta = sp.S.Zero
        middle = middle*modified_dh_transform(dh).subs(values)
    checks = (middle[0, 2], middle[1, 2], middle[2, 2]**2-1, middle[0, 3], middle[1, 3])
    return all(sp.trigsimp(value).is_zero is True for value in checks)


def complete_partial_shoulder_branches(tree, max_num_nodes=50):
    """Append a correlated shoulder tail and, when proved free, one gauge.

    A supported leaf must leave precisely the first three shoulder joints,
    optionally plus one joint coaxial with an unassigned shoulder joint.
    A gauge chooses one in-bounds value; this does not certify all bounded
    representatives of a continuous family. Every returned candidate still
    passes the generated API's normal FK checks.
    """
    robot = tree.robot
    if not spherical_shoulder_supported(robot):
        return 0
    shoulder = {unknown.symbol for unknown in robot.unknowns[:3]}
    required = {unknown.symbol for unknown in tree.solved_unknowns}
    if not shoulder <= required:
        return 0
    count = 0
    tail = None
    for leaf in list(tree.node_list):
        if not leaf.is_solution_node or leaf.dispatcher_child is not None:
            continue
        known = {solution.solved_variable for solution in tree.variable_solutions_from_root(leaf)}
        missing = required-known
        if not shoulder <= missing:
            continue
        values = branch_fixed_values(leaf)
        if values is None or shoulder & values.keys():
            continue
        other = missing-shoulder
        gauges = []
        if other:
            if len(other) != 1:
                continue
            symbol = next(iter(other))
            index = next(i for i, unknown in enumerate(robot.unknowns) if unknown.symbol == symbol)
            unknown = robot.unknowns[index]
            lower, upper = float(unknown.lower_bound), float(unknown.upper_bound)
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                continue
            value = values.get(symbol)
            method = SolutionMethod.OneVariableAlgebra.name
            if value is None:
                if not any(joints_are_coaxial(robot, i, index, values) for i in range(3)):
                    continue
                value = sp.Float(min(upper, max(lower, 0.)))
                method = SolutionMethod.CoaxialGauge.name
            elif not lower <= float(value) <= upper:
                continue
            gauges.append(VariableSolution.make_explicit_solution(
                symbol, [value], method))
        if tail is None:
            tail = spherical_shoulder_solutions(robot)
        solutions = gauges+tail
        if tree.num_nodes+2*len(solutions) > max_num_nodes:
            continue
        # Each appended node belongs to this leaf; existing alternatives stay attached.
        current = leaf
        for solution in copy.deepcopy(solutions):
            dispatcher, node = tree.append_non_degenerate_solution(solution, current)
            tree.add_node(dispatcher)
            tree.add_node(node)
            current = node
        count += 1
    return count
