"""Recover supported parallel-elbow arms at an aligned wrist.

In the shoulder plane the two arm links reach (x, y), after removing
the wrist offset. At wrist angle 0/pi the orientation gives the arm's
angle sum as a function of the last joint. Its squared reach is linear
trigonometric in that joint, so include its annulus-boundary roots.
These cuts provide candidate representatives, not a bounded-IK certificate.
"""
import math

import sympy as sp

from codegen.skeleton.tree_node import SolvedVariableDispatcherNode
from fk.fk_equations import Px, Py, Pz, ik_target_rotation
from solver.solved_variable import SolutionMethod, VariableSolution


def parallel_wrist_supported(robot):
    if (len(robot.unknowns) != 6 or len(robot.dh_params) != 6
            or robot.unknown_as_parameter_more_dof
            or not all(joint.is_revolute for joint in robot.unknowns)
            or any(dh.theta != joint.symbol for dh, joint in zip(robot.dh_params, robot.unknowns))):
        return False
    twists = (0, -sp.pi/2, 0, 0, -sp.pi/2, -sp.pi/2)
    if any(sp.simplify(dh.alpha-angle).is_zero is not True for dh, angle in zip(robot.dh_params, twists)):
        return False
    if any(sp.sympify(value).is_zero is not True for value in
           (robot.dh_params[0].a, robot.dh_params[0].d, robot.dh_params[1].a,
            robot.dh_params[4].a, robot.dh_params[5].a)):
        return False
    first, second = (sp.sympify(robot.dh_params[index].a).subs(robot.parameters_value) for index in (2, 3))
    # Avoid the folded zero-radius chart, which needs an additional free angle.
    for value in (first, second, first-second, first+second):
        if value.free_symbols or value.is_real is not True or value.is_finite is not True or value.is_zero is not False:
            return False
    lo, hi = map(float, (robot.unknowns[-1].lower_bound, robot.unknowns[-1].upper_bound))
    return math.isfinite(lo) and math.isfinite(hi) and 0 <= hi-lo <= 2*math.pi


def parallel_wrist_solutions(robot, wrist_value, representative_count=33):
    if not parallel_wrist_supported(robot) or representative_count < 2:
        return []
    wrist_value = sp.sympify(wrist_value)
    if wrist_value.is_zero is True:
        wrist_value = sp.S.Zero
    if wrist_value not in (sp.S.Zero, sp.pi):
        return []
    q0, q1, q2, q3, _, psi = (joint.symbol for joint in robot.unknowns)
    rotation = ik_target_rotation()
    c0, s0 = sp.cos(q0), sp.sin(q0)
    # Rx(pi/2) Rz(-q0) R, in the arm's planar coordinate convention.
    u00 = c0*rotation[0, 0]+s0*rotation[1, 0]
    u10 = -rotation[2, 0]
    norm2 = u00*u00+u10*u10
    chart = norm2 > sp.Float(1e-12)
    phase = sp.atan2(u10, u00)
    sign = sp.S.One if wrist_value == 0 else -sp.S.One
    shift = sp.S.Zero if wrist_value == 0 else -sp.pi
    total = phase+sign*psi+shift
    tool = robot.dh_params[5].d
    wrist = robot.dh_params[4].d
    first, second = robot.dh_params[2].a, robot.dh_params[3].a
    base_x = c0*(Px-tool*rotation[0, 2])+s0*(Py-tool*rotation[1, 2])
    base_y = -Pz+tool*rotation[2, 2]
    x, y = base_x+wrist*sp.sin(total), base_y-wrist*sp.cos(total)
    cosine = (x*x+y*y-first*first-second*second)/(2*first*second)
    elbow = sp.acos(cosine)
    shoulder = sp.atan2(y, x)-sp.atan2(second*sp.sin(q2), first+second*sp.cos(q2))

    low, high = map(float, (robot.unknowns[-1].lower_bound, robot.unknowns[-1].upper_bound))
    samples = [sp.Float(low+(high-low)*index/(representative_count-1)) for index in range(representative_count)]
    family = VariableSolution.make_explicit_solution(psi, list(dict.fromkeys(samples)),
        SolutionMethod.FamilyRepresentative.name, argument_valid_checkers=[chart]*len(set(samples)))
    cosine_phase = sign*u00/sp.sqrt(norm2)
    sine_phase = sign*u10/sp.sqrt(norm2)
    a = 2*wrist*(base_x*sine_phase-base_y*cosine_phase)
    b = 2*wrist*sign*(base_x*cosine_phase+base_y*sine_phase)
    constant = base_x*base_x+base_y*base_y+wrist*wrist
    rows = [(a, b, constant-boundary) for boundary in ((first-second)**2, (first+second)**2)]
    family.internal_solution.set_family_critical_equations(rows, (low, high), chart)
    method = SolutionMethod.OneVariableAlgebra.name
    return [family,
            VariableSolution.make_explicit_solution(q2, [elbow, -elbow], method),
            VariableSolution.make_explicit_solution(q1, [shoulder], method),
            VariableSolution.make_explicit_solution(q3, [total-q1-q2], method)]


def complete_parallel_wrist_branches(tree, max_num_nodes=50):
    """Append complete aligned-wrist tails, retaining each existing child."""
    robot = tree.robot
    if not parallel_wrist_supported(robot):
        return 0
    symbols = [joint.symbol for joint in robot.unknowns]
    before, missing = {symbols[0], symbols[4]}, {symbols[index] for index in (1, 2, 3, 5)}
    count = 0
    for node in list(tree.node_list):
        if not isinstance(node, SolvedVariableDispatcherNode):
            continue
        known = {solution.solved_variable for solution in tree.variable_solutions_from_root(node)}
        if not before <= known or known & missing:
            continue
        for index in range(node.num_degenerate_branches):
            if node.degenerate_branch_child(index) is not None:
                continue
            values = node.get_branch_variable_value(index)
            if set(values) != {symbols[4]}:
                continue
            solutions = parallel_wrist_solutions(robot, values[symbols[4]])
            if not solutions or tree.num_nodes+2*len(solutions)-1 > max_num_nodes:
                continue
            tree.append_branched_solution(solutions, node, index+1, preserve_first_dispatcher=False)
            count += 1
    return count
