"""Recover coefficient-zero branches without discarding the original solution.

Small remaining systems are solved again from the complete FK equations with
their failed coefficients excluded. Larger singular systems retain a finite
set of bounded representatives and continue through the existing formulas.
These representatives are candidates, not a completeness proof; the public
generated API must still validate them with FK.
"""

import copy
import math

import sympy as sp

from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, NoBranchDispatcherNode
from solver.branch_conditions import formula_degenerates_on_branch
from solver.solve_equations import SolveEquationOption, solve_from_snapshot
from solver.solved_variable import SolutionDegenerateRecord, SolutionMethod, VariableSolution
from solver.solver_snapshot import SolverSnapshot


def _interval(expression):
    """Conservative global bounds; never assume joint or target coordinates."""
    if not expression.free_symbols:
        try:
            value = float(expression)
            return (value, value) if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    if expression.func in (sp.sin, sp.cos):
        return -1.0, 1.0
    if expression.is_Add:
        intervals = [_interval(term) for term in expression.args]
        if all(interval is not None for interval in intervals):
            return sum(interval[0] for interval in intervals), sum(interval[1] for interval in intervals)
    if expression.is_Mul:
        result = (1.0, 1.0)
        for term in expression.args:
            interval = _interval(term)
            if interval is None:
                return None
            products = [a*b for a in result for b in interval]
            result = min(products), max(products)
        return result
    return None


def _provably_nonzero(record, parameter_values):
    for equation in record.equations:
        interval = _interval(sp.expand(equation.subs(parameter_values)))
        # Leave a margin for floating-point interval arithmetic and the runtime
        # dispatcher's 1e-6 zero test. Sampling is never used as a proof.
        if interval is not None and (interval[0] > 1.000001e-6 or interval[1] < -1.000001e-6):
            return True
    return False


def _remove_impossible_dispatcher(tree, node):
    replacement = NoBranchDispatcherNode()
    replacement.set_flatten_idx(node.flatten_idx)
    replacement.set_parent(node.parent)
    replacement.set_non_degenerate_child(node.non_degenerate_child)
    replacement.non_degenerate_child.solution.update_degenerate_record(
        SolutionDegenerateRecord.record_always_non_degenerate())
    tree.node_list[node.flatten_idx] = replacement
    if node.parent is None:
        tree._root_dispatcher = replacement
    else:
        index = node.parent.children.index(node)
        node.parent.set_child_by_index(replacement, index)


def _zero_equations_on_path(node):
    equations = list(node.degenerate_record.equations)
    child, parent = node, node.parent
    while parent is not None:
        if isinstance(parent, EquationAllZeroDispatcherNode) and parent.degenerate_child is child:
            equations.extend(parent.degenerate_record.equations)
        child, parent = parent, parent.parent
    return tuple(dict.fromkeys(equations))


def _nondegenerate_suffix(node):
    solutions = []
    while node is not None:
        if node.is_solution_node:
            solutions.append(copy.deepcopy(node.solution))
            node = node.dispatcher_child
        else:
            node = node.non_degenerate_child
    return solutions


def _representative_branch(node, remaining, prefix, count):
    if count < 2 or any(solution.solution_method == SolutionMethod.FamilyRepresentative.name
                        for solution in prefix):
        return []
    failed = node.non_degenerate_child.solved_variable
    unknown = next((unknown for unknown in remaining if unknown.symbol == failed), None)
    if unknown is None or not unknown.is_revolute:
        return []
    lower, upper = float(unknown.lower_bound), float(unknown.upper_bound)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        return []
    suffix = _nondegenerate_suffix(node.non_degenerate_child)[1:]
    symbols = [failed] + [solution.solved_variable for solution in suffix]
    if len(set(symbols)) != len(symbols) or set(symbols) != {unknown.symbol for unknown in remaining}:
        return []
    values = [sp.Float(lower + (upper-lower)*index/(count-1)) for index in range(count)]
    first = VariableSolution.make_explicit_solution(
        failed, list(dict.fromkeys(values)), SolutionMethod.FamilyRepresentative.name,
        degenerate_record=SolutionDegenerateRecord.record_always_non_degenerate())
    return [first] + suffix


def build_equation_zero_branches(tree, max_num_nodes=50, representative_count=33):
    """Extend complete branches only, keeping one representative joint per path."""
    visited = set()
    cache = {}
    equations = None
    while tree.num_nodes < max_num_nodes:
        node = next((node for node in tree.node_list
                     if isinstance(node, EquationAllZeroDispatcherNode)
                     and node.flatten_idx not in visited and node.degenerate_child is None), None)
        if node is None:
            break
        visited.add(node.flatten_idx)
        if _provably_nonzero(node.degenerate_record, tree.robot.parameters_value):
            _remove_impossible_dispatcher(tree, node)
            continue
        prefix = tree.variable_solutions_from_root(node)
        solved = {solution.solved_variable for solution in prefix}
        remaining = [unknown for unknown in tree.solved_unknowns if unknown.symbol not in solved]
        if not remaining:
            continue

        solutions = []
        if len(remaining) <= 2:
            zero_equations = _zero_equations_on_path(node)
            key = tuple(unknown.symbol for unknown in remaining), zero_equations
            if key not in cache:
                for (symbols, constraints), candidate in list(cache.items()):
                    if (symbols == key[0] and candidate and set(constraints) <= set(zero_equations)
                            and not any(formula_degenerates_on_branch(solution, zero_equations)
                                        for solution in candidate)):
                        cache[key] = candidate
                        break
            if key not in cache:
                if equations is None:
                    equations = tree.solved_equation_record
                snapshot = SolverSnapshot(
                    equations.scalar_equations, equations.sos_hint, remaining, equations.all_symbol_set,
                    parameter_values=equations.parameter_value_dict,
                    parameter_bounds=equations.parameter_bound_dict)
                option = SolveEquationOption(
                    zero_equations=zero_equations,
                    initially_deferred_symbols={node.non_degenerate_child.solved_variable}
                    if len(remaining) > 1 else set(),
                    direct_solutions_only=True)
                option.on_main_branch = False
                option.disable_polynomial_solution = True
                option.stop_on_first_unsolved = True
                print('Recovering coefficient-zero branch at node', node.flatten_idx,
                      'with unknowns', [str(unknown.symbol) for unknown in remaining], flush=True)
                candidate = solve_from_snapshot(snapshot, option)
                cache[key] = (candidate if {solution.solved_variable for solution in candidate}
                              == {unknown.symbol for unknown in remaining} else [])
            solutions = copy.deepcopy(cache[key])
        if not solutions:
            solutions = _representative_branch(node, remaining, prefix, representative_count)
        if not solutions:
            continue
        preserve = solutions[0].solution_method != SolutionMethod.FamilyRepresentative.name
        added_nodes = 2*len(solutions) - (0 if preserve else 1)
        if tree.num_nodes + added_nodes > max_num_nodes:
            print('Coefficient-zero branch exceeds the node budget:', node.flatten_idx)
            continue
        tree.append_branched_solution(solutions, node, 1, preserve_first_dispatcher=preserve)
