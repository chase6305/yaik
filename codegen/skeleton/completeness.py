"""Structural coverage diagnostics, without equating coverage to a proof."""

from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, SolvedVariableDispatcherNode
from solver.solved_variable import SolutionMethod


def audit_completeness(tree):
    required = {unknown.symbol for unknown in tree.solved_unknowns}
    unresolved, incomplete, families, numerical = [], [], [], []
    for node in tree.node_list:
        if isinstance(node, EquationAllZeroDispatcherNode) and node.degenerate_child is None:
            unresolved.append({'node': node.flatten_idx, 'condition': str(node.degenerate_checker),
                               'variable': str(node.non_degenerate_child.solved_variable)})
        elif isinstance(node, SolvedVariableDispatcherNode):
            unresolved.extend({'node': node.flatten_idx, 'condition': str(condition),
                               'variable': str(node.non_degenerate_child.solved_variable)}
                              for condition, child in zip(node.branch_conditions, node.children[1:])
                              if child is None)
        if not node.is_solution_node:
            continue
        solution = node.solution
        if solution.solution_method in (SolutionMethod.FamilyRepresentative.name, SolutionMethod.SphericalShoulder.name) or solution.family_critical_equations is not None:
            critical = solution.family_critical_equations
            families.append({'node': node.flatten_idx, 'variable': str(solution.solved_variable),
                             'strategy': 'analytic_limit_cells_with_seed_backup' if critical is not None else 'uniform_samples',
                             'seed_count': len(solution.explicit_solutions),
                             'critical_equations': len(critical) if critical is not None else 0,
                             'candidate_bound': solution.num_solutions(),
                             'critical_roots_certified': False})
        if not solution.is_explicit_solution:
            numerical.append({'node': node.flatten_idx, 'method': solution.impl_class_key})
        if node.dispatcher_child is None:
            solved = {solution.solved_variable for solution in tree.variable_solutions_from_root(node)}
            if required-solved:
                incomplete.append({'node': node.flatten_idx, 'missing': sorted(map(str, required-solved))})
    reasons = ['No end-to-end symbolic/interval completeness certificate is implemented.']
    if unresolved:
        reasons.append(f'{len(unresolved)} degenerate branch conditions have no implementation or verified infeasibility certificate.')
    if incomplete:
        reasons.append(f'{len(incomplete)} leaves do not assign every required joint.')
    if families:
        reasons.append('Continuous families use finite representatives and floating-point critical roots.')
    if numerical:
        reasons.append('Numerical root/linear kernels do not certify that every real solution is isolated.')
    return {'schema_version': 1, 'robot': tree.robot.name, 'tree_nodes': tree.num_nodes,
            'fixed_joints': list(map(str, tree.robot.unknown_as_parameter_more_dof)),
            'status': 'not_certified', 'certified_complete': False,
            'scope': 'valid SE(3) targets with the specified fixed redundant joints',
            'unresolved_branches': unresolved, 'incomplete_leaves': incomplete,
            'family_nodes': families, 'numerical_nodes': numerical, 'reasons': reasons,
            'required_proofs': [
                'All regular and singular kinematic cases cover the bounded joint domain.',
                'Every case is solved completely or proved infeasible under its path conditions.',
                'Continuous solution families are represented and intersected with all joint limits.',
                'All roots and floating-point decisions are verified with sound numerical bounds.',
                'Timeouts, node limits, and unresolved boxes are returned as unknown rather than infeasible.']}
