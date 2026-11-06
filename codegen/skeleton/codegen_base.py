from fk.robots import RobotDescription
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, SolvedVariableDispatcherNode
from typing import List
import sympy as sp
import copy
from functools import lru_cache
from sympy.core.relational import Relational
from codegen.skeleton.branch_certificates import certify_unreachable_branch
from solver.argument_guards import runtime_argument_guards
from codegen.skeleton.completeness import undispatched_degeneracies


class CodeGenerationBase(object):

    def __init__(self, tree: SkeletonTree):
        self._tree = tree
        if tree.root is None:
            raise ValueError('Cannot generate code from an empty solution tree')
        required = {unknown.symbol for unknown in tree.solved_unknowns}
        self.unhandled_degenerate_branches = []
        self.certified_unreachable_branches = []
        # Every additional real branch can require an independent copy of each base candidate.
        self.branch_capacity_multiplier = 1
        complete_branch = False
        missing = required
        for node in tree.node_list:
            self.unhandled_degenerate_branches.extend(
                (item['node'], item['condition']) for item in undispatched_degeneracies(node))
            if isinstance(node, EquationAllZeroDispatcherNode) and node.degenerate_child is None:
                certificate = certify_unreachable_branch(tree, node)
                if certificate is None:
                    self.unhandled_degenerate_branches.append((node.flatten_idx, str(node.degenerate_checker)))
                else:
                    self.certified_unreachable_branches.append((node.flatten_idx, certificate))
            elif isinstance(node, SolvedVariableDispatcherNode):
                self.branch_capacity_multiplier += max(0, sum(child is not None for child in node.children[1:]) - 1)
                self.unhandled_degenerate_branches.extend(
                    (node.flatten_idx, str(condition)) for condition, child in
                    zip(node.branch_conditions, node.children[1:]) if child is None)
            if not node.is_solution_node:
                continue
            if node.solution.is_polynomial and not node.is_parent_root_dispatcher():
                raise NotImplementedError('Polynomial code generation requires a root solution node')
            if all(child is None for child in node.children):
                solved = {solution.solved_variable for solution in tree.variable_solutions_from_root(node)}
                missing = required - solved
                complete_branch = complete_branch or not missing
        if not complete_branch:
            raise ValueError('No complete solution branch; missing ' + ', '.join(sorted(map(str, missing))))
        if self.unhandled_degenerate_branches:
            print('Generation coverage for {name}: {count} unimplemented degenerate branches.'.format(
                name=tree.robot.name, count=len(self.unhandled_degenerate_branches)))
        self.queue_capacity_bound = queue_capacity_bound(tree.root)
        from codegen.skeleton.fixed_joint_phases import fixed_joint_branch_centers
        self.fixed_joint_phase_centers = fixed_joint_branch_centers(tree)

    @property
    def skeleton_tree(self) -> SkeletonTree:
        return self._tree

    @property
    def robot(self) -> RobotDescription:
        return self._tree.robot

    # The naming utilities
    def ik_function_name(self, raw_ik: bool) -> str:
        """
        :param raw_ik: whether the function check the solved value with fk
        """
        robot_name: str = self.robot.name
        if raw_ik:
            ik_name = robot_name + '_ik_solve_raw'
        else:
            ik_name = robot_name + '_ik_solve'
        return ik_name

    def fk_function_name(self) -> str:
        robot_name: str = self.robot.name
        fk_name = robot_name + '_fk'
        return fk_name

    def twist_jacobian_func_name(self) -> str:
        robot_name: str = self.robot.name
        func_name = robot_name + '_twist_jacobian'
        return func_name

    def angular_velocity_jacobian_func_name(self) -> str:
        robot_name: str = self.robot.name
        func_name = robot_name + '_angular_velocity_jacobian'
        return func_name

    def transform_point_jacobian_func_name(self) -> str:
        robot_name: str = self.robot.name
        func_name = robot_name + '_transform_point_jacobian'
        return func_name

    def default_file_name(self) -> str:
        return self.robot.name + '_ik'

    # The symbolic utility
    def free_symbols(self, include_unknown_as_parameter: bool = True) -> List[sp.Symbol]:
        """
        All symbols that must be provided by the caller
        except the ik target
        """
        return self._tree.get_free_symbols(include_unknown_as_parameter)

    def frame_parameters(self) -> List[sp.Symbol]:
        """Runtime geometry needed by the base/tool coordinate conversions."""
        symbols = set()
        for transform in (self.robot.get_pre_transform_sp(), self.robot.get_post_transform_sp()):
            if transform is not None:
                symbols.update(transform.free_symbols)
        return [symbol for symbol in self.free_symbols(False) if symbol in symbols]

    def frame_transform_call(self, function_name: str, target: str) -> str:
        arguments = [target] + list(map(str, self.frame_parameters()))
        return function_name + '(' + ', '.join(arguments) + ')'

    def joint_offset_expression(self, symbol, value, cpp=False, subtract=False):
        if self.robot.get_unknown_offset() is None:
            return value
        index = next(i for i, unknown in enumerate(self.robot.unknowns) if unknown.symbol == symbol)
        offset = self.get_unknown_offset_name(symbol.name)
        unknown = self.robot.unknowns[index]
        if unknown.is_revolute and self.robot.get_unknown_offset()[index] != 0:
            helper = 'addRevoluteOffset' if cpp else 'add_revolute_offset'
            return helper + '(' + value + ', ' + ('-' if subtract else '') + offset + ')'
        return value + (' - ' if subtract else ' + ') + offset

    # The string utility
    @staticmethod
    def new_line():
        return '\n'

    @staticmethod
    def remove_whitespace(string_to_remove: str):
        new_str = copy.copy(string_to_remove)
        new_str.replace(' ', '')
        return new_str


def candidate_is_complete(tree, node):
    required = {unknown.symbol for unknown in tree.solved_unknowns}
    solved = {solution.solved_variable for solution in
              tree.variable_solutions_from_root(node, include_current_node=False)}
    return required <= solved


def queue_capacity_bound(node, has_prefix=False):
    """Bound all queue allocations, including invalid prefixes that retain slots.

    Coefficient-zero branches are exclusive. Variable-value conditions can
    overlap, in which case every implemented child needs an independent prefix.
    """
    if node is None:
        return 0
    if node.is_solution_node:
        roots = explicit_active_bound(node.solution) if node.solution.is_explicit_solution else node.solution.num_solutions()
        allocated = max(0, roots-1) if has_prefix else roots
        return allocated + roots*queue_capacity_bound(node.dispatcher_child, True)
    costs = [queue_capacity_bound(child, has_prefix) for child in node.children]
    if isinstance(node, SolvedVariableDispatcherNode):
        count = sum(child is not None for child in node.children[1:])
        copies = max(0, count-1) if has_prefix else 0
        return max(costs[0], copies + sum(costs[1:]))
    return max(costs, default=0)


def node_input_capacity_bounds(tree):
    """Bound each node's incoming indices without changing candidate limits.

    Different dispatcher children can each receive every incoming prefix.
    Formula nodes multiply that bound by their maximum active candidates.
    Runtime index storage can grow if a future formula exceeds its estimate.
    """
    capacities = [0]*tree.num_nodes

    def visit(node, incoming):
        if node is None:
            return
        capacities[node.flatten_idx] = incoming
        if node.is_solution_node:
            roots = (explicit_active_bound(node.solution) if node.solution.is_explicit_solution
                     else node.solution.num_solutions())
            visit(node.dispatcher_child, incoming*roots)
        else:
            for child in node.children:
                visit(child, incoming)

    visit(tree.root, 1)
    return capacities


@lru_cache(maxsize=4096)
def _guards_exclusive(first, second):
    expression = sp.And(first, second)
    comparisons, replacements = {}, {}
    for atom in expression.atoms(Relational):
        if atom.rel_op in ('<', '>='):
            key, negate = (atom.lhs, atom.rhs), atom.rel_op == '>='
        elif atom.rel_op in ('>', '<='):
            key, negate = (atom.rhs, atom.lhs), atom.rel_op == '<='
        else:
            continue
        symbol = comparisons.setdefault(key, sp.Dummy('comparison', boolean=True))
        replacements[atom] = ~symbol if negate else symbol
    return sp.satisfiable(expression.xreplace(replacements)) is False


def explicit_active_bound(solution):
    if solution.family_critical_equations is not None:
        return solution.num_solutions()
    groups = {}
    # Rounded inverse-trig domains can overlap an adjacent fallback guard.
    # Allocate against the guards emitted by safe code generation.
    for checker in runtime_argument_guards(solution.explicit_solutions, solution.argument_validity_checkers):
        groups[checker] = groups.get(checker, 0)+1
    buckets = []
    for checker, multiplicity in groups.items():
        for bucket in buckets:
            if all(_guards_exclusive(checker, previous) for previous, _ in bucket):
                bucket.append((checker, multiplicity))
                break
        else:
            buckets.append([(checker, multiplicity)])
    return sum(max(count for _, count in bucket) for bucket in buckets)


def cse_by_condition(expressions, conditions, safe_domains=False):
    """Share temporaries only within adjacent equal guards, preserving candidate order."""
    if safe_domains:
        conditions = runtime_argument_guards(expressions, conditions)
    start = 0
    while start < len(expressions):
        end = start + 1
        while end < len(expressions) and conditions[end] == conditions[start]:
            end += 1
        replacements, reduced = sp.cse(expressions[start:end])
        yield range(start, end), conditions[start], replacements, reduced
        start = end


# The indent string
def indent(n_indent: int = 1):
    indent_str = '    '
    all_indent_str = ''
    for i in range(n_indent):
        all_indent_str += indent_str
    return all_indent_str
