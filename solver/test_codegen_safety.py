"""Code-generation regressions for guarded expressions and partial branches."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import TestCase, mock
import copy
import math
import shutil
import subprocess
import tempfile

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, SolvedVariableDispatcherNode
from codegen.skeleton.tree_serialize import TreeSerializer
from fk.robots import DHEntry, RobotDescription
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionDegenerateRecord, SolutionMethod, VariableSolution


def explicit(symbol, expressions, **kwargs):
    return VariableSolution.make_explicit_solution(
        symbol, expressions, SolutionMethod.OneVariableAlgebra.name, **kwargs)


def robot(name, symbols):
    result = RobotDescription(name)
    result.unknowns = [Unknown(symbol, UnknownType.Translational.name) for symbol in symbols]
    result.dh_params = [DHEntry(0, 0, symbol, 0) for symbol in symbols]
    return result


def guarded_tree(non_root, interleaved=False):
    x, y, a = sp.symbols('x y a')
    result = robot('guarded_nonroot' if non_root else 'guarded_root', [x, y] if non_root else [x])
    result.symbolic_parameters = {a}
    result.parameters_bound = {a: (0, 2)}
    first = sp.sin(1 / a) + sp.cos(1 / a)
    second = sp.sin(1 / (a - 1)) + sp.cos(1 / (a - 1))
    expressions = [first, 2 * first, second]
    checkers = [sp.Abs(a) > 1e-6, sp.Abs(a) > 1e-6, sp.Abs(a - 1) > 1e-6]
    if interleaved:
        expressions = [expressions[i] for i in (0, 2, 1)]
        checkers = [checkers[i] for i in (0, 2, 1)]
    guarded = explicit(y if non_root else x, expressions, argument_valid_checkers=checkers)
    return SkeletonTree([explicit(x, [sp.S.Zero]), guarded] if non_root else [guarded], result)


def missing_branch_tree(variable=False, implemented_overlap=False, mixed=False):
    x, y, Pz = sp.symbols('x y Pz')
    name = 'missing_variable' if variable else 'missing_equation'
    if implemented_overlap:
        name += '_overlap'
    result = robot(name, [x, y])
    values = [sp.S.Zero, sp.S.One] if mixed else [sp.S.Zero]
    record = (SolutionDegenerateRecord.record_variable_value(
        {x: [sp.S.Zero, sp.Float(1e-7)] if implemented_overlap else [sp.S.Zero]})
        if variable else SolutionDegenerateRecord.record_all_equations([x]))
    tree = SkeletonTree([explicit(x, values), explicit(y, [Pz - x], degenerate_record=record)], result)
    if implemented_overlap:
        tree.append_branched_solution([explicit(y, [Pz - x])], tree.node_list[2], 2)
    return tree


def generated_python(tree):
    scope = {}
    exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
    return scope[tree.robot.name + '_ik_solve_raw']


def complete_prefix_tree():
    x, y, extra, Pz = sp.symbols('x y extra Pz')
    result = robot('complete_prefix', [x, y])
    return SkeletonTree([explicit(x, [sp.S.Zero]), explicit(y, [Pz]), explicit(extra, [sp.S.Zero],
        degenerate_record=SolutionDegenerateRecord.record_all_equations([x]))], result)


def overlapping_branch_tree(missing_first, root_count=1, overlap=True):
    x, y, extra, Pz = sp.symbols('x y extra Pz')
    name = 'deep_overlap_' + ('first' if missing_first else 'last')
    name += '_' + str(root_count)
    if not overlap:
        name += '_exclusive'
    result = robot(name, [x, y])
    tree = SkeletonTree([explicit(x, [sp.S.Zero] * root_count), explicit(y, [Pz],
        degenerate_record=SolutionDegenerateRecord.record_variable_value(
            {x: [sp.S.Zero, sp.Float(1e-7) if overlap else sp.S.One]}))], result)
    branches = [[explicit(extra, [sp.S.Zero]), explicit(y, [Pz],
        degenerate_record=SolutionDegenerateRecord.record_all_equations([x]))], [explicit(y, [Pz])]]
    if not missing_first:
        branches.reverse()
    for index, solutions in enumerate(branches, 1):
        tree.append_branched_solution(solutions, tree.node_list[2], index)
    return tree


def successful_overlap_tree():
    x, y, z, Pz = sp.symbols('x y z Pz')
    result = robot('successful_overlap', [x, y, z])
    tree = SkeletonTree([explicit(x, [sp.S.Zero]), explicit(y, [sp.S.Zero],
        degenerate_record=SolutionDegenerateRecord.record_variable_value({x: [sp.S.Zero, sp.Float(1e-7)]})),
        explicit(z, [Pz])], result)
    for index, value in enumerate((sp.S.One, -sp.S.One), 1):
        tree.append_branched_solution([explicit(y, [value]), explicit(z, [Pz - value])], tree.node_list[2], index)
    return tree


class TestCodegenSafety(TestCase):
    def test_guards_isolate_cse_and_keep_output_order(self):
        for non_root in (False, True):
            solve = generated_python(guarded_tree(non_root))
            for a in (0.0, 0.25, 1.0):
                with self.subTest(non_root=non_root, parameter=a):
                    expected = []
                    if a != 0:
                        first = math.sin(1 / a) + math.cos(1 / a)
                        expected.extend([first, 2 * first])
                    if a != 1:
                        expected.append(math.sin(1 / (a - 1)) + math.cos(1 / (a - 1)))
                    # Non-root solvers reuse the input slot for their final candidate.
                    if non_root and a != 1 and len(expected) > 1:
                        expected = expected[-1:] + expected[:-1]
                    output = solve(np.eye(4), a)
                    np.testing.assert_allclose([float(q[-1, 0]) for q in output], expected, atol=1e-14)
            with mock.patch('math.sin', wraps=math.sin) as sine:
                solve(np.eye(4), 0.25)
            # The first two candidates share one guarded expression evaluation.
            self.assertEqual(sine.call_count, 2)
            solve = generated_python(guarded_tree(non_root, interleaved=True))
            a = 0.25
            first = math.sin(1 / a) + math.cos(1 / a)
            middle = math.sin(1 / (a - 1)) + math.cos(1 / (a - 1))
            expected = [2 * first, first, middle] if non_root else [first, middle, 2 * first]
            np.testing.assert_allclose([float(q[-1, 0]) for q in solve(np.eye(4), a)], expected)

    def test_missing_branches_discard_partial_candidates(self):
        target = np.eye(4)
        target[2, 3] = 2
        for variable in (False, True):
            with self.subTest(variable=variable):
                tree = missing_branch_tree(variable)
                with redirect_stdout(StringIO()) as diagnostic:
                    generator = CodeGenerationPython(tree)
                self.assertEqual(len(generator.unhandled_degenerate_branches), 1)
                self.assertEqual(len(diagnostic.getvalue().splitlines()), 1)
                scope = {}
                exec('\n'.join(generator.generate_code()), scope)
                self.assertEqual(scope[tree.robot.name + '_ik_solve_raw'](target), [])
                mixed = missing_branch_tree(variable, mixed=True)
                output = generated_python(mixed)(target)
                np.testing.assert_allclose([q.ravel() for q in output], [[1, 1]])
        overlap = missing_branch_tree(variable=True, implemented_overlap=True)
        output = generated_python(overlap)(target)
        np.testing.assert_allclose([q.ravel() for q in output], [[0, 2]])
        # An unresolved auxiliary symbol must not erase already complete joint values.
        output = generated_python(complete_prefix_tree())(target)
        np.testing.assert_allclose([q.ravel() for q in output], [[0, 2]])

    def test_overlapping_real_branches_do_not_invalidate_each_other(self):
        target = np.eye(4)
        target[2, 3] = 2
        for missing_first in (False, True):
            for root_count in (1, 16):
                with self.subTest(missing_first=missing_first, root_count=root_count):
                    tree = overlapping_branch_tree(missing_first, root_count)
                    generator = CodeGenerationCpp(tree)
                    self.assertEqual(generator.branch_capacity_multiplier, 2)
                    output = generated_python(tree)(target)
                    np.testing.assert_allclose([q.ravel() for q in output], [[0, 2]] * root_count)
        solve = generated_python(overlapping_branch_tree(False, 16, overlap=False))
        with mock.patch('copy.copy', wraps=copy.copy) as copies:
            output = solve(target)
        self.assertEqual(copies.call_count, 0)
        np.testing.assert_allclose([q.ravel() for q in output], [[0, 2]] * 16)
        output = generated_python(successful_overlap_tree())(target)
        np.testing.assert_allclose([q.ravel() for q in output], [[0, 1, 1], [0, -1, 3]])

    def test_serialization_preserves_custom_branch_conditions_and_old_defaults(self):
        x = sp.Symbol('x')
        model = robot('serialization', [x])
        nodes = [EquationAllZeroDispatcherNode(SolutionDegenerateRecord.record_all_equations([x]), 1e-4),
                 SolvedVariableDispatcherNode(SolutionDegenerateRecord.record_variable_value({x: [sp.S.Zero]}), 1e-4)]
        for node in nodes:
            with self.subTest(node=type(node).__name__):
                node.set_flatten_idx(0)
                serializer = TreeSerializer(1, model)
                node.accept(serializer)
                saved = serializer.serialize_dict()['tree_nodes'][0]
                loaded = TreeSerializer.load_node_no_connection(saved)
                original = node.degenerate_checker if isinstance(node, EquationAllZeroDispatcherNode) else node.branch_conditions[0]
                restored = loaded.degenerate_checker if isinstance(loaded, EquationAllZeroDispatcherNode) else loaded.branch_conditions[0]
                self.assertEqual(restored, original)
                self.assertTrue(bool(restored.subs(x, 5e-5)))
                legacy = {key: value for key, value in saved.items() if key not in ('degenerate_checker', 'branch_conditions')}
                loaded = TreeSerializer.load_node_no_connection(legacy)
                default = loaded.degenerate_checker if isinstance(loaded, EquationAllZeroDispatcherNode) else loaded.branch_conditions[0]
                self.assertFalse(bool(default.subs(x, 5e-5)))
        for constant in (sp.S.Zero, sp.S.One):
            node = EquationAllZeroDispatcherNode(SolutionDegenerateRecord.record_all_equations([constant]))
            node.set_flatten_idx(0)
            serializer = TreeSerializer(1, model)
            node.accept(serializer)
            saved = serializer.serialize_dict()['tree_nodes'][0]
            loaded = TreeSerializer.load_node_no_connection(saved)
            self.assertEqual(loaded.degenerate_checker, node.degenerate_checker)
            self.assertEqual(loaded.degenerate_checker.free_symbols, set())

    def test_native_guards_and_partial_branch_handling(self):
        eigen = Path('/usr/include/eigen3')
        if not shutil.which('g++') or not (eigen / 'Eigen/Core').exists():
            self.skipTest('g++ and Eigen are needed for the C++ integration test')
        trees = [guarded_tree(False), guarded_tree(True), missing_branch_tree(False),
                 missing_branch_tree(True), missing_branch_tree(True, implemented_overlap=True), complete_prefix_tree()]
        trees.extend(overlapping_branch_tree(missing_first, root_count)
                     for missing_first in (False, True) for root_count in (1, 16))
        trees.extend([overlapping_branch_tree(False, 16, overlap=False), successful_overlap_tree()])
        code = '\n'.join('\n'.join(CodeGenerationCpp(tree).generate_code()).split('// Code below for debug')[0]
                         for tree in trees)
        main = r'''
int main() {
    const Eigen::Matrix4d target = Eigen::Matrix4d::Identity();
    const double expected = std::sin(1.0) + std::cos(1.0);
    for (double a : {0.0, 0.25, 1.0}) {
        guarded_root_ik::RawIKWorksace root;
        guarded_nonroot_ik::RawIKWorksace nonroot;
        std::feclearexcept(FE_ALL_EXCEPT);
        guarded_root_ik::computeRawIK(target, a, root);
        guarded_nonroot_ik::computeRawIK(target, a, nonroot);
        assert(!std::fetestexcept(FE_DIVBYZERO | FE_INVALID));
        const auto count = a == 0 ? 1 : (a == 1 ? 2 : 3);
        assert(root.raw_ik_out.size() == count && nonroot.raw_ik_out.size() == count);
        if (a == 1) {
            assert(std::abs(root.raw_ik_out[0][0] - expected) < 1e-14);
            assert(std::abs(root.raw_ik_out[1][0] - 2 * expected) < 1e-14);
            assert(std::abs(nonroot.raw_ik_out[0][1] - expected) < 1e-14);
            assert(std::abs(nonroot.raw_ik_out[1][1] - 2 * expected) < 1e-14);
        } else if (a == 0.25) {
            const double first = std::sin(1 / a) + std::cos(1 / a);
            const double last = std::sin(1 / (a - 1)) + std::cos(1 / (a - 1));
            assert(std::abs(root.raw_ik_out[0][0] - first) < 1e-14);
            assert(std::abs(root.raw_ik_out[2][0] - last) < 1e-14);
            assert(std::abs(nonroot.raw_ik_out[0][1] - last) < 1e-14);
            assert(std::abs(nonroot.raw_ik_out[1][1] - first) < 1e-14);
        }
    }
    Eigen::Matrix4d translated = target;
    translated(2, 3) = 2;
    missing_equation_ik::RawIKWorksace equation;
    missing_variable_ik::RawIKWorksace variable;
    missing_variable_overlap_ik::RawIKWorksace overlap;
    complete_prefix_ik::RawIKWorksace complete;
    missing_equation_ik::computeRawIK(translated, equation);
    missing_variable_ik::computeRawIK(translated, variable);
    missing_variable_overlap_ik::computeRawIK(translated, overlap);
    complete_prefix_ik::computeRawIK(translated, complete);
    assert(equation.raw_ik_out.empty() && variable.raw_ik_out.empty());
    assert(overlap.raw_ik_out.size() == 1);
    assert(overlap.raw_ik_out[0][0] == 0 && overlap.raw_ik_out[0][1] == 2);
    assert(complete.raw_ik_out.size() == 1);
    assert(complete.raw_ik_out[0][0] == 0 && complete.raw_ik_out[0][1] == 2);
    deep_overlap_last_16_exclusive_ik::RawIKWorksace exclusive;
    deep_overlap_last_16_exclusive_ik::computeRawIK(translated, exclusive);
    assert(exclusive.raw_ik_out.size() == 16 && exclusive.solution_queue.size() == 16);
    successful_overlap_ik::RawIKWorksace successful;
    successful_overlap_ik::computeRawIK(translated, successful);
    assert(successful.raw_ik_out.size() == 2);
    assert(successful.raw_ik_out[0][0] == 0 && successful.raw_ik_out[0][1] == 1 && successful.raw_ik_out[0][2] == 1);
    assert(successful.raw_ik_out[1][0] == 0 && successful.raw_ik_out[1][1] == -1 && successful.raw_ik_out[1][2] == 3);
}
'''
        overlap_checks = ''
        for missing_first in (False, True):
            for root_count in (1, 16):
                name = overlapping_branch_tree(missing_first, root_count).robot.name + '_ik'
                overlap_checks += '''
    {
        NAME::RawIKWorksace workspace;
        static_assert(NAME::max_n_solutions == 32, "capacity must cover both overlapping paths");
        NAME::computeRawIK(translated, workspace);
        assert(workspace.raw_ik_out.size() == COUNT);
        for (const auto& q : workspace.raw_ik_out) assert(q[0] == 0 && q[1] == 2);
    }
'''.replace('NAME', name).replace('COUNT', str(root_count))
        main = main.rsplit('}', 1)[0] + overlap_checks + '}\n'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'safety.cpp'
            executable = Path(directory) / 'safety'
            source.write_text('#include <cfenv>\n' + code + main)
            root = Path(__file__).resolve().parents[1]
            result = subprocess.run(['g++', '-std=c++14', '-O0', '-I' + str(root), '-I' + str(eigen),
                                     str(source), '-o', str(executable)], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
