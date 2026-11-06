"""Aligned-wrist families use reach critical points, including tangencies."""
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.completeness import audit_completeness
from codegen.skeleton.parallel_wrist import (
    complete_parallel_wrist_branches, parallel_wrist_solutions, parallel_wrist_supported)
from codegen.skeleton.tree_serialize import read_skeleton_tree, save_ik_tree

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'solver/fixtures/ur_10_pre_wrist_tree.yaml'
CASES = json.loads((ROOT/'solver/fixtures/empty_raw_recovery_cases.json').read_text())['cases']
NAMES = ('ur_10', 'ur_5', 'ur10_urdf')


def tree_for(name):
    if name == 'ur_10':
        return read_skeleton_tree(str(FIXTURE))
    return read_skeleton_tree(str(next((ROOT/'gallery').glob('**/'+name+'_ik.yaml'))))


def independent_fk_function(robot):
    values = dict(robot.parameters_value)
    fields = [sp.sympify(v).subs(values) for dh in robot.dh_params for v in dh.flatten()]
    evaluate = sp.lambdify([joint.symbol for joint in robot.unknowns], fields, 'numpy')
    offsets = np.array(robot.get_unknown_offset(), dtype=float)
    pre = np.array(robot.get_pre_transform_sp().subs(values), dtype=float)
    post = np.array(robot.get_post_transform_sp().subs(values), dtype=float)
    def fk(q):
        pose = pre.copy()
        fields = evaluate(*(np.asarray(q).ravel()+offsets))
        for start in range(0, len(fields), 4):
            alpha, a, d, theta = fields[start:start+4]
            ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
            pose = pose@np.array([[ct, -st, 0., a], [ca*st, ca*ct, -sa, -d*sa],
                                 [sa*st, sa*ct, ca, d*ca], [0., 0., 0., 1.]])
        return pose@post
    return fk


def targets(tree):
    fk = independent_fk_function(tree.robot)
    rng = np.random.default_rng(20270123)
    offsets = np.array(tree.robot.get_unknown_offset())
    for wrist in (0., math.pi):
        for elbow in (0., math.pi, 0.4):
            for _ in range(4):
                raw = rng.uniform(-math.pi, math.pi, 6)
                raw[2], raw[4] = elbow, wrist
                yield fk(raw-offsets)
    for case in CASES:
        if case['robot'] == tree.robot.name and not case['recovery_expected']:
            target = np.array(case['target'])
            np.testing.assert_allclose(fk(case['q']), target, rtol=0, atol=2e-14)
            yield target


class TestParallelWrist(TestCase):
    def test_completion_is_atomic_idempotent_and_preserves_existing_nodes(self):
        tree = tree_for('ur_10'); nodes = list(tree.node_list)
        existing = {n.flatten_idx:n.solution.to_dict() for n in nodes if n.is_solution_node}
        self.assertEqual(complete_parallel_wrist_branches(tree, tree.num_nodes+6), 0)
        self.assertEqual(tree.node_list, nodes)
        self.assertEqual(complete_parallel_wrist_branches(tree, 32), 2)
        self.assertEqual(tree.num_nodes, 32)
        self.assertEqual(tree.node_list[:len(nodes)], nodes)
        for index, value in existing.items():
            self.assertEqual(tree.node_list[index].solution.to_dict(), value)
        self.assertEqual(complete_parallel_wrist_branches(tree, 50), 0)
        self.assertFalse(audit_completeness(tree)['certified_complete'])

    def test_annulus_critical_points_recover_a_tangency_missed_by_the_grid(self):
        tree = tree_for('ur_10'); complete_parallel_wrist_branches(tree, 32)
        case = next(row for row in CASES if row['robot']=='ur_10' and row['index']==24)
        target = np.array(case['target'])
        fk = independent_fk_function(tree.robot)
        outputs = []
        for critical in (True, False):
            if not critical:
                for node in tree.node_list:
                    if node.is_solution_node and node.solution.internal_solution.is_explicit_solution:
                        impl = node.solution.internal_solution
                        if impl.family_critical_equations is not None:
                            impl.set_family_critical_equations([], impl.family_bounds, impl.family_critical_condition)
            scope = {}
            with redirect_stdout(StringIO()):
                exec('\n'.join(CodeGenerationPython(tree, refine_seeds=False).generate_code()), scope)
            outputs.append(scope['ur_10_ik_solve'](target))
        self.assertTrue(any(np.max(np.abs(fk(q)-target))<1e-7 for q in outputs[0]))
        self.assertEqual(outputs[1], [])

    def test_geometry_guard_rejects_unsupported_charts(self):
        robot = tree_for('ur_10').robot
        self.assertTrue(parallel_wrist_supported(robot))
        self.assertEqual(len(parallel_wrist_solutions(robot, sp.Float(0))), 4)
        self.assertEqual(parallel_wrist_solutions(robot, sp.Float('1e-30')), [])
        for field, value in (('alpha', sp.Rational(1, 10**30)), ('a', sp.Rational(1, 10**30))):
            changed = copy.deepcopy(robot)
            setattr(changed.dh_params[4], field, getattr(changed.dh_params[4], field)+value)
            self.assertFalse(parallel_wrist_supported(changed))
        changed = copy.deepcopy(robot)
        changed.dh_params[3].a = changed.dh_params[2].a
        self.assertFalse(parallel_wrist_supported(changed))
        changed = copy.deepcopy(robot)
        changed.unknown_as_parameter_more_dof = {changed.unknowns[5].symbol}
        self.assertFalse(parallel_wrist_supported(changed))

    def test_python_exact_wrist_and_elbow_targets_without_numerical_recovery(self):
        for name in NAMES:
            tree = tree_for(name); complete_parallel_wrist_branches(tree, 32)
            with tempfile.TemporaryDirectory(prefix='yaik-wrist-tree-') as temporary:
                path = Path(temporary)/'tree.yaml'; save_ik_tree(tree, str(path))
                tree = read_skeleton_tree(str(path))
            scope = {'__name__':'parallel_wrist_regression'}
            with redirect_stdout(StringIO()):
                exec('\n'.join(CodeGenerationPython(tree, refine_seeds=False).generate_code()), scope)
            fk = independent_fk_function(tree.robot)
            for index, target in enumerate(targets(tree)):
                with self.subTest(robot=name, index=index):
                    output = scope[name+'_ik_solve'](target)
                    self.assertTrue(output)
                    errors = [np.max(np.abs(fk(q)-target)) for q in output]
                    self.assertLess(min(errors), 1e-7)
                    self.assertLess(max(errors), 1e-4)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(), 'Requires g++ and Eigen3')
    def test_cpp_exact_wrist_and_elbow_targets_without_numerical_recovery(self):
        sections = []; rows = []; models = []
        for name in NAMES:
            tree = tree_for(name); complete_parallel_wrist_branches(tree, 32)
            with redirect_stdout(StringIO()):
                sections.append('\n'.join(CodeGenerationCpp(tree, use_safe_operator=True, refine_seeds=False).generate_code()).split('// Code below for debug')[0])
            models.append(tree)
        sections.append("""#include <iostream>
#include <iomanip>
int main() { int model; while(std::cin>>model) {
 Eigen::Matrix4d target; for(int i=0;i<4;++i)for(int j=0;j<4;++j)std::cin>>target(i,j);
 std::vector<std::array<double,6>> output;
 switch(model) {""")
        for index, tree in enumerate(models):
            sections.append('case '+str(index)+': output='+tree.robot.name+'_ik::computeIK(target); break;')
            rows.extend((index, target) for target in targets(tree))
        sections.append("""}
 std::cout<<output.size(); for(const auto& q:output)for(double v:q)std::cout<<' '<<std::setprecision(17)<<v;
 std::cout<<'\\n';
} }""")
        with tempfile.TemporaryDirectory(prefix='yaik-wrist-cpp-') as temporary:
            source, binary = Path(temporary)/'test.cpp', Path(temporary)/'test'
            source.write_text('\n'.join(sections))
            result = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3', str(source), '-o', str(binary)], capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = '\n'.join(str(index)+' '+' '.join(format(float(v), '.17g') for v in target.ravel()) for index, target in rows)+'\n'
            result = subprocess.run([str(binary)], input=data, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines(); self.assertEqual(len(lines), len(rows))
            fks = [independent_fk_function(tree.robot) for tree in models]
            for line, (index, target) in zip(lines, rows):
                fields = line.split(); count = int(fields[0]); self.assertGreater(count, 0)
                candidates = np.array(list(map(float, fields[1:]))).reshape(count, 6)
                errors = [np.max(np.abs(fks[index](q)-target)) for q in candidates]
                self.assertLess(min(errors), 1e-7)
                self.assertTrue(all(np.sum((fks[index](q)-target)**2) < 1e-6 for q in candidates))
