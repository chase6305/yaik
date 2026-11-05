"""Recover legacy incomplete shoulders only after checking the path geometry."""
from contextlib import redirect_stdout
import copy
from io import StringIO
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
from codegen.skeleton.complete_partial_branches import (
    branch_fixed_values, complete_partial_shoulder_branches, joints_are_coaxial)
from codegen.skeleton.completeness import audit_completeness
from codegen.skeleton.family_critical import spherical_shoulder_supported
from codegen.skeleton.tree_serialize import read_skeleton_tree, save_ik_tree

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'solver/fixtures/franka_partial_tree.yaml'


def legacy_tree():
    return read_skeleton_tree(str(FIXTURE))


def independent_fk(robot, q):
    offsets = np.array(robot.get_unknown_offset())
    values = dict(robot.parameters_value)
    values.update({joint.symbol: value for joint, value in zip(robot.unknowns, q+offsets)})
    pose = np.array(robot.get_pre_transform_sp().subs(values), dtype=float)
    for dh in robot.dh_params:
        alpha, a, d, theta = [float(sp.sympify(value).subs(values)) for value in dh.flatten()]
        ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
        pose = pose@np.array([[ct, -st, 0., a], [ca*st, ca*ct, -sa, -d*sa],
                              [sa*st, sa*ct, ca, d*ca], [0., 0., 0., 1.]])
    return pose@np.array(robot.get_post_transform_sp().subs(values), dtype=float)


def inputs():
    rng = np.random.default_rng(20261211)
    for middle in (0., -math.pi, -2*math.pi):
        for delta in (0., 1e-8, -1e-8):
            q = rng.uniform(-math.pi, math.pi, 7)
            q[1], q[3] = middle, delta
            yield q


class TestPartialBranches(TestCase):
    def test_exact_axis_check_rejects_small_offsets_and_unfixed_middle_angles(self):
        tree = legacy_tree(); values = branch_fixed_values(tree.node_list[12])
        self.assertEqual(values, {tree.robot.unknowns[3].symbol: sp.pi})
        self.assertTrue(joints_are_coaxial(tree.robot, 2, 4, values))
        self.assertFalse(joints_are_coaxial(tree.robot, 2, 4, {}))
        for epsilon in (sp.Rational(1, 10**30), sp.Rational(1, 10**300)):
            robot = copy.deepcopy(tree.robot)
            robot.dh_params[4].a += epsilon
            self.assertFalse(joints_are_coaxial(robot, 2, 4, values))
        robot = copy.deepcopy(tree.robot)
        robot.dh_params[1].d = sp.Rational(1, 10**30)
        self.assertFalse(spherical_shoulder_supported(robot))
        self.assertTrue(spherical_shoulder_supported(tree.robot))

    def test_completion_preserves_existing_paths_is_atomic_at_budget_and_is_idempotent(self):
        tree = legacy_tree(); nodes = list(tree.node_list)
        solutions = {node.flatten_idx: node.solution.to_dict() for node in nodes if node.is_solution_node}
        self.assertEqual(complete_partial_shoulder_branches(tree, tree.num_nodes+5), 0)
        self.assertEqual(tree.node_list, nodes)
        self.assertEqual(complete_partial_shoulder_branches(tree), 2)
        self.assertEqual(tree.node_list[:len(nodes)], nodes)
        for index, original in solutions.items():
            self.assertEqual(tree.node_list[index].solution.to_dict(), original)
        count = tree.num_nodes
        self.assertEqual(complete_partial_shoulder_branches(tree), 0)
        self.assertEqual(tree.num_nodes, count)
        report = audit_completeness(tree)
        self.assertFalse(report['incomplete_leaves'])
        self.assertFalse(report['certified_complete'])
        self.assertTrue(any(family['strategy']=='coaxial_gauge_representative' for family in report['family_nodes']))

    def test_python_independent_fk_roundtrip_and_serialization(self):
        tree = legacy_tree()
        complete_partial_shoulder_branches(tree)
        with tempfile.TemporaryDirectory(prefix='yaik-partial-') as temporary:
            path = Path(temporary)/'tree.yaml'; save_ik_tree(tree, str(path))
            tree = read_skeleton_tree(str(path))
        scope = {'__name__': 'partial_branch_regression'}
        with redirect_stdout(StringIO()):
            exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
        for q in inputs():
            target = independent_fk(tree.robot, q)
            np.testing.assert_allclose(scope['franka_panda_fk'](q), target, atol=2e-15, rtol=0)
            output = scope['franka_panda_ik_solve'](target, float(q[3]))
            self.assertTrue(output, q)
            errors = []
            for candidate in output:
                self.assertTrue(np.isfinite(candidate).all())
                self.assertEqual(float(candidate[3, 0]), q[3])
                errors.append(float(np.max(np.abs(independent_fk(tree.robot, candidate.ravel())-target))))
            self.assertLess(min(errors), 1e-7, q)
            self.assertLess(max(errors), 1e-4, q)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(), 'Requires g++ and Eigen3')
    def test_cpp_independent_targets_and_fixed_joint_preservation(self):
        tree = legacy_tree(); complete_partial_shoulder_branches(tree)
        with redirect_stdout(StringIO()):
            code = '\n'.join(CodeGenerationCpp(tree, use_safe_operator=True).generate_code()).split('// Code below for debug')[0]
        main = '''#include <cassert>
#include <iostream>
int main() {
    double fixed;
    while(std::cin >> fixed) {
        Eigen::Matrix4d target;
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)std::cin >> target(i,j);
        const auto output=franka_panda_ik::computeIK(target,fixed);
        bool witness=false;
        for(const auto& q:output) {
            assert(q[3]==fixed);
            if((franka_panda_ik::computeFK(q)-target).cwiseAbs().maxCoeff()<1e-7)witness=true;
        }
        assert(witness);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-partial-cpp-') as temporary:
            temporary = Path(temporary)
            (temporary/'test.cpp').write_text(code+main)
            result = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                     str(temporary/'test.cpp'), '-o', str(temporary/'test')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [format(float(q[3]), '.17g')+' '+' '.join(format(float(v), '.17g')
                    for v in independent_fk(tree.robot, q).ravel()) for q in inputs()]
            subprocess.run([str(temporary/'test')], input='\n'.join(rows)+'\n', text=True, check=True)
