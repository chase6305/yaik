"""URDF frame, analytical singularity, and bounded integration regressions."""

from pathlib import Path
import hashlib
import json
import math
import sys
from unittest import TestCase

import numpy as np
import sympy as sp
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate import PythonSolver, UrdfArm, cases_for, load_module, negative_checks, pose_error, run_solver
from fk.robots import RobotDescription


class TestMarvinM6(TestCase):
    def test_generated_fk_matches_exact_urdf_base_and_ee(self):
        rng = np.random.default_rng(696)
        for side in ('left', 'right'):
            arm = UrdfArm(side)
            generated, bounded = PythonSolver(side), PythonSolver(side, bounded=True)
            samples = list(rng.uniform(arm.lower, arm.upper, (80, 7)))
            samples += [np.zeros(7)] + [row for row in np.eye(7)*0.5]
            for q in samples:
                for solver in (generated, bounded):
                    self.assertLess(np.max(np.abs(solver.fk(q)-arm.fk(q))), 1e-11)
            # The 29.142 um shoulder Y origin must survive model conversion.
            self.assertAlmostEqual(arm.chain[0][0][1, 3], 0.000029142, places=14)
            # This target is for *_ee, including the 95 mm fixed tool origin.
            self.assertAlmostEqual(np.linalg.norm(arm.chain[-1][0][:3, 3]), 0.095)

    def test_source_snapshots_match_generation_provenance(self):
        provenance = json.loads((HERE / 'provenance.json').read_text())
        self.assertEqual(set(provenance['urdf_sha256']), {'left_arm.urdf', 'right_arm.urdf'})
        for name, digest in provenance['urdf_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE / 'urdf' / name).read_bytes()).hexdigest(), digest)

    def test_analytical_geometry_limits_and_frames_match_model_snapshot(self):
        module = load_module(HERE / 'solver.py', 'marvin_m6_geometry_invariants')
        for side in ('left', 'right'):
            arm, integration = UrdfArm(side), module.Solver(side)
            data = yaml.safe_load((HERE / f'marvin_m6_{side}_arm.yaml').read_text())
            model = RobotDescription.load_from_dict(data)
            values = model.parameters_value
            self.assertEqual(module.A, values[sp.Symbol('a_1')])
            self.assertEqual(module.A, values[sp.Symbol('a_2')])
            self.assertEqual(module.UPPER_ARM, values[sp.Symbol('d_0')])
            self.assertEqual(module.FOREARM, values[sp.Symbol('d_3')])
            np.testing.assert_allclose(module.OFFSETS, model.get_unknown_offset(), atol=1e-14, rtol=0)
            np.testing.assert_array_equal(module.LOWER, arm.lower)
            np.testing.assert_array_equal(module.UPPER, arm.upper)
            np.testing.assert_allclose([u.lower_bound-offset for u, offset in zip(model.unknowns, module.OFFSETS)], arm.lower, atol=1e-14, rtol=0)
            np.testing.assert_allclose([u.upper_bound-offset for u, offset in zip(model.unknowns, module.OFFSETS)], arm.upper, atol=1e-14, rtol=0)
            pre = np.array(model.get_pre_transform_sp().subs(values), dtype=float)
            post = np.array(model.get_post_transform_sp().subs(values), dtype=float)
            np.testing.assert_array_equal(integration.pre_r, pre[:3, :3])
            np.testing.assert_array_equal(integration.pre_p, pre[:3, 3])
            np.testing.assert_array_equal(module.POST_R, post[:3, :3])
            np.testing.assert_array_equal(module.POST_P, post[:3, 3])

    def test_bounded_analytical_random_limits_and_singular_targets(self):
        for side in ('left', 'right'):
            arm = UrdfArm(side)
            result = run_solver(PythonSolver(side, bounded=True), arm, cases_for(arm, 100, 696))
            self.assertTrue(result['passed'], result)

    def test_invalid_targets_and_free_joint_are_rejected(self):
        for side in ('left', 'right'):
            result = negative_checks(PythonSolver(side, bounded=True), UrdfArm(side))
            self.assertTrue(result['passed'], result)

    def test_reachable_pose_can_be_incompatible_with_requested_fixed_joint(self):
        # The inverse wrist translation lies on its Z axis here. A nonzero
        # sin(q5) demands an XY component at this elbow distance, so fixing J5
        # to pi/2 is incompatible even though q5=0 has bounded representatives.
        q = np.array([0.3, -0.5, 0.7, -2.0, 0.0, -0.634705735855024, 0.1])
        for side in ('left', 'right'):
            target = UrdfArm(side).fk(q)
            bounded = PythonSolver(side, bounded=True)
            self.assertTrue(bounded.ik(target, 0.0))
            for solver in (bounded, PythonSolver(side)):
                self.assertEqual(len(solver.ik(target, math.pi/2)), 0)

    def test_near_inverse_wrist_axis_retains_j7_endpoint_solution(self):
        q = np.array([1.4629432909116207, -0.6221208281415567,
                      -0.6753055498052598, -2.1225024040257785,
                      -1e-6, -0.582034496345317, math.pi/2])
        for side in ('left', 'right'):
            arm, solver = UrdfArm(side), PythonSolver(side, bounded=True)
            target = arm.fk(q)
            output = solver.ik(target, q[4])
            self.assertTrue(output)
            for candidate in output:
                position, rotation = pose_error(arm.fk(candidate), target)
                self.assertLessEqual(position, 1e-7)
                self.assertLessEqual(rotation, 1e-6)
                self.assertTrue(np.all(candidate >= arm.lower-1e-9))
                self.assertTrue(np.all(candidate <= arm.upper+1e-9))

    def test_left_target_cannot_be_used_unchanged_for_right_arm(self):
        q = np.array([0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1])
        left, right = UrdfArm('left'), UrdfArm('right')
        position, rotation = pose_error(left.fk(q), right.fk(q))
        self.assertGreater(position, 0.1)
        self.assertGreater(rotation, 3.0)


if __name__ == '__main__':
    import unittest
    unittest.main()
