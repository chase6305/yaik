"""URDF frame, analytical singularity, and bounded integration regressions."""

from pathlib import Path
import hashlib
import json
import math
import sys
from unittest import mock, TestCase

import numpy as np
import sympy as sp
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate import PythonSolver, UrdfArm, cases_for, load_module, negative_checks, pose_error, run_solver
from fk.robots import RobotDescription


class TestMarvinM6(TestCase):
    def test_wrist_family_basis_matches_full_rotation_products(self):
        module = load_module(HERE/'solver.py', 'marvin_m6_family_basis')
        rng = np.random.default_rng(20261107)
        for _ in range(30):
            t3, t4, t5 = rng.uniform(-math.pi, math.pi, 3)
            rotation = module._rz(float(rng.uniform(-math.pi, math.pi)))@module.RX_PLUS
            cosine, sine, constant = module._wrist_family_coefficients(rotation, t3, t4, t5)
            for t6 in (-math.pi, -math.pi/2, 0., 1e-12, math.pi/2, math.pi):
                expected = rotation@module._distal_rotation(t3, t4, t5, t6).T@module.RX_PI
                reconstructed = cosine*math.cos(t6)+sine*math.sin(t6)+constant
                np.testing.assert_allclose(reconstructed, expected, atol=2e-15, rtol=0)

    def test_equivalent_pi_offsets_from_blas_keep_the_saved_dh_convention(self):
        from gallery.marvin_m6 import generate
        real_converter = generate.convert_chain_into_robot

        def flipped_pi_offsets(*args, **kwargs):
            model = real_converter(*args, **kwargs)
            model.auxiliary_data.unknown_offset = [
                -value if abs(abs(value)-math.pi) < 1e-6 else value
                for value in model.auxiliary_data.unknown_offset]
            return model

        for side in ('left', 'right'):
            saved = yaml.safe_load((HERE / f'marvin_m6_{side}_arm.yaml').read_text())
            with mock.patch.object(generate, 'convert_chain_into_robot', side_effect=flipped_pi_offsets):
                converted = generate.urdf_model(HERE / 'urdf' / f'{side}_arm.urdf', side)
            self.assertEqual(generate.model_dict(converted), saved)

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

    def test_large_finite_angles_keep_their_urdf_offset_phase(self):
        # FK accepts finite angles outside IK's joint limits. Adding pi to a
        # huge float first would erase the offset, changing the actual pose.
        for side in ('left', 'right'):
            arm = UrdfArm(side)
            solvers = (PythonSolver(side), PythonSolver(side, bounded=True))
            for joint in range(7):
                for value in (2.**30, 1e20, -1e20, 1e300, -1e300):
                    q = np.zeros(7); q[joint] = value
                    expected = arm.fk(q)
                    for solver in solvers:
                        with self.subTest(side=side, joint=joint, value=value, bounded=solver.bounded):
                            np.testing.assert_allclose(solver.fk(q), expected, atol=1e-13, rtol=0)

    def test_source_snapshots_match_generation_provenance(self):
        provenance = json.loads((HERE / 'provenance.json').read_text())
        self.assertEqual(set(provenance['urdf_sha256']), {'left_arm.urdf', 'right_arm.urdf'})
        for name, digest in provenance['urdf_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE / 'urdf' / name).read_bytes()).hexdigest(), digest)
        self.assertIn('codegen/skeleton/guard_codegen.py', provenance['generator_sources_sha256'])
        self.assertIn('gallery/marvin_m6/generate.py', provenance['generator_sources_sha256'])
        for name, digest in (provenance['generator_sources_sha256'] | provenance['runtime_sha256']).items():
            self.assertEqual(hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest(), digest, name)

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
