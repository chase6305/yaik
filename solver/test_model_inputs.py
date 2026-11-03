"""Model preflight, bounded sampling, and reproducible offline generation."""
import contextlib
import io
import random
from unittest import TestCase, mock

import numpy as np
import sympy as sp

from fk.fk_equations import DHEntry
from fk.robot_models import puma_robot
from fk.robots import RobotAuxiliaryData, RobotDescription
from ik_solve import check_robot, run_ik, RunIKOption
from solver.build_equations import build_ik_equations
from solver.degenerate_analyse.numerical_check_data import generate_test_cases
from solver.equation_types import Unknown, UnknownType


def bounded_joint(kind=UnknownType.Translational.name, lower=0, upper=0.01):
    x = sp.Symbol('x')
    robot = RobotDescription('bounded_joint')
    robot.unknowns = [Unknown(x, kind, lower, upper)]
    robot.dh_params = [DHEntry(0, 0, x, 0) if kind == UnknownType.Translational.name
                       else DHEntry(0, 0, 0, x)]
    return robot


class TestModelInputs(TestCase):
    def assert_invalid(self, robot):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(check_robot(robot))

    def test_preflight_checks_symbols_inside_dh_expressions(self):
        robot = puma_robot()
        length = sp.Symbol('length')
        robot.dh_params[0].a = 2 * length
        self.assert_invalid(robot)
        robot.symbolic_parameters.add(length)
        robot.parameters_bound[length] = (0.1, 0.9)
        self.assertTrue(check_robot(robot))

    def test_preflight_rejects_nonfinite_values_and_bad_bounds(self):
        for lower, upper in [(np.nan, 1), (-1, np.inf), (1, 0)]:
            self.assert_invalid(bounded_joint(lower=lower, upper=upper))
        length = sp.Symbol('length')
        for bounds in [(2, 1), (0, np.nan), (0, np.inf), (0,)]:
            robot = puma_robot()
            robot.symbolic_parameters.add(length)
            robot.parameters_bound[length] = bounds
            self.assert_invalid(robot)
        for invalid in [np.nan, np.inf, -np.inf]:
            robot = puma_robot()
            robot.parameters_value[length] = invalid
            self.assert_invalid(robot)
            robot = bounded_joint()
            robot.dh_params[0].a = invalid
            self.assert_invalid(robot)

    def test_preflight_checks_joint_and_redundant_symbols(self):
        self.assert_invalid(RobotDescription('empty'))
        robot = puma_robot()
        robot.unknown_as_parameter_more_dof = [sp.Symbol('missing')]
        self.assert_invalid(robot)
        robot.unknown_as_parameter_more_dof = [robot.unknowns[0].symbol] * 2
        self.assert_invalid(robot)
        robot = puma_robot()
        robot.unknowns[1] = robot.unknowns[0]
        self.assert_invalid(robot)
        robot = bounded_joint()
        robot.dh_params[0].d *= 2
        self.assert_invalid(robot)
        robot = bounded_joint()
        robot.unknowns[0] = Unknown('not_a_symbol')
        self.assert_invalid(robot)
        robot = puma_robot()
        for index in (6, 7):
            symbol = sp.Symbol('th_' + str(index))
            robot.unknowns.append(Unknown(symbol))
            robot.dh_params.append(DHEntry(0, 0, 0, symbol))
        robot.unknown_as_parameter_more_dof = [robot.unknowns[0].symbol]
        self.assert_invalid(robot)
        robot.unknown_as_parameter_more_dof.append(robot.unknowns[1].symbol)
        self.assertTrue(check_robot(robot))

    def test_automatic_special_samples_respect_joint_type_and_limits(self):
        for robot in [bounded_joint(), bounded_joint(UnknownType.Revolute.name, 0.2, 0.4)]:
            equations = build_ik_equations(robot)
            samples = generate_test_cases(equations, n_test_cases=400, degenerate_ratio=0.5)
            joint = robot.unknowns[0]
            values = [float(sample[joint.symbol]) for sample in samples]
            self.assertTrue(all(joint.lower_bound <= value <= joint.upper_bound for value in values))
            self.assertIn(joint.lower_bound, values)
            self.assertIn(joint.upper_bound, values)
        joint = bounded_joint().unknowns[0]
        self.assertEqual(joint.degenerate_check_value, [sp.S.Zero])

    def test_preflight_rejects_coupled_dh_and_malformed_auxiliary_data(self):
        robot = bounded_joint()
        robot.dh_params[0].theta = robot.unknowns[0].symbol
        self.assert_invalid(robot)
        for field in ('symbolic_parameters', 'parameters_value', 'parameters_bound'):
            with self.subTest(parameter_field=field):
                robot = bounded_joint()
                joint = robot.unknowns[0].symbol
                if field == 'symbolic_parameters':
                    robot.symbolic_parameters.add(joint)
                else:
                    getattr(robot, field)[joint] = 0 if field == 'parameters_value' else (0, 0.01)
                self.assert_invalid(robot)  # Parameter substitution must not replace the DH joint.
                robot.auxiliary_data = RobotAuxiliaryData()
                robot.auxiliary_data.pre_transform_sp = sp.eye(4)
                robot.auxiliary_data.pre_transform_sp[0, 3] = joint
                self.assert_invalid(robot)
        robot = bounded_joint()
        alias = sp.Symbol(robot.unknowns[0].symbol.name, real=True)
        robot.symbolic_parameters.add(alias)
        robot.parameters_bound[alias] = (0, 0.01)
        self.assert_invalid(robot)
        for offsets in ([], [0, 0], [np.nan], [np.inf], 0.1):
            robot = bounded_joint()
            robot.auxiliary_data = RobotAuxiliaryData(unknown_offset=offsets)
            self.assert_invalid(robot)
        for matrix in (sp.eye(3), sp.eye(4) * sp.nan):
            for field in ('pre_transform_sp', 'post_transform_sp'):
                robot = bounded_joint()
                robot.auxiliary_data = RobotAuxiliaryData()
                setattr(robot.auxiliary_data, field, matrix)
                self.assert_invalid(robot)
        robot = bounded_joint()
        length = sp.Symbol('tool_length')
        robot.auxiliary_data = RobotAuxiliaryData(unknown_offset=[0.1])
        robot.auxiliary_data.post_transform_sp = sp.eye(4)
        robot.auxiliary_data.post_transform_sp[0, 3] = length
        self.assert_invalid(robot)
        robot.symbolic_parameters.add(length)
        robot.parameters_bound[length] = (0, 0.2)
        self.assertTrue(check_robot(robot))

    def test_explicit_branch_samples_and_empty_batches(self):
        robot = bounded_joint()
        robot.unknowns[0].degenerate_check_value[:] = [sp.Symbol('unused_special_value')]
        equations = build_ik_equations(robot)
        x = robot.unknowns[0].symbol
        samples = generate_test_cases(equations, {x: 2}, n_test_cases=3)
        self.assertTrue(all(sample[x] == 2 for sample in samples))
        self.assertEqual(len(generate_test_cases(equations, n_test_cases=3, degenerate_ratio=0)), 3)
        self.assertEqual(generate_test_cases(equations, n_test_cases=0), [])
        for kwargs in [{'n_test_cases': -1}, {'degenerate_ratio': 1.1}, {'degenerate_ratio': -0.1}]:
            with self.assertRaises(ValueError):
                generate_test_cases(equations, **kwargs)

    def test_seeded_runs_repeat_and_restore_callers_random_state(self):
        option = RunIKOption(random_seed=42)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with mock.patch('ik_solve._run_ik', side_effect=lambda *args: (random.random(), np.random.random())):
            first = run_ik(bounded_joint(), run_ik_option=option)
            second = run_ik(bounded_joint(), run_ik_option=option)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_state)
        restored = np.random.get_state()
        self.assertEqual(restored[0], numpy_state[0])
        np.testing.assert_array_equal(restored[1], numpy_state[1])
        self.assertEqual(restored[2:], numpy_state[2:])
        with mock.patch('ik_solve._run_ik', side_effect=RuntimeError('failed generation')):
            with self.assertRaises(RuntimeError):
                run_ik(bounded_joint(), run_ik_option=option)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
