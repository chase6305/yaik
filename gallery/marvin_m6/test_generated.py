"""Regressions must exercise the generated APIs themselves on both arms."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase, mock
import sys

import numpy as np

from gallery.marvin_m6 import validate
from gallery.marvin_m6.validate import CppSolver, PythonSolver, UrdfArm, cases_for, pose_error


class TestGeneratedSolvers(TestCase):
    def test_custom_directory_uses_its_kernels_and_runtime_without_leaking_imports(self):
        import python_run_import
        original_runtime = python_run_import
        with tempfile.TemporaryDirectory(prefix='marvin-custom-validation-') as temporary:
            directory = Path(temporary)
            (directory/'python_run_import.py').write_text('bundle_marker = 17\n')
            for side in ('left', 'right'):
                name = f'marvin_m6_{side}_arm'
                (directory/(name+'_ik_generated.py')).write_text(
                    'from python_run_import import bundle_marker\n'
                    f'def {name}_fk(q):\n'
                    '    import numpy as np\n'
                    '    target=np.eye(4); target[0,3]=bundle_marker; return target\n'
                    f'def {name}_ik_solve(target,q5):\n'
                    '    return []\n')
            solver = PythonSolver('left', solver_dir=directory)
            self.assertEqual(solver.fk(np.zeros(7))[0, 3], 17)
            self.assertEqual(solver.ik(np.eye(4), 0.), [])
            self.assertIs(sys.modules['python_run_import'], original_runtime)
            with mock.patch.object(validate, 'cases_for', return_value=[('known', np.zeros(7))]), \
                    mock.patch.object(validate, 'negative_checks', return_value={'passed': True}), \
                    redirect_stdout(StringIO()):
                output = directory/'report.json'
                status = validate.main(['--samples', '0', '--python-only', '--solver-dir', str(directory),
                                        '--output', str(output)])
            self.assertEqual(status, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report['generated_solver_dir'], str(directory))
            self.assertEqual(report['generated_runtime']['python'], str(directory/'python_run_import.py'))
            for side in ('left', 'right'):
                self.assertFalse(report['results'][f'python/{side}/generated']['passed'])
                self.assertTrue(report['results'][f'python/{side}/bounded_analytical']['passed'])
            self.assertIs(sys.modules['python_run_import'], original_runtime)

    def test_failed_custom_runtime_import_restores_the_callers_runtime(self):
        import python_run_import
        original_runtime = python_run_import
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'python_run_import.py').write_text("raise ValueError('broken bundled helper')\n")
            with self.assertRaisesRegex(ValueError, 'broken bundled helper'):
                PythonSolver('left', solver_dir=directory)
            self.assertIs(sys.modules['python_run_import'], original_runtime)

    def test_generated_apis_recover_zero_and_wrist_axis_targets(self):
        native = shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file()
        with tempfile.TemporaryDirectory(prefix='marvin-generated-test-') as temporary:
            for side in ('left', 'right'):
                arm = UrdfArm(side)
                solvers = {'python': PythonSolver(side)}
                if native:
                    solvers['cpp'] = CppSolver(side, temporary)
                cases = [(label, q) for label, q in cases_for(arm, 0, 20261103)
                         if label.startswith(('special/zero', 'special/J4_', 'wrist_axis/', 'wrist_near_axis/',
                                              'wrist_narrow_family/', 'wrist_near_limit/', 'wrist_vertical_axis/',
                                              'shoulder_limit/', 'elbow_repeated_limit/'))]
                for language, solver in solvers.items():
                    for joint in range(7):
                        for value in (2.**30, 1e20, -1e20, 1e300, -1e300):
                            angles = np.zeros(7); angles[joint] = value
                            with self.subTest(side=side, language=language, large_fk_joint=joint, value=value):
                                np.testing.assert_allclose(solver.fk(angles), arm.fk(angles), atol=1e-13, rtol=0)
                    for label, q in cases:
                        with self.subTest(side=side, language=language, case=label):
                            target = arm.fk(q)
                            valid = []
                            for candidate in solver.ik(target, q[4]):
                                normalized = arm.normalize(candidate)
                                if normalized is None or abs(normalized[4]-q[4]) > 1e-8:
                                    continue
                                position, rotation = pose_error(arm.fk(normalized), target)
                                if position <= 1e-7 and rotation <= 1e-6:
                                    valid.append(normalized)
                            self.assertTrue(valid, f'{side}/{language}/{label} lost every bounded solution')

    def test_validation_exit_status_rejects_generated_failures(self):
        def result(solver, arm, cases):
            return {'passed': solver.bounded, 'groups': {}, 'failures': [] if solver.bounded else [{}],
                    'max_fk_matrix_error': 0.0}
        with tempfile.TemporaryDirectory(prefix='marvin-validation-gate-') as temporary:
            output = Path(temporary)/'report.json'
            with mock.patch.object(validate, 'run_solver', side_effect=result), \
                    mock.patch.object(validate, 'negative_checks', return_value={'passed': True}), \
                    redirect_stdout(StringIO()):
                status = validate.main(['--samples', '0', '--python-only', '--output', str(output)])
            self.assertEqual(status, 1)
            self.assertFalse(json.loads(output.read_text())['passed'])
