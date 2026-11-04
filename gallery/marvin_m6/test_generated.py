"""Regressions must exercise the generated APIs themselves on both arms."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase, mock

from gallery.marvin_m6 import validate
from gallery.marvin_m6.validate import CppSolver, PythonSolver, UrdfArm, cases_for, pose_error


class TestGeneratedSolvers(TestCase):
    def test_generated_apis_recover_zero_and_wrist_axis_targets(self):
        native = shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file()
        with tempfile.TemporaryDirectory(prefix='marvin-generated-test-') as temporary:
            for side in ('left', 'right'):
                arm = UrdfArm(side)
                solvers = {'python': PythonSolver(side)}
                if native:
                    solvers['cpp'] = CppSolver(side, temporary)
                cases = [(label, q) for label, q in cases_for(arm, 0, 20261103)
                         if label.startswith(('special/zero', 'special/J4_', 'wrist_axis/', 'wrist_near_axis/'))]
                for language, solver in solvers.items():
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
