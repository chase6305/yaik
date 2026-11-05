"""Exercise the user CLI as a subprocess, including its files and exit codes."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase

import numpy as np
import sympy as sp
import yaml

from fk.robots import DHEntry, RobotDescription
from solver.equation_types import Unknown, UnknownType
from gallery.marvin_m6.validate import UrdfArm, pose_error


ROOT = Path(__file__).resolve().parent
MARVIN = ROOT / 'gallery/marvin_m6'
JOINTS = [0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]


class TestCLI(TestCase):
    def run_cli(self, *arguments, cwd=None, text=None, script='yaik.py', no_site=False):
        command = [sys.executable, '-B'] + (['-S'] if no_site else [])
        return subprocess.run(command + [str(ROOT / script), *map(str, arguments)],
                              cwd=cwd, input=text, text=True, capture_output=True, timeout=60)

    def test_help_and_model_list_need_no_third_party_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in (('--help',), ('marvin', 'ik', '--help'), ('audit', '--help'), ('models',)):
                with self.subTest(arguments=arguments):
                    result = self.run_cli(*arguments, cwd=temporary, no_site=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('puma', json.loads(result.stdout)['models'])
            for script in ('py_codegen.py', 'cpp_codegen.py'):
                result = self.run_cli('--help', script=script, cwd=temporary, no_site=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'usage: {script} ', result.stdout)
                self.assertNotIn(f'{script} codegen', result.stdout)

    def test_fk_json_pipes_into_bounded_ik_for_either_side(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                result = self.run_cli('marvin', 'fk', '--side', side, '--joints', json.dumps(JOINTS))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                data = json.loads(result.stdout)
                self.assertEqual(data['q5'], JOINTS[4])
                solution = self.run_cli('marvin', 'ik', '--side', side, '--target', '-', text=result.stdout)
                self.assertEqual(solution.returncode, 0, solution.stderr)
                solutions = json.loads(solution.stdout)['solutions']
                self.assertTrue(solutions)
                arm = UrdfArm(side)
                for candidate in solutions:
                    position, rotation = pose_error(arm.fk(candidate), np.asarray(data['target']))
                    self.assertLessEqual(position, 1e-7)
                    self.assertLessEqual(rotation, 1e-6)
                    self.assertEqual(candidate[4], JOINTS[4])
                    self.assertTrue(np.all(np.asarray(candidate) >= arm.lower))
                    self.assertTrue(np.all(np.asarray(candidate) <= arm.upper))

    def test_bad_json_frames_shapes_and_fixed_joint_have_clear_errors(self):
        matrix = np.eye(4).tolist()
        cases = [('{', []), (json.dumps(matrix), []),
                 (json.dumps({'target': matrix, 'side': 'right', 'q5': 0}), []),
                 (json.dumps({'target': matrix, 'q5': True}), []),
                 (json.dumps([[1, 0], [0, 1]]), ['--q5', '0']),
                 (json.dumps(matrix), ['--q5', 'nan']),
                 (json.dumps(matrix), ['--q5', '4']),
                 ('[[NaN]]', ['--q5', '0'])]
        for text, extra in cases:
            with self.subTest(text=text, extra=extra):
                result = self.run_cli('marvin', 'ik', '--side', 'left', '--target', '-', *extra, text=text)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
                self.assertIn('yaik: error:', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
        for joints in ('[1, 2]', '[true, 0, 0, 0, 0, 0, 0]', '[0, 0, 0, 0, 0, 0, "x"]'):
            result = self.run_cli('marvin', 'fk', '--side', 'left', '--joints', joints)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Traceback', result.stderr)

    def test_no_solution_has_json_output_and_exit_one(self):
        target = np.eye(4)
        target[:3, 3] = 10
        result = self.run_cli('marvin', 'ik', '--side', 'right', '--target', '-',
                              '--q5', '0.4', text=json.dumps(target.tolist()))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'no_solution')
        self.assertEqual(json.loads(result.stdout)['solutions'], [])
        self.assertEqual(json.loads(result.stdout)['reachability'], 'undetermined')
        self.assertFalse(json.loads(result.stdout)['certified_complete'])

    def test_malformed_exported_module_has_a_clear_input_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for side in ('left', 'right'):
                (directory/f'marvin_m6_{side}_arm_ik_generated.py').write_text('def broken(:\n')
            result = self.run_cli('marvin', 'validate', '--solver-dir', directory,
                                  '--samples', '0', '--python-only', '--output', directory/'report.json')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertIn('invalid syntax', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_completeness_audit_does_not_promote_regression_coverage_to_a_proof(self):
        tree = MARVIN / 'marvin_m6_left_arm_ik.yaml'
        result = self.run_cli('audit', tree)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report['certified_complete'])
        self.assertTrue(report['family_nodes'])
        self.assertTrue(report['required_proofs'])
        result = self.run_cli('audit', tree, '--require-complete')
        self.assertEqual(result.returncode, 1, result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'not-created'
            result = self.run_cli('codegen', tree, '--require-complete', '--output-dir', output)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'not_certified')
            self.assertFalse(output.exists())

    def test_file_output_and_errors_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results' / 'pose.json'
            result = self.run_cli('marvin', 'fk', '--side', 'right', '--joints', json.dumps(JOINTS),
                                  '--output', 'results/pose.json', cwd=temporary)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertEqual(json.loads(output.read_text())['side'], 'right')
            result = self.run_cli('marvin', 'ik', '--side', 'right', '--target', 'missing.json', cwd=temporary)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Traceback', result.stderr)

    def test_codegen_accepts_legacy_tuple_trees_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sentinel = output / 'keep.txt'
            sentinel.write_text('keep')
            tree = MARVIN / 'marvin_m6_left_arm_ik.yaml'
            result = self.run_cli('codegen', tree, '--output-dir', output, cwd=temporary)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['artifacts']), 2)
            for extension in ('py', 'cpp'):
                name = 'marvin_m6_left_arm_ik_generated.' + extension
                self.assertEqual((output / name).read_bytes(), (MARVIN / name).read_bytes())
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_codegen_can_bundle_runtime_for_an_external_python_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/'export'
            result = self.run_cli('codegen', MARVIN/'marvin_m6_left_arm_ik.yaml',
                                  '--with-runtime', '--output-dir', output, cwd=temporary)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['artifacts']), 4)
            self.assertEqual((output/'python_run_import.py').read_bytes(),
                             (ROOT/'python_run_import.py').read_bytes())
            self.assertEqual((output/'yaik_cpp_common.h').read_bytes(),
                             (ROOT/'yaik_cpp_common.h').read_bytes())
            script = '''import sys
sys.path.insert(0, sys.argv[1])
import marvin_m6_left_arm_ik_generated as arm
q=[.3,-.5,.7,-1.,.4,-.2,.1]
assert arm.marvin_m6_left_arm_ik_solve(arm.marvin_m6_left_arm_fk(q), q[4])
'''
            external = subprocess.run([sys.executable, '-B', '-I', '-c', script, str(output)],
                                      cwd=temporary, text=True, capture_output=True, timeout=30)
            self.assertEqual(external.returncode, 0, external.stderr)

    def test_marvin_bundle_is_checked_and_file_conflicts_preserve_existing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'export'
            result = self.run_cli('marvin', 'generate', '--with-runtime', '--output-dir', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output/'provenance.json').read_text())
            self.assertEqual(len(metadata['artifact_sha256']), 10)
            for name in ('python_run_import.py', 'yaik_cpp_common.h'):
                self.assertIn(name, metadata['artifact_sha256'])
            result = self.run_cli('marvin', 'check', '--output-dir', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            changed = json.loads(json.dumps(metadata))
            changed['generator_sources_sha256']['codegen/skeleton/guard_codegen.py'] = '0'*64
            (output/'provenance.json').write_text(json.dumps(changed))
            result = self.run_cli('marvin', 'check', '--output-dir', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Stale artifact', result.stderr)
            (output/'provenance.json').write_text(json.dumps(metadata))
            (output/'python_run_import.py').write_text('# stale helper\n')
            result = self.run_cli('marvin', 'check', '--output-dir', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Stale artifact', result.stderr)
            sentinel = output/'marvin_m6_left_arm.yaml'
            sentinel.write_text('old result')
            conflict = output/'marvin_m6_right_arm_ik_generated.cpp'
            conflict.unlink(); conflict.mkdir()
            before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
            result = self.run_cli('marvin', 'generate', '--with-runtime', '--output-dir', output)
            self.assertEqual(result.returncode, 2)
            self.assertIn('destination is not a file', result.stderr)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()})

    def test_fast_marvin_export_preserves_seed_and_rejects_mismatched_inputs_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, urdf = root / 'export', root / 'urdf'
            output.mkdir()
            (output / 'marvin_m6_user.py').write_text('# User file\n')
            result = self.run_cli('marvin', 'generate', '--output-dir', output, '--seed', '99')
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = json.loads((output / 'provenance.json').read_text())
            self.assertEqual(recorded['seed'], 42)
            self.assertEqual(len(recorded['artifact_sha256']), 8)
            self.assertNotIn('marvin_m6_user.py', recorded['artifact_sha256'])
            result = self.run_cli('marvin', 'check', '--output-dir', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            urdf.mkdir()
            for side in ('left', 'right'):
                text = (MARVIN / 'urdf' / f'{side}_arm.urdf').read_text()
                if side == 'right':
                    self.assertIn('0.018 -0.314 0', text)
                    text = text.replace('0.018 -0.314 0', '0.019 -0.314 0')
                (urdf / f'{side}_arm.urdf').write_text(text)
            result = self.run_cli('marvin', 'generate', '--urdf-dir', urdf, '--output-dir', output)
            self.assertEqual(result.returncode, 2)
            self.assertIn('does not match', result.stderr)
            self.assertIn('robot.parameters_value.', result.stderr)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)

    def test_export_records_unknown_provenance_without_claiming_a_search_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf, trees, output = root / 'urdf', root / 'trees', root / 'export'
            urdf.mkdir()
            trees.mkdir()
            for side in ('left', 'right'):
                name = f'{side}_arm.urdf'
                text = (MARVIN / 'urdf' / name).read_text()
                (urdf / name).write_text(text.replace('<robot', '<!-- User snapshot -->\n<robot', 1))
                name = f'marvin_m6_{side}_arm_ik.yaml'
                (trees / name).write_bytes((MARVIN / name).read_bytes())
            result = self.run_cli('marvin', 'generate', '--urdf-dir', urdf, '--tree-dir', trees,
                                  '--output-dir', output, '--seed', '99')
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = json.loads((output / 'provenance.json').read_text())
            self.assertIsNone(recorded['seed'])
            self.assertIsNone(recorded['source_revision'])
            result = self.run_cli('marvin', 'check', '--urdf-dir', urdf, '--output-dir', output)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_generation_of_a_small_model_and_legacy_codegen_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            robot = RobotDescription('cli_slide')
            variable = sp.Symbol('d_0')
            robot.unknowns = [Unknown(variable, UnknownType.Translational.name, lower_bound=0, upper_bound=1)]
            robot.dh_params = [DHEntry(0, 0, variable, 0)]
            model, output = root / 'slide.yaml', root / 'generated'
            model.write_text(yaml.safe_dump(robot.to_dict()))
            result = self.run_cli('generate', '--model', model, '--output-dir', output, '--language', 'python')
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(result.stdout)
            self.assertEqual(manifest['seed'], 42)
            self.assertTrue((output / 'cli_slide_ik.yaml').is_file())
            scope = {'__name__': 'cli_slide_test'}
            source = output / 'cli_slide_ik_generated.py'
            exec(compile(source.read_text(), str(source), 'exec'), scope)
            target = scope['cli_slide_fk'](np.array([0.25]))
            candidates = scope['cli_slide_ik_solve'](target)
            self.assertTrue(candidates)
            self.assertAlmostEqual(candidates[0][0], 0.25)
            result = self.run_cli(output / 'cli_slide_ik.yaml', '--output-dir', root / 'legacy',
                                  script='py_codegen.py')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['artifacts']), 1)

    def test_model_tree_confusion_and_mode_conflicts_are_reported_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in (
                    ('codegen', MARVIN / 'marvin_m6_left_arm.yaml'),
                    ('generate', '--model', MARVIN / 'marvin_m6_left_arm_ik.yaml'),
                    ('marvin', 'generate', '--solve', '--models-only'),
                    ('marvin', 'generate', '--models-only', '--with-runtime'),
                    ('generate', '--builtin', 'puma', '--seed', '-1')):
                with self.subTest(arguments=arguments):
                    result = self.run_cli(*arguments, cwd=temporary)
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn('Traceback', result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            tree_dir = Path(temporary) / 'broken-trees'
            tree_dir.mkdir()
            (tree_dir / 'marvin_m6_left_arm_ik.yaml').write_text('robot: [')
            output = Path(temporary) / 'export'
            result = self.run_cli('marvin', 'generate', '--tree-dir', tree_dir, '--output-dir', output)
            self.assertEqual(result.returncode, 2)
            self.assertIn('invalid IK tree YAML', result.stderr)
            self.assertNotIn('Traceback', result.stderr)
            self.assertFalse(output.exists())
