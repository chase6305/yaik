"""Fault injection protects existing outputs when an export cannot finish."""

from pathlib import Path
import tempfile
from unittest import TestCase, mock

from utility.artifacts import publish_artifacts


class TestArtifactPublication(TestCase):
    def test_destination_directory_is_rejected_before_any_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, output = root/'staging', root/'output'
            staging.mkdir(); output.mkdir()
            (staging/'a.py').write_text('new a')
            (staging/'z.cpp').write_text('new z')
            (output/'a.py').write_text('old a')
            (output/'z.cpp').mkdir()
            with self.assertRaisesRegex(ValueError, 'destination is not a file'):
                publish_artifacts(staging, output)
            self.assertEqual((output/'a.py').read_text(), 'old a')
            self.assertTrue((staging/'a.py').exists())
            self.assertEqual(sorted(path.name for path in output.iterdir()), ['a.py', 'z.cpp'])

    def test_write_error_or_interrupt_restores_files_and_removes_new_outputs(self):
        for failure in (OSError('disk write failed'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                staging, output = root/'staging', root/'output'
                staging.mkdir(); output.mkdir()
                for name in ('a.py', 'b.py', 'z.cpp'):
                    (staging/name).write_text('new '+name)
                (output/'a.py').write_text('old a')
                (output/'z.cpp').write_text('old z')
                (output/'keep.txt').write_text('unrelated')
                replace = Path.replace
                def fail_last(path, target):
                    if path == staging/'z.cpp':
                        raise failure
                    return replace(path, target)
                with mock.patch.object(Path, 'replace', fail_last):
                    with self.assertRaises(type(failure)):
                        publish_artifacts(staging, output)
                self.assertEqual((output/'a.py').read_text(), 'old a')
                self.assertEqual((output/'z.cpp').read_text(), 'old z')
                self.assertEqual((output/'keep.txt').read_text(), 'unrelated')
                self.assertFalse((output/'b.py').exists())
                self.assertFalse(list(output.glob('.yaik-publish-*')))

    def test_backup_failure_keeps_original_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, output = root/'staging', root/'output'
            staging.mkdir(); output.mkdir()
            (staging/'a.py').write_text('new')
            (output/'a.py').write_text('old')
            with mock.patch('utility.artifacts.shutil.copy2', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    publish_artifacts(staging, output)
            self.assertEqual((output/'a.py').read_text(), 'old')
            self.assertFalse(list(output.glob('.yaik-publish-*')))

    def test_failed_rollback_preserves_recovery_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, output = root/'staging', root/'output'
            staging.mkdir(); output.mkdir()
            (staging/'a.py').write_text('new a')
            (staging/'z.cpp').write_text('new z')
            (output/'a.py').write_text('old a')
            replace = Path.replace
            def fail_write_and_restore(path, target):
                if path == staging/'z.cpp' or path.parent.name.startswith('.yaik-publish-'):
                    raise OSError('cannot replace')
                return replace(path, target)
            with mock.patch.object(Path, 'replace', fail_write_and_restore):
                with self.assertRaisesRegex(OSError, 'rollback failed; backups are in'):
                    publish_artifacts(staging, output)
            backups = list(output.glob('.yaik-publish-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0]/'a.py').read_text(), 'old a')

    def test_success_replaces_symlink_itself_and_preserves_its_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging, output = root/'staging', root/'output'
            staging.mkdir(); output.mkdir()
            (root/'user-file').write_text('user data')
            (output/'a.py').symlink_to('../user-file')
            (output/'keep.txt').write_text('keep')
            (staging/'a.py').write_text('generated')
            self.assertEqual(publish_artifacts(staging, output), [str(output/'a.py')])
            self.assertFalse((output/'a.py').is_symlink())
            self.assertEqual((output/'a.py').read_text(), 'generated')
            self.assertEqual((root/'user-file').read_text(), 'user data')
            self.assertEqual((output/'keep.txt').read_text(), 'keep')
