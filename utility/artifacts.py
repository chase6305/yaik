"""Publish staged artifacts while preserving the previous set on write errors."""

from pathlib import Path
import shutil
import stat
import tempfile


def copy_runtime(staging, language='both'):
    """Bundle the matching runtime helpers with exported Python/C++ kernels."""
    root = Path(__file__).resolve().parents[1]
    names = (['python_run_import.py'] if language == 'python' else
             ['yaik_cpp_common.h'] if language == 'cpp' else
             ['python_run_import.py', 'yaik_cpp_common.h'])
    for name in names:
        shutil.copy2(root/name, Path(staging)/name)
    return [Path(staging)/name for name in names]


def publish_artifacts(staging, output):
    """Replace generated files, rolling back ordinary errors and interruptions.

    All destination types and backups are checked before the first replacement.
    Individual replacements are atomic. A process kill or power loss can still
    interrupt a multi-file publication; concurrent writers need external locking.
    """
    staging, output = Path(staging), Path(output)
    paths = sorted(staging.iterdir())
    existing = set()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Staged artifact must be a regular file: {path}')
        destination = output/path.name
        try:
            mode = destination.lstat().st_mode
        except FileNotFoundError:
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise ValueError(f'Artifact destination is not a file: {destination}')
        existing.add(path.name)
    backup = Path(tempfile.mkdtemp(prefix='.yaik-publish-', dir=output))
    preserve_backup = False
    try:
        for name in existing:
            shutil.copy2(output/name, backup/name, follow_symlinks=False)
        attempted = []
        try:
            for path in paths:
                attempted.append(path.name)
                path.replace(output/path.name)
        except BaseException:
            failures = []
            for name in reversed(attempted):
                try:
                    if name in existing:
                        (backup/name).replace(output/name)
                    else:
                        (output/name).unlink(missing_ok=True)
                except OSError as error:
                    failures.append(f'{name}: {error}')
            if failures:
                # Preserve backup files for manual recovery instead of deleting
                # the only remaining copy when restoring also fails.
                preserve_backup = True
                raise OSError(f'Artifact rollback failed; backups are in {backup}: '
                              + '; '.join(failures))
            raise
    finally:
        if not preserve_backup:
            shutil.rmtree(backup)
    return [str(output/path.name) for path in paths]
