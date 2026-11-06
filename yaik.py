"""Command-line generation, code export, and Marvin M6 FK/IK."""

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import json
import keyword
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
MARVIN = ROOT / 'gallery' / 'marvin_m6'
BUILTINS = {
    'puma': 'puma_robot', 'arm_robo': 'arm_robo', 'mini_dd': 'mini_dd_robot',
    'yaskawa_hc10': 'yaskawa_HC10_robot', 'ur10_urdf': 'ur10_urdf_robot',
    'kuka_iiwa': 'kuka_iiwa_robot', 'franka_panda': 'franka_panda_robot',
    'pr2_right_gripper': 'pr2_r_gripper_palm',
    'atlas_left_hand': 'atlas_l_hand', 'atlas_right_hand': 'atlas_r_hand',
}


def seed_value(value):
    value = int(value)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError('seed must be between 0 and 4294967295')
    return value


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def read_yaml(path):
    import yaml

    # Saved gallery trees contain legacy tuple tags, but need no object loader.
    class Loader(getattr(yaml, 'CSafeLoader', yaml.SafeLoader)):
        pass

    Loader.add_constructor('tag:yaml.org,2002:python/tuple',
                           lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
    try:
        data = yaml.load(Path(path).read_text(), Loader=Loader)
    except yaml.YAMLError as error:
        raise ValueError(f'{path}: invalid YAML: {error}') from None
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return data


def read_json(text, source):
    def invalid_constant(value):
        raise ValueError(f'{source}: {value} is not a finite JSON number')
    try:
        return json.loads(text, parse_constant=invalid_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f'{source}:{error.lineno}:{error.colno}: {error.msg}') from None


def emit(data, output=None):
    text = json.dumps(data, indent=2, allow_nan=False) + '\n'
    if output is None:
        sys.stdout.write(text)
    else:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Do not leave a partial result if writing is interrupted.
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
                stream.close()
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def check_name(name):
    if (not isinstance(name, str) or not name.isascii() or not name.isidentifier()
            or keyword.iskeyword(name)):
        raise ValueError('robot name must be an ASCII identifier, such as my_robot')


def export_code(tree, output_dir, language, with_runtime=False, refine_seeds=True):
    check_name(tree.robot.name)
    paths = []
    for extension in (('py', 'cpp') if language == 'both' else
                      ('py',) if language == 'python' else ('cpp',)):
        path = Path(output_dir) / f'{tree.robot.name}_ik_generated.{extension}'
        if extension == 'py':
            from codegen.py_codegen.codegen_py import generate_code_python
            generate_code_python(tree, str(path), refine_seeds=refine_seeds)
        else:
            from codegen.cpp_codegen.codegen_cpp import generate_code_cpp
            generate_code_cpp(tree, str(path), use_safe_operator=True, refine_seeds=refine_seeds)
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        paths.append(path)
    if with_runtime:
        from utility.artifacts import copy_runtime
        paths.extend(copy_runtime(output_dir, language))
    return paths


def publish(staging, output):
    from utility.artifacts import publish_artifacts
    return publish_artifacts(staging, output)


def codegen(args):
    from codegen.skeleton.tree_serialize import TreeSerializer
    output = args.output_dir.resolve()
    with redirect_stdout(sys.stderr):
        data = read_yaml(args.tree)
        if 'robot' not in data or 'tree_nodes' not in data:
            raise ValueError('codegen requires an *_ik.yaml solution tree; use generate for a robot model')
        tree = TreeSerializer.load_skeleton_tree(data)
        check_name(tree.robot.name)
        if getattr(args, 'require_complete', False):
            from codegen.skeleton.completeness import audit_completeness
            report = audit_completeness(tree)
            if not report['certified_complete']:
                # JSON is emitted after leaving the diagnostic redirection.
                incomplete_report = report
            else:
                incomplete_report = None
        else:
            incomplete_report = None
        if incomplete_report is None:
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.yaik-', dir=output) as staging:
                export_code(tree, staging, args.language, args.with_runtime, not getattr(args, 'no_extra_recovery', False))
                paths = publish(staging, output)
    if incomplete_report is not None:
        emit(incomplete_report)
        return 1
    emit({'status': 'ok', 'robot': tree.robot.name, 'artifacts': paths,
          'extra_recovery': not getattr(args, 'no_extra_recovery', False)})
    return 0


def audit(args):
    from codegen.skeleton.tree_serialize import TreeSerializer
    from codegen.skeleton.completeness import audit_completeness
    with redirect_stdout(sys.stderr):
        data = read_yaml(args.tree)
        if 'robot' not in data or 'tree_nodes' not in data:
            raise ValueError('audit requires an *_ik.yaml solution tree')
        tree = TreeSerializer.load_skeleton_tree(data)
        report = audit_completeness(tree)
    emit(report, args.output)
    return 1 if args.require_complete and not report['certified_complete'] else 0


def generate(args):
    import yaml
    from fk.robots import RobotDescription
    from ik_solve import check_robot, run_ik, resume_ik_tree, RunIKOption
    from codegen.skeleton.tree_serialize import read_skeleton_tree, save_ik_tree, TreeSerializer

    test_cases = str(args.test_cases.resolve()) if args.test_cases else None
    resume = getattr(args, 'resume_tree', None)
    max_nodes = getattr(args, 'max_nodes', 50)
    resume_record = None
    with redirect_stdout(sys.stderr):
        if resume:
            if test_cases:
                raise ValueError('--test-cases is for a new search; omit it with --resume-tree')
            source = resume.resolve()
            data = read_yaml(source)
            if 'robot' not in data or 'tree_nodes' not in data:
                raise ValueError('--resume-tree requires an *_ik.yaml solution tree')
            tree = TreeSerializer.load_skeleton_tree(data)
            robot = tree.robot
            resume_record = {'schema_version': 1, 'generation_mode': 'resume_saved_tree',
                             'input_tree_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                             'resume_seed': args.seed, 'branch_node_budget': max_nodes,
                             'nodes_before': tree.num_nodes, 'coverage_status': 'not_certified'}
        elif args.builtin:
            from fk import robot_models
            robot = getattr(robot_models, BUILTINS[args.builtin])()
        else:
            data = read_yaml(args.model)
            if 'tree_nodes' in data:
                raise ValueError('generate requires a robot model YAML; use codegen for an *_ik.yaml solution tree')
            robot = RobotDescription.load_from_dict(data)
        check_name(robot.name)
        if not check_robot(robot):
            raise ValueError('invalid robot model; see the input diagnostics above')
        if test_cases and not Path(test_cases).is_file():
            raise FileNotFoundError(test_cases)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.yaik-', dir=output) as staging:
            staging = Path(staging)
            with working_directory(staging):
                if resume:
                    # Branch diagnostics stay in scratch; publish the resumed tree and its exports.
                    with tempfile.TemporaryDirectory(prefix='yaik-resume-') as scratch:
                        with working_directory(scratch):
                            resume_ik_tree(tree, args.seed, max_nodes)
                    save_ik_tree(tree, str(staging/f'{robot.name}_ik.yaml'))
                    tree = read_skeleton_tree(str(staging/f'{robot.name}_ik.yaml'))
                else:
                    run_ik(robot, test_cases, RunIKOption(random_seed=args.seed, max_num_nodes=max_nodes))
                    tree = read_skeleton_tree(str(staging / f'{robot.name}_ik.yaml'))
                export_code(tree, staging, args.language, args.with_runtime, not getattr(args, 'no_extra_recovery', False))
                (staging / f'{robot.name}.yaml').write_text(yaml.safe_dump(robot.to_dict(), sort_keys=True))
                if resume_record is not None:
                    from utility.artifacts import generator_source_hashes
                    resume_record.update(nodes_after=tree.num_nodes, generator_sources_sha256=generator_source_hashes(),
                        extra_recovery=not getattr(args, 'no_extra_recovery', False),
                        artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted(staging.iterdir()) if path.is_file()})
                    (staging/f'{robot.name}_resume.json').write_text(json.dumps(resume_record, indent=2)+'\n')
            paths = publish(staging, output)
    result = {'status': 'ok', 'robot': robot.name, 'seed': args.seed, 'artifacts': paths,
              'extra_recovery': not getattr(args, 'no_extra_recovery', False)}
    if resume_record is not None:
        result.update({key: resume_record[key] for key in
                       ('generation_mode', 'input_tree_sha256', 'nodes_before', 'nodes_after', 'branch_node_budget')})
    emit(result)
    return 0


def marvin_fk(args):
    from gallery.marvin_m6.solver import Solver
    joints = read_json(args.joints, '--joints')
    if (not isinstance(joints, list) or len(joints) != 7 or
            any(isinstance(q, bool) or not isinstance(q, (int, float)) for q in joints)):
        raise ValueError('--joints must be a JSON array of seven real numbers')
    solver = Solver(args.side)
    target = solver.fk(joints)
    emit({'status': 'ok', 'side': args.side, 'base_frame': f'{args.side}_arm_base',
          'tip_frame': f'{args.side}_ee', 'joints': list(joints),
          'q5': float(joints[4]), 'target': target.tolist()}, args.output)
    return 0


def marvin_ik(args):
    from gallery.marvin_m6.solver import Solver
    text = sys.stdin.read() if args.target == '-' else Path(args.target).read_text()
    data = read_json(text, args.target)
    q5 = args.q5
    if isinstance(data, dict):
        for key, expected in (('side', args.side), ('base_frame', f'{args.side}_arm_base'),
                              ('tip_frame', f'{args.side}_ee')):
            if key in data and data[key] != expected:
                raise ValueError(f'target {key}={data[key]!r} does not match {expected!r}')
        if q5 is None:
            q5 = data.get('q5')
        if 'target' not in data:
            raise ValueError('target JSON object must contain a target matrix')
        target = data['target']
    else:
        target = data
    if q5 is None:
        raise ValueError('--q5 is required unless target JSON includes q5 (URDF J5, radians)')
    if isinstance(q5, bool) or not isinstance(q5, (int, float)):
        raise ValueError('q5 must be a real number in radians')
    solutions = Solver(args.side).ik(target, q5)
    emit({'status': 'ok' if solutions else 'no_solution', 'side': args.side,
          'q5': float(q5), 'count': len(solutions),
          'reachability': 'witnessed' if solutions else 'undetermined',
          'certified_complete': False,
          'solutions': [q.tolist() for q in solutions]}, args.output)
    return 0 if solutions else 1


def marvin_generate(args):
    from gallery.marvin_m6.generate import generate as generate_marvin
    with redirect_stdout(sys.stderr):
        paths = generate_marvin(args.urdf_dir, args.output_dir, args.seed,
                        codegen_only=not args.solve and not args.models_only, models_only=args.models_only,
                        tree_dir=args.tree_dir, with_runtime=args.with_runtime)
    emit({'status': 'ok', 'output_dir': str(args.output_dir.resolve()),
          'mode': 'models' if args.models_only else 'solve' if args.solve else 'codegen',
          'artifacts': paths})
    return 0


def marvin_check(args):
    from gallery.marvin_m6.generate import check
    with redirect_stdout(sys.stderr):
        check(args.urdf_dir, args.output_dir)
    emit({'status': 'ok', 'output_dir': str(args.output_dir.resolve())})
    return 0


def marvin_validate(args):
    from gallery.marvin_m6.validate import main as validate
    argv = ['--samples', str(args.samples), '--seed', str(args.seed),
            '--stress-samples', str(args.stress_samples),
            '--output', str(args.output.resolve()), '--urdf-dir', str(args.urdf_dir.resolve()),
            '--eigen-dir', str(args.eigen_dir.resolve()),
            '--solver-dir', str(args.solver_dir.resolve())]
    if args.python_only:
        argv.append('--python-only')
    with redirect_stdout(sys.stderr):
        status = validate(argv)
    emit({'status': 'ok' if status == 0 else 'failed', 'report': str(args.output.resolve())})
    return status


def generation_arguments(command):
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', type=Path, help='Modified-DH robot model YAML (not an IK tree)')
    source.add_argument('--builtin', choices=sorted(BUILTINS), help='Built-in robot from models')
    source.add_argument('--resume-tree', type=Path, help='Continue branch recovery from a saved *_ik.yaml; keep its main branch')
    command.add_argument('--test-cases', type=Path, help='Optional saved numerical samples')
    command.add_argument('--seed', type=seed_value, default=42, help='Offline sampling seed (default: 42)')
    command.add_argument('--max-nodes', type=positive_integer, default=50,
                         help='Branch expansion node budget; existing nodes are retained (default: 50)')
    command.add_argument('--output-dir', type=Path, default=Path('build'), help='Output directory (default: build)')
    command.add_argument('--language', choices=('both', 'python', 'cpp'), default='both')
    command.add_argument('--with-runtime', action='store_true',
                         help='Copy matching runtime helper files into the output directory')
    command.set_defaults(handler=generate)
    command.add_argument('--no-extra-recovery', action='store_true',
                         help='Skip additional seed refinement and nearby-target recovery; retain legacy checked paths')


def codegen_arguments(command):
    command.add_argument('tree', type=Path, help='Saved *_ik.yaml tree')
    command.add_argument('--output-dir', type=Path, default=Path('build'), help='Output directory (default: build)')
    command.add_argument('--language', choices=('both', 'python', 'cpp'), default='both')
    command.add_argument('--with-runtime', action='store_true',
                         help='Copy matching runtime helper files into the output directory')
    command.add_argument('--require-complete', action='store_true',
                         help='Reject export unless a verified completeness certificate is available')
    command.set_defaults(handler=codegen)
    command.add_argument('--no-extra-recovery', action='store_true',
                         help='Skip additional seed refinement and nearby-target recovery; retain legacy checked paths')


def parser(command=None, language=None):
    result = argparse.ArgumentParser(description=__doc__, epilog='Use COMMAND --help for options; see README.md for examples.')
    result.add_argument('--debug', action='store_true', help='Show a traceback for input or dependency errors')
    if command == 'generate':
        generation_arguments(result)
        return result
    if command == 'codegen':
        codegen_arguments(result)
        if language:
            result.set_defaults(language=language)
        return result
    commands = result.add_subparsers(dest='command')
    command = commands.add_parser('models', help='List built-in robot names')
    command.set_defaults(handler=lambda args: emit({'models': sorted(BUILTINS)}) or 0)

    command = commands.add_parser('generate', help='Run offline symbolic search from a robot model (can take minutes)')
    generation_arguments(command)

    command = commands.add_parser('codegen', help='Export Python/C++ from a saved IK tree, without symbolic search')
    codegen_arguments(command)

    command = commands.add_parser('audit', help='Inspect unresolved branches and completeness proof obligations')
    command.add_argument('tree', type=Path, help='Saved *_ik.yaml tree')
    command.add_argument('--output', type=Path, help='Write the audit JSON to a file')
    command.add_argument('--require-complete', action='store_true',
                         help='Exit 1 unless completeness is certified; default reports without failing')
    command.set_defaults(handler=audit)

    command = commands.add_parser('marvin', help='Marvin M6 V4.0 bounded FK/IK, export, and validation')
    marvin = command.add_subparsers(dest='marvin_command', required=True)
    command = marvin.add_parser('fk', help='Compute arm-base-to-EE pose from seven URDF joint angles')
    command.add_argument('--side', choices=('left', 'right'), required=True)
    command.add_argument('--joints', required=True, metavar='JSON', help='Quoted JSON array of seven angles in radians')
    command.add_argument('--output', type=Path, help='Write JSON to a file (default: stdout)')
    command.set_defaults(handler=marvin_fk)
    command = marvin.add_parser('ik', help='Solve a target with fixed URDF J5; exit 1 if no representative is found')
    command.add_argument('--side', choices=('left', 'right'), required=True)
    command.add_argument('--target', required=True, metavar='FILE', help='JSON 4x4 matrix or FK result; - reads stdin')
    command.add_argument('--q5', type=float, help='Fixed URDF J5 in radians; defaults to q5 from FK JSON')
    command.add_argument('--output', type=Path, help='Write JSON to a file (default: stdout)')
    command.set_defaults(handler=marvin_ik)

    command = marvin.add_parser('generate', help='Fast export from saved trees; use --solve for a new symbolic search')
    command.add_argument('--urdf-dir', type=Path, default=MARVIN / 'urdf')
    command.add_argument('--tree-dir', type=Path, help='Saved tree directory (default: existing output trees or gallery)')
    command.add_argument('--output-dir', type=Path, default=Path('build/marvin_m6'))
    command.add_argument('--with-runtime', action='store_true',
                         help='Bundle Python/C++ runtime helpers for use outside this repository')
    mode = command.add_mutually_exclusive_group()
    mode.add_argument('--solve', action='store_true', help='Run the offline symbolic search (can take minutes)')
    mode.add_argument('--models-only', action='store_true', help='Export modified-DH models without code generation')
    command.add_argument('--seed', type=seed_value, default=42, help='Sampling seed for --solve (default: 42)')
    command.set_defaults(handler=marvin_generate)
    command = marvin.add_parser('check', help='Check models, generated code, and provenance without modifying artifacts')
    command.add_argument('--urdf-dir', type=Path, default=MARVIN / 'urdf')
    command.add_argument('--output-dir', type=Path, default=MARVIN)
    command.set_defaults(handler=marvin_check)
    command = marvin.add_parser('validate', help='Validate gallery Python/C++ against independent URDF FK')
    command.add_argument('--samples', type=int, default=1000, help='Random samples per arm, plus singular/limit cases')
    command.add_argument('--stress-samples', type=int, default=0,
                         help='Additional reachable limit/shoulder/wrist stress cases per arm')
    command.add_argument('--seed', type=seed_value, default=20261103)
    command.add_argument('--output', type=Path, default=Path('build/marvin_m6/validation.json'))
    command.add_argument('--python-only', action='store_true', help='Skip native C++ compilation')
    command.add_argument('--solver-dir', type=Path, default=MARVIN,
                         help='Generated kernels to validate (default: gallery/marvin_m6)')
    command.add_argument('--urdf-dir', type=Path, default=MARVIN / 'urdf')
    command.add_argument('--eigen-dir', type=Path, default=Path('/usr/include/eigen3'))
    command.set_defaults(handler=marvin_validate)
    return result


def main(argv=None, command=None, language=None):
    argument_parser = parser(command, language)
    args = argument_parser.parse_args(argv)
    if not hasattr(args, 'handler'):
        argument_parser.print_help()
        return 0
    try:
        return args.handler(args)
    except ModuleNotFoundError as error:
        if args.debug:
            raise
        print(f'yaik: missing dependency {error.name!r}; install numpy for Marvin FK/IK, '
              'or python -m pip install -r requirements.txt for generation', file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, SyntaxError,
            AssertionError, subprocess.CalledProcessError) as error:
        if args.debug:
            raise
        print(f'yaik: error: {error or "invalid model or IK tree"}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
