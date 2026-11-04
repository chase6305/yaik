"""Generate the Marvin M6 V4.0 arm models and analytical IK gallery artifacts.

Run from any directory: python gallery/marvin_m6/generate.py [--codegen-only].
URDF rotations are Rz(yaw) Ry(pitch) Rx(roll), unlike the mmind RPY helper.
"""

import argparse
from importlib.metadata import version
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import sympy as sp
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codegen.cpp_codegen.codegen_cpp import generate_code_cpp
from codegen.py_codegen.codegen_py import generate_code_python
from codegen.skeleton.tree_serialize import read_skeleton_tree, save_ik_tree
from convert_kinematic_chain import convert_chain_into_robot, process_special_angle
from fk.chain_transform import ChainTransform, ChainTransformElement
from ik_solve import RunIKOption, run_ik
from solver.equation_types import Unknown
from utility.rotation_utils import rotation_x, rotation_y, rotation_z

HERE = Path(__file__).resolve().parent


def urdf_model(path, side):
    """Read a standalone 7-revolute-joint arm, retaining its base and EE frames."""
    document = ET.parse(path).getroot()
    joints = document.findall('joint')
    child_to_joint = {joint.find('child').get('link'): joint for joint in joints}
    ordered = []
    tip = side + '_ee'
    while tip in child_to_joint:
        joint = child_to_joint[tip]
        ordered.append(joint)
        tip = joint.find('parent').get('link')
    ordered.reverse()
    if tip != side + '_arm_base' or len(ordered) != 8:
        raise ValueError('Expected the standalone Marvin 7-DoF arm and fixed EE joint')

    chain = ChainTransform()
    limits = []
    for joint in ordered:
        origin = joint.find('origin')
        roll, pitch, yaw = map(float, origin.get('rpy', '0 0 0').split())
        placement = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
        placement[:3, 3] = list(map(float, origin.get('xyz', '0 0 0').split()))
        if joint.get('type') == 'fixed':
            chain.set_post_transform(chain.post_transform @ placement)
        elif joint.get('type') == 'revolute':
            axis = list(map(float, joint.find('axis').get('xyz').split()))
            chain.add_chain_element(ChainTransformElement(placement, axis))
            limit = joint.find('limit')
            limits.append((float(limit.get('lower')), float(limit.get('upper'))))
        else:
            raise ValueError('Only revolute joints and the fixed EE joint are supported')
    if len(limits) != 7:
        raise ValueError('Expected exactly seven revolute joints')

    # The chain converter numerically identifies the modified DH frames. Clean
    # numerical roundoff in cardinal rotations before symbolic code generation.
    robot = convert_chain_into_robot('marvin_m6_' + side + '_arm', chain)
    if robot is None:
        raise ValueError('Could not convert the URDF chain to modified DH')
    for transform in (robot.auxiliary_data.pre_transform_sp,
                      robot.auxiliary_data.post_transform_sp):
        for row in range(4):
            for col in range(4):
                value = transform[row, col]
                if not value.free_symbols:
                    nearest = round(float(value))
                    if abs(float(value) - nearest) < 1e-12:
                        transform[row, col] = sp.Integer(nearest)
    robot.parameters_value = {symbol: round(float(value), 12)
                              for symbol, value in robot.parameters_value.items()}
    offsets = []
    for offset in robot.auxiliary_data.unknown_offset:
        exact = process_special_angle(offset)
        offsets.append(float(exact) if exact is not None else float(offset))
    robot.auxiliary_data.unknown_offset = offsets
    # Bounds are expressed in the raw DH angle, q_original + offset. The
    # generated API still accepts/returns the original URDF joint angles.
    robot.unknowns = [Unknown(unknown.symbol,
                             lower_bound=lower + offset,
                             upper_bound=upper + offset)
                      for unknown, (lower, upper), offset in
                      zip(robot.unknowns, limits, offsets)]
    robot.unknown_as_parameter_more_dof = [robot.unknowns[4].symbol]
    return robot


def model_dict(robot):
    result = robot.to_dict()
    result['symbolic_parameters'].sort()
    return result


def generate(urdf_dir=HERE / "urdf", output_dir=HERE, seed=42, codegen_only=False, models_only=False):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    provenance = {
        'source': 'HumanoidAssets/Marvin_M6_S_CCS_696_V4.0',
        'source_repository': 'https://github.com/chase6305/HumanoidAssets',
        'source_revision': '0ead034d8b8d0e5dae10fb282db47ac7174edc39',
        'generator_base_revision': '821ba579680e8232b7d71efe87abd373f567f9f8',
        'dependencies': {package: version(package) for package in
                         ('numpy', 'sympy', 'attrs', 'PyYAML')},
        'seed': seed, 'fixed_joint': 'th_4 (URDF J5 wrist_pitch)',
        'urdf_sha256': {},
    }
    for side in ('left', 'right'):
        source = Path(urdf_dir) / (side + '_arm.urdf')
        provenance['urdf_sha256'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        robot = urdf_model(source, side)
        models[side] = robot
        with (output_dir / (robot.name + '.yaml')).open('w') as stream:
            yaml.safe_dump(model_dict(robot), stream, sort_keys=True)
    if models_only:
        return

    if not codegen_only:
        left, right = models['left'], models['right']
        dh_symbols = set().union(*(sp.sympify(value).free_symbols
                                   for dh in left.dh_params for value in dh.flatten()))
        if (left.dh_params != right.dh_params
                or [item.to_dict() for item in left.unknowns] !=
                   [item.to_dict() for item in right.unknowns]
                or left.get_unknown_offset() != right.get_unknown_offset()
                or any(left.parameters_value.get(symbol) != right.parameters_value.get(symbol)
                       for symbol in dh_symbols)):
            raise ValueError('Tree sharing requires identical internal arm geometry and raw bounds')
        # Both arms have the same internal DH chain. Solve it once; serialize
        # each arm with its own exact base transform and original joint bounds.
        with tempfile.TemporaryDirectory(prefix='marvin-m6-generation-') as temporary:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                run_ik(models['left'], run_ik_option=RunIKOption(random_seed=seed))
                tree = read_skeleton_tree(models['left'].name + '_ik.yaml')
            finally:
                os.chdir(previous_cwd)
            for side, robot in models.items():
                data = copy.deepcopy(tree)
                data._robot = robot
                save_ik_tree(data, str(output_dir / (robot.name + '_ik.yaml')))
    for robot in models.values():
        tree = read_skeleton_tree(str(output_dir / (robot.name + '_ik.yaml')))
        if model_dict(tree.robot) != model_dict(robot):
            raise ValueError('IK tree model does not match the supplied URDF: ' + robot.name)
        generate_code_python(tree, str(output_dir / (robot.name + '_ik_generated.py')))
        generate_code_cpp(tree, str(output_dir / (robot.name + '_ik_generated.cpp')),
                          use_safe_operator=True)
        # Remove padding from generated blank lines before recording hashes.
        for extension in ('py', 'cpp'):
            generated = output_dir / f'{robot.name}_ik_generated.{extension}'
            generated.write_text('\n'.join(line.rstrip() for line in
                                           generated.read_text().splitlines()) + '\n')
    provenance['artifact_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.glob('marvin_m6_*'))
        if path.is_file() and path.suffix in ('.yaml', '.py', '.cpp')
    }
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def check(urdf_dir=HERE / 'urdf', output_dir=HERE, seed=42):
    """Verify committed models and generated code without changing the gallery."""
    output_dir = Path(output_dir).resolve()
    with tempfile.TemporaryDirectory(prefix='marvin-m6-check-') as temporary:
        temporary = Path(temporary)
        for side in ('left', 'right'):
            filename = 'marvin_m6_' + side + '_arm_ik.yaml'
            shutil.copyfile(output_dir / filename, temporary / filename)
        generate(urdf_dir, temporary, seed, codegen_only=True)
        for generated in sorted(temporary.iterdir()):
            committed = output_dir / generated.name
            if not committed.is_file():
                raise SystemExit('Missing artifact: ' + str(committed))
            if generated.name == 'provenance.json':
                expected = json.loads(generated.read_text())
                recorded = json.loads(committed.read_text())
                # Dependency versions describe the original generation environment.
                # Reproduction in a different environment is accepted if the
                # actual generated bytes, models, and source hashes still agree.
                expected['dependencies'] = recorded.get('dependencies')
                stale = expected != recorded
            else:
                stale = generated.read_bytes() != committed.read_bytes()
            if stale:
                raise SystemExit('Stale artifact: ' + str(committed))
    print('Marvin M6 models, generated solvers, and provenance are up to date.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf-dir', type=Path, default=HERE / 'urdf')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--codegen-only', action='store_true',
                        help='Regenerate Python/C++ from the checked-in IK trees')
    parser.add_argument('--models-only', action='store_true')
    parser.add_argument('--check', action='store_true',
                        help='Fail if models, generated code, or provenance are stale; write nothing')
    args = parser.parse_args()
    if args.check:
        if args.models_only or args.codegen_only:
            parser.error('--check cannot be combined with --models-only or --codegen-only')
        check(args.urdf_dir, args.output_dir, args.seed)
    else:
        generate(args.urdf_dir, args.output_dir, args.seed, args.codegen_only, args.models_only)


if __name__ == '__main__':
    main()
