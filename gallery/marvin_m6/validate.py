"""Check both Marvin IK APIs against the original URDF, in Python and C++.

Targets and candidate residuals use XML origins/axes directly, without DH
conversion or generated FK. Both generated kernels and the bounded analytical
integration must solve every reachable regression case.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def origin_transform(joint):
    """URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    origin = joint.find('origin')
    if origin is None:
        return np.eye(4)
    roll, pitch, yaw = map(float, origin.get('rpy', '0 0 0').split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    result[:3, 3] = list(map(float, origin.get('xyz', '0 0 0').split()))
    return result


class UrdfArm:
    """Independent numerical FK for the exact standalone XML arm chain."""

    def __init__(self, side, urdf_dir=HERE / 'urdf'):
        self.side = side
        self.path = Path(urdf_dir) / (side + '_arm.urdf')
        document = ET.parse(self.path).getroot()
        by_child = {j.find('child').get('link'): j for j in document.findall('joint')}
        ordered, link = [], side + '_ee'
        while link in by_child:
            joint = by_child[link]
            ordered.append(joint)
            link = joint.find('parent').get('link')
        ordered.reverse()
        if link != side + '_arm_base':
            raise ValueError('Unexpected arm root')
        movable = [j for j in ordered if j.get('type') != 'fixed']
        if len(movable) != 7 or any(j.get('type') != 'revolute' for j in movable):
            raise ValueError('Expected seven revolute joints')
        self.names = [j.get('name') for j in movable]
        self.lower = np.array([float(j.find('limit').get('lower')) for j in movable])
        self.upper = np.array([float(j.find('limit').get('upper')) for j in movable])
        self.chain = []
        for joint in ordered:
            axis = None
            if joint.get('type') != 'fixed':
                axis = np.array(list(map(float, joint.find('axis').get('xyz').split())))
                axis /= np.linalg.norm(axis)
            self.chain.append((origin_transform(joint), axis))

    def fk(self, q):
        result, index = np.eye(4), 0
        for origin, axis in self.chain:
            result = result @ origin
            if axis is None:
                continue
            x, y, z = axis
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            motion = np.eye(4)
            motion[:3, :3] += math.sin(q[index])*cross + (1-math.cos(q[index]))*(cross @ cross)
            result = result @ motion
            index += 1
        return result

    def normalize(self, q):
        """Find the equivalent bounded representation, without clipping pose errors."""
        result = np.asarray(q, dtype=float).reshape(7).copy()
        if not np.isfinite(result).all():
            return None
        for i, value in enumerate(result):
            lo = math.ceil((self.lower[i] - 1e-9 - value)/(2*math.pi))
            hi = math.floor((self.upper[i] + 1e-9 - value)/(2*math.pi))
            if lo > hi:
                return None
            result[i] += min(range(lo, hi+1), key=lambda k: abs(value+2*math.pi*k))*2*math.pi
        return result


def pose_error(actual, target):
    relative = actual[:3, :3] @ target[:3, :3].T
    sine = 0.5*np.linalg.norm([relative[2, 1]-relative[1, 2],
                              relative[0, 2]-relative[2, 0],
                              relative[1, 0]-relative[0, 1]])
    cosine = np.clip((np.trace(relative)-1)*0.5, -1, 1)
    return float(np.linalg.norm(actual[:3, 3]-target[:3, 3])), float(np.arctan2(sine, cosine))


def stress_cases_for(arm, count, seed):
    """Known reachable witnesses biased toward singularities and limit corners."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 696, 20261104]))
    result = []
    for index in range(count):
        q = rng.uniform(arm.lower, arm.upper)
        kind = index % 7
        if kind == 0:
            mask = rng.random(7) < .75
            epsilon = rng.choice([0., 1e-12, 1e-9, 1e-6, 1e-3])
            edges = np.where(rng.random(7) < .5, arm.lower+epsilon, arm.upper-epsilon)
            q[mask] = edges[mask]
            name = 'joint_limits'
        elif kind == 1:
            q[1] = rng.choice([-1., 1.])*10.**rng.uniform(-14., -2.)
            for joint in (0, 2, 3, 5, 6):
                if rng.random() < .5:
                    q[joint] = rng.choice([arm.lower[joint], arm.upper[joint]])
            name = 'shoulder'
        elif kind in (2, 3, 5):
            for _ in range(100):
                elbow = rng.uniform(arm.lower[3], arm.upper[3])
                t3 = elbow+math.pi
                h = .018*math.cos(t3)+.018+.287*math.sin(t3)
                k = .018*math.sin(t3)-.287*math.cos(t3)+.314
                exact = kind == 2 or (kind == 5 and index % 3 == 0)
                q5 = 0. if exact else rng.choice([-1., 1.])*10.**rng.uniform(-14., -3.)
                q6 = math.atan2(h*math.cos(q5), k)-math.pi/2
                if arm.lower[5] <= q6 <= arm.upper[5]:
                    break
            else:
                raise ValueError('Could not generate an in-limit wrist-axis witness')
            q[3:6] = elbow, q5, q6
            for joint in range(3):
                if rng.random() < .8:
                    epsilon = rng.choice([0., 1e-12, 1e-9, 1e-6, 1e-3])
                    q[joint] = (arm.lower[joint]+epsilon if rng.random() < .5 else arm.upper[joint]-epsilon)
            name = 'wrist_axis' if kind == 2 else 'wrist_near_axis'
            if kind == 5:
                q[1] = 0. if index % 4 == 0 else rng.choice([-1., 1.])*10.**rng.uniform(-14., -2.)
                name = 'combined_shoulder_wrist'
        elif kind == 4:
            # Stationary elbow distance: the two analytical elbow roots meet.
            center = math.atan2(2*.018*(.314+.287), 2*.018**2-2*.287*.314)-math.pi
            q[3] = center+(0. if index % 4 == 0 else rng.choice([-1., 1.])*10.**rng.uniform(-15., -2.))
            for joint in (0, 1, 2, 5, 6):
                if rng.random() < .7:
                    q[joint] = rng.choice([arm.lower[joint], 0., arm.upper[joint]])
            name = 'elbow_repeated_root'
        else:
            q[4] = rng.choice([-1., 1.])*math.pi/2
            if index % 3:
                q[4] += rng.choice([-1., 1.])*10.**rng.uniform(-15., -2.)
            if index % 2 == 0:
                q[3] = rng.choice([-1., 1.])*10.**rng.uniform(-15., -2.)
            for joint in (0, 1, 2, 5, 6):
                if rng.random() < .7:
                    q[joint] = rng.choice([arm.lower[joint], 0., arm.upper[joint]])
            name = 'fixed_j5_half_pi'
        result.append((f'stress_{name}/{index}', q))
    return result


def cases_for(arm, count, seed, stress_samples=0):
    rng = np.random.default_rng(seed)
    cases = [('random/' + str(i), q) for i, q in enumerate(rng.uniform(arm.lower, arm.upper, (count, 7)))]
    midpoint = (arm.lower+arm.upper)*0.5
    cases.extend([('special/zero', np.zeros(7)), ('special/midpoint', midpoint.copy()),
                  ('special/all_lower', arm.lower.copy()), ('special/all_upper', arm.upper.copy())])
    for i in range(7):
        values = [('lower', arm.lower[i]), ('upper', arm.upper[i]),
                  ('zero', 0.0), ('near_zero_positive', 1e-8), ('near_zero_negative', -1e-8)]
        for name, value in values:
            q = midpoint.copy()
            q[i] = value
            cases.append((f'special/J{i+1}_{name}', q))
    for value in (-math.pi/2, math.pi/2):
        q = midpoint.copy()
        q[4] = value
        cases.append((f'special/q5_{value:+g}', q))
    # Reachable singular targets must retain a legal representative of their
    # continuous families. Randomize other joints to avoid testing only zero.
    for joint in (1, 3, 4, 5):
        for value in (0.0, -1e-10, 1e-10):
            for index in range(8):
                q = rng.uniform(arm.lower, arm.upper)
                q[joint] = value
                cases.append((f'near_singular/J{joint+1}_{value:+g}/{index}', q))
    # Construct inverse-wrist-axis continuous families deliberately: q5=0,
    # t5=atan2(h,k) makes the inverse target's XY translation vanish. J7 is
    # then free, so the integration must pick a bounded family representative.
    for elbow in np.linspace(arm.lower[3], arm.upper[3], 9):
        t3 = elbow + math.pi
        h = 0.018*math.cos(t3) + 0.018 + 0.287*math.sin(t3)
        k = 0.018*math.sin(t3) - 0.287*math.cos(t3) + 0.314
        q6 = math.atan2(h, k)-math.pi/2
        if not arm.lower[5] <= q6 <= arm.upper[5]:
            continue
        for index in range(8):
            q = rng.uniform(arm.lower, arm.upper)
            q[3:6] = elbow, 0, q6
            if index < 4:
                q[index % 3] = arm.lower[index % 3]+1e-8 if index % 2 else arm.upper[index % 3]-1e-8
                q[6] = arm.lower[6]+1e-8 if index % 2 else arm.upper[6]-1e-8
            cases.append((f'wrist_axis/elbow_{elbow:+g}/{index}', q))
    # Couple near-axis degeneracy to exact J7 bounds; a small roundoff error in
    # the inverse XY direction otherwise discards a known reachable endpoint.
    for elbow in (-2.4, -2.12, -1.9, -1.7):
        t3 = elbow + math.pi
        h = 0.018*math.cos(t3) + 0.018 + 0.287*math.sin(t3)
        k = 0.018*math.sin(t3) - 0.287*math.cos(t3) + 0.314
        for q5 in (-1e-6, 1e-6, -1e-8, 1e-8, -1e-10, 1e-10, -1e-12, 1e-12):
            q6 = math.atan2(h*math.cos(q5), k)-math.pi/2
            for q7 in (arm.lower[6], arm.upper[6]):
                q = rng.uniform(arm.lower, arm.upper)
                q[3:] = elbow, q5, q6, q7
                cases.append((f'wrist_near_axis/{elbow:+g}/{q5:+g}/{q7:+g}', q))
    recorded = json.loads(Path(__file__).with_name('regression_cases.json').read_text())
    for case in recorded['cases']:
        cases.append((case['label'], np.asarray(case['q'], dtype=float)))
    cases.extend(stress_cases_for(arm, stress_samples, seed))
    return cases


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PythonSolver:
    def __init__(self, side, bounded=False):
        self.bounded = bounded
        if bounded:
            module = load_module(HERE / 'solver.py', 'marvin_m6_analytical_integration')
            self.instance = module.Solver(side)
            self._fk, self._ik = self.instance.fk, self.instance.ik
        else:
            name = 'marvin_m6_' + side + '_arm'
            module = load_module(HERE / (name+'_ik_generated.py'), name+'_generated_validation')
            self._fk = getattr(module, name+'_fk')
            self._ik = getattr(module, name+'_ik_solve')

    def fk(self, q):
        return self._fk(np.asarray(q, dtype=float))

    def ik(self, target, q5):
        return [np.asarray(q).reshape(7) for q in self._ik(target, float(q5))]


class CppSolver:
    def __init__(self, side, build_dir, bounded=False, eigen_dir='/usr/include/eigen3'):
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.bounded = bounded
        self.side_index = 0 if side == 'left' else 1
        name = 'marvin_m6_' + side + '_arm'
        tag = 'bounded' if bounded else side + '_generated'
        includes = '#include "solver.hpp"' if bounded else ('#define main marvin_generated_demo_main\n#include "'+name+'_ik_generated.cpp"\n#undef main')
        side_arg = 'static_cast<marvin_m6::Side>(side)'
        side_parameter = 'int side, ' if bounded else ''
        ik_call = f'marvin_m6::solveIK(pose, q5, {side_arg})' if bounded else f'{name}_ik::computeIK(pose, q5)'
        fk_call = f'marvin_m6::computeFK(angles, {side_arg})' if bounded else f'{name}_ik::computeFK(angles)'
        wrapper = f'''{includes}
#include <stdexcept>
extern "C" void fk({side_parameter}const double* q, double* output) {{
    std::array<double, 7> angles;
    std::copy(q, q+7, angles.begin());
    auto pose = {fk_call};
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) output[4*i+j]=pose(i,j);
}}
extern "C" int ik({side_parameter}const double* target, double q5, double* output, int capacity) {{
    try {{
    Eigen::Matrix4d pose;
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) pose(i,j)=target[4*i+j];
    auto solutions = {ik_call};
    if (static_cast<int>(solutions.size())>capacity) return -static_cast<int>(solutions.size());
    for (size_t i=0; i<solutions.size(); ++i) for (int j=0; j<7; ++j) output[7*i+j]=solutions[i][j];
    return static_cast<int>(solutions.size());
    }} catch (const std::invalid_argument&) {{ return 0; }}
}}
'''
        # Both bounded arms use one header with a runtime side argument. Reuse
        # that compilation; key it by the source so stale builds are not reused.
        dependency = HERE / ('solver.hpp' if bounded else name + '_ik_generated.cpp')
        dependency_bytes = dependency.read_bytes()
        common_bytes = b'' if bounded else (ROOT / 'yaik_cpp_common.h').read_bytes()
        digest = hashlib.sha256(wrapper.encode() + dependency_bytes + common_bytes +
                                str(Path(eigen_dir).resolve()).encode()).hexdigest()[:16]
        source = Path(build_dir) / f'{tag}_{digest}.cpp'
        library = source.with_suffix('.so')
        if not library.is_file():
            source.write_text(wrapper)
            # Compile immutable input snapshots, then publish atomically. Two
            # validators sharing a cache must never load a partly written .so.
            with tempfile.TemporaryDirectory(prefix=f'.{tag}-', dir=build_dir) as staging:
                staging = Path(staging)
                staged_source = staging / 'wrapper.cpp'
                staged_source.write_text(wrapper)
                (staging / dependency.name).write_bytes(dependency_bytes)
                if not bounded:
                    (staging / 'yaik_cpp_common.h').write_bytes(common_bytes)
                staged_library = staging / 'solver.so'
                subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-I', str(staging),
                                '-I', str(ROOT), '-I', str(HERE), '-I', str(eigen_dir),
                                str(staged_source), '-o', str(staged_library)], check=True)
                staged_library.replace(library)
        self.lib = ctypes.CDLL(str(library))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        side_types = [ctypes.c_int] if bounded else []
        self.lib.fk.argtypes, self.lib.fk.restype = side_types + [array, array], None
        self.lib.ik.argtypes = side_types + [array, ctypes.c_double, array, ctypes.c_int]
        self.lib.ik.restype = ctypes.c_int

    def fk(self, q):
        output = np.empty((4, 4))
        side = (self.side_index,) if self.bounded else ()
        self.lib.fk(*side, np.ascontiguousarray(q, dtype=np.float64).reshape(7), output)
        return output

    def ik(self, target, q5):
        output = np.empty((256, 7))
        side = (self.side_index,) if self.bounded else ()
        count = self.lib.ik(*side, np.ascontiguousarray(target, dtype=np.float64), float(q5), output, len(output))
        if count < 0:
            output = np.empty((-count, 7))
            count = self.lib.ik(*side, np.ascontiguousarray(target, dtype=np.float64), float(q5), output, len(output))
            if count < 0:
                raise RuntimeError('C++ IK output changed between identical calls')
        return output[:count].copy()


def run_solver(solver, arm, cases, position_tolerance=1e-7, rotation_tolerance=1e-6):
    metrics, groups, failures = Counter(), {}, []
    maximum_fk = maximum_position = maximum_rotation = 0.0
    start = time.perf_counter()
    for label, q in cases:
        target = arm.fk(q)
        maximum_fk = max(maximum_fk, float(np.max(np.abs(solver.fk(q)-target))))
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            raw = solver.ik(target, q[4])
        metrics['warnings'] += len(emitted)
        metrics['returned'] += len(raw)
        accepted, unique, invalid = [], [], 0
        for candidate in raw:
            if not np.isfinite(candidate).all():
                metrics['nonfinite'] += 1
                invalid += 1
                continue
            p, r = pose_error(arm.fk(candidate), target)
            maximum_position, maximum_rotation = max(maximum_position, p), max(maximum_rotation, r)
            if p > position_tolerance or r > rotation_tolerance or abs(candidate[4]-q[4]) > 1e-8:
                metrics['bad_pose_or_free_joint'] += 1
                invalid += 1
                continue
            normalized = arm.normalize(candidate)
            if normalized is None:
                metrics['outside_limits'] += 1
                invalid += 1
                continue
            if solver.bounded and np.max(np.abs(normalized-candidate)) > 1e-8:
                metrics['unnormalized'] += 1
                invalid += 1
            accepted.append(normalized)
            if any(np.max(np.abs(np.remainder(normalized-other+math.pi, 2*math.pi)-math.pi)) < 1e-7 for other in unique):
                metrics['duplicates'] += 1
            else:
                unique.append(normalized)
        group = groups.setdefault(label.split('/')[0], Counter())
        group['samples'] += 1
        group['with_valid_solution'] += bool(accepted)
        # Generated kernels expose unbounded branches, but each reachable
        # target must still retain at least one strict, bounded solution.
        if not accepted or (solver.bounded and invalid):
            failures.append({'case': label, 'q': q.tolist(), 'target': target.tolist(),
                             'returned': len(raw), 'valid': len(accepted), 'invalid': invalid})
    return {'groups': {name: dict(values) for name, values in groups.items()},
            'candidates': dict(metrics), 'max_fk_matrix_error': maximum_fk,
            'max_returned_position_error_m': maximum_position,
            'max_returned_rotation_error_rad': maximum_rotation,
            'failures': failures, 'elapsed_seconds': time.perf_counter()-start,
            'passed': not failures and maximum_fk < 1e-11 and
                      (not solver.bounded or metrics['duplicates'] == metrics['warnings'] == 0)}


def negative_checks(solver, arm):
    target = arm.fk((arm.lower+arm.upper)*0.5)
    cases = []
    unreachable = target.copy()
    unreachable[:3, 3] = [10, 10, 10]
    cases.append(('unreachable', unreachable, 0.0))
    # The first three arm axes meet at the shoulder, and the final URDF entry
    # is the fixed tool transform. Put the wrist center exactly at that common
    # point: the model's positive inner reach radius makes this pose unreachable.
    collapsed = np.eye(4)
    tool = arm.chain[-1][0]
    collapsed[:3, 3] = arm.chain[0][0][:3, 3]+tool[:3, :3].T@tool[:3, 3]
    cases.append(('inside_empty_annulus', collapsed, 0.0))
    for invalid in (np.nan, np.inf, -np.inf):
        for row, col in ((0, 0), (0, 3), (3, 0)):
            bad = target.copy()
            bad[row, col] = invalid
            cases.append((f'nonfinite_target_{row}_{col}_{invalid}', bad, 0.0))
        cases.append((f'nonfinite_q5_{invalid}', target, invalid))
    if solver.bounded:
        cases.extend([('q5_lower_outside', target, arm.lower[4]-0.1),
                      ('q5_upper_outside', target, arm.upper[4]+0.1)])
        bad_rotation, bad_bottom, reflection = target.copy(), target.copy(), target.copy()
        bad_rotation[:3, 0] *= 2
        bad_bottom[3, 3] = 2
        reflection[:3, 0] *= -1
        cases.extend([('nonrigid_rotation', bad_rotation, 0.0),
                      ('nonhomogeneous', bad_bottom, 0.0), ('improper_rotation', reflection, 0.0)])
        huge_position, huge_rotation = target.copy(), target.copy()
        huge_position[:3, 3] = np.finfo(float).max
        huge_rotation[0, 0] = np.finfo(float).max
        cases.extend([('max_double_position', huge_position, 0.0),
                      ('max_double_rotation', huge_rotation, 0.0),
                      ('max_double_q5', target, np.finfo(float).max)])
    failures = []
    warning_count = 0
    for label, pose, q5 in cases:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            try:
                output = solver.ik(pose, q5)
            except ValueError:
                output = []
        warning_count += len(emitted)
        if len(output):
            failures.append(label)
    return {'samples': len(cases), 'warnings': warning_count, 'failures': failures,
            'passed': not failures and (not solver.bounded or warning_count == 0)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--stress-samples', type=int, default=0,
                        help='Additional reachable limit/shoulder/wrist stress cases per arm')
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--output', type=Path, default=HERE / 'validation_report.json')
    parser.add_argument('--python-only', action='store_true')
    parser.add_argument('--urdf-dir', type=Path, default=HERE / 'urdf')
    parser.add_argument('--eigen-dir', type=Path, default=Path('/usr/include/eigen3'))
    args = parser.parse_args(argv)
    if args.samples < 0 or args.stress_samples < 0:
        parser.error('--samples and --stress-samples must be nonnegative')
    if not args.python_only and shutil.which('g++') is None:
        parser.error('C++ validation requires g++ (or select --python-only)')
    if not args.python_only and not (args.eigen_dir / 'Eigen/Core').is_file():
        parser.error('Eigen3 headers were not found; set --eigen-dir or use --python-only')
    sources = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.hpp')) + sorted(HERE.glob('*_ik_generated.cpp'))
    sources += sorted(HERE.glob('*.yaml')) + [HERE / 'provenance.json']
    sources += [Path(__file__).with_name('regression_cases.json')]
    sources += sorted(Path(args.urdf_dir).glob('*_arm.urdf'))
    sources += [ROOT / name for name in ('python_run_import.py', 'yaik_cpp_common.h',
                                       'yaik.py', 'requirements.txt')]
    report = {'seed': args.seed, 'random_samples_per_arm': args.samples,
              'stress_samples_per_arm': args.stress_samples,
              'position_tolerance_m': 1e-7, 'rotation_tolerance_rad': 1e-6,
              'fk_matrix_tolerance': 1e-11, 'free_joint_tolerance_rad': 1e-8,
              'python_version': platform.python_version(), 'numpy_version': np.__version__,
              'sources': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'method': 'Independent XML FK generates reachable targets and verifies every candidate; fixed q5 uses known generating q[4]. Raw generator and bounded analytical integration are reported separately.',
              'results': {}}
    with tempfile.TemporaryDirectory(prefix='marvin-m6-validation-') as temporary:
        for side in ('left', 'right'):
            arm = UrdfArm(side, args.urdf_dir)
            cases = cases_for(arm, args.samples, args.seed, args.stress_samples)
            for language in ('python',) if args.python_only else ('python', 'cpp'):
                for bounded in (False, True):
                    key = f'{language}/{side}/' + ('bounded_analytical' if bounded else 'generated')
                    print('Checking ' + key, flush=True)
                    solver = PythonSolver(side, bounded) if language == 'python' else CppSolver(side, temporary, bounded, args.eigen_dir)
                    result = run_solver(solver, arm, cases)
                    result['negative_checks'] = negative_checks(solver, arm)
                    result['passed'] &= result['negative_checks']['passed']
                    report['results'][key] = result
                    print(f"  {result['groups']}; failures={len(result['failures'])}; FK={result['max_fk_matrix_error']:.3g}", flush=True)
    report['passed'] = all(result['passed'] for result in report['results'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f"Generated and bounded analytical validation: {'PASS' if report['passed'] else 'FAIL'}; {args.output}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
