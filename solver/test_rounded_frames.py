"""Rounded base/tool matrices still need mutually inverse target conversions."""
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_serialize import read_skeleton_tree
from fk.fk_equations import Pz
from fk.robots import DHEntry, RobotAuxiliaryData, RobotDescription
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod, VariableSolution
from utility.symbolic_utils import inverse_auxiliary_transform, inverse_transform, rotation_x

ROOT = Path(__file__).resolve().parents[1]


def rounded_tree():
    x = sp.Symbol('x')
    model = RobotDescription('rounded_frames')
    model.unknowns = [Unknown(x, UnknownType.Translational.name)]
    model.dh_params = [DHEntry(0, 0, x, 0)]
    value = sp.Float('0.707107')
    pre, post = sp.eye(4), sp.eye(4)
    pre[:3, :3] = sp.Matrix([[value, -value, 0], [value, value, 0], [0, 0, 1]])
    post[:3, :3] = sp.Matrix([[value, value, 0], [-value, value, 0], [0, 0, 1]])
    pre[:3, 3] = sp.Matrix([.2, -.1, .3])
    post[:3, 3] = sp.Matrix([.1, .2, -.05])
    model.auxiliary_data = RobotAuxiliaryData()
    model.auxiliary_data.pre_transform_sp = pre
    model.auxiliary_data.post_transform_sp = post
    solution = VariableSolution.make_explicit_solution(x, [Pz], SolutionMethod.OneVariableAlgebra.name)
    return SkeletonTree([solution], model)


class TestRoundedFrames(TestCase):
    def test_raw_kernel_requires_dh_target_and_returns_dh_joint_values(self):
        from solver.test_parameterized_frames import frame_tree, expected_pose
        tree = frame_tree()
        scope = {'__name__': 'raw_coordinate_contract'}
        exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
        parameters = (.4, -.7, .2, -.3)
        target = expected_pose(.3, *parameters)
        raw_target = scope['parameterized_frames_ik_target_original_to_raw'](target, *parameters)
        raw = scope['parameterized_frames_ik_solve_raw'](raw_target, *parameters)
        self.assertTrue(raw)
        for candidate in raw:
            self.assertAlmostEqual(float(candidate[0, 0]), .3+math.pi/7)
            original = candidate-math.pi/7
            np.testing.assert_allclose(scope['parameterized_frames_fk'](original, *parameters), target,
                                       atol=1e-14, rtol=0)

    def test_exact_rigid_path_and_invalid_frames(self):
        angle = sp.Symbol('angle')
        transform = rotation_x(angle)
        self.assertEqual(inverse_auxiliary_transform(transform), inverse_transform(transform))
        floating_identity = sp.Matrix(np.eye(4))
        self.assertEqual(inverse_auxiliary_transform(floating_identity), inverse_transform(floating_identity))
        for invalid in (sp.diag(1, 1, 0, 1), sp.diag(1, 1, 1, 2)):
            with self.assertRaises(ValueError):
                inverse_auxiliary_transform(invalid)
        scale = sp.Symbol('scale')
        with self.assertRaisesRegex(ValueError, 'nonsingular'):
            inverse_auxiliary_transform(sp.diag(scale, scale, 1, 1), {scale: 0.})

    def test_python_roundtrip_and_legacy_franka_reachable_witnesses(self):
        tree = rounded_tree()
        scope = {'__name__': 'rounded_frames_test'}
        exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
        target = scope['rounded_frames_fk']([.4])
        expected = np.eye(4); expected[2, 3] = .4
        raw = scope['rounded_frames_ik_target_original_to_raw'](target)
        np.testing.assert_allclose(raw, expected, atol=1e-14, rtol=0)
        restored = scope['rounded_frames_ik_target_raw_to_original'](raw)
        np.testing.assert_allclose(restored, target, atol=1e-14, rtol=0)
        self.assertTrue(scope['rounded_frames_ik_solve'](target))
        franka = read_skeleton_tree(str(ROOT/'gallery/franka_panda/franka_panda_ik.yaml'))
        with redirect_stdout(StringIO()):
            exec('\n'.join(CodeGenerationPython(franka).generate_code()), scope)
        for case in json.loads((ROOT/'solver/fixtures/rounded_frame_witnesses.json').read_text()):
            q = np.array(case['q']); target = np.array(case['target'])
            np.testing.assert_allclose(scope['franka_panda_fk'](q), target, atol=1e-14, rtol=0)
            output = scope['franka_panda_ik_solve'](target, q[3])
            self.assertTrue(output, case['index'])
            for candidate in output:
                self.assertEqual(float(candidate[3, 0]), q[3])
                self.assertLess(np.max(np.abs(scope['franka_panda_fk'](candidate)-target)), 1e-4)

    def test_saved_franka_artifacts_match_tree_and_recorded_sources(self):
        import hashlib
        folder = ROOT/'gallery/franka_panda'
        provenance = json.loads((folder/'provenance.json').read_text())
        tree = read_skeleton_tree(str(folder/'franka_panda_ik.yaml'))
        with redirect_stdout(StringIO()):
            for extension, generator in (('py', CodeGenerationPython(tree)),
                                         ('cpp', CodeGenerationCpp(tree, use_safe_operator=True))):
                generated = '\n'.join(line.rstrip() for line in generator.generate_code())+'\n'
                self.assertEqual((folder/('franka_panda_ik_generated.'+extension)).read_text(), generated)
        for name, expected in provenance['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(), expected, name)
        for name, expected in (provenance['generator_overrides_sha256'] | provenance['runtime_sha256']).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_roundtrip_and_legacy_franka_reachable_witnesses(self):
        trees = [rounded_tree(), read_skeleton_tree(str(ROOT/'gallery/franka_panda/franka_panda_ik.yaml'))]
        with redirect_stdout(StringIO()):
            sections = ['\n'.join(CodeGenerationCpp(tree, use_safe_operator=True).generate_code())
                        .split('// Code below for debug')[0] for tree in trees]
        checks = []
        for case in json.loads((ROOT/'solver/fixtures/rounded_frame_witnesses.json').read_text()):
            array = 'std::array<double,7>{{'+','.join(format(float(q), '.17g') for q in case['q'])+'}}'
            checks.append('''{
                const auto q=JOINTS;
                const auto pose=franka_panda_ik::computeFK(q);
                const auto raw=franka_panda_ik::endEffectorTargetOriginalToRaw(pose);
                const auto restored=franka_panda_ik::endEffectorTargetRawToOriginal(raw);
                assert((restored-pose).cwiseAbs().maxCoeff()<1e-13);
                const auto output=franka_panda_ik::computeIK(pose,q[3]);
                bool witnessed=false;
                for(const auto& candidate:output) {
                    assert(candidate[3]==q[3]);
                    if((franka_panda_ik::computeFK(candidate)-pose).cwiseAbs().maxCoeff()<1e-4)witnessed=true;
                }
                EXPECT_WITNESS
            }'''.replace('JOINTS', array).replace('EXPECT_WITNESS',
                'assert(witnessed);' if case['cpp_expected_witness'] else '// IK coverage remains unresolved.'))
        main = '''#include <cassert>
int main() {
    const auto target=rounded_frames_ik::computeFK({{.4}});
    Eigen::Matrix4d expected=Eigen::Matrix4d::Identity();expected(2,3)=.4;
    const auto raw=rounded_frames_ik::endEffectorTargetOriginalToRaw(target);
    assert((raw-expected).cwiseAbs().maxCoeff()<1e-14);
    assert((rounded_frames_ik::endEffectorTargetRawToOriginal(raw)-target).cwiseAbs().maxCoeff()<1e-14);
    assert(!rounded_frames_ik::computeIK(target).empty());
    WITNESSES
}
'''.replace('WITNESSES', '\n'.join(checks))
        with tempfile.TemporaryDirectory(prefix='yaik-rounded-frames-') as temporary:
            temporary = Path(temporary)
            (temporary/'test.cpp').write_text('\n'.join(sections)+main)
            compiled = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT), '-I', '/usr/include/eigen3',
                                       str(temporary/'test.cpp'), '-o', str(temporary/'test')], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(temporary/'test')], check=True)
