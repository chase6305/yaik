"""Base/tool parameters must reach FK, coordinate conversion, and checked IK."""

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
from fk.fk_equations import Pz
from fk.robots import DHEntry, RobotAuxiliaryData, RobotDescription
from ik_solve import check_robot
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod, VariableSolution


ROOT = Path(__file__).resolve().parents[1]


def frame_tree():
    x, mount, tool, length, shift, height = sp.symbols(
        'x a_mount b_tool c_length d_shift height')
    robot = RobotDescription('parameterized_frames')
    robot.unknowns = [Unknown(x, UnknownType.Translational.name, -1., 2.)]
    robot.dh_params = [DHEntry(0, 0, x, 0)]
    robot.symbolic_parameters = {mount, tool, length, shift, height}
    robot.parameters_bound = {symbol: (-1., 1.) for symbol in (mount, tool, length, shift)}
    robot.parameters_value = {height: sp.sqrt(2)/3}
    pre = sp.eye(4)
    pre[:3, :3] = sp.Matrix([[sp.cos(mount), 0, sp.sin(mount)],
                             [0, 1, 0], [-sp.sin(mount), 0, sp.cos(mount)]])
    pre[:3, 3] = sp.Matrix([shift, 0, height])
    post = sp.eye(4)
    post[:3, :3] = sp.Matrix([[sp.cos(tool), -sp.sin(tool), 0],
                              [sp.sin(tool), sp.cos(tool), 0], [0, 0, 1]])
    post[:3, 3] = sp.Matrix([length, sp.Rational(1, 8), 0])
    robot.auxiliary_data = RobotAuxiliaryData(unknown_offset=[sp.pi/7])
    robot.auxiliary_data.pre_transform_sp = pre
    robot.auxiliary_data.post_transform_sp = post
    solution = VariableSolution.make_explicit_solution(
        x, [Pz], SolutionMethod.OneVariableAlgebra.name)
    return SkeletonTree([solution], robot)


def expected_pose(q, mount, tool, length, shift):
    pre, raw, post = np.eye(4), np.eye(4), np.eye(4)
    c, s = math.cos(mount), math.sin(mount)
    pre[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    pre[:3, 3] = [shift, 0, math.sqrt(2)/3]
    raw[2, 3] = q+math.pi/7
    c, s = math.cos(tool), math.sin(tool)
    post[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    post[:3, 3] = [length, 1/8, 0]
    return pre@raw@post


class TestParameterizedFrames(TestCase):
    def test_python_runtime_rotations_translations_and_exact_constants(self):
        tree = frame_tree()
        self.assertTrue(check_robot(tree.robot))
        scope = {'__name__': 'parameterized_frames_test'}
        exec('\n'.join(CodeGenerationPython(tree).generate_code()), scope)
        for parameters in ((0., 0., 0., 0.), (.4, -.7, .2, -.3), (-.9, .6, -.5, .8)):
            with self.subTest(parameters=parameters):
                target = expected_pose(.3, *parameters)
                np.testing.assert_allclose(scope['parameterized_frames_fk']([.3], *parameters),
                                           target, atol=1e-14, rtol=0)
                raw = scope['parameterized_frames_ik_target_original_to_raw'](target, *parameters)
                np.testing.assert_allclose(raw[:3, :3], np.eye(3), atol=1e-14, rtol=0)
                self.assertAlmostEqual(raw[2, 3], .3+math.pi/7)
                restored = scope['parameterized_frames_ik_target_raw_to_original'](raw, *parameters)
                np.testing.assert_allclose(restored, target, atol=1e-14, rtol=0)
                epsilon = 1e-6
                delta = (expected_pose(.3+epsilon, *parameters)-expected_pose(.3-epsilon, *parameters))/(2*epsilon)
                jacobian = scope['parameterized_frames_transform_point_jacobian'](
                    [.3], *parameters, target[:3, 3])
                np.testing.assert_allclose(jacobian[:, 0], delta[:3, 3], atol=1e-10, rtol=0)
                output = scope['parameterized_frames_ik_solve'](target, *parameters)
                self.assertTrue(output)
                for candidate in output:
                    self.assertAlmostEqual(float(candidate[0, 0]), .3)

    @skipUnless(shutil.which('g++') and Path('/usr/include/eigen3/Eigen/Core').is_file(),
                'Requires g++ and Eigen3')
    def test_cpp_fk_checked_unchecked_and_coordinate_conversions(self):
        code = '\n'.join(CodeGenerationCpp(frame_tree(), use_safe_operator=True).generate_code())
        code = code.split('// Code below for debug')[0]
        main = r'''
#include <cassert>
int main() {
    using Solver = parameterized_frames_ik;
    for (const std::array<double, 4>& parameters : {
            std::array<double, 4>{{0., 0., 0., 0.}}, {{.4, -.7, .2, -.3}}, {{-.9, .6, -.5, .8}}}) {
        const double a=parameters[0], b=parameters[1], c=parameters[2], d=parameters[3];
        Eigen::Matrix4d pre=Eigen::Matrix4d::Identity(), raw=pre, post=pre;
        pre.block<3, 3>(0, 0)=Eigen::AngleAxisd(a, Eigen::Vector3d::UnitY()).toRotationMatrix();
        pre(0, 3)=d; pre(2, 3)=std::sqrt(2.)/3.;
        raw(2, 3)=.3+M_PI/7.;
        post.block<3, 3>(0, 0)=Eigen::AngleAxisd(b, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        post(0, 3)=c; post(1, 3)=.125;
        const Eigen::Matrix4d target=pre*raw*post;
        assert((Solver::computeFK({{.3}}, a,b,c,d)-target).cwiseAbs().maxCoeff()<1e-14);
        const auto converted=Solver::endEffectorTargetOriginalToRaw(target, a,b,c,d);
        assert((converted-raw).cwiseAbs().maxCoeff()<1e-14);
        assert((Solver::endEffectorTargetRawToOriginal(converted, a,b,c,d)-target).cwiseAbs().maxCoeff()<1e-14);
        Eigen::Vector3d world_point=target.block<3, 1>(0, 3);
        Eigen::Matrix<double, 3, 1> jacobian;
        Solver::computeTransformPointJacobian({{.3}}, a,b,c,d, world_point, jacobian);
        assert((jacobian-pre.block<3, 1>(0, 2)).cwiseAbs().maxCoeff()<1e-14);
        auto checked=Solver::computeIK(target, a,b,c,d);
        assert(!checked.empty());
        Solver::RawIKWorksace workspace;
        std::vector<std::array<double, 1>> unchecked;
        Solver::computeIKUnChecked(target, a,b,c,d, workspace, unchecked);
        assert(!unchecked.empty());
        for (const auto& q : checked) assert(std::abs(q[0]-.3)<1e-14);
        for (const auto& q : unchecked) assert(std::abs(q[0]-.3)<1e-14);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='yaik-frames-') as temporary:
            temporary = Path(temporary)
            source, executable = temporary/'test.cpp', temporary/'test'
            source.write_text(code+main)
            completed = subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT),
                                        '-I', '/usr/include/eigen3', str(source), '-o', str(executable)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            subprocess.run([str(executable)], check=True)
