"""Boundary coverage for generated runtime inputs, angles, and refinement."""
from pathlib import Path
from unittest import TestCase
import shutil
import subprocess
import tempfile

import numpy as np
import sympy as sp

from codegen.cpp_codegen.codegen_cpp import CodeGenerationCpp
from codegen.py_codegen.codegen_py import CodeGenerationPython
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_serialize import read_skeleton_tree
from fk.robots import DHEntry, RobotAuxiliaryData, RobotDescription
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod, VariableSolution


ROOT = Path(__file__).resolve().parents[1]


def runtime_tree():
    x, angle, length = sp.symbols('x y_angle z_length')
    robot = RobotDescription('runtime_safety')
    robot.unknowns = [Unknown(x, UnknownType.Translational.name),
                      Unknown(angle, lower_bound=-1e21, upper_bound=1e21)]
    robot.dh_params = [DHEntry(0, length, x, 0), DHEntry(0, 0, 0, angle)]
    robot.unknown_as_parameter_more_dof = [angle]
    robot.symbolic_parameters = {length}
    robot.parameters_bound[length] = (0.1, 0.9)
    solution = VariableSolution.make_polynomial_solution(
        x, SolutionMethod.PolynomialDirect.name,
        {0: (-sp.S.One, sp.S.One), 2: (sp.S.One, sp.S.One)})
    return SkeletonTree([solution], robot)


def large_angle_tree(offset=None):
    angle = sp.Symbol('angle')
    robot = RobotDescription('runtime_angle' if offset is None else 'runtime_angle_offset')
    robot.unknowns = [Unknown(angle, lower_bound=-1e21, upper_bound=1e21)]
    robot.dh_params = [DHEntry(0, 0, 0, angle)]
    if offset is not None:
        robot.auxiliary_data = RobotAuxiliaryData(unknown_offset=[offset])
    solution = VariableSolution.make_explicit_solution(
        angle, [sp.Float(1e20)], SolutionMethod.OneVariableAlgebra.name)
    return SkeletonTree([solution], robot)


class TestRuntimeSafety(TestCase):
    def python_solver(self):
        scope = {}
        exec('\n'.join(CodeGenerationPython(runtime_tree()).generate_code()), scope)
        return scope

    def test_python_rejects_nonfinite_free_and_geometric_parameters(self):
        scope = self.python_solver()
        target = scope['runtime_safety_fk']([1, 0.25], 0.4)
        for invalid in (np.nan, np.inf, -np.inf):
            for angle, length in ((invalid, 0.4), (0.25, invalid)):
                with self.subTest(angle=angle, length=length):
                    self.assertEqual(scope['runtime_safety_ik_solve'](target, y_angle=angle, z_length=length), [])

    def test_python_preserves_fixed_joint_representation(self):
        scope = self.python_solver()
        for angle in (2 * np.pi + 0.3, 1e20, -1e20):
            target = scope['runtime_safety_fk']([1, angle], 0.4)
            output = scope['runtime_safety_ik_solve'](target, y_angle=angle, z_length=0.4)
            self.assertTrue(output)
            for solution in output:
                self.assertEqual(solution[1, 0], angle)
                np.testing.assert_allclose(scope['runtime_safety_fk'](solution, 0.4), target, atol=1e-12)

    def test_python_pose_tolerance_contract(self):
        scope = self.python_solver()
        target = scope['runtime_safety_fk']([1, 0.25], 0.4)
        target[0, 3] += 9e-4
        # The fixed X offset cannot realize this target. Python uses max-entry error.
        self.assertEqual(scope['runtime_safety_ik_solve'](target, y_angle=0.25, z_length=0.4), [])

    def test_native_angles_inputs_refinement_and_tolerance_contract(self):
        eigen = Path('/usr/include/eigen3')
        if not shutil.which('g++') or not (eigen / 'Eigen/Core').exists():
            self.skipTest('g++ and Eigen are needed for the C++ integration test')
        iiwa = read_skeleton_tree(str(ROOT / 'gallery/kuka_iiwa/kuka_iiwa_ik.yaml'))
        code = '\n'.join('\n'.join(CodeGenerationCpp(tree, use_safe_operator=True).generate_code())
                         .split('// Code below for debug')[0]
                         for tree in (runtime_tree(), large_angle_tree(), large_angle_tree(0.7), iiwa))
        main = r'''
int main() {
    for (double angle : {0.25, 2 * M_PI + 0.3, 1e20, -1e20}) {
        const auto target = runtime_safety_ik::computeFK({{1, angle}}, 0.4);
        const auto output = runtime_safety_ik::computeIK(target, 0.4, angle);
        assert(!output.empty());
        for (const auto& q : output) {
            assert(q[1] == angle);
            assert((runtime_safety_ik::computeFK(q, 0.4) - target).squaredNorm() < 1e-20);
        }
    }
    const auto target = runtime_safety_ik::computeFK({{1, 0.25}}, 0.4);
    runtime_safety_ik::RawIKWorksace workspace;
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        auto stale = runtime_safety_ik::computeIK(target, 0.4, 0.25);
        assert(!stale.empty());
        runtime_safety_ik::computeIK(target, 0.4, invalid, workspace, stale);
        assert(stale.empty());
        assert(runtime_safety_ik::computeIK(target, invalid, 0.25).empty());
    }
    std::vector<std::array<double, 2>> mixed_joints{{{8, 6.5}}};
    runtime_safety_ik::wrapAngleToPi(mixed_joints);
    assert(mixed_joints[0][0] == 8 && mixed_joints[0][1] == 6.5);
    for (double angle : {1e20, -1e20, 2 * M_PI + 0.3}) {
        std::vector<std::array<double, 1>> normalized{{{angle}}};
        runtime_angle_ik::wrapAngleToPi(normalized);
        assert(std::abs(normalized[0][0]) <= M_PI);
        assert((runtime_angle_ik::computeFK(normalized[0]) -
                runtime_angle_ik::computeFK({{angle}})).squaredNorm() < 1e-20);
    }
    const auto angle_target = runtime_angle_ik::computeFK({{1e20}});
    const auto angle_output = runtime_angle_ik::computeIK(angle_target);
    assert(!angle_output.empty());
    for (const auto& q : angle_output) {
        assert(std::abs(q[0]) <= M_PI);
        assert((runtime_angle_ik::computeFK(q) - angle_target).squaredNorm() < 1e-20);
    }
    // Applying a small offset to a huge angle loses bits; check the normalized output again.
    const auto offset_target = runtime_angle_offset_ik::computeFK({{1e20}});
    const auto offset_output = runtime_angle_offset_ik::computeIK(offset_target);
    for (const auto& q : offset_output) {
        assert((runtime_angle_offset_ik::computeFK(q) - offset_target).squaredNorm() <
               runtime_angle_offset_ik::pose_tolerance_degenerate);
    }
    for (double fixed : {2 * M_PI + 0.3, 1e20, -1e20}) {
        const std::array<double, 7> q{{fixed, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5}};
        const auto pose = kuka_iiwa_ik::computeFK(q);
        const auto output = kuka_iiwa_ik::computeIK(pose, fixed);
        assert(!output.empty());
        for (const auto& solution : output) {
            assert(solution[0] == fixed);
            assert((kuka_iiwa_ik::computeFK(solution) - pose).squaredNorm() < kuka_iiwa_ik::pose_tolerance);
        }
    }
    // Preserve the established C++ squared-norm contract; Python rejects this target.
    auto approximate_target = target;
    approximate_target(0, 3) += 9e-4;
    const auto approximate = runtime_safety_ik::computeIK(approximate_target, 0.4, 0.25);
    assert(!approximate.empty());
    assert(std::abs(runtime_safety_ik::computeFK(approximate[0], 0.4)(0, 3) - approximate_target(0, 3)) > 8e-4);

    int fk_calls = 0, jacobian_calls = 0;
    auto fk = [&](const std::array<double, 1>& q) -> Eigen::Matrix4d {
        ++fk_calls;
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(2, 3) = q[0];
        return pose;
    };
    double scale = 1;
    auto jacobian = [&](const std::array<double, 1>&, Eigen::Matrix<double, 6, 1>& J) {
        ++jacobian_calls;
        J.setZero(); J(5, 0) = scale;
    };
    std::array<double, 1> q{{1}};
    const Eigen::Matrix4d exact = fk(q);
    fk_calls = jacobian_calls = 0;
    yaik_cpp::numericalRefinement<1>(fk, jacobian, exact, q);
    assert(fk_calls == 1 && jacobian_calls == 0 && q[0] == 1);
    q[0] = 0; jacobian_calls = 0;
    yaik_cpp::numericalRefinement<1>(fk, jacobian, exact, q);
    assert(jacobian_calls > 0 && jacobian_calls < 10 && std::abs(q[0] - 1) < 1e-10);
    q[0] = 0; scale = 0; jacobian_calls = 0;
    yaik_cpp::numericalRefinement<1>(fk, jacobian, exact, q);
    assert(jacobian_calls == 1 && q[0] == 0);
    scale = 0.4; jacobian_calls = 0;
    yaik_cpp::numericalRefinement<1>(fk, jacobian, exact, q);
    assert(jacobian_calls == 10);  // A nonmonotonic path retains its iteration budget.
    q[0] = 0; scale = 1; jacobian_calls = 0;
    auto half_turn = Eigen::Matrix4d::Identity().eval();
    half_turn(0, 0) = half_turn(1, 1) = -1;
    yaik_cpp::numericalRefinement<1>(fk, jacobian, half_turn, q);
    assert(jacobian_calls == 1 && (fk(q) - half_turn).squaredNorm() > 1);
    auto invalid_target = exact;
    invalid_target(0, 0) = std::numeric_limits<double>::quiet_NaN();
    fk_calls = jacobian_calls = 0;
    yaik_cpp::numericalRefinement<1>(fk, jacobian, invalid_target, q);
    assert(fk_calls == 0 && jacobian_calls == 0 && q[0] == 0);
    auto invalid_fk = [](const std::array<double, 1>&) -> Eigen::Matrix4d {
        return Eigen::Matrix4d::Constant(std::numeric_limits<double>::quiet_NaN());
    };
    yaik_cpp::numericalRefinement<1>(invalid_fk, jacobian, exact, q);
    assert(jacobian_calls == 0 && q[0] == 0);
    auto invalid_jacobian = [](const std::array<double, 1>&, Eigen::Matrix<double, 6, 1>& J) {
        J.setConstant(std::numeric_limits<double>::infinity());
    };
    yaik_cpp::numericalRefinement<1>(fk, invalid_jacobian, exact, q);
    assert(q[0] == 0);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'runtime.cpp'
            executable = Path(directory) / 'runtime'
            source.write_text(code + main)
            compiled = subprocess.run(['g++', '-std=c++14', '-O0', '-I' + str(ROOT), '-I' + str(eigen),
                                       str(source), '-o', str(executable)], capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stderr)
