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
from python_run_import import add_revolute_offset
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


def offset_fixed_tree():
    tree = runtime_tree()
    tree.robot.name = 'runtime_offset_fixed'
    tree.robot.auxiliary_data = RobotAuxiliaryData(unknown_offset=[0., .7])
    return tree


def periodic_angle_tree():
    model = large_angle_tree().robot
    model.name = 'runtime_periodic'
    model.unknowns[0] = Unknown(model.unknowns[0].symbol, lower_bound=0, upper_bound=float(sp.pi))
    solution = VariableSolution.make_explicit_solution(
        model.unknowns[0].symbol, [-sp.pi, sp.pi], SolutionMethod.OneVariableAlgebra.name)
    return SkeletonTree([solution], model)


def prismatic_periodic_tree():
    x, y = sp.symbols('x y')
    model = RobotDescription('runtime_prismatic')
    model.unknowns = [Unknown(symbol, UnknownType.Translational.name, -10, 10) for symbol in (x, y)]
    model.dh_params = [DHEntry(0, 0, symbol, 0) for symbol in (x, y)]
    solutions = [VariableSolution.make_explicit_solution(symbol, values, SolutionMethod.OneVariableAlgebra.name)
                 for symbol, values in ((x, [-sp.pi, sp.pi]), (y, [-x]))]
    return SkeletonTree(solutions, model)


class TestRuntimeSafety(TestCase):
    def test_large_offset_composition_matches_independent_rotation_products(self):
        def rz(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s], [s, c]])
        for angle, offset in ((1e20, .7), (.3, 1e20), (1e308, 1e308), (-1e20, -.7)):
            with self.subTest(angle=angle, offset=offset):
                np.testing.assert_allclose(rz(add_revolute_offset(angle, offset)),
                                           rz(angle)@rz(offset), atol=1e-14, rtol=0)

    def test_python_large_fixed_joint_offsets_preserve_fk_jacobian_and_input_value(self):
        scope = {}
        exec('\n'.join(CodeGenerationPython(offset_fixed_tree()).generate_code()), scope)
        for angle in (2*np.pi+.3, 1e20, -1e20, 1e300):
            c, s, co, so = np.cos(angle), np.sin(angle), np.cos(.7), np.sin(.7)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])@np.array([
                [co, -so, 0], [so, co, 0], [0, 0, 1]])
            expected = np.eye(4); expected[:3, :3] = rotation; expected[:3, 3] = [.4, 0, 1.]
            target = scope['runtime_offset_fixed_fk']([1., angle], .4)
            np.testing.assert_allclose(target, expected, atol=1e-14, rtol=0)
            output = scope['runtime_offset_fixed_ik_solve'](target, y_angle=angle, z_length=.4)
            self.assertTrue(output)
            for candidate in output:
                self.assertEqual(candidate[1, 0], angle)
                np.testing.assert_allclose(scope['runtime_offset_fixed_fk'](candidate, .4), expected, atol=1e-14, rtol=0)
            local = np.array([.2, -.1, .3])
            world = rotation@local+expected[:3, 3]
            jacobian = scope['runtime_offset_fixed_transform_point_jacobian']([1., angle], .4, world)
            relative = rotation@local
            np.testing.assert_allclose(jacobian[:, 1], [-relative[1], relative[0], 0.], atol=1e-14, rtol=0)

    def test_python_large_raw_root_recovers_a_valid_original_joint_angle(self):
        scope = {}
        exec('\n'.join(CodeGenerationPython(large_angle_tree(.7)).generate_code()), scope)
        target = np.eye(4)
        c, s = np.cos(1e20), np.sin(1e20)
        target[:2, :2] = [[c, -s], [s, c]]
        output = scope['runtime_angle_offset_ik_solve'](target)
        self.assertTrue(output)
        for candidate in output:
            np.testing.assert_allclose(scope['runtime_angle_offset_fk'](candidate), target, atol=1e-14, rtol=0)

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
                         for tree in (runtime_tree(), offset_fixed_tree(), large_angle_tree(), large_angle_tree(0.7),
                                      periodic_angle_tree(), prismatic_periodic_tree(), iiwa))
        main = r'''
int main() {
    std::vector<std::array<double, 1>> endpoints{{{-M_PI}}, {{M_PI}}};
    auto linear = endpoints;
    using SingleJointOutput = std::vector<std::array<double, 1>>;
    void (*linear_deduplicate)(SingleJointOutput&, double) = &yaik_cpp::removeDuplicate<1>;
    linear_deduplicate(linear, 1e-6);
    assert(linear.size() == 2);  // Omitting the mask preserves linear comparison.
    yaik_cpp::removeDuplicate<1>(endpoints, 1e-6, {{true}});
    assert(endpoints.size() == 1 && endpoints[0][0] == -M_PI);
    std::vector<std::array<double, 2>> mixed{{{-M_PI, 0}}, {{M_PI, 0}}};
    yaik_cpp::removeDuplicate<2>(mixed, 1e-6, {{false, true}});
    assert(mixed.size() == 2);
    for (double epsilon : {2e-7, 2e-5}) {
        std::vector<std::array<double, 1>> boundary{{{-M_PI + epsilon}}, {{M_PI - epsilon}}};
        yaik_cpp::removeDuplicate<1>(boundary, 1e-6, {{true}});
        assert(boundary.size() == (epsilon < 1e-6 ? 1 : 2));
    }
    const double huge = std::ldexp(2 * M_PI, 60);
    assert(std::abs(std::sin(huge)) > 1e-3);
    std::vector<std::array<double, 1>> distinct{{{huge}}, {{0}}};
    yaik_cpp::removeDuplicate<1>(distinct, 1e-6, {{true}});
    assert(distinct.size() == 2);  // remainder(huge, 2*pi)==0 would falsely merge these.
    std::vector<std::array<double, 1>> equivalent{{{1e20}}, {{std::atan2(std::sin(1e20), std::cos(1e20))}}};
    yaik_cpp::removeDuplicate<1>(equivalent, 1e-6, {{true}});
    assert(equivalent.size() == 1 && equivalent[0][0] == 1e20);
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        std::vector<std::array<double, 1>> nonfinite{{{invalid}}, {{invalid}}};
        std::feclearexcept(FE_ALL_EXCEPT);
        yaik_cpp::removeDuplicate<1>(nonfinite, 1e-6, {{true}});
        assert(nonfinite.size() == 2 && !std::fetestexcept(FE_INVALID | FE_OVERFLOW));
    }
    const double largest = std::numeric_limits<double>::max();
    std::vector<std::array<double, 1>> overflow{{{largest}}, {{-largest}}};
    std::feclearexcept(FE_ALL_EXCEPT);
    yaik_cpp::removeDuplicate<1>(overflow, 1e-6, {{true}});
    assert(overflow.size() == 2 && !std::fetestexcept(FE_INVALID | FE_OVERFLOW));
    const auto periodic_target = runtime_periodic_ik::computeFK({{M_PI}});
    const auto periodic_output = runtime_periodic_ik::computeIK(periodic_target);
    assert(periodic_output.size() == 2);
    std::vector<std::array<double, 1>> feasible;
    for (const auto& q : periodic_output) {
        assert((runtime_periodic_ik::computeFK(q) - periodic_target).squaredNorm() < 1e-20);
        if (q[0] >= 0 && q[0] <= M_PI) feasible.push_back(q);
    }
    // Screen the model's [0,pi] limits before opting into periodic deduplication.
    yaik_cpp::removeDuplicate<1>(feasible, 1e-6, {{true}});
    assert(feasible.size() == 1 && feasible[0][0] == M_PI);
    const auto prismatic_target = runtime_prismatic_ik::computeFK({{0, 0}});
    const auto prismatic_output = runtime_prismatic_ik::computeIK(prismatic_target);
    assert(prismatic_output.size() == 2);
    for (const auto& q : prismatic_output) {
        assert((runtime_prismatic_ik::computeFK(q) - prismatic_target).squaredNorm() < 1e-20);
    }
    for (double angle : {0.25, 2 * M_PI + 0.3, 1e20, -1e20}) {
        const auto target = runtime_safety_ik::computeFK({{1, angle}}, 0.4);
        const auto output = runtime_safety_ik::computeIK(target, 0.4, angle);
        assert(!output.empty());
        for (const auto& q : output) {
            assert(q[1] == angle);
            assert((runtime_safety_ik::computeFK(q, 0.4) - target).squaredNorm() < 1e-20);
        }
    }
    for (double angle : {2*M_PI+.3, 1e20, -1e20, 1e300}) {
        Eigen::Matrix4d expected=Eigen::Matrix4d::Identity();
        expected.block<3,3>(0,0)=(Eigen::AngleAxisd(angle,Eigen::Vector3d::UnitZ())*
                                  Eigen::AngleAxisd(.7,Eigen::Vector3d::UnitZ())).toRotationMatrix();
        expected(0,3)=.4; expected(2,3)=1.;
        const auto target=runtime_offset_fixed_ik::computeFK({{1.,angle}},.4);
        assert((target-expected).cwiseAbs().maxCoeff()<1e-14);
        const auto output=runtime_offset_fixed_ik::computeIK(target,.4,angle);
        assert(!output.empty());
        for(const auto& q : output) {
            assert(q[1]==angle);
            assert((runtime_offset_fixed_ik::computeFK(q,.4)-expected).cwiseAbs().maxCoeff()<1e-14);
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
    Eigen::Matrix4d raw_angle_target=Eigen::Matrix4d::Identity();
    raw_angle_target.block<3,3>(0,0)=Eigen::AngleAxisd(1e20,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const auto recovered_offset=runtime_angle_offset_ik::computeIK(raw_angle_target);
    assert(!recovered_offset.empty());
    for(const auto& candidate : recovered_offset)
        assert((runtime_angle_offset_ik::computeFK(candidate)-raw_angle_target).cwiseAbs().maxCoeff()<1e-14);
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
            source.write_text('#include <cfenv>\n' + code + main)
            compiled = subprocess.run(['g++', '-std=c++14', '-O0', '-I' + str(ROOT), '-I' + str(eigen),
                                       str(source), '-o', str(executable)], capture_output=True, text=True, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stderr)
