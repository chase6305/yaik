#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct yaskawa_HC10_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 384;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -1.5707963267948966, -3.141592653589793, -6.283185307179586, -6.283185307179586, -6.283185307179586}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 4.71238898038469, 3.141592653589793, 0.0, 0.0, 0.0}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 28;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_0 = 0.7;
static constexpr double d_1 = -0.5;
static constexpr double d_2 = -0.162;
static constexpr double d_3 = -0.13;
static constexpr double pre_transform_special_symbol_23 = 0.275;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 3.141592653589793;

// The transformation between raw and original ee target
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// ee_original = pre_transform * ee_raw * post_transform
// ee_raw = dh_forward_transform(theta_raw)
static Eigen::Matrix4d endEffectorTargetOriginalToRaw(const Eigen::Matrix4d& T_ee)
{
    const double r_11 = T_ee(0, 0);
    const double r_12 = T_ee(0, 1);
    const double r_13 = T_ee(0, 2);
    const double Px = T_ee(0, 3);
    const double r_21 = T_ee(1, 0);
    const double r_22 = T_ee(1, 1);
    const double r_23 = T_ee(1, 2);
    const double Py = T_ee(1, 3);
    const double r_31 = T_ee(2, 0);
    const double r_32 = T_ee(2, 1);
    const double r_33 = T_ee(2, 2);
    const double Pz = T_ee(2, 3);
    Eigen::Matrix4d ee_transformed;
    ee_transformed.setIdentity();
    ee_transformed(0, 0) = 1.0*r_13;
    ee_transformed(0, 1) = 1.0*r_12;
    ee_transformed(0, 2) = -1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_23;
    ee_transformed(1, 1) = 1.0*r_22;
    ee_transformed(1, 2) = -1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = 1.0*r_33;
    ee_transformed(2, 1) = 1.0*r_32;
    ee_transformed(2, 2) = -1.0*r_31;
    ee_transformed(2, 3) = 1.0*Pz - 1.0*pre_transform_special_symbol_23;
    return ee_transformed;
}

static Eigen::Matrix4d endEffectorTargetRawToOriginal(const Eigen::Matrix4d& T_ee)
{
    const double r_11 = T_ee(0, 0);
    const double r_12 = T_ee(0, 1);
    const double r_13 = T_ee(0, 2);
    const double Px = T_ee(0, 3);
    const double r_21 = T_ee(1, 0);
    const double r_22 = T_ee(1, 1);
    const double r_23 = T_ee(1, 2);
    const double Py = T_ee(1, 3);
    const double r_31 = T_ee(2, 0);
    const double r_32 = T_ee(2, 1);
    const double r_33 = T_ee(2, 2);
    const double Pz = T_ee(2, 3);
    Eigen::Matrix4d ee_transformed;
    ee_transformed.setIdentity();
    ee_transformed(0, 0) = -1.0*r_13;
    ee_transformed(0, 1) = 1.0*r_12;
    ee_transformed(0, 2) = 1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = -1.0*r_23;
    ee_transformed(1, 1) = 1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = -1.0*r_33;
    ee_transformed(2, 1) = 1.0*r_32;
    ee_transformed(2, 2) = 1.0*r_31;
    ee_transformed(2, 3) = 1.0*Pz + 1.0*pre_transform_special_symbol_23;
    return ee_transformed;
}

///************* Below are the actual FK and IK implementations *******************
static Eigen::Matrix4d computeFK(const std::array<double, robot_nq>& theta_input_original)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = std::sin(th_5);
    const double x1 = std::sin(th_0);
    const double x2 = std::cos(th_3);
    const double x3 = std::sin(th_3);
    const double x4 = std::cos(th_0);
    const double x5 = std::sin(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x5*x6;
    const double x8 = std::cos(th_1);
    const double x9 = std::cos(th_2);
    const double x10 = x8*x9;
    const double x11 = x10*x4 + x4*x7;
    const double x12 = -x1*x2 - x11*x3;
    const double x13 = std::cos(th_5);
    const double x14 = std::sin(th_4);
    const double x15 = x5*x9;
    const double x16 = x6*x8;
    const double x17 = x15*x4 - x16*x4;
    const double x18 = std::cos(th_4);
    const double x19 = -x1*x3 + x11*x2;
    const double x20 = -x14*x17 + x18*x19;
    const double x21 = -x14*x19 - x17*x18;
    const double x22 = a_0*x8;
    const double x23 = x1*x10 + x1*x7;
    const double x24 = x2*x4 - x23*x3;
    const double x25 = x1*x15 - x1*x16;
    const double x26 = x2*x23 + x3*x4;
    const double x27 = -x14*x25 + x18*x26;
    const double x28 = -x14*x26 - x18*x25;
    const double x29 = -x15 + x6*x8;
    const double x30 = x29*x3;
    const double x31 = x10 + x7;
    const double x32 = -x14*x31 + x18*x2*x29;
    const double x33 = -x14*x2*x29 - x18*x31;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x12 + x13*x20;
    ee_pose_raw(0, 1) = -x0*x20 - x12*x13;
    ee_pose_raw(0, 2) = x21;
    ee_pose_raw(0, 3) = d_1*x17 + d_2*x12 + d_3*x21 + x22*x4;
    ee_pose_raw(1, 0) = -x0*x24 + x13*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x13*x24;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = d_1*x25 + d_2*x24 + d_3*x28 + x1*x22;
    ee_pose_raw(2, 0) = x0*x30 + x13*x32;
    ee_pose_raw(2, 1) = -x0*x32 + x13*x29*x3;
    ee_pose_raw(2, 2) = x33;
    ee_pose_raw(2, 3) = -a_0*x5 + d_1*x31 - d_2*x30 + d_3*x33;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_2);
    const double x4 = std::cos(th_0);
    const double x5 = 1.0*x4;
    const double x6 = x3*x5;
    const double x7 = std::cos(th_1);
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = x2*x6 - x7*x9;
    const double x11 = std::cos(th_3);
    const double x12 = std::sin(th_3);
    const double x13 = x2*x9 + x6*x7;
    const double x14 = -x1*x11 - x12*x13;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x15 - x16*(-x1*x12 + x11*x13);
    const double x18 = x1*x3;
    const double x19 = x1*x8;
    const double x20 = x18*x2 - x19*x7;
    const double x21 = x18*x7 + x19*x2;
    const double x22 = 1.0*x11*x4 - x12*x21;
    const double x23 = -x15*x20 - x16*(x11*x21 + x12*x5);
    const double x24 = 1.0*x2;
    const double x25 = 1.0*x7;
    const double x26 = x24*x8 + x25*x3;
    const double x27 = -x24*x3 + 1.0*x7*x8;
    const double x28 = x12*x27;
    const double x29 = -x11*x16*x27 - x15*x26;
    const double x30 = a_0*x24;
    const double x31 = -pre_transform_special_symbol_23 + x30;
    const double x32 = -x31;
    const double x33 = a_0*x7;
    const double x34 = d_1*x20 + x1*x33;
    const double x35 = d_1*x26 + pre_transform_special_symbol_23 - x30;
    const double x36 = d_2*x28;
    const double x37 = d_1*x26 - x31 - x36;
    const double x38 = d_2*x22 + x34;
    const double x39 = d_3*x29 + x35 - x36;
    const double x40 = d_3*x23 + x38;
    const double x41 = d_1*x10 + x33*x5;
    const double x42 = d_2*x14 + x41;
    const double x43 = d_3*x17 + x42;
    const double x44 = a_0*x25;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x17;
    jacobian(1, 1) = x5;
    jacobian(1, 2) = -x5;
    jacobian(1, 3) = x20;
    jacobian(1, 4) = x22;
    jacobian(1, 5) = x23;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x26;
    jacobian(2, 4) = -x28;
    jacobian(2, 5) = x29;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x5;
    jacobian(3, 2) = x32*x5;
    jacobian(3, 3) = -x20*x35 + x26*x34;
    jacobian(3, 4) = -x22*x37 - x28*x38;
    jacobian(3, 5) = -x23*x39 + x29*x40;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = x1*x32;
    jacobian(4, 3) = x10*x35 - x26*x41;
    jacobian(4, 4) = x14*x37 + x28*x42;
    jacobian(4, 5) = x17*x39 - x29*x43;
    jacobian(5, 2) = -std::pow(x0, 2)*x44 - std::pow(x4, 2)*x44;
    jacobian(5, 3) = -x10*x34 + x20*x41;
    jacobian(5, 4) = -x14*x38 + x22*x42;
    jacobian(5, 5) = -x17*x40 + x23*x43;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_2);
    const double x3 = std::cos(th_0);
    const double x4 = 1.0*x3;
    const double x5 = x2*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x4*x7;
    const double x9 = x1*x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x1*x8 + x5*x6;
    const double x13 = std::cos(th_4);
    const double x14 = std::sin(th_4);
    const double x15 = x0*x2;
    const double x16 = x0*x7;
    const double x17 = x1*x15 - x16*x6;
    const double x18 = x1*x16 + x15*x6;
    const double x19 = 1.0*x1;
    const double x20 = x19*x7 + 1.0*x2*x6;
    const double x21 = -x19*x2 + 1.0*x6*x7;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = -x0*x10 - x11*x12;
    jacobian(0, 5) = -x13*x9 - x14*(-x0*x11 + x10*x12);
    jacobian(1, 1) = x4;
    jacobian(1, 2) = -x4;
    jacobian(1, 3) = x17;
    jacobian(1, 4) = 1.0*x10*x3 - x11*x18;
    jacobian(1, 5) = -x13*x17 - x14*(x10*x18 + x11*x4);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x20;
    jacobian(2, 4) = -x11*x21;
    jacobian(2, 5) = -x10*x14*x21 - x13*x20;
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];

    // Temp variable for efficiency
    const double x0 = 1.0*p_on_ee_y;
    const double x1 = std::cos(th_0);
    const double x2 = 1.0*x1;
    const double x3 = p_on_ee_z*x2;
    const double x4 = std::sin(th_1);
    const double x5 = 1.0*x4;
    const double x6 = a_0*x5;
    const double x7 = -pre_transform_special_symbol_23 + x6;
    const double x8 = -x7;
    const double x9 = std::sin(th_2);
    const double x10 = x5*x9;
    const double x11 = std::cos(th_2);
    const double x12 = std::cos(th_1);
    const double x13 = 1.0*x12;
    const double x14 = x11*x13;
    const double x15 = x10 + x14;
    const double x16 = std::sin(th_0);
    const double x17 = x11*x5;
    const double x18 = -x13*x16*x9 + x16*x17;
    const double x19 = d_1*x15 + pre_transform_special_symbol_23 - x6;
    const double x20 = a_0*x13;
    const double x21 = d_1*x18 + x16*x20;
    const double x22 = std::sin(th_3);
    const double x23 = 1.0*x12*x9 - x17;
    const double x24 = x22*x23;
    const double x25 = std::cos(th_3);
    const double x26 = x10*x16 + x14*x16;
    const double x27 = 1.0*x1*x25 - x22*x26;
    const double x28 = d_2*x24;
    const double x29 = d_1*x15 - x28 - x7;
    const double x30 = d_2*x27 + x21;
    const double x31 = std::cos(th_4);
    const double x32 = std::sin(th_4);
    const double x33 = -x15*x31 - x23*x25*x32;
    const double x34 = -x18*x31 - x32*(x2*x22 + x25*x26);
    const double x35 = d_3*x33 + x19 - x28;
    const double x36 = d_3*x34 + x30;
    const double x37 = 1.0*p_on_ee_x;
    const double x38 = 1.0*x16;
    const double x39 = p_on_ee_z*x38;
    const double x40 = x11*x2;
    const double x41 = x12*x2;
    const double x42 = x4*x40 - x41*x9;
    const double x43 = a_0*x41 + d_1*x42;
    const double x44 = x12*x40 + x2*x4*x9;
    const double x45 = -x22*x44 - x25*x38;
    const double x46 = d_2*x45 + x43;
    const double x47 = -x31*x42 - x32*(-x22*x38 + x25*x44);
    const double x48 = d_3*x47 + x46;
    const double x49 = x1*x37;
    const double x50 = x0*x16;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = 1.0*x1*x8 - x3;
    jacobian(0, 3) = -p_on_ee_y*x15 + p_on_ee_z*x18 + x15*x21 - x18*x19;
    jacobian(0, 4) = p_on_ee_y*x24 + p_on_ee_z*x27 - x24*x30 - x27*x29;
    jacobian(0, 5) = -p_on_ee_y*x33 + p_on_ee_z*x34 + x33*x36 - x34*x35;
    jacobian(1, 0) = x37;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x38 + x39;
    jacobian(1, 2) = 1.0*x16*x8 - x39;
    jacobian(1, 3) = p_on_ee_x*x15 - p_on_ee_z*x42 - x15*x43 + x19*x42;
    jacobian(1, 4) = -p_on_ee_x*x24 - p_on_ee_z*x45 + x22*x23*x46 + x29*x45;
    jacobian(1, 5) = p_on_ee_x*x33 - p_on_ee_z*x47 - x33*x48 + x35*x47;
    jacobian(2, 1) = -x49 - x50;
    jacobian(2, 2) = -std::pow(x1, 2)*x20 - std::pow(x16, 2)*x20 + x49 + x50;
    jacobian(2, 3) = -p_on_ee_x*x18 + p_on_ee_y*x42 + x18*x43 - x21*x42;
    jacobian(2, 4) = -p_on_ee_x*x27 + p_on_ee_y*x45 + x27*x46 - x30*x45;
    jacobian(2, 5) = -p_on_ee_x*x34 + p_on_ee_y*x47 + x34*x48 - x36*x47;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, SolutionQueueType& solution_queue, NodeIndexWorkspaceType& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    // Extracting the ik target symbols
    const double r_11 = T_ee(0, 0);
    const double r_12 = T_ee(0, 1);
    const double r_13 = T_ee(0, 2);
    const double Px = T_ee(0, 3);
    const double r_21 = T_ee(1, 0);
    const double r_22 = T_ee(1, 1);
    const double r_23 = T_ee(1, 2);
    const double Py = T_ee(1, 3);
    const double r_31 = T_ee(2, 0);
    const double r_32 = T_ee(2, 1);
    const double r_33 = T_ee(2, 2);
    const double Pz = T_ee(2, 3);
    const auto& ee_rotation = T_ee.block<3, 3>(0, 0);
    const auto& ee_translation = T_ee.block<3, 1>(0, 3);
    const Eigen::Vector3d inv_ee_translation = - ee_rotation.transpose() * ee_translation;
    const double inv_Px = inv_ee_translation(0);
    const double inv_Py = inv_ee_translation(1);
    const double inv_Pz = inv_ee_translation(2);

    solution_queue.reset();
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 48, 48, 96, 96, 192, 192, 384, 384, 384, 384, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192}};
    node_index_workspace.reset(node_input_capacities, max_n_solutions);

    using RawSolution = IntermediateSolution<intermediate_solution_size>;
    auto make_raw_solution = []() -> RawSolution { RawSolution result; result.fill(std::numeric_limits<double>::quiet_NaN()); return result; };

    auto append_solution_to_queue = [&solution_queue](RawSolution solution_2_add) -> int {
        return solution_queue.push_solution(solution_2_add);
    };

    auto add_input_index_to = [&node_index_workspace](int node_idx, int solution_idx) -> void {
        if (solution_idx < 0) return;
        node_index_workspace.append_index_to_node(node_idx, solution_idx);
    };

    // Code for non-branch dispatcher node 0
    // Actually, there is no code

    // Code for general_6dof solution node 1, solved variable is th_2
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;

        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 0) = d_2*r_21;
        R_l(0, 1) = d_2*r_22;
        R_l(0, 2) = d_2*r_11;
        R_l(0, 3) = d_2*r_12;
        R_l(0, 4) = Py - d_3*r_23;
        R_l(0, 5) = Px - d_3*r_13;
        R_l(1, 0) = d_2*r_11;
        R_l(1, 1) = d_2*r_12;
        R_l(1, 2) = -d_2*r_21;
        R_l(1, 3) = -d_2*r_22;
        R_l(1, 4) = Px - d_3*r_13;
        R_l(1, 5) = -Py + d_3*r_23;
        R_l(2, 6) = d_2*r_31;
        R_l(2, 7) = d_2*r_32;
        R_l(3, 0) = -r_21;
        R_l(3, 1) = -r_22;
        R_l(3, 2) = -r_11;
        R_l(3, 3) = -r_12;
        R_l(4, 0) = -r_11;
        R_l(4, 1) = -r_12;
        R_l(4, 2) = r_21;
        R_l(4, 3) = r_22;
        R_l(5, 6) = 2*Px*d_2*r_11 + 2*Py*d_2*r_21 + 2*Pz*d_2*r_31 - 2*d_2*d_3*r_11*r_13 - 2*d_2*d_3*r_21*r_23 - 2*d_2*d_3*r_31*r_33;
        R_l(5, 7) = 2*Px*d_2*r_12 + 2*Py*d_2*r_22 + 2*Pz*d_2*r_32 - 2*d_2*d_3*r_12*r_13 - 2*d_2*d_3*r_22*r_23 - 2*d_2*d_3*r_32*r_33;
        R_l(6, 0) = -Px*r_31 + Pz*r_11 - d_3*r_11*r_33 + d_3*r_13*r_31;
        R_l(6, 1) = -Px*r_32 + Pz*r_12 - d_3*r_12*r_33 + d_3*r_13*r_32;
        R_l(6, 2) = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31;
        R_l(6, 3) = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32;
        R_l(7, 0) = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31;
        R_l(7, 1) = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32;
        R_l(7, 2) = Px*r_31 - Pz*r_11 + d_3*r_11*r_33 - d_3*r_13*r_31;
        R_l(7, 3) = Px*r_32 - Pz*r_12 + d_3*r_12*r_33 - d_3*r_13*r_32;
        Eigen::Matrix<double, 8, 8> R_l_mat_inv = R_l.inverse();
        for(auto r = 0; r < R_l_mat_inv.rows(); r++) {
            for(auto c = 0; c < R_l_mat_inv.cols(); c++) {
                if (std::isnan(R_l_mat_inv(r, c)) || (!std::isfinite(R_l_mat_inv(r, c)))) return;
            }
        }

        const double R_l_inv_00 = R_l_mat_inv(0, 0);
        const double R_l_inv_01 = R_l_mat_inv(0, 1);
        const double R_l_inv_02 = R_l_mat_inv(0, 2);
        const double R_l_inv_03 = R_l_mat_inv(0, 3);
        const double R_l_inv_04 = R_l_mat_inv(0, 4);
        const double R_l_inv_05 = R_l_mat_inv(0, 5);
        const double R_l_inv_06 = R_l_mat_inv(0, 6);
        const double R_l_inv_07 = R_l_mat_inv(0, 7);
        const double R_l_inv_10 = R_l_mat_inv(1, 0);
        const double R_l_inv_11 = R_l_mat_inv(1, 1);
        const double R_l_inv_12 = R_l_mat_inv(1, 2);
        const double R_l_inv_13 = R_l_mat_inv(1, 3);
        const double R_l_inv_14 = R_l_mat_inv(1, 4);
        const double R_l_inv_15 = R_l_mat_inv(1, 5);
        const double R_l_inv_16 = R_l_mat_inv(1, 6);
        const double R_l_inv_17 = R_l_mat_inv(1, 7);
        const double R_l_inv_20 = R_l_mat_inv(2, 0);
        const double R_l_inv_21 = R_l_mat_inv(2, 1);
        const double R_l_inv_22 = R_l_mat_inv(2, 2);
        const double R_l_inv_23 = R_l_mat_inv(2, 3);
        const double R_l_inv_24 = R_l_mat_inv(2, 4);
        const double R_l_inv_25 = R_l_mat_inv(2, 5);
        const double R_l_inv_26 = R_l_mat_inv(2, 6);
        const double R_l_inv_27 = R_l_mat_inv(2, 7);
        const double R_l_inv_30 = R_l_mat_inv(3, 0);
        const double R_l_inv_31 = R_l_mat_inv(3, 1);
        const double R_l_inv_32 = R_l_mat_inv(3, 2);
        const double R_l_inv_33 = R_l_mat_inv(3, 3);
        const double R_l_inv_34 = R_l_mat_inv(3, 4);
        const double R_l_inv_35 = R_l_mat_inv(3, 5);
        const double R_l_inv_36 = R_l_mat_inv(3, 6);
        const double R_l_inv_37 = R_l_mat_inv(3, 7);
        const double R_l_inv_40 = R_l_mat_inv(4, 0);
        const double R_l_inv_41 = R_l_mat_inv(4, 1);
        const double R_l_inv_42 = R_l_mat_inv(4, 2);
        const double R_l_inv_43 = R_l_mat_inv(4, 3);
        const double R_l_inv_44 = R_l_mat_inv(4, 4);
        const double R_l_inv_45 = R_l_mat_inv(4, 5);
        const double R_l_inv_46 = R_l_mat_inv(4, 6);
        const double R_l_inv_47 = R_l_mat_inv(4, 7);
        const double R_l_inv_50 = R_l_mat_inv(5, 0);
        const double R_l_inv_51 = R_l_mat_inv(5, 1);
        const double R_l_inv_52 = R_l_mat_inv(5, 2);
        const double R_l_inv_53 = R_l_mat_inv(5, 3);
        const double R_l_inv_54 = R_l_mat_inv(5, 4);
        const double R_l_inv_55 = R_l_mat_inv(5, 5);
        const double R_l_inv_56 = R_l_mat_inv(5, 6);
        const double R_l_inv_57 = R_l_mat_inv(5, 7);
        const double R_l_inv_60 = R_l_mat_inv(6, 0);
        const double R_l_inv_61 = R_l_mat_inv(6, 1);
        const double R_l_inv_62 = R_l_mat_inv(6, 2);
        const double R_l_inv_63 = R_l_mat_inv(6, 3);
        const double R_l_inv_64 = R_l_mat_inv(6, 4);
        const double R_l_inv_65 = R_l_mat_inv(6, 5);
        const double R_l_inv_66 = R_l_mat_inv(6, 6);
        const double R_l_inv_67 = R_l_mat_inv(6, 7);
        const double R_l_inv_70 = R_l_mat_inv(7, 0);
        const double R_l_inv_71 = R_l_mat_inv(7, 1);
        const double R_l_inv_72 = R_l_mat_inv(7, 2);
        const double R_l_inv_73 = R_l_mat_inv(7, 3);
        const double R_l_inv_74 = R_l_mat_inv(7, 4);
        const double R_l_inv_75 = R_l_mat_inv(7, 5);
        const double R_l_inv_76 = R_l_mat_inv(7, 6);
        const double R_l_inv_77 = R_l_mat_inv(7, 7);

        // Temp variable for efficiency
        const double x0 = R_l_inv_60*r_31 + R_l_inv_70*r_32;
        const double x1 = a_0*x0;
        const double x2 = R_l_inv_66*r_31;
        const double x3 = R_l_inv_76*r_32;
        const double x4 = x2 + x3;
        const double x5 = d_1*x4;
        const double x6 = R_l_inv_62*r_31 + R_l_inv_72*r_32;
        const double x7 = d_1*x6;
        const double x8 = -x7;
        const double x9 = x1 + x5 + x8;
        const double x10 = R_l_inv_64*r_31;
        const double x11 = R_l_inv_74*r_32;
        const double x12 = d_3*r_33;
        const double x13 = Pz - x12;
        const double x14 = x13*x6;
        const double x15 = R_l_inv_65*r_31 + R_l_inv_75*r_32;
        const double x16 = std::pow(Px, 2);
        const double x17 = std::pow(Py, 2);
        const double x18 = std::pow(Pz, 2);
        const double x19 = std::pow(d_2, 2);
        const double x20 = std::pow(r_11, 2);
        const double x21 = x19*x20;
        const double x22 = std::pow(r_21, 2);
        const double x23 = x19*x22;
        const double x24 = std::pow(r_31, 2);
        const double x25 = x19*x24;
        const double x26 = std::pow(d_3, 2);
        const double x27 = std::pow(r_13, 2)*x26;
        const double x28 = std::pow(r_23, 2)*x26;
        const double x29 = std::pow(r_33, 2)*x26;
        const double x30 = d_3*r_13;
        const double x31 = 2*Px;
        const double x32 = x30*x31;
        const double x33 = d_3*r_23;
        const double x34 = 2*Py;
        const double x35 = x33*x34;
        const double x36 = 2*Pz;
        const double x37 = x12*x36;
        const double x38 = std::pow(a_0, 2);
        const double x39 = std::pow(d_1, 2);
        const double x40 = -x38 - x39;
        const double x41 = x16 + x17 + x18 + x21 + x23 + x25 + x27 + x28 + x29 - x32 - x35 - x37 + x40;
        const double x42 = x15*x41;
        const double x43 = x14 + x42;
        const double x44 = -x10 - x11 + x43;
        const double x45 = 2*a_0;
        const double x46 = x4*x45;
        const double x47 = x45*x6;
        const double x48 = 2*d_1;
        const double x49 = x0*x48;
        const double x50 = x47 + x49;
        const double x51 = x10 + x11;
        const double x52 = -x14 - x42 + x51;
        const double x53 = R_l_inv_67*r_31 + R_l_inv_77*r_32;
        const double x54 = x48*x53;
        const double x55 = 2*R_l_inv_63;
        const double x56 = 2*R_l_inv_73;
        const double x57 = r_31*x55 + r_32*x56;
        const double x58 = -x54 - x57;
        const double x59 = -x54 + x57;
        const double x60 = -x5;
        const double x61 = x1 + x60;
        const double x62 = x43 + x51;
        const double x63 = x62 + x8;
        const double x64 = -x1;
        const double x65 = x5 + x62 + x7;
        const double x66 = Px*r_11;
        const double x67 = Py*r_21;
        const double x68 = Pz*r_31;
        const double x69 = r_11*x30;
        const double x70 = r_21*x33;
        const double x71 = r_31*x12;
        const double x72 = x66 + x67 + x68 - x69 - x70 - x71;
        const double x73 = Px*r_12;
        const double x74 = Py*r_22;
        const double x75 = Pz*r_32;
        const double x76 = r_12*x30;
        const double x77 = r_22*x33;
        const double x78 = r_32*x12;
        const double x79 = x73 + x74 + x75 - x76 - x77 - x78;
        const double x80 = R_l_inv_62*x72 + R_l_inv_72*x79;
        const double x81 = d_1*x80;
        const double x82 = R_l_inv_60*x72 + R_l_inv_70*x79;
        const double x83 = a_0*x82;
        const double x84 = -x83;
        const double x85 = x81 + x84;
        const double x86 = R_l_inv_64*x72;
        const double x87 = R_l_inv_74*x79;
        const double x88 = R_l_inv_66*x72;
        const double x89 = R_l_inv_76*x79;
        const double x90 = x88 + x89;
        const double x91 = d_1*x90;
        const double x92 = x86 + x87 - x91;
        const double x93 = d_2*x20;
        const double x94 = d_2*x22;
        const double x95 = d_2*x24;
        const double x96 = x13*x80;
        const double x97 = R_l_inv_65*x72 + R_l_inv_75*x79;
        const double x98 = x41*x97;
        const double x99 = x93 + x94 + x95 - x96 - x98;
        const double x100 = x92 + x99;
        const double x101 = x45*x90;
        const double x102 = x45*x80;
        const double x103 = x48*x82;
        const double x104 = x102 + x103;
        const double x105 = -x81;
        const double x106 = x105 + x83;
        const double x107 = x86 + x87 + x91;
        const double x108 = x107 + x99;
        const double x109 = R_l_inv_67*x72 + R_l_inv_77*x79;
        const double x110 = x109*x48;
        const double x111 = -x45;
        const double x112 = x55*x72 + x56*x79;
        const double x113 = x111 + x112;
        const double x114 = -x110;
        const double x115 = x112 + x45;
        const double x116 = -d_2*x20 - d_2*x22 - d_2*x24 + x96 + x98;
        const double x117 = x107 + x116;
        const double x118 = Px*r_21;
        const double x119 = Py*r_11;
        const double x120 = r_11*x33 - r_21*x30 + x118 - x119;
        const double x121 = R_l_inv_64*x120;
        const double x122 = Px*r_22;
        const double x123 = Py*r_12;
        const double x124 = r_12*x33 - r_22*x30 + x122 - x123;
        const double x125 = R_l_inv_74*x124;
        const double x126 = R_l_inv_62*x120 + R_l_inv_72*x124;
        const double x127 = R_l_inv_65*x120 + R_l_inv_75*x124;
        const double x128 = x121 + x125 - x126*x13 - x127*x41;
        const double x129 = R_l_inv_66*x120 + R_l_inv_76*x124;
        const double x130 = d_1*x129;
        const double x131 = -x130;
        const double x132 = a_0 + x131;
        const double x133 = d_1*x126;
        const double x134 = R_l_inv_60*x120 + R_l_inv_70*x124;
        const double x135 = a_0*x134;
        const double x136 = -x135;
        const double x137 = x133 + x136;
        const double x138 = -x48;
        const double x139 = x129*x45;
        const double x140 = -x139;
        const double x141 = x126*x45;
        const double x142 = x134*x48;
        const double x143 = x141 + x142;
        const double x144 = -x133;
        const double x145 = x135 + x144;
        const double x146 = -a_0;
        const double x147 = x130 + x146;
        const double x148 = R_l_inv_67*x120 + R_l_inv_77*x124;
        const double x149 = x120*x55 + x124*x56;
        const double x150 = x148*x48 + x149;
        const double x151 = 2*d_1*x148 - x149;
        const double x152 = x121 + x125 + x126*x13 + x127*x41;
        const double x153 = Py*x20 + Py*x22 + Py*x24 - x20*x33 - x22*x33 - x24*x33;
        const double x154 = 2*d_2;
        const double x155 = x153*x154;
        const double x156 = Px*x20 + Px*x22 + Px*x24 - x20*x30 - x22*x30 - x24*x30;
        const double x157 = x154*x156;
        const double x158 = 2*x33;
        const double x159 = 2*x30;
        const double x160 = 2*r_11;
        const double x161 = r_23*x26;
        const double x162 = r_13*x161;
        const double x163 = 2*r_31;
        const double x164 = r_33*x161;
        const double x165 = std::pow(r_21, 3)*x19 - r_21*x16 + r_21*x17 - r_21*x18 + r_21*x21 + r_21*x25 - r_21*x27 + r_21*x28 - r_21*x29 + r_21*x37 + x118*x159 - x119*x159 - x158*x66 - x158*x67 - x158*x68 + x160*x162 + x163*x164 + x34*x66 + x34*x68 - x34*x71;
        const double x166 = 2*x162;
        const double x167 = r_13*r_33*x26;
        const double x168 = std::pow(r_11, 3)*x19 + r_11*x16 - r_11*x17 - r_11*x18 + r_11*x23 + r_11*x25 + r_11*x27 - r_11*x28 - r_11*x29 + r_11*x37 + r_21*x166 - x118*x158 + x119*x158 - x159*x66 - x159*x67 - x159*x68 + x163*x167 + x31*x67 + x31*x68 - x31*x71;
        const double x169 = r_21*x19;
        const double x170 = x160*x169;
        const double x171 = x163*x169;
        const double x172 = 2*r_32;
        const double x173 = r_12*x166 + r_12*x170 - r_22*x16 + r_22*x17 - r_22*x18 + r_22*x21 + 3*r_22*x23 + r_22*x25 - r_22*x27 + r_22*x28 - r_22*x29 + r_22*x37 + r_32*x171 + x122*x159 - x123*x159 - x158*x73 - x158*x74 - x158*x75 + x164*x172 + x34*x73 + x34*x75 - x34*x78;
        const double x174 = r_31*x160*x19;
        const double x175 = r_12*x16 - r_12*x17 - r_12*x18 + 3*r_12*x21 + r_12*x23 + r_12*x25 + r_12*x27 - r_12*x28 - r_12*x29 + r_12*x37 + r_22*x166 + r_22*x170 + r_32*x174 - x122*x158 + x123*x158 - x159*x73 - x159*x74 - x159*x75 + x167*x172 + x31*x74 + x31*x75 - x31*x78;
        const double x176 = R_l_inv_02*x165 + R_l_inv_12*x173 + R_l_inv_22*x168 + R_l_inv_32*x175 + R_l_inv_42*x155 + R_l_inv_52*x157;
        const double x177 = d_1*x176;
        const double x178 = R_l_inv_00*x165 + R_l_inv_10*x173 + R_l_inv_20*x168 + R_l_inv_30*x175 + R_l_inv_40*x155 + R_l_inv_50*x157;
        const double x179 = a_0*x178;
        const double x180 = -x179;
        const double x181 = x177 + x180;
        const double x182 = R_l_inv_04*x165;
        const double x183 = R_l_inv_14*x173;
        const double x184 = R_l_inv_24*x168;
        const double x185 = R_l_inv_34*x175;
        const double x186 = R_l_inv_06*x165 + R_l_inv_16*x173 + R_l_inv_26*x168 + R_l_inv_36*x175 + R_l_inv_46*x155 + R_l_inv_56*x157;
        const double x187 = d_1*x186;
        const double x188 = R_l_inv_44*x155;
        const double x189 = R_l_inv_54*x157;
        const double x190 = x182 + x183 + x184 + x185 - x187 + x188 + x189;
        const double x191 = R_l_inv_05*x165 + R_l_inv_15*x173 + R_l_inv_25*x168 + R_l_inv_35*x175 + R_l_inv_45*x155 + R_l_inv_55*x157;
        const double x192 = -x13*x176 - x191*x41;
        const double x193 = x190 + x192;
        const double x194 = x186*x45;
        const double x195 = x176*x45;
        const double x196 = x178*x48;
        const double x197 = x195 + x196;
        const double x198 = -x177;
        const double x199 = x179 + x198;
        const double x200 = x182 + x183 + x184 + x185 + x187 + x188 + x189;
        const double x201 = x192 + x200;
        const double x202 = R_l_inv_07*x165 + R_l_inv_17*x173 + R_l_inv_27*x168 + R_l_inv_37*x175 + R_l_inv_47*x155 + R_l_inv_57*x157;
        const double x203 = 2*x165;
        const double x204 = 2*x173;
        const double x205 = 2*x168;
        const double x206 = 2*x175;
        const double x207 = 4*d_2;
        const double x208 = x153*x207;
        const double x209 = x156*x207;
        const double x210 = R_l_inv_03*x203 + R_l_inv_13*x204 + R_l_inv_23*x205 + R_l_inv_33*x206 + R_l_inv_43*x208 + R_l_inv_53*x209 + 2*x38 - 2*x39;
        const double x211 = x202*x48 + x210;
        const double x212 = a_0*d_1;
        const double x213 = 8*x212;
        const double x214 = 2*d_1*x202 - x210;
        const double x215 = x13*x176 + x191*x41;
        const double x216 = x190 + x215;
        const double x217 = x200 + x215;
        const double x218 = R_l_inv_40*x157;
        const double x219 = R_l_inv_50*x155;
        const double x220 = R_l_inv_00*x168;
        const double x221 = R_l_inv_20*x165;
        const double x222 = R_l_inv_10*x175;
        const double x223 = R_l_inv_30*x173;
        const double x224 = x218 - x219 + x220 - x221 + x222 - x223;
        const double x225 = a_0*x224;
        const double x226 = R_l_inv_46*x157;
        const double x227 = R_l_inv_56*x155;
        const double x228 = R_l_inv_06*x168;
        const double x229 = R_l_inv_26*x165;
        const double x230 = R_l_inv_16*x175;
        const double x231 = R_l_inv_36*x173;
        const double x232 = x226 - x227 + x228 - x229 + x230 - x231;
        const double x233 = d_1*x232;
        const double x234 = R_l_inv_02*x168 + R_l_inv_12*x175 - R_l_inv_22*x165 - R_l_inv_32*x173 + R_l_inv_42*x157 - R_l_inv_52*x155;
        const double x235 = x13*x234;
        const double x236 = R_l_inv_05*x168 + R_l_inv_15*x175 - R_l_inv_25*x165 - R_l_inv_35*x173 + R_l_inv_45*x157 - R_l_inv_55*x155;
        const double x237 = x236*x41;
        const double x238 = x233 + x235 + x237;
        const double x239 = x225 + x238;
        const double x240 = d_1*x234;
        const double x241 = R_l_inv_24*x165;
        const double x242 = R_l_inv_34*x173;
        const double x243 = R_l_inv_04*x168;
        const double x244 = R_l_inv_14*x175;
        const double x245 = R_l_inv_44*x157;
        const double x246 = R_l_inv_54*x155;
        const double x247 = x241 + x242 - x243 - x244 - x245 + x246 + x38 + x39;
        const double x248 = -x240 + x247;
        const double x249 = x232*x45;
        const double x250 = x234*x45;
        const double x251 = x224*x48;
        const double x252 = x250 + x251;
        const double x253 = -x225;
        const double x254 = x240 + x253;
        const double x255 = x247 + x254;
        const double x256 = -x233 + x235 + x237;
        const double x257 = R_l_inv_43*x209;
        const double x258 = R_l_inv_53*x208;
        const double x259 = R_l_inv_03*x205;
        const double x260 = R_l_inv_23*x203;
        const double x261 = R_l_inv_13*x206;
        const double x262 = R_l_inv_33*x204;
        const double x263 = R_l_inv_07*x168 + R_l_inv_17*x175 - R_l_inv_27*x165 - R_l_inv_37*x173 + R_l_inv_47*x157 - R_l_inv_57*x155;
        const double x264 = x263*x48;
        const double x265 = x257 - x258 + x259 - x260 + x261 - x262 + x264;
        const double x266 = -x257 + x258 - x259 + x260 - x261 + x262 + x264;
        const double x267 = -x13*x234 + x233 - x236*x41;
        const double x268 = -x241 - x242 + x243 + x244 + x245 - x246 + x40;
        const double x269 = 2*x12;
        const double x270 = 2*x164;
        const double x271 = r_21*x270 + std::pow(r_31, 3)*x19 - r_31*x16 - r_31*x17 + r_31*x18 + r_31*x21 + r_31*x23 - r_31*x27 - r_31*x28 + r_31*x29 + r_31*x32 + r_31*x35 + x160*x167 - x269*x66 - x269*x67 - x269*x68 + x36*x66 + x36*x67 - x36*x69 - x36*x70;
        const double x272 = 2*r_12*x167 + r_12*x174 + r_22*x171 + r_22*x270 - r_32*x16 - r_32*x17 + r_32*x18 + r_32*x21 + r_32*x23 + 3*r_32*x25 - r_32*x27 - r_32*x28 + r_32*x29 + r_32*x32 + r_32*x35 - x269*x73 - x269*x74 - x269*x75 + x36*x73 + x36*x74 - x36*x76 - x36*x77;
        const double x273 = R_l_inv_60*x271 + R_l_inv_70*x272;
        const double x274 = a_0*x273;
        const double x275 = R_l_inv_66*x271 + R_l_inv_76*x272;
        const double x276 = d_1*x275;
        const double x277 = R_l_inv_62*x271 + R_l_inv_72*x272;
        const double x278 = d_1*x277;
        const double x279 = -x278;
        const double x280 = x274 + x276 + x279;
        const double x281 = R_l_inv_64*x271;
        const double x282 = R_l_inv_74*x272;
        const double x283 = R_l_inv_65*x271 + R_l_inv_75*x272;
        const double x284 = x36*x93;
        const double x285 = x36*x94;
        const double x286 = x36*x95;
        const double x287 = x13*x277 + x269*x93 + x269*x94 + x269*x95 + x283*x41 - x284 - x285 - x286;
        const double x288 = -x281 - x282 + x287;
        const double x289 = x275*x45;
        const double x290 = x277*x45;
        const double x291 = x273*x48;
        const double x292 = x290 + x291;
        const double x293 = x281 + x282;
        const double x294 = -2*d_2*d_3*r_33*x20 - 2*d_2*d_3*r_33*x22 - 2*d_2*d_3*r_33*x24 - x13*x277 - x283*x41 + x284 + x285 + x286 + x293;
        const double x295 = R_l_inv_67*x271 + R_l_inv_77*x272;
        const double x296 = x295*x48;
        const double x297 = 4*x212;
        const double x298 = x271*x55 + x272*x56;
        const double x299 = -x297 + x298;
        const double x300 = 4*x38;
        const double x301 = 4*x39;
        const double x302 = x300 - x301;
        const double x303 = -2*d_1*x295;
        const double x304 = -x276;
        const double x305 = x274 + x304;
        const double x306 = x287 + x293;
        const double x307 = x279 + x306;
        const double x308 = -x274;
        const double x309 = x276 + x278 + x306;
        const double x310 = x15*x45;
        const double x311 = x48*(x0 - x310);
        const double x312 = 4*d_1;
        const double x313 = -x48*(x0 + x310);
        const double x314 = 4*a_0;
        const double x315 = x314*x53;
        const double x316 = 8*R_l_inv_63;
        const double x317 = 8*R_l_inv_73;
        const double x318 = x45*x97;
        const double x319 = x48*(-x318 + x82);
        const double x320 = -x48*(x318 + x82);
        const double x321 = x109*x314;
        const double x322 = -2*a_0*x127;
        const double x323 = x134 - 1;
        const double x324 = x127*x45;
        const double x325 = -x148*x314;
        const double x326 = x134 + 1;
        const double x327 = x48*(2*a_0*x191 - x178);
        const double x328 = x48*(x178 + x191*x45);
        const double x329 = -x202*x314;
        const double x330 = 16*d_2;
        const double x331 = x153*x330;
        const double x332 = x156*x330;
        const double x333 = 8*x165;
        const double x334 = 8*x168;
        const double x335 = 8*x173;
        const double x336 = 8*x175;
        const double x337 = x224 + x45;
        const double x338 = x236*x45;
        const double x339 = -x263*x314;
        const double x340 = x48*(2*a_0*x283 - x273);
        const double x341 = x48*(x273 + x283*x45);
        const double x342 = x295*x314;
        const double x343 = x300 + x301;
        const double x344 = x61 + x7;
        const double x345 = x81 + x83;
        const double x346 = a_0 + x130;
        const double x347 = x136 + x144;
        const double x348 = x141 - x142;
        const double x349 = x133 + x135;
        const double x350 = x131 + x146;
        const double x351 = x180 + x198;
        const double x352 = x177 + x179;
        const double x353 = x225 + x240 + x247;
        const double x354 = x278 + x305;
        const double x355 = x297 + x298;

        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = -x44 - x9;
        A(0, 1) = x46 - x50;
        A(0, 2) = x52 + x9;
        A(0, 3) = x58;
        A(0, 4) = -4;
        A(0, 5) = x59;
        A(0, 6) = -x61 - x63;
        A(0, 7) = -x46 - x50;
        A(0, 8) = -x64 - x65;
        A(1, 0) = x100 + x85;
        A(1, 1) = x101 - x104;
        A(1, 2) = x106 + x108;
        A(1, 3) = -x110 - x113;
        A(1, 5) = x114 + x115;
        A(1, 6) = -x106 - x116 - x92;
        A(1, 7) = -x101 - x104;
        A(1, 8) = -x117 - x85;
        A(2, 0) = -x128 - x132 - x137;
        A(2, 1) = x138 + x140 + x143;
        A(2, 2) = -x128 - x145 - x147;
        A(2, 3) = x150;
        A(2, 5) = x151;
        A(2, 6) = x132 + x145 + x152;
        A(2, 7) = x139 + x143 + x48;
        A(2, 8) = x137 + x147 + x152;
        A(3, 0) = -x181 - x193;
        A(3, 1) = -x194 + x197;
        A(3, 2) = -x199 - x201;
        A(3, 3) = x211;
        A(3, 4) = x213;
        A(3, 5) = x214;
        A(3, 6) = x199 + x216;
        A(3, 7) = x194 + x197;
        A(3, 8) = x181 + x217;
        A(4, 0) = x239 + x248;
        A(4, 1) = -x249 + x252;
        A(4, 2) = x255 + x256;
        A(4, 3) = x265;
        A(4, 5) = x266;
        A(4, 6) = -x255 - x267;
        A(4, 7) = x249 + x252;
        A(4, 8) = x238 + x254 + x268;
        A(5, 0) = x280 + x288;
        A(5, 1) = -x289 + x292;
        A(5, 2) = -x280 - x294;
        A(5, 3) = x296 + x299;
        A(5, 4) = x302;
        A(5, 5) = -x299 - x303;
        A(5, 6) = x305 + x307;
        A(5, 7) = x289 + x292;
        A(5, 8) = x308 + x309;

        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x311;
        B(0, 1) = x312*(-x2 - x3 + x6);
        B(0, 2) = x313;
        B(0, 3) = x315 + 4;
        B(0, 4) = -r_31*x316 - r_32*x317;
        B(0, 5) = x315 - 4;
        B(0, 6) = x311;
        B(0, 7) = x312*(x4 + x6);
        B(0, 8) = x313;
        B(1, 0) = x319;
        B(1, 1) = x312*(x80 - x88 - x89);
        B(1, 2) = x320;
        B(1, 3) = x321;
        B(1, 4) = -x316*x72 - x317*x79;
        B(1, 5) = x321;
        B(1, 6) = x319;
        B(1, 7) = x312*(x80 + x90);
        B(1, 8) = x320;
        B(2, 0) = x48*(-x322 - x323);
        B(2, 1) = x312*(R_l_inv_66*x120 + R_l_inv_76*x124 - x126);
        B(2, 2) = x48*(x323 + x324);
        B(2, 3) = x325;
        B(2, 4) = x120*x316 + x124*x317;
        B(2, 5) = x325;
        B(2, 6) = x48*(-x322 - x326);
        B(2, 7) = -x312*(x126 + x129);
        B(2, 8) = x48*(x324 + x326);
        B(3, 0) = x327;
        B(3, 1) = x312*(R_l_inv_06*x165 + R_l_inv_16*x173 + R_l_inv_26*x168 + R_l_inv_36*x175 + 2*R_l_inv_46*d_2*x153 + 2*R_l_inv_56*d_2*x156 - x176);
        B(3, 2) = x328;
        B(3, 3) = x329;
        B(3, 4) = R_l_inv_03*x333 + R_l_inv_13*x335 + R_l_inv_23*x334 + R_l_inv_33*x336 + R_l_inv_43*x331 + R_l_inv_53*x332 - 8*x38 - 8*x39;
        B(3, 5) = x329;
        B(3, 6) = x327;
        B(3, 7) = -x312*(x176 + x186);
        B(3, 8) = x328;
        B(4, 0) = x48*(2*a_0*x236 - x337);
        B(4, 1) = x312*(x226 - x227 + x228 - x229 + x230 - x231 - x234);
        B(4, 2) = x48*(x111 + x224 + x338);
        B(4, 3) = x339;
        B(4, 4) = R_l_inv_03*x334 + R_l_inv_13*x336 - R_l_inv_23*x333 - R_l_inv_33*x335 + R_l_inv_43*x332 - R_l_inv_53*x331;
        B(4, 5) = x339;
        B(4, 6) = x48*(-x218 + x219 - x220 + x221 - x222 + x223 + x338 + x45);
        B(4, 7) = x312*(-x232 - x234);
        B(4, 8) = x48*(x337 + x338);
        B(5, 0) = x340;
        B(5, 1) = x312*(R_l_inv_66*x271 + R_l_inv_76*x272 - x277);
        B(5, 2) = x341;
        B(5, 3) = -x342 + x343;
        B(5, 4) = x271*x316 + x272*x317;
        B(5, 5) = -x342 - x343;
        B(5, 6) = x340;
        B(5, 7) = -x312*(x275 + x277);
        B(5, 8) = x341;

        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = -x344 - x44;
        C(0, 1) = x46 - x47 + x49;
        C(0, 2) = x344 + x52;
        C(0, 3) = x59;
        C(0, 4) = 4;
        C(0, 5) = x58;
        C(0, 6) = -x1 - x65;
        C(0, 7) = -x46 - x47 + x49;
        C(0, 8) = -x60 - x63 - x64;
        C(1, 0) = x105 + x108 + x84;
        C(1, 1) = x101 - x102 + x103;
        C(1, 2) = x100 + x345;
        C(1, 3) = x113 + x114;
        C(1, 5) = -x110 - x115;
        C(1, 6) = -x117 - x345;
        C(1, 7) = -x101 - x102 + x103;
        C(1, 8) = x345 - x86 - x87 + x91 + x99;
        C(2, 0) = -x128 - x346 - x347;
        C(2, 1) = x140 + x348 + x48;
        C(2, 2) = -x128 - x349 - x350;
        C(2, 3) = x151;
        C(2, 5) = x150;
        C(2, 6) = x152 + x346 + x349;
        C(2, 7) = x138 + x139 + x348;
        C(2, 8) = x152 + x347 + x350;
        C(3, 0) = -x201 - x351;
        C(3, 1) = 2*a_0*x176 - x194 - x196;
        C(3, 2) = -x193 - x352;
        C(3, 3) = x214;
        C(3, 4) = x213;
        C(3, 5) = x211;
        C(3, 6) = x217 + x352;
        C(3, 7) = x194 + x195 - x196;
        C(3, 8) = x216 + x351;
        C(4, 0) = x256 + x353;
        C(4, 1) = 2*a_0*x234 - x249 - x251;
        C(4, 2) = x238 + x248 + x253;
        C(4, 3) = x266;
        C(4, 5) = x265;
        C(4, 6) = x239 + x240 + x268;
        C(4, 7) = x249 + x250 - x251;
        C(4, 8) = -x267 - x353;
        C(5, 0) = x288 + x354;
        C(5, 1) = 2*a_0*x277 - x289 - x291;
        C(5, 2) = -x294 - x354;
        C(5, 3) = -x303 - x355;
        C(5, 4) = -x302;
        C(5, 5) = x296 + x355;
        C(5, 6) = x274 + x309;
        C(5, 7) = x289 + x290 - x291;
        C(5, 8) = x304 + x307 + x308;

        // Invoke the solver
        std::array<double, 48> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);

        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[2] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor();
    // Finish code for general_6dof solution node 0

    // Code for solved_variable dispatcher node 2
    auto SolvedVariableDispatcherNode_node_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(2);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(2);
        if (!this_input_valid)
            return;

        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[2];

            const bool degenerate_valid_0 = std::fabs(th_2 - 1.0/2.0*M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }

            const bool degenerate_valid_1 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }

            if (!taken_by_degenerate)
                add_input_index_to(3, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor();
    // Finish code for solved_variable dispatcher node 2

    // Code for explicit solution node 3, solved variable is th_3
    auto ExplicitSolutionNode_node_3_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(3);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(3);
        if (!this_input_valid)
            return;

        // The solution of non-root node 3
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[2];
            const bool condition_0 = ((a_0 != 0) && (d_2 != 0) && (std::cos(th_2) != 0) && ((1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 - std::pow(a_0, 2) + 2*a_0*d_1*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_0*d_2*std::cos(th_2))) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = 2*d_3;
                const double x1 = std::pow(d_3, 2);
                const double x2 = safe_asin((1.0/2.0)*(std::pow(Px, 2) - Px*r_13*x0 + std::pow(Py, 2) - Py*r_23*x0 + std::pow(Pz, 2) - Pz*r_33*x0 - std::pow(a_0, 2) + 2*a_0*d_1*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x1 + std::pow(r_23, 2)*x1 + std::pow(r_33, 2)*x1)/(a_0*d_2*std::cos(th_2)));
                RawSolution solution_0(this_solution);
                solution_0[3] = -x2;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = x2 + M_PI;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor();
    // Finish code for explicit solution node 3

    // Code for equation all-zero dispatcher node 4
    auto EquationAllZeroDispatcherNode_node_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(4);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(4);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[3];
            const bool checked_result = ((std::fabs(d_2*std::cos(th_3)) <= 9.9999999999999995e-7) && (std::fabs(Px - d_3*r_13) <= 9.9999999999999995e-7) && (std::fabs(Py - d_3*r_23) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4

    // Code for explicit solution node 5, solved variable is th_0
    auto ExplicitSolutionNode_node_5_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(5);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(5);
        if (!this_input_valid)
            return;

        // The solution of non-root node 5
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[3];
            const bool condition_0 = ((std::fabs(d_2*std::cos(th_3)) >= zero_tolerance) || (std::fabs(Px - d_3*r_13) >= zero_tolerance) || (std::fabs(Py - d_3*r_23) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = -Px + d_3*r_13;
                const double x1 = Py - d_3*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::cos(th_3);
                const double x4 = safe_sqrt(-std::pow(d_2, 2)*std::pow(x3, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x5 = d_2*x3;
                RawSolution solution_0(this_solution);
                solution_0[0] = x2 + std::atan2(x4, x5);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx_0);
                const double tmp_sol_value = x2 + std::atan2(-x4, x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_0_processor();
    // Finish code for explicit solution node 5

    // Code for solved_variable dispatcher node 6
    auto SolvedVariableDispatcherNode_node_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(6);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(6);
        if (!this_input_valid)
            return;

        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            bool forwarded_to_child = false;
            const double th_3 = this_solution[3];

            const bool degenerate_valid_0 = std::fabs(th_3) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(12, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(17, branch_input_index);
            }

            if (!taken_by_degenerate)
                add_input_index_to(7, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor();
    // Finish code for solved_variable dispatcher node 6

    // Code for explicit solution node 17, solved variable is th_5
    auto ExplicitSolutionNode_node_17_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(17);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(17);
        if (!this_input_valid)
            return;

        // The solution of non-root node 17
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(18, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_5_processor();
    // Finish code for explicit solution node 17

    // Code for solved_variable dispatcher node 18
    auto SolvedVariableDispatcherNode_node_18_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
        if (!this_input_valid)
            return;

        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[2];

            const bool degenerate_valid_0 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(25, node_input_i_idx_in_queue);
            }

            if (!taken_by_degenerate)
                add_input_index_to(19, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_18_processor();
    // Finish code for solved_variable dispatcher node 18

    // Code for explicit solution node 25, solved variable is th_1
    auto ExplicitSolutionNode_node_25_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(25);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(25);
        if (!this_input_valid)
            return;

        // The solution of non-root node 25
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((std::fabs(a_0 + d_1) >= 9.9999999999999995e-7) || (std::fabs(d_2*std::sin(th_3)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = a_0 + d_1;
                const double x1 = Pz - d_3*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_2*std::sin(th_3);
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_1_processor();
    // Finish code for explicit solution node 25

    // Code for non-branch dispatcher node 26
    // Actually, there is no code

    // Code for explicit solution node 27, solved variable is th_4
    auto ExplicitSolutionNode_node_27_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(26);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(26);
        if (!this_input_valid)
            return;

        // The solution of non-root node 27
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_0)*std::sin(th_3) + std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) - r_23*(-std::sin(th_0)*std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_0);
                const double x3 = std::sin(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_1);
                const double x6 = x1*x5;
                const double tmp_sol_value = std::atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_4_processor();
    // Finish code for explicit solution node 26

    // Code for explicit solution node 19, solved variable is th_1
    auto ExplicitSolutionNode_node_19_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(19);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(19);
        if (!this_input_valid)
            return;

        // The solution of non-root node 19
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const bool condition_0 = ((std::fabs(d_1*std::cos(th_2)) >= 9.9999999999999995e-7) || (std::fabs(a_0 - d_1*std::sin(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_3*r_33;
                const double x1 = a_0 - d_1*std::sin(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_1*std::cos(th_2);
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1_processor();
    // Finish code for explicit solution node 19

    // Code for non-branch dispatcher node 20
    // Actually, there is no code

    // Code for explicit solution node 21, solved variable is th_4
    auto ExplicitSolutionNode_node_21_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(20);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(20);
        if (!this_input_valid)
            return;

        // The solution of non-root node 21
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + r_23*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - r_33*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))) >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_33*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = x0*x1 - x2*x3;
                const double x5 = x0*x2 + x1*x3;
                const double x6 = r_13*std::cos(th_0);
                const double x7 = r_23*std::sin(th_0);
                const double tmp_sol_value = std::atan2(-r_33*x4 + x5*x6 + x5*x7, -r_33*x5 - x4*x6 - x4*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_4_processor();
    // Finish code for explicit solution node 20

    // Code for explicit solution node 12, solved variable is th_5
    auto ExplicitSolutionNode_node_12_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(12);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(12);
        if (!this_input_valid)
            return;

        // The solution of non-root node 12
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(13, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_5_processor();
    // Finish code for explicit solution node 12

    // Code for solved_variable dispatcher node 13
    auto SolvedVariableDispatcherNode_node_13_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;

        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[2];

            const bool degenerate_valid_0 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }

            if (!taken_by_degenerate)
                add_input_index_to(14, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor();
    // Finish code for solved_variable dispatcher node 13

    // Code for explicit solution node 22, solved variable is th_1
    auto ExplicitSolutionNode_node_22_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(22);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(22);
        if (!this_input_valid)
            return;

        // The solution of non-root node 22
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((std::fabs(a_0 + d_1) >= 9.9999999999999995e-7) || (std::fabs(d_2*std::sin(th_3)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = a_0 + d_1;
                const double x1 = Pz - d_3*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_2*std::sin(th_3);
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(23, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_1_processor();
    // Finish code for explicit solution node 22

    // Code for non-branch dispatcher node 23
    // Actually, there is no code

    // Code for explicit solution node 24, solved variable is th_4
    auto ExplicitSolutionNode_node_24_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(23);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(23);
        if (!this_input_valid)
            return;

        // The solution of non-root node 24
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_0)*std::sin(th_3) + std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) - r_23*(-std::sin(th_0)*std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_0);
                const double x3 = std::sin(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_1);
                const double x6 = x1*x5;
                const double tmp_sol_value = std::atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_4_processor();
    // Finish code for explicit solution node 23

    // Code for explicit solution node 14, solved variable is th_1
    auto ExplicitSolutionNode_node_14_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(14);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(14);
        if (!this_input_valid)
            return;

        // The solution of non-root node 14
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const bool condition_0 = ((std::fabs(d_1*std::cos(th_2)) >= 9.9999999999999995e-7) || (std::fabs(a_0 - d_1*std::sin(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_3*r_33;
                const double x1 = a_0 - d_1*std::sin(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_1*std::cos(th_2);
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_1_processor();
    // Finish code for explicit solution node 14

    // Code for non-branch dispatcher node 15
    // Actually, there is no code

    // Code for explicit solution node 16, solved variable is th_4
    auto ExplicitSolutionNode_node_16_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(15);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(15);
        if (!this_input_valid)
            return;

        // The solution of non-root node 16
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + r_23*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - r_33*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))) >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_33*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = x0*x1 - x2*x3;
                const double x5 = x0*x2 + x1*x3;
                const double x6 = r_13*std::cos(th_0);
                const double x7 = r_23*std::sin(th_0);
                const double tmp_sol_value = std::atan2(r_33*x4 - x5*x6 - x5*x7, -r_33*x5 - x4*x6 - x4*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor();
    // Finish code for explicit solution node 15

    // Code for explicit solution node 7, solved variable is th_4
    auto ExplicitSolutionNode_node_7_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(7);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(7);
        if (!this_input_valid)
            return;

        // The solution of non-root node 7
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((std::sin(th_3) != 0) && (std::fabs((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3)) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = safe_asin((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3));
                RawSolution solution_0(this_solution);
                solution_0[4] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx_0);
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_4_processor();
    // Finish code for explicit solution node 7

    // Code for equation all-zero dispatcher node 8
    auto EquationAllZeroDispatcherNode_node_8_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(8);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(8);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const bool checked_result = ((std::fabs(d_1*std::cos(th_2) - d_2*std::sin(th_2)*std::sin(th_3)) <= 9.9999999999999995e-7) && (std::fabs(-a_0 + d_1*std::sin(th_2) + d_2*std::sin(th_3)*std::cos(th_2)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8

    // Code for explicit solution node 9, solved variable is th_1
    auto ExplicitSolutionNode_node_9_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(9);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(9);
        if (!this_input_valid)
            return;

        // The solution of non-root node 9
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((std::fabs(d_1*std::cos(th_2) - d_2*std::sin(th_2)*std::sin(th_3)) >= 9.9999999999999995e-7) || (std::fabs(-a_0 + d_1*std::sin(th_2) + d_2*std::sin(th_3)*std::cos(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_3*r_33;
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(th_2);
                const double x3 = d_2*std::sin(th_3);
                const double x4 = a_0 - d_1*x1 - x2*x3;
                const double x5 = d_1*x2 - x1*x3;
                const double x6 = std::cos(th_0);
                const double x7 = std::sin(th_0);
                const double x8 = Px*x6 + Py*x7 - d_3*r_13*x6 - d_3*r_23*x7;
                const double tmp_sol_value = std::atan2(-x0*x4 + x5*x8, x0*x5 + x4*x8);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1_processor();
    // Finish code for explicit solution node 9

    // Code for non-branch dispatcher node 10
    // Actually, there is no code

    // Code for explicit solution node 11, solved variable is th_5
    auto ExplicitSolutionNode_node_11_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(10);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(10);
        if (!this_input_valid)
            return;

        // The solution of non-root node 11
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*((std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_3)*std::cos(th_0) + std::sin(th_0)*std::cos(th_3)) - r_21*((-std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))*std::sin(th_0)*std::sin(th_3) + std::cos(th_0)*std::cos(th_3)) - r_31*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_3)) >= zero_tolerance) || (std::fabs(r_12*((std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_3)*std::cos(th_0) + std::sin(th_0)*std::cos(th_3)) - r_22*((-std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))*std::sin(th_0)*std::sin(th_3) + std::cos(th_0)*std::cos(th_3)) - r_32*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_2);
                const double x3 = std::sin(th_2);
                const double x4 = std::cos(th_1);
                const double x5 = x0*(x1*x2 - x3*x4);
                const double x6 = std::sin(th_0);
                const double x7 = std::cos(th_3);
                const double x8 = std::cos(th_0);
                const double x9 = x1*x3 + x2*x4;
                const double x10 = x0*x8*x9 + x6*x7;
                const double x11 = -x0*x6*x9 + x7*x8;
                const double tmp_sol_value = std::atan2(r_11*x10 - r_21*x11 - r_31*x5, r_12*x10 - r_22*x11 - r_32*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_5_processor();
    // Finish code for explicit solution node 10

    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = raw_ik_out_i[0];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[1];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[2];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[3];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[4];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[5];  // th_5
        new_ik_i[5] = value_at_5;
        if (std::all_of(new_ik_i.begin(), new_ik_i.end(), [](double value) { return std::isfinite(value); }))
            ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    computeRawIK(T_ee_raw, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw;
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
        ik_out_i[4] = addRevoluteOffset(ik_out_i[4], -th_4_offset_original2raw);
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
        ik_output.push_back(ik_out_i);
    }
}

static void wrapAngleToPi(std::array<double, robot_nq>& solution_i)
{
    // Wrap angular value to [-pi, pi]
    // Revolute unknown th_0
    solution_i[0] = std::atan2(std::sin(solution_i[0]), std::cos(solution_i[0]));
    // Revolute unknown th_1
    solution_i[1] = std::atan2(std::sin(solution_i[1]), std::cos(solution_i[1]));
    // Revolute unknown th_2
    solution_i[2] = std::atan2(std::sin(solution_i[2]), std::cos(solution_i[2]));
    // Revolute unknown th_3
    solution_i[3] = std::atan2(std::sin(solution_i[3]), std::cos(solution_i[3]));
    // Revolute unknown th_4
    solution_i[4] = std::atan2(std::sin(solution_i[4]), std::cos(solution_i[4]));
    // Revolute unknown th_5
    solution_i[5] = std::atan2(std::sin(solution_i[5]), std::cos(solution_i[5]));
}

static void wrapAngleToPi(std::vector<std::array<double, robot_nq>>& ik_output)
{
    for (auto& solution_i : ik_output)
        wrapAngleToPi(solution_i);
}

static void computeIKForRawTarget(const Eigen::Matrix4d& T_ee, const Eigen::Matrix4d& T_ee_raw, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output, Eigen::Matrix4d& frame_disturbance, bool& frame_disturbance_ready, bool allow_disturbance, IkRefinementSeeds<robot_nq>* refinement_seeds = nullptr)
{
    if (!T_ee.allFinite()) { ik_output.clear(); return; }
    if (!T_ee_raw.allFinite()) { ik_output.clear(); return; }
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw;
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
        ik_out_i[4] = addRevoluteOffset(ik_out_i[4], -th_4_offset_original2raw);
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
        wrapAngleToPi(ik_out_i);
        const Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
        if (refinement_seeds)
        {
            const double refinement_error = (ee_pose_i-T_ee).cwiseAbs().maxCoeff();
            if (refinement_error < 1e-7) { refinement_seeds->strict_witness=true;refinement_seeds->clear(); }
            else if (!refinement_seeds->strict_witness) retainRefinementSeed<robot_nq>(*refinement_seeds, ik_out_i, refinement_error);
        }
        if (ee_pose_diff < pose_tolerance)
            ik_output.push_back(ik_out_i);
        std::array<double, robot_nq> snapped;
        if (snapNearJointLimits<robot_nq>(ik_out_i, snapped, jointLower(), jointUpper(), periodicLimitJoints(), fixedJoints()))
        {
            const Eigen::Matrix4d snapped_pose = computeFK(snapped);
            if ((snapped_pose.block<3, 1>(0, 3)-T_ee.block<3, 1>(0, 3)).norm() <= 1e-7 && (snapped_pose.block<3, 3>(0, 0)-T_ee.block<3, 3>(0, 0)).cwiseAbs().maxCoeff() <= 1e-7)
            {
                ik_output.push_back(snapped);
                if (refinement_seeds && (snapped_pose-T_ee).cwiseAbs().maxCoeff()<1e-7)
                    { refinement_seeds->strict_witness=true;refinement_seeds->clear(); }
            }
        }
    }

    if (!ik_output.empty())
    {
        removeDuplicateWithinClasses<robot_nq>(ik_output, zero_tolerance, ikCandidateClasses<robot_nq>(ik_output, jointLower(), jointUpper(), periodicLimitJoints(), [&](const std::array<double, robot_nq>& candidate) { return computeFK(candidate); }, T_ee));
        return;
    }

    // Disturbing method for degenerate handling
    if (!allow_disturbance) return;
    for (int perturbation = 0; perturbation < 3; ++perturbation)
    {
        Eigen::Matrix4d T_ee_raw_disturbed;
        if (perturbation != 0)
            T_ee_raw_disturbed = yaik_cpp::disturbTransform<double>(T_ee_raw, 1e-2, 1e-2, nullptr, perturbation-1);
        else
        {
            T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);
        }
        Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
        computeRawIK(T_ee_raw_disturbed, workspace);
        const auto& raw_ik_out_disturb = workspace.raw_ik_out;
        for(int i = 0; i < raw_ik_out_disturb.size(); i++)
        {
            auto ik_out_i = raw_ik_out_disturb[i];
            ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
            ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
            ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw;
            ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
            ik_out_i[4] = addRevoluteOffset(ik_out_i[4], -th_4_offset_original2raw);
            ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
            wrapAngleToPi(ik_out_i);
            Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
            double ee_pose_diff = (ee_pose_i - T_ee_disturbed).squaredNorm();
            if (!std::isfinite(ee_pose_diff) || ee_pose_diff > pose_tolerance_degenerate)
                continue;

            // Try numerical refinement
            yaik_cpp::numericalRefinement<robot_nq>(computeFK, computeTwistJacobian, T_ee, ik_out_i);
            wrapAngleToPi(ik_out_i);
            ee_pose_i = computeFK(ik_out_i);
            ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
            if (ee_pose_diff < pose_tolerance_degenerate)
            {
                ik_output.push_back(ik_out_i);
                if (refinement_seeds && (ee_pose_i-T_ee).cwiseAbs().maxCoeff()<1e-7)
                    { refinement_seeds->strict_witness=true;refinement_seeds->clear(); }
            }
        }
    }

    removeDuplicateWithinClasses<robot_nq>(ik_output, zero_tolerance, ikCandidateClasses<robot_nq>(ik_output, jointLower(), jointUpper(), periodicLimitJoints(), [&](const std::array<double, robot_nq>& candidate) { return computeFK(candidate); }, T_ee));
}

static void computeIK(const Eigen::Matrix4d& T_ee, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    ik_output.clear();
    if (!T_ee.allFinite()) return;
    Eigen::Matrix4d frame_disturbance;
    bool frame_disturbance_ready = false;
    IkRefinementSeeds<robot_nq> refinement_seeds;
    for (bool legacy : {false})
    {
        const Eigen::Matrix4d raw_target = legacy ? endEffectorTargetOriginalToRaw(T_ee) : endEffectorTargetOriginalToRaw(T_ee);
        std::vector<std::array<double, robot_nq>> candidates;
        computeIKForRawTarget(T_ee, raw_target, workspace, candidates, frame_disturbance, frame_disturbance_ready, legacy || ik_output.empty(), &refinement_seeds);
        for (const auto& candidate : candidates)
        {
            if (std::find(ik_output.begin(), ik_output.end(), candidate) == ik_output.end())
                ik_output.push_back(candidate);
        }
    }
    auto refinement_fk = [&](const std::array<double,robot_nq>& q) { return computeFK(q); };
    if (!refinement_seeds.strict_witness)
    {
        for (const auto& entry : refinement_seeds)
        {
            auto candidate=entry.second;
            if (refineIkSeed<robot_nq>(refinement_fk,T_ee,candidate,fixedJoints()))
            {
                ik_output.push_back(candidate);
                refinement_seeds.strict_witness=true;
                break;
            }
        }
    }
    if (!refinement_seeds.strict_witness)
    {
        refinement_seeds.clear();
        std::array<double,0> seed_parameters = {{}};
        for (const auto& shifted : nearbyIkTargets(T_ee))
        {
            if (!shifted.allFinite()) continue;
            for (bool legacy : {false})
            {
                const Eigen::Matrix4d raw_target = legacy ? endEffectorTargetOriginalToRaw(shifted) : endEffectorTargetOriginalToRaw(shifted);
                computeRawIK(raw_target, workspace);
                for (auto candidate : workspace.raw_ik_out)
                {
                    candidate[0] = candidate[0] - th_0_offset_original2raw;
                    candidate[1] = addRevoluteOffset(candidate[1], -th_1_offset_original2raw);
                    candidate[2] = candidate[2] - th_2_offset_original2raw;
                    candidate[3] = addRevoluteOffset(candidate[3], -th_3_offset_original2raw);
                    candidate[4] = addRevoluteOffset(candidate[4], -th_4_offset_original2raw);
                    candidate[5] = addRevoluteOffset(candidate[5], -th_5_offset_original2raw);
                    wrapAngleToPi(candidate);
                    retainRefinementSeed<robot_nq>(refinement_seeds,candidate,(refinement_fk(candidate)-T_ee).cwiseAbs().maxCoeff());
                }
            }
        }
        for (const auto& entry : refinement_seeds)
        {
            auto candidate=entry.second;
            if (refineIkSeed<robot_nq>(refinement_fk,T_ee,candidate,fixedJoints()))
            {
                ik_output.push_back(candidate);refinement_seeds.strict_witness=true;break;
            }
        }
    }
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct yaskawa_HC10_ik

// Code below for debug
void test_ik_solve_yaskawa_HC10()
{
    std::array<double, yaskawa_HC10_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = yaskawa_HC10_ik::computeFK(theta);
    auto ik_output = yaskawa_HC10_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = yaskawa_HC10_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_yaskawa_HC10();
    return 0;
}
