#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct abb_crb_15000_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 1152;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -1.5707963267948966, -3.141592653589793, -6.283185307179586, -6.283185307179586, -3.141592653589793}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 4.71238898038469, 3.141592653589793, 0.0, 0.0, 3.141592653589793}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 28;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_0 = 0.444;
static constexpr double a_1 = 0.11;
static constexpr double a_3 = 0.08;
static constexpr double d_2 = 0.47;
static constexpr double d_4 = 0.101;
static constexpr double pre_transform_special_symbol_23 = 0.265;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 0.0;

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
    ee_transformed(0, 1) = -1.0*r_12;
    ee_transformed(0, 2) = 1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_23;
    ee_transformed(1, 1) = -1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = 1.0*r_33;
    ee_transformed(2, 1) = -1.0*r_32;
    ee_transformed(2, 2) = 1.0*r_31;
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
    ee_transformed(0, 0) = 1.0*r_13;
    ee_transformed(0, 1) = -1.0*r_12;
    ee_transformed(0, 2) = 1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_23;
    ee_transformed(1, 1) = -1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = 1.0*r_33;
    ee_transformed(2, 1) = -1.0*r_32;
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
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;

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
    const double x10 = -x4*x7 + x4*x8*x9;
    const double x11 = x1*x2 - x10*x3;
    const double x12 = std::cos(th_5);
    const double x13 = std::sin(th_4);
    const double x14 = x5*x9;
    const double x15 = x6*x8;
    const double x16 = -x14*x4 - x15*x4;
    const double x17 = std::cos(th_4);
    const double x18 = x1*x3 + x10*x2;
    const double x19 = -x13*x16 + x17*x18;
    const double x20 = -x13*x18 - x16*x17;
    const double x21 = a_0*x8;
    const double x22 = -x1*x7 + x1*x8*x9;
    const double x23 = -x2*x4 - x22*x3;
    const double x24 = -x1*x14 - x1*x15;
    const double x25 = x2*x22 - x3*x4;
    const double x26 = -x13*x24 + x17*x25;
    const double x27 = -x13*x25 - x17*x24;
    const double x28 = -x14 - x15;
    const double x29 = x7 - x8*x9;
    const double x30 = -x13*x29 + x17*x2*x28;
    const double x31 = -x13*x2*x28 - x17*x29;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x11 + x12*x19;
    ee_pose_raw(0, 1) = -x0*x19 - x11*x12;
    ee_pose_raw(0, 2) = x20;
    ee_pose_raw(0, 3) = a_1*x10 + a_3*x19 + d_2*x16 + d_4*x20 + x21*x4;
    ee_pose_raw(1, 0) = -x0*x23 + x12*x26;
    ee_pose_raw(1, 1) = -x0*x26 - x12*x23;
    ee_pose_raw(1, 2) = x27;
    ee_pose_raw(1, 3) = a_1*x22 + a_3*x26 + d_2*x24 + d_4*x27 + x1*x21;
    ee_pose_raw(2, 0) = x0*x28*x3 + x12*x30;
    ee_pose_raw(2, 1) = -x0*x30 + x12*x28*x3;
    ee_pose_raw(2, 2) = x31;
    ee_pose_raw(2, 3) = -a_0*x5 + a_1*x28 + a_3*x30 + d_2*x29 + d_4*x31;
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
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = -x1;
    const double x3 = std::cos(th_2);
    const double x4 = std::sin(th_1);
    const double x5 = std::cos(th_0);
    const double x6 = 1.0*x5;
    const double x7 = x4*x6;
    const double x8 = std::sin(th_2);
    const double x9 = std::cos(th_1);
    const double x10 = x6*x9;
    const double x11 = -x10*x8 - x3*x7;
    const double x12 = std::cos(th_3);
    const double x13 = std::sin(th_3);
    const double x14 = 1.0*x3*x5*x9 - x7*x8;
    const double x15 = 1.0*x0*x12 - x13*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x1*x13 + x12*x14;
    const double x19 = -x11*x16 - x17*x18;
    const double x20 = x1*x4;
    const double x21 = x1*x9;
    const double x22 = -x20*x3 - x21*x8;
    const double x23 = 1.0*x0*x3*x9 - x20*x8;
    const double x24 = -x12*x6 - x13*x23;
    const double x25 = x12*x23 - x13*x6;
    const double x26 = -x16*x22 - x17*x25;
    const double x27 = 1.0*x8;
    const double x28 = 1.0*x3;
    const double x29 = x27*x4 - x28*x9;
    const double x30 = -x27*x9 - x28*x4;
    const double x31 = x13*x30;
    const double x32 = -x12*x17*x30 - x16*x29;
    const double x33 = 1.0*a_0;
    const double x34 = x33*x4;
    const double x35 = pre_transform_special_symbol_23 - x34;
    const double x36 = a_1*x30 + d_2*x29 + pre_transform_special_symbol_23 - x34;
    const double x37 = a_0*x21 + a_1*x23 + d_2*x22;
    const double x38 = a_3*(x12*x16*x30 - x17*x29) + d_4*x32 + x36;
    const double x39 = a_3*(x16*x25 - x17*x22) + d_4*x26 + x37;
    const double x40 = a_0*x10 + a_1*x14 + d_2*x11;
    const double x41 = a_3*(-x11*x17 + x16*x18) + d_4*x19 + x40;
    const double x42 = x33*x9;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x11;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x19;
    jacobian(1, 1) = x6;
    jacobian(1, 2) = x6;
    jacobian(1, 3) = x22;
    jacobian(1, 4) = x24;
    jacobian(1, 5) = x26;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x29;
    jacobian(2, 4) = -x31;
    jacobian(2, 5) = x32;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x6;
    jacobian(3, 2) = -x35*x6;
    jacobian(3, 3) = -x22*x36 + x29*x37;
    jacobian(3, 4) = -x24*x36 - x31*x37;
    jacobian(3, 5) = -x26*x38 + x32*x39;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = -x1*x35;
    jacobian(4, 3) = x11*x36 - x29*x40;
    jacobian(4, 4) = x15*x36 + x31*x40;
    jacobian(4, 5) = x19*x38 - x32*x41;
    jacobian(5, 2) = std::pow(x0, 2)*x42 + x42*std::pow(x5, 2);
    jacobian(5, 3) = -x11*x37 + x22*x40;
    jacobian(5, 4) = -x15*x37 + x24*x40;
    jacobian(5, 5) = -x19*x39 + x26*x41;
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
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = -x1;
    const double x3 = std::cos(th_2);
    const double x4 = std::sin(th_1);
    const double x5 = std::cos(th_0);
    const double x6 = 1.0*x5;
    const double x7 = x4*x6;
    const double x8 = std::sin(th_2);
    const double x9 = std::cos(th_1);
    const double x10 = -x3*x7 - x6*x8*x9;
    const double x11 = std::cos(th_3);
    const double x12 = std::sin(th_3);
    const double x13 = 1.0*x3*x5*x9 - x7*x8;
    const double x14 = std::cos(th_4);
    const double x15 = std::sin(th_4);
    const double x16 = x1*x4;
    const double x17 = -x1*x8*x9 - x16*x3;
    const double x18 = 1.0*x0*x3*x9 - x16*x8;
    const double x19 = 1.0*x8;
    const double x20 = 1.0*x3;
    const double x21 = x19*x4 - x20*x9;
    const double x22 = -x19*x9 - x20*x4;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = 1.0*x0*x11 - x12*x13;
    jacobian(0, 5) = -x10*x14 - x15*(x1*x12 + x11*x13);
    jacobian(1, 1) = x6;
    jacobian(1, 2) = x6;
    jacobian(1, 3) = x17;
    jacobian(1, 4) = -x11*x6 - x12*x18;
    jacobian(1, 5) = -x14*x17 - x15*(x11*x18 - x12*x6);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x21;
    jacobian(2, 4) = -x12*x22;
    jacobian(2, 5) = -x11*x15*x22 - x14*x21;
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
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
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
    const double x7 = pre_transform_special_symbol_23 - x6;
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = std::cos(th_2);
    const double x11 = std::cos(th_1);
    const double x12 = 1.0*x11;
    const double x13 = -x10*x12 + x9;
    const double x14 = std::sin(th_0);
    const double x15 = x10*x5;
    const double x16 = x12*x8;
    const double x17 = -x14*x15 - x14*x16;
    const double x18 = -x15 - x16;
    const double x19 = a_1*x18 + d_2*x13 + pre_transform_special_symbol_23 - x6;
    const double x20 = a_0*x12;
    const double x21 = 1.0*x10*x11*x14 - x14*x9;
    const double x22 = a_1*x21 + d_2*x17 + x14*x20;
    const double x23 = std::sin(th_3);
    const double x24 = x18*x23;
    const double x25 = std::cos(th_3);
    const double x26 = -x2*x25 - x21*x23;
    const double x27 = std::cos(th_4);
    const double x28 = std::sin(th_4);
    const double x29 = -x13*x27 - x18*x25*x28;
    const double x30 = -x2*x23 + x21*x25;
    const double x31 = -x17*x27 - x28*x30;
    const double x32 = a_3*(-x13*x28 + x18*x25*x27) + d_4*x29 + x19;
    const double x33 = a_3*(-x17*x28 + x27*x30) + d_4*x31 + x22;
    const double x34 = 1.0*p_on_ee_x;
    const double x35 = 1.0*x14;
    const double x36 = p_on_ee_z*x35;
    const double x37 = x2*x4;
    const double x38 = x11*x2;
    const double x39 = -x10*x37 - x38*x8;
    const double x40 = 1.0*x1*x10*x11 - x37*x8;
    const double x41 = a_0*x38 + a_1*x40 + d_2*x39;
    const double x42 = 1.0*x14*x25 - x23*x40;
    const double x43 = x23*x35 + x25*x40;
    const double x44 = -x27*x39 - x28*x43;
    const double x45 = a_3*(x27*x43 - x28*x39) + d_4*x44 + x41;
    const double x46 = x1*x34;
    const double x47 = x0*x14;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x7 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19;
    jacobian(0, 4) = p_on_ee_y*x24 + p_on_ee_z*x26 - x19*x26 - x22*x24;
    jacobian(0, 5) = -p_on_ee_y*x29 + p_on_ee_z*x31 + x29*x33 - x31*x32;
    jacobian(1, 0) = x34;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x35 + x36;
    jacobian(1, 2) = -x35*x7 + x36;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x39 - x13*x41 + x19*x39;
    jacobian(1, 4) = -p_on_ee_x*x24 - p_on_ee_z*x42 + x18*x23*x41 + x19*x42;
    jacobian(1, 5) = p_on_ee_x*x29 - p_on_ee_z*x44 - x29*x45 + x32*x44;
    jacobian(2, 1) = -x46 - x47;
    jacobian(2, 2) = std::pow(x1, 2)*x20 + std::pow(x14, 2)*x20 - x46 - x47;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x39 + x17*x41 - x22*x39;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x42 - x22*x42 + x26*x41;
    jacobian(2, 5) = -p_on_ee_x*x31 + p_on_ee_y*x44 + x31*x45 - x33*x44;
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
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 48, 48, 96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384}};
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

    // Code for general_6dof solution node 1, solved variable is th_0
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;

        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 3) = -a_0;
        R_l(0, 7) = -a_1;
        R_l(1, 2) = -a_0;
        R_l(1, 6) = -a_1;
        R_l(2, 4) = a_0;
        R_l(3, 6) = -1;
        R_l(4, 7) = 1;
        R_l(5, 5) = 2*a_0*a_1;
        R_l(6, 1) = -a_0;
        R_l(7, 0) = -a_0;
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
        const double x0 = -r_23;
        const double x1 = 2*r_21;
        const double x2 = -x1;
        const double x3 = 4*r_22;
        const double x4 = a_3*r_21;
        const double x5 = d_2*r_23;
        const double x6 = std::pow(r_21, 2);
        const double x7 = Py*x6;
        const double x8 = std::pow(r_22, 2);
        const double x9 = Py*x8;
        const double x10 = std::pow(r_23, 2);
        const double x11 = Py*x10;
        const double x12 = d_4*r_23;
        const double x13 = Px*r_11;
        const double x14 = r_21*x13;
        const double x15 = Px*r_12;
        const double x16 = r_22*x15;
        const double x17 = Px*r_13;
        const double x18 = r_23*x17;
        const double x19 = Pz*r_31;
        const double x20 = r_21*x19;
        const double x21 = Pz*r_32;
        const double x22 = r_22*x21;
        const double x23 = Pz*r_33;
        const double x24 = r_23*x23;
        const double x25 = x11 - x12 + x14 + x16 + x18 + x20 + x22 + x24 + x7 + x9;
        const double x26 = x25 - x5;
        const double x27 = x26 + x4;
        const double x28 = a_3*r_22;
        const double x29 = 2*x28;
        const double x30 = -x4;
        const double x31 = x26 + x30;
        const double x32 = d_2*x1;
        const double x33 = -x32;
        const double x34 = d_2*x3;
        const double x35 = x25 + x5;
        const double x36 = x35 + x4;
        const double x37 = x30 + x35;
        const double x38 = R_l_inv_50*a_0;
        const double x39 = a_3*x38;
        const double x40 = Py*r_21;
        const double x41 = x13 + x19 + x40;
        const double x42 = x38*x41;
        const double x43 = Py*r_23;
        const double x44 = -d_4 + x17 + x23 + x43;
        const double x45 = R_l_inv_52*a_0;
        const double x46 = x44*x45;
        const double x47 = d_4*r_21;
        const double x48 = r_23*x13;
        const double x49 = r_23*x19;
        const double x50 = r_21*x17;
        const double x51 = r_21*x23;
        const double x52 = x47 + x48 + x49 - x50 - x51;
        const double x53 = R_l_inv_57*a_0;
        const double x54 = x52*x53;
        const double x55 = a_0*r_22;
        const double x56 = R_l_inv_54*x55;
        const double x57 = -x56;
        const double x58 = d_2*r_22;
        const double x59 = R_l_inv_56*a_0;
        const double x60 = x58*x59;
        const double x61 = a_3*r_23;
        const double x62 = x53*x61;
        const double x63 = R_l_inv_55*a_0;
        const double x64 = 2*d_2;
        const double x65 = x44*x64;
        const double x66 = x63*x65;
        const double x67 = x39 + x42 + x46 + x54 + x57 + x60 + x62 + x66;
        const double x68 = a_0*r_21;
        const double x69 = R_l_inv_53*x68;
        const double x70 = d_4*r_22;
        const double x71 = r_23*x15;
        const double x72 = r_23*x21;
        const double x73 = r_22*x17;
        const double x74 = r_22*x23;
        const double x75 = x70 + x71 + x72 - x73 - x74;
        const double x76 = x59*x75;
        const double x77 = R_l_inv_57*d_2*x68;
        const double x78 = r_22*x13;
        const double x79 = r_22*x19;
        const double x80 = r_21*x15;
        const double x81 = r_21*x21;
        const double x82 = x78 + x79 - x80 - x81;
        const double x83 = x28 + x69 + x76 + x77 + x82;
        const double x84 = std::pow(a_3, 2);
        const double x85 = std::pow(d_2, 2);
        const double x86 = std::pow(d_4, 2);
        const double x87 = std::pow(a_0, 2);
        const double x88 = std::pow(a_1, 2);
        const double x89 = 2*d_4;
        const double x90 = 2*x12;
        const double x91 = Py*x1;
        const double x92 = Py*r_22;
        const double x93 = 2*x15;
        const double x94 = 2*x17;
        const double x95 = 2*x13;
        const double x96 = 2*x21;
        const double x97 = 2*x23;
        const double x98 = std::pow(Px, 2);
        const double x99 = std::pow(r_11, 2);
        const double x100 = x98*x99;
        const double x101 = std::pow(r_12, 2);
        const double x102 = x101*x98;
        const double x103 = std::pow(r_13, 2);
        const double x104 = x103*x98;
        const double x105 = std::pow(Py, 2);
        const double x106 = x105*x6;
        const double x107 = x105*x8;
        const double x108 = x10*x105;
        const double x109 = std::pow(Pz, 2);
        const double x110 = std::pow(r_31, 2);
        const double x111 = x109*x110;
        const double x112 = std::pow(r_32, 2);
        const double x113 = x109*x112;
        const double x114 = std::pow(r_33, 2)*x109;
        const double x115 = -Py*x90 + x100 + x102 + x104 + x106 + x107 + x108 + x111 + x113 + x114 + x13*x91 - x17*x89 + x19*x91 + x19*x95 + x21*x93 - x23*x89 + x23*x94 + x43*x94 + x43*x97 + x84 + x85 + x86 - x87 - x88 + x92*x93 + x92*x96;
        const double x116 = -R_l_inv_52*a_0*d_2 - R_l_inv_55*a_0*x115 - a_1;
        const double x117 = x15 + x21 + x92;
        const double x118 = a_0*x117;
        const double x119 = R_l_inv_51*x118;
        const double x120 = 2*a_3;
        const double x121 = x41*x63;
        const double x122 = x120*x121;
        const double x123 = -x119 - x122;
        const double x124 = 2*R_l_inv_50*x118;
        const double x125 = 2*x41;
        const double x126 = R_l_inv_51*a_0*x125;
        const double x127 = -4*R_l_inv_55*a_0*a_3*x117 + x124 + x126;
        const double x128 = R_l_inv_54*a_0;
        const double x129 = x1*x128;
        const double x130 = 2*x53;
        const double x131 = x130*x75;
        const double x132 = x32*x59;
        const double x133 = x129 + x131 - x132;
        const double x134 = a_3*x1;
        const double x135 = 2*x59;
        const double x136 = x135*x52;
        const double x137 = 2*R_l_inv_53;
        const double x138 = x137*x55;
        const double x139 = x130*x58;
        const double x140 = -x134 - x136 + x138 + x139;
        const double x141 = d_2*x45;
        const double x142 = x115*x63;
        const double x143 = -x62;
        const double x144 = a_1 + x141 + x142 + x143 + x42;
        const double x145 = -x39;
        const double x146 = -x46;
        const double x147 = -x66;
        const double x148 = x145 + x146 + x147 + x54 + x57 + x60;
        const double x149 = -x78 - x79 + x80 + x81;
        const double x150 = x149 + x28 + x69 + x76 + x77;
        const double x151 = x123 + x150;
        const double x152 = a_0*r_23*x137;
        const double x153 = x135*x82;
        const double x154 = x130*x5;
        const double x155 = -2*R_l_inv_50*a_0*x44;
        const double x156 = x120*x45;
        const double x157 = x125*x45;
        const double x158 = 4*a_3;
        const double x159 = d_2*x158*x63;
        const double x160 = 4*d_2;
        const double x161 = x121*x160;
        const double x162 = x155 + x156 + x157 + x159 + x161;
        const double x163 = 2*x70;
        const double x164 = 2*x73;
        const double x165 = 2*x74;
        const double x166 = 2*x71;
        const double x167 = 2*x72;
        const double x168 = x29*x59;
        const double x169 = x163 - x164 - x165 + x166 + x167 - x168;
        const double x170 = 4*x47;
        const double x171 = 4*x48;
        const double x172 = 4*x49;
        const double x173 = 4*x50;
        const double x174 = 4*x51;
        const double x175 = 4*x4;
        const double x176 = x175*x59;
        const double x177 = 8*d_2;
        const double x178 = R_l_inv_55*x118;
        const double x179 = 4*R_l_inv_52*x118 + x177*x178;
        const double x180 = x152 + x153 + x154;
        const double x181 = x169 + x180;
        const double x182 = x119 + x122;
        const double x183 = x182 + x83;
        const double x184 = -x54 + x56 - x60;
        const double x185 = x39 + x46 + x66;
        const double x186 = x184 + x185;
        const double x187 = -x129 - x131 + x132;
        const double x188 = x124 - x126 + x158*x178;
        const double x189 = x116 + x150 + x182;
        const double x190 = a_0*a_1;
        const double x191 = 2*x190;
        const double x192 = x87 + x88;
        const double x193 = R_l_inv_62*x192;
        const double x194 = R_l_inv_22*x191 + x193;
        const double x195 = d_2*x194;
        const double x196 = r_21*x84;
        const double x197 = std::pow(r_21, 3)*x105;
        const double x198 = R_l_inv_25*x191 + R_l_inv_65*x192;
        const double x199 = x115*x198;
        const double x200 = -r_21*x85;
        const double x201 = -r_21*x86;
        const double x202 = R_l_inv_23*x191 + R_l_inv_63*x192;
        const double x203 = -r_21*x202;
        const double x204 = R_l_inv_60*x192;
        const double x205 = R_l_inv_20*x191 + x204;
        const double x206 = x205*x41;
        const double x207 = -x206;
        const double x208 = R_l_inv_66*x192;
        const double x209 = R_l_inv_26*x191 + x208;
        const double x210 = -x209*x75;
        const double x211 = r_21*x100;
        const double x212 = r_21*x107;
        const double x213 = r_21*x108;
        const double x214 = r_21*x111;
        const double x215 = R_l_inv_67*x192;
        const double x216 = R_l_inv_27*x191 + x215;
        const double x217 = x216*x61;
        const double x218 = -x217;
        const double x219 = d_2*r_21;
        const double x220 = -x216*x219;
        const double x221 = -r_21*x102;
        const double x222 = -r_21*x104;
        const double x223 = -r_21*x113;
        const double x224 = -r_21*x114;
        const double x225 = x120*x5;
        const double x226 = -x225;
        const double x227 = -x13*x90;
        const double x228 = -x19*x90;
        const double x229 = x7*x95;
        const double x230 = x9*x95;
        const double x231 = x11*x95;
        const double x232 = d_4*x1;
        const double x233 = x17*x232;
        const double x234 = 2*x19;
        const double x235 = x234*x7;
        const double x236 = x234*x9;
        const double x237 = x11*x234;
        const double x238 = x23*x232;
        const double x239 = 2*r_11;
        const double x240 = r_12*x98;
        const double x241 = r_22*x240;
        const double x242 = x239*x241;
        const double x243 = 2*r_13;
        const double x244 = r_23*x98;
        const double x245 = r_11*x243*x244;
        const double x246 = 2*r_31;
        const double x247 = r_32*x109;
        const double x248 = r_22*x247;
        const double x249 = x246*x248;
        const double x250 = r_23*r_33;
        const double x251 = x109*x250;
        const double x252 = x246*x251;
        const double x253 = x15*x21;
        const double x254 = -x1*x253;
        const double x255 = x17*x23;
        const double x256 = -x1*x255;
        const double x257 = x13*x19;
        const double x258 = x1*x257;
        const double x259 = x22*x95;
        const double x260 = x24*x95;
        const double x261 = x16*x234;
        const double x262 = x18*x234;
        const double x263 = x195 + x196 + x197 + x199 + x200 + x201 + x203 + x207 + x210 + x211 + x212 + x213 + x214 + x218 + x220 + x221 + x222 + x223 + x224 + x226 + x227 + x228 + x229 + x230 + x231 + x233 + x235 + x236 + x237 + x238 + x242 + x245 + x249 + x252 + x254 + x256 + x258 + x259 + x260 + x261 + x262;
        const double x264 = R_l_inv_61*x192;
        const double x265 = x117*(R_l_inv_21*x191 + x264);
        const double x266 = a_3*x205;
        const double x267 = -x266;
        const double x268 = x194*x44;
        const double x269 = -x268;
        const double x270 = x198*x65;
        const double x271 = -x270;
        const double x272 = x198*x41;
        const double x273 = x120*x272;
        const double x274 = x265 + x267 + x269 + x271 + x273;
        const double x275 = R_l_inv_24*x191 + R_l_inv_64*x192;
        const double x276 = r_22*x275;
        const double x277 = x216*x52;
        const double x278 = x209*x58;
        const double x279 = d_4*x32;
        const double x280 = x5*x95;
        const double x281 = x234*x5;
        const double x282 = x17*x32;
        const double x283 = x23*x32;
        const double x284 = x276 - x277 - x278 - x279 - x280 - x281 + x282 + x283;
        const double x285 = x12*x120;
        const double x286 = x120*x7;
        const double x287 = x120*x9;
        const double x288 = x11*x120;
        const double x289 = x13*x134;
        const double x290 = x15*x29;
        const double x291 = x120*x18;
        const double x292 = x134*x19;
        const double x293 = x21*x29;
        const double x294 = x120*x24;
        const double x295 = -x285 + x286 + x287 + x288 + x289 + x290 + x291 + x292 + x293 + x294;
        const double x296 = x284 + x295;
        const double x297 = 4*x190;
        const double x298 = R_l_inv_26*x297 + 2*x208;
        const double x299 = x298*x52;
        const double x300 = R_l_inv_20*x297 + 2*x204;
        const double x301 = x117*x300;
        const double x302 = r_22*x85;
        const double x303 = -2*x302;
        const double x304 = r_22*x86;
        const double x305 = -2*x304;
        const double x306 = 2*x202;
        const double x307 = -r_22*x306;
        const double x308 = r_22*x84;
        const double x309 = 2*x308;
        const double x310 = std::pow(r_22, 3)*x105;
        const double x311 = 2*x310;
        const double x312 = 2*x216;
        const double x313 = -x312*x58;
        const double x314 = r_22*x100;
        const double x315 = -2*x314;
        const double x316 = r_22*x104;
        const double x317 = -2*x316;
        const double x318 = r_22*x111;
        const double x319 = -2*x318;
        const double x320 = r_22*x114;
        const double x321 = -2*x320;
        const double x322 = r_22*x102;
        const double x323 = 2*x322;
        const double x324 = r_22*x106;
        const double x325 = 2*x324;
        const double x326 = r_22*x108;
        const double x327 = 2*x326;
        const double x328 = r_22*x113;
        const double x329 = 2*x328;
        const double x330 = 4*x15;
        const double x331 = -x12*x330;
        const double x332 = 4*x21;
        const double x333 = -x12*x332;
        const double x334 = x330*x7;
        const double x335 = x330*x9;
        const double x336 = x11*x330;
        const double x337 = d_4*x3;
        const double x338 = x17*x337;
        const double x339 = x332*x7;
        const double x340 = x332*x9;
        const double x341 = x11*x332;
        const double x342 = x23*x337;
        const double x343 = 4*r_11;
        const double x344 = r_21*x343;
        const double x345 = x240*x344;
        const double x346 = 4*r_12;
        const double x347 = r_13*x244;
        const double x348 = x346*x347;
        const double x349 = 4*x247;
        const double x350 = r_21*r_31;
        const double x351 = x349*x350;
        const double x352 = x250*x349;
        const double x353 = -x257*x3;
        const double x354 = -x255*x3;
        const double x355 = x14*x332;
        const double x356 = x20*x330;
        const double x357 = x253*x3;
        const double x358 = x24*x330;
        const double x359 = x18*x332;
        const double x360 = x299 - x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x325 + x327 + x329 + x331 + x333 + x334 + x335 + x336 + x338 + x339 + x340 + x341 + x342 + x345 + x348 + x351 + x352 + x353 + x354 + x355 + x356 + x357 + x358 + x359;
        const double x361 = x41*(R_l_inv_21*x297 + 2*x264);
        const double x362 = x158*x198;
        const double x363 = x117*x362 - x361;
        const double x364 = R_l_inv_27*x297 + 2*x215;
        const double x365 = x364*x75;
        const double x366 = x1*x275;
        const double x367 = d_4*x34;
        const double x368 = x209*x32;
        const double x369 = x330*x5;
        const double x370 = x332*x5;
        const double x371 = x17*x34;
        const double x372 = x23*x34;
        const double x373 = -x365 - x366 - x367 + x368 - x369 - x370 + x371 + x372;
        const double x374 = -x286;
        const double x375 = -x287;
        const double x376 = -x288;
        const double x377 = -x289;
        const double x378 = -x290;
        const double x379 = -x291;
        const double x380 = -x292;
        const double x381 = -x293;
        const double x382 = -x294;
        const double x383 = x266 + x268 + x270 + x285 + x374 + x375 + x376 + x377 + x378 + x379 + x380 + x381 + x382;
        const double x384 = -d_2*x194;
        const double x385 = -x115*x198;
        const double x386 = x196 + x197 + x200 + x201 + x203 + x207 + x210 + x211 + x212 + x213 + x214 + x217 + x220 + x221 + x222 + x223 + x224 + x225 + x227 + x228 + x229 + x230 + x231 + x233 + x235 + x236 + x237 + x238 + x242 + x245 + x249 + x252 + x254 + x256 + x258 + x259 + x260 + x261 + x262 + x384 + x385;
        const double x387 = x265 + x273;
        const double x388 = 4*d_4;
        const double x389 = x388*x4;
        const double x390 = x209*x29;
        const double x391 = x158*x48;
        const double x392 = x158*x49;
        const double x393 = x17*x175;
        const double x394 = x175*x23;
        const double x395 = x41*(R_l_inv_22*x297 + 2*x193);
        const double x396 = -x300*x44;
        const double x397 = x120*x194;
        const double x398 = d_2*x362;
        const double x399 = x160*x272;
        const double x400 = x395 + x396 + x397 + x398 + x399;
        const double x401 = x298*x82;
        const double x402 = r_23*x84;
        const double x403 = 2*x402;
        const double x404 = r_23*x85;
        const double x405 = 2*x404;
        const double x406 = r_23*x306;
        const double x407 = r_23*x86;
        const double x408 = 2*x407;
        const double x409 = std::pow(r_23, 3)*x105;
        const double x410 = 2*x409;
        const double x411 = x388*x7;
        const double x412 = x388*x9;
        const double x413 = x11*x388;
        const double x414 = x312*x5;
        const double x415 = r_23*x100;
        const double x416 = 2*x415;
        const double x417 = r_23*x102;
        const double x418 = 2*x417;
        const double x419 = r_23*x111;
        const double x420 = 2*x419;
        const double x421 = r_23*x113;
        const double x422 = 2*x421;
        const double x423 = r_23*x104;
        const double x424 = 2*x423;
        const double x425 = r_23*x106;
        const double x426 = 2*x425;
        const double x427 = r_23*x107;
        const double x428 = 2*x427;
        const double x429 = r_23*x114;
        const double x430 = 2*x429;
        const double x431 = x13*x170;
        const double x432 = x15*x337;
        const double x433 = 4*x17;
        const double x434 = x12*x433;
        const double x435 = x170*x19;
        const double x436 = x21*x337;
        const double x437 = 4*x23;
        const double x438 = x12*x437;
        const double x439 = x433*x7;
        const double x440 = x433*x9;
        const double x441 = x11*x433;
        const double x442 = x437*x7;
        const double x443 = x437*x9;
        const double x444 = x11*x437;
        const double x445 = r_13*x98;
        const double x446 = x344*x445;
        const double x447 = r_13*x3;
        const double x448 = x240*x447;
        const double x449 = r_33*x109;
        const double x450 = 4*x350*x449;
        const double x451 = x247*x3;
        const double x452 = r_33*x451;
        const double x453 = x171*x19;
        const double x454 = x332*x71;
        const double x455 = x14*x437;
        const double x456 = x15*x3;
        const double x457 = x23*x456;
        const double x458 = x20*x433;
        const double x459 = x21*x3;
        const double x460 = x17*x459;
        const double x461 = x18*x437;
        const double x462 = -x401 - x403 - x405 - x406 + x408 + x410 - x411 - x412 - x413 - x414 - x416 - x418 - x420 - x422 + x424 + x426 + x428 + x430 - x431 - x432 - x434 - x435 - x436 - x438 + x439 + x440 + x441 + x442 + x443 + x444 + x446 + x448 + x450 + x452 - x453 - x454 + x455 + x457 + x458 + x460 + x461;
        const double x463 = 8*d_4;
        const double x464 = x28*x463;
        const double x465 = x175*x209;
        const double x466 = 8*a_3;
        const double x467 = x466*x71;
        const double x468 = x466*x72;
        const double x469 = 8*x28;
        const double x470 = x17*x469;
        const double x471 = x23*x469;
        const double x472 = 8*x190;
        const double x473 = x117*x177;
        const double x474 = x117*(R_l_inv_22*x472 + 4*x193) + x198*x473;
        const double x475 = x389 + x390 + x391 + x392 - x393 - x394;
        const double x476 = x462 + x475;
        const double x477 = -x265 - x273;
        const double x478 = x267 + x269 + x271 + x477;
        const double x479 = -x276 + x277 + x278 + x279 + x280 + x281 - x282 - x283;
        const double x480 = x295 + x479;
        const double x481 = -4*a_3*x117*x198 + x361;
        const double x482 = x365 + x366 + x367 - x368 + x369 + x370 - x371 - x372;
        const double x483 = R_l_inv_72*x192;
        const double x484 = R_l_inv_32*x191 + x483;
        const double x485 = d_2*x484;
        const double x486 = R_l_inv_34*x191 + R_l_inv_74*x192;
        const double x487 = r_22*x486;
        const double x488 = R_l_inv_35*x191 + R_l_inv_75*x192;
        const double x489 = x115*x488;
        const double x490 = -x308;
        const double x491 = -x302;
        const double x492 = -x304;
        const double x493 = R_l_inv_70*x192;
        const double x494 = R_l_inv_30*x191 + x493;
        const double x495 = x41*x494;
        const double x496 = -x495;
        const double x497 = R_l_inv_77*x192;
        const double x498 = R_l_inv_37*x191 + x497;
        const double x499 = -x498*x52;
        const double x500 = R_l_inv_76*x192;
        const double x501 = R_l_inv_36*x191 + x500;
        const double x502 = -x501*x58;
        const double x503 = -x314;
        const double x504 = -x316;
        const double x505 = -x318;
        const double x506 = -x320;
        const double x507 = -x15*x90;
        const double x508 = -x21*x90;
        const double x509 = x7*x93;
        const double x510 = x9*x93;
        const double x511 = x11*x93;
        const double x512 = x163*x17;
        const double x513 = x7*x96;
        const double x514 = x9*x96;
        const double x515 = x11*x96;
        const double x516 = x163*x23;
        const double x517 = r_11*x240;
        const double x518 = x1*x517;
        const double x519 = r_23*x240*x243;
        const double x520 = r_31*x1;
        const double x521 = x247*x520;
        const double x522 = x247*x250;
        const double x523 = 2*x522;
        const double x524 = -x234*x78;
        const double x525 = -x164*x23;
        const double x526 = x1*x13;
        const double x527 = x21*x526;
        const double x528 = x1*x19;
        const double x529 = x15*x528;
        const double x530 = x16*x96;
        const double x531 = x24*x93;
        const double x532 = x18*x96;
        const double x533 = x310 + x322 + x324 + x326 + x328 + x485 + x487 + x489 + x490 + x491 + x492 + x496 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532;
        const double x534 = R_l_inv_71*x192;
        const double x535 = x117*(R_l_inv_31*x191 + x534);
        const double x536 = a_3*x494;
        const double x537 = -x536;
        const double x538 = x44*x484;
        const double x539 = -x538;
        const double x540 = x488*x65;
        const double x541 = -x540;
        const double x542 = x41*x488;
        const double x543 = x120*x542;
        const double x544 = x535 + x537 + x539 + x541 + x543;
        const double x545 = x498*x61;
        const double x546 = x13*x29;
        const double x547 = x19*x29;
        const double x548 = x134*x15;
        const double x549 = x134*x21;
        const double x550 = -x545 - x546 - x547 + x548 + x549;
        const double x551 = R_l_inv_33*x191 + R_l_inv_73*x192;
        const double x552 = r_21*x551;
        const double x553 = x501*x75;
        const double x554 = x219*x498;
        const double x555 = x64*x70;
        const double x556 = x5*x93;
        const double x557 = x5*x96;
        const double x558 = x58*x94;
        const double x559 = x58*x97;
        const double x560 = -x552 - x553 - x554 - x555 - x556 - x557 + x558 + x559;
        const double x561 = x550 + x560;
        const double x562 = R_l_inv_30*x297 + 2*x493;
        const double x563 = x117*x562;
        const double x564 = R_l_inv_37*x297 + 2*x497;
        const double x565 = x564*x75;
        const double x566 = x1*x84;
        const double x567 = -x566;
        const double x568 = x1*x85;
        const double x569 = -x568;
        const double x570 = x1*x486;
        const double x571 = -x32*x501;
        const double x572 = 4*x13;
        const double x573 = 4*x19;
        const double x574 = 4*r_31*x251 + r_31*x451 + x1*x100 - x1*x102 - x1*x104 + x1*x107 + x1*x108 + x1*x111 - x1*x113 - x1*x114 - x1*x86 + x11*x572 + x11*x573 - x12*x572 - x12*x573 + x13*x459 + x14*x573 + x17*x170 + x170*x23 - x173*x23 + x18*x573 + x19*x456 + 2*x197 + x24*x572 + x3*x517 - x332*x80 + x343*x347 + x572*x7 + x572*x9 + x573*x7 + x573*x9;
        const double x575 = x563 + x565 + x567 + x569 + x570 + x571 + x574;
        const double x576 = R_l_inv_31*x297 + 2*x534;
        const double x577 = x158*x488;
        const double x578 = x117*x577;
        const double x579 = x41*x576 - x578;
        const double x580 = R_l_inv_36*x297 + 2*x500;
        const double x581 = x52*x580;
        const double x582 = 2*x551;
        const double x583 = r_22*x582;
        const double x584 = d_2*x170;
        const double x585 = 2*x498;
        const double x586 = x58*x585;
        const double x587 = x5*x572;
        const double x588 = x5*x573;
        const double x589 = d_2*x173;
        const double x590 = d_2*x174;
        const double x591 = -x581 + x583 - x584 + x586 - x587 - x588 + x589 + x590;
        const double x592 = -x548;
        const double x593 = -x549;
        const double x594 = x536 + x538 + x540 + x545 + x546 + x547 + x592 + x593;
        const double x595 = -x485;
        const double x596 = -x489;
        const double x597 = x310 + x322 + x324 + x326 + x328 + x487 + x490 + x491 + x492 + x496 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532 + x595 + x596;
        const double x598 = x535 + x543;
        const double x599 = x580*x82;
        const double x600 = r_23*x582;
        const double x601 = x29*x501;
        const double x602 = x5*x585;
        const double x603 = a_3*x34;
        const double x604 = -x160*x80;
        const double x605 = -x160*x81;
        const double x606 = x13*x34;
        const double x607 = x19*x34;
        const double x608 = x599 + x600 + x601 + x602 + x603 + x604 + x605 + x606 + x607;
        const double x609 = x41*(R_l_inv_32*x297 + 2*x483);
        const double x610 = x160*x542;
        const double x611 = -x609 - x610;
        const double x612 = x44*x562;
        const double x613 = -4*a_3*d_2*x488 - 2*a_3*x484 + x612;
        const double x614 = R_l_inv_32*x472 + 4*x483;
        const double x615 = x175*x501 + x177*x4;
        const double x616 = x609 + x610;
        const double x617 = x599 + x600 - x601 + x602 - x603 + x604 + x605 + x606 + x607;
        const double x618 = x552 + x553 + x554 + x555 + x556 + x557 - x558 - x559;
        const double x619 = x550 + x618;
        const double x620 = x310 + x322 + x324 + x326 + x328 + x485 + x487 + x489 + x490 + x491 + x492 + x495 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532;
        const double x621 = x536 + x538 + x540;
        const double x622 = -x563 + x565 + x567 + x569 + x570 + x571 + x574;
        const double x623 = x581 - x583 + x584 - x586 + x587 + x588 - x589 - x590;
        const double x624 = x545 + x546 + x547 + x592 + x593;
        const double x625 = x310 + x322 + x324 + x326 + x328 + x487 + x490 + x491 + x492 + x495 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532 + x595 + x596;
        const double x626 = -2*d_2*d_4*r_23;
        const double x627 = x64*x7;
        const double x628 = x64*x9;
        const double x629 = x11*x64;
        const double x630 = a_3*x32;
        const double x631 = x13*x32;
        const double x632 = x58*x93;
        const double x633 = x5*x94;
        const double x634 = x19*x32;
        const double x635 = x58*x96;
        const double x636 = x5*x97;
        const double x637 = x626 + x627 + x628 + x629 + x630 + x631 + x632 + x633 + x634 + x635 + x636;
        const double x638 = d_4*x134;
        const double x639 = x134*x17;
        const double x640 = x134*x23;
        const double x641 = x120*x48;
        const double x642 = x120*x49;
        const double x643 = x638 - x639 - x640 + x641 + x642;
        const double x644 = x7*x89;
        const double x645 = x89*x9;
        const double x646 = x11*x89;
        const double x647 = x7*x94;
        const double x648 = x9*x94;
        const double x649 = x11*x94;
        const double x650 = x7*x97;
        const double x651 = x9*x97;
        const double x652 = x11*x97;
        const double x653 = r_11*x1*x445;
        const double x654 = x241*x243;
        const double x655 = x449*x520;
        const double x656 = 2*r_33*x248;
        const double x657 = x13*x232;
        const double x658 = x15*x163;
        const double x659 = x17*x90;
        const double x660 = x19*x232;
        const double x661 = x163*x21;
        const double x662 = x23*x90;
        const double x663 = x23*x526;
        const double x664 = x16*x97;
        const double x665 = x17*x528;
        const double x666 = x22*x94;
        const double x667 = x18*x97;
        const double x668 = x234*x48;
        const double x669 = x166*x21;
        const double x670 = x402 - x404 - x407 - x409 + x415 + x417 + x419 + x421 - x423 - x425 - x427 - x429 + x644 + x645 + x646 - x647 - x648 - x649 - x650 - x651 - x652 - x653 - x654 - x655 - x656 + x657 + x658 + x659 + x660 + x661 + x662 - x663 - x664 - x665 - x666 - x667 + x668 + x669;
        const double x671 = x637 + x643 + x670;
        const double x672 = x58 + x75;
        const double x673 = -x638 + x639 + x640 - x641 - x642;
        const double x674 = x626 + x627 + x628 + x629 - x630 + x631 + x632 + x633 + x634 + x635 + x636;
        const double x675 = x670 + x673 + x674;
        const double x676 = x12*x158;
        const double x677 = x158*x7;
        const double x678 = x158*x9;
        const double x679 = x11*x158;
        const double x680 = x13*x175;
        const double x681 = a_3*x3;
        const double x682 = x15*x681;
        const double x683 = x158*x18;
        const double x684 = x175*x19;
        const double x685 = x21*x681;
        const double x686 = x158*x24;
        const double x687 = x566 + x568 + x574;
        const double x688 = -x676 + x677 + x678 + x679 + x680 + x682 + x683 + x684 + x685 + x686 + x687;
        const double x689 = 8*x12;
        const double x690 = 8*x70;
        const double x691 = 8*x21;
        const double x692 = 8*x19;
        const double x693 = 8*x15;
        const double x694 = 8*x23;
        const double x695 = 8*r_12;
        const double x696 = r_11*r_21;
        const double x697 = 8*x247;
        const double x698 = -x100*x3 + x102*x3 - x104*x3 + x106*x3 + x108*x3 + x11*x691 + x11*x693 - x111*x3 + x113*x3 - x114*x3 + x14*x691 - x15*x689 + x16*x691 + x17*x690 + x18*x691 + x20*x693 - x21*x689 + x23*x690 + x24*x693 + x3*x84 + x3*x85 - x3*x86 + 4*x310 + x347*x695 + x350*x697 + 8*x522 + x691*x7 + x691*x9 - x692*x78 + x693*x7 + x693*x9 - x694*x73 + x695*x696*x98;
        const double x699 = x676 - x677 - x678 - x679 - x680 - x682 - x683 - x684 - x685 - x686 + x687;
        const double x700 = -x402 + x404 + x407 + x409 - x415 - x417 - x419 - x421 + x423 + x425 + x427 + x429 - x644 - x645 - x646 + x647 + x648 + x649 + x650 + x651 + x652 + x653 + x654 + x655 + x656 - x657 - x658 - x659 - x660 - x661 - x662 + x663 + x664 + x665 + x666 + x667 - x668 - x669;
        const double x701 = x637 + x673 + x700;
        const double x702 = -x58 + x75;
        const double x703 = x643 + x674 + x700;
        const double x704 = -x243;
        const double x705 = r_11*x120;
        const double x706 = d_2*x243;
        const double x707 = 2*Px;
        const double x708 = 2*r_12;
        const double x709 = -d_4*x243 + r_11*x91 + x101*x707 + x103*x707 + x19*x239 + x21*x708 + x23*x243 + x243*x43 + x707*x99 + x708*x92;
        const double x710 = -x706 + x709;
        const double x711 = r_12*x158;
        const double x712 = -x705;
        const double x713 = d_2*x343;
        const double x714 = d_2*x695;
        const double x715 = x706 + x709;
        const double x716 = r_12*x120;
        const double x717 = R_l_inv_53*a_0;
        const double x718 = x239*x717;
        const double x719 = d_4*r_12;
        const double x720 = r_13*x92;
        const double x721 = r_13*x21;
        const double x722 = r_12*x43;
        const double x723 = r_12*x23;
        const double x724 = x719 + x720 + x721 - x722 - x723;
        const double x725 = x135*x724;
        const double x726 = a_3*x243;
        const double x727 = x53*x726;
        const double x728 = r_11*x64;
        const double x729 = x53*x728;
        const double x730 = x716 + x718 + x725 + x727 + x729;
        const double x731 = x128*x708;
        const double x732 = d_4*r_11;
        const double x733 = r_13*x40;
        const double x734 = r_13*x19;
        const double x735 = r_11*x43;
        const double x736 = r_11*x23;
        const double x737 = x732 + x733 + x734 - x735 - x736;
        const double x738 = x130*x737;
        const double x739 = r_12*x64;
        const double x740 = x59*x739;
        const double x741 = -x731 + x738 + x740;
        const double x742 = r_11*x92;
        const double x743 = 2*x742;
        const double x744 = r_11*x21;
        const double x745 = 2*x744;
        const double x746 = r_12*x91;
        const double x747 = r_12*x19;
        const double x748 = 2*x747;
        const double x749 = -x743 - x745 + x746 + x748;
        const double x750 = a_3*x343;
        const double x751 = 4*x59;
        const double x752 = x737*x751;
        const double x753 = x346*x717;
        const double x754 = d_2*r_12;
        const double x755 = 4*x53;
        const double x756 = x754*x755;
        const double x757 = -4*R_l_inv_56*a_0*d_2*r_11 + x128*x343 + x724*x755;
        const double x758 = x743 + x745 - x746 - x748;
        const double x759 = x716 + x718 + x725 - x727 + x729;
        const double x760 = 4*x719;
        const double x761 = Py*x447;
        const double x762 = 4*x721;
        const double x763 = 4*x722;
        const double x764 = 4*x723;
        const double x765 = x59*x711;
        const double x766 = r_12*x40;
        const double x767 = x742 + x744 - x747 - x766;
        const double x768 = 4*r_13;
        const double x769 = d_2*x768;
        const double x770 = x53*x769 + x717*x768 - x751*x767;
        const double x771 = 8*x735;
        const double x772 = 8*x736;
        const double x773 = r_11*x466;
        const double x774 = x731 - x738 - x740;
        const double x775 = -x298*x724;
        const double x776 = -x239*x85;
        const double x777 = -x239*x86;
        const double x778 = -r_11*x306;
        const double x779 = x239*x84;
        const double x780 = std::pow(r_11, 3);
        const double x781 = 2*x98;
        const double x782 = x780*x781;
        const double x783 = r_13*x158;
        const double x784 = d_2*x783;
        const double x785 = x216*x726;
        const double x786 = -x216*x728;
        const double x787 = -x107*x239;
        const double x788 = -x108*x239;
        const double x789 = -x113*x239;
        const double x790 = -x114*x239;
        const double x791 = x102*x239;
        const double x792 = x104*x239;
        const double x793 = x106*x239;
        const double x794 = x111*x239;
        const double x795 = -x388*x733;
        const double x796 = -x388*x734;
        const double x797 = Px*x99;
        const double x798 = 4*x40;
        const double x799 = x797*x798;
        const double x800 = Px*x798;
        const double x801 = x101*x800;
        const double x802 = x103*x800;
        const double x803 = x573*x797;
        const double x804 = Px*x573;
        const double x805 = x101*x804;
        const double x806 = x103*x804;
        const double x807 = Py*x343;
        const double x808 = x12*x807;
        const double x809 = d_4*x23;
        const double x810 = x343*x809;
        const double x811 = r_21*x105;
        const double x812 = r_12*x3;
        const double x813 = x811*x812;
        const double x814 = r_31*x247;
        const double x815 = x346*x814;
        const double x816 = r_23*x811;
        const double x817 = x768*x816;
        const double x818 = r_31*x449;
        const double x819 = x768*x818;
        const double x820 = Py*x3;
        const double x821 = -x744*x820;
        const double x822 = x23*x43;
        const double x823 = -x343*x822;
        const double x824 = x19*x40;
        const double x825 = x343*x824;
        const double x826 = x332*x766;
        const double x827 = x747*x820;
        const double x828 = x437*x733;
        const double x829 = 4*x43;
        const double x830 = x734*x829;
        const double x831 = x775 + x776 + x777 + x778 + x779 + x782 - x784 - x785 + x786 + x787 + x788 + x789 + x790 + x791 + x792 + x793 + x794 + x795 + x796 + x799 + x801 + x802 + x803 + x805 + x806 + x808 + x810 + x813 + x815 + x817 + x819 + x821 + x823 + x825 + x826 + x827 + x828 + x830;
        const double x832 = x364*x737;
        const double x833 = x275*x708;
        const double x834 = d_4*x713;
        const double x835 = x209*x739;
        const double x836 = x160*x733;
        const double x837 = x160*x734;
        const double x838 = x5*x807;
        const double x839 = x23*x713;
        const double x840 = -x832 + x833 - x834 - x835 - x836 - x837 + x838 + x839;
        const double x841 = d_4*x783;
        const double x842 = Px*x158;
        const double x843 = x842*x99;
        const double x844 = x101*x842;
        const double x845 = x103*x842;
        const double x846 = x4*x807;
        const double x847 = Py*x681;
        const double x848 = r_12*x847;
        const double x849 = x43*x783;
        const double x850 = x19*x750;
        const double x851 = x21*x711;
        const double x852 = x23*x783;
        const double x853 = -x841 + x843 + x844 + x845 + x846 + x848 + x849 + x850 + x851 + x852;
        const double x854 = x724*(R_l_inv_27*x472 + 4*x215);
        const double x855 = x275*x343;
        const double x856 = d_4*x714;
        const double x857 = x177*x720;
        const double x858 = x177*x721;
        const double x859 = Py*x695;
        const double x860 = R_l_inv_26*x472 + 4*x208;
        const double x861 = std::pow(r_12, 3);
        const double x862 = 4*x98;
        const double x863 = 4*x754;
        const double x864 = 8*x92;
        const double x865 = Px*x864;
        const double x866 = Px*x691;
        const double x867 = r_22*x105;
        const double x868 = 8*x867;
        const double x869 = r_13*r_23;
        const double x870 = r_13*r_33;
        const double x871 = 8*x40;
        const double x872 = x21*x695;
        const double x873 = 8*x43;
        const double x874 = r_11*r_31*x697 + x100*x346 + x101*x865 + x101*x866 + x103*x865 + x103*x866 + x104*x346 - x106*x346 + x107*x346 - x108*x346 - x111*x346 + x113*x346 - x114*x346 + x12*x859 - x202*x346 - x216*x863 + x346*x84 - x346*x85 - x346*x86 - x463*x720 - x463*x721 + x691*x797 + x692*x742 + x694*x720 + x695*x809 - x695*x822 - x695*x824 + x696*x868 + x697*x870 + x721*x873 + x737*x860 + x744*x871 + x797*x864 + x861*x862 + x868*x869 + x872*x92;
        const double x875 = x841 - x843 - x844 - x845 - x846 - x848 - x849 - x850 - x851 - x852;
        const double x876 = x775 + x776 + x777 + x778 + x779 + x782 + x784 + x785 + x786 + x787 + x788 + x789 + x790 + x791 + x792 + x793 + x794 + x795 + x796 + x799 + x801 + x802 + x803 + x805 + x806 + x808 + x810 + x813 + x815 + x817 + x819 + x821 + x823 + x825 + x826 + x827 + x828 + x830;
        const double x877 = 8*x732;
        const double x878 = a_3*x877;
        const double x879 = x209*x711;
        const double x880 = Py*r_13;
        const double x881 = 8*x4*x880;
        const double x882 = x466*x734;
        const double x883 = x466*x735;
        const double x884 = x466*x736;
        const double x885 = std::pow(r_13, 3);
        const double x886 = Px*x873;
        const double x887 = Px*x694;
        const double x888 = x105*x696;
        const double x889 = 8*r_11;
        const double x890 = r_33*x247;
        const double x891 = x695*x92;
        const double x892 = -8*Px*d_4*x101 - 8*Px*d_4*x103 - 8*Px*d_4*x99 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_4*r_11*r_21 - 8*Py*d_4*r_12*r_22 - 8*Py*d_4*r_13*r_23 - 8*Pz*d_4*r_11*r_31 - 8*Pz*d_4*r_12*r_32 - 8*Pz*d_4*r_13*r_33 - 4*d_2*r_13*x216 - 4*r_13*x105*x6 - 4*r_13*x105*x8 - 4*r_13*x109*x110 - 4*r_13*x109*x112 - 4*r_13*x202 + r_13*x23*x873 - 4*r_13*x84 - 4*r_13*x85 + r_23*x695*x867 + 8*r_23*x888 + x100*x768 + x101*x886 + x101*x887 + x102*x768 + x103*x886 + x103*x887 + x108*x768 + x114*x768 + x19*x771 + x23*x891 + x40*x772 + x43*x872 + x694*x797 + x695*x890 + x767*x860 + x768*x86 + x797*x873 + x818*x889 + x862*x885;
        const double x893 = x832 - x833 + x834 + x835 + x836 + x837 - x838 - x839;
        const double x894 = r_11*x847;
        const double x895 = x21*x750;
        const double x896 = Px*x820;
        const double x897 = Px*x332;
        const double x898 = Py*x346;
        const double x899 = r_23*x105;
        const double x900 = x343*x40;
        const double x901 = Py*x21*x812 - d_4*x761 + r_11*x19*x820 + x100*x708 + x101*x896 + x101*x897 + x103*x896 + x103*x897 + x104*x708 - x106*x708 + x107*x708 - x108*x708 - x111*x708 + x113*x708 - x114*x708 + x12*x898 + x21*x900 + x23*x760 + x23*x761 - x23*x763 + x3*x888 + x332*x797 + x343*x814 - x388*x721 + x43*x762 + x447*x899 + x486*x708 - x501*x739 - x564*x737 - x573*x766 - x708*x84 - x708*x85 - x708*x86 + x768*x890 + x781*x861 + x797*x820;
        const double x902 = -4*Py*a_3*r_12*r_21 - 4*Pz*a_3*r_12*r_31 - 2*a_3*r_13*x498 + x894 + x895 + x901;
        const double x903 = x580*x724;
        const double x904 = r_11*x582;
        const double x905 = d_2*x760;
        const double x906 = x498*x728;
        const double x907 = x34*x880;
        const double x908 = x160*x721;
        const double x909 = x5*x898;
        const double x910 = x23*x863;
        const double x911 = -x903 - x904 - x905 - x906 - x907 - x908 + x909 + x910;
        const double x912 = R_l_inv_36*x472 + 4*x500;
        const double x913 = x737*x912;
        const double x914 = x346*x551;
        const double x915 = d_2*x877;
        const double x916 = x498*x863;
        const double x917 = x177*x733;
        const double x918 = x177*x734;
        const double x919 = Py*x5;
        const double x920 = x889*x919;
        const double x921 = x177*x736;
        const double x922 = x343*x84;
        const double x923 = x343*x85;
        const double x924 = Px*x871;
        const double x925 = Px*x692;
        const double x926 = 8*r_13;
        const double x927 = Py*r_11*x689 + r_22*x695*x811 + x101*x924 + x101*x925 + x102*x343 + x103*x924 + x103*x925 + x104*x343 + x106*x343 - x107*x343 - x108*x343 + x111*x343 - x113*x343 - x114*x343 + x19*x891 + 8*x23*x733 - x23*x771 + x23*x877 - x343*x86 + x40*x872 + 8*x43*x734 - x463*x733 - x463*x734 - x691*x742 + x692*x797 + x695*x814 + x780*x862 + x797*x871 + x816*x926 + x818*x926 + x824*x889;
        const double x928 = x343*x486 - x501*x713 + x724*(R_l_inv_37*x472 + 4*x497) - x922 - x923 + x927;
        const double x929 = x19*x711 + x4*x898 + x498*x726 - x894 - x895 + x901;
        const double x930 = x551*x768;
        const double x931 = x498*x769;
        const double x932 = x40*x714;
        const double x933 = x19*x714;
        const double x934 = a_3*x714 + x501*x711;
        const double x935 = x903 + x904 + x905 + x906 + x907 + x908 - x909 - x910;
        const double x936 = x243*x85;
        const double x937 = x243*x86;
        const double x938 = x781*x885;
        const double x939 = x243*x84;
        const double x940 = x100*x243;
        const double x941 = x102*x243;
        const double x942 = x108*x243;
        const double x943 = x114*x243;
        const double x944 = x106*x243;
        const double x945 = x107*x243;
        const double x946 = x111*x243;
        const double x947 = x113*x243;
        const double x948 = Px*x388;
        const double x949 = x948*x99;
        const double x950 = x101*x948;
        const double x951 = x103*x948;
        const double x952 = x797*x829;
        const double x953 = Px*x829;
        const double x954 = x101*x953;
        const double x955 = x103*x953;
        const double x956 = x437*x797;
        const double x957 = Px*x437;
        const double x958 = x101*x957;
        const double x959 = x103*x957;
        const double x960 = x344*x899;
        const double x961 = x343*x818;
        const double x962 = x812*x899;
        const double x963 = x346*x890;
        const double x964 = d_4*x343;
        const double x965 = x40*x964;
        const double x966 = x719*x820;
        const double x967 = Py*x768;
        const double x968 = x12*x967;
        const double x969 = x19*x964;
        const double x970 = x21*x760;
        const double x971 = x768*x809;
        const double x972 = x23*x900;
        const double x973 = x19*x343*x43;
        const double x974 = x723*x820;
        const double x975 = x21*x763;
        const double x976 = x768*x822;
        const double x977 = x573*x733;
        const double x978 = x721*x820;
        const double x979 = a_3*x713;
        const double x980 = d_4*x750;
        const double x981 = x43*x750;
        const double x982 = -x981;
        const double x983 = x23*x750;
        const double x984 = -x983;
        const double x985 = x4*x967;
        const double x986 = x158*x734;
        const double x987 = x979 + x980 + x982 + x984 + x985 + x986;
        const double x988 = Px*x160;
        const double x989 = x988*x99;
        const double x990 = x101*x988;
        const double x991 = x103*x988;
        const double x992 = x40*x713;
        const double x993 = x754*x820;
        const double x994 = x768*x919;
        const double x995 = x19*x713;
        const double x996 = x21*x863;
        const double x997 = x23*x769;
        const double x998 = -4*d_2*d_4*r_13 + x989 + x990 + x991 + x992 + x993 + x994 + x995 + x996 + x997;
        const double x999 = x936 + x937 + x938 - x939 + x940 + x941 + x942 + x943 - x944 - x945 - x946 - x947 - x949 - x950 - x951 + x952 + x954 + x955 + x956 + x958 + x959 + x960 + x961 + x962 + x963 - x965 - x966 - x968 - x969 - x970 - x971 + x972 + x973 + x974 + x975 + x976 - x977 - x978;
        const double x1000 = r_13*x466;
        const double x1001 = d_4*x1000;
        const double x1002 = Px*x466;
        const double x1003 = x922 + x923 + x927;
        const double x1004 = 16*d_4;
        const double x1005 = 16*x23;
        const double x1006 = 16*x19;
        const double x1007 = 16*x92;
        const double x1008 = Px*x1007;
        const double x1009 = 16*x21;
        const double x1010 = Px*x1009;
        const double x1011 = 16*x867;
        const double x1012 = x998 + x999;
        const double x1013 = -x29;
        const double x1014 = -x42;
        const double x1015 = a_1 + x141 + x142;
        const double x1016 = x1014 + x1015 + x62;
        const double x1017 = x134 + x136 - x138 - x139;
        const double x1018 = x196 + x197 + x200 + x201 + x203 + x206 + x210 + x211 + x212 + x213 + x214 + x220 + x221 + x222 + x223 + x224 + x227 + x228 + x229 + x230 + x231 + x233 + x235 + x236 + x237 + x238 + x242 + x245 + x249 + x252 + x254 + x256 + x258 + x259 + x260 + x261 + x262;
        const double x1019 = x1018 + x266 + x268 + x270;
        const double x1020 = x218 + x226 + x384 + x385;
        const double x1021 = x299 + x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x325 + x327 + x329 + x331 + x333 + x334 + x335 + x336 + x338 + x339 + x340 + x341 + x342 + x345 + x348 + x351 + x352 + x353 + x354 + x355 + x356 + x357 + x358 + x359;
        const double x1022 = x195 + x199 + x217 + x225;
        const double x1023 = x1018 + x285 + x374 + x375 + x376 + x377 + x378 + x379 + x380 + x381 + x382;
        const double x1024 = -x535 - x543;
        const double x1025 = -x41*x576 + x578;
        const double x1026 = x1024 + x537 + x539 + x541;
        const double x1027 = d_2*x577 + x120*x484 - x612;

        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x0;
        A(0, 2) = x0;
        A(0, 3) = x2;
        A(0, 4) = -x3;
        A(0, 5) = x1;
        A(0, 6) = r_23;
        A(0, 8) = r_23;
        A(1, 0) = x27;
        A(1, 1) = x29;
        A(1, 2) = x31;
        A(1, 3) = x33;
        A(1, 4) = -x34;
        A(1, 5) = x32;
        A(1, 6) = x36;
        A(1, 7) = x29;
        A(1, 8) = x37;
        A(2, 0) = -x116 - x123 - x67 - x83;
        A(2, 1) = -x127 - x133 - x140;
        A(2, 2) = x144 + x148 + x151;
        A(2, 3) = x152 + x153 + x154 - x162 - x169;
        A(2, 4) = x170 + x171 + x172 - x173 - x174 - x176 - x179;
        A(2, 5) = -x156 + x157 - x159 + x161 + x181 + 2*x38*x44;
        A(2, 6) = x144 + x183 + x186;
        A(2, 7) = x140 + x187 + x188;
        A(2, 8) = -x145 - x146 - x147 - x184 - x189 - x42 - x62;
        A(3, 0) = x263 + x274 + x296;
        A(3, 1) = x360 + x363 + x373;
        A(3, 2) = -x284 - x383 - x386 - x387;
        A(3, 3) = x389 + x390 + x391 + x392 - x393 - x394 - x400 - x462;
        A(3, 4) = x464 - x465 + x467 + x468 - x470 - x471 - x474;
        A(3, 5) = x395 - x396 - x397 - x398 + x399 - x476;
        A(3, 6) = -x386 - x478 - x480;
        A(3, 7) = -x360 - x481 - x482;
        A(3, 8) = x263 + x383 + x477 + x479;
        A(4, 0) = -x533 - x544 - x561;
        A(4, 1) = x575 + x579 + x591;
        A(4, 2) = x560 + x594 + x597 + x598;
        A(4, 3) = -x608 - x611 - x613;
        A(4, 4) = x117*x614 + x473*x488 + x615;
        A(4, 5) = -x613 - x616 - x617;
        A(4, 6) = -x598 - x619 - x620 - x621;
        A(4, 7) = x579 + x622 + x623;
        A(4, 8) = x544 + x618 + x624 + x625;
        A(5, 0) = -x671;
        A(5, 1) = -x158*x672;
        A(5, 2) = -x675;
        A(5, 3) = x688;
        A(5, 4) = x698;
        A(5, 5) = -x699;
        A(5, 6) = -x701;
        A(5, 7) = x158*x702;
        A(5, 8) = -x703;

        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x704;
        B(0, 2) = x704;
        B(0, 3) = -x343;
        B(0, 4) = -x695;
        B(0, 5) = x343;
        B(0, 6) = x243;
        B(0, 8) = x243;
        B(1, 0) = x705 + x710;
        B(1, 1) = x711;
        B(1, 2) = x710 + x712;
        B(1, 3) = -x713;
        B(1, 4) = -x714;
        B(1, 5) = x713;
        B(1, 6) = x705 + x715;
        B(1, 7) = x711;
        B(1, 8) = x712 + x715;
        B(2, 0) = -x730 - x741 - x749;
        B(2, 1) = x750 + x752 - x753 - x756 - x757;
        B(2, 2) = x741 + x758 + x759;
        B(2, 3) = -x760 - x761 - x762 + x763 + x764 + x765 + x770;
        B(2, 4) = 8*Py*r_13*r_21 + 8*Pz*r_13*r_31 + 8*d_4*r_11 - x59*x773 - x771 - x772;
        B(2, 5) = x760 + x761 + x762 - x763 - x764 - x765 + x770;
        B(2, 6) = x749 + x759 + x774;
        B(2, 7) = -x750 - x752 + x753 + x756 - x757;
        B(2, 8) = -x730 - x758 - x774;
        B(3, 0) = x831 + x840 + x853;
        B(3, 1) = x209*x713 + x23*x714 + x5*x859 - x854 - x855 - x856 - x857 - x858 + x874;
        B(3, 2) = -x840 - x875 - x876;
        B(3, 3) = x878 + x879 + x881 + x882 - x883 - x884 - x892;
        B(3, 4) = x466*(2*Py*r_13*r_22 + 2*Pz*r_13*r_32 + 2*d_4*r_12 - r_11*x209 - 2*x722 - 2*x723);
        B(3, 5) = -x878 - x879 - x881 - x882 + x883 + x884 - x892;
        B(3, 6) = -x853 - x876 - x893;
        B(3, 7) = 8*Py*d_2*r_12*r_23 + 8*Pz*d_2*r_12*r_33 + 4*d_2*r_11*x209 - x854 - x855 - x856 - x857 - x858 - x874;
        B(3, 8) = x831 + x875 + x893;
        B(4, 0) = -x902 - x911;
        B(4, 1) = -x913 + x914 - x915 + x916 - x917 - x918 + x920 + x921 + x928;
        B(4, 2) = x911 + x929;
        B(4, 3) = 8*Py*d_2*r_11*r_22 + 8*Pz*d_2*r_11*r_32 + x767*x912 - x930 - x931 - x932 - x933 - x934;
        B(4, 4) = x773*(x501 + x64);
        B(4, 5) = x177*x742 + x177*x744 + x767*x912 - x930 - x931 - x932 - x933 + x934;
        B(4, 6) = -x902 - x935;
        B(4, 7) = x913 - x914 + x915 - x916 + x917 + x918 - x920 - x921 + x928;
        B(4, 8) = x929 + x935;
        B(5, 0) = x936 + x937 + x938 - x939 + x940 + x941 + x942 + x943 - x944 - x945 - x946 - x947 - x949 - x950 - x951 + x952 + x954 + x955 + x956 + x958 + x959 + x960 + x961 + x962 + x963 - x965 - x966 - x968 - x969 - x970 - x971 + x972 + x973 + x974 + x975 + x976 - x977 - x978 - x987 - x998;
        B(5, 1) = x466*(-x724 - x754);
        B(5, 2) = d_4*x769 + x987 - x989 - x990 - x991 - x992 - x993 - x994 - x995 - x996 - x997 + x999;
        B(5, 3) = Py*x4*x889 + a_3*x872 + x1000*x23 + x1000*x43 - x1001 + x1002*x101 + x1002*x103 + x1002*x99 + x1003 + x19*x773 + x28*x859;
        B(5, 4) = 16*Py*r_12*x12 + 16*r_11*x814 + r_12*x1007*x21 + x100*x695 - x1004*x720 - x1004*x721 + x1005*x719 + x1005*x720 - x1005*x722 + x1006*x742 - x1006*x766 + x1007*x797 + x1008*x101 + x1008*x103 + x1009*x797 + x101*x1010 + x1010*x103 + x1011*x696 + x1011*x869 + x104*x695 - x106*x695 + x107*x695 - x108*x695 - x111*x695 + x113*x695 - x114*x695 + 16*x247*x870 + 16*x40*x744 + 16*x43*x721 + x695*x84 + x695*x85 - x695*x86 + 8*x861*x98;
        B(5, 5) = 8*Px*a_3*x101 + 8*Px*a_3*x103 + 8*Px*a_3*x99 + 8*Py*a_3*r_11*r_21 + 8*Py*a_3*r_12*r_22 + 8*Py*a_3*r_13*r_23 + 8*Pz*a_3*r_11*r_31 + 8*Pz*a_3*r_12*r_32 + 8*Pz*a_3*r_13*r_33 - x1001 - x1003;
        B(5, 6) = -x1012 - x979 + x980 - x981 - x983 + x985 + x986;
        B(5, 7) = x466*(x719 + x720 + x721 - x722 - x723 - x754);
        B(5, 8) = -x1012 + x979 - x980 - x982 - x984 - x985 - x986;

        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = r_23;
        C(0, 2) = r_23;
        C(0, 3) = x1;
        C(0, 4) = x3;
        C(0, 5) = x2;
        C(0, 6) = x0;
        C(0, 8) = x0;
        C(1, 0) = -x27;
        C(1, 1) = x1013;
        C(1, 2) = -x31;
        C(1, 3) = x32;
        C(1, 4) = x34;
        C(1, 5) = x33;
        C(1, 6) = -x36;
        C(1, 7) = x1013;
        C(1, 8) = -x37;
        C(2, 0) = x1016 + x148 + x183;
        C(2, 1) = -x1017 - x127 - x187;
        C(2, 2) = -x1014 - x143 - x185 - x189 - x54 - x57 - x60;
        C(2, 3) = -x162 + x163 - x164 - x165 + x166 + x167 - x168 - x180;
        C(2, 4) = -x170 - x171 - x172 + x173 + x174 + x176 - x179;
        C(2, 5) = -x155 - x156 + x157 - x159 + x161 - x181;
        C(2, 6) = x1015 + x149 + x182 - x28 + x67 - x69 - x76 - x77;
        C(2, 7) = x1017 + x133 + x188;
        C(2, 8) = x1016 + x151 + x186;
        C(3, 0) = -x1019 - x1020 - x296 - x477;
        C(3, 1) = -x1021 - x373 - x481;
        C(3, 2) = x1022 + x1023 + x284 + x478;
        C(3, 3) = -x400 - x401 - x403 - x405 - x406 + x408 + x410 - x411 - x412 - x413 - x414 - x416 - x418 - x420 - x422 + x424 + x426 + x428 + x430 - x431 - x432 - x434 - x435 - x436 - x438 + x439 + x440 + x441 + x442 + x443 + x444 + x446 + x448 + x450 + x452 - x453 - x454 + x455 + x457 + x458 + x460 + x461 - x475;
        C(3, 4) = -x464 + x465 - x467 - x468 + x470 + x471 - x474;
        C(3, 5) = x300*x44 + x395 - x397 - x398 + x399 + x476;
        C(3, 6) = x1019 + x1022 + x387 + x480;
        C(3, 7) = x1021 + x363 + x482;
        C(3, 8) = -x1020 - x1023 - x274 - x479;
        C(4, 0) = x1024 + x561 + x621 + x625;
        C(4, 1) = -x1025 - x591 - x622;
        C(4, 2) = -x1026 - x560 - x620 - x624;
        C(4, 3) = x1027 + x608 + x616;
        C(4, 4) = 8*d_2*x117*x488 + x117*x614 - x615;
        C(4, 5) = x1027 + x611 + x617;
        C(4, 6) = x1026 + x597 + x619;
        C(4, 7) = -x1025 - x575 - x623;
        C(4, 8) = -x1024 - x533 - x594 - x618;
        C(5, 0) = x671;
        C(5, 1) = x158*x672;
        C(5, 2) = x675;
        C(5, 3) = -x688;
        C(5, 4) = -x698;
        C(5, 5) = x699;
        C(5, 6) = x701;
        C(5, 7) = -x158*x702;
        C(5, 8) = x703;

        // Invoke the solver
        std::array<double, 48> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);

        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[0] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor();
    // Finish code for general_6dof solution node 0

    // Code for equation all-zero dispatcher node 2
    auto EquationAllZeroDispatcherNode_node_2_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const bool checked_result = ((std::fabs(a_3*r_11*std::sin(th_0) - a_3*r_21*std::cos(th_0)) <= 9.9999999999999995e-7) && (std::fabs(a_3*r_12*std::sin(th_0) - a_3*r_22*std::cos(th_0)) <= 9.9999999999999995e-7) && (std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0) - d_4*r_13*std::sin(th_0) + d_4*r_23*std::cos(th_0)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2

    // Code for explicit solution node 3, solved variable is th_5
    auto ExplicitSolutionNode_node_3_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const bool condition_0 = ((std::fabs(a_3*r_11*std::sin(th_0) - a_3*r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(a_3*r_12*std::sin(th_0) - a_3*r_22*std::cos(th_0)) >= zero_tolerance) || (std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0) - d_4*r_13*std::sin(th_0) + d_4*r_23*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = a_3*x0;
                const double x2 = std::cos(th_0);
                const double x3 = -a_3*r_22*x2 + r_12*x1;
                const double x4 = a_3*r_21*x2 - r_11*x1;
                const double x5 = std::atan2(x3, x4);
                const double x6 = -Px*x0 + Py*x2 + d_4*r_13*x0 - d_4*r_23*x2;
                const double x7 = safe_sqrt(std::pow(x3, 2) + std::pow(x4, 2) - std::pow(x6, 2));
                RawSolution solution_0(this_solution);
                solution_0[7] = x5 + std::atan2(x7, x6);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = x5 + std::atan2(-x7, x6);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_5_processor();
    // Finish code for explicit solution node 3

    // Code for non-branch dispatcher node 4
    // Actually, there is no code

    // Code for explicit solution node 5, solved variable is th_3
    auto ExplicitSolutionNode_node_5_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(4);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(4);
        if (!this_input_valid)
            return;

        // The solution of non-root node 5
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            const bool condition_0 = std::fabs((r_11*std::sin(th_0) - r_21*std::cos(th_0))*std::sin(th_5) + (r_12*std::sin(th_0) - r_22*std::cos(th_0))*std::cos(th_5)) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = safe_acos((-r_11*x0 + r_21*x1)*std::sin(th_5) + (-r_12*x0 + r_22*x1)*std::cos(th_5));
                RawSolution solution_0(this_solution);
                solution_0[5] = x2;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx_0);
                const double tmp_sol_value = -x2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_3_processor();
    // Finish code for explicit solution node 4

    // Code for non-branch dispatcher node 6
    // Actually, there is no code

    // Code for explicit solution node 7, solved variable is th_2
    auto ExplicitSolutionNode_node_7_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(6);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(6);
        if (!this_input_valid)
            return;

        // The solution of non-root node 7
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[7];
            const bool condition_0 = ((2*std::fabs(a_0*a_1) >= zero_tolerance) || (2*std::fabs(a_0*d_2) >= zero_tolerance) || (std::fabs(std::pow(Px, 2) - 2*Px*a_3*r_11*std::cos(th_5) + 2*Px*a_3*r_12*std::sin(th_5) - 2*Px*d_4*r_13 + std::pow(Py, 2) - 2*Py*a_3*r_21*std::cos(th_5) + 2*Py*a_3*r_22*std::sin(th_5) - 2*Py*d_4*r_23 + std::pow(Pz, 2) - 2*Pz*a_3*r_31*std::cos(th_5) + 2*Pz*a_3*r_32*std::sin(th_5) - 2*Pz*d_4*r_33 - std::pow(a_0, 2) - std::pow(a_1, 2) + std::pow(a_3, 2)*std::pow(r_11, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_11*r_12*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_12, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_12, 2) + std::pow(a_3, 2)*std::pow(r_21, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_21*r_22*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_22, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_22, 2) + std::pow(a_3, 2)*std::pow(r_31, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_31*r_32*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_32, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_32, 2) + 2*a_3*d_4*r_11*r_13*std::cos(th_5) - 2*a_3*d_4*r_12*r_13*std::sin(th_5) + 2*a_3*d_4*r_21*r_23*std::cos(th_5) - 2*a_3*d_4*r_22*r_23*std::sin(th_5) + 2*a_3*d_4*r_31*r_33*std::cos(th_5) - 2*a_3*d_4*r_32*r_33*std::sin(th_5) - std::pow(d_2, 2) + std::pow(d_4, 2)*std::pow(r_13, 2) + std::pow(d_4, 2)*std::pow(r_23, 2) + std::pow(d_4, 2)*std::pow(r_33, 2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 2*a_0;
                const double x1 = std::atan2(-d_2*x0, a_1*x0);
                const double x2 = std::pow(a_1, 2);
                const double x3 = std::pow(a_0, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_2, 2);
                const double x6 = 2*d_4;
                const double x7 = r_13*x6;
                const double x8 = r_23*x6;
                const double x9 = r_33*x6;
                const double x10 = std::pow(a_3, 2);
                const double x11 = std::pow(r_12, 2)*x10;
                const double x12 = std::pow(r_22, 2)*x10;
                const double x13 = std::pow(r_32, 2)*x10;
                const double x14 = std::pow(d_4, 2);
                const double x15 = std::cos(th_5);
                const double x16 = 2*a_3;
                const double x17 = x15*x16;
                const double x18 = std::sin(th_5);
                const double x19 = x16*x18;
                const double x20 = a_3*x15;
                const double x21 = a_3*x18;
                const double x22 = std::pow(x15, 2);
                const double x23 = x10*x22;
                const double x24 = x10*std::sin(2*th_5);
                const double x25 = std::pow(Px, 2) - Px*r_11*x17 + Px*r_12*x19 - Px*x7 + std::pow(Py, 2) - Py*r_21*x17 + Py*r_22*x19 - Py*x8 + std::pow(Pz, 2) - Pz*r_31*x17 + Pz*r_32*x19 - Pz*x9 + std::pow(r_11, 2)*x23 - r_11*r_12*x24 + r_11*x20*x7 - r_12*x21*x7 + std::pow(r_13, 2)*x14 + std::pow(r_21, 2)*x23 - r_21*r_22*x24 + r_21*x20*x8 - r_22*x21*x8 + std::pow(r_23, 2)*x14 + std::pow(r_31, 2)*x23 - r_31*r_32*x24 + r_31*x20*x9 - r_32*x21*x9 + std::pow(r_33, 2)*x14 - x11*x22 + x11 - x12*x22 + x12 - x13*x22 + x13 - x2 - x3 - x5;
                const double x26 = safe_sqrt(x2*x4 - std::pow(x25, 2) + x4*x5);
                RawSolution solution_0(this_solution);
                solution_0[4] = x1 + std::atan2(x26, x25);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx_0);
                const double tmp_sol_value = x1 + std::atan2(-x26, x25);
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
    ExplicitSolutionNode_node_7_solve_th_2_processor();
    // Finish code for explicit solution node 6

    // Code for solved_variable dispatcher node 8
    auto SolvedVariableDispatcherNode_node_8_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            bool forwarded_to_child = false;
            const double th_3 = this_solution[5];

            const bool degenerate_valid_0 = std::fabs(th_3) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(14, branch_input_index);
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
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8

    // Code for explicit solution node 17, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor = [&]() -> void
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
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_33) >= zero_tolerance) || (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double tmp_sol_value = std::atan2(r_13*std::cos(th_0) + r_23*std::sin(th_0), r_33);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(18, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor();
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
            bool forwarded_to_child = false;
            const double th_2 = this_solution[4];

            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.3408918986629901) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(22, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.3408918986629901) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(25, branch_input_index);
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
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(0.973688178796424*a_1 - 0.22788446737788701*d_2) >= 9.9999999999999995e-7) || (std::fabs(a_0 + 0.22788446737788701*a_1 + 0.973688178796424*d_2) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = a_0 + 0.22788446737788701*a_1 + 0.973688178796424*d_2;
                const double x1 = a_3*std::sin(th_5);
                const double x2 = a_3*std::cos(th_5);
                const double x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1;
                const double x4 = 0.973688178796424*a_1 - 0.22788446737788701*d_2;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6;
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4);
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
            const double th_3 = this_solution[5];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(-r_13*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::cos(th_0) - r_23*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::sin(th_0) + r_33*(0.973688178796424*std::sin(th_1) + 0.22788446737788701*std::cos(th_1))) >= zero_tolerance) || (std::fabs(r_13*((0.973688178796424*std::sin(th_1) + 0.22788446737788601*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) + r_23*((0.973688178796424*std::sin(th_1) + 0.22788446737788601*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.22788446737788601*std::sin(th_1) - 0.973688178796424*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.973688178796424*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.973688178796424*x1;
                const double x8 = x0*(0.22788446737788601*x2 + x7);
                const double x9 = -0.22788446737788701*x1 - x3;
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(-0.22788446737788601*x1 - x3), -r_13*x6*x9 - r_23*x4*x9 + r_33*(0.22788446737788701*x2 + x7));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(0.973688178796424*a_1 - 0.22788446737788701*d_2) >= 9.9999999999999995e-7) || (std::fabs(-a_0 + 0.22788446737788701*a_1 + 0.973688178796424*d_2) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = a_0 - 0.22788446737788701*a_1 - 0.973688178796424*d_2;
                const double x1 = a_3*std::sin(th_5);
                const double x2 = a_3*std::cos(th_5);
                const double x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1;
                const double x4 = -0.973688178796424*a_1 + 0.22788446737788701*d_2;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6;
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4);
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
            const double th_3 = this_solution[5];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(-r_13*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::cos(th_0) - r_23*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::sin(th_0) + r_33*(0.973688178796424*std::sin(th_1) + 0.22788446737788701*std::cos(th_1))) >= zero_tolerance) || (std::fabs(-r_13*((-0.973688178796424*std::sin(th_1) - 0.22788446737788601*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.973688178796424*std::sin(th_1) - 0.22788446737788601*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.22788446737788601*std::sin(th_1) - 0.973688178796424*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.973688178796424*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.973688178796424*x1;
                const double x8 = x0*(-0.22788446737788601*x2 - x7);
                const double x9 = -0.22788446737788701*x1 - x3;
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(0.22788446737788601*x1 + x3), r_13*x6*x9 + r_23*x4*x9 - r_33*(0.22788446737788701*x2 + x7));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
            const double th_1th_2th_4_soa = this_solution[3];
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(a_1*std::sin(th_2) + d_2*std::cos(th_2)) >= 9.9999999999999995e-7) || (std::fabs(a_0 + a_1*std::cos(th_2) - d_2*std::sin(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = std::sin(th_2);
                const double x1 = std::cos(th_2);
                const double x2 = -a_1*x0 - d_2*x1;
                const double x3 = Px*std::cos(th_0) + Py*std::sin(th_0) + a_3*std::cos(th_1th_2th_4_soa) - d_4*std::sin(th_1th_2th_4_soa);
                const double x4 = a_0 + a_1*x1 - d_2*x0;
                const double x5 = Pz - a_3*r_31*std::cos(th_5) + a_3*r_32*std::sin(th_5) - d_4*r_33;
                const double tmp_sol_value = std::atan2(x2*x3 - x4*x5, x2*x5 + x3*x4);
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
            const double th_1 = this_solution[1];
            const double th_1th_2th_4_soa = this_solution[3];
            const double th_2 = this_solution[4];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -th_1 + th_1th_2th_4_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(a_1*std::sin(th_2) + d_2*std::cos(th_2)) >= 9.9999999999999995e-7) || (std::fabs(a_0 + a_1*std::cos(th_2) - d_2*std::sin(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_2);
                const double x2 = a_0 + a_1*x0 - d_2*x1;
                const double x3 = a_3*std::sin(th_5);
                const double x4 = a_3*std::cos(th_5);
                const double x5 = Pz - d_4*r_33 - r_31*x4 + r_32*x3;
                const double x6 = -a_1*x1 - d_2*x0;
                const double x7 = std::cos(th_0);
                const double x8 = std::sin(th_0);
                const double x9 = Px*x7 + Py*x8 - d_4*r_13*x7 - d_4*r_23*x8 - r_11*x4*x7 + r_12*x3*x7 - r_21*x4*x8 + r_22*x3*x8;
                const double tmp_sol_value = std::atan2(-x2*x5 + x6*x9, x2*x9 + x5*x6);
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
            const double th_2 = this_solution[4];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_33*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance) || (std::fabs(-r_13*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - r_23*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + r_33*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = x0*x1 + x2*x3;
                const double x5 = -x0*x2 + x1*x3;
                const double x6 = r_13*std::cos(th_0);
                const double x7 = r_23*std::sin(th_0);
                const double tmp_sol_value = std::atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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

    // Code for explicit solution node 9, solved variable is th_4
    auto ExplicitSolutionNode_node_9_solve_th_4_processor = [&]() -> void
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
            const double th_3 = this_solution[5];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(std::sin(th_3)) >= zero_tolerance) || (std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance) || (std::fabs((r_11*std::sin(th_0) - r_21*std::cos(th_0))*std::cos(th_5) + (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*std::sin(th_5)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_3);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(x0*(-r_13*x1 + r_23*x2), x0*(-(-r_11*x1 + r_21*x2)*std::cos(th_5) + (-r_12*x1 + r_22*x2)*std::sin(th_5)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor();
    // Finish code for explicit solution node 9

    // Code for non-branch dispatcher node 10
    // Actually, there is no code

    // Code for explicit solution node 11, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_3 = this_solution[5];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(std::sin(th_3)) >= zero_tolerance) || (std::fabs(r_31*std::sin(th_5) + r_32*std::cos(th_5)) >= zero_tolerance) || (std::fabs((r_11*std::cos(th_0) + r_21*std::sin(th_0))*std::sin(th_5) + (r_12*std::cos(th_0) + r_22*std::sin(th_0))*std::cos(th_5)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_3);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_5);
                const double x3 = std::cos(th_0);
                const double x4 = std::sin(th_0);
                const double tmp_sol_value = std::atan2(x0*(-r_31*x1 - r_32*x2), x0*(-x1*(-r_11*x3 - r_21*x4) + x2*(r_12*x3 + r_22*x4)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 10

    // Code for non-branch dispatcher node 12
    // Actually, there is no code

    // Code for explicit solution node 13, solved variable is th_1
    auto ExplicitSolutionNode_node_13_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(12);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(12);
        if (!this_input_valid)
            return;

        // The solution of non-root node 13
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1th_2_soa = this_solution[2];
            const double th_2 = this_solution[4];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = th_1th_2_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_1_processor();
    // Finish code for explicit solution node 12

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
        const double value_at_2 = raw_ik_out_i[4];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[5];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[6];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[7];  // th_5
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
        ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw;
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
        ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw;
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
            ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw;
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
                    candidate[5] = candidate[5] - th_5_offset_original2raw;
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

}; // struct abb_crb_15000_ik

// Code below for debug
void test_ik_solve_abb_crb_15000()
{
    std::array<double, abb_crb_15000_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = abb_crb_15000_ik::computeFK(theta);
    auto ik_output = abb_crb_15000_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = abb_crb_15000_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_abb_crb_15000();
    return 0;
}
