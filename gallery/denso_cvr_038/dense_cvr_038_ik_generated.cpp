#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct dense_cvr_038_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 1152;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -1.5707963267948966, -3.141592653589793, -6.283185307179586, -6.283185307179586, -3.141592653589793}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 4.71238898038469, 3.141592653589793, 0.0, 0.0, 3.141592653589793}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 35;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_0 = 0.165;
static constexpr double a_2 = 0.012;
static constexpr double d_1 = 0.02;
static constexpr double d_3 = 0.1775;
static constexpr double d_4 = -0.0445;
static constexpr double d_5 = 0.045;
static constexpr double pre_transform_special_symbol_23 = 0.18;

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
    const double x29 = x28*x3;
    const double x30 = x7 - x8*x9;
    const double x31 = -x13*x30 + x17*x2*x28;
    const double x32 = -x13*x2*x28 - x17*x30;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x11 + x12*x19;
    ee_pose_raw(0, 1) = -x0*x19 - x11*x12;
    ee_pose_raw(0, 2) = x20;
    ee_pose_raw(0, 3) = a_2*x10 - d_1*x1 + d_3*x16 + d_4*x11 + d_5*x20 + x21*x4;
    ee_pose_raw(1, 0) = -x0*x23 + x12*x26;
    ee_pose_raw(1, 1) = -x0*x26 - x12*x23;
    ee_pose_raw(1, 2) = x27;
    ee_pose_raw(1, 3) = a_2*x22 + d_1*x4 + d_3*x24 + d_4*x23 + d_5*x27 + x1*x21;
    ee_pose_raw(2, 0) = x0*x29 + x12*x31;
    ee_pose_raw(2, 1) = -x0*x31 + x12*x28*x3;
    ee_pose_raw(2, 2) = x32;
    ee_pose_raw(2, 3) = -a_0*x5 + a_2*x28 + d_3*x30 - d_4*x29 + d_5*x32;
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
    const double x18 = -x11*x16 - x17*(x1*x13 + x12*x14);
    const double x19 = x1*x4;
    const double x20 = x1*x9;
    const double x21 = -x19*x3 - x20*x8;
    const double x22 = 1.0*x0*x3*x9 - x19*x8;
    const double x23 = -x12*x6 - x13*x22;
    const double x24 = -x16*x21 - x17*(x12*x22 - x13*x6);
    const double x25 = 1.0*x8;
    const double x26 = 1.0*x3;
    const double x27 = x25*x4 - x26*x9;
    const double x28 = -x25*x9 - x26*x4;
    const double x29 = x13*x28;
    const double x30 = -x12*x17*x28 - x16*x27;
    const double x31 = 1.0*a_0*x4;
    const double x32 = pre_transform_special_symbol_23 - x31;
    const double x33 = a_2*x28 + d_3*x27 + pre_transform_special_symbol_23 - x31;
    const double x34 = a_0*x20 + d_1*x6;
    const double x35 = a_2*x22 + d_3*x21 + x34;
    const double x36 = -d_4*x29 + x33;
    const double x37 = d_4*x23 + x35;
    const double x38 = d_5*x30 + x36;
    const double x39 = d_5*x24 + x37;
    const double x40 = a_0*x10 - d_1*x1;
    const double x41 = a_2*x14 + d_3*x11 + x40;
    const double x42 = d_4*x15 + x41;
    const double x43 = d_5*x18 + x42;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x11;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x18;
    jacobian(1, 1) = x6;
    jacobian(1, 2) = x6;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x23;
    jacobian(1, 5) = x24;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x27;
    jacobian(2, 4) = -x29;
    jacobian(2, 5) = x30;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x6;
    jacobian(3, 2) = -x32*x6;
    jacobian(3, 3) = -x21*x33 + x27*x35;
    jacobian(3, 4) = -x23*x36 - x29*x37;
    jacobian(3, 5) = -x24*x38 + x30*x39;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = -x1*x32;
    jacobian(4, 3) = x11*x33 - x27*x41;
    jacobian(4, 4) = x15*x36 + x29*x42;
    jacobian(4, 5) = x18*x38 - x30*x43;
    jacobian(5, 2) = x1*x34 + x40*x6;
    jacobian(5, 3) = -x11*x35 + x21*x41;
    jacobian(5, 4) = -x15*x37 + x23*x42;
    jacobian(5, 5) = -x18*x39 + x24*x43;
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
    const double x19 = a_2*x18 + d_3*x13 + pre_transform_special_symbol_23 - x6;
    const double x20 = 1.0*x10*x11*x14 - x14*x9;
    const double x21 = a_0*x12*x14 + d_1*x2;
    const double x22 = a_2*x20 + d_3*x17 + x21;
    const double x23 = std::sin(th_3);
    const double x24 = x18*x23;
    const double x25 = std::cos(th_3);
    const double x26 = -x2*x25 - x20*x23;
    const double x27 = -d_4*x24 + x19;
    const double x28 = d_4*x26 + x22;
    const double x29 = std::cos(th_4);
    const double x30 = std::sin(th_4);
    const double x31 = -x13*x29 - x18*x25*x30;
    const double x32 = -x17*x29 - x30*(-x2*x23 + x20*x25);
    const double x33 = d_5*x31 + x27;
    const double x34 = d_5*x32 + x28;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x14;
    const double x37 = p_on_ee_z*x36;
    const double x38 = x2*x4;
    const double x39 = x11*x2;
    const double x40 = -x10*x38 - x39*x8;
    const double x41 = 1.0*x1*x10*x11 - x38*x8;
    const double x42 = a_0*x39 - d_1*x36;
    const double x43 = a_2*x41 + d_3*x40 + x42;
    const double x44 = 1.0*x14*x25 - x23*x41;
    const double x45 = d_4*x44 + x43;
    const double x46 = -x29*x40 - x30*(x23*x36 + x25*x41);
    const double x47 = d_5*x46 + x45;
    const double x48 = x0*x14 + x1*x35;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x7 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19;
    jacobian(0, 4) = p_on_ee_y*x24 + p_on_ee_z*x26 - x24*x28 - x26*x27;
    jacobian(0, 5) = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = -x36*x7 + x37;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x43 + x19*x40;
    jacobian(1, 4) = -p_on_ee_x*x24 - p_on_ee_z*x44 + x18*x23*x45 + x27*x44;
    jacobian(1, 5) = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46;
    jacobian(2, 1) = -x48;
    jacobian(2, 2) = 1.0*x1*x42 + 1.0*x14*x21 - x48;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x43 - x22*x40;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x44 + x26*x45 - x28*x44;
    jacobian(2, 5) = -p_on_ee_x*x32 + p_on_ee_y*x46 + x32*x47 - x34*x46;
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
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 48, 48, 96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 384, 384, 192, 384, 384, 192, 384, 384}};
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
        R_l(0, 6) = d_1;
        R_l(0, 7) = -a_2;
        R_l(1, 2) = -a_0;
        R_l(1, 6) = -a_2;
        R_l(1, 7) = -d_1;
        R_l(2, 4) = a_0;
        R_l(3, 6) = -1;
        R_l(4, 7) = 1;
        R_l(5, 5) = 2*a_0*a_2;
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
        const double x4 = d_4*r_22;
        const double x5 = -x4;
        const double x6 = d_3*r_23;
        const double x7 = -x6;
        const double x8 = std::pow(r_21, 2);
        const double x9 = Py*x8;
        const double x10 = std::pow(r_22, 2);
        const double x11 = Py*x10;
        const double x12 = std::pow(r_23, 2);
        const double x13 = Py*x12;
        const double x14 = d_5*r_23;
        const double x15 = Px*r_11;
        const double x16 = r_21*x15;
        const double x17 = Px*r_12;
        const double x18 = r_22*x17;
        const double x19 = Px*r_13;
        const double x20 = r_23*x19;
        const double x21 = Pz*r_31;
        const double x22 = r_21*x21;
        const double x23 = Pz*r_32;
        const double x24 = r_22*x23;
        const double x25 = Pz*r_33;
        const double x26 = r_23*x25;
        const double x27 = x11 + x13 - x14 + x16 + x18 + x20 + x22 + x24 + x26 + x9;
        const double x28 = d_1 + x27;
        const double x29 = x28 + x7;
        const double x30 = d_4*x1;
        const double x31 = d_3*x1;
        const double x32 = -x31;
        const double x33 = d_3*x3;
        const double x34 = x28 + x6;
        const double x35 = Py*r_21;
        const double x36 = x15 + x21 + x35;
        const double x37 = R_l_inv_50*a_0;
        const double x38 = x36*x37;
        const double x39 = Py*r_23;
        const double x40 = -d_5 + x19 + x25 + x39;
        const double x41 = R_l_inv_52*a_0;
        const double x42 = x40*x41;
        const double x43 = d_5*r_21;
        const double x44 = r_23*x15;
        const double x45 = r_23*x21;
        const double x46 = r_21*x19;
        const double x47 = r_21*x25;
        const double x48 = x43 + x44 + x45 - x46 - x47;
        const double x49 = R_l_inv_57*a_0;
        const double x50 = x48*x49;
        const double x51 = a_0*r_22;
        const double x52 = R_l_inv_54*x51;
        const double x53 = -x52;
        const double x54 = R_l_inv_56*d_3*x51;
        const double x55 = R_l_inv_55*a_0;
        const double x56 = 2*d_3;
        const double x57 = x40*x56;
        const double x58 = x55*x57;
        const double x59 = x38 + x42 + x50 + x53 + x54 + x58;
        const double x60 = d_4*r_21;
        const double x61 = a_0*r_21;
        const double x62 = R_l_inv_53*x61;
        const double x63 = d_5*r_22;
        const double x64 = r_23*x17;
        const double x65 = r_23*x23;
        const double x66 = r_22*x19;
        const double x67 = r_22*x25;
        const double x68 = x63 + x64 + x65 - x66 - x67;
        const double x69 = R_l_inv_56*a_0;
        const double x70 = x68*x69;
        const double x71 = Py*r_22;
        const double x72 = x17 + x23 + x71;
        const double x73 = R_l_inv_51*a_0;
        const double x74 = x72*x73;
        const double x75 = d_3*x61;
        const double x76 = R_l_inv_57*x75;
        const double x77 = 2*d_4;
        const double x78 = x55*x72;
        const double x79 = x77*x78;
        const double x80 = x60 + x62 + x70 - x74 + x76 + x79;
        const double x81 = d_4*r_23;
        const double x82 = x69*x81;
        const double x83 = r_22*x15;
        const double x84 = r_22*x21;
        const double x85 = r_21*x17;
        const double x86 = r_21*x23;
        const double x87 = x83 + x84 - x85 - x86;
        const double x88 = -x82 + x87;
        const double x89 = d_4*x73;
        const double x90 = std::pow(d_3, 2);
        const double x91 = std::pow(d_4, 2);
        const double x92 = std::pow(d_5, 2);
        const double x93 = std::pow(a_0, 2);
        const double x94 = std::pow(a_2, 2);
        const double x95 = -std::pow(d_1, 2);
        const double x96 = 2*d_5;
        const double x97 = Py*x14;
        const double x98 = Py*x1;
        const double x99 = 2*x17;
        const double x100 = 2*x19;
        const double x101 = 2*x15;
        const double x102 = 2*x71;
        const double x103 = 2*x39;
        const double x104 = std::pow(Px, 2);
        const double x105 = std::pow(r_11, 2);
        const double x106 = x104*x105;
        const double x107 = std::pow(r_12, 2);
        const double x108 = x104*x107;
        const double x109 = std::pow(r_13, 2);
        const double x110 = x104*x109;
        const double x111 = std::pow(Py, 2);
        const double x112 = x111*x8;
        const double x113 = x10*x111;
        const double x114 = x111*x12;
        const double x115 = std::pow(Pz, 2);
        const double x116 = std::pow(r_31, 2);
        const double x117 = x115*x116;
        const double x118 = std::pow(r_32, 2);
        const double x119 = x115*x118;
        const double x120 = std::pow(r_33, 2)*x115;
        const double x121 = x100*x25 + x100*x39 + x101*x21 + x102*x23 + x103*x25 + x106 + x108 + x110 + x112 + x113 + x114 + x117 + x119 + x120 + x15*x98 - x19*x96 + x21*x98 + x23*x99 - x25*x96 + x71*x99 + x90 + x91 + x92 - x93 - x94 + x95 - 2*x97;
        const double x122 = -R_l_inv_52*a_0*d_3 - R_l_inv_55*a_0*x121 - a_2 + x89;
        const double x123 = 2*x36;
        const double x124 = x123*x73;
        const double x125 = 2*x37;
        const double x126 = x125*x72;
        const double x127 = R_l_inv_54*a_0;
        const double x128 = x1*x127;
        const double x129 = 2*x49;
        const double x130 = x129*x68;
        const double x131 = x31*x69;
        const double x132 = -x131;
        const double x133 = x126 + x128 + x130 + x132;
        const double x134 = 2*x4;
        const double x135 = 2*x69;
        const double x136 = x135*x48;
        const double x137 = 2*R_l_inv_53;
        const double x138 = x137*x51;
        const double x139 = x51*x56;
        const double x140 = R_l_inv_57*x139;
        const double x141 = x134 - x136 + x138 + x140;
        const double x142 = d_3*x41;
        const double x143 = x121*x55;
        const double x144 = -x89;
        const double x145 = a_2 + x142 + x143 + x144 + x38;
        const double x146 = -x83;
        const double x147 = -x84;
        const double x148 = -x42;
        const double x149 = -x58;
        const double x150 = x146 + x147 + x148 + x149 + x82 + x85 + x86;
        const double x151 = x50 + x53 + x54;
        const double x152 = 2*x81;
        const double x153 = a_0*r_23;
        const double x154 = x137*x153;
        const double x155 = x135*x87;
        const double x156 = x129*x6;
        const double x157 = x125*x40 + x152 + x154 + x155 + x156;
        const double x158 = x123*x41;
        const double x159 = 4*d_3;
        const double x160 = x36*x55;
        const double x161 = x159*x160;
        const double x162 = -x158 - x161;
        const double x163 = 2*x63;
        const double x164 = 2*x64;
        const double x165 = 2*x65;
        const double x166 = 2*x66;
        const double x167 = 2*x67;
        const double x168 = x30*x69;
        const double x169 = -x163 - x164 - x165 + x166 + x167 + x168;
        const double x170 = 4*x72;
        const double x171 = x170*x41;
        const double x172 = 8*d_3;
        const double x173 = x172*x78;
        const double x174 = 4*x43;
        const double x175 = 4*x46;
        const double x176 = 4*x47;
        const double x177 = 4*x44;
        const double x178 = d_4*x3;
        const double x179 = x174 - x175 - x176 + x177 + x178*x69 + 4*x45;
        const double x180 = x158 + x161;
        const double x181 = x163 + x164 + x165 - x166 - x167 - x168;
        const double x182 = -x50;
        const double x183 = -x54;
        const double x184 = x182 + x183 + x42 + x52 + x58;
        const double x185 = x74 - x79;
        const double x186 = x185 + x60 + x62 + x70 + x76 + x88;
        const double x187 = 4*d_4;
        const double x188 = -x124 + x160*x187;
        const double x189 = x141 + x188;
        const double x190 = x122 + x185 + x60 + x62 + x70 + x76;
        const double x191 = 2*a_0;
        const double x192 = a_2*x191;
        const double x193 = d_1*x191;
        const double x194 = a_2*d_1;
        const double x195 = 2*x194;
        const double x196 = x93 + x94 + x95;
        const double x197 = R_l_inv_22*x192 + R_l_inv_32*x193 + R_l_inv_62*x196 + R_l_inv_72*x195;
        const double x198 = d_3*x197;
        const double x199 = std::pow(r_21, 3)*x111;
        const double x200 = R_l_inv_25*x192 + R_l_inv_35*x193 + R_l_inv_65*x196 + R_l_inv_75*x195;
        const double x201 = x121*x200;
        const double x202 = R_l_inv_21*x192 + R_l_inv_31*x193 + R_l_inv_61*x196 + R_l_inv_71*x195;
        const double x203 = d_4*x202;
        const double x204 = -x203;
        const double x205 = -r_21*x90;
        const double x206 = -r_21*x91;
        const double x207 = -r_21*x92;
        const double x208 = R_l_inv_23*x192 + R_l_inv_33*x193 + R_l_inv_63*x196 + R_l_inv_73*x195;
        const double x209 = -r_21*x208;
        const double x210 = R_l_inv_20*x192 + R_l_inv_30*x193 + R_l_inv_60*x196 + R_l_inv_70*x195;
        const double x211 = x210*x36;
        const double x212 = -x211;
        const double x213 = R_l_inv_66*x196;
        const double x214 = R_l_inv_26*x192 + R_l_inv_36*x193 + R_l_inv_76*x195 + x213;
        const double x215 = -x214*x68;
        const double x216 = r_21*x106;
        const double x217 = r_21*x113;
        const double x218 = r_21*x114;
        const double x219 = r_21*x117;
        const double x220 = R_l_inv_67*x196;
        const double x221 = R_l_inv_27*x192 + R_l_inv_37*x193 + R_l_inv_77*x195 + x220;
        const double x222 = d_3*r_21;
        const double x223 = -x221*x222;
        const double x224 = -r_21*x108;
        const double x225 = -r_21*x110;
        const double x226 = -r_21*x119;
        const double x227 = -r_21*x120;
        const double x228 = -x101*x14;
        const double x229 = 2*x21;
        const double x230 = -x14*x229;
        const double x231 = x101*x9;
        const double x232 = x101*x11;
        const double x233 = x101*x13;
        const double x234 = d_5*x1;
        const double x235 = x19*x234;
        const double x236 = x229*x9;
        const double x237 = x11*x229;
        const double x238 = x13*x229;
        const double x239 = x234*x25;
        const double x240 = 2*r_11;
        const double x241 = r_12*x104;
        const double x242 = r_22*x241;
        const double x243 = x240*x242;
        const double x244 = 2*r_13;
        const double x245 = r_23*x104;
        const double x246 = r_11*x244*x245;
        const double x247 = 2*r_31;
        const double x248 = r_32*x115;
        const double x249 = r_22*x248;
        const double x250 = x247*x249;
        const double x251 = r_23*r_33;
        const double x252 = x115*x251;
        const double x253 = x247*x252;
        const double x254 = x1*x17;
        const double x255 = -x23*x254;
        const double x256 = x19*x25;
        const double x257 = -x1*x256;
        const double x258 = x15*x21;
        const double x259 = x1*x258;
        const double x260 = x101*x24;
        const double x261 = x101*x26;
        const double x262 = x18*x229;
        const double x263 = x20*x229;
        const double x264 = x198 + x199 + x201 + x204 + x205 + x206 + x207 + x209 + x212 + x215 + x216 + x217 + x218 + x219 + x223 + x224 + x225 + x226 + x227 + x228 + x230 + x231 + x232 + x233 + x235 + x236 + x237 + x238 + x239 + x243 + x246 + x250 + x253 + x255 + x257 + x259 + x260 + x261 + x262 + x263;
        const double x265 = x202*x72;
        const double x266 = x197*x40;
        const double x267 = -x266;
        const double x268 = x200*x57;
        const double x269 = -x268;
        const double x270 = x200*x72;
        const double x271 = x270*x77;
        const double x272 = -x271;
        const double x273 = x265 + x267 + x269 + x272;
        const double x274 = x214*x81;
        const double x275 = x134*x15;
        const double x276 = x134*x21;
        const double x277 = x17*x30;
        const double x278 = x23*x30;
        const double x279 = x274 - x275 - x276 + x277 + x278;
        const double x280 = R_l_inv_24*x192 + R_l_inv_34*x193 + R_l_inv_64*x196 + R_l_inv_74*x195;
        const double x281 = r_22*x280;
        const double x282 = x221*x48;
        const double x283 = d_3*r_22;
        const double x284 = x214*x283;
        const double x285 = d_5*x31;
        const double x286 = x101*x6;
        const double x287 = x229*x6;
        const double x288 = x19*x31;
        const double x289 = x25*x31;
        const double x290 = x281 - x282 - x284 - x285 - x286 - x287 + x288 + x289;
        const double x291 = x279 + x290;
        const double x292 = 4*a_0;
        const double x293 = a_2*x292;
        const double x294 = d_1*x292;
        const double x295 = 4*x194;
        const double x296 = R_l_inv_26*x293 + R_l_inv_36*x294 + R_l_inv_76*x295 + 2*x213;
        const double x297 = x296*x48;
        const double x298 = 2*x23;
        const double x299 = x102 + x298 + x99;
        const double x300 = x210*x299;
        const double x301 = r_22*x90;
        const double x302 = -2*x301;
        const double x303 = r_22*x91;
        const double x304 = -2*x303;
        const double x305 = r_22*x92;
        const double x306 = -2*x305;
        const double x307 = 2*x208;
        const double x308 = -r_22*x307;
        const double x309 = std::pow(r_22, 3)*x111;
        const double x310 = 2*x309;
        const double x311 = r_22*x56;
        const double x312 = -x221*x311;
        const double x313 = r_22*x106;
        const double x314 = -2*x313;
        const double x315 = r_22*x110;
        const double x316 = -2*x315;
        const double x317 = r_22*x117;
        const double x318 = -2*x317;
        const double x319 = r_22*x120;
        const double x320 = -2*x319;
        const double x321 = r_22*x108;
        const double x322 = 2*x321;
        const double x323 = r_22*x112;
        const double x324 = 2*x323;
        const double x325 = r_22*x114;
        const double x326 = 2*x325;
        const double x327 = r_22*x119;
        const double x328 = 2*x327;
        const double x329 = 4*x17;
        const double x330 = -x14*x329;
        const double x331 = 4*x23;
        const double x332 = -x14*x331;
        const double x333 = x329*x9;
        const double x334 = x11*x329;
        const double x335 = x13*x329;
        const double x336 = d_5*x3;
        const double x337 = x19*x336;
        const double x338 = x331*x9;
        const double x339 = x11*x331;
        const double x340 = x13*x331;
        const double x341 = x25*x336;
        const double x342 = 4*r_11;
        const double x343 = r_21*x342;
        const double x344 = x241*x343;
        const double x345 = 4*r_12;
        const double x346 = r_13*x245;
        const double x347 = x345*x346;
        const double x348 = 4*x248;
        const double x349 = r_21*r_31;
        const double x350 = x348*x349;
        const double x351 = x251*x348;
        const double x352 = -x258*x3;
        const double x353 = -x256*x3;
        const double x354 = x16*x331;
        const double x355 = x22*x329;
        const double x356 = x17*x3;
        const double x357 = x23*x356;
        const double x358 = x26*x329;
        const double x359 = x20*x331;
        const double x360 = x297 - x300 + x302 + x304 + x306 + x308 + x310 + x312 + x314 + x316 + x318 + x320 + x322 + x324 + x326 + x328 + x330 + x332 + x333 + x334 + x335 + x337 + x338 + x339 + x340 + x341 + x344 + x347 + x350 + x351 + x352 + x353 + x354 + x355 + x357 + x358 + x359;
        const double x361 = x101 + x229 + x98;
        const double x362 = x202*x361;
        const double x363 = x200*x36;
        const double x364 = x187*x363 - x362;
        const double x365 = R_l_inv_27*x293 + R_l_inv_37*x294 + R_l_inv_77*x295 + 2*x220;
        const double x366 = x365*x68;
        const double x367 = x1*x280;
        const double x368 = d_5*x33;
        const double x369 = x214*x31;
        const double x370 = x329*x6;
        const double x371 = x331*x6;
        const double x372 = x19*x33;
        const double x373 = x25*x33;
        const double x374 = -x366 - x367 - x368 + x369 - x370 - x371 + x372 + x373;
        const double x375 = -x274;
        const double x376 = -x277;
        const double x377 = -x278;
        const double x378 = x266 + x268 + x275 + x276 + x375 + x376 + x377;
        const double x379 = -d_3*x197;
        const double x380 = -x121*x200;
        const double x381 = x199 + x203 + x205 + x206 + x207 + x209 + x212 + x215 + x216 + x217 + x218 + x219 + x223 + x224 + x225 + x226 + x227 + x228 + x230 + x231 + x232 + x233 + x235 + x236 + x237 + x238 + x239 + x243 + x246 + x250 + x253 + x255 + x257 + x259 + x260 + x261 + x262 + x263 + x379 + x380;
        const double x382 = x265 + x272;
        const double x383 = x197*x361;
        const double x384 = 2*x25;
        const double x385 = x100 + x103 + x384 - x96;
        const double x386 = x159*x363;
        const double x387 = -x210*x385 + x383 + x386;
        const double x388 = x214*x30;
        const double x389 = d_5*x178;
        const double x390 = x178*x19;
        const double x391 = x178*x25;
        const double x392 = x187*x64;
        const double x393 = x187*x65;
        const double x394 = -x388 + x389 - x390 - x391 + x392 + x393;
        const double x395 = x1*x23;
        const double x396 = -x254 - x395 + 2*x83 + 2*x84;
        const double x397 = x214*x396;
        const double x398 = r_23*x90;
        const double x399 = 2*x398;
        const double x400 = r_23*x91;
        const double x401 = 2*x400;
        const double x402 = r_23*x307;
        const double x403 = r_23*x92;
        const double x404 = 2*x403;
        const double x405 = std::pow(r_23, 3)*x111;
        const double x406 = 2*x405;
        const double x407 = 4*d_5;
        const double x408 = x407*x9;
        const double x409 = x11*x407;
        const double x410 = x13*x407;
        const double x411 = 2*x6;
        const double x412 = x221*x411;
        const double x413 = r_23*x106;
        const double x414 = 2*x413;
        const double x415 = r_23*x108;
        const double x416 = 2*x415;
        const double x417 = r_23*x117;
        const double x418 = 2*x417;
        const double x419 = r_23*x119;
        const double x420 = 2*x419;
        const double x421 = r_23*x110;
        const double x422 = 2*x421;
        const double x423 = r_23*x112;
        const double x424 = 2*x423;
        const double x425 = r_23*x113;
        const double x426 = 2*x425;
        const double x427 = r_23*x120;
        const double x428 = 2*x427;
        const double x429 = x15*x174;
        const double x430 = x17*x336;
        const double x431 = 4*x19;
        const double x432 = x14*x431;
        const double x433 = x174*x21;
        const double x434 = x23*x336;
        const double x435 = 4*x25;
        const double x436 = x14*x435;
        const double x437 = x431*x9;
        const double x438 = x11*x431;
        const double x439 = x13*x431;
        const double x440 = x435*x9;
        const double x441 = x11*x435;
        const double x442 = x13*x435;
        const double x443 = r_13*x104;
        const double x444 = x343*x443;
        const double x445 = x241*x3;
        const double x446 = r_13*x445;
        const double x447 = r_33*x115;
        const double x448 = 4*x349*x447;
        const double x449 = x248*x3;
        const double x450 = r_33*x449;
        const double x451 = x177*x21;
        const double x452 = x331*x64;
        const double x453 = x16*x435;
        const double x454 = x25*x356;
        const double x455 = x22*x431;
        const double x456 = x23*x3;
        const double x457 = x19*x456;
        const double x458 = x20*x435;
        const double x459 = -x397 - x399 - x401 - x402 + x404 + x406 - x408 - x409 - x410 - x412 - x414 - x416 - x418 - x420 + x422 + x424 + x426 + x428 - x429 - x430 - x432 - x433 - x434 - x436 + x437 + x438 + x439 + x440 + x441 + x442 + x444 + x446 + x448 + x450 - x451 - x452 + x453 + x454 + x455 + x457 + x458;
        const double x460 = Py*x3;
        const double x461 = x329 + x331 + x460;
        const double x462 = x197*x461;
        const double x463 = x172*x270;
        const double x464 = 8*d_5;
        const double x465 = 8*x60;
        const double x466 = 8*d_4;
        const double x467 = x178*x214 - x19*x465 - x25*x465 + x44*x466 + x45*x466 + x464*x60;
        const double x468 = x210*x385 + x383 + x386;
        const double x469 = x397 + x399 + x401 + x402 - x404 - x406 + x408 + x409 + x410 + x412 + x414 + x416 + x418 + x420 - x422 - x424 - x426 - x428 + x429 + x430 + x432 + x433 + x434 + x436 - x437 - x438 - x439 - x440 - x441 - x442 - x444 - x446 - x448 - x450 + x451 + x452 - x453 - x454 - x455 - x457 - x458;
        const double x470 = -x265 + x271;
        const double x471 = x267 + x269 + x470;
        const double x472 = -x281 + x282 + x284 + x285 + x286 + x287 - x288 - x289;
        const double x473 = x279 + x472;
        const double x474 = -4*d_4*x200*x36 + x362;
        const double x475 = x366 + x367 + x368 - x369 + x370 + x371 - x372 - x373;
        const double x476 = R_l_inv_23*x193 - R_l_inv_33*x192 + R_l_inv_63*x195 - R_l_inv_73*x196;
        const double x477 = r_21*x476;
        const double x478 = R_l_inv_20*x193 - R_l_inv_30*x192 + R_l_inv_60*x195 - R_l_inv_70*x196;
        const double x479 = x36*x478;
        const double x480 = R_l_inv_22*x193 - R_l_inv_32*x192 + R_l_inv_62*x195 - R_l_inv_72*x196;
        const double x481 = x40*x480;
        const double x482 = R_l_inv_36*x192;
        const double x483 = R_l_inv_76*x196;
        const double x484 = R_l_inv_26*x193 + R_l_inv_66*x195 - x482 - x483;
        const double x485 = x484*x68;
        const double x486 = R_l_inv_77*x196;
        const double x487 = R_l_inv_27*x193 - R_l_inv_37*x192 + R_l_inv_67*x195 - x486;
        const double x488 = x222*x487;
        const double x489 = x484*x81;
        const double x490 = -x489;
        const double x491 = x56*x63;
        const double x492 = -x491;
        const double x493 = x6*x77;
        const double x494 = R_l_inv_25*x193 - R_l_inv_35*x192 + R_l_inv_65*x195 - R_l_inv_75*x196;
        const double x495 = x494*x57;
        const double x496 = x6*x99;
        const double x497 = -x496;
        const double x498 = x298*x6;
        const double x499 = -x498;
        const double x500 = x56*x66;
        const double x501 = x56*x67;
        const double x502 = x477 + x479 + x481 + x485 + x488 + x490 + x492 + x493 + x495 + x497 + x499 + x500 + x501;
        const double x503 = x48*x487;
        const double x504 = -x301;
        const double x505 = -x305;
        const double x506 = R_l_inv_24*x193 - R_l_inv_34*x192 + R_l_inv_64*x195 - R_l_inv_74*x196;
        const double x507 = r_22*x506;
        const double x508 = -x507;
        const double x509 = R_l_inv_21*x193 - R_l_inv_31*x192 + R_l_inv_61*x195 - R_l_inv_71*x196;
        const double x510 = x509*x72;
        const double x511 = x283*x484;
        const double x512 = -x313;
        const double x513 = -x315;
        const double x514 = -x317;
        const double x515 = -x319;
        const double x516 = x494*x72;
        const double x517 = x516*x77;
        const double x518 = x14*x99;
        const double x519 = -x518;
        const double x520 = x14*x298;
        const double x521 = -x520;
        const double x522 = x9*x99;
        const double x523 = x11*x99;
        const double x524 = x13*x99;
        const double x525 = x163*x19;
        const double x526 = x298*x9;
        const double x527 = x11*x298;
        const double x528 = x13*x298;
        const double x529 = x163*x25;
        const double x530 = r_11*x1;
        const double x531 = x241*x530;
        const double x532 = r_23*x241*x244;
        const double x533 = r_31*x1;
        const double x534 = x248*x533;
        const double x535 = x248*x251;
        const double x536 = 2*x535;
        const double x537 = x229*x83;
        const double x538 = -x537;
        const double x539 = x384*x66;
        const double x540 = -x539;
        const double x541 = x15*x395;
        const double x542 = x21*x254;
        const double x543 = x18*x298;
        const double x544 = x26*x99;
        const double x545 = x20*x298;
        const double x546 = x303 + x309 + x321 + x323 + x325 + x327 + x503 + x504 + x505 + x508 - x510 + x511 + x512 + x513 + x514 + x515 + x517 + x519 + x521 + x522 + x523 + x524 + x525 + x526 + x527 + x528 + x529 + x531 + x532 + x534 + x536 + x538 + x540 + x541 + x542 + x543 + x544 + x545;
        const double x547 = d_4*x509;
        const double x548 = -d_3*x480 - x121*x494 + x547;
        const double x549 = x77*x9;
        const double x550 = x11*x77;
        const double x551 = x13*x77;
        const double x552 = x14*x77;
        const double x553 = x15*x30;
        const double x554 = x134*x17;
        const double x555 = x20*x77;
        const double x556 = x21*x30;
        const double x557 = x134*x23;
        const double x558 = x26*x77;
        const double x559 = -x549 - x550 - x551 + x552 - x553 - x554 - x555 - x556 - x557 - x558;
        const double x560 = x299*x478;
        const double x561 = R_l_inv_27*x294 - R_l_inv_37*x293 + R_l_inv_67*x295 - 2*x486;
        const double x562 = -x561*x68;
        const double x563 = x1*x90;
        const double x564 = -x563;
        const double x565 = -x1*x506;
        const double x566 = x1*x91;
        const double x567 = x31*x484;
        const double x568 = 4*x15;
        const double x569 = 4*x21;
        const double x570 = r_11*x445 + 4*r_31*x252 + r_31*x449 + x1*x106 - x1*x108 - x1*x110 + x1*x113 + x1*x114 + x1*x117 - x1*x119 - x1*x120 - x1*x92 + x11*x568 + x11*x569 + x13*x568 + x13*x569 - x14*x568 - x14*x569 + x15*x456 + x16*x569 + x174*x19 + x174*x25 - x175*x25 + 2*x199 + x20*x569 + x21*x356 + x26*x568 - x331*x85 + x342*x346 + x568*x9 + x569*x9;
        const double x571 = -x560 + x562 + x564 + x565 + x566 + x567 + x570;
        const double x572 = x361*x509;
        const double x573 = x36*x494;
        const double x574 = x187*x573 - x572;
        const double x575 = R_l_inv_26*x294 - R_l_inv_36*x293 + R_l_inv_66*x295 - 2*x483;
        const double x576 = x48*x575;
        const double x577 = 2*x476;
        const double x578 = r_22*x577;
        const double x579 = d_3*x174;
        const double x580 = x311*x487;
        const double x581 = x568*x6;
        const double x582 = x569*x6;
        const double x583 = d_3*x175;
        const double x584 = d_3*x176;
        const double x585 = x576 - x578 - x579 - x580 - x581 - x582 + x583 + x584;
        const double x586 = d_3*x480;
        const double x587 = x121*x494;
        const double x588 = -x547;
        const double x589 = -x493;
        const double x590 = x479 + x489 + x586 + x587 + x588 + x589;
        const double x591 = -x481;
        const double x592 = -x495;
        const double x593 = x477 + x485 + x488 + x492 + x497 + x499 + x500 + x501 + x591 + x592;
        const double x594 = x549 + x550 + x551 - x552 + x553 + x554 + x555 + x556 + x557 + x558;
        const double x595 = x546 + x594;
        const double x596 = x361*x480;
        const double x597 = x159*x573;
        const double x598 = x385*x478;
        const double x599 = x159*x60;
        const double x600 = -x599;
        const double x601 = x30*x484;
        const double x602 = x598 + x600 + x601;
        const double x603 = x396*x484;
        const double x604 = r_23*x577;
        const double x605 = x411*x487;
        const double x606 = x15*x33;
        const double x607 = x21*x33;
        const double x608 = x159*x85;
        const double x609 = x159*x86;
        const double x610 = x603 + x604 + x605 - x606 - x607 + x608 + x609;
        const double x611 = x178*x484;
        const double x612 = x172*x4;
        const double x613 = x172*x516 + x461*x480;
        const double x614 = x596 + x597;
        const double x615 = x610 + x614;
        const double x616 = x586 + x587 + x588;
        const double x617 = x510 - x517;
        const double x618 = x594 + x617;
        const double x619 = x560 + x562 + x564 + x565 + x566 + x567 + x570;
        const double x620 = -x576 + x578 + x579 + x580 + x581 + x582 - x583 - x584;
        const double x621 = -x479;
        const double x622 = x481 + x495 + x621;
        const double x623 = x490 + x493 + x616;
        const double x624 = -x477 - x485 - x488 + x491 + x496 + x498 - x500 - x501;
        const double x625 = x4*x56;
        const double x626 = x4*x96;
        const double x627 = x134*x19;
        const double x628 = -x627;
        const double x629 = x134*x25;
        const double x630 = -x629;
        const double x631 = R_l_inv_41*x193*x72;
        const double x632 = -x631;
        const double x633 = R_l_inv_44*d_1;
        const double x634 = 2*x51*x633;
        const double x635 = -x634;
        const double x636 = x64*x77;
        const double x637 = x65*x77;
        const double x638 = a_0*d_1;
        const double x639 = R_l_inv_40*x638;
        const double x640 = x123*x639;
        const double x641 = R_l_inv_43*x638;
        const double x642 = x1*x641;
        const double x643 = R_l_inv_46*x193*x68;
        const double x644 = R_l_inv_47*x193*x48;
        const double x645 = R_l_inv_46*d_1*x139;
        const double x646 = R_l_inv_47*x638;
        const double x647 = x31*x646;
        const double x648 = R_l_inv_45*x638;
        const double x649 = x187*x648*x72;
        const double x650 = x625 + x626 + x628 + x630 + x632 + x635 + x636 + x637 + x640 + x642 + x643 + x644 + x645 + x647 + x649;
        const double x651 = R_l_inv_42*x56*x638;
        const double x652 = -x651;
        const double x653 = R_l_inv_45*x121*x193;
        const double x654 = -x653;
        const double x655 = R_l_inv_41*x638*x77;
        const double x656 = R_l_inv_42*x193*x40;
        const double x657 = x294*x40;
        const double x658 = R_l_inv_45*d_3;
        const double x659 = x657*x658;
        const double x660 = x652 + x654 + x655 + x656 + x659;
        const double x661 = x56*x9;
        const double x662 = x11*x56;
        const double x663 = x13*x56;
        const double x664 = x6*x96;
        const double x665 = x15*x31;
        const double x666 = x18*x56;
        const double x667 = x100*x6;
        const double x668 = x21*x31;
        const double x669 = x24*x56;
        const double x670 = x384*x6;
        const double x671 = x9*x96;
        const double x672 = x11*x96;
        const double x673 = x13*x96;
        const double x674 = x15*x234;
        const double x675 = x163*x17;
        const double x676 = x100*x14;
        const double x677 = x21*x234;
        const double x678 = x163*x23;
        const double x679 = x14*x384;
        const double x680 = x100*x9;
        const double x681 = x100*x11;
        const double x682 = x100*x13;
        const double x683 = x384*x9;
        const double x684 = x11*x384;
        const double x685 = x13*x384;
        const double x686 = x443*x530;
        const double x687 = x242*x244;
        const double x688 = x447*x533;
        const double x689 = 2*r_33*x249;
        const double x690 = x229*x44;
        const double x691 = x298*x64;
        const double x692 = R_l_inv_46*x638;
        const double x693 = x152*x692;
        const double x694 = x1*x15*x25;
        const double x695 = x18*x384;
        const double x696 = x1*x19*x21;
        const double x697 = x100*x24;
        const double x698 = x20*x384;
        const double x699 = x398 - x400 + x403 + x405 - x413 - x415 - x417 - x419 + x421 + x423 + x425 + x427 - x671 - x672 - x673 - x674 - x675 - x676 - x677 - x678 - x679 + x680 + x681 + x682 + x683 + x684 + x685 + x686 + x687 + x688 + x689 - x690 - x691 - x693 + x694 + x695 + x696 + x697 + x698;
        const double x700 = -x661 - x662 - x663 + x664 - x665 - x666 - x667 - x668 - x669 - x670 + x699;
        const double x701 = x170*x639;
        const double x702 = x294*x36;
        const double x703 = x36*x648;
        const double x704 = x466*x703;
        const double x705 = R_l_inv_41*x702 + x701 - x704;
        const double x706 = 4*x61*x633;
        const double x707 = R_l_inv_47*x294;
        const double x708 = x68*x707;
        const double x709 = 4*d_1;
        const double x710 = R_l_inv_46*x709*x75;
        const double x711 = x600 + x706 + x708 - x710;
        const double x712 = x407*x60;
        const double x713 = x187*x44;
        const double x714 = x187*x45;
        const double x715 = R_l_inv_46*x294;
        const double x716 = x48*x715;
        const double x717 = x431*x60;
        const double x718 = x435*x60;
        const double x719 = x3*x641;
        const double x720 = x33*x646;
        const double x721 = -x712 - x713 - x714 - x716 + x717 + x718 + x719 + x720;
        const double x722 = -2*R_l_inv_41*a_0*d_1*d_4;
        const double x723 = -x656;
        const double x724 = -x659;
        const double x725 = x651 + x653 + x722 + x723 + x724;
        const double x726 = x661 + x662 + x663 - x664 + x665 + x666 + x667 + x668 + x669 + x670;
        const double x727 = -x398 + x400 - x403 - x405 + x413 + x415 + x417 + x419 - x421 - x423 - x425 - x427 + x671 + x672 + x673 + x674 + x675 + x676 + x677 + x678 + x679 - x680 - x681 - x682 - x683 - x684 - x685 - x686 - x687 - x688 - x689 + x690 + x691 + x693 - x694 - x695 - x696 - x697 - x698 + x726;
        const double x728 = -x566;
        const double x729 = R_l_inv_40*x657;
        const double x730 = -x60*x715;
        const double x731 = x563 + x570 + x728 - x729 + x730;
        const double x732 = R_l_inv_42*x702;
        const double x733 = x172*x703;
        const double x734 = x732 + x733;
        const double x735 = x15*x178;
        const double x736 = x178*x21;
        const double x737 = R_l_inv_43*x153*x709;
        const double x738 = x715*x87;
        const double x739 = x329*x60;
        const double x740 = x331*x60;
        const double x741 = x6*x707;
        const double x742 = -x735 - x736 - x737 - x738 + x739 + x740 - x741;
        const double x743 = 8*a_0;
        const double x744 = d_1*x743;
        const double x745 = 8*x14;
        const double x746 = 8*x17;
        const double x747 = 8*x63;
        const double x748 = 8*x23;
        const double x749 = 8*r_12;
        const double x750 = r_11*r_21;
        const double x751 = 8*x248;
        const double x752 = 8*x21;
        const double x753 = 8*x25;
        const double x754 = R_l_inv_46*x744;
        const double x755 = x104*x749*x750 - x106*x3 + x108*x3 + x11*x746 + x11*x748 - x110*x3 + x112*x3 + x114*x3 - x117*x3 + x119*x3 - x120*x3 + x13*x746 + x13*x748 + x16*x748 - x17*x745 + x18*x748 + x19*x747 + x20*x748 + x22*x746 - x23*x745 + x25*x747 + x26*x746 + x3*x90 - x3*x91 - x3*x92 + 4*x309 + x346*x749 + x349*x751 - x4*x754 + 8*x535 - x66*x753 + x746*x9 + x748*x9 - x752*x83;
        const double x756 = x563 + x570 + x728 + x729 + x730;
        const double x757 = x735 + x736 + x737 + x738 - x739 - x740 + x741;
        const double x758 = -x649;
        const double x759 = x631 + x640 + x699 + x726 + x758;
        const double x760 = x651 + x653 + x656 + x659 + x722;
        const double x761 = x626 + x628 + x630 + x636 + x637 + x642 + x643 + x647;
        const double x762 = -x625 + x634 - x644 - x645 + x761;
        const double x763 = -4*R_l_inv_41*a_0*d_1*x36 + x701 + x704;
        const double x764 = x599 - x706 - x708 + x710;
        const double x765 = -x640;
        const double x766 = x625 + x635 + x644 + x645 + x765;
        const double x767 = x632 + x649 + x699 + x726;
        const double x768 = -x626 + x627 + x629 - x636 - x637 - x642 - x643 - x647;
        const double x769 = -x244;
        const double x770 = r_12*x77;
        const double x771 = -x770;
        const double x772 = d_3*x244;
        const double x773 = 2*Px;
        const double x774 = r_11*x229 + r_11*x98 + r_12*x102 + r_12*x298 - r_13*x96 + x105*x773 + x107*x773 + x109*x773 + x244*x25 + x244*x39;
        const double x775 = -x772 + x774;
        const double x776 = d_4*x342;
        const double x777 = d_3*x342;
        const double x778 = d_3*x749;
        const double x779 = x772 + x774;
        const double x780 = 2*r_12;
        const double x781 = x127*x780;
        const double x782 = d_5*r_11;
        const double x783 = r_13*x35;
        const double x784 = r_13*x21;
        const double x785 = r_11*x39;
        const double x786 = r_11*x25;
        const double x787 = x782 + x783 + x784 - x785 - x786;
        const double x788 = x129*x787;
        const double x789 = r_12*x56;
        const double x790 = x69*x789;
        const double x791 = R_l_inv_53*a_0;
        const double x792 = d_5*r_12;
        const double x793 = r_13*x71;
        const double x794 = r_13*x23;
        const double x795 = r_12*x39;
        const double x796 = r_12*x25;
        const double x797 = x792 + x793 + x794 - x795 - x796;
        const double x798 = r_11*x56;
        const double x799 = r_11*x77 + x135*x797 + x240*x791 + x49*x798;
        const double x800 = -x781 + x788 + x790 + x799;
        const double x801 = r_11*x102;
        const double x802 = r_11*x298;
        const double x803 = r_12*x98;
        const double x804 = r_12*x229;
        const double x805 = d_4*x244;
        const double x806 = x69*x805;
        const double x807 = -x801 - x802 + x803 + x804 - x806;
        const double x808 = x127*x342;
        const double x809 = 4*x49*x797;
        const double x810 = r_12*x187;
        const double x811 = 4*x69;
        const double x812 = d_3*x345;
        const double x813 = x345*x791 + x49*x812 - x787*x811 + x810;
        const double x814 = x801 + x802 - x803 - x804 + x806;
        const double x815 = 4*x792;
        const double x816 = r_13*x460;
        const double x817 = r_13*x331;
        const double x818 = 4*x795;
        const double x819 = 4*x796;
        const double x820 = x69*x776;
        const double x821 = r_13*x187;
        const double x822 = r_11*x71;
        const double x823 = r_11*x23;
        const double x824 = r_12*x35;
        const double x825 = r_12*x21;
        const double x826 = x822 + x823 - x824 - x825;
        const double x827 = 4*r_13;
        const double x828 = d_3*x827;
        const double x829 = x49*x828 + x791*x827 - x811*x826 + x821;
        const double x830 = 8*x782;
        const double x831 = 8*x785;
        const double x832 = 8*x783;
        const double x833 = 8*x786;
        const double x834 = 8*x784;
        const double x835 = d_4*x749;
        const double x836 = x781 - x788 - x790 + x799;
        const double x837 = x214*x805;
        const double x838 = Py*x345;
        const double x839 = x60*x838;
        const double x840 = x21*x810;
        const double x841 = r_11*x460;
        const double x842 = d_4*x841;
        const double x843 = x23*x776;
        const double x844 = std::pow(r_11, 3);
        const double x845 = 2*x104;
        const double x846 = Px*x105;
        const double x847 = 4*x35;
        const double x848 = Px*x847;
        const double x849 = Px*x569;
        const double x850 = d_5*x25;
        const double x851 = r_21*x111;
        const double x852 = r_12*x3;
        const double x853 = r_31*x248;
        const double x854 = r_23*x851;
        const double x855 = r_31*x447;
        const double x856 = x25*x39;
        const double x857 = x21*x35;
        const double x858 = 4*x39;
        const double x859 = -r_11*x307 + x107*x848 + x107*x849 + x108*x240 + x109*x848 + x109*x849 + x110*x240 + x112*x240 - x113*x240 - x114*x240 + x117*x240 - x119*x240 - x120*x240 - x221*x798 - x240*x90 - x240*x91 - x240*x92 - x296*x797 + x331*x824 + x342*x850 - x342*x856 + x342*x857 + x342*x97 + x345*x853 - x407*x783 - x407*x784 + x435*x783 - x460*x823 + x460*x825 + x569*x846 + x784*x858 + x827*x854 + x827*x855 + x844*x845 + x846*x847 + x851*x852;
        const double x860 = x837 - x839 - x840 + x842 + x843 + x859;
        const double x861 = x365*x787;
        const double x862 = x280*x780;
        const double x863 = d_5*x777;
        const double x864 = r_12*x214;
        const double x865 = x56*x864;
        const double x866 = x159*x783;
        const double x867 = x159*x784;
        const double x868 = Py*x6;
        const double x869 = x342*x868;
        const double x870 = x25*x777;
        const double x871 = -x861 + x862 - x863 - x865 - x866 - x867 + x869 + x870;
        const double x872 = a_2*x743;
        const double x873 = 8*x194;
        const double x874 = x797*(R_l_inv_27*x872 + R_l_inv_37*x744 + R_l_inv_77*x873 + 4*x220);
        const double x875 = x280*x342;
        const double x876 = d_5*x778;
        const double x877 = x172*x793;
        const double x878 = x172*x794;
        const double x879 = std::pow(r_12, 3);
        const double x880 = 4*x104;
        const double x881 = 8*x71;
        const double x882 = Px*x881;
        const double x883 = Px*x748;
        const double x884 = r_22*x111;
        const double x885 = 8*x884;
        const double x886 = r_13*r_23;
        const double x887 = r_13*r_33;
        const double x888 = 8*x35;
        const double x889 = x71*x749;
        const double x890 = 8*x39;
        const double x891 = -d_3*x221*x345 + r_11*r_31*x751 + x106*x345 + x107*x882 + x107*x883 + x109*x882 + x109*x883 + x110*x345 - x112*x345 + x113*x345 - x114*x345 - x117*x345 + x119*x345 - x120*x345 - x208*x345 + x23*x889 - x345*x90 - x345*x91 - x345*x92 - x464*x793 - x464*x794 + x748*x846 + x749*x850 - x749*x856 - x749*x857 + x749*x97 + x750*x885 + x751*x887 + x752*x822 + x753*x793 + x787*(R_l_inv_26*x872 + R_l_inv_36*x744 + R_l_inv_76*x873 + 4*x213) + x794*x890 + x823*x888 + x846*x881 + x879*x880 + x885*x886;
        const double x892 = -x837 + x839 + x840 - x842 - x843 + x859;
        const double x893 = x214*x776;
        const double x894 = d_5*x835;
        const double x895 = x39*x835;
        const double x896 = x25*x835;
        const double x897 = 8*r_13;
        const double x898 = Py*x4;
        const double x899 = x897*x898;
        const double x900 = x466*x794;
        const double x901 = 4*x824;
        const double x902 = x23*x342;
        const double x903 = -4*x825 + x841 - x901 + x902;
        const double x904 = std::pow(r_13, 3);
        const double x905 = Px*x890;
        const double x906 = Px*x753;
        const double x907 = x111*x750;
        const double x908 = 8*r_11;
        const double x909 = r_33*x248;
        const double x910 = x23*x749;
        const double x911 = -8*Px*d_5*x105 - 8*Px*d_5*x107 - 8*Px*d_5*x109 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_5*r_11*r_21 - 8*Py*d_5*r_12*r_22 - 8*Py*d_5*r_13*r_23 - 8*Pz*d_5*r_11*r_31 - 8*Pz*d_5*r_12*r_32 - 8*Pz*d_5*r_13*r_33 - 4*d_3*r_13*x221 - 4*r_13*x10*x111 - 4*r_13*x111*x8 - 4*r_13*x115*x116 - 4*r_13*x115*x118 - 4*r_13*x208 + r_13*x25*x890 - 4*r_13*x90 - 4*r_13*x91 + r_23*x749*x884 + 8*r_23*x907 + x106*x827 + x107*x905 + x107*x906 + x108*x827 + x109*x905 + x109*x906 + x114*x827 + x120*x827 + x21*x831 + x214*x903 + x25*x889 + x35*x833 + x39*x910 + x749*x909 + x753*x846 + x827*x92 + x846*x890 + x855*x908 + x880*x904;
        const double x912 = x861 - x862 + x863 + x865 + x866 + x867 - x869 - x870;
        const double x913 = x561*x787;
        const double x914 = -x780*x90;
        const double x915 = -x780*x92;
        const double x916 = -x506*x780;
        const double x917 = x780*x91;
        const double x918 = x845*x879;
        const double x919 = x484*x805;
        const double x920 = -x112*x780;
        const double x921 = -x114*x780;
        const double x922 = -x117*x780;
        const double x923 = -x120*x780;
        const double x924 = x484*x789;
        const double x925 = x106*x780;
        const double x926 = x110*x780;
        const double x927 = x113*x780;
        const double x928 = x119*x780;
        const double x929 = d_3*x821;
        const double x930 = -d_5*x816;
        const double x931 = -d_5*x817;
        const double x932 = x460*x846;
        const double x933 = Px*x460;
        const double x934 = x107*x933;
        const double x935 = x109*x933;
        const double x936 = x331*x846;
        const double x937 = Px*x331;
        const double x938 = x107*x937;
        const double x939 = x109*x937;
        const double x940 = x345*x97;
        const double x941 = x25*x815;
        const double x942 = x3*x907;
        const double x943 = x342*x853;
        const double x944 = x111*x3*x886;
        const double x945 = x827*x909;
        const double x946 = -x21*x901;
        const double x947 = -x25*x818;
        const double x948 = x35*x902;
        const double x949 = x21*x841;
        const double x950 = r_12*x23;
        const double x951 = x460*x950;
        const double x952 = x25*x816;
        const double x953 = x39*x817;
        const double x954 = x913 + x914 + x915 + x916 + x917 + x918 - x919 + x920 + x921 + x922 + x923 + x924 + x925 + x926 + x927 + x928 + x929 + x930 + x931 + x932 + x934 + x935 + x936 + x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x946 + x947 + x948 + x949 + x951 + x952 + x953;
        const double x955 = x575*x797;
        const double x956 = r_11*x577;
        const double x957 = d_3*x815;
        const double x958 = x487*x798;
        const double x959 = Py*x33;
        const double x960 = r_13*x959;
        const double x961 = d_3*x817;
        const double x962 = x6*x838;
        const double x963 = x159*x796;
        const double x964 = x955 + x956 - x957 + x958 - x960 - x961 + x962 + x963;
        const double x965 = d_5*x821;
        const double x966 = -4*Px*d_4*x105 - 4*Px*d_4*x107 - 4*Px*d_4*x109 - 4*Py*d_4*r_11*r_21 - 4*Py*d_4*r_12*r_22 - 4*Py*d_4*r_13*r_23 - 4*Pz*d_4*r_11*r_31 - 4*Pz*d_4*r_12*r_32 - 4*Pz*d_4*r_13*r_33 + x965;
        const double x967 = x787*(R_l_inv_26*x744 - R_l_inv_36*x872 + R_l_inv_66*x873 - 4*x483);
        const double x968 = x345*x476;
        const double x969 = d_3*x830;
        const double x970 = d_3*x487;
        const double x971 = x345*x970;
        const double x972 = x172*x783;
        const double x973 = x172*x784;
        const double x974 = x868*x908;
        const double x975 = x172*x786;
        const double x976 = x342*x90;
        const double x977 = x342*x91;
        const double x978 = Px*x888;
        const double x979 = Px*x752;
        const double x980 = r_22*x749*x851 + x107*x978 + x107*x979 + x108*x342 + x109*x978 + x109*x979 + x110*x342 + x112*x342 - x113*x342 - x114*x342 + x117*x342 - x119*x342 - x120*x342 + x21*x889 + x25*x830 - x25*x831 + x25*x832 - x342*x92 + x35*x910 + x39*x834 - x464*x783 - x464*x784 - x748*x822 + x749*x853 + x752*x846 + x844*x880 + x846*x888 + x854*x897 + x855*x897 + x857*x908 + x908*x97;
        const double x981 = -x342*x506 + x484*x777 - x797*(R_l_inv_27*x744 - R_l_inv_37*x872 + R_l_inv_67*x873 - 4*x486) - x976 + x977 + x980;
        const double x982 = Px*x187;
        const double x983 = Py*x60;
        const double x984 = d_4*r_12;
        const double x985 = x105*x982 + x107*x982 + x109*x982 + x21*x776 + x25*x821 + x331*x984 + x342*x983 + x39*x821 + x460*x984 - x965;
        const double x986 = x913 + x914 + x915 + x916 + x917 + x918 + x919 + x920 + x921 + x922 + x923 + x924 + x925 + x926 + x927 + x928 - x929 + x930 + x931 + x932 + x934 + x935 + x936 + x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x946 + x947 + x948 + x949 + x951 + x952 + x953;
        const double x987 = r_11*x466;
        const double x988 = d_3*x987;
        const double x989 = -x988;
        const double x990 = x484*x776;
        const double x991 = x172*x822 + x172*x823 - x21*x778 - x35*x778 + x476*x827 - x484*x903 + x827*x970;
        const double x992 = -x955 - x956 + x957 - x958 + x960 + x961 - x962 - x963;
        const double x993 = Px*x159;
        const double x994 = x105*x993;
        const double x995 = x107*x993;
        const double x996 = x109*x993;
        const double x997 = x35*x777;
        const double x998 = r_12*x959;
        const double x999 = x827*x868;
        const double x1000 = x21*x777;
        const double x1001 = d_3*r_12*x331;
        const double x1002 = x25*x828;
        const double x1003 = d_3*x810;
        const double x1004 = x187*x792;
        const double x1005 = x187*x795;
        const double x1006 = -x1005;
        const double x1007 = x187*x796;
        const double x1008 = -x1007;
        const double x1009 = R_l_inv_44*x345*x638;
        const double x1010 = -x1009;
        const double x1011 = d_4*x816;
        const double x1012 = d_4*x817;
        const double x1013 = x342*x641;
        const double x1014 = x715*x797;
        const double x1015 = x707*x787;
        const double x1016 = x692*x812;
        const double x1017 = x646*x777;
        const double x1018 = x1003 + x1004 + x1006 + x1008 + x1010 + x1011 + x1012 + x1013 + x1014 + x1015 + x1016 + x1017;
        const double x1019 = x244*x91;
        const double x1020 = x244*x90;
        const double x1021 = x244*x92;
        const double x1022 = x845*x904;
        const double x1023 = Px*x407;
        const double x1024 = x1023*x105;
        const double x1025 = x1023*x107;
        const double x1026 = x1023*x109;
        const double x1027 = x112*x244;
        const double x1028 = x113*x244;
        const double x1029 = x117*x244;
        const double x1030 = x119*x244;
        const double x1031 = x106*x244;
        const double x1032 = x108*x244;
        const double x1033 = x114*x244;
        const double x1034 = x120*x244;
        const double x1035 = d_5*x342;
        const double x1036 = x1035*x35;
        const double x1037 = x460*x792;
        const double x1038 = x827*x97;
        const double x1039 = x1035*x21;
        const double x1040 = x331*x792;
        const double x1041 = x827*x850;
        const double x1042 = x846*x858;
        const double x1043 = Px*x858;
        const double x1044 = x1043*x107;
        const double x1045 = x1043*x109;
        const double x1046 = x435*x846;
        const double x1047 = Px*x435;
        const double x1048 = x1047*x107;
        const double x1049 = x1047*x109;
        const double x1050 = r_23*x111;
        const double x1051 = x1050*x343;
        const double x1052 = x342*x855;
        const double x1053 = x1050*x852;
        const double x1054 = x345*x909;
        const double x1055 = x569*x783;
        const double x1056 = x460*x794;
        const double x1057 = x692*x821;
        const double x1058 = x25*x342*x35;
        const double x1059 = x21*x342*x39;
        const double x1060 = x460*x796;
        const double x1061 = x331*x795;
        const double x1062 = x827*x856;
        const double x1063 = -x1019 + x1020 + x1021 + x1022 - x1024 - x1025 - x1026 - x1027 - x1028 - x1029 - x1030 + x1031 + x1032 + x1033 + x1034 - x1036 - x1037 - x1038 - x1039 - x1040 - x1041 + x1042 + x1044 + x1045 + x1046 + x1048 + x1049 + x1051 + x1052 + x1053 + x1054 - x1055 - x1056 - x1057 + x1058 + x1059 + x1060 + x1061 + x1062;
        const double x1064 = r_11*x172*x692;
        const double x1065 = -Py*r_13*x465 - d_4*x830 - x466*x784 + x466*x785 + x466*x786 + x641*x749 + x646*x778 - x754*x787;
        const double x1066 = -4*d_3*d_5*r_13 + x1000 + x1001 + x1002 + x994 + x995 + x996 + x997 + x998 + x999;
        const double x1067 = x749*x983;
        const double x1068 = x21*x835;
        const double x1069 = R_l_inv_43*r_13*x744;
        const double x1070 = r_13*x172*x646;
        const double x1071 = -x692*x987 + x976 - x977 + x980;
        const double x1072 = 16*d_5;
        const double x1073 = 16*x25;
        const double x1074 = 16*x21;
        const double x1075 = 16*x71;
        const double x1076 = Px*x1075;
        const double x1077 = 16*x23;
        const double x1078 = Px*x1077;
        const double x1079 = 16*x884;
        const double x1080 = x1063 + x1066;
        const double x1081 = -d_1 + x27;
        const double x1082 = x1081 + x7;
        const double x1083 = -x30;
        const double x1084 = x1081 + x6;
        const double x1085 = -x38;
        const double x1086 = x1085 + x151;
        const double x1087 = a_2 + x142 + x143 + x144;
        const double x1088 = x146 + x147 + x82 + x85 + x86;
        const double x1089 = -2*R_l_inv_50*a_0*x40 + x152 + x154 + x155 + x156;
        const double x1090 = x1087 + x1088;
        const double x1091 = x199 + x205 + x206 + x207 + x209 + x211 + x215 + x216 + x217 + x218 + x219 + x223 + x224 + x225 + x226 + x227 + x228 + x230 + x231 + x232 + x233 + x235 + x236 + x237 + x238 + x239 + x243 + x246 + x250 + x253 + x255 + x257 + x259 + x260 + x261 + x262 + x263;
        const double x1092 = x1091 + x266 + x268;
        const double x1093 = x203 + x379 + x380;
        const double x1094 = x297 + x300 + x302 + x304 + x306 + x308 + x310 + x312 + x314 + x316 + x318 + x320 + x322 + x324 + x326 + x328 + x330 + x332 + x333 + x334 + x335 + x337 + x338 + x339 + x340 + x341 + x344 + x347 + x350 + x351 + x352 + x353 + x354 + x355 + x357 + x358 + x359;
        const double x1095 = x198 + x201 + x204;
        const double x1096 = x1091 + x275 + x276 + x375 + x376 + x377;
        const double x1097 = x388 - x389 + x390 + x391 - x392 - x393;
        const double x1098 = x303 + x309 + x321 + x323 + x325 + x327 + x503 + x504 + x505 + x508 + x511 + x512 + x513 + x514 + x515 + x519 + x521 + x522 + x523 + x524 + x525 + x526 + x527 + x528 + x529 + x531 + x532 + x534 + x536 + x538 + x540 + x541 + x542 + x543 + x544 + x545;
        const double x1099 = x1098 + x559 + x617;
        const double x1100 = -4*d_4*x36*x494 + x572;
        const double x1101 = x1098 + x548 + x618;
        const double x1102 = x631 + x758 + x761 + x766;
        const double x1103 = x712 + x713 + x714 + x716 - x717 - x718 - x719 - x720;
        const double x1104 = -x732 - x733;
        const double x1105 = x652 + x654 + x655 + x723 + x724;

        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x0;
        A(0, 2) = x0;
        A(0, 3) = x2;
        A(0, 4) = -x3;
        A(0, 5) = x1;
        A(0, 6) = r_23;
        A(0, 8) = r_23;
        A(1, 0) = x29 + x5;
        A(1, 1) = x30;
        A(1, 2) = x29 + x4;
        A(1, 3) = x32;
        A(1, 4) = -x33;
        A(1, 5) = x31;
        A(1, 6) = x34 + x5;
        A(1, 7) = x30;
        A(1, 8) = x34 + x4;
        A(2, 0) = -x122 - x59 - x80 - x88;
        A(2, 1) = 4*R_l_inv_55*a_0*d_4*x36 - x124 - x133 - x141;
        A(2, 2) = x145 + x150 + x151 + x80;
        A(2, 3) = x157 + x162 + x169;
        A(2, 4) = -x171 - x173 + x179;
        A(2, 5) = x157 + x180 + x181;
        A(2, 6) = x145 + x184 + x186;
        A(2, 7) = x126 - x128 - x130 + x131 + x189;
        A(2, 8) = -x150 - x182 - x183 - x190 - x38 - x52;
        A(3, 0) = x264 + x273 + x291;
        A(3, 1) = x360 + x364 + x374;
        A(3, 2) = -x290 - x378 - x381 - x382;
        A(3, 3) = -x387 - x394 - x459;
        A(3, 4) = -x462 - x463 + x467;
        A(3, 5) = x394 + x468 + x469;
        A(3, 6) = -x381 - x471 - x473;
        A(3, 7) = -x360 - x474 - x475;
        A(3, 8) = x264 + x378 + x470 + x472;
        A(4, 0) = -x502 - x546 - x548 - x559;
        A(4, 1) = x571 + x574 + x585;
        A(4, 2) = x590 + x593 + x595;
        A(4, 3) = -x596 - x597 + x602 + x610;
        A(4, 4) = x611 - x612 - x613;
        A(4, 5) = x598 + x599 - x601 + x615;
        A(4, 6) = x301 - x303 + x305 - x309 + x313 + x315 + x317 + x319 - x321 - x323 - x325 - x327 + x502 - x503 + x507 - x511 + x518 + x520 - x522 - x523 - x524 - x525 - x526 - x527 - x528 - x529 - x531 - x532 - x534 - x536 + x537 + x539 - x541 - x542 - x543 - x544 - x545 + x616 + x618;
        A(4, 7) = x574 + x619 + x620;
        A(4, 8) = x595 + x622 + x623 + x624;
        A(5, 0) = x650 + x660 + x700;
        A(5, 1) = x705 + x711 + x721;
        A(5, 2) = -x650 - x725 - x727;
        A(5, 3) = x731 + x734 + x742;
        A(5, 4) = R_l_inv_42*x72*x744 + 16*x638*x658*x72 + x755;
        A(5, 5) = -x734 - x756 - x757;
        A(5, 6) = -x759 - x760 - x762;
        A(5, 7) = -x721 - x763 - x764;
        A(5, 8) = -x760 - x766 - x767 - x768;

        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x769;
        B(0, 2) = x769;
        B(0, 3) = -x342;
        B(0, 4) = -x749;
        B(0, 5) = x342;
        B(0, 6) = x244;
        B(0, 8) = x244;
        B(1, 0) = x771 + x775;
        B(1, 1) = x776;
        B(1, 2) = x770 + x775;
        B(1, 3) = -x777;
        B(1, 4) = -x778;
        B(1, 5) = x777;
        B(1, 6) = x771 + x779;
        B(1, 7) = x776;
        B(1, 8) = x770 + x779;
        B(2, 0) = -x800 - x807;
        B(2, 1) = 4*R_l_inv_56*a_0*d_3*r_11 - x808 - x809 - x813;
        B(2, 2) = x800 + x814;
        B(2, 3) = -x815 - x816 - x817 + x818 + x819 + x820 + x829;
        B(2, 4) = x69*x835 + x830 - x831 + x832 - x833 + x834;
        B(2, 5) = x815 + x816 + x817 - x818 - x819 - x820 + x829;
        B(2, 6) = x807 + x836;
        B(2, 7) = x69*x777 - x808 - x809 + x813;
        B(2, 8) = -x814 - x836;
        B(3, 0) = x860 + x871;
        B(3, 1) = x214*x777 + x25*x778 + x749*x868 - x874 - x875 - x876 - x877 - x878 + x891;
        B(3, 2) = -x871 - x892;
        B(3, 3) = x893 - x894 + x895 + x896 - x899 - x900 - x911;
        B(3, 4) = x466*(-r_11*x103 - r_11*x384 + r_11*x96 + r_13*x98 + x21*x244 + x864);
        B(3, 5) = -x893 + x894 - x895 - x896 + x899 + x900 - x911;
        B(3, 6) = -x860 - x912;
        B(3, 7) = 8*Py*d_3*r_12*r_23 + 8*Pz*d_3*r_12*r_33 + 4*d_3*r_11*x214 - x874 - x875 - x876 - x877 - x878 - x891;
        B(3, 8) = x892 + x912;
        B(4, 0) = -x954 - x964 - x966;
        B(4, 1) = x967 - x968 - x969 - x971 - x972 - x973 + x974 + x975 + x981;
        B(4, 2) = x964 + x985 + x986;
        B(4, 3) = x989 + x990 + x991;
        B(4, 4) = x835*(2*R_l_inv_26*a_0*d_1 + 2*R_l_inv_66*a_2*d_1 - x482 - x483 - x56);
        B(4, 5) = x988 - x990 + x991;
        B(4, 6) = -x966 - x986 - x992;
        B(4, 7) = -x967 + x968 + x969 + x971 + x972 + x973 - x974 - x975 + x981;
        B(4, 8) = x954 + x985 + x992;
        B(5, 0) = d_5*x828 - x1000 - x1001 - x1002 + x1018 + x1063 - x994 - x995 - x996 - x997 - x998 - x999;
        B(5, 1) = R_l_inv_44*r_11*x744 + R_l_inv_47*x744*x797 - x1064 + x1065 + x989;
        B(5, 2) = -x1018 - x1019 + x1020 + x1021 + x1022 - x1024 - x1025 - x1026 - x1027 - x1028 - x1029 - x1030 + x1031 + x1032 + x1033 + x1034 - x1036 - x1037 - x1038 - x1039 - x1040 - x1041 + x1042 + x1044 + x1045 + x1046 + x1048 + x1049 + x1051 + x1052 + x1053 + x1054 - x1055 - x1056 - x1057 + x1058 + x1059 + x1060 + x1061 + x1062 - x1066;
        B(5, 3) = -x1067 - x1068 - x1069 - x1070 + x1071 + x466*x823 + x754*x826 + x898*x908;
        B(5, 4) = 16*r_11*x853 + 16*r_12*x97 + 8*x104*x879 + x106*x749 + x107*x1076 + x107*x1078 - x1072*x793 - x1072*x794 + x1073*x792 + x1073*x793 - x1073*x795 + x1074*x822 - x1074*x824 + x1075*x846 + x1075*x950 + x1076*x109 + x1077*x846 + x1078*x109 + x1079*x750 + x1079*x886 + x110*x749 - x112*x749 + x113*x749 - x114*x749 - x117*x749 + x119*x749 - x120*x749 + 16*x248*x887 + 16*x35*x823 + 16*x39*x794 - 16*x692*x984 + x749*x90 - x749*x91 - x749*x92;
        B(5, 5) = 8*Py*d_4*r_11*r_22 + 8*Pz*d_4*r_11*r_32 + 8*R_l_inv_46*a_0*d_1*x826 - x1067 - x1068 - x1069 - x1070 - x1071;
        B(5, 6) = x1003 - x1004 - x1006 - x1008 - x1009 - x1011 - x1012 - x1013 - x1014 + x1015 + x1016 - x1017 - x1080;
        B(5, 7) = 8*R_l_inv_44*a_0*d_1*r_11 + 8*R_l_inv_47*a_0*d_1*x797 - x1064 - x1065 - x988;
        B(5, 8) = -x1003 + x1004 - x1005 - x1007 - x1010 + x1011 + x1012 + x1013 + x1014 - x1015 - x1016 + x1017 - x1080;

        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = r_23;
        C(0, 2) = r_23;
        C(0, 3) = x1;
        C(0, 4) = x3;
        C(0, 5) = x2;
        C(0, 6) = x0;
        C(0, 8) = x0;
        C(1, 0) = -x1082 - x5;
        C(1, 1) = x1083;
        C(1, 2) = -x1082 - x4;
        C(1, 3) = x31;
        C(1, 4) = x33;
        C(1, 5) = x32;
        C(1, 6) = -x1084 - x5;
        C(1, 7) = x1083;
        C(1, 8) = -x1084 - x4;
        C(2, 0) = x1086 + x1087 + x148 + x149 + x186;
        C(2, 1) = -x126 + x128 + x130 + x132 + x189;
        C(2, 2) = -x1086 - x1088 - x190 - x42 - x58;
        C(2, 3) = -x1089 - x169 - x180;
        C(2, 4) = -x171 - x173 - x179;
        C(2, 5) = -x1089 - x162 - x181;
        C(2, 6) = x1090 + x185 + x59 - x60 - x62 - x70 - x76;
        C(2, 7) = x133 - x134 + x136 - x138 - x140 + x188;
        C(2, 8) = x1085 + x1090 + x184 + x80;
        C(3, 0) = -x1092 - x1093 - x291 - x470;
        C(3, 1) = -x1094 - x374 - x474;
        C(3, 2) = x1095 + x1096 + x290 + x471;
        C(3, 3) = -x1097 - x387 - x469;
        C(3, 4) = -x462 - x463 - x467;
        C(3, 5) = x1097 + x459 + x468;
        C(3, 6) = x1092 + x1095 + x382 + x473;
        C(3, 7) = x1094 + x364 + x475;
        C(3, 8) = -x1093 - x1096 - x273 - x472;
        C(4, 0) = x1099 + x593 + x621 + x623;
        C(4, 1) = -x1100 - x585 - x619;
        C(4, 2) = -x1101 - x477 - x485 - x488 - x489 - x492 - x497 - x499 - x500 - x501 - x589 - x622;
        C(4, 3) = x385*x478 - x600 - x601 - x615;
        C(4, 4) = -x611 + x612 - x613;
        C(4, 5) = x602 - x603 - x604 - x605 + x606 + x607 - x608 - x609 + x614;
        C(4, 6) = x1099 + x481 + x495 + x590 + x624;
        C(4, 7) = -x1100 - x571 - x620;
        C(4, 8) = -x1101 - x479 - x490 - x493 - x591 - x592 - x624;
        C(5, 0) = -x1102 - x700 - x725;
        C(5, 1) = x1103 + x705 + x764;
        C(5, 2) = x1102 + x660 + x727;
        C(5, 3) = -x1104 - x742 - x756;
        C(5, 4) = 8*R_l_inv_42*a_0*d_1*x72 + 16*R_l_inv_45*a_0*d_1*d_3*x72 - x755;
        C(5, 5) = x1104 + x731 + x757;
        C(5, 6) = x1105 + x762 + x765 + x767;
        C(5, 7) = -x1103 - x711 - x763;
        C(5, 8) = x1105 + x625 + x635 + x644 + x645 + x759 + x768;

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

    // Code for non-branch dispatcher node 2
    // Actually, there is no code

    // Code for explicit solution node 3, solved variable is th_3
    auto ExplicitSolutionNode_node_3_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(2);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(2);
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
            const bool condition_0 = ((d_4 != 0) && (std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0) + d_1 - d_5*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_4) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = safe_acos((Px*x0 - Py*x1 + d_1 + d_5*(-r_13*x0 + r_23*x1))/d_4);
                RawSolution solution_0(this_solution);
                solution_0[5] = x2;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = -x2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
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
    // Finish code for explicit solution node 2

    // Code for non-branch dispatcher node 4
    // Actually, there is no code

    // Code for explicit solution node 5, solved variable is th_2
    auto ExplicitSolutionNode_node_5_solve_th_2_processor = [&]() -> void
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
            const double th_3 = this_solution[5];
            const bool condition_0 = ((2*std::fabs(a_0*d_3) >= zero_tolerance) || (std::fabs(2*a_0*a_2 - 2*a_0*d_4*std::sin(th_3)) >= zero_tolerance) || (std::fabs(std::pow(Px, 2) - 2*Px*d_5*r_13 + std::pow(Py, 2) - 2*Py*d_5*r_23 + std::pow(Pz, 2) - 2*Pz*d_5*r_33 - std::pow(a_0, 2) - std::pow(a_2, 2) + 2*a_2*d_4*std::sin(th_3) - std::pow(d_1, 2) + 2*d_1*d_4*std::cos(th_3) - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(d_5, 2)*std::pow(r_13, 2) + std::pow(d_5, 2)*std::pow(r_23, 2) + std::pow(d_5, 2)*std::pow(r_33, 2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 2*a_0;
                const double x1 = d_4*std::sin(th_3);
                const double x2 = a_2*x0 - x0*x1;
                const double x3 = std::atan2(-d_3*x0, x2);
                const double x4 = std::pow(a_0, 2);
                const double x5 = std::pow(d_3, 2);
                const double x6 = 2*d_5;
                const double x7 = std::pow(d_5, 2);
                const double x8 = std::pow(Px, 2) - Px*r_13*x6 + std::pow(Py, 2) - Py*r_23*x6 + std::pow(Pz, 2) - Pz*r_33*x6 - std::pow(a_2, 2) + 2*a_2*x1 - std::pow(d_1, 2) + 2*d_1*d_4*std::cos(th_3) - std::pow(d_4, 2) + std::pow(r_13, 2)*x7 + std::pow(r_23, 2)*x7 + std::pow(r_33, 2)*x7 - x4 - x5;
                const double x9 = safe_sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x8, 2));
                RawSolution solution_0(this_solution);
                solution_0[4] = x3 + std::atan2(x9, x8);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx_0);
                const double tmp_sol_value = x3 + std::atan2(-x9, x8);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_2_processor();
    // Finish code for explicit solution node 4

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
                add_input_index_to(19, branch_input_index);
            }

            if (!taken_by_degenerate)
                add_input_index_to(7, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor();
    // Finish code for solved_variable dispatcher node 6

    // Code for explicit solution node 19, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor = [&]() -> void
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
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_33) >= zero_tolerance) || (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double tmp_sol_value = std::atan2(r_13*std::cos(th_0) + r_23*std::sin(th_0), r_33);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor();
    // Finish code for explicit solution node 19

    // Code for non-branch dispatcher node 20
    // Actually, there is no code

    // Code for explicit solution node 21, solved variable is th_5
    auto ExplicitSolutionNode_node_21_solve_th_5_processor = [&]() -> void
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
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_5_processor();
    // Finish code for explicit solution node 20

    // Code for solved_variable dispatcher node 22
    auto SolvedVariableDispatcherNode_node_22_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(22);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(22);
        if (!this_input_valid)
            return;

        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[4];

            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }

            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }

            if (!taken_by_degenerate)
                add_input_index_to(23, node_input_i_idx_in_queue);
            else if (!(degenerate_valid_0))
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_22_processor();
    // Finish code for solved_variable dispatcher node 22

    // Code for explicit solution node 32, solved variable is th_1
    auto ExplicitSolutionNode_node_32_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(32);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(32);
        if (!this_input_valid)
            return;

        // The solution of non-root node 32
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance) || (std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = safe_sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(-a_2 + 0.067605633802817006*d_3, 2));
                const double x8 = -0.99772254304723296*a_2 + 0.067451664882066498*d_3;
                RawSolution solution_0(this_solution);
                solution_0[1] = x6 + std::atan2(x7, x8);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(33, appended_idx_0);
                const double tmp_sol_value = x6 + std::atan2(-x7, x8);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(33, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_1_processor();
    // Finish code for explicit solution node 32

    // Code for non-branch dispatcher node 33
    // Actually, there is no code

    // Code for explicit solution node 34, solved variable is th_4
    auto ExplicitSolutionNode_node_34_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(33);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(33);
        if (!this_input_valid)
            return;

        // The solution of non-root node 34
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(-r_13*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_0) - r_23*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::sin(th_0) + r_33*(0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))) >= zero_tolerance) || (std::fabs(-r_13*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.067451664882066498*std::sin(th_1) - 0.99772254304723296*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = 0.067451664882066498*x1 - 0.99772254304723296*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.99772254304723296*x1 + 0.067451664882066498*x2;
                const double x8 = -x0*x7;
                const double x9 = -x3;
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_34_solve_th_4_processor();
    // Finish code for explicit solution node 33

    // Code for explicit solution node 23, solved variable is th_1
    auto ExplicitSolutionNode_node_23_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(23);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(23);
        if (!this_input_valid)
            return;

        // The solution of non-root node 23
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1th_2th_4_soa = this_solution[3];
            const double th_2 = this_solution[4];
            const bool condition_0 = ((std::fabs(a_2*std::sin(th_2) + d_3*std::cos(th_2)) >= 9.9999999999999995e-7) || (std::fabs(a_0 + a_2*std::cos(th_2) - d_3*std::sin(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_0 + a_2*x1 - d_3*x2;
                const double x4 = -a_2*x2 - d_3*x1;
                const double x5 = Px*std::cos(th_0) + Py*std::sin(th_0) - d_5*std::sin(th_1th_2th_4_soa);
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x5, x0*x4 + x3*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(24, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor();
    // Finish code for explicit solution node 23

    // Code for non-branch dispatcher node 24
    // Actually, there is no code

    // Code for explicit solution node 25, solved variable is th_4
    auto ExplicitSolutionNode_node_25_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(24);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(24);
        if (!this_input_valid)
            return;

        // The solution of non-root node 25
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
    ExplicitSolutionNode_node_25_solve_th_4_processor();
    // Finish code for explicit solution node 24

    // Code for explicit solution node 14, solved variable is th_5
    auto ExplicitSolutionNode_node_14_solve_th_5_processor = [&]() -> void
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
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor();
    // Finish code for explicit solution node 14

    // Code for solved_variable dispatcher node 15
    auto SolvedVariableDispatcherNode_node_15_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(15);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(15);
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

            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(26, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(29, branch_input_index);
            }

            if (!taken_by_degenerate)
                add_input_index_to(16, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor();
    // Finish code for solved_variable dispatcher node 15

    // Code for explicit solution node 29, solved variable is th_1
    auto ExplicitSolutionNode_node_29_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(29);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(29);
        if (!this_input_valid)
            return;

        // The solution of non-root node 29
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance) || (std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = safe_sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(a_2 - 0.067605633802817006*d_3, 2));
                const double x8 = 0.99772254304723296*a_2 - 0.067451664882066498*d_3;
                RawSolution solution_0(this_solution);
                solution_0[1] = x6 + std::atan2(x7, x8);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(30, appended_idx_0);
                const double tmp_sol_value = x6 + std::atan2(-x7, x8);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_1_processor();
    // Finish code for explicit solution node 29

    // Code for non-branch dispatcher node 30
    // Actually, there is no code

    // Code for explicit solution node 31, solved variable is th_4
    auto ExplicitSolutionNode_node_31_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(30);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(30);
        if (!this_input_valid)
            return;

        // The solution of non-root node 31
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(-r_13*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_0) - r_23*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::sin(th_0) + r_33*(0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))) >= zero_tolerance) || (std::fabs(r_13*((0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) + r_23*((0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.067451664882066498*std::sin(th_1) - 0.99772254304723296*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.067451664882066498*x1 + 0.99772254304723296*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.99772254304723296*x1 + 0.067451664882066498*x2;
                const double x8 = x0*x7;
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, -r_13*x3*x6 - r_23*x3*x4 + r_33*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_4_processor();
    // Finish code for explicit solution node 30

    // Code for explicit solution node 26, solved variable is th_1
    auto ExplicitSolutionNode_node_26_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(26);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(26);
        if (!this_input_valid)
            return;

        // The solution of non-root node 26
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            const bool condition_0 = ((std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance) || (std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = safe_sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(-a_2 + 0.067605633802817006*d_3, 2));
                const double x8 = -0.99772254304723296*a_2 + 0.067451664882066498*d_3;
                RawSolution solution_0(this_solution);
                solution_0[1] = x6 + std::atan2(x7, x8);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(27, appended_idx_0);
                const double tmp_sol_value = x6 + std::atan2(-x7, x8);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(27, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_1_processor();
    // Finish code for explicit solution node 26

    // Code for non-branch dispatcher node 27
    // Actually, there is no code

    // Code for explicit solution node 28, solved variable is th_4
    auto ExplicitSolutionNode_node_28_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(27);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(27);
        if (!this_input_valid)
            return;

        // The solution of non-root node 28
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(-r_13*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_0) - r_23*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::sin(th_0) + r_33*(0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))) >= zero_tolerance) || (std::fabs(-r_13*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.067451664882066498*std::sin(th_1) - 0.99772254304723296*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = 0.067451664882066498*x1 - 0.99772254304723296*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.99772254304723296*x1 + 0.067451664882066498*x2;
                const double x8 = -x0*x7;
                const double x9 = -x3;
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_4_processor();
    // Finish code for explicit solution node 27

    // Code for explicit solution node 16, solved variable is th_1
    auto ExplicitSolutionNode_node_16_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(16);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(16);
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
            const double th_2 = this_solution[4];
            const bool condition_0 = ((std::fabs(a_2*std::sin(th_2) + d_3*std::cos(th_2)) >= 9.9999999999999995e-7) || (std::fabs(a_0 + a_2*std::cos(th_2) - d_3*std::sin(th_2)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_0 + a_2*x1 - d_3*x2;
                const double x4 = -a_2*x2 - d_3*x1;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6;
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x7, x0*x4 + x3*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1_processor();
    // Finish code for explicit solution node 16

    // Code for non-branch dispatcher node 17
    // Actually, there is no code

    // Code for explicit solution node 18, solved variable is th_4
    auto ExplicitSolutionNode_node_18_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(17);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(17);
        if (!this_input_valid)
            return;

        // The solution of non-root node 18
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
    ExplicitSolutionNode_node_18_solve_th_4_processor();
    // Finish code for explicit solution node 17

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
            const double th_3 = this_solution[5];
            const bool condition_0 = ((std::sin(th_3) != 0) && (std::fabs((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3)) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = safe_asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
                RawSolution solution_0(this_solution);
                solution_0[6] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx_0);
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
            const double th_0 = this_solution[0];
            const bool checked_result = ((std::fabs(Pz) <= 9.9999999999999995e-7) && (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8

    // Code for explicit solution node 9, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            const double th_3 = this_solution[5];
            const double th_4 = this_solution[6];
            const bool condition_0 = ((std::fabs(Pz) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = -a_0*std::cos(th_2) - a_2 + d_4*std::sin(th_3) + d_5*std::sin(th_4)*std::cos(th_3);
                const double x1 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x2 = a_0*std::sin(th_2) - d_3 + d_5*std::cos(th_4);
                const double tmp_sol_value = std::atan2(Pz*x0 - x1*x2, Pz*x2 + x0*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 9

    // Code for non-branch dispatcher node 10
    // Actually, there is no code

    // Code for explicit solution node 11, solved variable is th_1
    auto ExplicitSolutionNode_node_11_solve_th_1_processor = [&]() -> void
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
            const double th_1th_2_soa = this_solution[2];
            const double th_2 = this_solution[4];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = th_1th_2_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor();
    // Finish code for explicit solution node 10

    // Code for non-branch dispatcher node 12
    // Actually, there is no code

    // Code for explicit solution node 13, solved variable is th_5
    auto ExplicitSolutionNode_node_13_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_1th_2_soa = this_solution[2];
            const double th_3 = this_solution[5];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*(std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)*std::cos(th_1th_2_soa)) - r_21*(std::sin(th_0)*std::sin(th_3)*std::cos(th_1th_2_soa) + std::cos(th_0)*std::cos(th_3)) + r_31*std::sin(th_1th_2_soa)*std::sin(th_3)) >= zero_tolerance) || (std::fabs(r_12*(std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)*std::cos(th_1th_2_soa)) - r_22*(std::sin(th_0)*std::sin(th_3)*std::cos(th_1th_2_soa) + std::cos(th_0)*std::cos(th_3)) + r_32*std::sin(th_1th_2_soa)*std::sin(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_3);
                const double x1 = x0*std::sin(th_1th_2_soa);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1th_2_soa);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                const double tmp_sol_value = std::atan2(-r_11*x6 + r_21*x7 - r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_5_processor();
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

static void computeIKForRawTarget(const Eigen::Matrix4d& T_ee, const Eigen::Matrix4d& T_ee_raw, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output, Eigen::Matrix4d& frame_disturbance, bool& frame_disturbance_ready, bool allow_disturbance, std::vector<std::pair<double,std::array<double,robot_nq>>>* refinement_seeds = nullptr)
{
    if (!T_ee.allFinite()) { ik_output.clear(); return; }
    if (!T_ee_raw.allFinite()) { ik_output.clear(); return; }
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    bool strict_raw_witness = false;
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
            if (refinement_error < 1e-7) { strict_raw_witness=true;refinement_seeds->clear(); }
            else if (!strict_raw_witness) retainRefinementSeed(*refinement_seeds, ik_out_i, refinement_error);
        }
        if (ee_pose_diff < pose_tolerance)
            ik_output.push_back(ik_out_i);
        std::array<double, robot_nq> snapped;
        if (snapNearJointLimits<robot_nq>(ik_out_i, snapped, jointLower(), jointUpper(), periodicLimitJoints(), fixedJoints()))
        {
            const Eigen::Matrix4d snapped_pose = computeFK(snapped);
            if ((snapped_pose.block<3, 1>(0, 3)-T_ee.block<3, 1>(0, 3)).norm() <= 1e-7 && (snapped_pose.block<3, 3>(0, 0)-T_ee.block<3, 3>(0, 0)).cwiseAbs().maxCoeff() <= 1e-7)
                ik_output.push_back(snapped);
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
                ik_output.push_back(ik_out_i);
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
    std::vector<std::pair<double,std::array<double,robot_nq>>> refinement_seeds;
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
    if (!refinement_seeds.empty())
    {
        auto refinement_fk = [&](const std::array<double,robot_nq>& q) { return computeFK(q); };
        bool strict_witness = false;
        for (const auto& candidate : ik_output)
            if ((refinement_fk(candidate)-T_ee).cwiseAbs().maxCoeff()<1e-7) { strict_witness=true;break; }
        if (!strict_witness)
        {
            for (const auto& entry : refinement_seeds)
            {
                auto candidate=entry.second;
                if (refineIkSeed<robot_nq>(refinement_fk,T_ee,candidate,fixedJoints()))
                {
                    ik_output.push_back(candidate);
                    break;
                }
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

}; // struct dense_cvr_038_ik

// Code below for debug
void test_ik_solve_dense_cvr_038()
{
    std::array<double, dense_cvr_038_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = dense_cvr_038_ik::computeFK(theta);
    auto ik_output = dense_cvr_038_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = dense_cvr_038_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_dense_cvr_038();
    return 0;
}
