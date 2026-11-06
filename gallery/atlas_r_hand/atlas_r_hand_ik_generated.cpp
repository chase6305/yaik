#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct atlas_r_hand_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 384;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -4.71238898038469, -6.283185307179586, -6.283185307179586, -6.283185307179586, -6.283185307179586, -4.71238898038469}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{false, true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{true, false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 22;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_1 = 0.11;
static constexpr double a_2 = 0.016;
static constexpr double a_3 = 0.0092;
static constexpr double a_4 = 0.00921;
static constexpr double d_2 = -0.306;
static constexpr double d_4 = -0.29955;
static constexpr double pre_transform_s0 = 0.1406;
static constexpr double pre_transform_s1 = -0.2256;
static constexpr double pre_transform_s2 = 0.2326;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = 1.5707963267948966;
static constexpr double th_2_offset_original2raw = 3.141592653589793;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 3.141592653589793;
static constexpr double th_6_offset_original2raw = 1.5707963267948966;

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
    ee_transformed(0, 0) = r_21;
    ee_transformed(0, 1) = -r_23;
    ee_transformed(0, 2) = -r_22;
    ee_transformed(0, 3) = -Py + pre_transform_s1;
    ee_transformed(1, 0) = -r_11;
    ee_transformed(1, 1) = r_13;
    ee_transformed(1, 2) = r_12;
    ee_transformed(1, 3) = Px - pre_transform_s0;
    ee_transformed(2, 0) = -r_31;
    ee_transformed(2, 1) = r_33;
    ee_transformed(2, 2) = r_32;
    ee_transformed(2, 3) = Pz - pre_transform_s2;
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
    ee_transformed(0, 0) = -r_21;
    ee_transformed(0, 1) = r_23;
    ee_transformed(0, 2) = r_22;
    ee_transformed(0, 3) = Py + pre_transform_s0;
    ee_transformed(1, 0) = r_11;
    ee_transformed(1, 1) = -r_13;
    ee_transformed(1, 2) = -r_12;
    ee_transformed(1, 3) = -Px + pre_transform_s1;
    ee_transformed(2, 0) = -r_31;
    ee_transformed(2, 1) = r_33;
    ee_transformed(2, 2) = r_32;
    ee_transformed(2, 3) = Pz + pre_transform_s2;
    return ee_transformed;
}

///************* Below are the actual FK and IK implementations *******************
static Eigen::Matrix4d computeFK(const std::array<double, robot_nq>& theta_input_original)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = addRevoluteOffset(theta_input_original[6], th_6_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = std::sin(th_6);
    const double x1 = std::cos(th_4);
    const double x2 = std::sin(th_0);
    const double x3 = std::cos(th_2);
    const double x4 = x2*x3;
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_0);
    const double x8 = x6*x7;
    const double x9 = x4 - x5*x8;
    const double x10 = std::sin(th_4);
    const double x11 = std::sin(th_3);
    const double x12 = std::sin(th_1);
    const double x13 = x12*x7;
    const double x14 = std::cos(th_3);
    const double x15 = x2*x6;
    const double x16 = x3*x7;
    const double x17 = x15 + x16*x5;
    const double x18 = x11*x13 + x14*x17;
    const double x19 = -x1*x9 - x10*x18;
    const double x20 = std::cos(th_6);
    const double x21 = std::sin(th_5);
    const double x22 = -x11*x17 + x12*x14*x7;
    const double x23 = std::cos(th_5);
    const double x24 = x1*x18 - x10*x9;
    const double x25 = -x21*x22 + x23*x24;
    const double x26 = a_2*x5;
    const double x27 = -x15*x5 - x16;
    const double x28 = x12*x2;
    const double x29 = x4*x5 - x8;
    const double x30 = x11*x28 + x14*x29;
    const double x31 = -x1*x27 - x10*x30;
    const double x32 = -x11*x29 + x12*x14*x2;
    const double x33 = x1*x30 - x10*x27;
    const double x34 = -x21*x32 + x23*x33;
    const double x35 = x12*x6;
    const double x36 = x12*x3;
    const double x37 = x11*x5 - x14*x36;
    const double x38 = -x1*x35 - x10*x37;
    const double x39 = x11*x36 + x14*x5;
    const double x40 = x1*x37 - x10*x35;
    const double x41 = -x21*x39 + x23*x40;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x19 + x20*x25;
    ee_pose_raw(0, 1) = -x0*x25 - x19*x20;
    ee_pose_raw(0, 2) = -x21*x24 - x22*x23;
    ee_pose_raw(0, 3) = a_1*x7 + a_3*x17 + a_4*x18 - d_2*x13 + d_4*x22 + x26*x7;
    ee_pose_raw(1, 0) = -x0*x31 + x20*x34;
    ee_pose_raw(1, 1) = -x0*x34 - x20*x31;
    ee_pose_raw(1, 2) = -x21*x33 - x23*x32;
    ee_pose_raw(1, 3) = a_1*x2 + a_3*x29 + a_4*x30 - d_2*x28 + d_4*x32 + x2*x26;
    ee_pose_raw(2, 0) = -x0*x38 + x20*x41;
    ee_pose_raw(2, 1) = -x0*x41 - x20*x38;
    ee_pose_raw(2, 2) = -x21*x40 - x23*x39;
    ee_pose_raw(2, 3) = -a_2*x12 - a_3*x36 + a_4*x37 - d_2*x5 + d_4*x39;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = addRevoluteOffset(theta_input_original[6], th_6_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x1*x7;
    const double x9 = -x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x1*x4;
    const double x13 = -x0*x7 + x12*x6;
    const double x14 = x1*x10*x2 - x11*x13;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = x10*x13 + x11*x3;
    const double x18 = -x15*x9 - x16*x17;
    const double x19 = std::cos(th_5);
    const double x20 = std::sin(th_5);
    const double x21 = -x14*x19 - x20*(x15*x17 - x16*x9);
    const double x22 = x0*x2;
    const double x23 = x0*x6*x7 - x12;
    const double x24 = -x5*x6 - x8;
    const double x25 = -x10*x22 - x11*x24;
    const double x26 = x10*x24 - x11*x22;
    const double x27 = -x15*x23 - x16*x26;
    const double x28 = -x19*x25 - x20*(x15*x26 - x16*x23);
    const double x29 = x2*x7;
    const double x30 = x2*x4;
    const double x31 = x10*x6 + x11*x30;
    const double x32 = -x10*x30 + x11*x6;
    const double x33 = -x15*x29 - x16*x32;
    const double x34 = -x19*x31 - x20*(x15*x32 - x16*x29);
    const double x35 = a_2*x2 + d_2*x6 - pre_transform_s2;
    const double x36 = -x35;
    const double x37 = a_2*x6;
    const double x38 = x0*x37;
    const double x39 = a_1*x0;
    const double x40 = -pre_transform_s1 + x39;
    const double x41 = d_2*x0*x2 - x38 - x40;
    const double x42 = a_3*x30 + x35;
    const double x43 = -x42;
    const double x44 = a_3*x24 + d_2*x22 + pre_transform_s1 - x38 - x39;
    const double x45 = a_4*x32 + d_4*x31 - x42;
    const double x46 = a_4*x26 + d_4*x25 + x44;
    const double x47 = a_1*x1 + pre_transform_s0;
    const double x48 = -d_2*x3 + x1*x37 + x47;
    const double x49 = a_3*x13 + x48;
    const double x50 = a_4*x17 + d_4*x14 + x49;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x18;
    jacobian(0, 6) = x21;
    jacobian(1, 1) = x1;
    jacobian(1, 2) = x22;
    jacobian(1, 3) = x23;
    jacobian(1, 4) = x25;
    jacobian(1, 5) = x27;
    jacobian(1, 6) = x28;
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x6;
    jacobian(2, 3) = x29;
    jacobian(2, 4) = x31;
    jacobian(2, 5) = x33;
    jacobian(2, 6) = x34;
    jacobian(3, 0) = pre_transform_s1;
    jacobian(3, 1) = -pre_transform_s2*x1;
    jacobian(3, 2) = -x22*x36 - x41*x6;
    jacobian(3, 3) = x2*x44*x7 - x23*x43;
    jacobian(3, 4) = -x25*x45 + x31*x46;
    jacobian(3, 5) = -x27*x45 + x33*x46;
    jacobian(3, 6) = -x28*x45 + x34*x46;
    jacobian(4, 0) = -pre_transform_s0;
    jacobian(4, 1) = pre_transform_s2*x0;
    jacobian(4, 2) = -x3*x36 + x48*x6;
    jacobian(4, 3) = -x29*x49 + x43*x9;
    jacobian(4, 4) = x14*x45 - x31*x50;
    jacobian(4, 5) = x18*x45 - x33*x50;
    jacobian(4, 6) = x21*x45 - x34*x50;
    jacobian(5, 1) = x0*x40 + x1*x47;
    jacobian(5, 2) = x22*x48 + x3*x41;
    jacobian(5, 3) = x23*x49 - x44*x9;
    jacobian(5, 4) = -x14*x46 + x25*x50;
    jacobian(5, 5) = -x18*x46 + x27*x50;
    jacobian(5, 6) = -x21*x46 + x28*x50;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = addRevoluteOffset(theta_input_original[6], th_6_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x1*x7;
    const double x9 = -x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x1*x4;
    const double x13 = -x0*x7 + x12*x6;
    const double x14 = x1*x10*x2 - x11*x13;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = x10*x13 + x11*x3;
    const double x18 = std::cos(th_5);
    const double x19 = std::sin(th_5);
    const double x20 = x0*x2;
    const double x21 = x0*x6*x7 - x12;
    const double x22 = -x5*x6 - x8;
    const double x23 = -x10*x20 - x11*x22;
    const double x24 = x10*x22 - x11*x20;
    const double x25 = x2*x7;
    const double x26 = x2*x4;
    const double x27 = x10*x6 + x11*x26;
    const double x28 = -x10*x26 + x11*x6;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = -x15*x9 - x16*x17;
    jacobian(0, 6) = -x14*x18 - x19*(x15*x17 - x16*x9);
    jacobian(1, 1) = x1;
    jacobian(1, 2) = x20;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x23;
    jacobian(1, 5) = -x15*x21 - x16*x24;
    jacobian(1, 6) = -x18*x23 - x19*(x15*x24 - x16*x21);
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x6;
    jacobian(2, 3) = x25;
    jacobian(2, 4) = x27;
    jacobian(2, 5) = -x15*x25 - x16*x28;
    jacobian(2, 6) = -x18*x27 - x19*(x15*x28 - x16*x25);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = addRevoluteOffset(theta_input_original[6], th_6_offset_original2raw);
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];

    // Temp variable for efficiency
    const double x0 = -pre_transform_s1;
    const double x1 = std::sin(th_0);
    const double x2 = std::cos(th_1);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_0);
    const double x5 = p_on_ee_z*x4;
    const double x6 = a_2*x3 + d_2*x2 - pre_transform_s2;
    const double x7 = -x6;
    const double x8 = x3*x4;
    const double x9 = a_2*x2;
    const double x10 = x4*x9;
    const double x11 = a_1*x4;
    const double x12 = x0 + x11;
    const double x13 = d_2*x3*x4 - x10 - x12;
    const double x14 = std::sin(th_2);
    const double x15 = x14*x3;
    const double x16 = std::cos(th_2);
    const double x17 = x1*x16;
    const double x18 = x14*x2*x4 - x17;
    const double x19 = x16*x3;
    const double x20 = a_3*x19 + x6;
    const double x21 = -x20;
    const double x22 = x1*x14;
    const double x23 = x16*x4;
    const double x24 = -x2*x23 - x22;
    const double x25 = a_3*x24 + d_2*x8 + pre_transform_s1 - x10 - x11;
    const double x26 = std::cos(th_3);
    const double x27 = std::sin(th_3);
    const double x28 = x19*x27 + x2*x26;
    const double x29 = -x24*x27 - x26*x8;
    const double x30 = -x19*x26 + x2*x27;
    const double x31 = a_4*x30 + d_4*x28 - x20;
    const double x32 = x24*x26 - x27*x8;
    const double x33 = a_4*x32 + d_4*x29 + x25;
    const double x34 = std::cos(th_4);
    const double x35 = std::sin(th_4);
    const double x36 = -x15*x34 - x30*x35;
    const double x37 = -x18*x34 - x32*x35;
    const double x38 = std::cos(th_5);
    const double x39 = std::sin(th_5);
    const double x40 = -x28*x38 - x39*(-x15*x35 + x30*x34);
    const double x41 = -x29*x38 - x39*(-x18*x35 + x32*x34);
    const double x42 = x1*x3;
    const double x43 = a_1*x1 + pre_transform_s0;
    const double x44 = -d_2*x42 + x1*x9 + x43;
    const double x45 = -x2*x22 - x23;
    const double x46 = -x14*x4 + x17*x2;
    const double x47 = a_3*x46 + x44;
    const double x48 = x1*x26*x3 - x27*x46;
    const double x49 = x26*x46 + x27*x42;
    const double x50 = a_4*x49 + d_4*x48 + x47;
    const double x51 = -x34*x45 - x35*x49;
    const double x52 = -x38*x48 - x39*(x34*x49 - x35*x45);
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y - x0;
    jacobian(0, 1) = p_on_ee_z*x1 - pre_transform_s2*x1;
    jacobian(0, 2) = p_on_ee_y*x2 - x13*x2 + x3*x5 - x7*x8;
    jacobian(0, 3) = -p_on_ee_y*x15 + p_on_ee_z*x18 + x14*x25*x3 - x18*x21;
    jacobian(0, 4) = -p_on_ee_y*x28 + p_on_ee_z*x29 + x28*x33 - x29*x31;
    jacobian(0, 5) = -p_on_ee_y*x36 + p_on_ee_z*x37 - x31*x37 + x33*x36;
    jacobian(0, 6) = -p_on_ee_y*x40 + p_on_ee_z*x41 - x31*x41 + x33*x40;
    jacobian(1, 0) = p_on_ee_x - pre_transform_s0;
    jacobian(1, 1) = pre_transform_s2*x4 - x5;
    jacobian(1, 2) = -p_on_ee_x*x2 + p_on_ee_z*x1*x3 + x2*x44 - x42*x7;
    jacobian(1, 3) = p_on_ee_x*x15 - p_on_ee_z*x45 - x15*x47 + x21*x45;
    jacobian(1, 4) = p_on_ee_x*x28 - p_on_ee_z*x48 - x28*x50 + x31*x48;
    jacobian(1, 5) = p_on_ee_x*x36 - p_on_ee_z*x51 + x31*x51 - x36*x50;
    jacobian(1, 6) = p_on_ee_x*x40 - p_on_ee_z*x52 + x31*x52 - x40*x50;
    jacobian(2, 1) = -p_on_ee_x*x1 + p_on_ee_y*x4 + x1*x43 + x12*x4;
    jacobian(2, 2) = -p_on_ee_x*x8 - p_on_ee_y*x42 + x1*x13*x3 + x3*x4*x44;
    jacobian(2, 3) = -p_on_ee_x*x18 + p_on_ee_y*x45 + x18*x47 - x25*x45;
    jacobian(2, 4) = -p_on_ee_x*x29 + p_on_ee_y*x48 + x29*x50 - x33*x48;
    jacobian(2, 5) = -p_on_ee_x*x37 + p_on_ee_y*x51 - x33*x51 + x37*x50;
    jacobian(2, 6) = -p_on_ee_x*x41 + p_on_ee_y*x52 - x33*x52 + x41*x50;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, double th_0, SolutionQueueType& solution_queue, NodeIndexWorkspaceType& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 48, 48, 96, 96, 96, 96, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192}};
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
        R_l(0, 4) = -Pz;
        R_l(0, 5) = Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2);
        R_l(1, 4) = Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2);
        R_l(1, 5) = Pz;
        R_l(2, 0) = r_31;
        R_l(2, 1) = r_32;
        R_l(2, 2) = -r_11*std::cos(th_0) - r_21*std::sin(th_0);
        R_l(2, 3) = -r_12*std::cos(th_0) - r_22*std::sin(th_0);
        R_l(3, 0) = -r_11*std::cos(th_0) - r_21*std::sin(th_0);
        R_l(3, 1) = -r_12*std::cos(th_0) - r_22*std::sin(th_0);
        R_l(3, 2) = -r_31;
        R_l(3, 3) = -r_32;
        R_l(4, 6) = r_11*std::sin(th_0) - r_21*std::cos(th_0);
        R_l(4, 7) = r_12*std::sin(th_0) - r_22*std::cos(th_0);
        R_l(5, 6) = -Pz*r_31 - (-Px*std::sin(th_0) + Py*std::cos(th_0))*(-r_11*std::sin(th_0) + r_21*std::cos(th_0)) - (r_11*std::cos(th_0) + r_21*std::sin(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(5, 7) = -Pz*r_32 - (-Px*std::sin(th_0) + Py*std::cos(th_0))*(-r_12*std::sin(th_0) + r_22*std::cos(th_0)) - (r_12*std::cos(th_0) + r_22*std::sin(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(6, 0) = (-Px*std::sin(th_0) + Py*std::cos(th_0))*(r_11*std::cos(th_0) + r_21*std::sin(th_0)) - (-r_11*std::sin(th_0) + r_21*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(6, 1) = (-Px*std::sin(th_0) + Py*std::cos(th_0))*(r_12*std::cos(th_0) + r_22*std::sin(th_0)) - (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(6, 2) = -Pz*(-r_11*std::sin(th_0) + r_21*std::cos(th_0)) + r_31*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(6, 3) = -Pz*(-r_12*std::sin(th_0) + r_22*std::cos(th_0)) + r_32*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(7, 0) = -Pz*(-r_11*std::sin(th_0) + r_21*std::cos(th_0)) + r_31*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(7, 1) = -Pz*(-r_12*std::sin(th_0) + r_22*std::cos(th_0)) + r_32*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(7, 2) = (Px*std::sin(th_0) - Py*std::cos(th_0))*(r_11*std::cos(th_0) + r_21*std::sin(th_0)) + (-r_11*std::sin(th_0) + r_21*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(7, 3) = (Px*std::sin(th_0) - Py*std::cos(th_0))*(r_12*std::cos(th_0) + r_22*std::sin(th_0)) + (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
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
        const double x0 = std::sin(th_0);
        const double x1 = Px*x0;
        const double x2 = std::cos(th_0);
        const double x3 = -Py*x2 + x1;
        const double x4 = std::pow(Pz, 2);
        const double x5 = -x3;
        const double x6 = std::pow(x5, 2);
        const double x7 = Px*x2 + Py*x0 - a_1*std::pow(x0, 2) - a_1*std::pow(x2, 2);
        const double x8 = std::pow(x7, 2);
        const double x9 = -x4 - x6 - x8;
        const double x10 = 2*a_3;
        const double x11 = a_4*x10;
        const double x12 = -x11;
        const double x13 = 2*a_4;
        const double x14 = a_2*x13;
        const double x15 = 2*d_2;
        const double x16 = d_4*x15;
        const double x17 = x12 + x14 + x16;
        const double x18 = std::pow(a_2, 2);
        const double x19 = std::pow(a_3, 2);
        const double x20 = std::pow(a_4, 2);
        const double x21 = std::pow(d_2, 2);
        const double x22 = std::pow(d_4, 2);
        const double x23 = a_2*x10;
        const double x24 = x18 + x19 + x20 + x21 + x22 - x23;
        const double x25 = x17 + x24;
        const double x26 = x25 + x9;
        const double x27 = 4*a_2;
        const double x28 = d_4*x27;
        const double x29 = -x28;
        const double x30 = 4*a_3;
        const double x31 = d_4*x30;
        const double x32 = 4*a_4;
        const double x33 = d_2*x32;
        const double x34 = x31 + x33;
        const double x35 = x29 + x34;
        const double x36 = -x35;
        const double x37 = x17 - x18 - x19 - x20 - x21 - x22 + x23;
        const double x38 = -x37 - x4 - x6 - x8;
        const double x39 = r_11*x2 + r_21*x0;
        const double x40 = Pz*x39;
        const double x41 = r_31*x7;
        const double x42 = x40 - x41;
        const double x43 = r_12*x2 + r_22*x0;
        const double x44 = Pz*x43;
        const double x45 = r_32*x7;
        const double x46 = x44 - x45;
        const double x47 = R_l_inv_67*x42 + R_l_inv_77*x46;
        const double x48 = a_4*x47;
        const double x49 = R_l_inv_61*x42 + R_l_inv_71*x46;
        const double x50 = d_4*x49;
        const double x51 = R_l_inv_66*x42 + R_l_inv_76*x46;
        const double x52 = d_4*x51;
        const double x53 = R_l_inv_60*x42 + R_l_inv_70*x46;
        const double x54 = a_4*x53;
        const double x55 = -x54;
        const double x56 = x48 + x50 + x52 + x55;
        const double x57 = a_3*x53;
        const double x58 = a_3*x47;
        const double x59 = -x58;
        const double x60 = x57 + x59;
        const double x61 = a_2*x53;
        const double x62 = d_2*x49 - x61;
        const double x63 = a_2*x47;
        const double x64 = d_2*x51;
        const double x65 = R_l_inv_64*x42;
        const double x66 = R_l_inv_74*x46;
        const double x67 = x63 + x64 - x65 - x66;
        const double x68 = x62 + x67;
        const double x69 = R_l_inv_65*x42 + R_l_inv_75*x46;
        const double x70 = x10*x69;
        const double x71 = -x70;
        const double x72 = x13*x69;
        const double x73 = 2*d_4;
        const double x74 = 2*R_l_inv_62;
        const double x75 = 2*R_l_inv_72;
        const double x76 = 2*a_2;
        const double x77 = -x15 + x42*x74 + x46*x75 + x69*x76;
        const double x78 = -x73 + x77;
        const double x79 = -x48;
        const double x80 = x50 + x58 + x79;
        const double x81 = -x52;
        const double x82 = x57 + x81;
        const double x83 = -x63 - x64 + x65 + x66;
        const double x84 = x62 + x83;
        const double x85 = x47*x73;
        const double x86 = x13*x51;
        const double x87 = x13*x49;
        const double x88 = x53*x73;
        const double x89 = x87 + x88;
        const double x90 = 4*R_l_inv_63;
        const double x91 = 4*R_l_inv_73;
        const double x92 = 4*d_2;
        const double x93 = x69*x92;
        const double x94 = -x57;
        const double x95 = -d_2*x49;
        const double x96 = x58 + x61 + x95;
        const double x97 = -2*a_4*x69;
        const double x98 = x73 + x77;
        const double x99 = x52 + x54;
        const double x100 = x48 + x84;
        const double x101 = -r_11*x0 + r_21*x2;
        const double x102 = 2*Pz;
        const double x103 = x102*x5;
        const double x104 = 2*x7;
        const double x105 = -r_31*x4 + r_31*x6 + r_31*x8 - x101*x103 - x104*x40;
        const double x106 = -r_12*x0 + r_22*x2;
        const double x107 = -r_32*x4 + r_32*x6 + r_32*x8 - x103*x106 - x104*x44;
        const double x108 = x7*(2*Py*x2 - 2*x1);
        const double x109 = x101*x108 + x102*x41 - x39*x4 - x39*x6 + x39*x8;
        const double x110 = x102*x45 + x106*x108 - x4*x43 - x43*x6 + x43*x8;
        const double x111 = R_l_inv_00*x105 + R_l_inv_10*x107 + R_l_inv_20*x109 + R_l_inv_30*x110;
        const double x112 = a_3*x111;
        const double x113 = -x112;
        const double x114 = R_l_inv_01*x105 + R_l_inv_11*x107 + R_l_inv_21*x109 + R_l_inv_31*x110;
        const double x115 = d_4*x114;
        const double x116 = -x115;
        const double x117 = a_2*x111;
        const double x118 = R_l_inv_07*x105 + R_l_inv_17*x107 + R_l_inv_27*x109 + R_l_inv_37*x110;
        const double x119 = a_3*x118;
        const double x120 = a_4*x111;
        const double x121 = -d_2*x114;
        const double x122 = R_l_inv_06*x105 + R_l_inv_16*x107 + R_l_inv_26*x109 + R_l_inv_36*x110;
        const double x123 = d_4*x122;
        const double x124 = -x123;
        const double x125 = x117 + x119 + x120 + x121 + x124;
        const double x126 = R_l_inv_04*x105;
        const double x127 = R_l_inv_14*x107;
        const double x128 = R_l_inv_24*x109;
        const double x129 = R_l_inv_34*x110;
        const double x130 = a_2*x118;
        const double x131 = -x130;
        const double x132 = a_4*x118;
        const double x133 = d_2*x122;
        const double x134 = -x133;
        const double x135 = x126 + x127 + x128 + x129 + x131 - x132 + x134;
        const double x136 = 2*x19;
        const double x137 = 2*x20;
        const double x138 = a_3*x27;
        const double x139 = R_l_inv_05*x105 + R_l_inv_15*x107 + R_l_inv_25*x109 + R_l_inv_35*x110;
        const double x140 = x10*x139;
        const double x141 = x136 + x137 - x138 - x140;
        const double x142 = a_4*x30;
        const double x143 = -x142;
        const double x144 = -2*x21;
        const double x145 = -2*x22;
        const double x146 = 2*x18;
        const double x147 = 4*d_4;
        const double x148 = d_2*x147;
        const double x149 = 2*x105;
        const double x150 = R_l_inv_02*x149;
        const double x151 = 2*x107;
        const double x152 = R_l_inv_12*x151;
        const double x153 = 2*x109;
        const double x154 = R_l_inv_22*x153;
        const double x155 = 2*x110;
        const double x156 = R_l_inv_32*x155;
        const double x157 = x139*x76;
        const double x158 = x144 + x145 + x146 - x148 + x150 + x152 + x154 + x156 + x157;
        const double x159 = x143 + x158;
        const double x160 = a_4*x27 + x13*x139;
        const double x161 = -x120;
        const double x162 = x119 + x161;
        const double x163 = x112 + x115;
        const double x164 = x135 + x163;
        const double x165 = d_2*x114 - x117;
        const double x166 = x124 + x165;
        const double x167 = x118*x73;
        const double x168 = x122*x13;
        const double x169 = x114*x13;
        const double x170 = x111*x73;
        const double x171 = x169 + x170;
        const double x172 = 8*d_2;
        const double x173 = a_3*x172;
        const double x174 = 8*d_4;
        const double x175 = a_4*x174;
        const double x176 = 4*x105;
        const double x177 = 4*x107;
        const double x178 = R_l_inv_03*x176 + R_l_inv_13*x177 + 4*R_l_inv_23*x109 + 4*R_l_inv_33*x110 - a_2*x172 - x139*x92;
        const double x179 = x117 + x121 + x123;
        const double x180 = x126 + x127 + x128 + x129 + x131 + x132 + x134;
        const double x181 = x113 + x115 + x180;
        const double x182 = -4*a_2*a_4 - 2*a_4*x139;
        const double x183 = x144 + x145 + x146 + x148 + x150 + x152 + x154 + x156 + x157;
        const double x184 = x142 + x183;
        const double x185 = x112 + x116 + x180;
        const double x186 = R_l_inv_07*x109 + R_l_inv_17*x110 - R_l_inv_27*x105 - R_l_inv_37*x107;
        const double x187 = a_4*x186;
        const double x188 = R_l_inv_01*x109 + R_l_inv_11*x110 - R_l_inv_21*x105 - R_l_inv_31*x107;
        const double x189 = d_4*x188;
        const double x190 = R_l_inv_06*x109 + R_l_inv_16*x110 - R_l_inv_26*x105 - R_l_inv_36*x107;
        const double x191 = d_4*x190;
        const double x192 = R_l_inv_00*x109 + R_l_inv_10*x110 - R_l_inv_20*x105 - R_l_inv_30*x107;
        const double x193 = a_4*x192;
        const double x194 = -x193;
        const double x195 = x187 + x189 + x191 + x194;
        const double x196 = a_3*x192;
        const double x197 = a_3*x186;
        const double x198 = -x197;
        const double x199 = x196 + x198;
        const double x200 = a_2*x192;
        const double x201 = d_2*x188 - x200;
        const double x202 = R_l_inv_24*x105;
        const double x203 = R_l_inv_34*x107;
        const double x204 = a_2*x186;
        const double x205 = d_2*x190;
        const double x206 = R_l_inv_04*x109;
        const double x207 = R_l_inv_14*x110;
        const double x208 = x202 + x203 + x204 + x205 - x206 - x207;
        const double x209 = x201 + x208;
        const double x210 = R_l_inv_05*x109 + R_l_inv_15*x110 - R_l_inv_25*x105 - R_l_inv_35*x107;
        const double x211 = x28 - x31;
        const double x212 = R_l_inv_02*x153;
        const double x213 = R_l_inv_12*x155;
        const double x214 = x210*x76;
        const double x215 = R_l_inv_22*x149;
        const double x216 = R_l_inv_32*x151;
        const double x217 = d_2*x27;
        const double x218 = -x212 - x213 - x214 + x215 + x216 + x217;
        const double x219 = d_2*x30;
        const double x220 = x10*x210;
        const double x221 = a_4*x147;
        const double x222 = -x219 + x220 + x221;
        const double x223 = x218 + x222;
        const double x224 = -x196;
        const double x225 = -d_2*x188;
        const double x226 = x200 + x224 + x225;
        const double x227 = -x189;
        const double x228 = x191 + x193;
        const double x229 = x187 + x227 + x228;
        const double x230 = x186*x73;
        const double x231 = x13*x190;
        const double x232 = x13*x188;
        const double x233 = x192*x73;
        const double x234 = x232 + x233;
        const double x235 = 8*a_3;
        const double x236 = a_2*x235;
        const double x237 = 4*x21;
        const double x238 = 4*x19;
        const double x239 = -x237 + x238;
        const double x240 = 4*x20;
        const double x241 = 4*x18;
        const double x242 = 4*x22;
        const double x243 = -x240 + x241 + x242;
        const double x244 = -4*R_l_inv_03*x109 - 4*R_l_inv_13*x110 + R_l_inv_23*x176 + R_l_inv_33*x177 + x210*x92 + x239 + x243;
        const double x245 = -x202 - x203 - x204 - x205 + x206 + x207;
        const double x246 = x197 + x245;
        const double x247 = -x33;
        const double x248 = x13*x210 + x247 + x31;
        const double x249 = r_31*x103 - x101*x4 + x101*x6 - x101*x8 + x108*x39;
        const double x250 = r_32*x103 - x106*x4 + x106*x6 - x106*x8 + x108*x43;
        const double x251 = R_l_inv_61*x249 + R_l_inv_71*x250;
        const double x252 = d_4*x251;
        const double x253 = R_l_inv_60*x249 + R_l_inv_70*x250;
        const double x254 = a_3*x253;
        const double x255 = R_l_inv_67*x249 + R_l_inv_77*x250;
        const double x256 = a_4*x255;
        const double x257 = d_2*x251;
        const double x258 = R_l_inv_66*x249 + R_l_inv_76*x250;
        const double x259 = d_4*x258;
        const double x260 = a_2*x253;
        const double x261 = -x260;
        const double x262 = x254 + x256 + x257 + x259 + x261;
        const double x263 = a_2*x255;
        const double x264 = d_2*x258;
        const double x265 = R_l_inv_64*x249;
        const double x266 = -x265;
        const double x267 = R_l_inv_74*x250;
        const double x268 = -x267;
        const double x269 = a_4*x253;
        const double x270 = x263 + x264 + x266 + x268 - x269;
        const double x271 = a_3*x255;
        const double x272 = -x271;
        const double x273 = x25 + x272;
        const double x274 = R_l_inv_65*x249 + R_l_inv_75*x250;
        const double x275 = x13*x274;
        const double x276 = x10*x274;
        const double x277 = x249*x74 + x250*x75 + x274*x76;
        const double x278 = -x276 + x277;
        const double x279 = -x254;
        const double x280 = x256 + x279;
        const double x281 = -d_2*x251;
        const double x282 = x259 + x260 + x281;
        const double x283 = -x252;
        const double x284 = x269 + x283;
        const double x285 = x263 + x264 + x266 + x268 + x284;
        const double x286 = x255*x73;
        const double x287 = x13*x258;
        const double x288 = x13*x251;
        const double x289 = x253*x73;
        const double x290 = x288 + x289;
        const double x291 = x249*x90 + x250*x91 - x274*x92;
        const double x292 = -x14;
        const double x293 = x11 - x16;
        const double x294 = -x256;
        const double x295 = x254 + x294;
        const double x296 = -x259;
        const double x297 = x257 + x261;
        const double x298 = x296 + x297;
        const double x299 = -2*a_4*x274;
        const double x300 = -x10 + x13;
        const double x301 = -x10 - x13;
        const double x302 = 4*x65;
        const double x303 = 4*x66;
        const double x304 = 4*x63;
        const double x305 = 4*x64;
        const double x306 = 4*x52;
        const double x307 = -x136 - x137;
        const double x308 = x142 + x158 + x307;
        const double x309 = 4*x123;
        const double x310 = -x118*x27 + 4*x126 + 4*x127 + 4*x128 + 4*x129 - 4*x133;
        const double x311 = a_3*x174 + a_4*x172;
        const double x312 = x143 + x183 + x307;
        const double x313 = x218 + x28;
        const double x314 = 4*x191;
        const double x315 = x186*x27 + 4*x202 + 4*x203 + 4*x205 - 4*x206 - 4*x207;
        const double x316 = 8*a_2*a_4;
        const double x317 = x212 + x213 + x214 - x215 - x216 - x217 + x28;
        const double x318 = 4*x265;
        const double x319 = 4*x267;
        const double x320 = x255*x27;
        const double x321 = 4*x264;
        const double x322 = a_4*x235 + d_4*x172 + 4*x259;
        const double x323 = -x277;
        const double x324 = x18 + x19 + x20 + x21 + x22 + x23;
        const double x325 = x324 + x9;
        const double x326 = x12 + x16 + x292;
        const double x327 = x325 + x326;
        const double x328 = x28 + x34;
        const double x329 = -x328;
        const double x330 = x14 + x293;
        const double x331 = x325 + x330;
        const double x332 = x50 + x54;
        const double x333 = x85 + x86;
        const double x334 = -x119;
        const double x335 = x120 + x334;
        const double x336 = x136 + x137 + x138 + x140;
        const double x337 = x167 + x168;
        const double x338 = -x187;
        const double x339 = x197 + x338;
        const double x340 = x200 + x225;
        const double x341 = x196 + x208 + x340;
        const double x342 = x189 + x228;
        const double x343 = x230 + x231;
        const double x344 = x271 + x324;
        const double x345 = x326 + x344;
        const double x346 = x270 + x283;
        const double x347 = x276 + x277;
        const double x348 = x252 + x263 + x264 + x266 + x268 + x269;
        const double x349 = x286 + x287 + x328;
        const double x350 = x330 + x344;

        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x3;
        A(0, 2) = x3;
        A(0, 6) = x3;
        A(0, 8) = x3;
        A(1, 0) = x26;
        A(1, 2) = x26;
        A(1, 3) = x36;
        A(1, 5) = x36;
        A(1, 6) = x38;
        A(1, 8) = x38;
        A(2, 0) = x56 + x60 + x68;
        A(2, 1) = x71 + x72 + x78;
        A(2, 2) = x55 + x80 + x82 + x84;
        A(2, 3) = x85 - x86 - x89;
        A(2, 4) = -x27 + x30 + x42*x90 + x46*x91 - x93;
        A(2, 5) = -x85 + x86 - x89;
        A(2, 6) = -x56 - x83 - x94 - x96;
        A(2, 7) = -x71 - x97 - x98;
        A(2, 8) = x100 - x50 + x57 + x58 + x99;
        A(3, 0) = -x113 - x116 - x125 - x135;
        A(3, 1) = x141 + x159 + x160;
        A(3, 2) = x162 + x164 + x166;
        A(3, 3) = x167 - x168 - x171;
        A(3, 4) = x173 + x175 + x178;
        A(3, 5) = -x167 + x168 - x171;
        A(3, 6) = -x162 - x179 - x181;
        A(3, 7) = -x141 - x182 - x184;
        A(3, 8) = x119 + x120 + x123 + x165 + x185;
        A(4, 0) = x195 + x199 + x209;
        A(4, 1) = 2*a_4*x210 - x211 - x223 - x33;
        A(4, 2) = -x198 - x208 - x226 - x229;
        A(4, 3) = x230 - x231 - x234;
        A(4, 4) = x236 - x244;
        A(4, 5) = -x230 + x231 - x234;
        A(4, 6) = -x195 - x226 - x246;
        A(4, 7) = x223 + x248 + x29;
        A(4, 8) = x196 + x201 + x229 + x246;
        A(5, 0) = x252 + x262 + x270 + x273;
        A(5, 1) = x275 + x278;
        A(5, 2) = -x273 - x280 - x282 - x285;
        A(5, 3) = x286 - x287 - x290 - x35;
        A(5, 4) = x291;
        A(5, 5) = -x211 - x247 - x286 + x287 - x290;
        A(5, 6) = x24 + x272 + x285 + x292 + x293 + x295 + x298;
        A(5, 7) = -x278 - x299;
        A(5, 8) = x262 - x263 - x264 + x265 + x267 + x271 + x284 + x37;

        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x300;
        B(0, 2) = x300;
        B(0, 3) = x147;
        B(0, 5) = x147;
        B(0, 6) = x301;
        B(0, 8) = x301;
        B(2, 0) = x78;
        B(2, 1) = x302 + x303 - x304 - x305 - x306;
        B(2, 2) = -x78;
        B(2, 3) = x32;
        B(2, 4) = x235*x51;
        B(2, 5) = -x32;
        B(2, 6) = x98;
        B(2, 7) = -x302 - x303 + x304 + x305 - x306;
        B(2, 8) = -x98;
        B(3, 0) = x308;
        B(3, 1) = -x309 + x310;
        B(3, 2) = -x308;
        B(3, 3) = x311;
        B(3, 4) = x122*x235;
        B(3, 5) = -x311;
        B(3, 6) = x312;
        B(3, 7) = -x309 - x310;
        B(3, 8) = -x312;
        B(4, 0) = -x313;
        B(4, 1) = -x314 - x315;
        B(4, 2) = x313;
        B(4, 3) = x316;
        B(4, 4) = x190*x235;
        B(4, 5) = -x316;
        B(4, 6) = x317;
        B(4, 7) = -x314 + x315;
        B(4, 8) = -x317;
        B(5, 0) = x277;
        B(5, 1) = -x237 + x238 - x243 + x318 + x319 - x320 - x321 - x322;
        B(5, 2) = x323;
        B(5, 4) = 16*a_3*d_2 + 16*a_4*d_4 + x235*x258;
        B(5, 6) = x277;
        B(5, 7) = -x239 - x240 + x241 + x242 - x318 - x319 + x320 + x321 - x322;
        B(5, 8) = x323;

        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x3;
        C(0, 2) = x3;
        C(0, 6) = x3;
        C(0, 8) = x3;
        C(1, 0) = x327;
        C(1, 2) = x327;
        C(1, 3) = x329;
        C(1, 5) = x329;
        C(1, 6) = x331;
        C(1, 8) = x331;
        C(2, 0) = x100 + x332 + x59 + x81 + x94;
        C(2, 1) = -x70 - x78 - x97;
        C(2, 2) = x68 + x80 + x94 + x99;
        C(2, 3) = x333 - x87 + x88;
        C(2, 4) = 4*R_l_inv_63*x42 + 4*R_l_inv_73*x46 - x27 - x30 - x93;
        C(2, 5) = -x333 - x87 + x88;
        C(2, 6) = -x332 - x48 - x67 - x82 - x96;
        C(2, 7) = x70 + x72 + x98;
        C(2, 8) = -x50 - x60 - x61 - x79 - x83 - x95 - x99;
        C(3, 0) = x166 + x181 + x335;
        C(3, 1) = -x159 - x182 - x336;
        C(3, 2) = -x117 - x121 - x124 - x161 - x185 - x334;
        C(3, 3) = -x169 + x170 + x337;
        C(3, 4) = -x173 - x175 + x178;
        C(3, 5) = -x169 + x170 - x337;
        C(3, 6) = -x125 + x126 + x127 + x128 + x129 - x130 - x132 - x133 - x163;
        C(3, 7) = x160 + x184 + x336;
        C(3, 8) = -x164 - x179 - x335;
        C(4, 0) = -x191 - x194 - x227 - x339 - x341;
        C(4, 1) = x219 - x220 - x221 + x248 + x313;
        C(4, 2) = x209 + x224 + x339 + x342;
        C(4, 3) = -x232 + x233 + x343;
        C(4, 4) = -x236 - x244;
        C(4, 5) = -x232 + x233 - x343;
        C(4, 6) = -x187 - x189 + x191 - x193 - x197 - x341;
        C(4, 7) = x222 + x248 + x317;
        C(4, 8) = -x199 - x245 - x338 - x340 - x342;
        C(5, 0) = -x282 - x295 - x345 - x346;
        C(5, 1) = -x299 - x347;
        C(5, 2) = x259 + x279 + x294 + x297 + x345 + x348;
        C(5, 3) = -x288 + x289 + x349;
        C(5, 4) = x291;
        C(5, 5) = -x288 + x289 - x349;
        C(5, 6) = -x254 - x256 - x260 - x281 - x296 - x348 - x350;
        C(5, 7) = x275 + x347;
        C(5, 8) = x280 + x298 + x346 + x350;

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
            const double th_2 = this_solution[2];
            const bool checked_result = ((std::fabs(2*a_2*a_4*std::cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) <= 9.9999999999999995e-7) && (std::fabs(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) <= 9.9999999999999995e-7) && (std::fabs(-std::pow(Px, 2) + 2*Px*a_1*std::cos(th_0) - std::pow(Py, 2) + 2*Py*a_1*std::sin(th_0) - std::pow(Pz, 2) - std::pow(a_1, 2) + std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + std::pow(a_3, 2) + std::pow(a_4, 2) + std::pow(d_2, 2) + std::pow(d_4, 2)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2

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
            const bool condition_0 = ((std::fabs(2*a_2*a_4*std::cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) >= zero_tolerance) || (std::fabs(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) >= zero_tolerance) || (std::fabs(-std::pow(Px, 2) + 2*Px*a_1*std::cos(th_0) - std::pow(Py, 2) + 2*Py*a_1*std::sin(th_0) - std::pow(Pz, 2) - std::pow(a_1, 2) + std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + std::pow(a_3, 2) + std::pow(a_4, 2) + std::pow(d_2, 2) + std::pow(d_4, 2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 2*d_4;
                const double x1 = 2*a_4;
                const double x2 = a_2*std::cos(th_2);
                const double x3 = -a_3*x0 - d_2*x1 - x0*x2;
                const double x4 = a_3*x1 - d_2*x0 + x1*x2;
                const double x5 = std::atan2(x3, x4);
                const double x6 = 2*a_1;
                const double x7 = std::pow(Px, 2) - Px*x6*std::cos(th_0) + std::pow(Py, 2) - Py*x6*std::sin(th_0) + std::pow(Pz, 2) + std::pow(a_1, 2) - std::pow(a_2, 2) - std::pow(a_3, 2) - 2*a_3*x2 - std::pow(a_4, 2) - std::pow(d_2, 2) - std::pow(d_4, 2);
                const double x8 = safe_sqrt(std::pow(x3, 2) + std::pow(x4, 2) - std::pow(x7, 2));
                RawSolution solution_0(this_solution);
                solution_0[3] = x5 + std::atan2(x8, x7);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = x5 + std::atan2(-x8, x7);
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
            const bool checked_result = ((std::fabs(Pz) <= 9.9999999999999995e-7) && (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4

    // Code for explicit solution node 5, solved variable is th_1
    auto ExplicitSolutionNode_node_5_solve_th_1_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const bool condition_0 = ((std::fabs(Pz) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = std::cos(th_2);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_3);
                const double x3 = -a_2 - a_3*x0 - a_4*x0*x1 + d_4*x0*x2;
                const double x4 = -Px*std::cos(th_0) - Py*std::sin(th_0) + a_1;
                const double x5 = a_4*x2 - d_2 + d_4*x1;
                const double tmp_sol_value = std::atan2(Pz*x3 - x4*x5, Pz*x5 + x3*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1_processor();
    // Finish code for explicit solution node 5

    // Code for non-branch dispatcher node 6
    // Actually, there is no code

    // Code for explicit solution node 7, solved variable is th_5
    auto ExplicitSolutionNode_node_7_solve_th_5_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const bool condition_0 = std::fabs(r_13*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_23*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) + r_33*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_1);
                const double x3 = std::cos(th_2);
                const double x4 = std::sin(th_3);
                const double x5 = std::sin(th_0);
                const double x6 = x4*std::sin(th_2);
                const double x7 = std::cos(th_0);
                const double x8 = x0*x3*x4 - x1*x2;
                const double x9 = safe_acos(r_13*(x5*x6 + x7*x8) + r_23*(x5*x8 - x6*x7) - r_33*(x0*x1 + x2*x3*x4));
                RawSolution solution_0(this_solution);
                solution_0[6] = x9;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx_0);
                const double tmp_sol_value = -x9;
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
    ExplicitSolutionNode_node_7_solve_th_5_processor();
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
            const double th_5 = this_solution[6];

            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(12, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
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

    // Code for explicit solution node 17, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance) || (std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                const double tmp_sol_value = std::atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1);
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
    ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 17

    // Code for non-branch dispatcher node 18
    // Actually, there is no code

    // Code for explicit solution node 19, solved variable is th_4
    auto ExplicitSolutionNode_node_19_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
        if (!this_input_valid)
            return;

        // The solution of non-root node 19
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_4_processor();
    // Finish code for explicit solution node 18

    // Code for non-branch dispatcher node 20
    // Actually, there is no code

    // Code for explicit solution node 21, solved variable is th_6
    auto ExplicitSolutionNode_node_21_solve_th_6_processor = [&]() -> void
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
            const double th_4 = this_solution[4];
            const double th_4th_6_soa = this_solution[5];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_6_processor();
    // Finish code for explicit solution node 20

    // Code for explicit solution node 12, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance) || (std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                const double tmp_sol_value = std::atan2(-r_11*x6 + r_21*x7 - r_31*x1, r_12*x6 - r_22*x7 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(13, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 12

    // Code for non-branch dispatcher node 13
    // Actually, there is no code

    // Code for explicit solution node 14, solved variable is th_4
    auto ExplicitSolutionNode_node_14_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;

        // The solution of non-root node 14
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_4_processor();
    // Finish code for explicit solution node 13

    // Code for non-branch dispatcher node 15
    // Actually, there is no code

    // Code for explicit solution node 16, solved variable is th_6
    auto ExplicitSolutionNode_node_16_solve_th_6_processor = [&]() -> void
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
            const double negative_th_6_positive_th_4__soa = this_solution[0];
            const double th_4 = this_solution[4];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_6_processor();
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const double th_5 = this_solution[6];
            const bool condition_0 = ((std::fabs(std::sin(th_5)) >= zero_tolerance) || (std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance) || (std::fabs(r_13*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::cos(th_3)) + r_23*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::sin(th_0) - std::sin(th_2)*std::cos(th_0)*std::cos(th_3)) - r_33*(std::sin(th_1)*std::cos(th_2)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = std::cos(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = std::cos(th_1);
                const double x7 = x2*x6;
                const double x8 = std::sin(th_3);
                const double x9 = std::cos(th_3);
                const double x10 = x4*x9;
                const double x11 = x2*x9;
                const double x12 = x1*x8 + x10*x6;
                const double tmp_sol_value = std::atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(-r_13*(x11*x3 + x12*x5) - r_23*(-x11*x5 + x12*x3) - r_33*(-x1*x10 + x6*x8)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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

    // Code for explicit solution node 11, solved variable is th_6
    auto ExplicitSolutionNode_node_11_solve_th_6_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const double th_5 = this_solution[6];
            const bool condition_0 = ((std::fabs(std::sin(th_5)) >= zero_tolerance) || (std::fabs(r_12*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_22*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) + r_32*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) >= zero_tolerance) || (std::fabs(-r_11*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) - r_21*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) + r_31*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::cos(th_1);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_1);
                const double x4 = std::sin(th_3);
                const double x5 = x4*std::cos(th_2);
                const double x6 = x1*x2 + x3*x5;
                const double x7 = std::cos(th_0);
                const double x8 = x4*std::sin(th_2);
                const double x9 = x7*x8;
                const double x10 = std::sin(th_0);
                const double x11 = -x1*x5 + x2*x3;
                const double x12 = x10*x8;
                const double x13 = -x11;
                const double tmp_sol_value = std::atan2(x0*(r_12*(x11*x7 - x12) + r_22*(x10*x11 + x9) + r_32*x6), x0*(r_11*(x12 + x13*x7) + r_21*(x10*x13 - x9) - r_31*x6));
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_6_processor();
    // Finish code for explicit solution node 10

    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = th_0;  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[1];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[2];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[3];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[4];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[6];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[7];  // th_6
        new_ik_i[6] = value_at_6;
        if (std::all_of(new_ik_i.begin(), new_ik_i.end(), [](double value) { return std::isfinite(value); }))
            ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, double th_0, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    computeRawIK(T_ee_raw, th_0, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_0 + th_0_offset_original2raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = th_0;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
        ik_out_i[4] = addRevoluteOffset(ik_out_i[4], -th_4_offset_original2raw);
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
        ik_out_i[6] = addRevoluteOffset(ik_out_i[6], -th_6_offset_original2raw);
        ik_output.push_back(ik_out_i);
    }
}

static void wrapAngleToPi(std::array<double, robot_nq>& solution_i)
{
    // Wrap angular value to [-pi, pi]
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
    // Revolute unknown th_6
    solution_i[6] = std::atan2(std::sin(solution_i[6]), std::cos(solution_i[6]));
}

static void wrapAngleToPi(std::vector<std::array<double, robot_nq>>& ik_output)
{
    for (auto& solution_i : ik_output)
        wrapAngleToPi(solution_i);
}

static void computeIKForRawTarget(const Eigen::Matrix4d& T_ee, const Eigen::Matrix4d& T_ee_raw, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output, Eigen::Matrix4d& frame_disturbance, bool& frame_disturbance_ready, bool allow_disturbance, std::vector<std::pair<double,std::array<double,robot_nq>>>* refinement_seeds = nullptr)
{
    if (!T_ee.allFinite()) { ik_output.clear(); return; }
    if (!std::isfinite(th_0)) { ik_output.clear(); return; }
    if (!T_ee_raw.allFinite()) { ik_output.clear(); return; }
    computeRawIK(T_ee_raw, th_0 + th_0_offset_original2raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    bool strict_raw_witness = false;
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = th_0;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
        ik_out_i[4] = addRevoluteOffset(ik_out_i[4], -th_4_offset_original2raw);
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
        ik_out_i[6] = addRevoluteOffset(ik_out_i[6], -th_6_offset_original2raw);
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
        computeRawIK(T_ee_raw_disturbed, th_0 + th_0_offset_original2raw, workspace);
        const auto& raw_ik_out_disturb = workspace.raw_ik_out;
        for(int i = 0; i < raw_ik_out_disturb.size(); i++)
        {
            auto ik_out_i = raw_ik_out_disturb[i];
            ik_out_i[0] = th_0;
            ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
            ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
            ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
            ik_out_i[4] = addRevoluteOffset(ik_out_i[4], -th_4_offset_original2raw);
            ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
            ik_out_i[6] = addRevoluteOffset(ik_out_i[6], -th_6_offset_original2raw);
            wrapAngleToPi(ik_out_i);
            Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
            double ee_pose_diff = (ee_pose_i - T_ee_disturbed).squaredNorm();
            if (!std::isfinite(ee_pose_diff) || ee_pose_diff > pose_tolerance_degenerate)
                continue;

            // Try numerical refinement
            yaik_cpp::numericalRefinement<robot_nq>(computeFK, computeTwistJacobian, T_ee, ik_out_i, 1e-4, std::array<bool, robot_nq>{{true, false, false, false, false, false, false}});
            wrapAngleToPi(ik_out_i);
            ee_pose_i = computeFK(ik_out_i);
            ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
            if (ee_pose_diff < pose_tolerance_degenerate)
                ik_output.push_back(ik_out_i);
        }
    }

    removeDuplicateWithinClasses<robot_nq>(ik_output, zero_tolerance, ikCandidateClasses<robot_nq>(ik_output, jointLower(), jointUpper(), periodicLimitJoints(), [&](const std::array<double, robot_nq>& candidate) { return computeFK(candidate); }, T_ee));
}

static void computeIK(const Eigen::Matrix4d& T_ee, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    ik_output.clear();
    if (!T_ee.allFinite()) return;
    if (!std::isfinite(th_0)) return;
    Eigen::Matrix4d frame_disturbance;
    bool frame_disturbance_ready = false;
    std::vector<std::pair<double,std::array<double,robot_nq>>> refinement_seeds;
    for (bool legacy : {false})
    {
        const Eigen::Matrix4d raw_target = legacy ? endEffectorTargetOriginalToRaw(T_ee) : endEffectorTargetOriginalToRaw(T_ee);
        std::vector<std::array<double, robot_nq>> candidates;
        computeIKForRawTarget(T_ee, raw_target, th_0, workspace, candidates, frame_disturbance, frame_disturbance_ready, legacy || ik_output.empty(), &refinement_seeds);
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

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_0)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_0, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct atlas_r_hand_ik

// Code below for debug
void test_ik_solve_atlas_r_hand()
{
    std::array<double, atlas_r_hand_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = atlas_r_hand_ik::computeFK(theta);
    auto ik_output = atlas_r_hand_ik::computeIK(ee_pose, theta[0]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = atlas_r_hand_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_atlas_r_hand();
    return 0;
}
