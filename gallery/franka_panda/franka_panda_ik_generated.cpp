#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct franka_panda_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 1584;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -6.283185307179586, -6.283185307179586, -6.283185307179586, -3.141592653589793, -6.283185307179586, -3.141592653589793}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 0.0, 0.0, 0.0, 3.141592653589793, 0.0, 3.141592653589793}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, false, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, true, false, false, false}}; }
static constexpr int n_tree_nodes = 50;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_3 = 0.00825;
static constexpr double a_5 = 0.088;
static constexpr double d_2 = 0.316;
static constexpr double d_4 = 0.384;
static constexpr double post_transform_d5 = 0.107;
static constexpr double post_transform_sqrt2_over2 = 0.707107;
static constexpr double pre_transform_d4 = 0.333;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = 3.141592653589793;
static constexpr double th_2_offset_original2raw = 3.141592653589793;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 0.0;
static constexpr double th_5_offset_original2raw = 3.141592653589793;
static constexpr double th_6_offset_original2raw = 0.0;

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
    ee_transformed(0, 0) = (1.0/2.0)*(r_11 + r_12)/post_transform_sqrt2_over2;
    ee_transformed(0, 1) = (1.0/2.0)*(-r_11 + r_12)/post_transform_sqrt2_over2;
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px - post_transform_d5*r_13;
    ee_transformed(1, 0) = (1.0/2.0)*(r_21 + r_22)/post_transform_sqrt2_over2;
    ee_transformed(1, 1) = (1.0/2.0)*(-r_21 + r_22)/post_transform_sqrt2_over2;
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py - post_transform_d5*r_23;
    ee_transformed(2, 0) = (1.0/2.0)*(r_31 + r_32)/post_transform_sqrt2_over2;
    ee_transformed(2, 1) = (1.0/2.0)*(-r_31 + r_32)/post_transform_sqrt2_over2;
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz - post_transform_d5*r_33 - pre_transform_d4;
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
    ee_transformed(0, 0) = post_transform_sqrt2_over2*(r_11 - r_12);
    ee_transformed(0, 1) = post_transform_sqrt2_over2*(r_11 + r_12);
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px + post_transform_d5*r_13;
    ee_transformed(1, 0) = post_transform_sqrt2_over2*(r_21 - r_22);
    ee_transformed(1, 1) = post_transform_sqrt2_over2*(r_21 + r_22);
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py + post_transform_d5*r_23;
    ee_transformed(2, 0) = post_transform_sqrt2_over2*(r_31 - r_32);
    ee_transformed(2, 1) = post_transform_sqrt2_over2*(r_31 + r_32);
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz + post_transform_d5*r_33 + pre_transform_d4;
    return ee_transformed;
}

static Eigen::Matrix4d endEffectorTargetOriginalToRawLegacyFrame(const Eigen::Matrix4d& T_ee)
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
    ee_transformed(0, 0) = post_transform_sqrt2_over2*(r_11 + r_12);
    ee_transformed(0, 1) = post_transform_sqrt2_over2*(-r_11 + r_12);
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px - post_transform_d5*r_13;
    ee_transformed(1, 0) = post_transform_sqrt2_over2*(r_21 + r_22);
    ee_transformed(1, 1) = post_transform_sqrt2_over2*(-r_21 + r_22);
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py - post_transform_d5*r_23;
    ee_transformed(2, 0) = post_transform_sqrt2_over2*(r_31 + r_32);
    ee_transformed(2, 1) = post_transform_sqrt2_over2*(-r_31 + r_32);
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz - post_transform_d5*r_33 - pre_transform_d4;
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
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::sin(th_6);
    const double x1 = std::cos(th_4);
    const double x2 = std::sin(th_0);
    const double x3 = std::cos(th_2);
    const double x4 = x2*x3;
    const double x5 = std::sin(th_2);
    const double x6 = std::cos(th_0);
    const double x7 = std::cos(th_1);
    const double x8 = -x4 + x5*x6*x7;
    const double x9 = std::sin(th_4);
    const double x10 = std::sin(th_3);
    const double x11 = std::sin(th_1);
    const double x12 = x11*x6;
    const double x13 = std::cos(th_3);
    const double x14 = x2*x5;
    const double x15 = x3*x6;
    const double x16 = x14 + x15*x7;
    const double x17 = -x10*x12 + x13*x16;
    const double x18 = x1*x8 - x17*x9;
    const double x19 = std::cos(th_6);
    const double x20 = std::sin(th_5);
    const double x21 = x10*x16 + x12*x13;
    const double x22 = std::cos(th_5);
    const double x23 = x1*x17 + x8*x9;
    const double x24 = -x20*x21 + x22*x23;
    const double x25 = x14*x7 + x15;
    const double x26 = x11*x2;
    const double x27 = x4*x7 - x5*x6;
    const double x28 = -x10*x26 + x13*x27;
    const double x29 = x1*x25 - x28*x9;
    const double x30 = x10*x27 + x13*x26;
    const double x31 = x1*x28 + x25*x9;
    const double x32 = -x20*x30 + x22*x31;
    const double x33 = x11*x5;
    const double x34 = x11*x3;
    const double x35 = -x10*x7 - x13*x34;
    const double x36 = -x1*x33 - x35*x9;
    const double x37 = x10*x34 - x13*x7;
    const double x38 = -x37;
    const double x39 = x1*x35 - x33*x9;
    const double x40 = -x20*x38 + x22*x39;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x18 + x19*x24;
    ee_pose_raw(0, 1) = -x0*x24 + x18*x19;
    ee_pose_raw(0, 2) = x20*x23 + x21*x22;
    ee_pose_raw(0, 3) = a_3*x16 + a_3*x17 + a_5*x24 - d_2*x12 + d_4*x21;
    ee_pose_raw(1, 0) = x0*x29 + x19*x32;
    ee_pose_raw(1, 1) = -x0*x32 + x19*x29;
    ee_pose_raw(1, 2) = x20*x31 + x22*x30;
    ee_pose_raw(1, 3) = a_3*x27 + a_3*x28 + a_5*x32 - d_2*x26 + d_4*x30;
    ee_pose_raw(2, 0) = x0*x36 + x19*x40;
    ee_pose_raw(2, 1) = -x0*x40 + x19*x36;
    ee_pose_raw(2, 2) = x20*x39 + x22*x38;
    ee_pose_raw(2, 3) = -a_3*x34 + a_3*x35 + a_5*x40 - d_2*x7 - d_4*x37;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_0);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x2*x6*x7 - x5;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x0*x6;
    const double x12 = x2*x4;
    const double x13 = x11 + x12*x7;
    const double x14 = x10*x13 + x3*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x3 + x13*x9;
    const double x18 = x15*x8 - x16*x17;
    const double x19 = std::cos(th_5);
    const double x20 = std::sin(th_5);
    const double x21 = x15*x17 + x16*x8;
    const double x22 = x14*x19 + x20*x21;
    const double x23 = x0*x1;
    const double x24 = x11*x7 + x12;
    const double x25 = -x2*x6 + x5*x7;
    const double x26 = x10*x25 + x23*x9;
    const double x27 = -x10*x23 + x25*x9;
    const double x28 = x15*x24 - x16*x27;
    const double x29 = x15*x27 + x16*x24;
    const double x30 = x19*x26 + x20*x29;
    const double x31 = x1*x6;
    const double x32 = x1*x4;
    const double x33 = x10*x32 - x7*x9;
    const double x34 = -x33;
    const double x35 = -x10*x7 - x32*x9;
    const double x36 = -x15*x31 - x16*x35;
    const double x37 = x15*x35 - x16*x31;
    const double x38 = x19*x34 + x20*x37;
    const double x39 = d_2*x7;
    const double x40 = -pre_transform_d4 + x39;
    const double x41 = -x40;
    const double x42 = a_3*x32;
    const double x43 = x40 + x42;
    const double x44 = -x43;
    const double x45 = a_3*x25 - d_2*x23;
    const double x46 = d_4*x33;
    const double x47 = a_3*x35 - x43 - x46;
    const double x48 = a_3*x27 + d_4*x26 + x45;
    const double x49 = a_3*x35 + a_5*(x19*x37 - x20*x34) + pre_transform_d4 - x39 - x42 - x46;
    const double x50 = a_5*(x19*x29 - x20*x26) + x48;
    const double x51 = a_3*x13 - d_2*x3;
    const double x52 = a_3*x17 + d_4*x14 + x51;
    const double x53 = a_5*(-x14*x20 + x19*x21) + x52;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x18;
    jacobian(0, 6) = x22;
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x23;
    jacobian(1, 3) = x24;
    jacobian(1, 4) = x26;
    jacobian(1, 5) = x28;
    jacobian(1, 6) = x30;
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x7;
    jacobian(2, 3) = -x31;
    jacobian(2, 4) = x34;
    jacobian(2, 5) = x36;
    jacobian(2, 6) = x38;
    jacobian(3, 1) = -pre_transform_d4*x2;
    jacobian(3, 2) = x23*x39 + x23*x41;
    jacobian(3, 3) = -x24*x44 - x31*x45;
    jacobian(3, 4) = -x26*x47 + x34*x48;
    jacobian(3, 5) = -x28*x47 + x36*x48;
    jacobian(3, 6) = -x30*x49 + x38*x50;
    jacobian(4, 1) = -pre_transform_d4*x0;
    jacobian(4, 2) = -x3*x39 - x3*x41;
    jacobian(4, 3) = x31*x51 + x44*x8;
    jacobian(4, 4) = x14*x47 - x34*x52;
    jacobian(4, 5) = x18*x47 - x36*x52;
    jacobian(4, 6) = x22*x49 - x38*x53;
    jacobian(5, 3) = x24*x51 - x45*x8;
    jacobian(5, 4) = -x14*x48 + x26*x52;
    jacobian(5, 5) = -x18*x48 + x28*x52;
    jacobian(5, 6) = -x22*x50 + x30*x53;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_0);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x2*x6*x7 - x5;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x0*x6;
    const double x12 = x2*x4;
    const double x13 = x11 + x12*x7;
    const double x14 = x10*x13 + x3*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x3 + x13*x9;
    const double x18 = std::cos(th_5);
    const double x19 = std::sin(th_5);
    const double x20 = x0*x1;
    const double x21 = x11*x7 + x12;
    const double x22 = -x2*x6 + x5*x7;
    const double x23 = x10*x22 + x20*x9;
    const double x24 = -x10*x20 + x22*x9;
    const double x25 = x1*x6;
    const double x26 = x1*x4;
    const double x27 = -x10*x26 + x7*x9;
    const double x28 = -x10*x7 - x26*x9;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x15*x8 - x16*x17;
    jacobian(0, 6) = x14*x18 + x19*(x15*x17 + x16*x8);
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x20;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x23;
    jacobian(1, 5) = x15*x21 - x16*x24;
    jacobian(1, 6) = x18*x23 + x19*(x15*x24 + x16*x21);
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x7;
    jacobian(2, 3) = -x25;
    jacobian(2, 4) = x27;
    jacobian(2, 5) = -x15*x25 - x16*x28;
    jacobian(2, 6) = x18*x27 + x19*(x15*x28 - x16*x25);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = addRevoluteOffset(theta_input_original[5], th_5_offset_original2raw);
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];

    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::cos(th_1);
    const double x2 = std::sin(th_1);
    const double x3 = std::sin(th_0);
    const double x4 = p_on_ee_z*x3;
    const double x5 = d_2*x1;
    const double x6 = x2*x3;
    const double x7 = -pre_transform_d4 + x5;
    const double x8 = -x7;
    const double x9 = std::sin(th_2);
    const double x10 = x2*x9;
    const double x11 = std::cos(th_2);
    const double x12 = x0*x11;
    const double x13 = x3*x9;
    const double x14 = x1*x13 + x12;
    const double x15 = x11*x2;
    const double x16 = a_3*x15;
    const double x17 = x16 + x7;
    const double x18 = -x17;
    const double x19 = x11*x3;
    const double x20 = -x0*x9 + x1*x19;
    const double x21 = a_3*x20 - d_2*x6;
    const double x22 = std::cos(th_3);
    const double x23 = std::sin(th_3);
    const double x24 = -x1*x22 + x15*x23;
    const double x25 = -x24;
    const double x26 = x20*x23 + x22*x6;
    const double x27 = d_4*x24;
    const double x28 = -x1*x23 - x15*x22;
    const double x29 = a_3*x28 - x17 - x27;
    const double x30 = x20*x22 - x23*x6;
    const double x31 = a_3*x30 + d_4*x26 + x21;
    const double x32 = std::cos(th_4);
    const double x33 = std::sin(th_4);
    const double x34 = -x10*x32 - x28*x33;
    const double x35 = x14*x32 - x30*x33;
    const double x36 = std::cos(th_5);
    const double x37 = std::sin(th_5);
    const double x38 = -x10*x33 + x28*x32;
    const double x39 = x25*x36 + x37*x38;
    const double x40 = x14*x33 + x30*x32;
    const double x41 = x26*x36 + x37*x40;
    const double x42 = a_3*x28 + a_5*(-x25*x37 + x36*x38) + pre_transform_d4 - x16 - x27 - x5;
    const double x43 = a_5*(-x26*x37 + x36*x40) + x31;
    const double x44 = x0*x2;
    const double x45 = x0*x1*x9 - x19;
    const double x46 = x1*x12 + x13;
    const double x47 = a_3*x46 - d_2*x44;
    const double x48 = x22*x44 + x23*x46;
    const double x49 = x22*x46 - x23*x44;
    const double x50 = a_3*x49 + d_4*x48 + x47;
    const double x51 = x32*x45 - x33*x49;
    const double x52 = x32*x49 + x33*x45;
    const double x53 = x36*x48 + x37*x52;
    const double x54 = a_5*(x36*x52 - x37*x48) + x50;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y;
    jacobian(0, 1) = p_on_ee_z*x0 - pre_transform_d4*x0;
    jacobian(0, 2) = p_on_ee_y*x1 - x2*x4 + x5*x6 + x6*x8;
    jacobian(0, 3) = p_on_ee_y*x10 + p_on_ee_z*x14 - x10*x21 - x14*x18;
    jacobian(0, 4) = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x31 - x26*x29;
    jacobian(0, 5) = -p_on_ee_y*x34 + p_on_ee_z*x35 - x29*x35 + x31*x34;
    jacobian(0, 6) = -p_on_ee_y*x39 + p_on_ee_z*x41 + x39*x43 - x41*x42;
    jacobian(1, 0) = p_on_ee_x;
    jacobian(1, 1) = -pre_transform_d4*x3 + x4;
    jacobian(1, 2) = -p_on_ee_x*x1 + p_on_ee_z*x0*x2 - x44*x5 - x44*x8;
    jacobian(1, 3) = -p_on_ee_x*x10 - p_on_ee_z*x45 + x18*x45 + x2*x47*x9;
    jacobian(1, 4) = p_on_ee_x*x25 - p_on_ee_z*x48 - x25*x50 + x29*x48;
    jacobian(1, 5) = p_on_ee_x*x34 - p_on_ee_z*x51 + x29*x51 - x34*x50;
    jacobian(1, 6) = p_on_ee_x*x39 - p_on_ee_z*x53 - x39*x54 + x42*x53;
    jacobian(2, 1) = -p_on_ee_x*x0 - p_on_ee_y*x3;
    jacobian(2, 2) = p_on_ee_x*x6 - p_on_ee_y*x44;
    jacobian(2, 3) = -p_on_ee_x*x14 + p_on_ee_y*x45 + x14*x47 - x21*x45;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x48 + x26*x50 - x31*x48;
    jacobian(2, 5) = -p_on_ee_x*x35 + p_on_ee_y*x51 - x31*x51 + x35*x50;
    jacobian(2, 6) = -p_on_ee_x*x41 + p_on_ee_y*x53 + x41*x54 - x43*x53;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, double th_3, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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
    node_index_workspace.reset(n_tree_nodes);

    using RawSolution = IntermediateSolution<intermediate_solution_size>;
    auto make_raw_solution = []() -> RawSolution { RawSolution result; result.fill(std::numeric_limits<double>::quiet_NaN()); return result; };

    auto append_solution_to_queue = [&solution_queue](RawSolution solution_2_add) -> int {
        return solution_queue.push_solution(solution_2_add);
    };

    auto add_input_index_to = [&node_index_workspace](int node_idx, int solution_idx) -> void {
        if (solution_idx < 0) return;
        node_index_workspace.append_index_to_node(node_idx, solution_idx);
    };

    // Code for equation all-zero dispatcher node 0
    auto EquationAllZeroDispatcherNode_node_0_processor= [&]()
    {
        if (!node_index_workspace.node_input_validity_vector[0])
            return;
        const bool checked_result = ((2*std::fabs(a_5*inv_Px) <= 9.9999999999999995e-7) && (2*std::fabs(a_5*inv_Py) <= 9.9999999999999995e-7) && (std::fabs(2*std::pow(a_3, 2)*std::cos(th_3) + 2*std::pow(a_3, 2) + 2*a_3*d_2*std::sin(th_3) + 2*a_3*d_4*std::sin(th_3) - std::pow(a_5, 2) + std::pow(d_2, 2) - 2*d_2*d_4*std::cos(th_3) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2)) <= 9.9999999999999995e-7));
        if (!checked_result)  // To non-degenerate node
            node_index_workspace.node_input_validity_vector[1] = true;
        else
            node_index_workspace.node_input_validity_vector[30] = true;
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_0_processor();
    // Finish code for equation all-zero dispatcher node 0

    // Code for explicit solution node 30, solved variable is th_6
    auto ExplicitSolutionNode_node_30_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(30);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(30);
        if (!this_input_valid)
            return;

        // The explicit solution of root node
        const bool condition_0 = true;
        if (condition_0)
        {
            auto solution_0 = make_raw_solution();
            solution_0[5] = -3.14159265358979;
            int appended_idx_0 = append_solution_to_queue(solution_0);
            add_input_index_to(31, appended_idx_0);

            auto solution_1 = make_raw_solution();
            solution_1[5] = -2.9452431127404299;
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(31, appended_idx_1);

            auto solution_2 = make_raw_solution();
            solution_2[5] = -2.7488935718910699;
            int appended_idx_2 = append_solution_to_queue(solution_2);
            add_input_index_to(31, appended_idx_2);

            auto solution_3 = make_raw_solution();
            solution_3[5] = -2.5525440310417098;
            int appended_idx_3 = append_solution_to_queue(solution_3);
            add_input_index_to(31, appended_idx_3);

            auto solution_4 = make_raw_solution();
            solution_4[5] = -2.35619449019234;
            int appended_idx_4 = append_solution_to_queue(solution_4);
            add_input_index_to(31, appended_idx_4);

            auto solution_5 = make_raw_solution();
            solution_5[5] = -2.1598449493429799;
            int appended_idx_5 = append_solution_to_queue(solution_5);
            add_input_index_to(31, appended_idx_5);

            auto solution_6 = make_raw_solution();
            solution_6[5] = -1.96349540849362;
            int appended_idx_6 = append_solution_to_queue(solution_6);
            add_input_index_to(31, appended_idx_6);

            auto solution_7 = make_raw_solution();
            solution_7[5] = -1.76714586764426;
            int appended_idx_7 = append_solution_to_queue(solution_7);
            add_input_index_to(31, appended_idx_7);

            auto solution_8 = make_raw_solution();
            solution_8[5] = -1.5707963267949001;
            int appended_idx_8 = append_solution_to_queue(solution_8);
            add_input_index_to(31, appended_idx_8);

            auto solution_9 = make_raw_solution();
            solution_9[5] = -1.37444678594553;
            int appended_idx_9 = append_solution_to_queue(solution_9);
            add_input_index_to(31, appended_idx_9);

            auto solution_10 = make_raw_solution();
            solution_10[5] = -1.17809724509617;
            int appended_idx_10 = append_solution_to_queue(solution_10);
            add_input_index_to(31, appended_idx_10);

            auto solution_11 = make_raw_solution();
            solution_11[5] = -0.98174770424681101;
            int appended_idx_11 = append_solution_to_queue(solution_11);
            add_input_index_to(31, appended_idx_11);

            auto solution_12 = make_raw_solution();
            solution_12[5] = -0.78539816339744795;
            int appended_idx_12 = append_solution_to_queue(solution_12);
            add_input_index_to(31, appended_idx_12);

            auto solution_13 = make_raw_solution();
            solution_13[5] = -0.58904862254808599;
            int appended_idx_13 = append_solution_to_queue(solution_13);
            add_input_index_to(31, appended_idx_13);

            auto solution_14 = make_raw_solution();
            solution_14[5] = -0.39269908169872397;
            int appended_idx_14 = append_solution_to_queue(solution_14);
            add_input_index_to(31, appended_idx_14);

            auto solution_15 = make_raw_solution();
            solution_15[5] = -0.19634954084936199;
            int appended_idx_15 = append_solution_to_queue(solution_15);
            add_input_index_to(31, appended_idx_15);

            auto solution_16 = make_raw_solution();
            solution_16[5] = 0.0;
            int appended_idx_16 = append_solution_to_queue(solution_16);
            add_input_index_to(31, appended_idx_16);

            auto solution_17 = make_raw_solution();
            solution_17[5] = 0.19634954084936199;
            int appended_idx_17 = append_solution_to_queue(solution_17);
            add_input_index_to(31, appended_idx_17);

            auto solution_18 = make_raw_solution();
            solution_18[5] = 0.39269908169872397;
            int appended_idx_18 = append_solution_to_queue(solution_18);
            add_input_index_to(31, appended_idx_18);

            auto solution_19 = make_raw_solution();
            solution_19[5] = 0.58904862254808599;
            int appended_idx_19 = append_solution_to_queue(solution_19);
            add_input_index_to(31, appended_idx_19);

            auto solution_20 = make_raw_solution();
            solution_20[5] = 0.78539816339744795;
            int appended_idx_20 = append_solution_to_queue(solution_20);
            add_input_index_to(31, appended_idx_20);

            auto solution_21 = make_raw_solution();
            solution_21[5] = 0.98174770424681101;
            int appended_idx_21 = append_solution_to_queue(solution_21);
            add_input_index_to(31, appended_idx_21);

            auto solution_22 = make_raw_solution();
            solution_22[5] = 1.17809724509617;
            int appended_idx_22 = append_solution_to_queue(solution_22);
            add_input_index_to(31, appended_idx_22);

            auto solution_23 = make_raw_solution();
            solution_23[5] = 1.37444678594553;
            int appended_idx_23 = append_solution_to_queue(solution_23);
            add_input_index_to(31, appended_idx_23);

            auto solution_24 = make_raw_solution();
            solution_24[5] = 1.5707963267949001;
            int appended_idx_24 = append_solution_to_queue(solution_24);
            add_input_index_to(31, appended_idx_24);

            auto solution_25 = make_raw_solution();
            solution_25[5] = 1.76714586764426;
            int appended_idx_25 = append_solution_to_queue(solution_25);
            add_input_index_to(31, appended_idx_25);

            auto solution_26 = make_raw_solution();
            solution_26[5] = 1.96349540849362;
            int appended_idx_26 = append_solution_to_queue(solution_26);
            add_input_index_to(31, appended_idx_26);

            auto solution_27 = make_raw_solution();
            solution_27[5] = 2.1598449493429799;
            int appended_idx_27 = append_solution_to_queue(solution_27);
            add_input_index_to(31, appended_idx_27);

            auto solution_28 = make_raw_solution();
            solution_28[5] = 2.35619449019234;
            int appended_idx_28 = append_solution_to_queue(solution_28);
            add_input_index_to(31, appended_idx_28);

            auto solution_29 = make_raw_solution();
            solution_29[5] = 2.5525440310417098;
            int appended_idx_29 = append_solution_to_queue(solution_29);
            add_input_index_to(31, appended_idx_29);

            auto solution_30 = make_raw_solution();
            solution_30[5] = 2.7488935718910699;
            int appended_idx_30 = append_solution_to_queue(solution_30);
            add_input_index_to(31, appended_idx_30);

            auto solution_31 = make_raw_solution();
            solution_31[5] = 2.9452431127404299;
            int appended_idx_31 = append_solution_to_queue(solution_31);
            add_input_index_to(31, appended_idx_31);

            auto solution_32 = make_raw_solution();
            solution_32[5] = 3.14159265358979;
            int appended_idx_32 = append_solution_to_queue(solution_32);
            add_input_index_to(31, appended_idx_32);

        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_30_solve_th_6_processor();
    // Finish code for explicit solution node 30

    // Code for solved_variable dispatcher node 31
    auto SolvedVariableDispatcherNode_node_31_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(31);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(31);
        if (!this_input_valid)
            return;

        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;

            const bool degenerate_valid_0 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }

            const bool degenerate_valid_1 = std::fabs(th_3 - 3.0893893222715398 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }

            if (!taken_by_degenerate)
                add_input_index_to(32, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_31_processor();
    // Finish code for solved_variable dispatcher node 31

    // Code for explicit solution node 32, solved variable is th_4
    auto ExplicitSolutionNode_node_32_solve_th_4_processor = [&]() -> void
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
            const double th_6 = this_solution[5];
            const bool condition_0 = ((a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3) != 0) && (std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3))) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = safe_asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3)));
                RawSolution solution_0(this_solution);
                solution_0[3] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(33, appended_idx_0);
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(33, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_4_processor();
    // Finish code for explicit solution node 32

    // Code for equation all-zero dispatcher node 33
    auto EquationAllZeroDispatcherNode_node_33_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(33);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(33);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_6 = this_solution[5];
            const bool checked_result = ((std::fabs(inv_Pz) <= 9.9999999999999995e-7) && (std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(34, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_33_processor();
    // Finish code for equation all-zero dispatcher node 33

    // Code for explicit solution node 34, solved variable is th_5
    auto ExplicitSolutionNode_node_34_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(34);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(34);
        if (!this_input_valid)
            return;

        // The solution of non-root node 34
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[3];
            const double th_6 = this_solution[5];
            const bool condition_0 = ((std::fabs(inv_Pz) >= 9.9999999999999995e-7) || (std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_3);
                const double x2 = a_3*x0 + a_3 + d_2*x1;
                const double x3 = std::cos(th_4);
                const double x4 = a_3*x1 - d_2*x0 + d_4;
                const double x5 = -a_5 - inv_Px*std::cos(th_6) + inv_Py*std::sin(th_6);
                const double tmp_sol_value = std::atan2(-inv_Pz*x2*x3 - x4*x5, -inv_Pz*x4 + x2*x3*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(35, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_34_solve_th_5_processor();
    // Finish code for explicit solution node 34

    // Code for non-branch dispatcher node 35
    // Actually, there is no code

    // Code for linear type2 solution node 36, solved variable is th_2
    auto LinearSinCosType_2_SolverNode_node_36_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(35);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(35);
        if (!this_input_valid)
            return;
        for (int linear_prefix = 0; linear_prefix < this_node_input_index.size(); ++linear_prefix)
        {
            const int queue_index = this_node_input_index[linear_prefix];
            if (!solution_queue.is_solution_valid(queue_index)) continue;
            const auto& this_solution = solution_queue.get_solution(queue_index);
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const double x0 = std::cos(th_6);
            const double x1 = std::sin(th_3);
            const double x2 = std::sin(th_5);
            const double x3 = x1*x2;
            const double x4 = std::sin(th_6);
            const double x5 = std::sin(th_4);
            const double x6 = std::cos(th_3);
            const double x7 = x5*x6;
            const double x8 = std::cos(th_4);
            const double x9 = std::cos(th_5);
            const double x10 = x6*x9;
            const double x11 = -x0*x10*x8 + x0*x3 + x4*x7;
            const double x12 = x4*x8;
            const double x13 = x5*x9;
            const double x14 = x0*x13 + x12;
            const double x15 = x0*x7 + x10*x12 - x3*x4;
            const double x16 = -x0*x8 + x13*x4;
            const double x17 = x6*x8;
            const double x18 = x1*x9 + x17*x2;
            const double x19 = x2*x5;
            const double x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1;
            const double x21 = a_5*x13;
            Eigen::Matrix<double, 8, 4> A_matrix;
            A_matrix.setZero();
            A_matrix(0, 0) = r_11;
            A_matrix(0, 1) = -r_21;
            A_matrix(0, 2) = x11;
            A_matrix(0, 3) = x14;
            A_matrix(1, 0) = r_12;
            A_matrix(1, 1) = -r_22;
            A_matrix(1, 2) = x15;
            A_matrix(1, 3) = -x16;
            A_matrix(2, 0) = r_13;
            A_matrix(2, 1) = -r_23;
            A_matrix(2, 2) = -x18;
            A_matrix(2, 3) = x19;
            A_matrix(3, 0) = Px;
            A_matrix(3, 1) = -Py;
            A_matrix(3, 2) = -x20;
            A_matrix(3, 3) = x21;
            A_matrix(4, 0) = -r_11;
            A_matrix(4, 1) = r_21;
            A_matrix(4, 2) = -x11;
            A_matrix(4, 3) = -x14;
            A_matrix(5, 0) = -r_12;
            A_matrix(5, 1) = r_22;
            A_matrix(5, 2) = -x15;
            A_matrix(5, 3) = x16;
            A_matrix(6, 0) = -r_13;
            A_matrix(6, 1) = r_23;
            A_matrix(6, 2) = x18;
            A_matrix(6, 3) = -x19;
            A_matrix(7, 0) = -Px;
            A_matrix(7, 1) = Py;
            A_matrix(7, 2) = x20;
            A_matrix(7, 3) = -x21;
            const auto linear_values = ::yaik_cpp::linear_solver::solveLinearType2Candidates<8>(A_matrix, {{{0, 1, 2}}, {{0, 1, 3}}, {{0, 1, 6}}, {{0, 1, 7}}, {{0, 2, 3}}});
            for (std::size_t linear_index = 0; linear_index < linear_values.size(); ++linear_index)
            {
                auto candidate = this_solution;
                candidate[2] = linear_values[linear_index];
                int candidate_index;
                if (linear_index+1 == linear_values.size())
                {
                    solution_queue.get_solution(queue_index) = candidate;
                    candidate_index = queue_index;
                }
                else candidate_index = append_solution_to_queue(candidate);
                add_input_index_to(37, candidate_index);
            }
            if (linear_values.empty()) solution_queue.solutions_validity[queue_index] = false;
        }
    };
    // Invoke the processor
    LinearSinCosType_2_SolverNode_node_36_solve_th_2_processor();
    // Finish code for general_6dof solution node 35

    // Code for equation all-zero dispatcher node 37
    auto EquationAllZeroDispatcherNode_node_37_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(37);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(37);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool checked_result = ((std::fabs(r_13) <= 9.9999999999999995e-7) && (std::fabs(r_23) <= 9.9999999999999995e-7) && (std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(38, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_37_processor();
    // Finish code for equation all-zero dispatcher node 37

    // Code for explicit solution node 38, solved variable is th_0
    auto ExplicitSolutionNode_node_38_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(38);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(38);
        if (!this_input_valid)
            return;

        // The solution of non-root node 38
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool condition_0 = ((std::fabs(r_13) >= zero_tolerance) || (std::fabs(r_23) >= zero_tolerance) || (std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::atan2(r_13, -r_23);
                const double x1 = std::sin(th_2);
                const double x2 = std::sin(th_5);
                const double x3 = x1*x2*std::cos(th_3)*std::cos(th_4) + x1*std::sin(th_3)*std::cos(th_5) - x2*std::sin(th_4)*std::cos(th_2);
                const double x4 = safe_sqrt(std::pow(r_13, 2) + std::pow(r_23, 2) - std::pow(x3, 2));
                RawSolution solution_0(this_solution);
                solution_0[0] = x0 + std::atan2(x4, x3);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(39, appended_idx_0);
                const double tmp_sol_value = x0 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(39, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_38_solve_th_0_processor();
    // Finish code for explicit solution node 38

    // Code for equation all-zero dispatcher node 39
    auto EquationAllZeroDispatcherNode_node_39_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(39);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(39);
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
                add_input_index_to(40, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_39_processor();
    // Finish code for equation all-zero dispatcher node 39

    // Code for explicit solution node 40, solved variable is th_1
    auto ExplicitSolutionNode_node_40_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(40);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(40);
        if (!this_input_valid)
            return;

        // The solution of non-root node 40
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool condition_0 = ((std::fabs(Pz) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x1 = std::sin(th_3);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_5);
                const double x4 = a_5*std::cos(th_5);
                const double x5 = x4*std::cos(th_4);
                const double x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5;
                const double x7 = std::cos(th_2);
                const double x8 = a_3*x7;
                const double x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*std::sin(th_2)*std::sin(th_4) - x8;
                const double tmp_sol_value = std::atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_40_solve_th_1_processor();
    // Finish code for explicit solution node 40

    // Code for explicit solution node 1, solved variable is th_6
    auto ExplicitSolutionNode_node_1_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(1);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(1);
        if (!this_input_valid)
            return;

        // The explicit solution of root node
        const bool condition_0 = ((2*std::fabs(a_5*inv_Px) >= zero_tolerance) || (2*std::fabs(a_5*inv_Py) >= zero_tolerance) || (std::fabs(2*std::pow(a_3, 2)*std::cos(th_3) + 2*std::pow(a_3, 2) + 2*a_3*d_2*std::sin(th_3) + 2*a_3*d_4*std::sin(th_3) - std::pow(a_5, 2) + std::pow(d_2, 2) - 2*d_2*d_4*std::cos(th_3) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2)) >= zero_tolerance));
        if (condition_0)
        {
            const double x0 = 2*a_5;
            const double x1 = std::atan2(inv_Py*x0, -inv_Px*x0);
            const double x2 = std::pow(inv_Px, 2);
            const double x3 = std::pow(a_5, 2);
            const double x4 = 4*x3;
            const double x5 = std::pow(inv_Py, 2);
            const double x6 = 2*std::pow(a_3, 2);
            const double x7 = a_3*std::sin(th_3);
            const double x8 = std::cos(th_3);
            const double x9 = -std::pow(d_2, 2) + 2*d_2*d_4*x8 - 2*d_2*x7 - std::pow(d_4, 2) - 2*d_4*x7 + std::pow(inv_Pz, 2) + x2 + x3 + x5 - x6*x8 - x6;
            const double x10 = safe_sqrt(x2*x4 + x4*x5 - std::pow(x9, 2));
            auto solution_0 = make_raw_solution();
            solution_0[5] = x1 + std::atan2(x10, x9);
            int appended_idx_0 = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx_0);

            auto solution_1 = make_raw_solution();
            solution_1[5] = x1 + std::atan2(-x10, x9);
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx_1);

        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_6_processor();
    // Finish code for explicit solution node 1

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
            bool forwarded_to_child = false;

            const bool degenerate_valid_0 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(12, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_3 - 3.0893893222715398 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(13, branch_input_index);
            }

            if (!taken_by_degenerate)
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor();
    // Finish code for solved_variable dispatcher node 2

    // Code for explicit solution node 13, solved variable is th_4
    auto ExplicitSolutionNode_node_13_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;

        // The solution of non-root node 13
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_6 = this_solution[5];
            const bool condition_0 = ((1.99863771551522*a_3 - 0.0521796239019005*d_2 != 0) && (std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2)) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = safe_asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2));
                RawSolution solution_0(this_solution);
                solution_0[3] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(14, appended_idx_0);
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_4_processor();
    // Finish code for explicit solution node 13

    // Code for non-branch dispatcher node 14
    // Actually, there is no code

    // Code for explicit solution node 15, solved variable is th_5
    auto ExplicitSolutionNode_node_15_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(14);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(14);
        if (!this_input_valid)
            return;

        // The solution of non-root node 15
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[3];
            const double th_6 = this_solution[5];
            const bool condition_0 = ((std::fabs(0.0521796239019005*a_3 + 0.99863771551521896*d_2 - 1.0*d_4) >= 9.9999999999999995e-7) || (std::fabs(1.99863771551522*a_3*std::cos(th_4) - 0.0521796239019005*d_2*std::cos(th_4)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = std::cos(th_4);
                const double x1 = -1.99863771551522*a_3*x0 + 0.0521796239019005*d_2*x0;
                const double x2 = 0.0521796239019005*a_3 + 0.99863771551521896*d_2 - 1.0*d_4;
                const double x3 = 1.0*a_5 + 1.0*inv_Px*std::cos(th_6) - 1.0*inv_Py*std::sin(th_6);
                const double tmp_sol_value = std::atan2(inv_Pz*x1 - x2*x3, inv_Pz*x2 + x1*x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(24, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_5_processor();
    // Finish code for explicit solution node 14

    // Code for non-branch dispatcher node 24
    // Actually, there is no code

    // Code for explicit solution node 25, solved variable is th_2
    auto ExplicitSolutionNode_node_25_solve_th_2_processor = [&]() -> void
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
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) > 9.9999999999999992e-25;
            if (condition_0)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = std::cos(th_5);
                const double x6 = x0*x5;
                const double x7 = r_31*(-x2*x4 - x3*x6) + r_32*(-x2*x3 + x4*x6) - r_33*x0*x1;
                const double x8 = std::sin(th_3);
                const double x9 = std::cos(th_3);
                const double x10 = x0*x9;
                const double x11 = x1*x8 - x2*x5*x9;
                const double x12 = r_31*(-x10*x4 - x11*x3) + r_32*(-x10*x3 + x11*x4) + r_33*(x1*x2*x9 + x5*x8);
                RawSolution solution_0(this_solution);
                solution_0[2] = -std::atan2(-x7, -x12);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(26, appended_idx_0);
                RawSolution solution_1(this_solution);
                solution_1[2] = -std::atan2(x7, x12);
                int appended_idx_1 = append_solution_to_queue(solution_1);
                add_input_index_to(26, appended_idx_1);
            }
            const bool condition_2 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)));
            if (condition_2)
            {
                const double x0 = 6*M_PI;
                const double x1 = std::sin(th_4);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_4);
                const double x4 = std::cos(th_6);
                const double x5 = std::sin(th_6);
                const double x6 = x1*std::cos(th_5);
                const double x7 = -x3*x4 + x5*x6;
                const double x8 = -x3*x5 - x4*x6;
                const double x9 = std::atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2);
                RawSolution solution_2(this_solution);
                solution_2[2] = (1.0/2.0)*std::max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, -x0 + x9 + 3.14159265358979);
                int appended_idx_2 = append_solution_to_queue(solution_2);
                add_input_index_to(26, appended_idx_2);
            }
            const bool condition_3 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)));
            if (condition_3)
            {
                const double x0 = 4*M_PI;
                const double x1 = std::sin(th_4);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_4);
                const double x4 = std::cos(th_6);
                const double x5 = std::sin(th_6);
                const double x6 = x1*std::cos(th_5);
                const double x7 = -x3*x4 + x5*x6;
                const double x8 = -x3*x5 - x4*x6;
                const double x9 = std::atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2);
                RawSolution solution_3(this_solution);
                solution_3[2] = (1.0/2.0)*std::max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, -x0 + x9 + 3.14159265358979);
                int appended_idx_3 = append_solution_to_queue(solution_3);
                add_input_index_to(26, appended_idx_3);
            }
            const bool condition_4 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)))));
            if (condition_4)
            {
                const double x0 = 2*M_PI;
                const double x1 = std::sin(th_4);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_4);
                const double x4 = std::cos(th_6);
                const double x5 = std::sin(th_6);
                const double x6 = x1*std::cos(th_5);
                const double x7 = -x3*x4 + x5*x6;
                const double x8 = -x3*x5 - x4*x6;
                const double x9 = std::atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2);
                RawSolution solution_4(this_solution);
                solution_4[2] = (1.0/2.0)*std::max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, -x0 + x9 + 3.14159265358979);
                int appended_idx_4 = append_solution_to_queue(solution_4);
                add_input_index_to(26, appended_idx_4);
            }
            const bool condition_5 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_5)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1);
                RawSolution solution_5(this_solution);
                solution_5[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_5 = append_solution_to_queue(solution_5);
                add_input_index_to(26, appended_idx_5);
            }
            const bool condition_6 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 2*M_PI) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 2*M_PI)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_6)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 2*M_PI;
                RawSolution solution_6(this_solution);
                solution_6[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_6 = append_solution_to_queue(solution_6);
                add_input_index_to(26, appended_idx_6);
            }
            const bool condition_7 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 4*M_PI) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 4*M_PI)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_7)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 4*M_PI;
                RawSolution solution_7(this_solution);
                solution_7[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_7 = append_solution_to_queue(solution_7);
                add_input_index_to(26, appended_idx_7);
            }
            const bool condition_8 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 6*M_PI) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 6*M_PI)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_8)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 6*M_PI;
                RawSolution solution_8(this_solution);
                solution_8[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_8 = append_solution_to_queue(solution_8);
                add_input_index_to(26, appended_idx_8);
            }
            const bool condition_9 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)));
            if (condition_9)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_9(this_solution);
                solution_9[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979 + 6*M_PI) + (1.0/2.0)*std::min(3.14159265358979, -x12 + 3.14159265358979 + 6*M_PI);
                int appended_idx_9 = append_solution_to_queue(solution_9);
                add_input_index_to(26, appended_idx_9);
            }
            const bool condition_10 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)));
            if (condition_10)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_10(this_solution);
                solution_10[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979 + 4*M_PI) + (1.0/2.0)*std::min(3.14159265358979, -x12 + 3.14159265358979 + 4*M_PI);
                int appended_idx_10 = append_solution_to_queue(solution_10);
                add_input_index_to(26, appended_idx_10);
            }
            const bool condition_11 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)))));
            if (condition_11)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_11(this_solution);
                solution_11[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979 + 2*M_PI) + (1.0/2.0)*std::min(3.14159265358979, -x12 + 3.14159265358979 + 2*M_PI);
                int appended_idx_11 = append_solution_to_queue(solution_11);
                add_input_index_to(26, appended_idx_11);
            }
            const bool condition_12 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) <= std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)))));
            if (condition_12)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_12(this_solution);
                solution_12[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                int appended_idx_12 = append_solution_to_queue(solution_12);
                add_input_index_to(26, appended_idx_12);
            }
            const bool condition_13 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI - 3.14159265358979) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) > std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)))));
            if (condition_13)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*M_PI;
                RawSolution solution_13(this_solution);
                solution_13[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                int appended_idx_13 = append_solution_to_queue(solution_13);
                add_input_index_to(26, appended_idx_13);
            }
            const bool condition_14 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI - 3.14159265358979) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) > std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI + 3.14159265358979)))));
            if (condition_14)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*M_PI;
                RawSolution solution_14(this_solution);
                solution_14[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                int appended_idx_14 = append_solution_to_queue(solution_14);
                add_input_index_to(26, appended_idx_14);
            }
            const bool condition_15 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 6*M_PI - 3.14159265358979) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 6*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) > std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI + 3.14159265358979)))));
            if (condition_15)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*M_PI;
                const double tmp_sol_value = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_2_processor();
    // Finish code for explicit solution node 24

    // Code for non-branch dispatcher node 26
    // Actually, there is no code

    // Code for explicit solution node 27, solved variable is th_1
    auto ExplicitSolutionNode_node_27_solve_th_1_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = std::cos(th_5);
                const double x6 = x0*x5;
                const double x7 = std::sin(th_3);
                const double x8 = x5*x7;
                const double x9 = std::cos(th_3);
                const double x10 = x1*x9;
                const double x11 = x0*x9;
                const double x12 = x1*x7;
                const double x13 = x12 - x2*x5*x9;
                const double x14 = x0*x7;
                const double x15 = x10 + x2*x8;
                const double tmp_sol_value = std::atan2((-r_31*(-x11*x4 - x13*x3) - r_32*(-x11*x3 + x13*x4) - r_33*(x10*x2 + x8))*std::cos(th_2) - (-r_31*(-x2*x4 - x3*x6) - r_32*(-x2*x3 + x4*x6) + r_33*x0*x1)*std::sin(th_2), -r_31*(-x14*x4 + x15*x3) - r_32*(-x14*x3 - x15*x4) - r_33*(x12*x2 - x5*x9));
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(28, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_1_processor();
    // Finish code for explicit solution node 26

    // Code for non-branch dispatcher node 28
    // Actually, there is no code

    // Code for explicit solution node 29, solved variable is th_0
    auto ExplicitSolutionNode_node_29_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(28);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(28);
        if (!this_input_valid)
            return;

        // The solution of non-root node 29
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_5);
                const double x2 = std::sin(th_4);
                const double x3 = x1*x2;
                const double x4 = std::cos(th_4);
                const double x5 = std::cos(th_6);
                const double x6 = std::sin(th_6);
                const double x7 = std::cos(th_5);
                const double x8 = x2*x7;
                const double x9 = -x4*x5 + x6*x8;
                const double x10 = -x4*x6 - x5*x8;
                const double x11 = r_11*x10 + r_12*x9 - r_13*x3;
                const double x12 = r_21*x10 + r_22*x9 - r_23*x3;
                const double x13 = std::sin(th_2);
                const double x14 = std::cos(th_1);
                const double x15 = x13*x14;
                const double x16 = std::sin(th_3);
                const double x17 = x16*x7;
                const double x18 = std::cos(th_3);
                const double x19 = x1*x18;
                const double x20 = x17 + x19*x4;
                const double x21 = x18*x2;
                const double x22 = x1*x16;
                const double x23 = -x18*x4*x7 + x22;
                const double x24 = -x21*x6 - x23*x5;
                const double x25 = -x21*x5 + x23*x6;
                const double x26 = r_11*x24 + r_12*x25 + r_13*x20;
                const double x27 = std::sin(th_1);
                const double x28 = -x18*x7 + x22*x4;
                const double x29 = x16*x2;
                const double x30 = x17*x4 + x19;
                const double x31 = -x29*x6 + x30*x5;
                const double x32 = -x29*x5 - x30*x6;
                const double x33 = r_21*x24 + r_22*x25 + r_23*x20;
                const double tmp_sol_value = std::atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28));
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_0_processor();
    // Finish code for explicit solution node 28

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
            const double th_6 = this_solution[5];
            const bool condition_0 = ((std::fabs(inv_Pz) >= zero_tolerance) || (std::fabs(d_2 + d_4) >= zero_tolerance) || (std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = d_2 + d_4;
                const double tmp_sol_value = std::atan2((a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6))/x0, -inv_Pz/x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(16, node_input_i_idx_in_queue);
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

    // Code for non-branch dispatcher node 16
    // Actually, there is no code

    // Code for explicit solution node 17, solved variable is th_4
    auto ExplicitSolutionNode_node_17_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(16);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(16);
        if (!this_input_valid)
            return;

        // The solution of non-root node 17
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = 0.0;
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
    ExplicitSolutionNode_node_17_solve_th_4_processor();
    // Finish code for explicit solution node 16

    // Code for non-branch dispatcher node 18
    // Actually, there is no code

    // Code for explicit solution node 19, solved variable is th_2
    auto ExplicitSolutionNode_node_19_solve_th_2_processor = [&]() -> void
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
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) > 9.9999999999999992e-25;
            if (condition_0)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = std::cos(th_5);
                const double x6 = x0*x5;
                const double x7 = r_31*(-x2*x4 - x3*x6) + r_32*(-x2*x3 + x4*x6) - r_33*x0*x1;
                const double x8 = std::sin(th_3);
                const double x9 = std::cos(th_3);
                const double x10 = x0*x9;
                const double x11 = x1*x8 - x2*x5*x9;
                const double x12 = r_31*(-x10*x4 - x11*x3) + r_32*(-x10*x3 + x11*x4) + r_33*(x1*x2*x9 + x5*x8);
                RawSolution solution_0(this_solution);
                solution_0[2] = -std::atan2(-x7, -x12);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(20, appended_idx_0);
                RawSolution solution_1(this_solution);
                solution_1[2] = -std::atan2(x7, x12);
                int appended_idx_1 = append_solution_to_queue(solution_1);
                add_input_index_to(20, appended_idx_1);
            }
            const bool condition_2 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)));
            if (condition_2)
            {
                const double x0 = 6*M_PI;
                const double x1 = std::sin(th_4);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_4);
                const double x4 = std::cos(th_6);
                const double x5 = std::sin(th_6);
                const double x6 = x1*std::cos(th_5);
                const double x7 = -x3*x4 + x5*x6;
                const double x8 = -x3*x5 - x4*x6;
                const double x9 = std::atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2);
                RawSolution solution_2(this_solution);
                solution_2[2] = (1.0/2.0)*std::max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, -x0 + x9 + 3.14159265358979);
                int appended_idx_2 = append_solution_to_queue(solution_2);
                add_input_index_to(20, appended_idx_2);
            }
            const bool condition_3 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)));
            if (condition_3)
            {
                const double x0 = 4*M_PI;
                const double x1 = std::sin(th_4);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_4);
                const double x4 = std::cos(th_6);
                const double x5 = std::sin(th_6);
                const double x6 = x1*std::cos(th_5);
                const double x7 = -x3*x4 + x5*x6;
                const double x8 = -x3*x5 - x4*x6;
                const double x9 = std::atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2);
                RawSolution solution_3(this_solution);
                solution_3[2] = (1.0/2.0)*std::max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, -x0 + x9 + 3.14159265358979);
                int appended_idx_3 = append_solution_to_queue(solution_3);
                add_input_index_to(20, appended_idx_3);
            }
            const bool condition_4 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)))));
            if (condition_4)
            {
                const double x0 = 2*M_PI;
                const double x1 = std::sin(th_4);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_4);
                const double x4 = std::cos(th_6);
                const double x5 = std::sin(th_6);
                const double x6 = x1*std::cos(th_5);
                const double x7 = -x3*x4 + x5*x6;
                const double x8 = -x3*x5 - x4*x6;
                const double x9 = std::atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2);
                RawSolution solution_4(this_solution);
                solution_4[2] = (1.0/2.0)*std::max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, -x0 + x9 + 3.14159265358979);
                int appended_idx_4 = append_solution_to_queue(solution_4);
                add_input_index_to(20, appended_idx_4);
            }
            const bool condition_5 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_5)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1);
                RawSolution solution_5(this_solution);
                solution_5[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_5 = append_solution_to_queue(solution_5);
                add_input_index_to(20, appended_idx_5);
            }
            const bool condition_6 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 2*M_PI) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 2*M_PI)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_6)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 2*M_PI;
                RawSolution solution_6(this_solution);
                solution_6[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_6 = append_solution_to_queue(solution_6);
                add_input_index_to(20, appended_idx_6);
            }
            const bool condition_7 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 4*M_PI) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 4*M_PI)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_7)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 4*M_PI;
                RawSolution solution_7(this_solution);
                solution_7[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_7 = append_solution_to_queue(solution_7);
                add_input_index_to(20, appended_idx_7);
            }
            const bool condition_8 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) < 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 6*M_PI) <= std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 6*M_PI)) && (((std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 6*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, std::atan2(r_11*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) + r_12*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) - r_13*std::sin(th_4)*std::sin(th_5), -r_21*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_22*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_23*std::sin(th_4)*std::sin(th_5)) - 2*M_PI + 3.14159265358979)))));
            if (condition_8)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = x0*std::cos(th_5);
                const double x6 = -x2*x3 + x4*x5;
                const double x7 = -x2*x4 - x3*x5;
                const double x8 = std::atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 6*M_PI;
                RawSolution solution_8(this_solution);
                solution_8[2] = (1.0/2.0)*std::max(-3.14159265358979, x8 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, x8 + 3.14159265358979);
                int appended_idx_8 = append_solution_to_queue(solution_8);
                add_input_index_to(20, appended_idx_8);
            }
            const bool condition_9 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)));
            if (condition_9)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_9(this_solution);
                solution_9[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979 + 6*M_PI) + (1.0/2.0)*std::min(3.14159265358979, -x12 + 3.14159265358979 + 6*M_PI);
                int appended_idx_9 = append_solution_to_queue(solution_9);
                add_input_index_to(20, appended_idx_9);
            }
            const bool condition_10 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)));
            if (condition_10)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_10(this_solution);
                solution_10[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979 + 4*M_PI) + (1.0/2.0)*std::min(3.14159265358979, -x12 + 3.14159265358979 + 4*M_PI);
                int appended_idx_10 = append_solution_to_queue(solution_10);
                add_input_index_to(20, appended_idx_10);
            }
            const bool condition_11 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)))));
            if (condition_11)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_11(this_solution);
                solution_11[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979 + 2*M_PI) + (1.0/2.0)*std::min(3.14159265358979, -x12 + 3.14159265358979 + 2*M_PI);
                int appended_idx_11 = append_solution_to_queue(solution_11);
                add_input_index_to(20, appended_idx_11);
            }
            const bool condition_12 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) <= std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)))));
            if (condition_12)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5);
                RawSolution solution_12(this_solution);
                solution_12[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                int appended_idx_12 = append_solution_to_queue(solution_12);
                add_input_index_to(20, appended_idx_12);
            }
            const bool condition_13 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI - 3.14159265358979) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) > std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)))));
            if (condition_13)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*M_PI;
                RawSolution solution_13(this_solution);
                solution_13[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                int appended_idx_13 = append_solution_to_queue(solution_13);
                add_input_index_to(20, appended_idx_13);
            }
            const bool condition_14 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI - 3.14159265358979) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) > std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI + 3.14159265358979)))));
            if (condition_14)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*M_PI;
                RawSolution solution_14(this_solution);
                solution_14[2] = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                int appended_idx_14 = append_solution_to_queue(solution_14);
                add_input_index_to(20, appended_idx_14);
            }
            const bool condition_15 = ((-r_31*((std::sin(th_3)*std::cos(th_4)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3))*std::cos(th_6) - std::sin(th_3)*std::sin(th_4)*std::sin(th_6)) - r_32*((-std::sin(th_3)*std::cos(th_4)*std::cos(th_5) - std::sin(th_5)*std::cos(th_3))*std::sin(th_6) - std::sin(th_3)*std::sin(th_4)*std::cos(th_6)) - r_33*(std::sin(th_3)*std::sin(th_5)*std::cos(th_4) - std::cos(th_3)*std::cos(th_5)) >= 0) && (std::pow(r_31*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_32*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_33*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), 2) + std::pow(-r_31*(-std::sin(th_4)*std::cos(th_5)*std::cos(th_6) - std::sin(th_6)*std::cos(th_4)) - r_32*(std::sin(th_4)*std::sin(th_6)*std::cos(th_5) - std::cos(th_4)*std::cos(th_6)) + r_33*std::sin(th_4)*std::sin(th_5), 2) <= 1.0e-14) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 6*M_PI - 3.14159265358979) <= std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 6*M_PI + 3.14159265358979)) && (((std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979) > std::min(3.14159265358979, 3.14159265358979 - std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))))) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 2*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 2*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 4*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 4*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 3.14159265358979 + 6*M_PI) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) + 3.14159265358979 + 6*M_PI)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI - 3.14159265358979) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 4*M_PI + 3.14159265358979)) && (std::max(-3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI - 3.14159265358979) > std::min(3.14159265358979, -std::atan2(r_21*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_22*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_23*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4)), r_11*((-std::sin(th_3)*std::sin(th_5) + std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::cos(th_6) - std::sin(th_4)*std::sin(th_6)*std::cos(th_3)) + r_12*((std::sin(th_3)*std::sin(th_5) - std::cos(th_3)*std::cos(th_4)*std::cos(th_5))*std::sin(th_6) - std::sin(th_4)*std::cos(th_3)*std::cos(th_6)) + r_13*(std::sin(th_3)*std::cos(th_5) + std::sin(th_5)*std::cos(th_3)*std::cos(th_4))) - 2*M_PI + 3.14159265358979)))));
            if (condition_15)
            {
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_5);
                const double x2 = std::sin(th_5);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_4);
                const double x5 = x0*x1 + x2*x3*x4;
                const double x6 = std::sin(th_6);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_6);
                const double x9 = x0*x2 - x1*x3*x4;
                const double x10 = -x6*x7 - x8*x9;
                const double x11 = x6*x9 - x7*x8;
                const double x12 = std::atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*M_PI;
                const double tmp_sol_value = (1.0/2.0)*std::max(-3.14159265358979, -x12 - 3.14159265358979) + (1.0/2.0)*std::min(3.14159265358979, 3.14159265358979 - x12);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_2_processor();
    // Finish code for explicit solution node 18

    // Code for non-branch dispatcher node 20
    // Actually, there is no code

    // Code for explicit solution node 21, solved variable is th_1
    auto ExplicitSolutionNode_node_21_solve_th_1_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::sin(th_4);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_6);
                const double x4 = std::sin(th_6);
                const double x5 = std::cos(th_5);
                const double x6 = x0*x5;
                const double x7 = std::sin(th_3);
                const double x8 = x5*x7;
                const double x9 = std::cos(th_3);
                const double x10 = x1*x9;
                const double x11 = x0*x9;
                const double x12 = x1*x7;
                const double x13 = x12 - x2*x5*x9;
                const double x14 = x0*x7;
                const double x15 = x10 + x2*x8;
                const double tmp_sol_value = std::atan2((-r_31*(-x11*x4 - x13*x3) - r_32*(-x11*x3 + x13*x4) - r_33*(x10*x2 + x8))*std::cos(th_2) - (-r_31*(-x2*x4 - x3*x6) - r_32*(-x2*x3 + x4*x6) + r_33*x0*x1)*std::sin(th_2), -r_31*(-x14*x4 + x15*x3) - r_32*(-x14*x3 - x15*x4) - r_33*(x12*x2 - x5*x9));
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_1_processor();
    // Finish code for explicit solution node 20

    // Code for non-branch dispatcher node 22
    // Actually, there is no code

    // Code for explicit solution node 23, solved variable is th_0
    auto ExplicitSolutionNode_node_23_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(22);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(22);
        if (!this_input_valid)
            return;

        // The solution of non-root node 23
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_5);
                const double x2 = std::sin(th_4);
                const double x3 = x1*x2;
                const double x4 = std::cos(th_4);
                const double x5 = std::cos(th_6);
                const double x6 = std::sin(th_6);
                const double x7 = std::cos(th_5);
                const double x8 = x2*x7;
                const double x9 = -x4*x5 + x6*x8;
                const double x10 = -x4*x6 - x5*x8;
                const double x11 = r_11*x10 + r_12*x9 - r_13*x3;
                const double x12 = r_21*x10 + r_22*x9 - r_23*x3;
                const double x13 = std::sin(th_2);
                const double x14 = std::cos(th_1);
                const double x15 = x13*x14;
                const double x16 = std::sin(th_3);
                const double x17 = x16*x7;
                const double x18 = std::cos(th_3);
                const double x19 = x1*x18;
                const double x20 = x17 + x19*x4;
                const double x21 = x18*x2;
                const double x22 = x1*x16;
                const double x23 = -x18*x4*x7 + x22;
                const double x24 = -x21*x6 - x23*x5;
                const double x25 = -x21*x5 + x23*x6;
                const double x26 = r_11*x24 + r_12*x25 + r_13*x20;
                const double x27 = std::sin(th_1);
                const double x28 = -x18*x7 + x22*x4;
                const double x29 = x16*x2;
                const double x30 = x17*x4 + x19;
                const double x31 = -x29*x6 + x30*x5;
                const double x32 = -x29*x5 - x30*x6;
                const double x33 = r_21*x24 + r_22*x25 + r_23*x20;
                const double tmp_sol_value = std::atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28));
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_0_processor();
    // Finish code for explicit solution node 22

    // Code for explicit solution node 3, solved variable is th_4
    auto ExplicitSolutionNode_node_3_solve_th_4_processor = [&]() -> void
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
            const double th_6 = this_solution[5];
            const bool condition_0 = ((a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3) != 0) && (std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3))) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = safe_asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3)));
                RawSolution solution_0(this_solution);
                solution_0[3] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = M_PI - x0;
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
    ExplicitSolutionNode_node_3_solve_th_4_processor();
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
            const double th_6 = this_solution[5];
            const bool checked_result = ((std::fabs(inv_Pz) <= 9.9999999999999995e-7) && (std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
            else
                add_input_index_to(41, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4

    // Code for explicit solution node 41, solved variable is th_5
    auto ExplicitSolutionNode_node_41_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(41);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(41);
        if (!this_input_valid)
            return;

        // The solution of non-root node 41
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool condition_0 = true;
            if (condition_0)
            {
                RawSolution solution_0(this_solution);
                solution_0[4] = -3.14159265358979;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(42, appended_idx_0);
                RawSolution solution_1(this_solution);
                solution_1[4] = -2.9452431127404299;
                int appended_idx_1 = append_solution_to_queue(solution_1);
                add_input_index_to(42, appended_idx_1);
                RawSolution solution_2(this_solution);
                solution_2[4] = -2.7488935718910699;
                int appended_idx_2 = append_solution_to_queue(solution_2);
                add_input_index_to(42, appended_idx_2);
                RawSolution solution_3(this_solution);
                solution_3[4] = -2.5525440310417098;
                int appended_idx_3 = append_solution_to_queue(solution_3);
                add_input_index_to(42, appended_idx_3);
                RawSolution solution_4(this_solution);
                solution_4[4] = -2.35619449019234;
                int appended_idx_4 = append_solution_to_queue(solution_4);
                add_input_index_to(42, appended_idx_4);
                RawSolution solution_5(this_solution);
                solution_5[4] = -2.1598449493429799;
                int appended_idx_5 = append_solution_to_queue(solution_5);
                add_input_index_to(42, appended_idx_5);
                RawSolution solution_6(this_solution);
                solution_6[4] = -1.96349540849362;
                int appended_idx_6 = append_solution_to_queue(solution_6);
                add_input_index_to(42, appended_idx_6);
                RawSolution solution_7(this_solution);
                solution_7[4] = -1.76714586764426;
                int appended_idx_7 = append_solution_to_queue(solution_7);
                add_input_index_to(42, appended_idx_7);
                RawSolution solution_8(this_solution);
                solution_8[4] = -1.5707963267949001;
                int appended_idx_8 = append_solution_to_queue(solution_8);
                add_input_index_to(42, appended_idx_8);
                RawSolution solution_9(this_solution);
                solution_9[4] = -1.37444678594553;
                int appended_idx_9 = append_solution_to_queue(solution_9);
                add_input_index_to(42, appended_idx_9);
                RawSolution solution_10(this_solution);
                solution_10[4] = -1.17809724509617;
                int appended_idx_10 = append_solution_to_queue(solution_10);
                add_input_index_to(42, appended_idx_10);
                RawSolution solution_11(this_solution);
                solution_11[4] = -0.98174770424681101;
                int appended_idx_11 = append_solution_to_queue(solution_11);
                add_input_index_to(42, appended_idx_11);
                RawSolution solution_12(this_solution);
                solution_12[4] = -0.78539816339744795;
                int appended_idx_12 = append_solution_to_queue(solution_12);
                add_input_index_to(42, appended_idx_12);
                RawSolution solution_13(this_solution);
                solution_13[4] = -0.58904862254808599;
                int appended_idx_13 = append_solution_to_queue(solution_13);
                add_input_index_to(42, appended_idx_13);
                RawSolution solution_14(this_solution);
                solution_14[4] = -0.39269908169872397;
                int appended_idx_14 = append_solution_to_queue(solution_14);
                add_input_index_to(42, appended_idx_14);
                RawSolution solution_15(this_solution);
                solution_15[4] = -0.19634954084936199;
                int appended_idx_15 = append_solution_to_queue(solution_15);
                add_input_index_to(42, appended_idx_15);
                RawSolution solution_16(this_solution);
                solution_16[4] = 0.0;
                int appended_idx_16 = append_solution_to_queue(solution_16);
                add_input_index_to(42, appended_idx_16);
                RawSolution solution_17(this_solution);
                solution_17[4] = 0.19634954084936199;
                int appended_idx_17 = append_solution_to_queue(solution_17);
                add_input_index_to(42, appended_idx_17);
                RawSolution solution_18(this_solution);
                solution_18[4] = 0.39269908169872397;
                int appended_idx_18 = append_solution_to_queue(solution_18);
                add_input_index_to(42, appended_idx_18);
                RawSolution solution_19(this_solution);
                solution_19[4] = 0.58904862254808599;
                int appended_idx_19 = append_solution_to_queue(solution_19);
                add_input_index_to(42, appended_idx_19);
                RawSolution solution_20(this_solution);
                solution_20[4] = 0.78539816339744795;
                int appended_idx_20 = append_solution_to_queue(solution_20);
                add_input_index_to(42, appended_idx_20);
                RawSolution solution_21(this_solution);
                solution_21[4] = 0.98174770424681101;
                int appended_idx_21 = append_solution_to_queue(solution_21);
                add_input_index_to(42, appended_idx_21);
                RawSolution solution_22(this_solution);
                solution_22[4] = 1.17809724509617;
                int appended_idx_22 = append_solution_to_queue(solution_22);
                add_input_index_to(42, appended_idx_22);
                RawSolution solution_23(this_solution);
                solution_23[4] = 1.37444678594553;
                int appended_idx_23 = append_solution_to_queue(solution_23);
                add_input_index_to(42, appended_idx_23);
                RawSolution solution_24(this_solution);
                solution_24[4] = 1.5707963267949001;
                int appended_idx_24 = append_solution_to_queue(solution_24);
                add_input_index_to(42, appended_idx_24);
                RawSolution solution_25(this_solution);
                solution_25[4] = 1.76714586764426;
                int appended_idx_25 = append_solution_to_queue(solution_25);
                add_input_index_to(42, appended_idx_25);
                RawSolution solution_26(this_solution);
                solution_26[4] = 1.96349540849362;
                int appended_idx_26 = append_solution_to_queue(solution_26);
                add_input_index_to(42, appended_idx_26);
                RawSolution solution_27(this_solution);
                solution_27[4] = 2.1598449493429799;
                int appended_idx_27 = append_solution_to_queue(solution_27);
                add_input_index_to(42, appended_idx_27);
                RawSolution solution_28(this_solution);
                solution_28[4] = 2.35619449019234;
                int appended_idx_28 = append_solution_to_queue(solution_28);
                add_input_index_to(42, appended_idx_28);
                RawSolution solution_29(this_solution);
                solution_29[4] = 2.5525440310417098;
                int appended_idx_29 = append_solution_to_queue(solution_29);
                add_input_index_to(42, appended_idx_29);
                RawSolution solution_30(this_solution);
                solution_30[4] = 2.7488935718910699;
                int appended_idx_30 = append_solution_to_queue(solution_30);
                add_input_index_to(42, appended_idx_30);
                RawSolution solution_31(this_solution);
                solution_31[4] = 2.9452431127404299;
                int appended_idx_31 = append_solution_to_queue(solution_31);
                add_input_index_to(42, appended_idx_31);
                const double tmp_sol_value = 3.14159265358979;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(42, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_5_processor();
    // Finish code for explicit solution node 41

    // Code for non-branch dispatcher node 42
    // Actually, there is no code

    // Code for linear type2 solution node 43, solved variable is th_2
    auto LinearSinCosType_2_SolverNode_node_43_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(42);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(42);
        if (!this_input_valid)
            return;
        for (int linear_prefix = 0; linear_prefix < this_node_input_index.size(); ++linear_prefix)
        {
            const int queue_index = this_node_input_index[linear_prefix];
            if (!solution_queue.is_solution_valid(queue_index)) continue;
            const auto& this_solution = solution_queue.get_solution(queue_index);
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const double x0 = std::cos(th_6);
            const double x1 = std::sin(th_3);
            const double x2 = std::sin(th_5);
            const double x3 = x1*x2;
            const double x4 = std::sin(th_6);
            const double x5 = std::sin(th_4);
            const double x6 = std::cos(th_3);
            const double x7 = x5*x6;
            const double x8 = std::cos(th_4);
            const double x9 = std::cos(th_5);
            const double x10 = x6*x9;
            const double x11 = -x0*x10*x8 + x0*x3 + x4*x7;
            const double x12 = x4*x8;
            const double x13 = x5*x9;
            const double x14 = x0*x13 + x12;
            const double x15 = x0*x7 + x10*x12 - x3*x4;
            const double x16 = -x0*x8 + x13*x4;
            const double x17 = x6*x8;
            const double x18 = x1*x9 + x17*x2;
            const double x19 = x2*x5;
            const double x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1;
            const double x21 = a_5*x13;
            Eigen::Matrix<double, 8, 4> A_matrix;
            A_matrix.setZero();
            A_matrix(0, 0) = r_11;
            A_matrix(0, 1) = -r_21;
            A_matrix(0, 2) = x11;
            A_matrix(0, 3) = x14;
            A_matrix(1, 0) = r_12;
            A_matrix(1, 1) = -r_22;
            A_matrix(1, 2) = x15;
            A_matrix(1, 3) = -x16;
            A_matrix(2, 0) = r_13;
            A_matrix(2, 1) = -r_23;
            A_matrix(2, 2) = -x18;
            A_matrix(2, 3) = x19;
            A_matrix(3, 0) = Px;
            A_matrix(3, 1) = -Py;
            A_matrix(3, 2) = -x20;
            A_matrix(3, 3) = x21;
            A_matrix(4, 0) = -r_11;
            A_matrix(4, 1) = r_21;
            A_matrix(4, 2) = -x11;
            A_matrix(4, 3) = -x14;
            A_matrix(5, 0) = -r_12;
            A_matrix(5, 1) = r_22;
            A_matrix(5, 2) = -x15;
            A_matrix(5, 3) = x16;
            A_matrix(6, 0) = -r_13;
            A_matrix(6, 1) = r_23;
            A_matrix(6, 2) = x18;
            A_matrix(6, 3) = -x19;
            A_matrix(7, 0) = -Px;
            A_matrix(7, 1) = Py;
            A_matrix(7, 2) = x20;
            A_matrix(7, 3) = -x21;
            const auto linear_values = ::yaik_cpp::linear_solver::solveLinearType2Candidates<8>(A_matrix, {{{0, 1, 2}}, {{0, 1, 3}}, {{0, 1, 6}}, {{0, 1, 7}}, {{0, 2, 3}}});
            for (std::size_t linear_index = 0; linear_index < linear_values.size(); ++linear_index)
            {
                auto candidate = this_solution;
                candidate[2] = linear_values[linear_index];
                int candidate_index;
                if (linear_index+1 == linear_values.size())
                {
                    solution_queue.get_solution(queue_index) = candidate;
                    candidate_index = queue_index;
                }
                else candidate_index = append_solution_to_queue(candidate);
                add_input_index_to(44, candidate_index);
            }
            if (linear_values.empty()) solution_queue.solutions_validity[queue_index] = false;
        }
    };
    // Invoke the processor
    LinearSinCosType_2_SolverNode_node_43_solve_th_2_processor();
    // Finish code for general_6dof solution node 42

    // Code for equation all-zero dispatcher node 44
    auto EquationAllZeroDispatcherNode_node_44_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(44);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(44);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool checked_result = ((std::fabs(r_13) <= 9.9999999999999995e-7) && (std::fabs(r_23) <= 9.9999999999999995e-7) && (std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(45, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_44_processor();
    // Finish code for equation all-zero dispatcher node 44

    // Code for explicit solution node 45, solved variable is th_0
    auto ExplicitSolutionNode_node_45_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(45);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(45);
        if (!this_input_valid)
            return;

        // The solution of non-root node 45
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool condition_0 = ((std::fabs(r_13) >= zero_tolerance) || (std::fabs(r_23) >= zero_tolerance) || (std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::atan2(r_13, -r_23);
                const double x1 = std::sin(th_2);
                const double x2 = std::sin(th_5);
                const double x3 = x1*x2*std::cos(th_3)*std::cos(th_4) + x1*std::sin(th_3)*std::cos(th_5) - x2*std::sin(th_4)*std::cos(th_2);
                const double x4 = safe_sqrt(std::pow(r_13, 2) + std::pow(r_23, 2) - std::pow(x3, 2));
                RawSolution solution_0(this_solution);
                solution_0[0] = x0 + std::atan2(x4, x3);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(46, appended_idx_0);
                const double tmp_sol_value = x0 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(46, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_45_solve_th_0_processor();
    // Finish code for explicit solution node 45

    // Code for equation all-zero dispatcher node 46
    auto EquationAllZeroDispatcherNode_node_46_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(46);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(46);
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
                add_input_index_to(47, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_46_processor();
    // Finish code for equation all-zero dispatcher node 46

    // Code for explicit solution node 47, solved variable is th_1
    auto ExplicitSolutionNode_node_47_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(47);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(47);
        if (!this_input_valid)
            return;

        // The solution of non-root node 47
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool condition_0 = ((std::fabs(Pz) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x1 = std::sin(th_3);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_5);
                const double x4 = a_5*std::cos(th_5);
                const double x5 = x4*std::cos(th_4);
                const double x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5;
                const double x7 = std::cos(th_2);
                const double x8 = a_3*x7;
                const double x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*std::sin(th_2)*std::sin(th_4) - x8;
                const double tmp_sol_value = std::atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_47_solve_th_1_processor();
    // Finish code for explicit solution node 47

    // Code for explicit solution node 5, solved variable is th_5
    auto ExplicitSolutionNode_node_5_solve_th_5_processor = [&]() -> void
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
            const double th_4 = this_solution[3];
            const double th_6 = this_solution[5];
            const bool condition_0 = ((std::fabs(inv_Pz) >= 9.9999999999999995e-7) || (std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_3);
                const double x2 = a_3*x0 + a_3 + d_2*x1;
                const double x3 = std::cos(th_4);
                const double x4 = a_3*x1 - d_2*x0 + d_4;
                const double x5 = -a_5 - inv_Px*std::cos(th_6) + inv_Py*std::sin(th_6);
                const double tmp_sol_value = std::atan2(-inv_Pz*x2*x3 - x4*x5, -inv_Pz*x4 + x2*x3*x5);
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
    ExplicitSolutionNode_node_5_solve_th_5_processor();
    // Finish code for explicit solution node 5

    // Code for non-branch dispatcher node 6
    // Actually, there is no code

    // Code for linear type2 solution node 7, solved variable is th_2
    auto LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(6);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(6);
        if (!this_input_valid)
            return;
        for (int linear_prefix = 0; linear_prefix < this_node_input_index.size(); ++linear_prefix)
        {
            const int queue_index = this_node_input_index[linear_prefix];
            if (!solution_queue.is_solution_valid(queue_index)) continue;
            const auto& this_solution = solution_queue.get_solution(queue_index);
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const double x0 = std::cos(th_6);
            const double x1 = std::sin(th_3);
            const double x2 = std::sin(th_5);
            const double x3 = x1*x2;
            const double x4 = std::sin(th_6);
            const double x5 = std::sin(th_4);
            const double x6 = std::cos(th_3);
            const double x7 = x5*x6;
            const double x8 = std::cos(th_4);
            const double x9 = std::cos(th_5);
            const double x10 = x6*x9;
            const double x11 = -x0*x10*x8 + x0*x3 + x4*x7;
            const double x12 = x4*x8;
            const double x13 = x5*x9;
            const double x14 = x0*x13 + x12;
            const double x15 = x0*x7 + x10*x12 - x3*x4;
            const double x16 = -x0*x8 + x13*x4;
            const double x17 = x6*x8;
            const double x18 = x1*x9 + x17*x2;
            const double x19 = x2*x5;
            const double x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1;
            const double x21 = a_5*x13;
            Eigen::Matrix<double, 8, 4> A_matrix;
            A_matrix.setZero();
            A_matrix(0, 0) = r_11;
            A_matrix(0, 1) = -r_21;
            A_matrix(0, 2) = x11;
            A_matrix(0, 3) = x14;
            A_matrix(1, 0) = r_12;
            A_matrix(1, 1) = -r_22;
            A_matrix(1, 2) = x15;
            A_matrix(1, 3) = -x16;
            A_matrix(2, 0) = r_13;
            A_matrix(2, 1) = -r_23;
            A_matrix(2, 2) = -x18;
            A_matrix(2, 3) = x19;
            A_matrix(3, 0) = Px;
            A_matrix(3, 1) = -Py;
            A_matrix(3, 2) = -x20;
            A_matrix(3, 3) = x21;
            A_matrix(4, 0) = -r_11;
            A_matrix(4, 1) = r_21;
            A_matrix(4, 2) = -x11;
            A_matrix(4, 3) = -x14;
            A_matrix(5, 0) = -r_12;
            A_matrix(5, 1) = r_22;
            A_matrix(5, 2) = -x15;
            A_matrix(5, 3) = x16;
            A_matrix(6, 0) = -r_13;
            A_matrix(6, 1) = r_23;
            A_matrix(6, 2) = x18;
            A_matrix(6, 3) = -x19;
            A_matrix(7, 0) = -Px;
            A_matrix(7, 1) = Py;
            A_matrix(7, 2) = x20;
            A_matrix(7, 3) = -x21;
            const auto linear_values = ::yaik_cpp::linear_solver::solveLinearType2Candidates<8>(A_matrix, {{{0, 1, 2}}, {{0, 1, 3}}, {{0, 1, 6}}, {{0, 1, 7}}, {{0, 2, 3}}});
            for (std::size_t linear_index = 0; linear_index < linear_values.size(); ++linear_index)
            {
                auto candidate = this_solution;
                candidate[2] = linear_values[linear_index];
                int candidate_index;
                if (linear_index+1 == linear_values.size())
                {
                    solution_queue.get_solution(queue_index) = candidate;
                    candidate_index = queue_index;
                }
                else candidate_index = append_solution_to_queue(candidate);
                add_input_index_to(8, candidate_index);
            }
            if (linear_values.empty()) solution_queue.solutions_validity[queue_index] = false;
        }
    };
    // Invoke the processor
    LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor();
    // Finish code for general_6dof solution node 6

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
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool checked_result = ((std::fabs(r_13) <= 9.9999999999999995e-7) && (std::fabs(r_23) <= 9.9999999999999995e-7) && (std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8

    // Code for explicit solution node 9, solved variable is th_0
    auto ExplicitSolutionNode_node_9_solve_th_0_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool condition_0 = ((std::fabs(r_13) >= zero_tolerance) || (std::fabs(r_23) >= zero_tolerance) || (std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::atan2(r_13, -r_23);
                const double x1 = std::sin(th_2);
                const double x2 = std::sin(th_5);
                const double x3 = x1*x2*std::cos(th_3)*std::cos(th_4) + x1*std::sin(th_3)*std::cos(th_5) - x2*std::sin(th_4)*std::cos(th_2);
                const double x4 = safe_sqrt(std::pow(r_13, 2) + std::pow(r_23, 2) - std::pow(x3, 2));
                RawSolution solution_0(this_solution);
                solution_0[0] = x0 + std::atan2(x4, x3);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx_0);
                const double tmp_sol_value = x0 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor();
    // Finish code for explicit solution node 9

    // Code for equation all-zero dispatcher node 10
    auto EquationAllZeroDispatcherNode_node_10_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(10);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(10);
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
                add_input_index_to(11, node_input_i_idx_in_queue);
            else
                add_input_index_to(48, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor();
    // Finish code for equation all-zero dispatcher node 10

    // Code for equation all-zero dispatcher node 48
    auto EquationAllZeroDispatcherNode_node_48_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(48);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(48);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const bool checked_result = ((std::fabs(r_31) <= 9.9999999999999995e-7) && (std::fabs(r_32) <= 9.9999999999999995e-7) && (std::fabs(r_33) <= 9.9999999999999995e-7) && (std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0)) <= 9.9999999999999995e-7) && (std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0)) <= 9.9999999999999995e-7) && (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(49, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_48_processor();
    // Finish code for equation all-zero dispatcher node 48

    // Code for explicit solution node 49, solved variable is th_1
    auto ExplicitSolutionNode_node_49_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(49);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(49);
        if (!this_input_valid)
            return;

        // The solution of non-root node 49
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            const bool condition_0 = ((((std::fabs(r_33) >= 9.9999999999999995e-7) || (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= 9.9999999999999995e-7))) && (((((std::fabs(r_31) < 9.9999999999999995e-7) && (std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0)) < 9.9999999999999995e-7))) || (std::max(std::fabs(r_33), std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0))) >= std::max(std::fabs(r_31), std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0)))))) && (((((std::fabs(r_32) < 9.9999999999999995e-7) && (std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0)) < 9.9999999999999995e-7))) || (std::max(std::fabs(r_33), std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0))) >= std::max(std::fabs(r_32), std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0)))))));
            if (condition_0)
            {
                const double x0 = -r_13*std::cos(th_0) - r_23*std::sin(th_0);
                const double x1 = std::cos(th_3);
                const double x2 = std::cos(th_5);
                const double x3 = std::sin(th_3);
                const double x4 = std::sin(th_5);
                const double x5 = x4*std::cos(th_4);
                const double x6 = x1*x2 - x3*x5;
                const double x7 = std::cos(th_2);
                const double x8 = -x1*x5*x7 - x2*x3*x7 - x4*std::sin(th_2)*std::sin(th_4);
                RawSolution solution_0(this_solution);
                solution_0[1] = std::atan2(r_33*x8 - x0*x6, r_33*x6 + x0*x8);
                int appended_idx_0 = append_solution_to_queue(solution_0);
            }
            const bool condition_1 = ((((std::fabs(r_31) >= 9.9999999999999995e-7) || (std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0)) >= 9.9999999999999995e-7))) && (((((std::fabs(r_32) < 9.9999999999999995e-7) && (std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0)) < 9.9999999999999995e-7))) || (std::max(std::fabs(r_31), std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0))) >= std::max(std::fabs(r_32), std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0)))))) && (((((std::fabs(r_33) < 9.9999999999999995e-7) && (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) < 9.9999999999999995e-7))) || (std::max(std::fabs(r_31), std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0))) > std::max(std::fabs(r_33), std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)))))));
            if (condition_1)
            {
                const double x0 = -r_11*std::cos(th_0) - r_21*std::sin(th_0);
                const double x1 = std::cos(th_3);
                const double x2 = std::cos(th_6);
                const double x3 = std::sin(th_5);
                const double x4 = std::sin(th_3);
                const double x5 = std::sin(th_4);
                const double x6 = std::sin(th_6);
                const double x7 = std::cos(th_4);
                const double x8 = x2*std::cos(th_5);
                const double x9 = x7*x8;
                const double x10 = -x1*x2*x3 + x4*x5*x6 - x4*x9;
                const double x11 = std::sin(th_2);
                const double x12 = std::cos(th_2);
                const double x13 = x1*x12*x5*x6 - x1*x12*x9 - x11*x5*x8 - x11*x6*x7 + x12*x2*x3*x4;
                RawSolution solution_1(this_solution);
                solution_1[1] = std::atan2(r_31*x13 - x0*x10, r_31*x10 + x0*x13);
                int appended_idx_1 = append_solution_to_queue(solution_1);
            }
            const bool condition_2 = ((((std::fabs(r_32) >= 9.9999999999999995e-7) || (std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0)) >= 9.9999999999999995e-7))) && (((((std::fabs(r_31) < 9.9999999999999995e-7) && (std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0)) < 9.9999999999999995e-7))) || (std::max(std::fabs(r_32), std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0))) > std::max(std::fabs(r_31), std::fabs(r_11*std::cos(th_0) + r_21*std::sin(th_0)))))) && (((((std::fabs(r_33) < 9.9999999999999995e-7) && (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) < 9.9999999999999995e-7))) || (std::max(std::fabs(r_32), std::fabs(r_12*std::cos(th_0) + r_22*std::sin(th_0))) > std::max(std::fabs(r_33), std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)))))));
            if (condition_2)
            {
                const double x0 = -r_12*std::cos(th_0) - r_22*std::sin(th_0);
                const double x1 = std::sin(th_3);
                const double x2 = std::cos(th_6);
                const double x3 = std::sin(th_4);
                const double x4 = x2*x3;
                const double x5 = std::cos(th_3);
                const double x6 = std::sin(th_6);
                const double x7 = x6*std::sin(th_5);
                const double x8 = std::cos(th_4);
                const double x9 = x6*std::cos(th_5);
                const double x10 = x8*x9;
                const double x11 = x1*x10 + x1*x4 + x5*x7;
                const double x12 = std::sin(th_2);
                const double x13 = std::cos(th_2);
                const double x14 = x13*x5;
                const double x15 = -x1*x13*x7 + x10*x14 - x12*x2*x8 + x12*x3*x9 + x14*x4;
                const double tmp_sol_value = std::atan2(r_32*x15 - x0*x11, r_32*x11 + x0*x15);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_1_processor();
    // Finish code for explicit solution node 49

    // Code for explicit solution node 11, solved variable is th_1
    auto ExplicitSolutionNode_node_11_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(11);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(11);
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
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool condition_0 = ((std::fabs(Pz) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x1 = std::sin(th_3);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_5);
                const double x4 = a_5*std::cos(th_5);
                const double x5 = x4*std::cos(th_4);
                const double x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5;
                const double x7 = std::cos(th_2);
                const double x8 = a_3*x7;
                const double x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*std::sin(th_2)*std::sin(th_4) - x8;
                const double tmp_sol_value = std::atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor();
    // Finish code for explicit solution node 11

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
        const double value_at_3 = th_3;  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[3];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[4];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[5];  // th_6
        new_ik_i[6] = value_at_6;
        if (std::all_of(new_ik_i.begin(), new_ik_i.end(), [](double value) { return std::isfinite(value); }))
            ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, double th_3, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
    computeRawIK(T_ee_raw, th_3, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, double th_3, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, addRevoluteOffset(th_3, th_3_offset_original2raw), workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = th_3;
        ik_out_i[4] = ik_out_i[4] - th_4_offset_original2raw;
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
        ik_out_i[6] = ik_out_i[6] - th_6_offset_original2raw;
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

static void computeIKForRawTarget(const Eigen::Matrix4d& T_ee, const Eigen::Matrix4d& T_ee_raw, double th_3, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output, Eigen::Matrix4d& frame_disturbance, bool& frame_disturbance_ready, bool allow_disturbance)
{
    if (!T_ee.allFinite()) { ik_output.clear(); return; }
    if (!std::isfinite(th_3)) { ik_output.clear(); return; }
    if (!T_ee_raw.allFinite()) { ik_output.clear(); return; }
    computeRawIK(T_ee_raw, addRevoluteOffset(th_3, th_3_offset_original2raw), workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = th_3;
        ik_out_i[4] = ik_out_i[4] - th_4_offset_original2raw;
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
        ik_out_i[6] = ik_out_i[6] - th_6_offset_original2raw;
        wrapAngleToPi(ik_out_i);
        const Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
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
            if (frame_disturbance_ready)
                T_ee_raw_disturbed = frame_disturbance * T_ee_raw;
            else
            {
                T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw, 1e-2, 1e-2, &frame_disturbance);
                frame_disturbance_ready = true;
            }
        }
        Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
        computeRawIK(T_ee_raw_disturbed, addRevoluteOffset(th_3, th_3_offset_original2raw), workspace);
        const auto& raw_ik_out_disturb = workspace.raw_ik_out;
        for(int i = 0; i < raw_ik_out_disturb.size(); i++)
        {
            auto ik_out_i = raw_ik_out_disturb[i];
            ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
            ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
            ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
            ik_out_i[3] = th_3;
            ik_out_i[4] = ik_out_i[4] - th_4_offset_original2raw;
            ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
            ik_out_i[6] = ik_out_i[6] - th_6_offset_original2raw;
            wrapAngleToPi(ik_out_i);
            Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
            double ee_pose_diff = (ee_pose_i - T_ee_disturbed).squaredNorm();
            if (!std::isfinite(ee_pose_diff) || ee_pose_diff > pose_tolerance_degenerate)
                continue;

            // Try numerical refinement
            yaik_cpp::numericalRefinement<robot_nq>(computeFK, computeTwistJacobian, T_ee, ik_out_i, 1e-4, std::array<bool, robot_nq>{{false, false, false, true, false, false, false}});
            wrapAngleToPi(ik_out_i);
            ee_pose_i = computeFK(ik_out_i);
            ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
            if (ee_pose_diff < pose_tolerance_degenerate)
                ik_output.push_back(ik_out_i);
        }
    }

    removeDuplicateWithinClasses<robot_nq>(ik_output, zero_tolerance, ikCandidateClasses<robot_nq>(ik_output, jointLower(), jointUpper(), periodicLimitJoints(), [&](const std::array<double, robot_nq>& candidate) { return computeFK(candidate); }, T_ee));
}

static void computeIK(const Eigen::Matrix4d& T_ee, double th_3, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    ik_output.clear();
    if (!T_ee.allFinite()) return;
    if (!std::isfinite(th_3)) return;
    Eigen::Matrix4d frame_disturbance;
    bool frame_disturbance_ready = false;
    for (bool legacy : {true, false})
    {
        const Eigen::Matrix4d raw_target = legacy ? endEffectorTargetOriginalToRawLegacyFrame(T_ee) : endEffectorTargetOriginalToRaw(T_ee);
        std::vector<std::array<double, robot_nq>> candidates;
        computeIKForRawTarget(T_ee, raw_target, th_3, workspace, candidates, frame_disturbance, frame_disturbance_ready, legacy || ik_output.empty());
        for (const auto& candidate : candidates)
        {
            if (std::find(ik_output.begin(), ik_output.end(), candidate) == ik_output.end())
                ik_output.push_back(candidate);
        }
    }
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_3)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_3, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct franka_panda_ik

// Code below for debug
void test_ik_solve_franka_panda()
{
    std::array<double, franka_panda_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = franka_panda_ik::computeFK(theta);
    auto ik_output = franka_panda_ik::computeIK(ee_pose, theta[3]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = franka_panda_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_franka_panda();
    return 0;
}
