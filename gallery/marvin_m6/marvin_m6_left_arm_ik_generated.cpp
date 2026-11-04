#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct marvin_m6_left_arm_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 12;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_1 = 0.018;
static constexpr double a_2 = 0.018;
static constexpr double d_0 = 0.287;
static constexpr double d_3 = 0.314;
static constexpr double post_transform_special_symbol_13 = -0.095;
static constexpr double pre_transform_special_symbol_03 = 0.2215;
static constexpr double pre_transform_special_symbol_13 = 2.9142e-05;
static constexpr double pre_transform_special_symbol_23 = 1.121;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -3.141592653589793;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 0.0;
static constexpr double th_5_offset_original2raw = 1.5707963267948966;
static constexpr double th_6_offset_original2raw = -1.5707963267948966;

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
    ee_transformed(0, 0) = -r_22;
    ee_transformed(0, 1) = -r_23;
    ee_transformed(0, 2) = r_21;
    ee_transformed(0, 3) = Py + post_transform_special_symbol_13*r_23 - pre_transform_special_symbol_13;
    ee_transformed(1, 0) = -r_32;
    ee_transformed(1, 1) = -r_33;
    ee_transformed(1, 2) = r_31;
    ee_transformed(1, 3) = Pz + post_transform_special_symbol_13*r_33 - pre_transform_special_symbol_23;
    ee_transformed(2, 0) = -r_12;
    ee_transformed(2, 1) = -r_13;
    ee_transformed(2, 2) = r_11;
    ee_transformed(2, 3) = Px + post_transform_special_symbol_13*r_13 - pre_transform_special_symbol_03;
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
    ee_transformed(0, 0) = r_33;
    ee_transformed(0, 1) = -r_31;
    ee_transformed(0, 2) = -r_32;
    ee_transformed(0, 3) = Pz + post_transform_special_symbol_13*r_32 + pre_transform_special_symbol_03;
    ee_transformed(1, 0) = r_13;
    ee_transformed(1, 1) = -r_11;
    ee_transformed(1, 2) = -r_12;
    ee_transformed(1, 3) = Px + post_transform_special_symbol_13*r_12 + pre_transform_special_symbol_13;
    ee_transformed(2, 0) = r_23;
    ee_transformed(2, 1) = -r_21;
    ee_transformed(2, 2) = -r_22;
    ee_transformed(2, 3) = Py + post_transform_special_symbol_13*r_22 + pre_transform_special_symbol_23;
    return ee_transformed;
}

///************* Below are the actual FK and IK implementations *******************
static Eigen::Matrix4d computeFK(const std::array<double, robot_nq>& theta_input_original)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
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
    ee_pose_raw(0, 0) = -x0*x18 + x19*x24;
    ee_pose_raw(0, 1) = -x0*x24 - x18*x19;
    ee_pose_raw(0, 2) = -x20*x23 - x21*x22;
    ee_pose_raw(0, 3) = a_1*x16 + a_2*x17 - d_0*x12 + d_3*x21;
    ee_pose_raw(1, 0) = -x0*x29 + x19*x32;
    ee_pose_raw(1, 1) = -x0*x32 - x19*x29;
    ee_pose_raw(1, 2) = -x20*x31 - x22*x30;
    ee_pose_raw(1, 3) = a_1*x27 + a_2*x28 - d_0*x26 + d_3*x30;
    ee_pose_raw(2, 0) = -x0*x36 + x19*x40;
    ee_pose_raw(2, 1) = -x0*x40 - x19*x36;
    ee_pose_raw(2, 2) = -x20*x39 - x22*x38;
    ee_pose_raw(2, 3) = -a_1*x34 + a_2*x35 - d_0*x7 - d_3*x37;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::cos(th_1);
    const double x1 = std::sin(th_1);
    const double x2 = std::sin(th_2);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_3);
    const double x5 = std::sin(th_3);
    const double x6 = std::cos(th_2);
    const double x7 = x1*x6;
    const double x8 = -x0*x4 + x5*x7;
    const double x9 = -x8;
    const double x10 = std::cos(th_4);
    const double x11 = std::sin(th_4);
    const double x12 = -x0*x5 - x4*x7;
    const double x13 = -x10*x3 - x11*x12;
    const double x14 = std::cos(th_5);
    const double x15 = std::sin(th_5);
    const double x16 = -x14*x9 - x15*(x10*x12 - x11*x3);
    const double x17 = std::sin(th_0);
    const double x18 = std::cos(th_0);
    const double x19 = x1*x18;
    const double x20 = x17*x6;
    const double x21 = x0*x18*x2 - x20;
    const double x22 = x17*x2;
    const double x23 = x18*x6;
    const double x24 = x0*x23 + x22;
    const double x25 = x19*x4 + x24*x5;
    const double x26 = -x19*x5 + x24*x4;
    const double x27 = x10*x21 - x11*x26;
    const double x28 = -x14*x25 - x15*(x10*x26 + x11*x21);
    const double x29 = x1*x17;
    const double x30 = x0*x22 + x23;
    const double x31 = x0*x20 - x18*x2;
    const double x32 = x29*x4 + x31*x5;
    const double x33 = -x29*x5 + x31*x4;
    const double x34 = x10*x30 - x11*x33;
    const double x35 = -x14*x32 - x15*(x10*x33 + x11*x30);
    const double x36 = d_0*x29;
    const double x37 = pre_transform_special_symbol_23 - x36;
    const double x38 = -pre_transform_special_symbol_13;
    const double x39 = d_0*x19;
    const double x40 = -x38 - x39;
    const double x41 = a_1*x24 + pre_transform_special_symbol_13 - x39;
    const double x42 = a_1*x31 + pre_transform_special_symbol_23 - x36;
    const double x43 = a_2*x26 + d_3*x25 + x41;
    const double x44 = a_2*x33 + d_3*x32 + x42;
    const double x45 = d_0*x0 - pre_transform_special_symbol_03;
    const double x46 = -x45;
    const double x47 = a_1*x7 + x45;
    const double x48 = -x47;
    const double x49 = a_2*x12 - d_3*x8 - x47;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = 1;
    jacobian(0, 2) = -x0;
    jacobian(0, 3) = -x3;
    jacobian(0, 4) = x9;
    jacobian(0, 5) = x13;
    jacobian(0, 6) = x16;
    jacobian(1, 1) = -x17;
    jacobian(1, 2) = -x19;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x25;
    jacobian(1, 5) = x27;
    jacobian(1, 6) = x28;
    jacobian(2, 1) = x18;
    jacobian(2, 2) = -x29;
    jacobian(2, 3) = x30;
    jacobian(2, 4) = x32;
    jacobian(2, 5) = x34;
    jacobian(2, 6) = x35;
    jacobian(3, 1) = pre_transform_special_symbol_13*x18 + pre_transform_special_symbol_23*x17;
    jacobian(3, 2) = x19*x37 - x29*x40;
    jacobian(3, 3) = -x21*x42 + x30*x41;
    jacobian(3, 4) = -x25*x44 + x32*x43;
    jacobian(3, 5) = -x27*x44 + x34*x43;
    jacobian(3, 6) = -x28*x44 + x35*x43;
    jacobian(4, 0) = pre_transform_special_symbol_23;
    jacobian(4, 1) = -pre_transform_special_symbol_03*x18;
    jacobian(4, 2) = -x0*x37 + x29*x46;
    jacobian(4, 3) = -x3*x42 - x30*x48;
    jacobian(4, 4) = -x32*x49 + x44*x9;
    jacobian(4, 5) = x13*x44 - x34*x49;
    jacobian(4, 6) = x16*x44 - x35*x49;
    jacobian(5, 0) = x38;
    jacobian(5, 1) = -pre_transform_special_symbol_03*x17;
    jacobian(5, 2) = x0*x40 - x19*x46;
    jacobian(5, 3) = x21*x48 + x3*x41;
    jacobian(5, 4) = x25*x49 - x43*x9;
    jacobian(5, 5) = -x13*x43 + x27*x49;
    jacobian(5, 6) = -x16*x43 + x28*x49;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::cos(th_1);
    const double x1 = std::sin(th_1);
    const double x2 = std::sin(th_2);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_3);
    const double x5 = std::sin(th_3);
    const double x6 = std::cos(th_2);
    const double x7 = x1*x6;
    const double x8 = x0*x4 - x5*x7;
    const double x9 = std::cos(th_4);
    const double x10 = std::sin(th_4);
    const double x11 = -x0*x5 - x4*x7;
    const double x12 = std::cos(th_5);
    const double x13 = std::sin(th_5);
    const double x14 = std::sin(th_0);
    const double x15 = std::cos(th_0);
    const double x16 = x1*x15;
    const double x17 = x14*x6;
    const double x18 = x0*x15*x2 - x17;
    const double x19 = x14*x2;
    const double x20 = x15*x6;
    const double x21 = x0*x20 + x19;
    const double x22 = x16*x4 + x21*x5;
    const double x23 = -x16*x5 + x21*x4;
    const double x24 = x1*x14;
    const double x25 = x0*x19 + x20;
    const double x26 = x0*x17 - x15*x2;
    const double x27 = x24*x4 + x26*x5;
    const double x28 = -x24*x5 + x26*x4;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = 1;
    jacobian(0, 2) = -x0;
    jacobian(0, 3) = -x3;
    jacobian(0, 4) = x8;
    jacobian(0, 5) = -x10*x11 - x3*x9;
    jacobian(0, 6) = -x12*x8 - x13*(-x10*x3 + x11*x9);
    jacobian(1, 1) = -x14;
    jacobian(1, 2) = -x16;
    jacobian(1, 3) = x18;
    jacobian(1, 4) = x22;
    jacobian(1, 5) = -x10*x23 + x18*x9;
    jacobian(1, 6) = -x12*x22 - x13*(x10*x18 + x23*x9);
    jacobian(2, 1) = x15;
    jacobian(2, 2) = -x24;
    jacobian(2, 3) = x25;
    jacobian(2, 4) = x27;
    jacobian(2, 5) = -x10*x28 + x25*x9;
    jacobian(2, 6) = -x12*x27 - x13*(x10*x25 + x28*x9);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];

    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = x0*x2;
    const double x5 = -pre_transform_special_symbol_23;
    const double x6 = d_0*x3;
    const double x7 = -x5 - x6;
    const double x8 = -pre_transform_special_symbol_13;
    const double x9 = d_0*x4;
    const double x10 = -x8 - x9;
    const double x11 = std::cos(th_2);
    const double x12 = x0*x11;
    const double x13 = std::cos(th_1);
    const double x14 = std::sin(th_2);
    const double x15 = x1*x14;
    const double x16 = x12 + x13*x15;
    const double x17 = x1*x11;
    const double x18 = x0*x13*x14 - x17;
    const double x19 = x12*x13 + x15;
    const double x20 = a_1*x19 + pre_transform_special_symbol_13 - x9;
    const double x21 = -x0*x14 + x13*x17;
    const double x22 = a_1*x21 + pre_transform_special_symbol_23 - x6;
    const double x23 = std::cos(th_3);
    const double x24 = std::sin(th_3);
    const double x25 = x21*x24 + x23*x3;
    const double x26 = x19*x24 + x23*x4;
    const double x27 = x19*x23 - x24*x4;
    const double x28 = a_2*x27 + d_3*x26 + x20;
    const double x29 = x21*x23 - x24*x3;
    const double x30 = a_2*x29 + d_3*x25 + x22;
    const double x31 = std::cos(th_4);
    const double x32 = std::sin(th_4);
    const double x33 = x16*x31 - x29*x32;
    const double x34 = x18*x31 - x27*x32;
    const double x35 = std::cos(th_5);
    const double x36 = std::sin(th_5);
    const double x37 = -x25*x35 - x36*(x16*x32 + x29*x31);
    const double x38 = -x26*x35 - x36*(x18*x32 + x27*x31);
    const double x39 = p_on_ee_x*x0;
    const double x40 = p_on_ee_x*x1;
    const double x41 = d_0*x13 - pre_transform_special_symbol_03;
    const double x42 = -x41;
    const double x43 = x14*x2;
    const double x44 = x11*x2;
    const double x45 = a_1*x44 + x41;
    const double x46 = -x45;
    const double x47 = -x13*x23 + x24*x44;
    const double x48 = -x47;
    const double x49 = -x13*x24 - x23*x44;
    const double x50 = a_2*x49 - d_3*x47 - x45;
    const double x51 = -x31*x43 - x32*x49;
    const double x52 = -x35*x48 - x36*(x31*x49 - x32*x43);
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -p_on_ee_y*x0 - p_on_ee_z*x1 + pre_transform_special_symbol_13*x0 + pre_transform_special_symbol_23*x1;
    jacobian(0, 2) = p_on_ee_y*x3 - p_on_ee_z*x4 - x10*x3 + x4*x7;
    jacobian(0, 3) = -p_on_ee_y*x16 + p_on_ee_z*x18 + x16*x20 - x18*x22;
    jacobian(0, 4) = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x28 - x26*x30;
    jacobian(0, 5) = -p_on_ee_y*x33 + p_on_ee_z*x34 + x28*x33 - x30*x34;
    jacobian(0, 6) = -p_on_ee_y*x37 + p_on_ee_z*x38 + x28*x37 - x30*x38;
    jacobian(1, 0) = -p_on_ee_z - x5;
    jacobian(1, 1) = -pre_transform_special_symbol_03*x0 + x39;
    jacobian(1, 2) = p_on_ee_z*x13 + x1*x2*x42 - x13*x7 - x2*x40;
    jacobian(1, 3) = p_on_ee_x*x16 + p_on_ee_z*x43 - x16*x46 - x22*x43;
    jacobian(1, 4) = p_on_ee_x*x25 - p_on_ee_z*x48 - x25*x50 + x30*x48;
    jacobian(1, 5) = p_on_ee_x*x33 - p_on_ee_z*x51 + x30*x51 - x33*x50;
    jacobian(1, 6) = p_on_ee_x*x37 - p_on_ee_z*x52 + x30*x52 - x37*x50;
    jacobian(2, 0) = p_on_ee_y + x8;
    jacobian(2, 1) = -pre_transform_special_symbol_03*x1 + x40;
    jacobian(2, 2) = -p_on_ee_y*x13 + x10*x13 + x2*x39 - x4*x42;
    jacobian(2, 3) = -p_on_ee_x*x18 - p_on_ee_y*x43 + x14*x2*x20 + x18*x46;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x48 + x26*x50 - x28*x48;
    jacobian(2, 5) = -p_on_ee_x*x34 + p_on_ee_y*x51 - x28*x51 + x34*x50;
    jacobian(2, 6) = -p_on_ee_x*x38 + p_on_ee_y*x52 - x28*x52 + x38*x50;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, double th_4, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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
    auto make_raw_solution = []() -> RawSolution { return {}; };

    auto append_solution_to_queue = [&solution_queue](RawSolution solution_2_add) -> int {
        return solution_queue.push_solution(solution_2_add);
    };

    auto add_input_index_to = [&node_index_workspace](int node_idx, int solution_idx) -> void {
        if (solution_idx < 0) return;
        node_index_workspace.append_index_to_node(node_idx, solution_idx);
    };

    // Code for non-branch dispatcher node 0
    // Actually, there is no code

    // Code for explicit solution node 1, solved variable is th_3
    auto ExplicitSolutionNode_node_1_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;

        // The explicit solution of root node
        const bool condition_0 = std::fabs(2*a_1*a_2 - 2*d_0*d_3) >= zero_tolerance || std::fabs(2*a_1*d_3 + 2*a_2*d_0) >= zero_tolerance || std::fabs(std::pow(a_1, 2) + std::pow(a_2, 2) + std::pow(d_0, 2) + std::pow(d_3, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2)) >= zero_tolerance;
        if (condition_0)
        {
            const double x0 = 2*d_3;
            const double x1 = 2*a_2;
            const double x2 = a_1*x0 + d_0*x1;
            const double x3 = a_1*x1 - d_0*x0;
            const double x4 = std::atan2(x2, x3);
            const double x5 = -std::pow(a_1, 2) - std::pow(a_2, 2) - std::pow(d_0, 2) - std::pow(d_3, 2) + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2);
            const double x6 = safe_sqrt(std::pow(x2, 2) + std::pow(x3, 2) - std::pow(x5, 2));
            auto solution_0 = make_raw_solution();
            solution_0[3] = x4 + std::atan2(x6, x5);
            int appended_idx_0 = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx_0);

            auto solution_1 = make_raw_solution();
            solution_1[3] = x4 + std::atan2(-x6, x5);
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx_1);

        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor();
    // Finish code for explicit solution node 0

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
            const double th_3 = this_solution[3];
            const bool checked_result = std::fabs(inv_Px) <= 9.9999999999999995e-7 && std::fabs(inv_Py) <= 9.9999999999999995e-7 && std::fabs((a_1*std::cos(th_3) + a_2 + d_0*std::sin(th_3))*std::sin(th_4)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2

    // Code for explicit solution node 3, solved variable is th_6
    auto ExplicitSolutionNode_node_3_solve_th_6_processor = [&]() -> void
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
            const double th_3 = this_solution[3];
            const bool condition_0 = std::fabs(inv_Px) >= zero_tolerance || std::fabs(inv_Py) >= zero_tolerance || std::fabs((a_1*std::cos(th_3) + a_2 + d_0*std::sin(th_3))*std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::atan2(inv_Px, inv_Py);
                const double x1 = std::sin(th_4);
                const double x2 = a_1*std::cos(th_3) + a_2 + d_0*std::sin(th_3);
                const double x3 = safe_sqrt(std::pow(inv_Px, 2) + std::pow(inv_Py, 2) - std::pow(x1, 2)*std::pow(x2, 2));
                const double x4 = -x1*x2;
                RawSolution solution_0(this_solution);
                solution_0[5] = x0 + std::atan2(x3, x4);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = x0 + std::atan2(-x3, x4);
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
    ExplicitSolutionNode_node_3_solve_th_6_processor();
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
            const bool checked_result = std::fabs(inv_Pz) <= 9.9999999999999995e-7 && std::fabs(inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4

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
            const double th_3 = this_solution[3];
            const double th_6 = this_solution[5];
            const bool condition_0 = std::fabs(inv_Pz) >= 9.9999999999999995e-7 || std::fabs(inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_3);
                const double x2 = (a_1*x0 + a_2 + d_0*x1)*std::cos(th_4);
                const double x3 = -inv_Px*std::cos(th_6) + inv_Py*std::sin(th_6);
                const double x4 = a_1*x1 - d_0*x0 + d_3;
                const double tmp_sol_value = std::atan2(inv_Pz*x2 - x3*x4, inv_Pz*x4 + x2*x3);
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

        // The solution of non-root node 7
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            // Temp variable for efficiency
            const double x0 = std::sin(th_6);
            const double x1 = std::sin(th_4);
            const double x2 = std::cos(th_3);
            const double x3 = x1*x2;
            const double x4 = std::sin(th_3);
            const double x5 = std::sin(th_5);
            const double x6 = std::cos(th_6);
            const double x7 = std::cos(th_4);
            const double x8 = x6*x7;
            const double x9 = std::cos(th_5);
            const double x10 = x0*x3 + x2*x8*x9 - x4*x5*x6;
            const double x11 = x1*x9;
            const double x12 = -x0*x7 + x11*x6;
            const double x13 = -x0*x2*x7*x9 + x0*x4*x5 + x3*x6;
            const double x14 = x0*x11 + x8;
            const double x15 = x2*x5*x7 + x4*x9;
            const double x16 = x1*x5;
            const double x17 = a_1 + a_2*x2 + d_3*x4;
            // End of temp variables
            Eigen::Matrix<double, 8, 4> A_matrix;
            A_matrix.setZero();
            A_matrix(0, 0) = r_11;
            A_matrix(0, 1) = -r_21;
            A_matrix(0, 2) = -x10;
            A_matrix(0, 3) = x12;
            A_matrix(1, 0) = r_12;
            A_matrix(1, 1) = -r_22;
            A_matrix(1, 2) = -x13;
            A_matrix(1, 3) = -x14;
            A_matrix(2, 0) = r_13;
            A_matrix(2, 1) = -r_23;
            A_matrix(2, 2) = x15;
            A_matrix(2, 3) = -x16;
            A_matrix(3, 0) = Px;
            A_matrix(3, 1) = -Py;
            A_matrix(3, 2) = -x17;
            A_matrix(4, 0) = -r_11;
            A_matrix(4, 1) = r_21;
            A_matrix(4, 2) = x10;
            A_matrix(4, 3) = -x12;
            A_matrix(5, 0) = -r_12;
            A_matrix(5, 1) = r_22;
            A_matrix(5, 2) = x13;
            A_matrix(5, 3) = x14;
            A_matrix(6, 0) = -r_13;
            A_matrix(6, 1) = r_23;
            A_matrix(6, 2) = -x15;
            A_matrix(6, 3) = x16;
            A_matrix(7, 0) = -Px;
            A_matrix(7, 1) = Py;
            A_matrix(7, 2) = x17;
            // Code for solving
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 2, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 3, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 6, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 7, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 2, 3, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
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
            const double th_3 = this_solution[3];
            const bool checked_result = std::fabs(Px) <= 9.9999999999999995e-7 && std::fabs(Py) <= 9.9999999999999995e-7 && std::fabs((a_1 + a_2*std::cos(th_3) + d_3*std::sin(th_3))*std::sin(th_2)) <= 9.9999999999999995e-7;
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
            const double th_3 = this_solution[3];
            const bool condition_0 = std::fabs(Px) >= zero_tolerance || std::fabs(Py) >= zero_tolerance || std::fabs((a_1 + a_2*std::cos(th_3) + d_3*std::sin(th_3))*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::atan2(Px, -Py);
                const double x1 = std::sin(th_2);
                const double x2 = a_1 + a_2*std::cos(th_3) + d_3*std::sin(th_3);
                const double x3 = safe_sqrt(std::pow(Px, 2) + std::pow(Py, 2) - std::pow(x1, 2)*std::pow(x2, 2));
                const double x4 = x1*x2;
                RawSolution solution_0(this_solution);
                solution_0[0] = x0 + std::atan2(x3, x4);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx_0);
                const double tmp_sol_value = x0 + std::atan2(-x3, x4);
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
            const bool checked_result = std::fabs(Pz) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor();
    // Finish code for equation all-zero dispatcher node 10

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
            const double th_3 = this_solution[3];
            const bool condition_0 = std::fabs(Pz) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_3);
                const double x2 = (a_1 + a_2*x0 + d_3*x1)*std::cos(th_2);
                const double x3 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x4 = -a_2*x1 - d_0 + d_3*x0;
                const double tmp_sol_value = std::atan2(-Pz*x2 - x3*x4, Pz*x4 - x2*x3);
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
        const double value_at_3 = raw_ik_out_i[3];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = th_4;  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[4];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[5];  // th_6
        new_ik_i[6] = value_at_6;
        ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, double th_4, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
    computeRawIK(T_ee_raw, th_4, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, double th_4, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_4 + th_4_offset_original2raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] -= th_0_offset_original2raw;
        ik_out_i[1] -= th_1_offset_original2raw;
        ik_out_i[2] -= th_2_offset_original2raw;
        ik_out_i[3] -= th_3_offset_original2raw;
        ik_out_i[4] -= th_4_offset_original2raw;
        ik_out_i[5] -= th_5_offset_original2raw;
        ik_out_i[6] -= th_6_offset_original2raw;
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

static void computeIK(const Eigen::Matrix4d& T_ee, double th_4, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    if (!T_ee.allFinite()) { ik_output.clear(); return; }
    if (!std::isfinite(th_4)) { ik_output.clear(); return; }
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_4 + th_4_offset_original2raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] -= th_0_offset_original2raw;
        ik_out_i[1] -= th_1_offset_original2raw;
        ik_out_i[2] -= th_2_offset_original2raw;
        ik_out_i[3] -= th_3_offset_original2raw;
        ik_out_i[4] -= th_4_offset_original2raw;
        ik_out_i[5] -= th_5_offset_original2raw;
        ik_out_i[6] -= th_6_offset_original2raw;
        wrapAngleToPi(ik_out_i);
        const Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
        if (ee_pose_diff < pose_tolerance)
            ik_output.push_back(ik_out_i);
    }

    if (!ik_output.empty())
    {
        removeDuplicate<robot_nq>(ik_output, zero_tolerance);
        return;
    }

    // Disturbing method for degenerate handling
    Eigen::Matrix4d T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);
    Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
    computeRawIK(T_ee_raw_disturbed, th_4 + th_4_offset_original2raw, workspace);
    const auto& raw_ik_out_disturb = workspace.raw_ik_out;
    for(int i = 0; i < raw_ik_out_disturb.size(); i++)
    {
        auto ik_out_i = raw_ik_out_disturb[i];
        ik_out_i[0] -= th_0_offset_original2raw;
        ik_out_i[1] -= th_1_offset_original2raw;
        ik_out_i[2] -= th_2_offset_original2raw;
        ik_out_i[3] -= th_3_offset_original2raw;
        ik_out_i[4] -= th_4_offset_original2raw;
        ik_out_i[5] -= th_5_offset_original2raw;
        ik_out_i[6] -= th_6_offset_original2raw;
        wrapAngleToPi(ik_out_i);
        Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee_disturbed).squaredNorm();
        if (!std::isfinite(ee_pose_diff) || ee_pose_diff > pose_tolerance_degenerate)
            continue;

        // Try numerical refinement
        yaik_cpp::numericalRefinement<robot_nq>(computeFK, computeTwistJacobian, T_ee, ik_out_i, 1e-4, std::array<bool, robot_nq>{{false, false, false, false, true, false, false}});
        wrapAngleToPi(ik_out_i);
        ee_pose_i = computeFK(ik_out_i);
        ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
        if (ee_pose_diff < pose_tolerance_degenerate)
            ik_output.push_back(ik_out_i);
    }

    removeDuplicate<robot_nq>(ik_output, zero_tolerance);
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_4)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_4, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct marvin_m6_left_arm_ik

// Code below for debug
void test_ik_solve_marvin_m6_left_arm()
{
    std::array<double, marvin_m6_left_arm_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = marvin_m6_left_arm_ik::computeFK(theta);
    auto ik_output = marvin_m6_left_arm_ik::computeIK(ee_pose, theta[4]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = marvin_m6_left_arm_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_marvin_m6_left_arm();
}
