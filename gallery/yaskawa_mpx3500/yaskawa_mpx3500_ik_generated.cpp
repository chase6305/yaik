#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct yaskawa_mpx3500_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 192;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -1.5707963267948966, -3.141592653589793, -1.5707963267948966, -6.283185307179586, -1.5707963267948966}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 4.71238898038469, 3.141592653589793, 4.71238898038469, 0.0, 4.71238898038469}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 14;
static constexpr int intermediate_solution_size = 7;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_0 = 1.3;
static constexpr double alpha_3 = -1.0470795620393143;
static constexpr double alpha_5 = -1.0470795620393143;
static constexpr double d_1 = -1.4;
static constexpr double d_2 = -0.11397670814688408;
static constexpr double d_4 = -0.12300000000000003;
static constexpr double pre_transform_special_symbol_23 = 0.8;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = -0.0;
static constexpr double th_3_offset_original2raw = -1.5707963267948966;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = -1.5707963267948966;

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
    const double x1 = std::cos(alpha_3);
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x6*x7;
    const double x9 = x2*x5 - x2*x8;
    const double x10 = std::sin(alpha_3);
    const double x11 = std::sin(th_0);
    const double x12 = std::cos(th_3);
    const double x13 = std::sin(th_3);
    const double x14 = x3*x6;
    const double x15 = x4*x7;
    const double x16 = x14*x2 + x15*x2;
    const double x17 = -x11*x12 - x13*x16;
    const double x18 = x10*x17;
    const double x19 = x1*x9 - x18;
    const double x20 = std::sin(alpha_5);
    const double x21 = x19*x20;
    const double x22 = std::cos(th_5);
    const double x23 = std::sin(th_4);
    const double x24 = x10*x9;
    const double x25 = std::cos(th_4);
    const double x26 = -x11*x13 + x12*x16;
    const double x27 = x1*x17;
    const double x28 = x23*x24 + x23*x27 + x25*x26;
    const double x29 = std::cos(alpha_5);
    const double x30 = -x23*x26 + x24*x25 + x25*x27;
    const double x31 = x29*x30;
    const double x32 = x19*x29;
    const double x33 = x20*x30;
    const double x34 = a_0*x7;
    const double x35 = x11*x5 - x11*x8;
    const double x36 = x11*x14 + x11*x15;
    const double x37 = x12*x2 - x13*x36;
    const double x38 = x10*x37;
    const double x39 = x1*x35 - x38;
    const double x40 = x20*x39;
    const double x41 = x10*x35;
    const double x42 = x12*x36 + x13*x2;
    const double x43 = x1*x37;
    const double x44 = x23*x41 + x23*x43 + x25*x42;
    const double x45 = -x23*x42 + x25*x41 + x25*x43;
    const double x46 = x29*x45;
    const double x47 = x29*x39;
    const double x48 = x20*x45;
    const double x49 = x14 + x15;
    const double x50 = x1*x49;
    const double x51 = -x5 + x6*x7;
    const double x52 = x13*x51;
    const double x53 = x10*x52;
    const double x54 = x50 + x53;
    const double x55 = x20*x54;
    const double x56 = x12*x51;
    const double x57 = x1*x52;
    const double x58 = x10*x23*x49 - x23*x57 + x25*x56;
    const double x59 = x10*x25*x49 - x23*x56 - x25*x57;
    const double x60 = x29*x59;
    const double x61 = x29*x54;
    const double x62 = x20*x59;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x21 + x0*x31 + x22*x28;
    ee_pose_raw(0, 1) = -x0*x28 + x21*x22 + x22*x31;
    ee_pose_raw(0, 2) = x32 - x33;
    ee_pose_raw(0, 3) = d_1*x9 + d_2*x1*x9 - d_2*x18 + d_4*x32 - d_4*x33 + x2*x34;
    ee_pose_raw(1, 0) = x0*x40 + x0*x46 + x22*x44;
    ee_pose_raw(1, 1) = -x0*x44 + x22*x40 + x22*x46;
    ee_pose_raw(1, 2) = x47 - x48;
    ee_pose_raw(1, 3) = d_1*x35 + d_2*x1*x35 - d_2*x38 + d_4*x47 - d_4*x48 + x11*x34;
    ee_pose_raw(2, 0) = x0*x55 + x0*x60 + x22*x58;
    ee_pose_raw(2, 1) = -x0*x58 + x22*x55 + x22*x60;
    ee_pose_raw(2, 2) = x61 - x62;
    ee_pose_raw(2, 3) = -a_0*x3 + d_1*x49 + d_2*x50 + d_2*x53 + d_4*x61 - d_4*x62;
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
    const double x11 = std::cos(alpha_3);
    const double x12 = std::sin(alpha_3);
    const double x13 = std::cos(th_3);
    const double x14 = std::sin(th_3);
    const double x15 = x2*x9 + x6*x7;
    const double x16 = -x1*x13 - x14*x15;
    const double x17 = x12*x16;
    const double x18 = x10*x11 - x17;
    const double x19 = std::cos(alpha_5);
    const double x20 = x18*x19;
    const double x21 = std::sin(alpha_5);
    const double x22 = std::cos(th_4);
    const double x23 = x12*x22;
    const double x24 = std::sin(th_4);
    const double x25 = x11*x22;
    const double x26 = x21*(x10*x23 + x16*x25 - x24*(-x1*x14 + x13*x15));
    const double x27 = x20 - x26;
    const double x28 = x1*x3;
    const double x29 = x1*x8;
    const double x30 = x2*x28 - x29*x7;
    const double x31 = x2*x29 + x28*x7;
    const double x32 = 1.0*x13*x4 - x14*x31;
    const double x33 = x12*x32;
    const double x34 = x11*x30 - x33;
    const double x35 = x19*x34;
    const double x36 = x21*(x23*x30 - x24*(x13*x31 + x14*x5) + x25*x32);
    const double x37 = x35 - x36;
    const double x38 = 1.0*x2;
    const double x39 = 1.0*x7;
    const double x40 = x3*x39 + x38*x8;
    const double x41 = x11*x40;
    const double x42 = -x3*x38 + 1.0*x7*x8;
    const double x43 = x14*x42;
    const double x44 = x12*x43;
    const double x45 = x41 + x44;
    const double x46 = x19*x45;
    const double x47 = x21*(x12*x22*x40 - x13*x24*x42 - x25*x43);
    const double x48 = x46 - x47;
    const double x49 = a_0*x38;
    const double x50 = pre_transform_special_symbol_23 - x49;
    const double x51 = a_0*x7;
    const double x52 = d_1*x30 + x1*x51;
    const double x53 = d_1*x40 + pre_transform_special_symbol_23 - x49;
    const double x54 = d_2*x41 + d_2*x44 + x53;
    const double x55 = d_2*x11*x30 - d_2*x33 + x52;
    const double x56 = d_4*x46 - d_4*x47 + x54;
    const double x57 = d_4*x35 - d_4*x36 + x55;
    const double x58 = d_1*x10 + x5*x51;
    const double x59 = d_2*x10*x11 - d_2*x17 + x58;
    const double x60 = d_4*x20 - d_4*x26 + x59;
    const double x61 = a_0*x39;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = x18;
    jacobian(0, 5) = x27;
    jacobian(1, 1) = x5;
    jacobian(1, 2) = -x5;
    jacobian(1, 3) = x30;
    jacobian(1, 4) = x34;
    jacobian(1, 5) = x37;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x40;
    jacobian(2, 4) = x45;
    jacobian(2, 5) = x48;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x5;
    jacobian(3, 2) = x5*x50;
    jacobian(3, 3) = -x30*x53 + x40*x52;
    jacobian(3, 4) = -x34*x54 + x45*x55;
    jacobian(3, 5) = -x37*x56 + x48*x57;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = x1*x50;
    jacobian(4, 3) = x10*x53 - x40*x58;
    jacobian(4, 4) = x18*x54 - x45*x59;
    jacobian(4, 5) = x27*x56 - x48*x60;
    jacobian(5, 2) = -std::pow(x0, 2)*x61 - std::pow(x4, 2)*x61;
    jacobian(5, 3) = -x10*x52 + x30*x58;
    jacobian(5, 4) = -x18*x55 + x34*x59;
    jacobian(5, 5) = -x27*x57 + x37*x60;
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
    const double x10 = std::cos(alpha_3);
    const double x11 = std::sin(alpha_3);
    const double x12 = std::cos(th_3);
    const double x13 = std::sin(th_3);
    const double x14 = x1*x8 + x5*x6;
    const double x15 = -x0*x12 - x13*x14;
    const double x16 = x10*x9 - x11*x15;
    const double x17 = std::cos(alpha_5);
    const double x18 = std::sin(alpha_5);
    const double x19 = std::cos(th_4);
    const double x20 = x11*x19;
    const double x21 = std::sin(th_4);
    const double x22 = x10*x19;
    const double x23 = x0*x2;
    const double x24 = x0*x7;
    const double x25 = x1*x23 - x24*x6;
    const double x26 = x1*x24 + x23*x6;
    const double x27 = 1.0*x12*x3 - x13*x26;
    const double x28 = x10*x25 - x11*x27;
    const double x29 = 1.0*x1;
    const double x30 = 1.0*x2*x6 + x29*x7;
    const double x31 = -x2*x29 + 1.0*x6*x7;
    const double x32 = x13*x31;
    const double x33 = x10*x30 + x11*x32;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x16;
    jacobian(0, 5) = x16*x17 - x18*(x15*x22 + x20*x9 - x21*(-x0*x13 + x12*x14));
    jacobian(1, 1) = x4;
    jacobian(1, 2) = -x4;
    jacobian(1, 3) = x25;
    jacobian(1, 4) = x28;
    jacobian(1, 5) = x17*x28 - x18*(x20*x25 - x21*(x12*x26 + x13*x4) + x22*x27);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x30;
    jacobian(2, 4) = x33;
    jacobian(2, 5) = x17*x33 - x18*(x11*x19*x30 - x12*x21*x31 - x22*x32);
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
    const double x7 = pre_transform_special_symbol_23 - x6;
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = std::cos(th_2);
    const double x11 = std::cos(th_1);
    const double x12 = 1.0*x11;
    const double x13 = x10*x12;
    const double x14 = x13 + x9;
    const double x15 = std::sin(th_0);
    const double x16 = x10*x5;
    const double x17 = -x12*x15*x8 + x15*x16;
    const double x18 = d_1*x14 + pre_transform_special_symbol_23 - x6;
    const double x19 = a_0*x12;
    const double x20 = d_1*x17 + x15*x19;
    const double x21 = std::cos(alpha_3);
    const double x22 = x14*x21;
    const double x23 = std::sin(alpha_3);
    const double x24 = std::sin(th_3);
    const double x25 = 1.0*x11*x8 - x16;
    const double x26 = x24*x25;
    const double x27 = x23*x26;
    const double x28 = x22 + x27;
    const double x29 = std::cos(th_3);
    const double x30 = x13*x15 + x15*x9;
    const double x31 = 1.0*x1*x29 - x24*x30;
    const double x32 = x23*x31;
    const double x33 = x17*x21 - x32;
    const double x34 = d_2*x22 + d_2*x27 + x18;
    const double x35 = d_2*x17*x21 - d_2*x32 + x20;
    const double x36 = std::cos(alpha_5);
    const double x37 = x28*x36;
    const double x38 = std::sin(alpha_5);
    const double x39 = std::sin(th_4);
    const double x40 = std::cos(th_4);
    const double x41 = x21*x40;
    const double x42 = x38*(x14*x23*x40 - x25*x29*x39 - x26*x41);
    const double x43 = x37 - x42;
    const double x44 = x33*x36;
    const double x45 = x23*x40;
    const double x46 = x38*(x17*x45 + x31*x41 - x39*(x2*x24 + x29*x30));
    const double x47 = x44 - x46;
    const double x48 = d_4*x37 - d_4*x42 + x34;
    const double x49 = d_4*x44 - d_4*x46 + x35;
    const double x50 = 1.0*p_on_ee_x;
    const double x51 = 1.0*x15;
    const double x52 = p_on_ee_z*x51;
    const double x53 = x10*x2;
    const double x54 = x11*x2;
    const double x55 = x4*x53 - x54*x8;
    const double x56 = a_0*x54 + d_1*x55;
    const double x57 = x11*x53 + x2*x4*x8;
    const double x58 = -x24*x57 - x29*x51;
    const double x59 = x23*x58;
    const double x60 = x21*x55 - x59;
    const double x61 = d_2*x21*x55 - d_2*x59 + x56;
    const double x62 = x36*x60;
    const double x63 = x38*(-x39*(-x24*x51 + x29*x57) + x41*x58 + x45*x55);
    const double x64 = x62 - x63;
    const double x65 = d_4*x62 - d_4*x63 + x61;
    const double x66 = x1*x50;
    const double x67 = x0*x15;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = 1.0*x1*x7 - x3;
    jacobian(0, 3) = -p_on_ee_y*x14 + p_on_ee_z*x17 + x14*x20 - x17*x18;
    jacobian(0, 4) = -p_on_ee_y*x28 + p_on_ee_z*x33 + x28*x35 - x33*x34;
    jacobian(0, 5) = -p_on_ee_y*x43 + p_on_ee_z*x47 + x43*x49 - x47*x48;
    jacobian(1, 0) = x50;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x51 + x52;
    jacobian(1, 2) = 1.0*x15*x7 - x52;
    jacobian(1, 3) = p_on_ee_x*x14 - p_on_ee_z*x55 - x14*x56 + x18*x55;
    jacobian(1, 4) = p_on_ee_x*x28 - p_on_ee_z*x60 - x28*x61 + x34*x60;
    jacobian(1, 5) = p_on_ee_x*x43 - p_on_ee_z*x64 - x43*x65 + x48*x64;
    jacobian(2, 1) = -x66 - x67;
    jacobian(2, 2) = -std::pow(x1, 2)*x19 - std::pow(x15, 2)*x19 + x66 + x67;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x55 + x17*x56 - x20*x55;
    jacobian(2, 4) = -p_on_ee_x*x33 + p_on_ee_y*x60 + x33*x61 - x35*x60;
    jacobian(2, 5) = -p_on_ee_x*x47 + p_on_ee_y*x64 + x47*x65 - x49*x64;
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
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 48, 48, 96, 96, 192, 192, 192, 192, 192, 192, 192, 192}};
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

    // Code for general_6dof solution node 1, solved variable is th_3
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;

        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 0) = -d_2*r_21*std::sin(alpha_5);
        R_l(0, 1) = -d_2*r_22*std::sin(alpha_5);
        R_l(0, 2) = -d_2*r_11*std::sin(alpha_5);
        R_l(0, 3) = -d_2*r_12*std::sin(alpha_5);
        R_l(0, 4) = Py - d_2*r_23*std::cos(alpha_5) - d_4*r_23;
        R_l(0, 5) = Px - d_2*r_13*std::cos(alpha_5) - d_4*r_13;
        R_l(1, 0) = -d_2*r_11*std::sin(alpha_5);
        R_l(1, 1) = -d_2*r_12*std::sin(alpha_5);
        R_l(1, 2) = d_2*r_21*std::sin(alpha_5);
        R_l(1, 3) = d_2*r_22*std::sin(alpha_5);
        R_l(1, 4) = Px - d_2*r_13*std::cos(alpha_5) - d_4*r_13;
        R_l(1, 5) = -Py + d_2*r_23*std::cos(alpha_5) + d_4*r_23;
        R_l(2, 6) = -d_2*r_31*std::sin(alpha_5);
        R_l(2, 7) = -d_2*r_32*std::sin(alpha_5);
        R_l(3, 0) = r_21*std::sin(alpha_5);
        R_l(3, 1) = r_22*std::sin(alpha_5);
        R_l(3, 2) = r_11*std::sin(alpha_5);
        R_l(3, 3) = r_12*std::sin(alpha_5);
        R_l(3, 4) = r_23*std::cos(alpha_5);
        R_l(3, 5) = r_13*std::cos(alpha_5);
        R_l(4, 0) = r_11*std::sin(alpha_5);
        R_l(4, 1) = r_12*std::sin(alpha_5);
        R_l(4, 2) = -r_21*std::sin(alpha_5);
        R_l(4, 3) = -r_22*std::sin(alpha_5);
        R_l(4, 4) = r_13*std::cos(alpha_5);
        R_l(4, 5) = -r_23*std::cos(alpha_5);
        R_l(5, 6) = -2*Px*d_2*r_11*std::sin(alpha_5) - 2*Py*d_2*r_21*std::sin(alpha_5) - 2*Pz*d_2*r_31*std::sin(alpha_5) + 2*std::pow(d_2, 2)*r_11*r_13*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_21*r_23*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_31*r_33*std::sin(alpha_5)*std::cos(alpha_5) + 2*d_2*d_4*r_11*r_13*std::sin(alpha_5) + 2*d_2*d_4*r_21*r_23*std::sin(alpha_5) + 2*d_2*d_4*r_31*r_33*std::sin(alpha_5);
        R_l(5, 7) = -2*Px*d_2*r_12*std::sin(alpha_5) - 2*Py*d_2*r_22*std::sin(alpha_5) - 2*Pz*d_2*r_32*std::sin(alpha_5) + 2*std::pow(d_2, 2)*r_12*r_13*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_22*r_23*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_32*r_33*std::sin(alpha_5)*std::cos(alpha_5) + 2*d_2*d_4*r_12*r_13*std::sin(alpha_5) + 2*d_2*d_4*r_22*r_23*std::sin(alpha_5) + 2*d_2*d_4*r_32*r_33*std::sin(alpha_5);
        R_l(6, 0) = Px*r_31*std::sin(alpha_5) - Pz*r_11*std::sin(alpha_5) + d_4*r_11*r_33*std::sin(alpha_5) - d_4*r_13*r_31*std::sin(alpha_5);
        R_l(6, 1) = Px*r_32*std::sin(alpha_5) - Pz*r_12*std::sin(alpha_5) + d_4*r_12*r_33*std::sin(alpha_5) - d_4*r_13*r_32*std::sin(alpha_5);
        R_l(6, 2) = -Py*r_31*std::sin(alpha_5) + Pz*r_21*std::sin(alpha_5) - d_4*r_21*r_33*std::sin(alpha_5) + d_4*r_23*r_31*std::sin(alpha_5);
        R_l(6, 3) = -Py*r_32*std::sin(alpha_5) + Pz*r_22*std::sin(alpha_5) - d_4*r_22*r_33*std::sin(alpha_5) + d_4*r_23*r_32*std::sin(alpha_5);
        R_l(6, 4) = Px*r_33*std::cos(alpha_5) - Pz*r_13*std::cos(alpha_5);
        R_l(6, 5) = -Py*r_33*std::cos(alpha_5) + Pz*r_23*std::cos(alpha_5);
        R_l(7, 0) = -Py*r_31*std::sin(alpha_5) + Pz*r_21*std::sin(alpha_5) - d_4*r_21*r_33*std::sin(alpha_5) + d_4*r_23*r_31*std::sin(alpha_5);
        R_l(7, 1) = -Py*r_32*std::sin(alpha_5) + Pz*r_22*std::sin(alpha_5) - d_4*r_22*r_33*std::sin(alpha_5) + d_4*r_23*r_32*std::sin(alpha_5);
        R_l(7, 2) = -Px*r_31*std::sin(alpha_5) + Pz*r_11*std::sin(alpha_5) - d_4*r_11*r_33*std::sin(alpha_5) + d_4*r_13*r_31*std::sin(alpha_5);
        R_l(7, 3) = -Px*r_32*std::sin(alpha_5) + Pz*r_12*std::sin(alpha_5) - d_4*r_12*r_33*std::sin(alpha_5) + d_4*r_13*r_32*std::sin(alpha_5);
        R_l(7, 4) = -Py*r_33*std::cos(alpha_5) + Pz*r_23*std::cos(alpha_5);
        R_l(7, 5) = -Px*r_33*std::cos(alpha_5) + Pz*r_13*std::cos(alpha_5);
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
        const double x0 = std::sin(alpha_5);
        const double x1 = r_31*x0;
        const double x2 = r_32*x0;
        const double x3 = R_l_inv_60*x1 + R_l_inv_70*x2;
        const double x4 = a_0*x3;
        const double x5 = -x4;
        const double x6 = std::cos(alpha_3);
        const double x7 = -x6;
        const double x8 = R_l_inv_62*x1 + R_l_inv_72*x2;
        const double x9 = d_1*x8;
        const double x10 = R_l_inv_67*x1 + R_l_inv_77*x2;
        const double x11 = a_0*x6;
        const double x12 = x10*x11;
        const double x13 = R_l_inv_66*x1 + R_l_inv_76*x2;
        const double x14 = std::sin(alpha_3);
        const double x15 = d_1*x14;
        const double x16 = x13*x15;
        const double x17 = x12 + x16 + x7 + x9;
        const double x18 = std::cos(alpha_5);
        const double x19 = r_33*x18;
        const double x20 = d_4*r_33;
        const double x21 = d_2*x19;
        const double x22 = -Pz + x20 + x21;
        const double x23 = x22*x8;
        const double x24 = R_l_inv_65*x1 + R_l_inv_75*x2;
        const double x25 = std::pow(Px, 2);
        const double x26 = std::pow(Py, 2);
        const double x27 = std::pow(Pz, 2);
        const double x28 = std::pow(a_0, 2);
        const double x29 = std::pow(d_1, 2);
        const double x30 = std::pow(d_4, 2);
        const double x31 = std::pow(r_13, 2);
        const double x32 = x30*x31;
        const double x33 = std::pow(r_23, 2);
        const double x34 = x30*x33;
        const double x35 = std::pow(r_33, 2);
        const double x36 = x30*x35;
        const double x37 = d_4*x18;
        const double x38 = x31*x37;
        const double x39 = 2*d_2;
        const double x40 = x33*x37;
        const double x41 = x35*x37;
        const double x42 = std::pow(r_11, 2);
        const double x43 = std::pow(d_2, 2);
        const double x44 = std::pow(x0, 2);
        const double x45 = x43*x44;
        const double x46 = x42*x45;
        const double x47 = std::pow(x18, 2);
        const double x48 = x43*x47;
        const double x49 = x31*x48;
        const double x50 = std::pow(r_21, 2);
        const double x51 = x45*x50;
        const double x52 = x33*x48;
        const double x53 = std::pow(r_31, 2);
        const double x54 = x45*x53;
        const double x55 = x35*x48;
        const double x56 = 2*Px*d_2*r_13*x18 + 2*Px*d_4*r_13 + 2*Py*d_2*r_23*x18 + 2*Py*d_4*r_23 + 2*Pz*d_2*r_33*x18 + 2*Pz*d_4*r_33 - x25 - x26 - x27 + x28 + x29 - x32 - x34 - x36 - x38*x39 - x39*x40 - x39*x41 - x46 - x49 - x51 - x52 - x54 - x55;
        const double x57 = x24*x56;
        const double x58 = x14*(R_l_inv_64*x1 + R_l_inv_74*x2);
        const double x59 = x19 + x23 + x57 - x58;
        const double x60 = 2*R_l_inv_63;
        const double x61 = 2*R_l_inv_73;
        const double x62 = x1*x60 + x2*x61;
        const double x63 = x6*x62;
        const double x64 = 2*d_1;
        const double x65 = x3*x64;
        const double x66 = x63 + x65;
        const double x67 = 4*a_0*d_1*x24 - x66;
        const double x68 = x4 + x58;
        const double x69 = 2*a_0;
        const double x70 = x69*x8;
        const double x71 = a_0*x14;
        const double x72 = 2*x71;
        const double x73 = x13*x72;
        const double x74 = x70 + x73;
        const double x75 = 4*x9;
        const double x76 = 4*x14;
        const double x77 = d_1*x76;
        const double x78 = x13*x77;
        const double x79 = x4 + x59;
        const double x80 = -x16;
        const double x81 = x12 + x80;
        const double x82 = x6 - x9;
        const double x83 = 4*d_1;
        const double x84 = a_0*x83;
        const double x85 = x24*x84 + x66;
        const double x86 = x7 + x9;
        const double x87 = -x12;
        const double x88 = x16 + x87;
        const double x89 = d_1*x6;
        const double x90 = r_11*x0;
        const double x91 = d_4*r_13;
        const double x92 = x90*x91;
        const double x93 = r_21*x0;
        const double x94 = d_4*r_23;
        const double x95 = x93*x94;
        const double x96 = r_13*x18;
        const double x97 = x39*x96;
        const double x98 = r_23*x18;
        const double x99 = x39*x98;
        const double x100 = 2*x21;
        const double x101 = Px*r_11*x0 + Py*r_21*x0 + Pz*r_31*x0 - x1*x100 - x1*x20 - x90*x97 - x92 - x93*x99 - x95;
        const double x102 = r_12*x0;
        const double x103 = x102*x91;
        const double x104 = r_22*x0;
        const double x105 = x104*x94;
        const double x106 = Px*r_12*x0 + Py*r_22*x0 + Pz*r_32*x0 - x100*x2 - x102*x97 - x103 - x104*x99 - x105 - x2*x20;
        const double x107 = R_l_inv_62*x101 + R_l_inv_72*x106;
        const double x108 = d_1*x107;
        const double x109 = -x107*x22;
        const double x110 = R_l_inv_65*x101 + R_l_inv_75*x106;
        const double x111 = -x110*x56;
        const double x112 = d_2*x44;
        const double x113 = x112*x42;
        const double x114 = d_2*x47;
        const double x115 = x114*x31;
        const double x116 = x112*x50;
        const double x117 = x114*x33;
        const double x118 = x112*x53;
        const double x119 = x114*x35;
        const double x120 = Px*x18;
        const double x121 = r_13*x120;
        const double x122 = -x121;
        const double x123 = Py*x18;
        const double x124 = r_23*x123;
        const double x125 = -x124;
        const double x126 = Pz*x19;
        const double x127 = -x126;
        const double x128 = -x108 + x109 + x111 + x113 + x115 + x116 + x117 + x118 + x119 + x122 + x125 + x127 + x38 + x40 + x41 + x89;
        const double x129 = R_l_inv_60*x101 + R_l_inv_70*x106;
        const double x130 = a_0*x129;
        const double x131 = x14*(R_l_inv_64*x101 + R_l_inv_74*x106);
        const double x132 = x130 + x131;
        const double x133 = R_l_inv_67*x101 + R_l_inv_77*x106;
        const double x134 = x11*x133;
        const double x135 = -x134;
        const double x136 = R_l_inv_66*x101 + R_l_inv_76*x106;
        const double x137 = x136*x15;
        const double x138 = -x137;
        const double x139 = x135 + x138;
        const double x140 = 2*x11;
        const double x141 = x101*x60 + x106*x61;
        const double x142 = x141*x6;
        const double x143 = x129*x64;
        const double x144 = x142 + x143;
        const double x145 = 4*a_0*d_1*x110 - x140 - x144;
        const double x146 = x134 + x137;
        const double x147 = x108 + x109 + x111 + x113 + x115 + x116 + x117 + x118 + x119 + x122 + x125 + x127 + x38 + x40 + x41 + x89;
        const double x148 = x107*x69;
        const double x149 = x136*x72;
        const double x150 = x148 + x149;
        const double x151 = 4*x108;
        const double x152 = x136*x77;
        const double x153 = -x130;
        const double x154 = x131 + x153;
        const double x155 = x135 + x137;
        const double x156 = x110*x84 - x140 + x144;
        const double x157 = x134 + x138;
        const double x158 = Px*x0;
        const double x159 = r_21*x158;
        const double x160 = Py*r_11*x0 + d_4*r_13*r_21*x0 - x159 - x90*x94;
        const double x161 = r_22*x158;
        const double x162 = Py*r_12*x0 + d_4*r_13*r_22*x0 - x102*x94 - x161;
        const double x163 = R_l_inv_60*x160 + R_l_inv_70*x162;
        const double x164 = a_0*x163;
        const double x165 = -x164;
        const double x166 = -x71;
        const double x167 = R_l_inv_62*x160 + R_l_inv_72*x162;
        const double x168 = d_1*x167;
        const double x169 = R_l_inv_67*x160 + R_l_inv_77*x162;
        const double x170 = x11*x169;
        const double x171 = R_l_inv_66*x160 + R_l_inv_76*x162;
        const double x172 = x15*x171;
        const double x173 = x168 + x170 + x172;
        const double x174 = x167*x22;
        const double x175 = R_l_inv_65*x160 + R_l_inv_75*x162;
        const double x176 = x175*x56;
        const double x177 = x14*(R_l_inv_64*x160 + R_l_inv_74*x162);
        const double x178 = r_13*x123;
        const double x179 = -Px*r_23*x18;
        const double x180 = x174 + x176 - x177 + x178 + x179;
        const double x181 = 2*x15;
        const double x182 = x160*x60 + x162*x61;
        const double x183 = x182*x6;
        const double x184 = x163*x64;
        const double x185 = x183 + x184;
        const double x186 = x181 + x185;
        const double x187 = r_23*x120;
        const double x188 = x164 + x71;
        const double x189 = x167*x69;
        const double x190 = x171*x72;
        const double x191 = 4*x168;
        const double x192 = x171*x77;
        const double x193 = -2*a_0*x167;
        const double x194 = -x190;
        const double x195 = -x168;
        const double x196 = x180 + x188;
        const double x197 = -x172;
        const double x198 = x170 + x197;
        const double x199 = x175*x84;
        const double x200 = -x170;
        const double x201 = x172 + x200;
        const double x202 = r_11*x158;
        const double x203 = 2*Py;
        const double x204 = Pz*x1;
        const double x205 = 2*d_4;
        const double x206 = r_13*x205;
        const double x207 = 2*Pz;
        const double x208 = x20*x207;
        const double x209 = 2*r_23;
        const double x210 = x209*x30;
        const double x211 = r_13*x90;
        const double x212 = r_33*x210;
        const double x213 = std::pow(x0, 3)*x43;
        const double x214 = 4*d_2;
        const double x215 = r_23*x214*x37;
        const double x216 = 4*x21;
        const double x217 = x216*x94;
        const double x218 = r_21*x213;
        const double x219 = x214*x40;
        const double x220 = x209*x48;
        const double x221 = r_33*x1;
        const double x222 = 3*x52;
        const double x223 = -2*Px*d_4*r_11*r_23*x0 - 4*Py*d_2*r_11*r_13*x0*x18 - 4*Py*d_2*r_21*r_23*x0*x18 - 4*Py*d_2*r_31*r_33*x0*x18 - 2*Py*d_4*r_11*r_13*x0 - 2*Py*d_4*r_21*r_23*x0 - 2*Py*d_4*r_31*r_33*x0 - 2*Pz*d_4*r_23*r_31*x0 + std::pow(r_21, 3)*x213 - r_21*x0*x25 - r_21*x0*x27 - r_21*x0*x30*x31 - r_21*x0*x30*x35 + x1*x212 + x1*x217 + x159*x206 + x202*x203 + x203*x204 + x208*x93 + x210*x211 + x211*x215 + x211*x220 + x218*x42 + x218*x53 + x219*x93 + x220*x221 + x222*x93 + x26*x93 + x34*x93 + x49*x93 + x55*x93;
        const double x224 = -x223;
        const double x225 = 2*Px;
        const double x226 = Py*x0;
        const double x227 = r_23*x205;
        const double x228 = r_13*x210;
        const double x229 = 2*x30;
        const double x230 = r_13*x229;
        const double x231 = r_13*x215;
        const double x232 = x216*x91;
        const double x233 = r_11*x213;
        const double x234 = x214*x38;
        const double x235 = r_13*x220;
        const double x236 = 2*x48;
        const double x237 = r_13*x236;
        const double x238 = 3*x49;
        const double x239 = 4*Px*d_2*r_11*r_13*x0*x18 + 4*Px*d_2*r_21*r_23*x0*x18 + 4*Px*d_2*r_31*r_33*x0*x18 + 2*Px*d_4*r_11*r_13*x0 + 2*Px*d_4*r_21*r_23*x0 + 2*Px*d_4*r_31*r_33*x0 + 2*Py*d_4*r_13*r_21*x0 + 2*Pz*d_4*r_13*r_31*x0 - std::pow(r_11, 3)*x213 + r_11*x0*x26 + r_11*x0*x27 + r_11*x0*x30*x33 + r_11*x0*x30*x35 - r_11*x226*x227 - x1*x232 - x159*x203 - x204*x225 - x208*x90 - x221*x230 - x221*x237 - x228*x93 - x231*x93 - x233*x50 - x233*x53 - x234*x90 - x235*x93 - x238*x90 - x25*x90 - x32*x90 - x52*x90 - x55*x90;
        const double x240 = r_12*x158;
        const double x241 = Pz*x2;
        const double x242 = r_22*x213;
        const double x243 = 2*x218;
        const double x244 = r_11*x243;
        const double x245 = r_31*r_32;
        const double x246 = r_33*x2;
        const double x247 = -2*Px*d_4*r_12*r_23*x0 - 4*Py*d_2*r_12*r_13*x0*x18 - 4*Py*d_2*r_22*r_23*x0*x18 - 4*Py*d_2*r_32*r_33*x0*x18 - 2*Py*d_4*r_12*r_13*x0 - 2*Py*d_4*r_22*r_23*x0 - 2*Py*d_4*r_32*r_33*x0 - 2*Pz*d_4*r_23*r_32*x0 + r_12*x244 - r_22*x0*x25 - r_22*x0*x27 - r_22*x0*x30*x31 - r_22*x0*x30*x35 + x102*x228 + x102*x231 + x102*x235 + x104*x208 + x104*x219 + x104*x222 + x104*x26 + x104*x34 + x104*x49 + x104*x55 + x161*x206 + x2*x212 + x2*x217 + x203*x240 + x203*x241 + x220*x246 + x242*x42 + 3*x242*x50 + x242*x53 + x243*x245;
        const double x248 = -x247;
        const double x249 = r_33*x230;
        const double x250 = r_12*x213;
        const double x251 = 2*x233;
        const double x252 = 4*Px*d_2*r_12*r_13*x0*x18 + 4*Px*d_2*r_22*r_23*x0*x18 + 4*Px*d_2*r_32*r_33*x0*x18 + 2*Px*d_4*r_12*r_13*x0 + 2*Px*d_4*r_22*r_23*x0 + 2*Px*d_4*r_32*r_33*x0 + 2*Py*d_4*r_13*r_22*x0 + 2*Pz*d_4*r_13*r_32*x0 + r_12*x0*x26 + r_12*x0*x27 + r_12*x0*x30*x33 + r_12*x0*x30*x35 - r_12*x226*x227 - r_22*x244 - x102*x208 - x102*x234 - x102*x238 - x102*x25 - x102*x32 - x102*x52 - x102*x55 - x104*x228 - x104*x231 - x104*x235 - x161*x203 - x2*x232 - x2*x249 - x225*x241 - x237*x246 - x245*x251 - 3*x250*x42 - x250*x50 - x250*x53;
        const double x253 = std::pow(r_23, 3);
        const double x254 = x18*x30;
        const double x255 = std::pow(x18, 3)*x43;
        const double x256 = x114*x205;
        const double x257 = r_23*x255;
        const double x258 = 2*r_21*x45;
        const double x259 = r_11*x258;
        const double x260 = r_31*x19;
        const double x261 = -2*Py*d_2*x31*x47 - 2*Py*d_2*x33*x47 - 2*Py*d_2*x35*x47 - 2*Py*d_2*x42*x44 - 2*Py*d_2*x44*x50 - 2*Py*d_2*x44*x53 - 2*Py*d_4*x18*x31 - 2*Py*d_4*x18*x33 - 2*Py*d_4*x18*x35 - r_23*x18*x25 - r_23*x18*x27 + x113*x227 + x115*x227 + x116*x227 + x118*x227 + x119*x227 + x121*x203 + x126*x203 + x253*x254 + x253*x255 + x253*x256 + x257*x31 + x257*x35 + x258*x260 + x259*x96 + x26*x98 + x32*x98 + x36*x98 + x46*x98 + 3*x51*x98 + x54*x98;
        const double x262 = -x261;
        const double x263 = std::pow(r_13, 3);
        const double x264 = r_13*x255;
        const double x265 = 2*r_11*x45;
        const double x266 = 2*Px*d_2*x31*x47 + 2*Px*d_2*x33*x47 + 2*Px*d_2*x35*x47 + 2*Px*d_2*x42*x44 + 2*Px*d_2*x44*x50 + 2*Px*d_2*x44*x53 + 2*Px*d_4*x18*x31 + 2*Px*d_4*x18*x33 + 2*Px*d_4*x18*x35 + r_13*x18*x26 + r_13*x18*x27 - x113*x206 - x116*x206 - x117*x206 - x118*x206 - x119*x206 - x126*x225 - x187*x203 - x25*x96 - x254*x263 - x255*x263 - x256*x263 - x259*x98 - x260*x265 - x264*x33 - x264*x35 - x34*x96 - x36*x96 - 3*x46*x96 - x51*x96 - x54*x96;
        const double x267 = R_l_inv_02*x224 + R_l_inv_12*x248 + R_l_inv_22*x239 + R_l_inv_32*x252 + R_l_inv_42*x262 + R_l_inv_52*x266;
        const double x268 = d_1*x267;
        const double x269 = R_l_inv_07*x224 + R_l_inv_17*x248 + R_l_inv_27*x239 + R_l_inv_37*x252 + R_l_inv_47*x262 + R_l_inv_57*x266;
        const double x270 = x11*x269;
        const double x271 = R_l_inv_06*x224 + R_l_inv_16*x248 + R_l_inv_26*x239 + R_l_inv_36*x252 + R_l_inv_46*x262 + R_l_inv_56*x266;
        const double x272 = x15*x271;
        const double x273 = x268 + x270 + x272;
        const double x274 = x14*(R_l_inv_04*x224 + R_l_inv_14*x248 + R_l_inv_24*x239 + R_l_inv_34*x252 + R_l_inv_44*x262 + R_l_inv_54*x266);
        const double x275 = -x274;
        const double x276 = x22*x267;
        const double x277 = R_l_inv_05*x224 + R_l_inv_15*x248 + R_l_inv_25*x239 + R_l_inv_35*x252 + R_l_inv_45*x262 + R_l_inv_55*x266;
        const double x278 = x277*x56;
        const double x279 = x276 + x278;
        const double x280 = x275 + x279;
        const double x281 = R_l_inv_00*x224 + R_l_inv_10*x248 + R_l_inv_20*x239 + R_l_inv_30*x252 + R_l_inv_40*x262 + R_l_inv_50*x266;
        const double x282 = a_0*x281;
        const double x283 = x69*x89;
        const double x284 = -x282 - x283;
        const double x285 = x28*x6;
        const double x286 = 2*x285;
        const double x287 = 2*R_l_inv_03;
        const double x288 = 2*R_l_inv_23;
        const double x289 = 2*R_l_inv_13;
        const double x290 = 2*R_l_inv_33;
        const double x291 = 2*R_l_inv_43;
        const double x292 = 2*R_l_inv_53;
        const double x293 = x224*x287 + x239*x288 + x248*x289 + x252*x290 + x262*x291 + x266*x292;
        const double x294 = x293*x6;
        const double x295 = x281*x64;
        const double x296 = x29*x6;
        const double x297 = 2*x296;
        const double x298 = x294 + x295 + x297;
        const double x299 = x286 + x298;
        const double x300 = 4*a_0*d_1*x277 - x299;
        const double x301 = x282 + x283;
        const double x302 = x274 + x301;
        const double x303 = -x276 - x278;
        const double x304 = x267*x69;
        const double x305 = x271*x72;
        const double x306 = x304 + x305;
        const double x307 = -x286 + x298;
        const double x308 = 4*x268;
        const double x309 = x271*x77;
        const double x310 = -x268;
        const double x311 = x280 + x301;
        const double x312 = -x272;
        const double x313 = x270 + x312;
        const double x314 = x277*x84 + x299;
        const double x315 = x268 - x270;
        const double x316 = R_l_inv_02*x239 + R_l_inv_12*x252 + R_l_inv_22*x223 + R_l_inv_32*x247 + R_l_inv_42*x266 + R_l_inv_52*x261;
        const double x317 = d_1*x316;
        const double x318 = R_l_inv_07*x239 + R_l_inv_17*x252 + R_l_inv_27*x223 + R_l_inv_37*x247 + R_l_inv_47*x266 + R_l_inv_57*x261;
        const double x319 = x11*x318;
        const double x320 = R_l_inv_06*x239 + R_l_inv_16*x252 + R_l_inv_26*x223 + R_l_inv_36*x247 + R_l_inv_46*x266 + R_l_inv_56*x261;
        const double x321 = x15*x320;
        const double x322 = x317 + x319 + x321;
        const double x323 = R_l_inv_00*x239 + R_l_inv_10*x252 + R_l_inv_20*x223 + R_l_inv_30*x247 + R_l_inv_40*x266 + R_l_inv_50*x261;
        const double x324 = a_0*x323;
        const double x325 = x22*x316;
        const double x326 = R_l_inv_05*x239 + R_l_inv_15*x252 + R_l_inv_25*x223 + R_l_inv_35*x247 + R_l_inv_45*x266 + R_l_inv_55*x261;
        const double x327 = x326*x56;
        const double x328 = x325 + x327;
        const double x329 = -x324 + x328;
        const double x330 = x14*x28;
        const double x331 = x14*x29;
        const double x332 = x14*(R_l_inv_04*x239 + R_l_inv_14*x252 + R_l_inv_24*x223 + R_l_inv_34*x247 + R_l_inv_44*x266 + R_l_inv_54*x261);
        const double x333 = x330 + x331 - x332;
        const double x334 = x223*x288 + x239*x287 + x247*x290 + x252*x289 + x261*x292 + x266*x291;
        const double x335 = x334*x6;
        const double x336 = x323*x64;
        const double x337 = -x335 - x336;
        const double x338 = x71*x83;
        const double x339 = x326*x84;
        const double x340 = x338 + x339;
        const double x341 = -x330 - x331 + x332;
        const double x342 = x324 - x325 - x327;
        const double x343 = x316*x69;
        const double x344 = x320*x72;
        const double x345 = x343 + x344;
        const double x346 = x335 + x336;
        const double x347 = 4*x317;
        const double x348 = x320*x77;
        const double x349 = -x317;
        const double x350 = -x321;
        const double x351 = x319 + x350;
        const double x352 = x324 + x328 + x333;
        const double x353 = x317 - x319;
        const double x354 = Px*x206;
        const double x355 = Py*x227;
        const double x356 = r_33*x211;
        const double x357 = r_31*x213;
        const double x358 = x214*x41;
        const double x359 = r_33*x220;
        const double x360 = 3*x55;
        const double x361 = 2*Px*d_4*r_11*r_33*x0 + 2*Py*d_4*r_21*r_33*x0 + 4*Pz*d_2*r_11*r_13*x0*x18 + 4*Pz*d_2*r_21*r_23*x0*x18 + 4*Pz*d_2*r_31*r_33*x0*x18 + 2*Pz*d_4*r_11*r_13*x0 + 2*Pz*d_4*r_21*r_23*x0 + 2*Pz*d_4*r_31*r_33*x0 - r_21*x207*x226 - std::pow(r_31, 3)*x213 + r_31*x0*x25 + r_31*x0*x26 + r_31*x0*x30*x31 + r_31*x0*x30*x33 - x1*x27 - x1*x354 - x1*x355 - x1*x358 - x1*x36 - x1*x360 - x1*x49 - x1*x52 - x202*x207 - x212*x93 - x216*x92 - x216*x95 - x229*x356 - x236*x356 - x357*x42 - x357*x50 - x359*x93;
        const double x362 = r_32*x213;
        const double x363 = 2*Px*d_4*r_12*r_33*x0 + 2*Py*d_4*r_22*r_33*x0 + 4*Pz*d_2*r_12*r_13*x0*x18 + 4*Pz*d_2*r_22*r_23*x0*x18 + 4*Pz*d_2*r_32*r_33*x0*x18 + 2*Pz*d_4*r_12*r_13*x0 + 2*Pz*d_4*r_22*r_23*x0 + 2*Pz*d_4*r_32*r_33*x0 - r_12*r_31*x251 - r_22*r_31*x243 - r_22*x207*x226 + r_32*x0*x25 + r_32*x0*x26 + r_32*x0*x30*x31 + r_32*x0*x30*x33 - r_33*x102*x237 - x102*x249 - x103*x216 - x104*x212 - x104*x359 - x105*x216 - x2*x27 - x2*x354 - x2*x355 - x2*x358 - x2*x36 - x2*x360 - x2*x49 - x2*x52 - x207*x240 - x362*x42 - x362*x50 - 3*x362*x53;
        const double x364 = R_l_inv_60*x361 + R_l_inv_70*x363;
        const double x365 = a_0*x364;
        const double x366 = x14*(R_l_inv_64*x361 + R_l_inv_74*x363);
        const double x367 = R_l_inv_62*x361 + R_l_inv_72*x363;
        const double x368 = -x22*x367;
        const double x369 = R_l_inv_65*x361 + R_l_inv_75*x363;
        const double x370 = -x369*x56;
        const double x371 = x19*x27;
        const double x372 = std::pow(r_33, 3);
        const double x373 = x255*x372;
        const double x374 = x254*x372;
        const double x375 = -x19*x25;
        const double x376 = -x19*x26;
        const double x377 = r_33*x255;
        const double x378 = x31*x377;
        const double x379 = x33*x377;
        const double x380 = x19*x32;
        const double x381 = x19*x34;
        const double x382 = -x113*x207;
        const double x383 = -x115*x207;
        const double x384 = -x116*x207;
        const double x385 = -x117*x207;
        const double x386 = -x118*x207;
        const double x387 = -x119*x207;
        const double x388 = -x207*x38;
        const double x389 = -x207*x40;
        const double x390 = -x207*x41;
        const double x391 = x121*x207;
        const double x392 = x124*x207;
        const double x393 = x256*x372;
        const double x394 = x19*x46;
        const double x395 = x19*x51;
        const double x396 = 2*x20;
        const double x397 = x113*x396;
        const double x398 = x115*x396;
        const double x399 = x116*x396;
        const double x400 = x117*x396;
        const double x401 = x118*x396;
        const double x402 = 3*x19*x54;
        const double x403 = r_31*x265*x96;
        const double x404 = r_31*x258*x98;
        const double x405 = x365 + x366 + x368 + x370 + x371 + x373 + x374 + x375 + x376 + x378 + x379 + x380 + x381 + x382 + x383 + x384 + x385 + x386 + x387 + x388 + x389 + x390 + x391 + x392 + x393 + x394 + x395 + x397 + x398 + x399 + x400 + x401 + x402 + x403 + x404;
        const double x406 = R_l_inv_67*x361 + R_l_inv_77*x363;
        const double x407 = x11*x406;
        const double x408 = -x407;
        const double x409 = R_l_inv_66*x361 + R_l_inv_76*x363;
        const double x410 = x15*x409;
        const double x411 = -x410;
        const double x412 = x408 + x411;
        const double x413 = d_1*x367;
        const double x414 = x285 - x296 - x413;
        const double x415 = x361*x60 + x363*x61;
        const double x416 = x415*x6;
        const double x417 = x364*x64;
        const double x418 = x416 + x417;
        const double x419 = 4*a_0*d_1*x369 - x418;
        const double x420 = x407 + x410;
        const double x421 = -x285 + x296 + x413;
        const double x422 = x409*x72;
        const double x423 = x418 + x422;
        const double x424 = 4*x6;
        const double x425 = a_0*d_1*x424 + x367*x69;
        const double x426 = x409*x77;
        const double x427 = x28*x424 + x29*x424 + 4*x413;
        const double x428 = x368 + x370 + x371 + x373 + x374 + x375 + x376 + x378 + x379 + x380 + x381 + x382 + x383 + x384 + x385 + x386 + x387 + x388 + x389 + x390 + x391 + x392 + x393 + x394 + x395 + x397 + x398 + x399 + x400 + x401 + x402 + x403 + x404 + x421;
        const double x429 = -x365;
        const double x430 = x366 + x429;
        const double x431 = x408 + x410;
        const double x432 = x369*x84 + x418;
        const double x433 = x407 + x411;
        const double x434 = x368 + x370 + x371 + x373 + x374 + x375 + x376 + x378 + x379 + x380 + x381 + x382 + x383 + x384 + x385 + x386 + x387 + x388 + x389 + x390 + x391 + x392 + x393 + x394 + x395 + x397 + x398 + x399 + x400 + x401 + x402 + x403 + x404 + x414;
        const double x435 = x14*x62;
        const double x436 = x10*x181;
        const double x437 = -x435 - x436;
        const double x438 = -x76;
        const double x439 = 4*x71;
        const double x440 = x10*x439;
        const double x441 = x435 - x436;
        const double x442 = 8*R_l_inv_63;
        const double x443 = 8*R_l_inv_73;
        const double x444 = x133*x181;
        const double x445 = x14*x141;
        const double x446 = x445 + x72;
        const double x447 = x133*x439;
        const double x448 = x14*x182;
        const double x449 = x169*x181;
        const double x450 = -x448 - x449;
        const double x451 = x169*x439;
        const double x452 = x448 - x449;
        const double x453 = x181*x269;
        const double x454 = x14*x293 + 2*x330 - 2*x331;
        const double x455 = -x453 - x454;
        const double x456 = x269*x439;
        const double x457 = -x453 + x454;
        const double x458 = -8*d_1*x71;
        const double x459 = 8*R_l_inv_03;
        const double x460 = 8*R_l_inv_23;
        const double x461 = 8*R_l_inv_13;
        const double x462 = 8*R_l_inv_33;
        const double x463 = 8*R_l_inv_43;
        const double x464 = 8*R_l_inv_53;
        const double x465 = x181*x318;
        const double x466 = x14*x334;
        const double x467 = -x465 - x466;
        const double x468 = x318*x439;
        const double x469 = -x465 + x466;
        const double x470 = -x338;
        const double x471 = x14*x415;
        const double x472 = x181*x406;
        const double x473 = x471 + x472;
        const double x474 = 4*x330;
        const double x475 = 4*x331;
        const double x476 = x474 + x475;
        const double x477 = x474 - x475;
        const double x478 = x19 + x23 + x57;
        const double x479 = x478 + x86;
        const double x480 = x5 + x58;
        const double x481 = x478 + x82;
        const double x482 = -x131;
        const double x483 = x130 + x482;
        const double x484 = x153 + x482;
        const double x485 = x174 + x176 + x177 + x178 + x179;
        const double x486 = x168 + x485;
        const double x487 = x165 + x71;
        const double x488 = x195 + x485;
        const double x489 = -x181 + x185;
        const double x490 = x164 + x166;
        const double x491 = x268 + x313;
        const double x492 = x279 + x302;
        const double x493 = x317 + x351;
        const double x494 = x324 + x328 + x341;
        const double x495 = -x366;
        const double x496 = x365 + x495;
        const double x497 = x429 + x495;

        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = -x17 - x5 - x59;
        A(0, 1) = x67;
        A(0, 2) = x17 - x19 - x23 - x57 + x68;
        A(0, 3) = x66 + x74;
        A(0, 4) = 4*x6 - x75 - x78;
        A(0, 5) = -x63 - x65 + x74;
        A(0, 6) = -x79 - x81 - x82;
        A(0, 7) = x85;
        A(0, 8) = -x79 - x86 - x88;
        A(1, 0) = x128 + x132 + x139;
        A(1, 1) = x145;
        A(1, 2) = x132 + x146 + x147;
        A(1, 3) = x144 + x150;
        A(1, 4) = -x151 - x152;
        A(1, 5) = -x142 - x143 + x150;
        A(1, 6) = x147 + x154 + x155;
        A(1, 7) = x156;
        A(1, 8) = x128 + x154 + x157;
        A(2, 0) = -x165 - x166 - x173 - x180;
        A(2, 1) = 4*a_0*d_1*x175 - x186;
        A(2, 2) = x173 - x174 - x176 + x177 - x178 + x187 + x188;
        A(2, 3) = x186 + x189 + x190;
        A(2, 4) = -x191 - x192;
        A(2, 5) = -x186 - x193 - x194;
        A(2, 6) = -x195 - x196 - x198;
        A(2, 7) = x186 + x199;
        A(2, 8) = -x168 - x196 - x201;
        A(3, 0) = -x273 - x280 - x284;
        A(3, 1) = x300;
        A(3, 2) = x273 + x302 + x303;
        A(3, 3) = x306 + x307;
        A(3, 4) = -x308 - x309;
        A(3, 5) = x286 - x294 - x295 - x297 + x306;
        A(3, 6) = -x310 - x311 - x313;
        A(3, 7) = x314;
        A(3, 8) = -x272 - x311 - x315;
        A(4, 0) = -x322 - x329 - x333;
        A(4, 1) = x337 + x340;
        A(4, 2) = x322 + x341 + x342;
        A(4, 3) = x345 + x346;
        A(4, 4) = -x347 - x348;
        A(4, 5) = x337 + x345;
        A(4, 6) = -x349 - x351 - x352;
        A(4, 7) = x340 + x346;
        A(4, 8) = -x321 - x352 - x353;
        A(5, 0) = x405 + x412 + x414;
        A(5, 1) = x419;
        A(5, 2) = x405 + x420 + x421;
        A(5, 3) = x423 + x425;
        A(5, 4) = -x426 - x427;
        A(5, 5) = -x416 - x417 + x422 + x425;
        A(5, 6) = x428 + x430 + x431;
        A(5, 7) = x432;
        A(5, 8) = x430 + x433 + x434;

        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x437;
        B(0, 1) = x438 + x440;
        B(0, 2) = x441;
        B(0, 3) = x76;
        B(0, 4) = x14*(-x1*x442 - x2*x443);
        B(0, 5) = x438;
        B(0, 6) = x441;
        B(0, 7) = x440 + x76;
        B(0, 8) = x437;
        B(1, 0) = -x444 - x446;
        B(1, 1) = x447;
        B(1, 2) = -x444 + x446;
        B(1, 4) = x14*(-x101*x442 - x106*x443);
        B(1, 6) = -x444 + x445 - x72;
        B(1, 7) = x447;
        B(1, 8) = -x444 - x445 + x72;
        B(2, 0) = x450;
        B(2, 1) = x451;
        B(2, 2) = x452;
        B(2, 4) = x14*(-x160*x442 - x162*x443);
        B(2, 6) = x452;
        B(2, 7) = x451;
        B(2, 8) = x450;
        B(3, 0) = x455;
        B(3, 1) = x456;
        B(3, 2) = x457;
        B(3, 3) = x458;
        B(3, 4) = -x14*(x224*x459 + x239*x460 + x248*x461 + x252*x462 + x262*x463 + x266*x464) + 8*x330 + 8*x331;
        B(3, 5) = x458;
        B(3, 6) = x457;
        B(3, 7) = x456;
        B(3, 8) = x455;
        B(4, 0) = x467;
        B(4, 1) = x468;
        B(4, 2) = x469;
        B(4, 4) = x14*(-x223*x460 - x239*x459 - x247*x462 - x252*x461 - x261*x464 - x266*x463);
        B(4, 6) = x469;
        B(4, 7) = x468;
        B(4, 8) = x467;
        B(5, 0) = -x470 - x473;
        B(5, 1) = 4*a_0*x14*x406 - x476;
        B(5, 2) = x338 + x471 - x472;
        B(5, 3) = -x477;
        B(5, 4) = x14*(-x361*x442 - x363*x443);
        B(5, 5) = x477;
        B(5, 6) = -x338 + x471 - x472;
        B(5, 7) = x406*x439 + x476;
        B(5, 8) = -x338 - x473;

        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = -x479 - x480 - x81;
        C(0, 1) = x67;
        C(0, 2) = -x480 - x481 - x88;
        C(0, 3) = x66 + x70 - x73;
        C(0, 4) = x424 - x75 + x78;
        C(0, 5) = 2*a_0*x8 - x66 - x73;
        C(0, 6) = -x12 - x16 - x481 - x68;
        C(0, 7) = x85;
        C(0, 8) = -x479 - x68 - x80 - x87;
        C(1, 0) = x128 + x155 + x483;
        C(1, 1) = x145;
        C(1, 2) = x147 + x157 + x483;
        C(1, 3) = x144 + x148 - x149;
        C(1, 4) = -x151 + x152;
        C(1, 5) = 2*a_0*x107 - x144 - x149;
        C(1, 6) = x139 + x147 + x484;
        C(1, 7) = x156;
        C(1, 8) = x128 + x146 + x484;
        C(2, 0) = -x198 - x486 - x487;
        C(2, 1) = x181 - x183 - x184 + x199;
        C(2, 2) = -x201 - x487 - x488;
        C(2, 3) = x189 + x194 + x489;
        C(2, 4) = -x191 + x192;
        C(2, 5) = -x190 - x193 - x489;
        C(2, 6) = -x170 - x172 - x488 - x490;
        C(2, 7) = x199 + x489;
        C(2, 8) = -x197 - x200 - x486 - x490;
        C(3, 0) = -x274 - x279 - x284 - x491;
        C(3, 1) = x300;
        C(3, 2) = x275 + x301 + x303 + x491;
        C(3, 3) = x304 - x305 + x307;
        C(3, 4) = -x308 + x309;
        C(3, 5) = 2*a_0*x267 - x305 - x307;
        C(3, 6) = -x270 - x272 - x310 - x492;
        C(3, 7) = x314;
        C(3, 8) = -x312 - x315 - x492;
        C(4, 0) = -x329 - x341 - x493;
        C(4, 1) = 4*a_0*d_1*x326 - x338 - x346;
        C(4, 2) = x333 + x342 + x493;
        C(4, 3) = x343 - x344 + x346;
        C(4, 4) = -x347 + x348;
        C(4, 5) = 2*a_0*x316 - x344 - x346;
        C(4, 6) = -x319 - x321 - x349 - x494;
        C(4, 7) = x339 + x346 + x470;
        C(4, 8) = -x350 - x353 - x494;
        C(5, 0) = x431 + x434 + x496;
        C(5, 1) = x419;
        C(5, 2) = x428 + x433 + x496;
        C(5, 3) = x418 - x422 + x425;
        C(5, 4) = x426 - x427;
        C(5, 5) = 4*a_0*d_1*x6 + 2*a_0*x367 - x423;
        C(5, 6) = x412 + x428 + x497;
        C(5, 7) = x432;
        C(5, 8) = x420 + x434 + x497;

        // Invoke the solver
        std::array<double, 48> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);

        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[4] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor();
    // Finish code for general_6dof solution node 0

    // Code for non-branch dispatcher node 2
    // Actually, there is no code

    // Code for explicit solution node 3, solved variable is th_2
    auto ExplicitSolutionNode_node_3_solve_th_2_processor = [&]() -> void
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
            const double th_3 = this_solution[4];
            const bool condition_0 = ((2*std::fabs(a_0*d_2*std::sin(alpha_3)*std::sin(th_3)) >= zero_tolerance) || (std::fabs(2*a_0*d_1 + 2*a_0*d_2*std::cos(alpha_3)) >= zero_tolerance) || (std::fabs(-std::pow(a_0, 2) - std::pow(d_1, 2) - 2*d_1*d_2*std::cos(alpha_3) - std::pow(d_2, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 2*a_0;
                const double x1 = d_2*std::cos(alpha_3);
                const double x2 = -d_1*x0 - x0*x1;
                const double x3 = std::sin(alpha_3);
                const double x4 = std::sin(th_3);
                const double x5 = std::atan2(x2, d_2*x0*x3*x4);
                const double x6 = std::pow(a_0, 2);
                const double x7 = std::pow(d_2, 2);
                const double x8 = -std::pow(d_1, 2) - 2*d_1*x1 + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x6 - x7;
                const double x9 = safe_sqrt(std::pow(x2, 2) + 4*std::pow(x3, 2)*std::pow(x4, 2)*x6*x7 - std::pow(x8, 2));
                RawSolution solution_0(this_solution);
                solution_0[3] = x5 + std::atan2(x9, x8);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = x5 + std::atan2(-x9, x8);
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
    ExplicitSolutionNode_node_3_solve_th_2_processor();
    // Finish code for explicit solution node 2

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
            const double th_3 = this_solution[4];
            const bool checked_result = ((std::fabs(Px - d_4*r_13) <= 9.9999999999999995e-7) && (std::fabs(Py - d_4*r_23) <= 9.9999999999999995e-7) && (std::fabs(d_2*std::sin(alpha_3)*std::cos(th_3)) <= 9.9999999999999995e-7));
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
            const double th_3 = this_solution[4];
            const bool condition_0 = ((std::fabs(d_2*std::sin(alpha_3)*std::cos(th_3)) >= zero_tolerance) || (std::fabs(Px - d_4*r_13) >= zero_tolerance) || (std::fabs(Py - d_4*r_23) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = Px - d_4*r_13;
                const double x1 = -Py + d_4*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::sin(alpha_3);
                const double x4 = std::cos(th_3);
                const double x5 = safe_sqrt(-std::pow(d_2, 2)*std::pow(x3, 2)*std::pow(x4, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x6 = d_2*x3*x4;
                RawSolution solution_0(this_solution);
                solution_0[1] = x2 + std::atan2(x5, x6);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx_0);
                const double tmp_sol_value = x2 + std::atan2(-x5, x6);
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
    ExplicitSolutionNode_node_5_solve_th_0_processor();
    // Finish code for explicit solution node 5

    // Code for equation all-zero dispatcher node 6
    auto EquationAllZeroDispatcherNode_node_6_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool checked_result = ((std::fabs(Pz - d_4*r_33) <= 9.9999999999999995e-7) && (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_4*r_13*std::cos(th_0) - d_4*r_23*std::sin(th_0)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(7, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_6_processor();
    // Finish code for equation all-zero dispatcher node 6

    // Code for explicit solution node 7, solved variable is th_1
    auto ExplicitSolutionNode_node_7_solve_th_1_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[4];
            const bool condition_0 = ((std::fabs(Pz - d_4*r_33) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_4*r_13*std::cos(th_0) - d_4*r_23*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_4*r_33;
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(alpha_3);
                const double x3 = std::cos(th_2);
                const double x4 = d_2*std::sin(alpha_3)*std::sin(th_3);
                const double x5 = -a_0 + d_1*x1 + d_2*x1*x2 - x3*x4;
                const double x6 = d_1*x3 + d_2*x2*x3 + x1*x4;
                const double x7 = std::cos(th_0);
                const double x8 = std::sin(th_0);
                const double x9 = -Px*x7 - Py*x8 + d_4*r_13*x7 + d_4*r_23*x8;
                const double tmp_sol_value = std::atan2(x0*x5 - x6*x9, x0*x6 + x5*x9);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor();
    // Finish code for explicit solution node 7

    // Code for non-branch dispatcher node 8
    // Actually, there is no code

    // Code for explicit solution node 9, solved variable is negative_th_2_positive_th_1__soa
    auto ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(8);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(8);
        if (!this_input_valid)
            return;

        // The solution of non-root node 9
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = th_1 - th_2;
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
    ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor();
    // Finish code for explicit solution node 8

    // Code for non-branch dispatcher node 10
    // Actually, there is no code

    // Code for explicit solution node 11, solved variable is th_4
    auto ExplicitSolutionNode_node_11_solve_th_4_processor = [&]() -> void
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const double th_0 = this_solution[1];
            const double th_3 = this_solution[4];
            const bool condition_0 = ((std::fabs(std::sin(alpha_5)) >= zero_tolerance) || (std::fabs(std::sin(alpha_3)*std::sin(alpha_5)) >= zero_tolerance) || (std::fabs(r_13*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_0) + r_23*std::sin(negative_th_2_positive_th_1__soa)*std::sin(th_0) + r_33*std::cos(negative_th_2_positive_th_1__soa) - std::cos(alpha_3)*std::cos(alpha_5)) >= zero_tolerance) || (std::fabs(-r_13*(-std::sin(th_0)*std::sin(th_3) + std::cos(negative_th_2_positive_th_1__soa)*std::cos(th_0)*std::cos(th_3)) - r_23*(std::sin(th_0)*std::cos(negative_th_2_positive_th_1__soa)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_3)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(alpha_5);
                const double x1 = std::sin(negative_th_2_positive_th_1__soa);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_0);
                const double x6 = std::cos(negative_th_2_positive_th_1__soa);
                const double tmp_sol_value = std::atan2(x0*(r_13*(x2*x4*x6 - x3*x5) + r_23*(x2*x5*x6 + x3*x4) - r_33*x1*x2), x0*(-r_13*x1*x4 - r_23*x1*x5 - r_33*x6 + std::cos(alpha_3)*std::cos(alpha_5))/std::sin(alpha_3));
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_4_processor();
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const bool condition_0 = ((std::fabs(d_2*std::sin(alpha_5)) >= zero_tolerance) || (std::fabs(-a_0*(r_31*std::sin(th_1) - (r_11*std::cos(th_0) + r_21*std::sin(th_0))*std::cos(th_1)) + d_1*(r_11*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_0) + r_21*std::sin(negative_th_2_positive_th_1__soa)*std::sin(th_0) + r_31*std::cos(negative_th_2_positive_th_1__soa)) + inv_Px) >= zero_tolerance) || (std::fabs(-a_0*(r_32*std::sin(th_1) - (r_12*std::cos(th_0) + r_22*std::sin(th_0))*std::cos(th_1)) + d_1*(r_12*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_0) + r_22*std::sin(negative_th_2_positive_th_1__soa)*std::sin(th_0) + r_32*std::cos(negative_th_2_positive_th_1__soa)) + inv_Py) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::cos(negative_th_2_positive_th_1__soa);
                const double x1 = std::sin(negative_th_2_positive_th_1__soa);
                const double x2 = std::cos(th_0);
                const double x3 = r_11*x2;
                const double x4 = std::sin(th_0);
                const double x5 = r_21*x4;
                const double x6 = std::sin(th_1);
                const double x7 = std::cos(th_1);
                const double x8 = 1/(d_2*std::sin(alpha_5));
                const double x9 = r_12*x2;
                const double x10 = r_22*x4;
                const double tmp_sol_value = std::atan2(x8*(-a_0*(-r_31*x6 + x7*(x3 + x5)) - d_1*(r_31*x0 + x1*x3 + x1*x5) - inv_Px), x8*(-a_0*(-r_32*x6 + x7*(x10 + x9)) - d_1*(r_32*x0 + x1*x10 + x1*x9) - inv_Py));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
        const double value_at_0 = raw_ik_out_i[1];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[2];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[3];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[4];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[5];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[6];  // th_5
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
        ik_out_i[5] = addRevoluteOffset(ik_out_i[5], -th_5_offset_original2raw);
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

}; // struct yaskawa_mpx3500_ik

// Code below for debug
void test_ik_solve_yaskawa_mpx3500()
{
    std::array<double, yaskawa_mpx3500_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = yaskawa_mpx3500_ik::computeFK(theta);
    auto ik_output = yaskawa_mpx3500_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = yaskawa_mpx3500_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_yaskawa_mpx3500();
    return 0;
}
