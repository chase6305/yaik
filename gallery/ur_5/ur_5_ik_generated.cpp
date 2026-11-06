#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct ur_5_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 704;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793, -6.283185307179586, -3.141592653589793}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 0.0, 3.141592653589793}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 32;
static constexpr int intermediate_solution_size = 9;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_1 = 0.425;
static constexpr double a_3 = 0.39225;
static constexpr double d_0 = 0.220941;
static constexpr double d_2 = -0.1719;
static constexpr double d_4 = 0.060108999999999996;
static constexpr double d_5 = 0.09465;
static constexpr double d_6 = 0.0823;
static constexpr double pre_transform_special_symbol_23 = 0.089159;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -0.0;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = -0.0;
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
    ee_transformed(0, 0) = 1.0*r_11;
    ee_transformed(0, 1) = 1.0*r_12 + 6.1232339957367697e-17*r_13;
    ee_transformed(0, 2) = 6.1232339957367697e-17*r_12 - 1.0*r_13;
    ee_transformed(0, 3) = -1.0*Px;
    ee_transformed(1, 0) = 1.0*r_21;
    ee_transformed(1, 1) = 1.0*r_22 + 6.1232339957367697e-17*r_23;
    ee_transformed(1, 2) = 6.1232339957367697e-17*r_22 - 1.0*r_23;
    ee_transformed(1, 3) = -1.0*Py;
    ee_transformed(2, 0) = -1.0*r_31;
    ee_transformed(2, 1) = -1.0*r_32 - 6.1232339957367697e-17*r_33;
    ee_transformed(2, 2) = -6.1232339957367697e-17*r_32 + 1.0*r_33;
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
    ee_transformed(0, 0) = 1.0*r_11;
    ee_transformed(0, 1) = 1.0*r_12 + 6.1232339957367697e-17*r_13;
    ee_transformed(0, 2) = 6.1232339957367697e-17*r_12 - 1.0*r_13;
    ee_transformed(0, 3) = -1.0*Px;
    ee_transformed(1, 0) = 1.0*r_21;
    ee_transformed(1, 1) = 1.0*r_22 + 6.1232339957367697e-17*r_23;
    ee_transformed(1, 2) = 6.1232339957367697e-17*r_22 - 1.0*r_23;
    ee_transformed(1, 3) = -1.0*Py;
    ee_transformed(2, 0) = -1.0*r_31;
    ee_transformed(2, 1) = -1.0*r_32 - 6.1232339957367697e-17*r_33;
    ee_transformed(2, 2) = -6.1232339957367697e-17*r_32 + 1.0*r_33;
    ee_transformed(2, 3) = 1.0*Pz + 1.0*pre_transform_special_symbol_23;
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
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = std::sin(th_5);
    const double x1 = std::sin(th_3);
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::sin(th_2);
    const double x5 = x3*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::cos(th_2);
    const double x8 = -x2*x5 + x2*x6*x7;
    const double x9 = std::cos(th_3);
    const double x10 = x3*x7;
    const double x11 = x4*x6;
    const double x12 = -x10*x2 - x11*x2;
    const double x13 = -x1*x8 + x12*x9;
    const double x14 = std::cos(th_5);
    const double x15 = std::sin(th_0);
    const double x16 = std::sin(th_4);
    const double x17 = std::cos(th_4);
    const double x18 = x1*x12 + x8*x9;
    const double x19 = x15*x16 + x17*x18;
    const double x20 = x15*x17 - x16*x18;
    const double x21 = a_1*x6;
    const double x22 = -x15*x5 + x15*x6*x7;
    const double x23 = -x10*x15 - x11*x15;
    const double x24 = -x1*x22 + x23*x9;
    const double x25 = x1*x23 + x22*x9;
    const double x26 = -x16*x2 + x17*x25;
    const double x27 = -x16*x25 - x17*x2;
    const double x28 = x5 - x6*x7;
    const double x29 = -x10 - x11;
    const double x30 = -x1*x29 + x28*x9;
    const double x31 = x1*x28 + x29*x9;
    const double x32 = x17*x31;
    const double x33 = x16*x31;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x13 + x14*x19;
    ee_pose_raw(0, 1) = -x0*x19 - x13*x14;
    ee_pose_raw(0, 2) = x20;
    ee_pose_raw(0, 3) = a_3*x8 - d_0*x15 - d_2*x15 - d_4*x15 + d_5*x13 + d_6*x20 + x2*x21;
    ee_pose_raw(1, 0) = -x0*x24 + x14*x26;
    ee_pose_raw(1, 1) = -x0*x26 - x14*x24;
    ee_pose_raw(1, 2) = x27;
    ee_pose_raw(1, 3) = a_3*x22 + d_0*x2 + d_2*x2 + d_4*x2 + d_5*x24 + d_6*x27 + x15*x21;
    ee_pose_raw(2, 0) = -x0*x30 + x14*x32;
    ee_pose_raw(2, 1) = -x0*x32 - x14*x30;
    ee_pose_raw(2, 2) = -x33;
    ee_pose_raw(2, 3) = -a_1*x3 + a_3*x29 + d_5*x30 - d_6*x33;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = std::sin(th_3);
    const double x2 = std::sin(th_1);
    const double x3 = std::sin(th_2);
    const double x4 = std::cos(th_0);
    const double x5 = 1.0*x4;
    const double x6 = x3*x5;
    const double x7 = std::cos(th_1);
    const double x8 = std::cos(th_2);
    const double x9 = x5*x8;
    const double x10 = x2*x6 - x7*x9;
    const double x11 = std::cos(th_3);
    const double x12 = x2*x9 + x6*x7;
    const double x13 = -x1*x10 + x11*x12;
    const double x14 = std::cos(th_4);
    const double x15 = std::sin(th_4);
    const double x16 = -x0*x14 - x15*(x1*x12 + x10*x11);
    const double x17 = -x5;
    const double x18 = x0*x3;
    const double x19 = x0*x8;
    const double x20 = x18*x2 - x19*x7;
    const double x21 = x18*x7 + x19*x2;
    const double x22 = -x1*x20 + x11*x21;
    const double x23 = 1.0*x14*x4 - x15*(x1*x21 + x11*x20);
    const double x24 = 1.0*x3;
    const double x25 = 1.0*x8;
    const double x26 = x2*x24 - x25*x7;
    const double x27 = -x2*x25 - x24*x7;
    const double x28 = -x1*x27 + x11*x26;
    const double x29 = x15*(x1*x26 + x11*x27);
    const double x30 = 1.0*a_1*x2;
    const double x31 = pre_transform_special_symbol_23 - x30;
    const double x32 = a_3*x27 + pre_transform_special_symbol_23 - x30;
    const double x33 = d_5*x28 + x32;
    const double x34 = a_1*x7;
    const double x35 = d_0*x5 + d_2*x5 + x0*x34;
    const double x36 = -a_3*x20 + d_4*x5 + x35;
    const double x37 = -d_5*x22 + x36;
    const double x38 = -x37;
    const double x39 = -d_6*x29 + x33;
    const double x40 = d_6*x23 - x37;
    const double x41 = d_0*x0 + d_2*x0 - x34*x5;
    const double x42 = a_3*x10 + d_4*x0 + x41;
    const double x43 = d_5*x13 + x42;
    const double x44 = d_6*x16 + x43;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x0;
    jacobian(0, 4) = x13;
    jacobian(0, 5) = x16;
    jacobian(1, 1) = x17;
    jacobian(1, 2) = x17;
    jacobian(1, 3) = x17;
    jacobian(1, 4) = x22;
    jacobian(1, 5) = x23;
    jacobian(2, 0) = 1.0;
    jacobian(2, 4) = x28;
    jacobian(2, 5) = -x29;
    jacobian(3, 1) = pre_transform_special_symbol_23*x5;
    jacobian(3, 2) = x31*x5;
    jacobian(3, 3) = x32*x5;
    jacobian(3, 4) = -x22*x33 + x28*x38;
    jacobian(3, 5) = -x23*x39 - x29*x40;
    jacobian(4, 1) = pre_transform_special_symbol_23*x0;
    jacobian(4, 2) = x0*x31;
    jacobian(4, 3) = x0*x32;
    jacobian(4, 4) = x13*x33 - x28*x43;
    jacobian(4, 5) = x16*x39 + x29*x44;
    jacobian(5, 2) = x0*x35 - x41*x5;
    jacobian(5, 3) = x0*x36 - x42*x5;
    jacobian(5, 4) = -x13*x38 + x22*x43;
    jacobian(5, 5) = -x16*x40 + x23*x44;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = addRevoluteOffset(theta_input_original[4], th_4_offset_original2raw);
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;

    // Temp variable for efficiency
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = std::sin(th_3);
    const double x2 = std::sin(th_1);
    const double x3 = std::sin(th_2);
    const double x4 = std::cos(th_0);
    const double x5 = 1.0*x4;
    const double x6 = x3*x5;
    const double x7 = std::cos(th_1);
    const double x8 = std::cos(th_2);
    const double x9 = x5*x8;
    const double x10 = x2*x6 - x7*x9;
    const double x11 = std::cos(th_3);
    const double x12 = x2*x9 + x6*x7;
    const double x13 = std::cos(th_4);
    const double x14 = std::sin(th_4);
    const double x15 = -x5;
    const double x16 = x0*x3;
    const double x17 = x0*x8;
    const double x18 = x16*x2 - x17*x7;
    const double x19 = x16*x7 + x17*x2;
    const double x20 = 1.0*x3;
    const double x21 = 1.0*x8;
    const double x22 = x2*x20 - x21*x7;
    const double x23 = -x2*x21 - x20*x7;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x0;
    jacobian(0, 4) = -x1*x10 + x11*x12;
    jacobian(0, 5) = -x0*x13 - x14*(x1*x12 + x10*x11);
    jacobian(1, 1) = x15;
    jacobian(1, 2) = x15;
    jacobian(1, 3) = x15;
    jacobian(1, 4) = -x1*x18 + x11*x19;
    jacobian(1, 5) = 1.0*x13*x4 - x14*(x1*x19 + x11*x18);
    jacobian(2, 0) = 1.0;
    jacobian(2, 4) = -x1*x23 + x11*x22;
    jacobian(2, 5) = -x14*(x1*x22 + x11*x23);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
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
    const double x6 = a_1*x5;
    const double x7 = pre_transform_special_symbol_23 - x6;
    const double x8 = std::cos(th_2);
    const double x9 = x5*x8;
    const double x10 = std::sin(th_2);
    const double x11 = std::cos(th_1);
    const double x12 = 1.0*x11;
    const double x13 = x10*x12;
    const double x14 = -x13 - x9;
    const double x15 = a_3*x14 + pre_transform_special_symbol_23 - x6;
    const double x16 = std::cos(th_3);
    const double x17 = x10*x5;
    const double x18 = x12*x8;
    const double x19 = x17 - x18;
    const double x20 = std::sin(th_3);
    const double x21 = -x14*x20 + x16*x19;
    const double x22 = std::sin(th_0);
    const double x23 = x17*x22 - x18*x22;
    const double x24 = x13*x22 + x22*x9;
    const double x25 = x16*x24 - x20*x23;
    const double x26 = d_5*x21 + x15;
    const double x27 = a_1*x12*x22 + d_0*x2 + d_2*x2;
    const double x28 = -a_3*x23 + d_4*x2 + x27;
    const double x29 = -d_5*x25 + x28;
    const double x30 = -x29;
    const double x31 = std::sin(th_4);
    const double x32 = x14*x16 + x19*x20;
    const double x33 = x31*x32;
    const double x34 = std::cos(th_4);
    const double x35 = 1.0*x1*x34 - x31*(x16*x23 + x20*x24);
    const double x36 = -d_6*x33 + x26;
    const double x37 = d_6*x35 - x29;
    const double x38 = 1.0*p_on_ee_x;
    const double x39 = 1.0*x22;
    const double x40 = p_on_ee_z*x39;
    const double x41 = x10*x2;
    const double x42 = x2*x8;
    const double x43 = -x11*x42 + x4*x41;
    const double x44 = x11*x41 + x4*x42;
    const double x45 = x16*x44 - x20*x43;
    const double x46 = -a_1*x11*x2 + d_0*x39 + d_2*x39;
    const double x47 = a_3*x43 + d_4*x39 + x46;
    const double x48 = d_5*x45 + x47;
    const double x49 = -x31*(x16*x43 + x20*x44) - x34*x39;
    const double x50 = d_6*x49 + x48;
    const double x51 = x0*x22 + x1*x38;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = 1.0*pre_transform_special_symbol_23*x1 - x3;
    jacobian(0, 2) = 1.0*x1*x7 - x3;
    jacobian(0, 3) = 1.0*x1*x15 - x3;
    jacobian(0, 4) = -p_on_ee_y*x21 + p_on_ee_z*x25 + x21*x30 - x25*x26;
    jacobian(0, 5) = p_on_ee_y*x33 + p_on_ee_z*x35 - x33*x37 - x35*x36;
    jacobian(1, 0) = x38;
    jacobian(1, 1) = 1.0*pre_transform_special_symbol_23*x22 - x40;
    jacobian(1, 2) = 1.0*x22*x7 - x40;
    jacobian(1, 3) = 1.0*x15*x22 - x40;
    jacobian(1, 4) = p_on_ee_x*x21 - p_on_ee_z*x45 - x21*x48 + x26*x45;
    jacobian(1, 5) = -p_on_ee_x*x33 - p_on_ee_z*x49 + x31*x32*x50 + x36*x49;
    jacobian(2, 1) = x51;
    jacobian(2, 2) = -x2*x46 + x27*x39 + x51;
    jacobian(2, 3) = -x2*x47 + x28*x39 + x51;
    jacobian(2, 4) = -p_on_ee_x*x25 + p_on_ee_y*x45 + x25*x48 - x30*x45;
    jacobian(2, 5) = -p_on_ee_x*x35 + p_on_ee_y*x49 + x35*x50 - x37*x49;
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
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 4, 176, 176, 352, 352, 352, 352, 4, 176, 176, 352, 352, 352, 352}};
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

    // Code for explicit solution node 1, solved variable is th_0
    auto ExplicitSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;

        // The explicit solution of root node
        const bool condition_0 = ((std::fabs(d_0 + d_2 + d_4) >= zero_tolerance) || (std::fabs(Px - d_6*r_13) >= zero_tolerance) || (std::fabs(Py - d_6*r_23) >= zero_tolerance));
        if (condition_0)
        {
            const double x0 = Px - d_6*r_13;
            const double x1 = -Py + d_6*r_23;
            const double x2 = std::atan2(x0, x1);
            const double x3 = -d_0 - d_2 - d_4;
            const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
            auto solution_0 = make_raw_solution();
            solution_0[0] = x2 + std::atan2(x4, x3);
            int appended_idx_0 = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx_0);

            auto solution_1 = make_raw_solution();
            solution_1[0] = x2 + std::atan2(-x4, x3);
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx_1);

        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor();
    // Finish code for explicit solution node 0

    // Code for non-branch dispatcher node 2
    // Actually, there is no code

    // Code for explicit solution node 3, solved variable is th_4
    auto ExplicitSolutionNode_node_3_solve_th_4_processor = [&]() -> void
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
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = safe_acos(r_13*std::sin(th_0) - r_23*std::cos(th_0));
                RawSolution solution_0(this_solution);
                solution_0[7] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = -x0;
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
    ExplicitSolutionNode_node_3_solve_th_4_processor();
    // Finish code for explicit solution node 2

    // Code for solved_variable dispatcher node 4
    auto SolvedVariableDispatcherNode_node_4_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            bool forwarded_to_child = false;
            const double th_4 = this_solution[7];

            const bool degenerate_valid_0 = std::fabs(th_4) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(18, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_4 - M_PI) <= 9.9999999999999995e-7;
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
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_4_processor();
    // Finish code for solved_variable dispatcher node 4

    // Code for explicit solution node 25, solved variable is th_5
    auto ExplicitSolutionNode_node_25_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(25);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(25);
        if (!this_input_valid)
            return;
        for (int family_prefix = 0; family_prefix < this_node_input_index.size(); ++family_prefix)
        {
            const int queue_index = this_node_input_index[family_prefix];
            if (!solution_queue.is_solution_valid(queue_index)) continue;
            const auto& this_solution = solution_queue.get_solution(queue_index);
            const double th_0 = this_solution[0];
            std::vector<double> family_values;
            if (std::pow(r_31, 2) + std::pow(r_11*std::cos(th_0) + r_21*std::sin(th_0), 2) > 9.9999999999999998e-13)
            {
                family_values.push_back(-3.14159265358979);
                family_values.push_back(-2.9452431127404299);
                family_values.push_back(-2.7488935718910699);
                family_values.push_back(-2.5525440310417098);
                family_values.push_back(-2.35619449019234);
                family_values.push_back(-2.1598449493429799);
                family_values.push_back(-1.96349540849362);
                family_values.push_back(-1.76714586764426);
                family_values.push_back(-1.5707963267949001);
                family_values.push_back(-1.37444678594553);
                family_values.push_back(-1.17809724509617);
                family_values.push_back(-0.98174770424681101);
                family_values.push_back(-0.78539816339744795);
                family_values.push_back(-0.58904862254808599);
                family_values.push_back(-0.39269908169872397);
                family_values.push_back(-0.19634954084936199);
                family_values.push_back(0.0);
                family_values.push_back(0.19634954084936199);
                family_values.push_back(0.39269908169872397);
                family_values.push_back(0.58904862254808599);
                family_values.push_back(0.78539816339744795);
                family_values.push_back(0.98174770424681101);
                family_values.push_back(1.17809724509617);
                family_values.push_back(1.37444678594553);
                family_values.push_back(1.5707963267949001);
                family_values.push_back(1.76714586764426);
                family_values.push_back(1.96349540849362);
                family_values.push_back(2.1598449493429799);
                family_values.push_back(2.35619449019234);
                family_values.push_back(2.5525440310417098);
                family_values.push_back(2.7488935718910699);
                family_values.push_back(2.9452431127404299);
                family_values.push_back(3.14159265358979);
            }
            if (std::pow(r_31, 2) + std::pow(r_11*std::cos(th_0) + r_21*std::sin(th_0), 2) > 9.9999999999999998e-13)
            {
                const double x0 = -Pz + d_6*r_33;
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                const double x3 = r_11*x1 + r_21*x2;
                const double x4 = std::pow(std::pow(r_31, 2) + std::pow(x3, 2), -1.0/2.0);
                const double x5 = -x3*x4;
                const double x6 = x1*(Px - d_6*r_13) + x2*(Py - d_6*r_23);
                const double x7 = r_31*x4;
                const double x8 = 2*d_5;
                const double x9 = x8*(-x0*x5 + x6*x7);
                const double x10 = -x8*(x0*x7 + x5*x6);
                const double x11 = std::pow(d_5, 2) + std::pow(x0, 2) + std::pow(x6, 2);
                family_values = ::yaik_cpp::trigFamilyRepresentatives({{{x9, x10, x11 - std::pow(a_1 - a_3, 2)}}, {{x9, x10, x11 - std::pow(a_1 + a_3, 2)}}}, -3.141592653589793, 3.141592653589793, family_values);
            }
            for (std::size_t family_index = 0; family_index < family_values.size(); ++family_index)
            {
                auto candidate = this_solution;
                candidate[8] = family_values[family_index];
                int candidate_index;
                if (family_index+1 == family_values.size())
                {
                    solution_queue.get_solution(queue_index) = candidate;
                    candidate_index = queue_index;
                }
                else candidate_index = append_solution_to_queue(candidate);
                add_input_index_to(26, candidate_index);
            }
            if (family_values.empty()) solution_queue.solutions_validity[queue_index] = false;
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_5_processor();
    // Finish code for explicit solution node 25

    // Code for non-branch dispatcher node 26
    // Actually, there is no code

    // Code for explicit solution node 27, solved variable is th_2
    auto ExplicitSolutionNode_node_27_solve_th_2_processor = [&]() -> void
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
            const double th_5 = this_solution[8];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = th_5 - std::atan2(-r_31, r_11*x0 + r_21*x1);
                const double x3 = safe_acos((1.0/2.0)*(-std::pow(a_1, 2) - std::pow(a_3, 2) + std::pow(-Pz + d_5*std::cos(x2) + d_6*r_33, 2) + std::pow(d_5*std::sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23), 2))/(a_1*a_3));
                RawSolution solution_0(this_solution);
                solution_0[4] = x3;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(28, appended_idx_0);
                const double tmp_sol_value = -x3;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(28, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_2_processor();
    // Finish code for explicit solution node 26

    // Code for non-branch dispatcher node 28
    // Actually, there is no code

    // Code for explicit solution node 29, solved variable is th_1
    auto ExplicitSolutionNode_node_29_solve_th_1_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[8];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = th_5 - std::atan2(-r_31, r_11*x0 + r_21*x1);
                const double tmp_sol_value = -std::atan2(a_3*std::sin(th_2), a_1 + a_3*std::cos(th_2)) + std::atan2(-Pz + d_5*std::cos(x2) + d_6*r_33, d_5*std::sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23));
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
    // Finish code for explicit solution node 28

    // Code for non-branch dispatcher node 30
    // Actually, there is no code

    // Code for explicit solution node 31, solved variable is th_3
    auto ExplicitSolutionNode_node_31_solve_th_3_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[8];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = -th_1 - th_2 - th_5 + std::atan2(-r_31, r_11*std::cos(th_0) + r_21*std::sin(th_0)) - M_PI;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_3_processor();
    // Finish code for explicit solution node 30

    // Code for explicit solution node 18, solved variable is th_5
    auto ExplicitSolutionNode_node_18_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
        if (!this_input_valid)
            return;
        for (int family_prefix = 0; family_prefix < this_node_input_index.size(); ++family_prefix)
        {
            const int queue_index = this_node_input_index[family_prefix];
            if (!solution_queue.is_solution_valid(queue_index)) continue;
            const auto& this_solution = solution_queue.get_solution(queue_index);
            const double th_0 = this_solution[0];
            std::vector<double> family_values;
            if (std::pow(r_31, 2) + std::pow(r_11*std::cos(th_0) + r_21*std::sin(th_0), 2) > 9.9999999999999998e-13)
            {
                family_values.push_back(-3.14159265358979);
                family_values.push_back(-2.9452431127404299);
                family_values.push_back(-2.7488935718910699);
                family_values.push_back(-2.5525440310417098);
                family_values.push_back(-2.35619449019234);
                family_values.push_back(-2.1598449493429799);
                family_values.push_back(-1.96349540849362);
                family_values.push_back(-1.76714586764426);
                family_values.push_back(-1.5707963267949001);
                family_values.push_back(-1.37444678594553);
                family_values.push_back(-1.17809724509617);
                family_values.push_back(-0.98174770424681101);
                family_values.push_back(-0.78539816339744795);
                family_values.push_back(-0.58904862254808599);
                family_values.push_back(-0.39269908169872397);
                family_values.push_back(-0.19634954084936199);
                family_values.push_back(0.0);
                family_values.push_back(0.19634954084936199);
                family_values.push_back(0.39269908169872397);
                family_values.push_back(0.58904862254808599);
                family_values.push_back(0.78539816339744795);
                family_values.push_back(0.98174770424681101);
                family_values.push_back(1.17809724509617);
                family_values.push_back(1.37444678594553);
                family_values.push_back(1.5707963267949001);
                family_values.push_back(1.76714586764426);
                family_values.push_back(1.96349540849362);
                family_values.push_back(2.1598449493429799);
                family_values.push_back(2.35619449019234);
                family_values.push_back(2.5525440310417098);
                family_values.push_back(2.7488935718910699);
                family_values.push_back(2.9452431127404299);
                family_values.push_back(3.14159265358979);
            }
            if (std::pow(r_31, 2) + std::pow(r_11*std::cos(th_0) + r_21*std::sin(th_0), 2) > 9.9999999999999998e-13)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = r_11*x0 + r_21*x1;
                const double x3 = std::pow(std::pow(r_31, 2) + std::pow(x2, 2), -1.0/2.0);
                const double x4 = -Pz + d_6*r_33;
                const double x5 = x3*x4;
                const double x6 = x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23);
                const double x7 = 2*d_5;
                const double x8 = x7*(-r_31*x3*x6 - x2*x5);
                const double x9 = x7*(-r_31*x5 + x2*x3*x6);
                const double x10 = std::pow(d_5, 2) + std::pow(x4, 2) + std::pow(x6, 2);
                family_values = ::yaik_cpp::trigFamilyRepresentatives({{{x8, x9, x10 - std::pow(a_1 - a_3, 2)}}, {{x8, x9, x10 - std::pow(a_1 + a_3, 2)}}}, -3.141592653589793, 3.141592653589793, family_values);
            }
            for (std::size_t family_index = 0; family_index < family_values.size(); ++family_index)
            {
                auto candidate = this_solution;
                candidate[8] = family_values[family_index];
                int candidate_index;
                if (family_index+1 == family_values.size())
                {
                    solution_queue.get_solution(queue_index) = candidate;
                    candidate_index = queue_index;
                }
                else candidate_index = append_solution_to_queue(candidate);
                add_input_index_to(19, candidate_index);
            }
            if (family_values.empty()) solution_queue.solutions_validity[queue_index] = false;
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_5_processor();
    // Finish code for explicit solution node 18

    // Code for non-branch dispatcher node 19
    // Actually, there is no code

    // Code for explicit solution node 20, solved variable is th_2
    auto ExplicitSolutionNode_node_20_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(19);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(19);
        if (!this_input_valid)
            return;

        // The solution of non-root node 20
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[8];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = th_5 + std::atan2(-r_31, r_11*x0 + r_21*x1);
                const double x3 = safe_acos((1.0/2.0)*(-std::pow(a_1, 2) - std::pow(a_3, 2) + std::pow(-Pz - d_5*std::cos(x2) + d_6*r_33, 2) + std::pow(d_5*std::sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23), 2))/(a_1*a_3));
                RawSolution solution_0(this_solution);
                solution_0[4] = x3;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(21, appended_idx_0);
                const double tmp_sol_value = -x3;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(21, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_2_processor();
    // Finish code for explicit solution node 19

    // Code for non-branch dispatcher node 21
    // Actually, there is no code

    // Code for explicit solution node 22, solved variable is th_1
    auto ExplicitSolutionNode_node_22_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(21);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(21);
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[8];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = th_5 + std::atan2(-r_31, r_11*x0 + r_21*x1);
                const double tmp_sol_value = -std::atan2(a_3*std::sin(th_2), a_1 + a_3*std::cos(th_2)) + std::atan2(-Pz - d_5*std::cos(x2) + d_6*r_33, d_5*std::sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23));
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
    // Finish code for explicit solution node 21

    // Code for non-branch dispatcher node 23
    // Actually, there is no code

    // Code for explicit solution node 24, solved variable is th_3
    auto ExplicitSolutionNode_node_24_solve_th_3_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[8];
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = -th_1 - th_2 + th_5 + std::atan2(-r_31, r_11*std::cos(th_0) + r_21*std::sin(th_0));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_3_processor();
    // Finish code for explicit solution node 23

    // Code for explicit solution node 5, solved variable is th_1th_2th_3_soa
    auto ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_4 = this_solution[7];
            const bool condition_0 = ((std::fabs(r_33) >= zero_tolerance) || (std::fabs(std::sin(th_4)) >= zero_tolerance) || (std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_4);
                const double tmp_sol_value = std::atan2(r_33*x0, x0*(-r_13*std::cos(th_0) - r_23*std::sin(th_0)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor();
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
            const double th_0 = this_solution[0];
            const double th_4 = this_solution[7];
            const bool condition_0 = ((std::fabs(std::sin(th_4)) >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(x0*(-r_12*x1 + r_22*x2), x0*(r_11*x1 - r_21*x2));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
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

    // Code for non-branch dispatcher node 8
    // Actually, there is no code

    // Code for explicit solution node 9, solved variable is th_2
    auto ExplicitSolutionNode_node_9_solve_th_2_processor = [&]() -> void
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
            const double th_5 = this_solution[8];
            const bool condition_0 = ((a_1 != 0) && (a_3 != 0) && ((1.0/2.0)*std::fabs((std::pow(a_1, 2) + std::pow(a_3, 2) + std::pow(d_0, 2) + 2*d_0*d_2 + 2*d_0*d_4 + std::pow(d_2, 2) + 2*d_2*d_4 + std::pow(d_4, 2) - std::pow(d_5, 2) + 2*d_5*inv_Px*std::sin(th_5) + 2*d_5*inv_Py*std::cos(th_5) - std::pow(d_6, 2) - 2*d_6*inv_Pz - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2))/(a_1*a_3)) <= 70368744177665.0/70368744177664.0));
            if (condition_0)
            {
                const double x0 = 2*d_0;
                const double x1 = 2*d_5;
                const double x2 = safe_acos((1.0/2.0)*(-std::pow(a_1, 2) - std::pow(a_3, 2) - std::pow(d_0, 2) - std::pow(d_2, 2) - 2*d_2*d_4 - d_2*x0 - std::pow(d_4, 2) - d_4*x0 + std::pow(d_5, 2) + std::pow(d_6, 2) + 2*d_6*inv_Pz + std::pow(inv_Px, 2) - inv_Px*x1*std::sin(th_5) + std::pow(inv_Py, 2) - inv_Py*x1*std::cos(th_5) + std::pow(inv_Pz, 2))/(a_1*a_3));
                RawSolution solution_0(this_solution);
                solution_0[4] = x2;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx_0);
                const double tmp_sol_value = -x2;
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
    ExplicitSolutionNode_node_9_solve_th_2_processor();
    // Finish code for explicit solution node 8

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
            const double th_5 = this_solution[8];
            const bool checked_result = ((std::fabs(Pz + d_5*r_31*std::sin(th_5) + d_5*r_32*std::cos(th_5) - d_6*r_33) <= 9.9999999999999995e-7) && (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_5*r_11*std::sin(th_5)*std::cos(th_0) + d_5*r_12*std::cos(th_0)*std::cos(th_5) + d_5*r_21*std::sin(th_0)*std::sin(th_5) + d_5*r_22*std::sin(th_0)*std::cos(th_5) - d_6*r_13*std::cos(th_0) - d_6*r_23*std::sin(th_0)) <= 9.9999999999999995e-7));
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[8];
            const bool condition_0 = ((std::fabs(Pz + d_5*r_31*std::sin(th_5) + d_5*r_32*std::cos(th_5) - d_6*r_33) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_5*r_11*std::sin(th_5)*std::cos(th_0) + d_5*r_12*std::cos(th_0)*std::cos(th_5) + d_5*r_21*std::sin(th_0)*std::sin(th_5) + d_5*r_22*std::sin(th_0)*std::cos(th_5) - d_6*r_13*std::cos(th_0) - d_6*r_23*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = -a_1 - a_3*std::cos(th_2);
                const double x1 = d_5*std::sin(th_5);
                const double x2 = d_5*std::cos(th_5);
                const double x3 = Pz - d_6*r_33 + r_31*x1 + r_32*x2;
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_0);
                const double x6 = -Px*x4 - Py*x5 + d_6*r_13*x4 + d_6*r_23*x5 - r_11*x1*x4 - r_12*x2*x4 - r_21*x1*x5 - r_22*x2*x5;
                const double x7 = a_3*std::sin(th_2);
                const double tmp_sol_value = std::atan2(x0*x3 + x6*x7, x0*x6 - x3*x7);
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
    // Finish code for explicit solution node 11

    // Code for non-branch dispatcher node 12
    // Actually, there is no code

    // Code for explicit solution node 13, solved variable is th_2th_3_soa
    auto ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_1th_2th_3_soa = this_solution[3];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -th_1 + th_1th_2th_3_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor();
    // Finish code for explicit solution node 12

    // Code for non-branch dispatcher node 14
    // Actually, there is no code

    // Code for explicit solution node 15, solved variable is th_3
    auto ExplicitSolutionNode_node_15_solve_th_3_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            const double th_2th_3_soa = this_solution[5];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -th_2 + th_2th_3_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(16, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_3_processor();
    // Finish code for explicit solution node 14

    // Code for non-branch dispatcher node 16
    // Actually, there is no code

    // Code for explicit solution node 17, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[4];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = th_1 + th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 16

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
        const double value_at_3 = raw_ik_out_i[6];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[7];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[8];  // th_5
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
        ik_out_i[1] = ik_out_i[1] - th_1_offset_original2raw;
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw;
        ik_out_i[3] = ik_out_i[3] - th_3_offset_original2raw;
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
        ik_out_i[1] = ik_out_i[1] - th_1_offset_original2raw;
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw;
        ik_out_i[3] = ik_out_i[3] - th_3_offset_original2raw;
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
            ik_out_i[1] = ik_out_i[1] - th_1_offset_original2raw;
            ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw;
            ik_out_i[3] = ik_out_i[3] - th_3_offset_original2raw;
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
                    candidate[1] = candidate[1] - th_1_offset_original2raw;
                    candidate[2] = candidate[2] - th_2_offset_original2raw;
                    candidate[3] = candidate[3] - th_3_offset_original2raw;
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

}; // struct ur_5_ik

// Code below for debug
void test_ik_solve_ur_5()
{
    std::array<double, ur_5_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = ur_5_ik::computeFK(theta);
    auto ik_output = ur_5_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = ur_5_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_ur_5();
    return 0;
}
