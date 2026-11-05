#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct kawasaki_CP130L_ik {

// Constants for solver
static constexpr int robot_nq = 4;
static constexpr int max_n_solutions = 32;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -1.5707963267948966, -1.5707963267948966, -4.71238898038469}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 4.71238898038469, 4.71238898038469, 1.5707963267948963}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false}}; }
static constexpr int n_tree_nodes = 50;
static constexpr int intermediate_solution_size = 5;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.255;
static constexpr double a_1 = 0.26;
static constexpr double a_2 = 0.25;
static constexpr double d_3 = 0.24;
static constexpr double pre_transform_special_symbol_23 = 0.75;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = -1.5707963267948966;
static constexpr double th_3_offset_original2raw = 1.5707963267948968;

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
    ee_transformed(0, 0) = -1.0*r_11;
    ee_transformed(0, 1) = 1.0*r_12;
    ee_transformed(0, 2) = -1.0*r_13;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_21;
    ee_transformed(1, 1) = -1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_23;
    ee_transformed(1, 3) = -1.0*Py;
    ee_transformed(2, 0) = 1.0*r_31;
    ee_transformed(2, 1) = -1.0*r_32;
    ee_transformed(2, 2) = 1.0*r_33;
    ee_transformed(2, 3) = -1.0*Pz + 1.0*pre_transform_special_symbol_23;
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
    ee_transformed(0, 0) = -1.0*r_11;
    ee_transformed(0, 1) = 1.0*r_12;
    ee_transformed(0, 2) = -1.0*r_13;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_21;
    ee_transformed(1, 1) = -1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_23;
    ee_transformed(1, 3) = -1.0*Py;
    ee_transformed(2, 0) = 1.0*r_31;
    ee_transformed(2, 1) = -1.0*r_32;
    ee_transformed(2, 2) = 1.0*r_33;
    ee_transformed(2, 3) = -1.0*Pz + 1.0*pre_transform_special_symbol_23;
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

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_3);
    const double x2 = std::cos(th_3);
    const double x3 = std::cos(th_0);
    const double x4 = std::sin(th_1);
    const double x5 = std::sin(th_2);
    const double x6 = x4*x5;
    const double x7 = std::cos(th_1);
    const double x8 = std::cos(th_2);
    const double x9 = x7*x8;
    const double x10 = x3*x6 + x3*x9;
    const double x11 = x4*x8;
    const double x12 = x11*x3 - x3*x5*x7;
    const double x13 = a_1*x7;
    const double x14 = x0*x6 + x0*x9;
    const double x15 = x0*x11 - x0*x5*x7;
    const double x16 = x11 - x5*x7;
    const double x17 = x6 + x9;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x1 + x10*x2;
    ee_pose_raw(0, 1) = -x0*x2 - x1*x10;
    ee_pose_raw(0, 2) = -x12;
    ee_pose_raw(0, 3) = a_0*x3 + a_2*x10 - d_3*x12 + x13*x3;
    ee_pose_raw(1, 0) = x1*x3 + x14*x2;
    ee_pose_raw(1, 1) = -x1*x14 + x2*x3;
    ee_pose_raw(1, 2) = -x15;
    ee_pose_raw(1, 3) = a_0*x0 + a_2*x14 - d_3*x15 + x0*x13;
    ee_pose_raw(2, 0) = x16*x2;
    ee_pose_raw(2, 1) = -x1*x16;
    ee_pose_raw(2, 2) = x17;
    ee_pose_raw(2, 3) = a_1*x4 + a_2*x16 + d_3*x17;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = std::cos(th_0);
    const double x3 = 1.0*x2;
    const double x4 = std::cos(th_2);
    const double x5 = std::sin(th_1);
    const double x6 = x4*x5;
    const double x7 = std::sin(th_2);
    const double x8 = std::cos(th_1);
    const double x9 = -1.0*x2*x7*x8 + x3*x6;
    const double x10 = -x9;
    const double x11 = x1*x6 - x1*x7*x8;
    const double x12 = 1.0*x5;
    const double x13 = x12*x7 + 1.0*x4*x8;
    const double x14 = -x13;
    const double x15 = a_1*x12 - pre_transform_special_symbol_23;
    const double x16 = -x15;
    const double x17 = a_2*(-x12*x4 + 1.0*x7*x8) - d_3*x13 - x15;
    const double x18 = x5*x7;
    const double x19 = x1*x8;
    const double x20 = a_0*x1 + a_1*x19;
    const double x21 = a_2*(-x1*x18 - x19*x4) + d_3*x11 - x20;
    const double x22 = x3*x8;
    const double x23 = a_0*x3 + a_1*x22;
    const double x24 = a_2*(x18*x3 + x22*x4) - d_3*x9 + x23;
    const double x25 = 1.0*a_0;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = -x1;
    jacobian(0, 3) = x10;
    jacobian(1, 1) = x3;
    jacobian(1, 2) = -x3;
    jacobian(1, 3) = x11;
    jacobian(2, 0) = -1.0;
    jacobian(2, 3) = x14;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x3;
    jacobian(3, 2) = x16*x3;
    jacobian(3, 3) = -x11*x17 + x14*x21;
    jacobian(4, 1) = pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = -x1*x16;
    jacobian(4, 3) = x10*x17 - x14*x24;
    jacobian(5, 1) = std::pow(x0, 2)*x25 + std::pow(x2, 2)*x25;
    jacobian(5, 2) = -x1*x20 - x23*x3;
    jacobian(5, 3) = -x10*x21 + x11*x24;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);

    // Temp variable for efficiency
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = std::cos(th_0);
    const double x2 = 1.0*x1;
    const double x3 = std::cos(th_2);
    const double x4 = std::sin(th_1);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x0;
    jacobian(0, 3) = 1.0*x1*x6*x7 - x2*x5;
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x2;
    jacobian(1, 3) = x0*x5 - x0*x6*x7;
    jacobian(2, 0) = -1.0;
    jacobian(2, 3) = -1.0*x3*x7 - 1.0*x4*x6;
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = addRevoluteOffset(theta_input_original[1], th_1_offset_original2raw);
    const double th_2 = addRevoluteOffset(theta_input_original[2], th_2_offset_original2raw);
    const double th_3 = addRevoluteOffset(theta_input_original[3], th_3_offset_original2raw);
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
    const double x6 = a_1*x5 - pre_transform_special_symbol_23;
    const double x7 = -x6;
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = std::cos(th_2);
    const double x11 = std::cos(th_1);
    const double x12 = 1.0*x11;
    const double x13 = x10*x12;
    const double x14 = x13 + x9;
    const double x15 = -x14;
    const double x16 = std::sin(th_0);
    const double x17 = x10*x5;
    const double x18 = -x12*x16*x8 + x16*x17;
    const double x19 = a_2*(1.0*x11*x8 - x17) - d_3*x14 - x6;
    const double x20 = 1.0*x16;
    const double x21 = a_0*x20 + a_1*x12*x16;
    const double x22 = a_2*(-x13*x16 - x16*x9) + d_3*x18 - x21;
    const double x23 = 1.0*p_on_ee_x;
    const double x24 = p_on_ee_z*x20;
    const double x25 = x2*x4;
    const double x26 = -1.0*x1*x11*x8 + x10*x25;
    const double x27 = -x26;
    const double x28 = x11*x2;
    const double x29 = a_0*x2 + a_1*x28;
    const double x30 = a_2*(x10*x28 + x25*x8) - d_3*x26 + x29;
    const double x31 = x1*x23;
    const double x32 = x0*x16;
    const double x33 = 1.0*a_0;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = 1.0*x1*x7 - x3;
    jacobian(0, 3) = -p_on_ee_y*x15 + p_on_ee_z*x18 + x15*x22 - x18*x19;
    jacobian(1, 0) = -x23;
    jacobian(1, 1) = 1.0*pre_transform_special_symbol_23*x16 - x24;
    jacobian(1, 2) = -x20*x7 + x24;
    jacobian(1, 3) = p_on_ee_x*x15 - p_on_ee_z*x27 - x15*x30 + x19*x27;
    jacobian(2, 1) = std::pow(x1, 2)*x33 + std::pow(x16, 2)*x33 - x31 + x32;
    jacobian(2, 2) = -x2*x29 - x20*x21 + x31 - x32;
    jacobian(2, 3) = -p_on_ee_x*x18 + p_on_ee_y*x27 + x18*x30 - x22*x27;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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

    // Code for explicit solution node 1, solved variable is negative_th_2_positive_th_1__soa
    auto ExplicitSolutionNode_node_1_solve_negative_th_2_positive_th_1__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;

        // The explicit solution of root node
        const bool condition_0 = std::fabs(r_33) <= 70368744177665.0/70368744177664.0;
        if (condition_0)
        {
            const double x0 = safe_acos(r_33);
            auto solution_0 = make_raw_solution();
            solution_0[0] = x0;
            int appended_idx_0 = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx_0);

            auto solution_1 = make_raw_solution();
            solution_1[0] = -x0;
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx_1);

        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_negative_th_2_positive_th_1__soa_processor();
    // Finish code for explicit solution node 0

    // Code for non-branch dispatcher node 2
    // Actually, there is no code

    // Code for explicit solution node 3, solved variable is th_1
    auto ExplicitSolutionNode_node_3_solve_th_1_processor = [&]() -> void
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const bool condition_0 = std::fabs((-Pz + a_2*std::sin(negative_th_2_positive_th_1__soa) + d_3*std::cos(negative_th_2_positive_th_1__soa))/a_1) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = safe_asin((Pz - a_2*std::sin(negative_th_2_positive_th_1__soa) - d_3*std::cos(negative_th_2_positive_th_1__soa))/a_1);
                RawSolution solution_0(this_solution);
                solution_0[2] = x0;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_1_processor();
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const double th_1 = this_solution[2];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -negative_th_2_positive_th_1__soa + th_1;
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];

            const bool degenerate_valid_0 = std::fabs(negative_th_2_positive_th_1__soa) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(10, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(negative_th_2_positive_th_1__soa - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                int branch_input_index = node_input_i_idx_in_queue;
                if (forwarded_to_child)
                    branch_input_index = append_solution_to_queue(this_solution);
                forwarded_to_child = true;
                add_input_index_to(14, branch_input_index);
            }

            if (!taken_by_degenerate)
                add_input_index_to(7, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor();
    // Finish code for solved_variable dispatcher node 6

    // Code for equation all-zero dispatcher node 14
    auto EquationAllZeroDispatcherNode_node_14_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(14);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(14);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool checked_result = std::fabs(r_13) <= 9.9999999999999995e-7 && std::fabs(r_23) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(15, node_input_i_idx_in_queue);
            else
                add_input_index_to(22, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_14_processor();
    // Finish code for equation all-zero dispatcher node 14

    // Code for equation all-zero dispatcher node 22
    auto EquationAllZeroDispatcherNode_node_22_processor = [&]() -> void
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
            const bool checked_result = std::fabs(Px) <= 9.9999999999999995e-7 && std::fabs(Py) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(23, node_input_i_idx_in_queue);
            else
                add_input_index_to(30, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_22_processor();
    // Finish code for equation all-zero dispatcher node 22

    // Code for equation all-zero dispatcher node 30
    auto EquationAllZeroDispatcherNode_node_30_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(30);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(30);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool checked_result = std::fabs(std::pow(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) + d_3*(-std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1)), 2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(31, node_input_i_idx_in_queue);
            else
                add_input_index_to(38, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_30_processor();
    // Finish code for equation all-zero dispatcher node 30

    // Code for equation all-zero dispatcher node 38
    auto EquationAllZeroDispatcherNode_node_38_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(38);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(38);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool checked_result = std::fabs(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) + d_3*(-std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(39, node_input_i_idx_in_queue);
            else
                add_input_index_to(46, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_38_processor();
    // Finish code for equation all-zero dispatcher node 38

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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool checked_result = std::fabs(r_13*std::sin(th_1)) <= 9.9999999999999995e-7 && std::fabs(r_23*std::sin(th_1)) <= 9.9999999999999995e-7 && std::fabs(r_33*std::cos(th_1) - std::cos(th_2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(47, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_46_processor();
    // Finish code for equation all-zero dispatcher node 46

    // Code for explicit solution node 47, solved variable is th_0
    auto ExplicitSolutionNode_node_47_solve_th_0_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(r_13*std::sin(th_1)) >= zero_tolerance || std::fabs(r_23*std::sin(th_1)) >= zero_tolerance || std::fabs(r_33*std::cos(th_1) - std::cos(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_1);
                const double x1 = std::atan2(r_23*x0, r_13*x0);
                const double x2 = std::pow(x0, 2);
                const double x3 = r_33*std::cos(th_1) - std::cos(th_2);
                const double x4 = safe_sqrt(std::pow(r_13, 2)*x2 + std::pow(r_23, 2)*x2 - std::pow(x3, 2));
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x4, x3);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(48, appended_idx_0);
                const double tmp_sol_value = x1 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(48, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_47_solve_th_0_processor();
    // Finish code for explicit solution node 47

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
            const bool checked_result = false;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(49, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_48_processor();
    // Finish code for equation all-zero dispatcher node 48

    // Code for explicit solution node 49, solved variable is th_3
    auto ExplicitSolutionNode_node_49_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_3_processor();
    // Finish code for explicit solution node 49

    // Code for explicit solution node 39, solved variable is th_0
    auto ExplicitSolutionNode_node_39_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(39);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(39);
        if (!this_input_valid)
            return;

        // The solution of non-root node 39
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(Py/(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) - d_3*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1)))) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_2);
                const double x4 = safe_asin(Py/(a_0 + a_1*x0 + a_2*(x0*x3 + x1*x2) + d_3*(x0*x2 - x1*x3)));
                RawSolution solution_0(this_solution);
                solution_0[1] = x4;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(40, appended_idx_0);
                const double tmp_sol_value = M_PI - x4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(40, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_0_processor();
    // Finish code for explicit solution node 39

    // Code for equation all-zero dispatcher node 40
    auto EquationAllZeroDispatcherNode_node_40_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(40);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(40);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool checked_result = false;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(41, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_40_processor();
    // Finish code for equation all-zero dispatcher node 40

    // Code for explicit solution node 41, solved variable is th_3
    auto ExplicitSolutionNode_node_41_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_3_processor();
    // Finish code for explicit solution node 41

    // Code for explicit solution node 31, solved variable is th_0
    auto ExplicitSolutionNode_node_31_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(31);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(31);
        if (!this_input_valid)
            return;

        // The solution of non-root node 31
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(Px) >= zero_tolerance || std::fabs(Py) >= zero_tolerance || std::fabs(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) + d_3*(-std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_2);
                const double x4 = 1.0/(a_0 + a_1*x0 + a_2*(x0*x3 + x1*x2) + d_3*(x0*x2 - x1*x3));
                const double tmp_sol_value = std::atan2(Py*x4, Px*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_0_processor();
    // Finish code for explicit solution node 31

    // Code for non-branch dispatcher node 32
    // Actually, there is no code

    // Code for explicit solution node 33, solved variable is th_3
    auto ExplicitSolutionNode_node_33_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(32);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(32);
        if (!this_input_valid)
            return;

        // The solution of non-root node 33
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_3_processor();
    // Finish code for explicit solution node 32

    // Code for explicit solution node 23, solved variable is th_0
    auto ExplicitSolutionNode_node_23_solve_th_0_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(Px) >= 9.9999999999999995e-7 || std::fabs(Py) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                const double x0 = th_1 - th_2;
                const double x1 = -a_0 - a_1*std::cos(th_1) - a_2*std::cos(x0) + d_3*std::sin(x0);
                const double tmp_sol_value = std::atan2(-Py*x1, -Px*x1);
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
    ExplicitSolutionNode_node_23_solve_th_0_processor();
    // Finish code for explicit solution node 23

    // Code for non-branch dispatcher node 24
    // Actually, there is no code

    // Code for explicit solution node 25, solved variable is th_3
    auto ExplicitSolutionNode_node_25_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_3_processor();
    // Finish code for explicit solution node 24

    // Code for explicit solution node 15, solved variable is th_0
    auto ExplicitSolutionNode_node_15_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(15);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(15);
        if (!this_input_valid)
            return;

        // The solution of non-root node 15
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(r_13) >= 9.9999999999999995e-7 || std::fabs(r_23) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                const double x0 = std::sin(th_1 - th_2);
                const double tmp_sol_value = std::atan2(-r_23*x0, -r_13*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(16, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_0_processor();
    // Finish code for explicit solution node 15

    // Code for non-branch dispatcher node 16
    // Actually, there is no code

    // Code for explicit solution node 17, solved variable is th_3
    auto ExplicitSolutionNode_node_17_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_3_processor();
    // Finish code for explicit solution node 16

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
            const bool checked_result = std::fabs(r_13) <= 9.9999999999999995e-7 && std::fabs(r_23) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue);
            else
                add_input_index_to(18, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor();
    // Finish code for equation all-zero dispatcher node 10

    // Code for equation all-zero dispatcher node 18
    auto EquationAllZeroDispatcherNode_node_18_processor = [&]() -> void
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
            const bool checked_result = std::fabs(Px) <= 9.9999999999999995e-7 && std::fabs(Py) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(19, node_input_i_idx_in_queue);
            else
                add_input_index_to(26, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_18_processor();
    // Finish code for equation all-zero dispatcher node 18

    // Code for equation all-zero dispatcher node 26
    auto EquationAllZeroDispatcherNode_node_26_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(26);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(26);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool checked_result = std::fabs(std::pow(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) + d_3*(-std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1)), 2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(27, node_input_i_idx_in_queue);
            else
                add_input_index_to(34, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_26_processor();
    // Finish code for equation all-zero dispatcher node 26

    // Code for equation all-zero dispatcher node 34
    auto EquationAllZeroDispatcherNode_node_34_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(34);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(34);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool checked_result = std::fabs(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) + d_3*(-std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(35, node_input_i_idx_in_queue);
            else
                add_input_index_to(42, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_34_processor();
    // Finish code for equation all-zero dispatcher node 34

    // Code for equation all-zero dispatcher node 42
    auto EquationAllZeroDispatcherNode_node_42_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(42);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(42);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool checked_result = std::fabs(r_13*std::sin(th_1)) <= 9.9999999999999995e-7 && std::fabs(r_23*std::sin(th_1)) <= 9.9999999999999995e-7 && std::fabs(r_33*std::cos(th_1) - std::cos(th_2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(43, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_42_processor();
    // Finish code for equation all-zero dispatcher node 42

    // Code for explicit solution node 43, solved variable is th_0
    auto ExplicitSolutionNode_node_43_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(43);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(43);
        if (!this_input_valid)
            return;

        // The solution of non-root node 43
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(r_13*std::sin(th_1)) >= zero_tolerance || std::fabs(r_23*std::sin(th_1)) >= zero_tolerance || std::fabs(r_33*std::cos(th_1) - std::cos(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_1);
                const double x1 = std::atan2(r_23*x0, r_13*x0);
                const double x2 = std::pow(x0, 2);
                const double x3 = r_33*std::cos(th_1) - std::cos(th_2);
                const double x4 = safe_sqrt(std::pow(r_13, 2)*x2 + std::pow(r_23, 2)*x2 - std::pow(x3, 2));
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x4, x3);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(44, appended_idx_0);
                const double tmp_sol_value = x1 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(44, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_43_solve_th_0_processor();
    // Finish code for explicit solution node 43

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
            const bool checked_result = false;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(45, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_44_processor();
    // Finish code for equation all-zero dispatcher node 44

    // Code for explicit solution node 45, solved variable is th_3
    auto ExplicitSolutionNode_node_45_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_45_solve_th_3_processor();
    // Finish code for explicit solution node 45

    // Code for explicit solution node 35, solved variable is th_0
    auto ExplicitSolutionNode_node_35_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(35);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(35);
        if (!this_input_valid)
            return;

        // The solution of non-root node 35
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(Py/(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) - d_3*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1)))) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_2);
                const double x4 = safe_asin(Py/(a_0 + a_1*x0 + a_2*(x0*x3 + x1*x2) + d_3*(x0*x2 - x1*x3)));
                RawSolution solution_0(this_solution);
                solution_0[1] = x4;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(36, appended_idx_0);
                const double tmp_sol_value = M_PI - x4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(36, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_0_processor();
    // Finish code for explicit solution node 35

    // Code for non-branch dispatcher node 36
    // Actually, there is no code

    // Code for explicit solution node 37, solved variable is th_3
    auto ExplicitSolutionNode_node_37_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(36);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(36);
        if (!this_input_valid)
            return;

        // The solution of non-root node 37
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_3_processor();
    // Finish code for explicit solution node 36

    // Code for explicit solution node 27, solved variable is th_0
    auto ExplicitSolutionNode_node_27_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(27);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(27);
        if (!this_input_valid)
            return;

        // The solution of non-root node 27
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(Px) >= zero_tolerance || std::fabs(Py) >= zero_tolerance || std::fabs(a_0 + a_1*std::cos(th_1) + a_2*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2)) + d_3*(-std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_2);
                const double x4 = 1.0/(a_0 + a_1*x0 + a_2*(x0*x3 + x1*x2) + d_3*(x0*x2 - x1*x3));
                const double tmp_sol_value = std::atan2(Py*x4, Px*x4);
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
    ExplicitSolutionNode_node_27_solve_th_0_processor();
    // Finish code for explicit solution node 27

    // Code for non-branch dispatcher node 28
    // Actually, there is no code

    // Code for explicit solution node 29, solved variable is th_3
    auto ExplicitSolutionNode_node_29_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_3_processor();
    // Finish code for explicit solution node 28

    // Code for explicit solution node 19, solved variable is th_0
    auto ExplicitSolutionNode_node_19_solve_th_0_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(Px) >= 9.9999999999999995e-7 || std::fabs(Py) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                const double x0 = th_1 - th_2;
                const double x1 = -a_0 - a_1*std::cos(th_1) - a_2*std::cos(x0) + d_3*std::sin(x0);
                const double tmp_sol_value = std::atan2(-Py*x1, -Px*x1);
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
    ExplicitSolutionNode_node_19_solve_th_0_processor();
    // Finish code for explicit solution node 19

    // Code for non-branch dispatcher node 20
    // Actually, there is no code

    // Code for explicit solution node 21, solved variable is th_3
    auto ExplicitSolutionNode_node_21_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_3_processor();
    // Finish code for explicit solution node 20

    // Code for explicit solution node 11, solved variable is th_0
    auto ExplicitSolutionNode_node_11_solve_th_0_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const bool condition_0 = std::fabs(r_13) >= 9.9999999999999995e-7 || std::fabs(r_23) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                const double x0 = std::sin(th_1 - th_2);
                const double tmp_sol_value = std::atan2(-r_23*x0, -r_13*x0);
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
    ExplicitSolutionNode_node_11_solve_th_0_processor();
    // Finish code for explicit solution node 11

    // Code for non-branch dispatcher node 12
    // Actually, there is no code

    // Code for explicit solution node 13, solved variable is th_3
    auto ExplicitSolutionNode_node_13_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_3_processor();
    // Finish code for explicit solution node 12

    // Code for explicit solution node 7, solved variable is th_3
    auto ExplicitSolutionNode_node_7_solve_th_3_processor = [&]() -> void
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const bool condition_0 = std::fabs(r_31) >= zero_tolerance || std::fabs(r_32) >= zero_tolerance || std::fabs(std::sin(negative_th_2_positive_th_1__soa)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(negative_th_2_positive_th_1__soa);
                const double tmp_sol_value = std::atan2(-r_32*x0, r_31*x0);
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
    ExplicitSolutionNode_node_7_solve_th_3_processor();
    // Finish code for explicit solution node 7

    // Code for non-branch dispatcher node 8
    // Actually, there is no code

    // Code for explicit solution node 9, solved variable is th_0
    auto ExplicitSolutionNode_node_9_solve_th_0_processor = [&]() -> void
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const bool condition_0 = std::fabs(r_13) >= zero_tolerance || std::fabs(r_23) >= zero_tolerance || std::fabs(std::sin(negative_th_2_positive_th_1__soa)) >= zero_tolerance;
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(negative_th_2_positive_th_1__soa);
                const double tmp_sol_value = std::atan2(-r_23*x0, -r_13*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor();
    // Finish code for explicit solution node 8

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
        ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
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
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
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
}

static void wrapAngleToPi(std::vector<std::array<double, robot_nq>>& ik_output)
{
    for (auto& solution_i : ik_output)
        wrapAngleToPi(solution_i);
}

static void computeIK(const Eigen::Matrix4d& T_ee, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    if (!T_ee.allFinite()) { ik_output.clear(); return; }
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
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
    Eigen::Matrix4d T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);
    Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
    computeRawIK(T_ee_raw_disturbed, workspace);
    const auto& raw_ik_out_disturb = workspace.raw_ik_out;
    for(int i = 0; i < raw_ik_out_disturb.size(); i++)
    {
        auto ik_out_i = raw_ik_out_disturb[i];
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw;
        ik_out_i[1] = addRevoluteOffset(ik_out_i[1], -th_1_offset_original2raw);
        ik_out_i[2] = addRevoluteOffset(ik_out_i[2], -th_2_offset_original2raw);
        ik_out_i[3] = addRevoluteOffset(ik_out_i[3], -th_3_offset_original2raw);
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

    removeDuplicateWithinClasses<robot_nq>(ik_output, zero_tolerance, ikCandidateClasses<robot_nq>(ik_output, jointLower(), jointUpper(), periodicLimitJoints(), [&](const std::array<double, robot_nq>& candidate) { return computeFK(candidate); }, T_ee));
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct kawasaki_CP130L_ik

// Code below for debug
void test_ik_solve_kawasaki_CP130L()
{
    std::array<double, kawasaki_CP130L_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = kawasaki_CP130L_ik::computeFK(theta);
    auto ik_output = kawasaki_CP130L_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = kawasaki_CP130L_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_kawasaki_CP130L();
    return 0;
}
