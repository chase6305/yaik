#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct puma_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 32;
static std::array<double, robot_nq> jointLower() { return {{-3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793}}; }
static std::array<double, robot_nq> jointUpper() { return {{3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793}}; }
static std::array<bool, robot_nq> periodicLimitJoints() { return {{true, true, true, true, true, true}}; }
static std::array<bool, robot_nq> fixedJoints() { return {{false, false, false, false, false, false}}; }
static constexpr int n_tree_nodes = 24;
static constexpr int intermediate_solution_size = 9;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using SolutionQueueType = ::yaik_cpp::CompactSolutionQueue<intermediate_solution_size, max_n_solutions>;
using NodeIndexWorkspaceType = ::yaik_cpp::CompactNodeIndexWorkspace;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq, SolutionQueueType, NodeIndexWorkspaceType>;

// Robot parameters
static constexpr double a_2 = 0.432;
static constexpr double a_3 = 0.0203;
static constexpr double d_1 = 0.6;
static constexpr double d_3 = 0.1245;
static constexpr double d_4 = 0.432;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw

// The transformation between raw and original ee target
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// ee_original = pre_transform * ee_raw * post_transform
// ee_raw = dh_forward_transform(theta_raw)
static Eigen::Matrix4d endEffectorTargetOriginalToRaw(const Eigen::Matrix4d& T_ee)
{
    return T_ee;
}

static Eigen::Matrix4d endEffectorTargetRawToOriginal(const Eigen::Matrix4d& T_ee)
{
    return T_ee;
}

///************* Below are the actual FK and IK implementations *******************
static Eigen::Matrix4d computeFK(const std::array<double, robot_nq>& theta_input_original)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];

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
    const double x20 = a_2*x8;
    const double x21 = -x1*x7 + x1*x8*x9;
    const double x22 = -x2*x4 - x21*x3;
    const double x23 = -x1*x14 - x1*x15;
    const double x24 = x2*x21 - x3*x4;
    const double x25 = -x13*x23 + x17*x24;
    const double x26 = -x14 - x15;
    const double x27 = x26*x3;
    const double x28 = x7 - x8*x9;
    const double x29 = -x13*x28 + x17*x2*x26;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x11 + x12*x19;
    ee_pose_raw(0, 1) = -x0*x19 + x11*x12;
    ee_pose_raw(0, 2) = x13*x18 + x16*x17;
    ee_pose_raw(0, 3) = a_3*x10 - d_3*x1 + d_4*x16 + x20*x4;
    ee_pose_raw(1, 0) = x0*x22 + x12*x25;
    ee_pose_raw(1, 1) = -x0*x25 + x12*x22;
    ee_pose_raw(1, 2) = x13*x24 + x17*x23;
    ee_pose_raw(1, 3) = a_3*x21 + d_3*x4 + d_4*x23 + x1*x20;
    ee_pose_raw(2, 0) = -x0*x27 + x12*x29;
    ee_pose_raw(2, 1) = -x0*x29 - x12*x27;
    ee_pose_raw(2, 2) = x13*x2*x26 + x17*x28;
    ee_pose_raw(2, 3) = -a_2*x5 + a_3*x26 + d_1 + d_4*x28;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = -x0;
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x6*x7;
    const double x9 = -x2*x5 - x2*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x3*x6;
    const double x13 = -x12*x2 + x2*x4*x7;
    const double x14 = x0*x10 - x11*x13;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = x15*x9 + x16*(x0*x11 + x10*x13);
    const double x18 = -x0*x5 - x0*x8;
    const double x19 = -x0*x12 + x0*x4*x7;
    const double x20 = -x10*x2 - x11*x19;
    const double x21 = x15*x18 + x16*(x10*x19 - x11*x2);
    const double x22 = x12 - x4*x7;
    const double x23 = -x5 - x8;
    const double x24 = x11*x23;
    const double x25 = x10*x16*x23 + x15*x22;
    const double x26 = a_2*x3;
    const double x27 = d_1 - x26;
    const double x28 = a_3*x23 + d_1 + d_4*x22 - x26;
    const double x29 = a_2*x7;
    const double x30 = d_3*x2 + x0*x29;
    const double x31 = a_3*x19 + d_4*x18 + x30;
    const double x32 = -d_3*x0 + x2*x29;
    const double x33 = a_3*x13 + d_4*x9 + x32;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x17;
    jacobian(1, 1) = x2;
    jacobian(1, 2) = x2;
    jacobian(1, 3) = x18;
    jacobian(1, 4) = x20;
    jacobian(1, 5) = x21;
    jacobian(2, 0) = 1;
    jacobian(2, 3) = x22;
    jacobian(2, 4) = -x24;
    jacobian(2, 5) = x25;
    jacobian(3, 1) = -d_1*x2;
    jacobian(3, 2) = -x2*x27;
    jacobian(3, 3) = -x18*x28 + x22*x31;
    jacobian(3, 4) = -x20*x28 - x24*x31;
    jacobian(3, 5) = -x21*x28 + x25*x31;
    jacobian(4, 1) = -d_1*x0;
    jacobian(4, 2) = -x0*x27;
    jacobian(4, 3) = -x22*x33 + x28*x9;
    jacobian(4, 4) = x14*x28 + x24*x33;
    jacobian(4, 5) = x17*x28 - x25*x33;
    jacobian(5, 2) = x0*x30 + x2*x32;
    jacobian(5, 3) = x18*x33 - x31*x9;
    jacobian(5, 4) = -x14*x31 + x20*x33;
    jacobian(5, 5) = -x17*x31 + x21*x33;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];

    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = -x0;
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x6*x7;
    const double x9 = -x2*x5 - x2*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x3*x6;
    const double x13 = -x12*x2 + x2*x4*x7;
    const double x14 = std::cos(th_4);
    const double x15 = std::sin(th_4);
    const double x16 = -x0*x5 - x0*x8;
    const double x17 = -x0*x12 + x0*x4*x7;
    const double x18 = x12 - x4*x7;
    const double x19 = -x5 - x8;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x0*x10 - x11*x13;
    jacobian(0, 5) = x14*x9 + x15*(x0*x11 + x10*x13);
    jacobian(1, 1) = x2;
    jacobian(1, 2) = x2;
    jacobian(1, 3) = x16;
    jacobian(1, 4) = -x10*x2 - x11*x17;
    jacobian(1, 5) = x14*x16 + x15*(x10*x17 - x11*x2);
    jacobian(2, 0) = 1;
    jacobian(2, 3) = x18;
    jacobian(2, 4) = -x11*x19;
    jacobian(2, 5) = x10*x15*x19 + x14*x18;
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];

    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = a_2*x1;
    const double x3 = d_1 - x2;
    const double x4 = std::sin(th_2);
    const double x5 = x1*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::cos(th_2);
    const double x8 = x5 - x6*x7;
    const double x9 = std::sin(th_0);
    const double x10 = x1*x7;
    const double x11 = x4*x6;
    const double x12 = -x10*x9 - x11*x9;
    const double x13 = -x10 - x11;
    const double x14 = a_3*x13 + d_1 + d_4*x8 - x2;
    const double x15 = -x5*x9 + x6*x7*x9;
    const double x16 = a_2*x6;
    const double x17 = d_3*x0 + x16*x9;
    const double x18 = a_3*x15 + d_4*x12 + x17;
    const double x19 = std::sin(th_3);
    const double x20 = x13*x19;
    const double x21 = std::cos(th_3);
    const double x22 = -x0*x21 - x15*x19;
    const double x23 = std::cos(th_4);
    const double x24 = std::sin(th_4);
    const double x25 = x13*x21*x24 + x23*x8;
    const double x26 = x12*x23 + x24*(-x0*x19 + x15*x21);
    const double x27 = -x0*x10 - x0*x11;
    const double x28 = -x0*x5 + x0*x6*x7;
    const double x29 = -d_3*x9 + x0*x16;
    const double x30 = a_3*x28 + d_4*x27 + x29;
    const double x31 = -x19*x28 + x21*x9;
    const double x32 = x23*x27 + x24*(x19*x9 + x21*x28);
    const double x33 = p_on_ee_x*x0 + p_on_ee_y*x9;
    // End of temp variables

    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y;
    jacobian(0, 1) = -d_1*x0 + p_on_ee_z*x0;
    jacobian(0, 2) = p_on_ee_z*x0 - x0*x3;
    jacobian(0, 3) = -p_on_ee_y*x8 + p_on_ee_z*x12 - x12*x14 + x18*x8;
    jacobian(0, 4) = p_on_ee_y*x20 + p_on_ee_z*x22 - x14*x22 - x18*x20;
    jacobian(0, 5) = -p_on_ee_y*x25 + p_on_ee_z*x26 - x14*x26 + x18*x25;
    jacobian(1, 0) = p_on_ee_x;
    jacobian(1, 1) = -d_1*x9 + p_on_ee_z*x9;
    jacobian(1, 2) = p_on_ee_z*x9 - x3*x9;
    jacobian(1, 3) = p_on_ee_x*x8 - p_on_ee_z*x27 + x14*x27 - x30*x8;
    jacobian(1, 4) = -p_on_ee_x*x20 - p_on_ee_z*x31 + x13*x19*x30 + x14*x31;
    jacobian(1, 5) = p_on_ee_x*x25 - p_on_ee_z*x32 + x14*x32 - x25*x30;
    jacobian(2, 1) = -x33;
    jacobian(2, 2) = x0*x29 + x17*x9 - x33;
    jacobian(2, 3) = -p_on_ee_x*x12 + p_on_ee_y*x27 + x12*x30 - x18*x27;
    jacobian(2, 4) = -p_on_ee_x*x22 + p_on_ee_y*x31 - x18*x31 + x22*x30;
    jacobian(2, 5) = -p_on_ee_x*x26 + p_on_ee_y*x32 - x18*x32 + x26*x30;
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
    static constexpr std::array<std::size_t, n_tree_nodes> node_input_capacities = {{1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8}};
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
        const bool condition_0 = ((std::fabs(Px) >= zero_tolerance) || (std::fabs(Py) >= zero_tolerance) || (std::fabs(d_3) >= zero_tolerance));
        if (condition_0)
        {
            const double x0 = std::atan2(Px, -Py);
            const double x1 = safe_sqrt(std::pow(Px, 2) + std::pow(Py, 2) - std::pow(d_3, 2));
            const double x2 = -d_3;
            auto solution_0 = make_raw_solution();
            solution_0[1] = x0 + std::atan2(x1, x2);
            int appended_idx_0 = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx_0);

            auto solution_1 = make_raw_solution();
            solution_1[1] = x0 + std::atan2(-x1, x2);
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx_1);

        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor();
    // Finish code for explicit solution node 0

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
            const bool condition_0 = ((2*std::fabs(a_2*a_3) >= zero_tolerance) || (2*std::fabs(a_2*d_4) >= zero_tolerance) || (std::fabs(-std::pow(Px, 2) - std::pow(Py, 2) - std::pow(Pz, 2) + 2*Pz*d_1 + std::pow(a_2, 2) + std::pow(a_3, 2) - std::pow(d_1, 2) + std::pow(d_3, 2) + std::pow(d_4, 2)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 2*a_2;
                const double x1 = std::atan2(-d_4*x0, a_3*x0);
                const double x2 = std::pow(a_3, 2);
                const double x3 = std::pow(a_2, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_4, 2);
                const double x6 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_1 + std::pow(d_1, 2) - std::pow(d_3, 2) - x2 - x3 - x5;
                const double x7 = safe_sqrt(x2*x4 + x4*x5 - std::pow(x6, 2));
                RawSolution solution_0(this_solution);
                solution_0[4] = x1 + std::atan2(x7, x6);
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx_0);
                const double tmp_sol_value = x1 + std::atan2(-x7, x6);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
            const double th_0 = this_solution[1];
            const bool checked_result = ((std::fabs(Pz - d_1) <= 9.9999999999999995e-7) && (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7));
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
            else
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
        }
    };

    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4

    // Code for explicit solution node 5, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_5_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const double th_2 = this_solution[4];
            const bool condition_0 = ((std::fabs(Pz - d_1) >= 9.9999999999999995e-7) || (std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7));
            if (condition_0)
            {
                const double x0 = Pz - d_1;
                const double x1 = -a_2*std::cos(th_2) - a_3;
                const double x2 = a_2*std::sin(th_2) - d_4;
                const double x3 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double tmp_sol_value = std::atan2(x0*x1 - x2*x3, x0*x2 + x1*x3);
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
    ExplicitSolutionNode_node_5_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 5

    // Code for non-branch dispatcher node 6
    // Actually, there is no code

    // Code for explicit solution node 7, solved variable is th_1
    auto ExplicitSolutionNode_node_7_solve_th_1_processor = [&]() -> void
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
            const double th_1th_2_soa = this_solution[3];
            const double th_2 = this_solution[4];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = th_1th_2_soa - th_2;
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
    // Finish code for explicit solution node 6

    // Code for non-branch dispatcher node 8
    // Actually, there is no code

    // Code for explicit solution node 9, solved variable is th_4
    auto ExplicitSolutionNode_node_9_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const bool condition_0 = std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_33*(std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))) <= 70368744177665.0/70368744177664.0;
            if (condition_0)
            {
                const double x0 = std::sin(th_1);
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(th_1);
                const double x3 = std::cos(th_2);
                const double x4 = x0*x3 + x1*x2;
                const double x5 = safe_acos(-r_13*x4*std::cos(th_0) - r_23*x4*std::sin(th_0) - r_33*(-x0*x1 + x2*x3));
                RawSolution solution_0(this_solution);
                solution_0[7] = x5;
                int appended_idx_0 = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx_0);
                const double tmp_sol_value = -x5;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
    // Finish code for explicit solution node 8

    // Code for solved_variable dispatcher node 10
    auto SolvedVariableDispatcherNode_node_10_processor = [&]() -> void
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
                add_input_index_to(14, branch_input_index);
            }

            const bool degenerate_valid_1 = std::fabs(th_4 - M_PI) <= 9.9999999999999995e-7;
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
                add_input_index_to(11, node_input_i_idx_in_queue);
        }
    };

    // Invoke the processor
    SolvedVariableDispatcherNode_node_10_processor();
    // Finish code for solved_variable dispatcher node 10

    // Code for explicit solution node 19, solved variable is negative_th_5_positive_th_3__soa
    auto ExplicitSolutionNode_node_19_solve_negative_th_5_positive_th_3__soa_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, r_12*x0 - r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_negative_th_5_positive_th_3__soa_processor();
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
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
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

    // Code for non-branch dispatcher node 22
    // Actually, there is no code

    // Code for explicit solution node 23, solved variable is th_5
    auto ExplicitSolutionNode_node_23_solve_th_5_processor = [&]() -> void
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
            const double negative_th_5_positive_th_3__soa = this_solution[0];
            const double th_3 = this_solution[5];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -negative_th_5_positive_th_3__soa + th_3;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_5_processor();
    // Finish code for explicit solution node 22

    // Code for explicit solution node 14, solved variable is th_3th_5_soa
    auto ExplicitSolutionNode_node_14_solve_th_3th_5_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const bool condition_0 = ((1 >= zero_tolerance) || (std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance) || (std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_3th_5_soa_processor();
    // Finish code for explicit solution node 14

    // Code for non-branch dispatcher node 15
    // Actually, there is no code

    // Code for explicit solution node 16, solved variable is th_3
    auto ExplicitSolutionNode_node_16_solve_th_3_processor = [&]() -> void
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
            const bool condition_0 = true;
            if (condition_0)
            {
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_3_processor();
    // Finish code for explicit solution node 15

    // Code for non-branch dispatcher node 17
    // Actually, there is no code

    // Code for explicit solution node 18, solved variable is th_5
    auto ExplicitSolutionNode_node_18_solve_th_5_processor = [&]() -> void
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
            const double th_3 = this_solution[5];
            const double th_3th_5_soa = this_solution[6];
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                const double tmp_sol_value = -th_3 + th_3th_5_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_5_processor();
    // Finish code for explicit solution node 17

    // Code for explicit solution node 11, solved variable is th_3
    auto ExplicitSolutionNode_node_11_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[7];
            const bool condition_0 = ((std::fabs(std::sin(th_4)) >= zero_tolerance) || (std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance) || (std::fabs(-r_13*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - r_23*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + r_33*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_1);
                const double x4 = std::cos(th_2);
                const double x5 = std::sin(th_2);
                const double x6 = std::cos(th_1);
                const double x7 = -x3*x5 + x4*x6;
                const double tmp_sol_value = std::atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x7 + r_23*x1*x7 - r_33*(x3*x4 + x5*x6)));
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
    ExplicitSolutionNode_node_11_solve_th_3_processor();
    // Finish code for explicit solution node 11

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
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[7];
            const bool condition_0 = ((std::fabs(std::sin(th_4)) >= zero_tolerance) || (std::fabs(r_12*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_22*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_32*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance) || (std::fabs(r_11*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_21*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_31*(std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))) >= zero_tolerance));
            if (condition_0)
            {
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = std::cos(th_2);
                const double x5 = -x1*x2 + x3*x4;
                const double x6 = x1*x4 + x2*x3;
                const double x7 = x6*std::cos(th_0);
                const double x8 = x6*std::sin(th_0);
                const double tmp_sol_value = std::atan2(x0*(-r_12*x7 - r_22*x8 - r_32*x5), x0*(r_11*x7 + r_21*x8 + r_31*x5));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
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
        const double value_at_2 = raw_ik_out_i[4];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[5];  // th_3
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
    const Eigen::Matrix4d& T_ee_raw = T_ee;
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
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

}; // struct puma_ik

// Code below for debug
void test_ik_solve_puma()
{
    std::array<double, puma_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = puma_ik::computeFK(theta);
    auto ik_output = puma_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = puma_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_puma();
    return 0;
}
