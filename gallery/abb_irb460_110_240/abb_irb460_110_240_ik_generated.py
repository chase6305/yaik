import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 4
n_tree_nodes: int = 50
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.26
a_1: float = 0.945
a_2: float = 0.22
d_3: float = 0.2515
pre_transform_special_symbol_23: float = 0.7425

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = 1.5707963267948966
th_3_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def abb_irb460_110_240_ik_target_original_to_raw(T_ee: np.ndarray):
    r_11: float = T_ee[0, 0]
    r_12: float = T_ee[0, 1]
    r_13: float = T_ee[0, 2]
    Px: float = T_ee[0, 3]
    r_21: float = T_ee[1, 0]
    r_22: float = T_ee[1, 1]
    r_23: float = T_ee[1, 2]
    Py: float = T_ee[1, 3]
    r_31: float = T_ee[2, 0]
    r_32: float = T_ee[2, 1]
    r_33: float = T_ee[2, 2]
    Pz: float = T_ee[2, 3]
    ee_transformed = np.eye(4)
    ee_transformed[0, 0] = -1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = -1.0*r_13
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = -1.0*r_21
    ee_transformed[1, 1] = 1.0*r_22
    ee_transformed[1, 2] = -1.0*r_23
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = -1.0*r_31
    ee_transformed[2, 1] = 1.0*r_32
    ee_transformed[2, 2] = -1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def abb_irb460_110_240_ik_target_raw_to_original(T_ee: np.ndarray):
    r_11: float = T_ee[0, 0]
    r_12: float = T_ee[0, 1]
    r_13: float = T_ee[0, 2]
    Px: float = T_ee[0, 3]
    r_21: float = T_ee[1, 0]
    r_22: float = T_ee[1, 1]
    r_23: float = T_ee[1, 2]
    Py: float = T_ee[1, 3]
    r_31: float = T_ee[2, 0]
    r_32: float = T_ee[2, 1]
    r_33: float = T_ee[2, 2]
    Pz: float = T_ee[2, 3]
    ee_transformed = np.eye(4)
    ee_transformed[0, 0] = -1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = -1.0*r_13
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = -1.0*r_21
    ee_transformed[1, 1] = 1.0*r_22
    ee_transformed[1, 2] = -1.0*r_23
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = -1.0*r_31
    ee_transformed[2, 1] = 1.0*r_32
    ee_transformed[2, 2] = -1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def abb_irb460_110_240_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.sin(th_3)
    x2 = 1.0*x1
    x3 = math.cos(th_0)
    x4 = math.sin(th_1)
    x5 = math.sin(th_2)
    x6 = x4*x5
    x7 = math.cos(th_1)
    x8 = math.cos(th_2)
    x9 = -x3*x6 + x3*x7*x8
    x10 = math.cos(th_3)
    x11 = 1.0*x10
    x12 = x4*x8
    x13 = x12*x3
    x14 = x5*x7
    x15 = x14*x3
    x16 = 1.0*a_0
    x17 = 1.0*a_1
    x18 = x17*x7
    x19 = 1.0*a_2
    x20 = 1.0*d_3
    x21 = -x0*x6 + x0*x7*x8
    x22 = x0*x12
    x23 = x0*x14
    x24 = -x12 - x14
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x2 - x11*x9
    ee_pose[0, 1] = 1.0*x0*x10 - x2*x9
    ee_pose[0, 2] = 1.0*x13 + 1.0*x15
    ee_pose[0, 3] = x16*x3 + x18*x3 + x19*x9 + x20*(-x13 - x15)
    ee_pose[1, 0] = 1.0*x1*x3 - x11*x21
    ee_pose[1, 1] = -x11*x3 - x2*x21
    ee_pose[1, 2] = 1.0*x22 + 1.0*x23
    ee_pose[1, 3] = x0*x16 + x0*x18 + x19*x21 + x20*(-x22 - x23)
    ee_pose[2, 0] = -x11*x24
    ee_pose[2, 1] = -x2*x24
    ee_pose[2, 2] = -1.0*x6 + 1.0*x7*x8
    ee_pose[2, 3] = 1.0*pre_transform_special_symbol_23 - x17*x4 + x19*x24 + x20*(x6 - x7*x8)
    return ee_pose


def abb_irb460_110_240_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = -x1
    x3 = math.cos(th_0)
    x4 = 1.0*x3
    x5 = math.sin(th_1)
    x6 = math.cos(th_2)
    x7 = x5*x6
    x8 = math.sin(th_2)
    x9 = math.cos(th_1)
    x10 = x8*x9
    x11 = -x10*x4 - x4*x7
    x12 = -x1*x10 - x1*x7
    x13 = 1.0*x5
    x14 = 1.0*x9
    x15 = x13*x8 - x14*x6
    x16 = a_1*x13
    x17 = pre_transform_special_symbol_23 - x16
    x18 = a_2*(-x13*x6 - x14*x8) + d_3*x15 + pre_transform_special_symbol_23 - x16
    x19 = x5*x8
    x20 = a_0*x1 + a_1*x1*x9
    x21 = a_2*(1.0*x0*x6*x9 - x1*x19) + d_3*x12 + x20
    x22 = a_0*x4 + a_1*x4*x9
    x23 = a_2*(-x19*x4 + 1.0*x3*x6*x9) + d_3*x11 + x22
    x24 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 4))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x11
    jacobian_output[1, 1] = x4
    jacobian_output[1, 2] = x4
    jacobian_output[1, 3] = x12
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x15
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x4
    jacobian_output[3, 2] = -x17*x4
    jacobian_output[3, 3] = -x12*x18 + x15*x21
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x17
    jacobian_output[4, 3] = x11*x18 - x15*x23
    jacobian_output[5, 1] = x0**2*x24 + x24*x3**2
    jacobian_output[5, 2] = x1*x20 + x22*x4
    jacobian_output[5, 3] = -x11*x21 + x12*x23
    return jacobian_output


def abb_irb460_110_240_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = -x0
    x2 = 1.0*math.cos(th_0)
    x3 = math.sin(th_1)
    x4 = math.cos(th_2)
    x5 = x3*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x6*x7
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 4))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = -x2*x5 - x2*x8
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = x2
    jacobian_output[1, 3] = -x0*x5 - x0*x8
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = 1.0*x3*x6 - 1.0*x4*x7
    return jacobian_output


def abb_irb460_110_240_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = 1.0*p_on_ee_y
    x1 = math.cos(th_0)
    x2 = 1.0*x1
    x3 = p_on_ee_z*x2
    x4 = math.sin(th_1)
    x5 = 1.0*x4
    x6 = a_1*x5
    x7 = pre_transform_special_symbol_23 - x6
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = math.cos(th_2)
    x11 = math.cos(th_1)
    x12 = 1.0*x11
    x13 = -x10*x12 + x9
    x14 = math.sin(th_0)
    x15 = x10*x5
    x16 = x12*x8
    x17 = -x14*x15 - x14*x16
    x18 = a_2*(-x15 - x16) + d_3*x13 + pre_transform_special_symbol_23 - x6
    x19 = 1.0*x14
    x20 = a_0*x19 + a_1*x12*x14
    x21 = a_2*(1.0*x10*x11*x14 - x14*x9) + d_3*x17 + x20
    x22 = 1.0*p_on_ee_x
    x23 = p_on_ee_z*x19
    x24 = x2*x4
    x25 = x11*x2
    x26 = -x10*x24 - x25*x8
    x27 = a_0*x2 + a_1*x25
    x28 = a_2*(1.0*x1*x10*x11 - x24*x8) + d_3*x26 + x27
    x29 = x1*x22
    x30 = x0*x14
    x31 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 4))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x7 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x21 - x17*x18
    jacobian_output[1, 0] = x22
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x19 + x23
    jacobian_output[1, 2] = -x19*x7 + x23
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x26 - x13*x28 + x18*x26
    jacobian_output[2, 1] = x1**2*x31 + x14**2*x31 - x29 - x30
    jacobian_output[2, 2] = 1.0*x1*x27 + 1.0*x14*x20 - x29 - x30
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x26 + x17*x28 - x21*x26
    return jacobian_output


def abb_irb460_110_240_ik_solve_raw(T_ee: np.ndarray):
    # Extracting the ik target symbols
    r_11 = T_ee[0, 0]
    r_12 = T_ee[0, 1]
    r_13 = T_ee[0, 2]
    Px = T_ee[0, 3]
    r_21 = T_ee[1, 0]
    r_22 = T_ee[1, 1]
    r_23 = T_ee[1, 2]
    Py = T_ee[1, 3]
    r_31 = T_ee[2, 0]
    r_32 = T_ee[2, 1]
    r_33 = T_ee[2, 2]
    Pz = T_ee[2, 3]
    inv_ee_translation = - T_ee[0:3, 0:3].T.dot(T_ee[0:3, 3])
    inv_Px = inv_ee_translation[0]
    inv_Py = inv_ee_translation[1]
    inv_Pz = inv_ee_translation[2]

    # A new ik type. Should be a fixed array in C++
    IkSolution = NewType("IkSolution", List[float])
    def make_ik_solution():
        return IkSolution([100000.0] * 5)

    solution_queue: List[IkSolution] = list()
    queue_element_validity: List[bool] = list()
    def append_solution_to_queue(solution_2_add: IkSolution):
        index_4_appended = len(solution_queue)
        solution_queue.append(solution_2_add)
        queue_element_validity.append(True)
        return index_4_appended

    # Init for workspace as empty list. A list of fixed size array for each node
    max_n_solutions: int = 16
    node_input_index: List[List[int]] = list()
    node_input_validity: List[bool] = list()
    for i in range(50):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for non-branch dispatcher node 0
    # Actually, there is no code

    # Code for explicit solution node 1, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_1_solve_th_1th_2_soa_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The explicit solution of root node
        condition_0: bool = (abs(r_33) <= 70368744177665/70368744177664)
        if condition_0:
            x0 = safe_acos(-r_33)
            solution_0: IkSolution = make_ik_solution()
            solution_0[2] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)

            solution_1: IkSolution = make_ik_solution()
            solution_1[2] = -x0
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)

    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 0

    # Code for non-branch dispatcher node 2
    # Actually, there is no code

    # Code for explicit solution node 3, solved variable is th_1
    def ExplicitSolutionNode_node_3_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[2]
        this_input_valid: bool = node_input_validity[2]
        if not this_input_valid:
            return

        # The solution of non-root node 3
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1th_2_soa = this_solution[2]
            condition_0: bool = (abs((Pz + a_2*math.sin(th_1th_2_soa) + d_3*math.cos(th_1th_2_soa))/a_1) <= 70368744177665/70368744177664)
            if condition_0:
                x0 = safe_asin((Pz + a_2*math.sin(th_1th_2_soa) + d_3*math.cos(th_1th_2_soa))/a_1)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = -x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[1] = x0 + math.pi
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_1_processor()
    # Finish code for explicit solution node 2

    # Code for non-branch dispatcher node 4
    # Actually, there is no code

    # Code for explicit solution node 5, solved variable is th_2
    def ExplicitSolutionNode_node_5_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
        if not this_input_valid:
            return

        # The solution of non-root node 5
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_1th_2_soa = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[3] = -th_1 + th_1th_2_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_2_processor()
    # Finish code for explicit solution node 4

    # Code for solved_variable dispatcher node 6
    def SolvedVariableDispatcherNode_node_6_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            forwarded_to_child = False
            th_1th_2_soa = this_solution[2]
            degenerate_valid_0 = (abs(th_1th_2_soa) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(10, branch_input_index)

            th_1th_2_soa = this_solution[2]
            degenerate_valid_1 = (abs(th_1th_2_soa - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(14, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6

    # Code for equation all-zero dispatcher node 14
    def EquationAllZeroDispatcherNode_node_14_processor():
        this_node_input_index: List[int] = node_input_index[14]
        this_input_valid: bool = node_input_validity[14]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = (abs(r_13) <= 1.0e-6) and (abs(r_23) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                add_input_index_to(22, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_14_processor()
    # Finish code for equation all-zero dispatcher node 14

    # Code for equation all-zero dispatcher node 22
    def EquationAllZeroDispatcherNode_node_22_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = (abs(Px) <= 1.0e-6) and (abs(Py) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(23, node_input_i_idx_in_queue)
            else:
                add_input_index_to(30, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_22_processor()
    # Finish code for equation all-zero dispatcher node 22

    # Code for equation all-zero dispatcher node 30
    def EquationAllZeroDispatcherNode_node_30_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            checked_result: bool = (abs((a_0 + a_1*math.cos(th_1) + a_2*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2)) + d_3*(-math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1)))**2) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(31, node_input_i_idx_in_queue)
            else:
                add_input_index_to(38, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_30_processor()
    # Finish code for equation all-zero dispatcher node 30

    # Code for equation all-zero dispatcher node 38
    def EquationAllZeroDispatcherNode_node_38_processor():
        this_node_input_index: List[int] = node_input_index[38]
        this_input_valid: bool = node_input_validity[38]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            checked_result: bool = (abs(a_0 + a_1*math.cos(th_1) + a_2*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2)) + d_3*(-math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(39, node_input_i_idx_in_queue)
            else:
                add_input_index_to(46, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_38_processor()
    # Finish code for equation all-zero dispatcher node 38

    # Code for equation all-zero dispatcher node 46
    def EquationAllZeroDispatcherNode_node_46_processor():
        this_node_input_index: List[int] = node_input_index[46]
        this_input_valid: bool = node_input_validity[46]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            checked_result: bool = (abs(r_13*math.cos(th_1)) <= 1.0e-6) and (abs(r_23*math.cos(th_1)) <= 1.0e-6) and (abs(r_33*math.sin(th_1) - math.sin(th_2)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(47, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_46_processor()
    # Finish code for equation all-zero dispatcher node 46

    # Code for explicit solution node 47, solved variable is th_0
    def ExplicitSolutionNode_node_47_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[47]
        this_input_valid: bool = node_input_validity[47]
        if not this_input_valid:
            return

        # The solution of non-root node 47
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(r_13*math.cos(th_1)) >= zero_tolerance) or (abs(r_23*math.cos(th_1)) >= zero_tolerance) or (abs(r_33*math.sin(th_1) - math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.atan2(-r_23*x0, -r_13*x0)
                x2 = x0**2
                x3 = -r_33*math.sin(th_1) + math.sin(th_2)
                x4 = safe_sqrt(r_13**2*x2 + r_23**2*x2 - x3**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x1 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(48, appended_idx)
                this_solution[0] = x1 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(48, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_47_solve_th_0_processor()
    # Finish code for explicit solution node 47

    # Code for equation all-zero dispatcher node 48
    def EquationAllZeroDispatcherNode_node_48_processor():
        this_node_input_index: List[int] = node_input_index[48]
        this_input_valid: bool = node_input_validity[48]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = False
            if not checked_result:  # To non-degenerate node
                add_input_index_to(49, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_48_processor()
    # Finish code for equation all-zero dispatcher node 48

    # Code for explicit solution node 49, solved variable is th_3
    def ExplicitSolutionNode_node_49_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[49]
        this_input_valid: bool = node_input_validity[49]
        if not this_input_valid:
            return

        # The solution of non-root node 49
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_3_processor()
    # Finish code for explicit solution node 49

    # Code for explicit solution node 39, solved variable is th_0
    def ExplicitSolutionNode_node_39_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[39]
        this_input_valid: bool = node_input_validity[39]
        if not this_input_valid:
            return

        # The solution of non-root node 39
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(Py/(a_0 + a_1*math.cos(th_1) - a_2*(math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2)) - d_3*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1)))) <= 70368744177665/70368744177664)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.cos(th_2)
                x4 = safe_asin(Py/(a_0 + a_1*x0 + a_2*(x0*x3 - x1*x2) + d_3*(-x0*x2 - x1*x3)))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x4
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(40, appended_idx)
                this_solution[0] = math.pi - x4
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(40, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_0_processor()
    # Finish code for explicit solution node 39

    # Code for equation all-zero dispatcher node 40
    def EquationAllZeroDispatcherNode_node_40_processor():
        this_node_input_index: List[int] = node_input_index[40]
        this_input_valid: bool = node_input_validity[40]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = False
            if not checked_result:  # To non-degenerate node
                add_input_index_to(41, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_40_processor()
    # Finish code for equation all-zero dispatcher node 40

    # Code for explicit solution node 41, solved variable is th_3
    def ExplicitSolutionNode_node_41_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[41]
        this_input_valid: bool = node_input_validity[41]
        if not this_input_valid:
            return

        # The solution of non-root node 41
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_3_processor()
    # Finish code for explicit solution node 41

    # Code for explicit solution node 31, solved variable is th_0
    def ExplicitSolutionNode_node_31_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[31]
        this_input_valid: bool = node_input_validity[31]
        if not this_input_valid:
            return

        # The solution of non-root node 31
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(a_0 + a_1*math.cos(th_1) + a_2*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2)) + d_3*(-math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.cos(th_2)
                x4 = 1/(a_0 + a_1*x0 + a_2*(x0*x3 - x1*x2) + d_3*(-x0*x2 - x1*x3))
                this_solution[0] = math.atan2(Py*x4, Px*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(32, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_0_processor()
    # Finish code for explicit solution node 31

    # Code for non-branch dispatcher node 32
    # Actually, there is no code

    # Code for explicit solution node 33, solved variable is th_3
    def ExplicitSolutionNode_node_33_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[32]
        this_input_valid: bool = node_input_validity[32]
        if not this_input_valid:
            return

        # The solution of non-root node 33
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_3_processor()
    # Finish code for explicit solution node 32

    # Code for explicit solution node 23, solved variable is th_0
    def ExplicitSolutionNode_node_23_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[23]
        this_input_valid: bool = node_input_validity[23]
        if not this_input_valid:
            return

        # The solution of non-root node 23
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(Px) >= 1.0e-6) or (abs(Py) >= 1.0e-6)
            if condition_0:
                x0 = th_1 + th_2
                x1 = -a_0 - a_1*math.cos(th_1) - a_2*math.cos(x0) + d_3*math.sin(x0)
                this_solution[0] = math.atan2(-Py*x1, -Px*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_0_processor()
    # Finish code for explicit solution node 23

    # Code for non-branch dispatcher node 24
    # Actually, there is no code

    # Code for explicit solution node 25, solved variable is th_3
    def ExplicitSolutionNode_node_25_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[24]
        this_input_valid: bool = node_input_validity[24]
        if not this_input_valid:
            return

        # The solution of non-root node 25
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_3_processor()
    # Finish code for explicit solution node 24

    # Code for explicit solution node 15, solved variable is th_0
    def ExplicitSolutionNode_node_15_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[15]
        this_input_valid: bool = node_input_validity[15]
        if not this_input_valid:
            return

        # The solution of non-root node 15
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(r_13) >= 1.0e-6) or (abs(r_23) >= 1.0e-6)
            if condition_0:
                x0 = math.sin(th_1 + th_2)
                this_solution[0] = math.atan2(-r_23*x0, -r_13*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(16, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_0_processor()
    # Finish code for explicit solution node 15

    # Code for non-branch dispatcher node 16
    # Actually, there is no code

    # Code for explicit solution node 17, solved variable is th_3
    def ExplicitSolutionNode_node_17_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[16]
        this_input_valid: bool = node_input_validity[16]
        if not this_input_valid:
            return

        # The solution of non-root node 17
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_3_processor()
    # Finish code for explicit solution node 16

    # Code for equation all-zero dispatcher node 10
    def EquationAllZeroDispatcherNode_node_10_processor():
        this_node_input_index: List[int] = node_input_index[10]
        this_input_valid: bool = node_input_validity[10]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = (abs(r_13) <= 1.0e-6) and (abs(r_23) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue)
            else:
                add_input_index_to(18, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor()
    # Finish code for equation all-zero dispatcher node 10

    # Code for equation all-zero dispatcher node 18
    def EquationAllZeroDispatcherNode_node_18_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = (abs(Px) <= 1.0e-6) and (abs(Py) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(19, node_input_i_idx_in_queue)
            else:
                add_input_index_to(26, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_18_processor()
    # Finish code for equation all-zero dispatcher node 18

    # Code for equation all-zero dispatcher node 26
    def EquationAllZeroDispatcherNode_node_26_processor():
        this_node_input_index: List[int] = node_input_index[26]
        this_input_valid: bool = node_input_validity[26]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            checked_result: bool = (abs((a_0 + a_1*math.cos(th_1) + a_2*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2)) + d_3*(-math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1)))**2) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(27, node_input_i_idx_in_queue)
            else:
                add_input_index_to(34, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_26_processor()
    # Finish code for equation all-zero dispatcher node 26

    # Code for equation all-zero dispatcher node 34
    def EquationAllZeroDispatcherNode_node_34_processor():
        this_node_input_index: List[int] = node_input_index[34]
        this_input_valid: bool = node_input_validity[34]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            checked_result: bool = (abs(a_0 + a_1*math.cos(th_1) + a_2*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2)) + d_3*(-math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(35, node_input_i_idx_in_queue)
            else:
                add_input_index_to(42, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_34_processor()
    # Finish code for equation all-zero dispatcher node 34

    # Code for equation all-zero dispatcher node 42
    def EquationAllZeroDispatcherNode_node_42_processor():
        this_node_input_index: List[int] = node_input_index[42]
        this_input_valid: bool = node_input_validity[42]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            checked_result: bool = (abs(r_13*math.cos(th_1)) <= 1.0e-6) and (abs(r_23*math.cos(th_1)) <= 1.0e-6) and (abs(r_33*math.sin(th_1) - math.sin(th_2)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(43, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_42_processor()
    # Finish code for equation all-zero dispatcher node 42

    # Code for explicit solution node 43, solved variable is th_0
    def ExplicitSolutionNode_node_43_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[43]
        this_input_valid: bool = node_input_validity[43]
        if not this_input_valid:
            return

        # The solution of non-root node 43
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(r_13*math.cos(th_1)) >= zero_tolerance) or (abs(r_23*math.cos(th_1)) >= zero_tolerance) or (abs(r_33*math.sin(th_1) - math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.atan2(-r_23*x0, -r_13*x0)
                x2 = x0**2
                x3 = -r_33*math.sin(th_1) + math.sin(th_2)
                x4 = safe_sqrt(r_13**2*x2 + r_23**2*x2 - x3**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x1 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(44, appended_idx)
                this_solution[0] = x1 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(44, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_43_solve_th_0_processor()
    # Finish code for explicit solution node 43

    # Code for equation all-zero dispatcher node 44
    def EquationAllZeroDispatcherNode_node_44_processor():
        this_node_input_index: List[int] = node_input_index[44]
        this_input_valid: bool = node_input_validity[44]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = False
            if not checked_result:  # To non-degenerate node
                add_input_index_to(45, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_44_processor()
    # Finish code for equation all-zero dispatcher node 44

    # Code for explicit solution node 45, solved variable is th_3
    def ExplicitSolutionNode_node_45_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[45]
        this_input_valid: bool = node_input_validity[45]
        if not this_input_valid:
            return

        # The solution of non-root node 45
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_45_solve_th_3_processor()
    # Finish code for explicit solution node 45

    # Code for explicit solution node 35, solved variable is th_0
    def ExplicitSolutionNode_node_35_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[35]
        this_input_valid: bool = node_input_validity[35]
        if not this_input_valid:
            return

        # The solution of non-root node 35
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(Py/(a_0 + a_1*math.cos(th_1) - a_2*(math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2)) - d_3*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1)))) <= 70368744177665/70368744177664)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.cos(th_2)
                x4 = safe_asin(Py/(a_0 + a_1*x0 + a_2*(x0*x3 - x1*x2) + d_3*(-x0*x2 - x1*x3)))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x4
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(36, appended_idx)
                this_solution[0] = math.pi - x4
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(36, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_0_processor()
    # Finish code for explicit solution node 35

    # Code for non-branch dispatcher node 36
    # Actually, there is no code

    # Code for explicit solution node 37, solved variable is th_3
    def ExplicitSolutionNode_node_37_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[36]
        this_input_valid: bool = node_input_validity[36]
        if not this_input_valid:
            return

        # The solution of non-root node 37
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_3_processor()
    # Finish code for explicit solution node 36

    # Code for explicit solution node 27, solved variable is th_0
    def ExplicitSolutionNode_node_27_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[27]
        this_input_valid: bool = node_input_validity[27]
        if not this_input_valid:
            return

        # The solution of non-root node 27
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(a_0 + a_1*math.cos(th_1) + a_2*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2)) + d_3*(-math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.cos(th_2)
                x4 = 1/(a_0 + a_1*x0 + a_2*(x0*x3 - x1*x2) + d_3*(-x0*x2 - x1*x3))
                this_solution[0] = math.atan2(Py*x4, Px*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_0_processor()
    # Finish code for explicit solution node 27

    # Code for non-branch dispatcher node 28
    # Actually, there is no code

    # Code for explicit solution node 29, solved variable is th_3
    def ExplicitSolutionNode_node_29_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[28]
        this_input_valid: bool = node_input_validity[28]
        if not this_input_valid:
            return

        # The solution of non-root node 29
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_3_processor()
    # Finish code for explicit solution node 28

    # Code for explicit solution node 19, solved variable is th_0
    def ExplicitSolutionNode_node_19_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[19]
        this_input_valid: bool = node_input_validity[19]
        if not this_input_valid:
            return

        # The solution of non-root node 19
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(Px) >= 1.0e-6) or (abs(Py) >= 1.0e-6)
            if condition_0:
                x0 = th_1 + th_2
                x1 = -a_0 - a_1*math.cos(th_1) - a_2*math.cos(x0) + d_3*math.sin(x0)
                this_solution[0] = math.atan2(-Py*x1, -Px*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_0_processor()
    # Finish code for explicit solution node 19

    # Code for non-branch dispatcher node 20
    # Actually, there is no code

    # Code for explicit solution node 21, solved variable is th_3
    def ExplicitSolutionNode_node_21_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[20]
        this_input_valid: bool = node_input_validity[20]
        if not this_input_valid:
            return

        # The solution of non-root node 21
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_3_processor()
    # Finish code for explicit solution node 20

    # Code for explicit solution node 11, solved variable is th_0
    def ExplicitSolutionNode_node_11_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[11]
        this_input_valid: bool = node_input_validity[11]
        if not this_input_valid:
            return

        # The solution of non-root node 11
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[3]
            condition_0: bool = (abs(r_13) >= 1.0e-6) or (abs(r_23) >= 1.0e-6)
            if condition_0:
                x0 = math.sin(th_1 + th_2)
                this_solution[0] = math.atan2(-r_23*x0, -r_13*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_0_processor()
    # Finish code for explicit solution node 11

    # Code for non-branch dispatcher node 12
    # Actually, there is no code

    # Code for explicit solution node 13, solved variable is th_3
    def ExplicitSolutionNode_node_13_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[12]
        this_input_valid: bool = node_input_validity[12]
        if not this_input_valid:
            return

        # The solution of non-root node 13
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[4] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_3_processor()
    # Finish code for explicit solution node 12

    # Code for explicit solution node 7, solved variable is th_3
    def ExplicitSolutionNode_node_7_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[7]
        this_input_valid: bool = node_input_validity[7]
        if not this_input_valid:
            return

        # The solution of non-root node 7
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1th_2_soa = this_solution[2]
            condition_0: bool = (abs(r_31) >= zero_tolerance) or (abs(r_32) >= zero_tolerance) or (abs(math.sin(th_1th_2_soa)) >= zero_tolerance)
            if condition_0:
                x0 = 1/math.sin(th_1th_2_soa)
                this_solution[4] = math.atan2(r_32*x0, -r_31*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_3_processor()
    # Finish code for explicit solution node 7

    # Code for non-branch dispatcher node 8
    # Actually, there is no code

    # Code for explicit solution node 9, solved variable is th_0
    def ExplicitSolutionNode_node_9_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return

        # The solution of non-root node 9
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1th_2_soa = this_solution[2]
            condition_0: bool = (abs(r_13) >= zero_tolerance) or (abs(r_23) >= zero_tolerance) or (abs(math.sin(th_1th_2_soa)) >= zero_tolerance)
            if condition_0:
                x0 = 1/math.sin(th_1th_2_soa)
                this_solution[0] = math.atan2(-r_23*x0, -r_13*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor()
    # Finish code for explicit solution node 8

    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = ik_out_i[0]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[1]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[3]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[4]  # th_3
        new_ik_i[3] = value_at_3
        ik_out.append(new_ik_i)
    return ik_out


def abb_irb460_110_240_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    T_ee_raw_in = abb_irb460_110_240_ik_target_original_to_raw(T_ee)
    ik_output_raw = abb_irb460_110_240_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = add_revolute_offset(ik_out_i[2], -th_2_offset_original2raw)
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ee_pose_i = abb_irb460_110_240_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -1.5707963267948966, -4.71238898038469, -6.283185307179586], [3.141592653589793, 4.71238898038469, 1.5707963267948966, 0.0], [True, True, True, True], [False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = abb_irb460_110_240_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
    return ik_output


def test_ik_solve_abb_irb460_110_240():
    theta_in = np.random.random(size=(4, ))
    ee_pose = abb_irb460_110_240_fk(theta_in)
    ik_output = abb_irb460_110_240_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = abb_irb460_110_240_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_abb_irb460_110_240()
