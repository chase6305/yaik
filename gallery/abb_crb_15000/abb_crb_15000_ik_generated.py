import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 28
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.444
a_1: float = 0.11
a_3: float = 0.08
d_2: float = 0.47
d_4: float = 0.101
pre_transform_special_symbol_23: float = 0.265

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def abb_crb_15000_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_13
    ee_transformed[0, 1] = -1.0*r_12
    ee_transformed[0, 2] = 1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_23
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = 1.0*r_33
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def abb_crb_15000_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_13
    ee_transformed[0, 1] = -1.0*r_12
    ee_transformed[0, 2] = 1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_23
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = 1.0*r_33
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def abb_crb_15000_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_4)
    x1 = math.cos(th_0)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x5*x6
    x8 = -x1*x4 - x1*x7
    x9 = x0*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_0)
    x12 = math.sin(th_3)
    x13 = math.cos(th_3)
    x14 = x2*x5
    x15 = -x1*x14 + x1*x3*x6
    x16 = x11*x12 + x13*x15
    x17 = x10*x16
    x18 = math.cos(th_5)
    x19 = 1.0*x11*x13 - 1.0*x12*x15
    x20 = math.sin(th_5)
    x21 = 1.0*x0*x16 - 1.0*x10*x8
    x22 = 1.0*a_0
    x23 = x22*x6
    x24 = 1.0*a_1
    x25 = 1.0*d_2
    x26 = 1.0*d_4
    x27 = -x11*x4 - x11*x7
    x28 = x0*x27
    x29 = -x11*x14 + x11*x3*x6
    x30 = -x1*x12 + x13*x29
    x31 = x10*x30
    x32 = -1.0*x1*x13 - 1.0*x12*x29
    x33 = 1.0*x0*x30 - 1.0*x10*x27
    x34 = x14 - x3*x6
    x35 = x0*x34
    x36 = -x4 - x7
    x37 = x10*x13*x36
    x38 = 1.0*x12*x36
    x39 = 1.0*x0*x13*x36 - 1.0*x10*x34
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x17 - 1.0*x9
    ee_pose[0, 1] = x18*x19 + x20*x21
    ee_pose[0, 2] = x18*x21 - x19*x20
    ee_pose[0, 3] = a_3*x21 + x1*x23 + x15*x24 + x25*x8 + x26*(-x17 - x9)
    ee_pose[1, 0] = -1.0*x28 - 1.0*x31
    ee_pose[1, 1] = x18*x32 + x20*x33
    ee_pose[1, 2] = x18*x33 - x20*x32
    ee_pose[1, 3] = a_3*x33 + x11*x23 + x24*x29 + x25*x27 + x26*(-x28 - x31)
    ee_pose[2, 0] = -1.0*x35 - 1.0*x37
    ee_pose[2, 1] = -x18*x38 + x20*x39
    ee_pose[2, 2] = x18*x39 + x20*x38
    ee_pose[2, 3] = a_3*x39 + 1.0*pre_transform_special_symbol_23 - x2*x22 + x24*x36 + x25*x34 + x26*(-x35 - x37)
    return ee_pose


def abb_crb_15000_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = -x1
    x3 = math.cos(th_2)
    x4 = math.sin(th_1)
    x5 = math.cos(th_0)
    x6 = 1.0*x5
    x7 = x4*x6
    x8 = math.sin(th_2)
    x9 = math.cos(th_1)
    x10 = x6*x9
    x11 = -x10*x8 - x3*x7
    x12 = math.cos(th_3)
    x13 = math.sin(th_3)
    x14 = 1.0*x3*x5*x9 - x7*x8
    x15 = 1.0*x0*x12 - x13*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x1*x13 + x12*x14
    x19 = -x11*x16 - x17*x18
    x20 = x1*x4
    x21 = x1*x9
    x22 = -x20*x3 - x21*x8
    x23 = 1.0*x0*x3*x9 - x20*x8
    x24 = -x12*x6 - x13*x23
    x25 = x12*x23 - x13*x6
    x26 = -x16*x22 - x17*x25
    x27 = 1.0*x8
    x28 = 1.0*x3
    x29 = x27*x4 - x28*x9
    x30 = -x27*x9 - x28*x4
    x31 = x13*x30
    x32 = -x12*x17*x30 - x16*x29
    x33 = 1.0*a_0
    x34 = x33*x4
    x35 = pre_transform_special_symbol_23 - x34
    x36 = a_1*x30 + d_2*x29 + pre_transform_special_symbol_23 - x34
    x37 = a_0*x21 + a_1*x23 + d_2*x22
    x38 = a_3*(x12*x16*x30 - x17*x29) + d_4*x32 + x36
    x39 = a_3*(x16*x25 - x17*x22) + d_4*x26 + x37
    x40 = a_0*x10 + a_1*x14 + d_2*x11
    x41 = a_3*(-x11*x17 + x16*x18) + d_4*x19 + x40
    x42 = x33*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x11
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x19
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x6
    jacobian_output[1, 3] = x22
    jacobian_output[1, 4] = x24
    jacobian_output[1, 5] = x26
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x29
    jacobian_output[2, 4] = -x31
    jacobian_output[2, 5] = x32
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x6
    jacobian_output[3, 2] = -x35*x6
    jacobian_output[3, 3] = -x22*x36 + x29*x37
    jacobian_output[3, 4] = -x24*x36 - x31*x37
    jacobian_output[3, 5] = -x26*x38 + x32*x39
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x35
    jacobian_output[4, 3] = x11*x36 - x29*x40
    jacobian_output[4, 4] = x15*x36 + x31*x40
    jacobian_output[4, 5] = x19*x38 - x32*x41
    jacobian_output[5, 2] = x0**2*x42 + x42*x5**2
    jacobian_output[5, 3] = -x11*x37 + x22*x40
    jacobian_output[5, 4] = -x15*x37 + x24*x40
    jacobian_output[5, 5] = -x19*x39 + x26*x41
    return jacobian_output


def abb_crb_15000_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = -x1
    x3 = math.cos(th_2)
    x4 = math.sin(th_1)
    x5 = math.cos(th_0)
    x6 = 1.0*x5
    x7 = x4*x6
    x8 = math.sin(th_2)
    x9 = math.cos(th_1)
    x10 = -x3*x7 - x6*x8*x9
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = 1.0*x3*x5*x9 - x7*x8
    x14 = math.cos(th_4)
    x15 = math.sin(th_4)
    x16 = x1*x4
    x17 = -x1*x8*x9 - x16*x3
    x18 = 1.0*x0*x3*x9 - x16*x8
    x19 = 1.0*x8
    x20 = 1.0*x3
    x21 = x19*x4 - x20*x9
    x22 = -x19*x9 - x20*x4
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = 1.0*x0*x11 - x12*x13
    jacobian_output[0, 5] = -x10*x14 - x15*(x1*x12 + x11*x13)
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x6
    jacobian_output[1, 3] = x17
    jacobian_output[1, 4] = -x11*x6 - x12*x18
    jacobian_output[1, 5] = -x14*x17 - x15*(x11*x18 - x12*x6)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x21
    jacobian_output[2, 4] = -x12*x22
    jacobian_output[2, 5] = -x11*x15*x22 - x14*x21
    return jacobian_output


def abb_crb_15000_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw
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
    x6 = a_0*x5
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
    x18 = -x15 - x16
    x19 = a_1*x18 + d_2*x13 + pre_transform_special_symbol_23 - x6
    x20 = a_0*x12
    x21 = 1.0*x10*x11*x14 - x14*x9
    x22 = a_1*x21 + d_2*x17 + x14*x20
    x23 = math.sin(th_3)
    x24 = x18*x23
    x25 = math.cos(th_3)
    x26 = -x2*x25 - x21*x23
    x27 = math.cos(th_4)
    x28 = math.sin(th_4)
    x29 = -x13*x27 - x18*x25*x28
    x30 = -x2*x23 + x21*x25
    x31 = -x17*x27 - x28*x30
    x32 = a_3*(-x13*x28 + x18*x25*x27) + d_4*x29 + x19
    x33 = a_3*(-x17*x28 + x27*x30) + d_4*x31 + x22
    x34 = 1.0*p_on_ee_x
    x35 = 1.0*x14
    x36 = p_on_ee_z*x35
    x37 = x2*x4
    x38 = x11*x2
    x39 = -x10*x37 - x38*x8
    x40 = 1.0*x1*x10*x11 - x37*x8
    x41 = a_0*x38 + a_1*x40 + d_2*x39
    x42 = 1.0*x14*x25 - x23*x40
    x43 = x23*x35 + x25*x40
    x44 = -x27*x39 - x28*x43
    x45 = a_3*(x27*x43 - x28*x39) + d_4*x44 + x41
    x46 = x1*x34
    x47 = x0*x14
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x7 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x26 - x19*x26 - x22*x24
    jacobian_output[0, 5] = -p_on_ee_y*x29 + p_on_ee_z*x31 + x29*x33 - x31*x32
    jacobian_output[1, 0] = x34
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x35 + x36
    jacobian_output[1, 2] = -x35*x7 + x36
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x39 - x13*x41 + x19*x39
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x42 + x18*x23*x41 + x19*x42
    jacobian_output[1, 5] = p_on_ee_x*x29 - p_on_ee_z*x44 - x29*x45 + x32*x44
    jacobian_output[2, 1] = -x46 - x47
    jacobian_output[2, 2] = x1**2*x20 + x14**2*x20 - x46 - x47
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x39 + x17*x41 - x22*x39
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x42 - x22*x42 + x26*x41
    jacobian_output[2, 5] = -p_on_ee_x*x31 + p_on_ee_y*x44 + x31*x45 - x33*x44
    return jacobian_output


def abb_crb_15000_ik_solve_raw(T_ee: np.ndarray):
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
        return IkSolution([math.nan] * 8)

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
    for i in range(28):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for non-branch dispatcher node 0
    # Actually, there is no code

    # Code for explicit solution node 1, solved variable is th_0
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 3] = -a_0
        R_l[0, 7] = -a_1
        R_l[1, 2] = -a_0
        R_l[1, 6] = -a_1
        R_l[2, 4] = a_0
        R_l[3, 6] = -1
        R_l[4, 7] = 1
        R_l[5, 5] = 2*a_0*a_1
        R_l[6, 1] = -a_0
        R_l[7, 0] = -a_0
        try:
            R_l_mat_inv = np.linalg.inv(R_l)
        except:
            return
        R_l_inv_00 = R_l_mat_inv[0, 0]
        R_l_inv_01 = R_l_mat_inv[0, 1]
        R_l_inv_02 = R_l_mat_inv[0, 2]
        R_l_inv_03 = R_l_mat_inv[0, 3]
        R_l_inv_04 = R_l_mat_inv[0, 4]
        R_l_inv_05 = R_l_mat_inv[0, 5]
        R_l_inv_06 = R_l_mat_inv[0, 6]
        R_l_inv_07 = R_l_mat_inv[0, 7]
        R_l_inv_10 = R_l_mat_inv[1, 0]
        R_l_inv_11 = R_l_mat_inv[1, 1]
        R_l_inv_12 = R_l_mat_inv[1, 2]
        R_l_inv_13 = R_l_mat_inv[1, 3]
        R_l_inv_14 = R_l_mat_inv[1, 4]
        R_l_inv_15 = R_l_mat_inv[1, 5]
        R_l_inv_16 = R_l_mat_inv[1, 6]
        R_l_inv_17 = R_l_mat_inv[1, 7]
        R_l_inv_20 = R_l_mat_inv[2, 0]
        R_l_inv_21 = R_l_mat_inv[2, 1]
        R_l_inv_22 = R_l_mat_inv[2, 2]
        R_l_inv_23 = R_l_mat_inv[2, 3]
        R_l_inv_24 = R_l_mat_inv[2, 4]
        R_l_inv_25 = R_l_mat_inv[2, 5]
        R_l_inv_26 = R_l_mat_inv[2, 6]
        R_l_inv_27 = R_l_mat_inv[2, 7]
        R_l_inv_30 = R_l_mat_inv[3, 0]
        R_l_inv_31 = R_l_mat_inv[3, 1]
        R_l_inv_32 = R_l_mat_inv[3, 2]
        R_l_inv_33 = R_l_mat_inv[3, 3]
        R_l_inv_34 = R_l_mat_inv[3, 4]
        R_l_inv_35 = R_l_mat_inv[3, 5]
        R_l_inv_36 = R_l_mat_inv[3, 6]
        R_l_inv_37 = R_l_mat_inv[3, 7]
        R_l_inv_40 = R_l_mat_inv[4, 0]
        R_l_inv_41 = R_l_mat_inv[4, 1]
        R_l_inv_42 = R_l_mat_inv[4, 2]
        R_l_inv_43 = R_l_mat_inv[4, 3]
        R_l_inv_44 = R_l_mat_inv[4, 4]
        R_l_inv_45 = R_l_mat_inv[4, 5]
        R_l_inv_46 = R_l_mat_inv[4, 6]
        R_l_inv_47 = R_l_mat_inv[4, 7]
        R_l_inv_50 = R_l_mat_inv[5, 0]
        R_l_inv_51 = R_l_mat_inv[5, 1]
        R_l_inv_52 = R_l_mat_inv[5, 2]
        R_l_inv_53 = R_l_mat_inv[5, 3]
        R_l_inv_54 = R_l_mat_inv[5, 4]
        R_l_inv_55 = R_l_mat_inv[5, 5]
        R_l_inv_56 = R_l_mat_inv[5, 6]
        R_l_inv_57 = R_l_mat_inv[5, 7]
        R_l_inv_60 = R_l_mat_inv[6, 0]
        R_l_inv_61 = R_l_mat_inv[6, 1]
        R_l_inv_62 = R_l_mat_inv[6, 2]
        R_l_inv_63 = R_l_mat_inv[6, 3]
        R_l_inv_64 = R_l_mat_inv[6, 4]
        R_l_inv_65 = R_l_mat_inv[6, 5]
        R_l_inv_66 = R_l_mat_inv[6, 6]
        R_l_inv_67 = R_l_mat_inv[6, 7]
        R_l_inv_70 = R_l_mat_inv[7, 0]
        R_l_inv_71 = R_l_mat_inv[7, 1]
        R_l_inv_72 = R_l_mat_inv[7, 2]
        R_l_inv_73 = R_l_mat_inv[7, 3]
        R_l_inv_74 = R_l_mat_inv[7, 4]
        R_l_inv_75 = R_l_mat_inv[7, 5]
        R_l_inv_76 = R_l_mat_inv[7, 6]
        R_l_inv_77 = R_l_mat_inv[7, 7]

        # Temp variable for efficiency
        x0 = -r_23
        x1 = 2*r_21
        x2 = -x1
        x3 = 4*r_22
        x4 = a_3*r_21
        x5 = d_2*r_23
        x6 = r_21**2
        x7 = Py*x6
        x8 = r_22**2
        x9 = Py*x8
        x10 = r_23**2
        x11 = Py*x10
        x12 = d_4*r_23
        x13 = Px*r_11
        x14 = r_21*x13
        x15 = Px*r_12
        x16 = r_22*x15
        x17 = Px*r_13
        x18 = r_23*x17
        x19 = Pz*r_31
        x20 = r_21*x19
        x21 = Pz*r_32
        x22 = r_22*x21
        x23 = Pz*r_33
        x24 = r_23*x23
        x25 = x11 - x12 + x14 + x16 + x18 + x20 + x22 + x24 + x7 + x9
        x26 = x25 - x5
        x27 = x26 + x4
        x28 = a_3*r_22
        x29 = 2*x28
        x30 = -x4
        x31 = x26 + x30
        x32 = d_2*x1
        x33 = -x32
        x34 = d_2*x3
        x35 = x25 + x5
        x36 = x35 + x4
        x37 = x30 + x35
        x38 = R_l_inv_50*a_0
        x39 = a_3*x38
        x40 = Py*r_21
        x41 = x13 + x19 + x40
        x42 = x38*x41
        x43 = Py*r_23
        x44 = -d_4 + x17 + x23 + x43
        x45 = R_l_inv_52*a_0
        x46 = x44*x45
        x47 = d_4*r_21
        x48 = r_23*x13
        x49 = r_23*x19
        x50 = r_21*x17
        x51 = r_21*x23
        x52 = x47 + x48 + x49 - x50 - x51
        x53 = R_l_inv_57*a_0
        x54 = x52*x53
        x55 = a_0*r_22
        x56 = R_l_inv_54*x55
        x57 = -x56
        x58 = d_2*r_22
        x59 = R_l_inv_56*a_0
        x60 = x58*x59
        x61 = a_3*r_23
        x62 = x53*x61
        x63 = R_l_inv_55*a_0
        x64 = 2*d_2
        x65 = x44*x64
        x66 = x63*x65
        x67 = x39 + x42 + x46 + x54 + x57 + x60 + x62 + x66
        x68 = a_0*r_21
        x69 = R_l_inv_53*x68
        x70 = d_4*r_22
        x71 = r_23*x15
        x72 = r_23*x21
        x73 = r_22*x17
        x74 = r_22*x23
        x75 = x70 + x71 + x72 - x73 - x74
        x76 = x59*x75
        x77 = R_l_inv_57*d_2*x68
        x78 = r_22*x13
        x79 = r_22*x19
        x80 = r_21*x15
        x81 = r_21*x21
        x82 = x78 + x79 - x80 - x81
        x83 = x28 + x69 + x76 + x77 + x82
        x84 = a_3**2
        x85 = d_2**2
        x86 = d_4**2
        x87 = a_0**2
        x88 = a_1**2
        x89 = 2*d_4
        x90 = 2*x12
        x91 = Py*x1
        x92 = Py*r_22
        x93 = 2*x15
        x94 = 2*x17
        x95 = 2*x13
        x96 = 2*x21
        x97 = 2*x23
        x98 = Px**2
        x99 = r_11**2
        x100 = x98*x99
        x101 = r_12**2
        x102 = x101*x98
        x103 = r_13**2
        x104 = x103*x98
        x105 = Py**2
        x106 = x105*x6
        x107 = x105*x8
        x108 = x10*x105
        x109 = Pz**2
        x110 = r_31**2
        x111 = x109*x110
        x112 = r_32**2
        x113 = x109*x112
        x114 = r_33**2*x109
        x115 = -Py*x90 + x100 + x102 + x104 + x106 + x107 + x108 + x111 + x113 + x114 + x13*x91 - x17*x89 + x19*x91 + x19*x95 + x21*x93 - x23*x89 + x23*x94 + x43*x94 + x43*x97 + x84 + x85 + x86 - x87 - x88 + x92*x93 + x92*x96
        x116 = -R_l_inv_52*a_0*d_2 - R_l_inv_55*a_0*x115 - a_1
        x117 = x15 + x21 + x92
        x118 = a_0*x117
        x119 = R_l_inv_51*x118
        x120 = 2*a_3
        x121 = x41*x63
        x122 = x120*x121
        x123 = -x119 - x122
        x124 = 2*R_l_inv_50*x118
        x125 = 2*x41
        x126 = R_l_inv_51*a_0*x125
        x127 = -4*R_l_inv_55*a_0*a_3*x117 + x124 + x126
        x128 = R_l_inv_54*a_0
        x129 = x1*x128
        x130 = 2*x53
        x131 = x130*x75
        x132 = x32*x59
        x133 = x129 + x131 - x132
        x134 = a_3*x1
        x135 = 2*x59
        x136 = x135*x52
        x137 = 2*R_l_inv_53
        x138 = x137*x55
        x139 = x130*x58
        x140 = -x134 - x136 + x138 + x139
        x141 = d_2*x45
        x142 = x115*x63
        x143 = -x62
        x144 = a_1 + x141 + x142 + x143 + x42
        x145 = -x39
        x146 = -x46
        x147 = -x66
        x148 = x145 + x146 + x147 + x54 + x57 + x60
        x149 = -x78 - x79 + x80 + x81
        x150 = x149 + x28 + x69 + x76 + x77
        x151 = x123 + x150
        x152 = a_0*r_23*x137
        x153 = x135*x82
        x154 = x130*x5
        x155 = -2*R_l_inv_50*a_0*x44
        x156 = x120*x45
        x157 = x125*x45
        x158 = 4*a_3
        x159 = d_2*x158*x63
        x160 = 4*d_2
        x161 = x121*x160
        x162 = x155 + x156 + x157 + x159 + x161
        x163 = 2*x70
        x164 = 2*x73
        x165 = 2*x74
        x166 = 2*x71
        x167 = 2*x72
        x168 = x29*x59
        x169 = x163 - x164 - x165 + x166 + x167 - x168
        x170 = 4*x47
        x171 = 4*x48
        x172 = 4*x49
        x173 = 4*x50
        x174 = 4*x51
        x175 = 4*x4
        x176 = x175*x59
        x177 = 8*d_2
        x178 = R_l_inv_55*x118
        x179 = 4*R_l_inv_52*x118 + x177*x178
        x180 = x152 + x153 + x154
        x181 = x169 + x180
        x182 = x119 + x122
        x183 = x182 + x83
        x184 = -x54 + x56 - x60
        x185 = x39 + x46 + x66
        x186 = x184 + x185
        x187 = -x129 - x131 + x132
        x188 = x124 - x126 + x158*x178
        x189 = x116 + x150 + x182
        x190 = a_0*a_1
        x191 = 2*x190
        x192 = x87 + x88
        x193 = R_l_inv_62*x192
        x194 = R_l_inv_22*x191 + x193
        x195 = d_2*x194
        x196 = r_21*x84
        x197 = r_21**3*x105
        x198 = R_l_inv_25*x191 + R_l_inv_65*x192
        x199 = x115*x198
        x200 = -r_21*x85
        x201 = -r_21*x86
        x202 = R_l_inv_23*x191 + R_l_inv_63*x192
        x203 = -r_21*x202
        x204 = R_l_inv_60*x192
        x205 = R_l_inv_20*x191 + x204
        x206 = x205*x41
        x207 = -x206
        x208 = R_l_inv_66*x192
        x209 = R_l_inv_26*x191 + x208
        x210 = -x209*x75
        x211 = r_21*x100
        x212 = r_21*x107
        x213 = r_21*x108
        x214 = r_21*x111
        x215 = R_l_inv_67*x192
        x216 = R_l_inv_27*x191 + x215
        x217 = x216*x61
        x218 = -x217
        x219 = d_2*r_21
        x220 = -x216*x219
        x221 = -r_21*x102
        x222 = -r_21*x104
        x223 = -r_21*x113
        x224 = -r_21*x114
        x225 = x120*x5
        x226 = -x225
        x227 = -x13*x90
        x228 = -x19*x90
        x229 = x7*x95
        x230 = x9*x95
        x231 = x11*x95
        x232 = d_4*x1
        x233 = x17*x232
        x234 = 2*x19
        x235 = x234*x7
        x236 = x234*x9
        x237 = x11*x234
        x238 = x23*x232
        x239 = 2*r_11
        x240 = r_12*x98
        x241 = r_22*x240
        x242 = x239*x241
        x243 = 2*r_13
        x244 = r_23*x98
        x245 = r_11*x243*x244
        x246 = 2*r_31
        x247 = r_32*x109
        x248 = r_22*x247
        x249 = x246*x248
        x250 = r_23*r_33
        x251 = x109*x250
        x252 = x246*x251
        x253 = x15*x21
        x254 = -x1*x253
        x255 = x17*x23
        x256 = -x1*x255
        x257 = x13*x19
        x258 = x1*x257
        x259 = x22*x95
        x260 = x24*x95
        x261 = x16*x234
        x262 = x18*x234
        x263 = x195 + x196 + x197 + x199 + x200 + x201 + x203 + x207 + x210 + x211 + x212 + x213 + x214 + x218 + x220 + x221 + x222 + x223 + x224 + x226 + x227 + x228 + x229 + x230 + x231 + x233 + x235 + x236 + x237 + x238 + x242 + x245 + x249 + x252 + x254 + x256 + x258 + x259 + x260 + x261 + x262
        x264 = R_l_inv_61*x192
        x265 = x117*(R_l_inv_21*x191 + x264)
        x266 = a_3*x205
        x267 = -x266
        x268 = x194*x44
        x269 = -x268
        x270 = x198*x65
        x271 = -x270
        x272 = x198*x41
        x273 = x120*x272
        x274 = x265 + x267 + x269 + x271 + x273
        x275 = R_l_inv_24*x191 + R_l_inv_64*x192
        x276 = r_22*x275
        x277 = x216*x52
        x278 = x209*x58
        x279 = d_4*x32
        x280 = x5*x95
        x281 = x234*x5
        x282 = x17*x32
        x283 = x23*x32
        x284 = x276 - x277 - x278 - x279 - x280 - x281 + x282 + x283
        x285 = x12*x120
        x286 = x120*x7
        x287 = x120*x9
        x288 = x11*x120
        x289 = x13*x134
        x290 = x15*x29
        x291 = x120*x18
        x292 = x134*x19
        x293 = x21*x29
        x294 = x120*x24
        x295 = -x285 + x286 + x287 + x288 + x289 + x290 + x291 + x292 + x293 + x294
        x296 = x284 + x295
        x297 = 4*x190
        x298 = R_l_inv_26*x297 + 2*x208
        x299 = x298*x52
        x300 = R_l_inv_20*x297 + 2*x204
        x301 = x117*x300
        x302 = r_22*x85
        x303 = -2*x302
        x304 = r_22*x86
        x305 = -2*x304
        x306 = 2*x202
        x307 = -r_22*x306
        x308 = r_22*x84
        x309 = 2*x308
        x310 = r_22**3*x105
        x311 = 2*x310
        x312 = 2*x216
        x313 = -x312*x58
        x314 = r_22*x100
        x315 = -2*x314
        x316 = r_22*x104
        x317 = -2*x316
        x318 = r_22*x111
        x319 = -2*x318
        x320 = r_22*x114
        x321 = -2*x320
        x322 = r_22*x102
        x323 = 2*x322
        x324 = r_22*x106
        x325 = 2*x324
        x326 = r_22*x108
        x327 = 2*x326
        x328 = r_22*x113
        x329 = 2*x328
        x330 = 4*x15
        x331 = -x12*x330
        x332 = 4*x21
        x333 = -x12*x332
        x334 = x330*x7
        x335 = x330*x9
        x336 = x11*x330
        x337 = d_4*x3
        x338 = x17*x337
        x339 = x332*x7
        x340 = x332*x9
        x341 = x11*x332
        x342 = x23*x337
        x343 = 4*r_11
        x344 = r_21*x343
        x345 = x240*x344
        x346 = 4*r_12
        x347 = r_13*x244
        x348 = x346*x347
        x349 = 4*x247
        x350 = r_21*r_31
        x351 = x349*x350
        x352 = x250*x349
        x353 = -x257*x3
        x354 = -x255*x3
        x355 = x14*x332
        x356 = x20*x330
        x357 = x253*x3
        x358 = x24*x330
        x359 = x18*x332
        x360 = x299 - x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x325 + x327 + x329 + x331 + x333 + x334 + x335 + x336 + x338 + x339 + x340 + x341 + x342 + x345 + x348 + x351 + x352 + x353 + x354 + x355 + x356 + x357 + x358 + x359
        x361 = x41*(R_l_inv_21*x297 + 2*x264)
        x362 = x158*x198
        x363 = x117*x362 - x361
        x364 = R_l_inv_27*x297 + 2*x215
        x365 = x364*x75
        x366 = x1*x275
        x367 = d_4*x34
        x368 = x209*x32
        x369 = x330*x5
        x370 = x332*x5
        x371 = x17*x34
        x372 = x23*x34
        x373 = -x365 - x366 - x367 + x368 - x369 - x370 + x371 + x372
        x374 = -x286
        x375 = -x287
        x376 = -x288
        x377 = -x289
        x378 = -x290
        x379 = -x291
        x380 = -x292
        x381 = -x293
        x382 = -x294
        x383 = x266 + x268 + x270 + x285 + x374 + x375 + x376 + x377 + x378 + x379 + x380 + x381 + x382
        x384 = -d_2*x194
        x385 = -x115*x198
        x386 = x196 + x197 + x200 + x201 + x203 + x207 + x210 + x211 + x212 + x213 + x214 + x217 + x220 + x221 + x222 + x223 + x224 + x225 + x227 + x228 + x229 + x230 + x231 + x233 + x235 + x236 + x237 + x238 + x242 + x245 + x249 + x252 + x254 + x256 + x258 + x259 + x260 + x261 + x262 + x384 + x385
        x387 = x265 + x273
        x388 = 4*d_4
        x389 = x388*x4
        x390 = x209*x29
        x391 = x158*x48
        x392 = x158*x49
        x393 = x17*x175
        x394 = x175*x23
        x395 = x41*(R_l_inv_22*x297 + 2*x193)
        x396 = -x300*x44
        x397 = x120*x194
        x398 = d_2*x362
        x399 = x160*x272
        x400 = x395 + x396 + x397 + x398 + x399
        x401 = x298*x82
        x402 = r_23*x84
        x403 = 2*x402
        x404 = r_23*x85
        x405 = 2*x404
        x406 = r_23*x306
        x407 = r_23*x86
        x408 = 2*x407
        x409 = r_23**3*x105
        x410 = 2*x409
        x411 = x388*x7
        x412 = x388*x9
        x413 = x11*x388
        x414 = x312*x5
        x415 = r_23*x100
        x416 = 2*x415
        x417 = r_23*x102
        x418 = 2*x417
        x419 = r_23*x111
        x420 = 2*x419
        x421 = r_23*x113
        x422 = 2*x421
        x423 = r_23*x104
        x424 = 2*x423
        x425 = r_23*x106
        x426 = 2*x425
        x427 = r_23*x107
        x428 = 2*x427
        x429 = r_23*x114
        x430 = 2*x429
        x431 = x13*x170
        x432 = x15*x337
        x433 = 4*x17
        x434 = x12*x433
        x435 = x170*x19
        x436 = x21*x337
        x437 = 4*x23
        x438 = x12*x437
        x439 = x433*x7
        x440 = x433*x9
        x441 = x11*x433
        x442 = x437*x7
        x443 = x437*x9
        x444 = x11*x437
        x445 = r_13*x98
        x446 = x344*x445
        x447 = r_13*x3
        x448 = x240*x447
        x449 = r_33*x109
        x450 = 4*x350*x449
        x451 = x247*x3
        x452 = r_33*x451
        x453 = x171*x19
        x454 = x332*x71
        x455 = x14*x437
        x456 = x15*x3
        x457 = x23*x456
        x458 = x20*x433
        x459 = x21*x3
        x460 = x17*x459
        x461 = x18*x437
        x462 = -x401 - x403 - x405 - x406 + x408 + x410 - x411 - x412 - x413 - x414 - x416 - x418 - x420 - x422 + x424 + x426 + x428 + x430 - x431 - x432 - x434 - x435 - x436 - x438 + x439 + x440 + x441 + x442 + x443 + x444 + x446 + x448 + x450 + x452 - x453 - x454 + x455 + x457 + x458 + x460 + x461
        x463 = 8*d_4
        x464 = x28*x463
        x465 = x175*x209
        x466 = 8*a_3
        x467 = x466*x71
        x468 = x466*x72
        x469 = 8*x28
        x470 = x17*x469
        x471 = x23*x469
        x472 = 8*x190
        x473 = x117*x177
        x474 = x117*(R_l_inv_22*x472 + 4*x193) + x198*x473
        x475 = x389 + x390 + x391 + x392 - x393 - x394
        x476 = x462 + x475
        x477 = -x265 - x273
        x478 = x267 + x269 + x271 + x477
        x479 = -x276 + x277 + x278 + x279 + x280 + x281 - x282 - x283
        x480 = x295 + x479
        x481 = -4*a_3*x117*x198 + x361
        x482 = x365 + x366 + x367 - x368 + x369 + x370 - x371 - x372
        x483 = R_l_inv_72*x192
        x484 = R_l_inv_32*x191 + x483
        x485 = d_2*x484
        x486 = R_l_inv_34*x191 + R_l_inv_74*x192
        x487 = r_22*x486
        x488 = R_l_inv_35*x191 + R_l_inv_75*x192
        x489 = x115*x488
        x490 = -x308
        x491 = -x302
        x492 = -x304
        x493 = R_l_inv_70*x192
        x494 = R_l_inv_30*x191 + x493
        x495 = x41*x494
        x496 = -x495
        x497 = R_l_inv_77*x192
        x498 = R_l_inv_37*x191 + x497
        x499 = -x498*x52
        x500 = R_l_inv_76*x192
        x501 = R_l_inv_36*x191 + x500
        x502 = -x501*x58
        x503 = -x314
        x504 = -x316
        x505 = -x318
        x506 = -x320
        x507 = -x15*x90
        x508 = -x21*x90
        x509 = x7*x93
        x510 = x9*x93
        x511 = x11*x93
        x512 = x163*x17
        x513 = x7*x96
        x514 = x9*x96
        x515 = x11*x96
        x516 = x163*x23
        x517 = r_11*x240
        x518 = x1*x517
        x519 = r_23*x240*x243
        x520 = r_31*x1
        x521 = x247*x520
        x522 = x247*x250
        x523 = 2*x522
        x524 = -x234*x78
        x525 = -x164*x23
        x526 = x1*x13
        x527 = x21*x526
        x528 = x1*x19
        x529 = x15*x528
        x530 = x16*x96
        x531 = x24*x93
        x532 = x18*x96
        x533 = x310 + x322 + x324 + x326 + x328 + x485 + x487 + x489 + x490 + x491 + x492 + x496 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532
        x534 = R_l_inv_71*x192
        x535 = x117*(R_l_inv_31*x191 + x534)
        x536 = a_3*x494
        x537 = -x536
        x538 = x44*x484
        x539 = -x538
        x540 = x488*x65
        x541 = -x540
        x542 = x41*x488
        x543 = x120*x542
        x544 = x535 + x537 + x539 + x541 + x543
        x545 = x498*x61
        x546 = x13*x29
        x547 = x19*x29
        x548 = x134*x15
        x549 = x134*x21
        x550 = -x545 - x546 - x547 + x548 + x549
        x551 = R_l_inv_33*x191 + R_l_inv_73*x192
        x552 = r_21*x551
        x553 = x501*x75
        x554 = x219*x498
        x555 = x64*x70
        x556 = x5*x93
        x557 = x5*x96
        x558 = x58*x94
        x559 = x58*x97
        x560 = -x552 - x553 - x554 - x555 - x556 - x557 + x558 + x559
        x561 = x550 + x560
        x562 = R_l_inv_30*x297 + 2*x493
        x563 = x117*x562
        x564 = R_l_inv_37*x297 + 2*x497
        x565 = x564*x75
        x566 = x1*x84
        x567 = -x566
        x568 = x1*x85
        x569 = -x568
        x570 = x1*x486
        x571 = -x32*x501
        x572 = 4*x13
        x573 = 4*x19
        x574 = 4*r_31*x251 + r_31*x451 + x1*x100 - x1*x102 - x1*x104 + x1*x107 + x1*x108 + x1*x111 - x1*x113 - x1*x114 - x1*x86 + x11*x572 + x11*x573 - x12*x572 - x12*x573 + x13*x459 + x14*x573 + x17*x170 + x170*x23 - x173*x23 + x18*x573 + x19*x456 + 2*x197 + x24*x572 + x3*x517 - x332*x80 + x343*x347 + x572*x7 + x572*x9 + x573*x7 + x573*x9
        x575 = x563 + x565 + x567 + x569 + x570 + x571 + x574
        x576 = R_l_inv_31*x297 + 2*x534
        x577 = x158*x488
        x578 = x117*x577
        x579 = x41*x576 - x578
        x580 = R_l_inv_36*x297 + 2*x500
        x581 = x52*x580
        x582 = 2*x551
        x583 = r_22*x582
        x584 = d_2*x170
        x585 = 2*x498
        x586 = x58*x585
        x587 = x5*x572
        x588 = x5*x573
        x589 = d_2*x173
        x590 = d_2*x174
        x591 = -x581 + x583 - x584 + x586 - x587 - x588 + x589 + x590
        x592 = -x548
        x593 = -x549
        x594 = x536 + x538 + x540 + x545 + x546 + x547 + x592 + x593
        x595 = -x485
        x596 = -x489
        x597 = x310 + x322 + x324 + x326 + x328 + x487 + x490 + x491 + x492 + x496 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532 + x595 + x596
        x598 = x535 + x543
        x599 = x580*x82
        x600 = r_23*x582
        x601 = x29*x501
        x602 = x5*x585
        x603 = a_3*x34
        x604 = -x160*x80
        x605 = -x160*x81
        x606 = x13*x34
        x607 = x19*x34
        x608 = x599 + x600 + x601 + x602 + x603 + x604 + x605 + x606 + x607
        x609 = x41*(R_l_inv_32*x297 + 2*x483)
        x610 = x160*x542
        x611 = -x609 - x610
        x612 = x44*x562
        x613 = -4*a_3*d_2*x488 - 2*a_3*x484 + x612
        x614 = R_l_inv_32*x472 + 4*x483
        x615 = x175*x501 + x177*x4
        x616 = x609 + x610
        x617 = x599 + x600 - x601 + x602 - x603 + x604 + x605 + x606 + x607
        x618 = x552 + x553 + x554 + x555 + x556 + x557 - x558 - x559
        x619 = x550 + x618
        x620 = x310 + x322 + x324 + x326 + x328 + x485 + x487 + x489 + x490 + x491 + x492 + x495 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532
        x621 = x536 + x538 + x540
        x622 = -x563 + x565 + x567 + x569 + x570 + x571 + x574
        x623 = x581 - x583 + x584 - x586 + x587 + x588 - x589 - x590
        x624 = x545 + x546 + x547 + x592 + x593
        x625 = x310 + x322 + x324 + x326 + x328 + x487 + x490 + x491 + x492 + x495 + x499 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x518 + x519 + x521 + x523 + x524 + x525 + x527 + x529 + x530 + x531 + x532 + x595 + x596
        x626 = -2*d_2*d_4*r_23
        x627 = x64*x7
        x628 = x64*x9
        x629 = x11*x64
        x630 = a_3*x32
        x631 = x13*x32
        x632 = x58*x93
        x633 = x5*x94
        x634 = x19*x32
        x635 = x58*x96
        x636 = x5*x97
        x637 = x626 + x627 + x628 + x629 + x630 + x631 + x632 + x633 + x634 + x635 + x636
        x638 = d_4*x134
        x639 = x134*x17
        x640 = x134*x23
        x641 = x120*x48
        x642 = x120*x49
        x643 = x638 - x639 - x640 + x641 + x642
        x644 = x7*x89
        x645 = x89*x9
        x646 = x11*x89
        x647 = x7*x94
        x648 = x9*x94
        x649 = x11*x94
        x650 = x7*x97
        x651 = x9*x97
        x652 = x11*x97
        x653 = r_11*x1*x445
        x654 = x241*x243
        x655 = x449*x520
        x656 = 2*r_33*x248
        x657 = x13*x232
        x658 = x15*x163
        x659 = x17*x90
        x660 = x19*x232
        x661 = x163*x21
        x662 = x23*x90
        x663 = x23*x526
        x664 = x16*x97
        x665 = x17*x528
        x666 = x22*x94
        x667 = x18*x97
        x668 = x234*x48
        x669 = x166*x21
        x670 = x402 - x404 - x407 - x409 + x415 + x417 + x419 + x421 - x423 - x425 - x427 - x429 + x644 + x645 + x646 - x647 - x648 - x649 - x650 - x651 - x652 - x653 - x654 - x655 - x656 + x657 + x658 + x659 + x660 + x661 + x662 - x663 - x664 - x665 - x666 - x667 + x668 + x669
        x671 = x637 + x643 + x670
        x672 = x58 + x75
        x673 = -x638 + x639 + x640 - x641 - x642
        x674 = x626 + x627 + x628 + x629 - x630 + x631 + x632 + x633 + x634 + x635 + x636
        x675 = x670 + x673 + x674
        x676 = x12*x158
        x677 = x158*x7
        x678 = x158*x9
        x679 = x11*x158
        x680 = x13*x175
        x681 = a_3*x3
        x682 = x15*x681
        x683 = x158*x18
        x684 = x175*x19
        x685 = x21*x681
        x686 = x158*x24
        x687 = x566 + x568 + x574
        x688 = -x676 + x677 + x678 + x679 + x680 + x682 + x683 + x684 + x685 + x686 + x687
        x689 = 8*x12
        x690 = 8*x70
        x691 = 8*x21
        x692 = 8*x19
        x693 = 8*x15
        x694 = 8*x23
        x695 = 8*r_12
        x696 = r_11*r_21
        x697 = 8*x247
        x698 = -x100*x3 + x102*x3 - x104*x3 + x106*x3 + x108*x3 + x11*x691 + x11*x693 - x111*x3 + x113*x3 - x114*x3 + x14*x691 - x15*x689 + x16*x691 + x17*x690 + x18*x691 + x20*x693 - x21*x689 + x23*x690 + x24*x693 + x3*x84 + x3*x85 - x3*x86 + 4*x310 + x347*x695 + x350*x697 + 8*x522 + x691*x7 + x691*x9 - x692*x78 + x693*x7 + x693*x9 - x694*x73 + x695*x696*x98
        x699 = x676 - x677 - x678 - x679 - x680 - x682 - x683 - x684 - x685 - x686 + x687
        x700 = -x402 + x404 + x407 + x409 - x415 - x417 - x419 - x421 + x423 + x425 + x427 + x429 - x644 - x645 - x646 + x647 + x648 + x649 + x650 + x651 + x652 + x653 + x654 + x655 + x656 - x657 - x658 - x659 - x660 - x661 - x662 + x663 + x664 + x665 + x666 + x667 - x668 - x669
        x701 = x637 + x673 + x700
        x702 = -x58 + x75
        x703 = x643 + x674 + x700
        x704 = -x243
        x705 = r_11*x120
        x706 = d_2*x243
        x707 = 2*Px
        x708 = 2*r_12
        x709 = -d_4*x243 + r_11*x91 + x101*x707 + x103*x707 + x19*x239 + x21*x708 + x23*x243 + x243*x43 + x707*x99 + x708*x92
        x710 = -x706 + x709
        x711 = r_12*x158
        x712 = -x705
        x713 = d_2*x343
        x714 = d_2*x695
        x715 = x706 + x709
        x716 = r_12*x120
        x717 = R_l_inv_53*a_0
        x718 = x239*x717
        x719 = d_4*r_12
        x720 = r_13*x92
        x721 = r_13*x21
        x722 = r_12*x43
        x723 = r_12*x23
        x724 = x719 + x720 + x721 - x722 - x723
        x725 = x135*x724
        x726 = a_3*x243
        x727 = x53*x726
        x728 = r_11*x64
        x729 = x53*x728
        x730 = x716 + x718 + x725 + x727 + x729
        x731 = x128*x708
        x732 = d_4*r_11
        x733 = r_13*x40
        x734 = r_13*x19
        x735 = r_11*x43
        x736 = r_11*x23
        x737 = x732 + x733 + x734 - x735 - x736
        x738 = x130*x737
        x739 = r_12*x64
        x740 = x59*x739
        x741 = -x731 + x738 + x740
        x742 = r_11*x92
        x743 = 2*x742
        x744 = r_11*x21
        x745 = 2*x744
        x746 = r_12*x91
        x747 = r_12*x19
        x748 = 2*x747
        x749 = -x743 - x745 + x746 + x748
        x750 = a_3*x343
        x751 = 4*x59
        x752 = x737*x751
        x753 = x346*x717
        x754 = d_2*r_12
        x755 = 4*x53
        x756 = x754*x755
        x757 = -4*R_l_inv_56*a_0*d_2*r_11 + x128*x343 + x724*x755
        x758 = x743 + x745 - x746 - x748
        x759 = x716 + x718 + x725 - x727 + x729
        x760 = 4*x719
        x761 = Py*x447
        x762 = 4*x721
        x763 = 4*x722
        x764 = 4*x723
        x765 = x59*x711
        x766 = r_12*x40
        x767 = x742 + x744 - x747 - x766
        x768 = 4*r_13
        x769 = d_2*x768
        x770 = x53*x769 + x717*x768 - x751*x767
        x771 = 8*x735
        x772 = 8*x736
        x773 = r_11*x466
        x774 = x731 - x738 - x740
        x775 = -x298*x724
        x776 = -x239*x85
        x777 = -x239*x86
        x778 = -r_11*x306
        x779 = x239*x84
        x780 = r_11**3
        x781 = 2*x98
        x782 = x780*x781
        x783 = r_13*x158
        x784 = d_2*x783
        x785 = x216*x726
        x786 = -x216*x728
        x787 = -x107*x239
        x788 = -x108*x239
        x789 = -x113*x239
        x790 = -x114*x239
        x791 = x102*x239
        x792 = x104*x239
        x793 = x106*x239
        x794 = x111*x239
        x795 = -x388*x733
        x796 = -x388*x734
        x797 = Px*x99
        x798 = 4*x40
        x799 = x797*x798
        x800 = Px*x798
        x801 = x101*x800
        x802 = x103*x800
        x803 = x573*x797
        x804 = Px*x573
        x805 = x101*x804
        x806 = x103*x804
        x807 = Py*x343
        x808 = x12*x807
        x809 = d_4*x23
        x810 = x343*x809
        x811 = r_21*x105
        x812 = r_12*x3
        x813 = x811*x812
        x814 = r_31*x247
        x815 = x346*x814
        x816 = r_23*x811
        x817 = x768*x816
        x818 = r_31*x449
        x819 = x768*x818
        x820 = Py*x3
        x821 = -x744*x820
        x822 = x23*x43
        x823 = -x343*x822
        x824 = x19*x40
        x825 = x343*x824
        x826 = x332*x766
        x827 = x747*x820
        x828 = x437*x733
        x829 = 4*x43
        x830 = x734*x829
        x831 = x775 + x776 + x777 + x778 + x779 + x782 - x784 - x785 + x786 + x787 + x788 + x789 + x790 + x791 + x792 + x793 + x794 + x795 + x796 + x799 + x801 + x802 + x803 + x805 + x806 + x808 + x810 + x813 + x815 + x817 + x819 + x821 + x823 + x825 + x826 + x827 + x828 + x830
        x832 = x364*x737
        x833 = x275*x708
        x834 = d_4*x713
        x835 = x209*x739
        x836 = x160*x733
        x837 = x160*x734
        x838 = x5*x807
        x839 = x23*x713
        x840 = -x832 + x833 - x834 - x835 - x836 - x837 + x838 + x839
        x841 = d_4*x783
        x842 = Px*x158
        x843 = x842*x99
        x844 = x101*x842
        x845 = x103*x842
        x846 = x4*x807
        x847 = Py*x681
        x848 = r_12*x847
        x849 = x43*x783
        x850 = x19*x750
        x851 = x21*x711
        x852 = x23*x783
        x853 = -x841 + x843 + x844 + x845 + x846 + x848 + x849 + x850 + x851 + x852
        x854 = x724*(R_l_inv_27*x472 + 4*x215)
        x855 = x275*x343
        x856 = d_4*x714
        x857 = x177*x720
        x858 = x177*x721
        x859 = Py*x695
        x860 = R_l_inv_26*x472 + 4*x208
        x861 = r_12**3
        x862 = 4*x98
        x863 = 4*x754
        x864 = 8*x92
        x865 = Px*x864
        x866 = Px*x691
        x867 = r_22*x105
        x868 = 8*x867
        x869 = r_13*r_23
        x870 = r_13*r_33
        x871 = 8*x40
        x872 = x21*x695
        x873 = 8*x43
        x874 = r_11*r_31*x697 + x100*x346 + x101*x865 + x101*x866 + x103*x865 + x103*x866 + x104*x346 - x106*x346 + x107*x346 - x108*x346 - x111*x346 + x113*x346 - x114*x346 + x12*x859 - x202*x346 - x216*x863 + x346*x84 - x346*x85 - x346*x86 - x463*x720 - x463*x721 + x691*x797 + x692*x742 + x694*x720 + x695*x809 - x695*x822 - x695*x824 + x696*x868 + x697*x870 + x721*x873 + x737*x860 + x744*x871 + x797*x864 + x861*x862 + x868*x869 + x872*x92
        x875 = x841 - x843 - x844 - x845 - x846 - x848 - x849 - x850 - x851 - x852
        x876 = x775 + x776 + x777 + x778 + x779 + x782 + x784 + x785 + x786 + x787 + x788 + x789 + x790 + x791 + x792 + x793 + x794 + x795 + x796 + x799 + x801 + x802 + x803 + x805 + x806 + x808 + x810 + x813 + x815 + x817 + x819 + x821 + x823 + x825 + x826 + x827 + x828 + x830
        x877 = 8*x732
        x878 = a_3*x877
        x879 = x209*x711
        x880 = Py*r_13
        x881 = 8*x4*x880
        x882 = x466*x734
        x883 = x466*x735
        x884 = x466*x736
        x885 = r_13**3
        x886 = Px*x873
        x887 = Px*x694
        x888 = x105*x696
        x889 = 8*r_11
        x890 = r_33*x247
        x891 = x695*x92
        x892 = -8*Px*d_4*x101 - 8*Px*d_4*x103 - 8*Px*d_4*x99 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_4*r_11*r_21 - 8*Py*d_4*r_12*r_22 - 8*Py*d_4*r_13*r_23 - 8*Pz*d_4*r_11*r_31 - 8*Pz*d_4*r_12*r_32 - 8*Pz*d_4*r_13*r_33 - 4*d_2*r_13*x216 - 4*r_13*x105*x6 - 4*r_13*x105*x8 - 4*r_13*x109*x110 - 4*r_13*x109*x112 - 4*r_13*x202 + r_13*x23*x873 - 4*r_13*x84 - 4*r_13*x85 + r_23*x695*x867 + 8*r_23*x888 + x100*x768 + x101*x886 + x101*x887 + x102*x768 + x103*x886 + x103*x887 + x108*x768 + x114*x768 + x19*x771 + x23*x891 + x40*x772 + x43*x872 + x694*x797 + x695*x890 + x767*x860 + x768*x86 + x797*x873 + x818*x889 + x862*x885
        x893 = x832 - x833 + x834 + x835 + x836 + x837 - x838 - x839
        x894 = r_11*x847
        x895 = x21*x750
        x896 = Px*x820
        x897 = Px*x332
        x898 = Py*x346
        x899 = r_23*x105
        x900 = x343*x40
        x901 = Py*x21*x812 - d_4*x761 + r_11*x19*x820 + x100*x708 + x101*x896 + x101*x897 + x103*x896 + x103*x897 + x104*x708 - x106*x708 + x107*x708 - x108*x708 - x111*x708 + x113*x708 - x114*x708 + x12*x898 + x21*x900 + x23*x760 + x23*x761 - x23*x763 + x3*x888 + x332*x797 + x343*x814 - x388*x721 + x43*x762 + x447*x899 + x486*x708 - x501*x739 - x564*x737 - x573*x766 - x708*x84 - x708*x85 - x708*x86 + x768*x890 + x781*x861 + x797*x820
        x902 = -4*Py*a_3*r_12*r_21 - 4*Pz*a_3*r_12*r_31 - 2*a_3*r_13*x498 + x894 + x895 + x901
        x903 = x580*x724
        x904 = r_11*x582
        x905 = d_2*x760
        x906 = x498*x728
        x907 = x34*x880
        x908 = x160*x721
        x909 = x5*x898
        x910 = x23*x863
        x911 = -x903 - x904 - x905 - x906 - x907 - x908 + x909 + x910
        x912 = R_l_inv_36*x472 + 4*x500
        x913 = x737*x912
        x914 = x346*x551
        x915 = d_2*x877
        x916 = x498*x863
        x917 = x177*x733
        x918 = x177*x734
        x919 = Py*x5
        x920 = x889*x919
        x921 = x177*x736
        x922 = x343*x84
        x923 = x343*x85
        x924 = Px*x871
        x925 = Px*x692
        x926 = 8*r_13
        x927 = Py*r_11*x689 + r_22*x695*x811 + x101*x924 + x101*x925 + x102*x343 + x103*x924 + x103*x925 + x104*x343 + x106*x343 - x107*x343 - x108*x343 + x111*x343 - x113*x343 - x114*x343 + x19*x891 + 8*x23*x733 - x23*x771 + x23*x877 - x343*x86 + x40*x872 + 8*x43*x734 - x463*x733 - x463*x734 - x691*x742 + x692*x797 + x695*x814 + x780*x862 + x797*x871 + x816*x926 + x818*x926 + x824*x889
        x928 = x343*x486 - x501*x713 + x724*(R_l_inv_37*x472 + 4*x497) - x922 - x923 + x927
        x929 = x19*x711 + x4*x898 + x498*x726 - x894 - x895 + x901
        x930 = x551*x768
        x931 = x498*x769
        x932 = x40*x714
        x933 = x19*x714
        x934 = a_3*x714 + x501*x711
        x935 = x903 + x904 + x905 + x906 + x907 + x908 - x909 - x910
        x936 = x243*x85
        x937 = x243*x86
        x938 = x781*x885
        x939 = x243*x84
        x940 = x100*x243
        x941 = x102*x243
        x942 = x108*x243
        x943 = x114*x243
        x944 = x106*x243
        x945 = x107*x243
        x946 = x111*x243
        x947 = x113*x243
        x948 = Px*x388
        x949 = x948*x99
        x950 = x101*x948
        x951 = x103*x948
        x952 = x797*x829
        x953 = Px*x829
        x954 = x101*x953
        x955 = x103*x953
        x956 = x437*x797
        x957 = Px*x437
        x958 = x101*x957
        x959 = x103*x957
        x960 = x344*x899
        x961 = x343*x818
        x962 = x812*x899
        x963 = x346*x890
        x964 = d_4*x343
        x965 = x40*x964
        x966 = x719*x820
        x967 = Py*x768
        x968 = x12*x967
        x969 = x19*x964
        x970 = x21*x760
        x971 = x768*x809
        x972 = x23*x900
        x973 = x19*x343*x43
        x974 = x723*x820
        x975 = x21*x763
        x976 = x768*x822
        x977 = x573*x733
        x978 = x721*x820
        x979 = a_3*x713
        x980 = d_4*x750
        x981 = x43*x750
        x982 = -x981
        x983 = x23*x750
        x984 = -x983
        x985 = x4*x967
        x986 = x158*x734
        x987 = x979 + x980 + x982 + x984 + x985 + x986
        x988 = Px*x160
        x989 = x988*x99
        x990 = x101*x988
        x991 = x103*x988
        x992 = x40*x713
        x993 = x754*x820
        x994 = x768*x919
        x995 = x19*x713
        x996 = x21*x863
        x997 = x23*x769
        x998 = -4*d_2*d_4*r_13 + x989 + x990 + x991 + x992 + x993 + x994 + x995 + x996 + x997
        x999 = x936 + x937 + x938 - x939 + x940 + x941 + x942 + x943 - x944 - x945 - x946 - x947 - x949 - x950 - x951 + x952 + x954 + x955 + x956 + x958 + x959 + x960 + x961 + x962 + x963 - x965 - x966 - x968 - x969 - x970 - x971 + x972 + x973 + x974 + x975 + x976 - x977 - x978
        x1000 = r_13*x466
        x1001 = d_4*x1000
        x1002 = Px*x466
        x1003 = x922 + x923 + x927
        x1004 = 16*d_4
        x1005 = 16*x23
        x1006 = 16*x19
        x1007 = 16*x92
        x1008 = Px*x1007
        x1009 = 16*x21
        x1010 = Px*x1009
        x1011 = 16*x867
        x1012 = x998 + x999
        x1013 = -x29
        x1014 = -x42
        x1015 = a_1 + x141 + x142
        x1016 = x1014 + x1015 + x62
        x1017 = x134 + x136 - x138 - x139
        x1018 = x196 + x197 + x200 + x201 + x203 + x206 + x210 + x211 + x212 + x213 + x214 + x220 + x221 + x222 + x223 + x224 + x227 + x228 + x229 + x230 + x231 + x233 + x235 + x236 + x237 + x238 + x242 + x245 + x249 + x252 + x254 + x256 + x258 + x259 + x260 + x261 + x262
        x1019 = x1018 + x266 + x268 + x270
        x1020 = x218 + x226 + x384 + x385
        x1021 = x299 + x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x325 + x327 + x329 + x331 + x333 + x334 + x335 + x336 + x338 + x339 + x340 + x341 + x342 + x345 + x348 + x351 + x352 + x353 + x354 + x355 + x356 + x357 + x358 + x359
        x1022 = x195 + x199 + x217 + x225
        x1023 = x1018 + x285 + x374 + x375 + x376 + x377 + x378 + x379 + x380 + x381 + x382
        x1024 = -x535 - x543
        x1025 = -x41*x576 + x578
        x1026 = x1024 + x537 + x539 + x541
        x1027 = d_2*x577 + x120*x484 - x612
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x0
        A[0, 2] = x0
        A[0, 3] = x2
        A[0, 4] = -x3
        A[0, 5] = x1
        A[0, 6] = r_23
        A[0, 8] = r_23
        A[1, 0] = x27
        A[1, 1] = x29
        A[1, 2] = x31
        A[1, 3] = x33
        A[1, 4] = -x34
        A[1, 5] = x32
        A[1, 6] = x36
        A[1, 7] = x29
        A[1, 8] = x37
        A[2, 0] = -x116 - x123 - x67 - x83
        A[2, 1] = -x127 - x133 - x140
        A[2, 2] = x144 + x148 + x151
        A[2, 3] = x152 + x153 + x154 - x162 - x169
        A[2, 4] = x170 + x171 + x172 - x173 - x174 - x176 - x179
        A[2, 5] = -x156 + x157 - x159 + x161 + x181 + 2*x38*x44
        A[2, 6] = x144 + x183 + x186
        A[2, 7] = x140 + x187 + x188
        A[2, 8] = -x145 - x146 - x147 - x184 - x189 - x42 - x62
        A[3, 0] = x263 + x274 + x296
        A[3, 1] = x360 + x363 + x373
        A[3, 2] = -x284 - x383 - x386 - x387
        A[3, 3] = x389 + x390 + x391 + x392 - x393 - x394 - x400 - x462
        A[3, 4] = x464 - x465 + x467 + x468 - x470 - x471 - x474
        A[3, 5] = x395 - x396 - x397 - x398 + x399 - x476
        A[3, 6] = -x386 - x478 - x480
        A[3, 7] = -x360 - x481 - x482
        A[3, 8] = x263 + x383 + x477 + x479
        A[4, 0] = -x533 - x544 - x561
        A[4, 1] = x575 + x579 + x591
        A[4, 2] = x560 + x594 + x597 + x598
        A[4, 3] = -x608 - x611 - x613
        A[4, 4] = x117*x614 + x473*x488 + x615
        A[4, 5] = -x613 - x616 - x617
        A[4, 6] = -x598 - x619 - x620 - x621
        A[4, 7] = x579 + x622 + x623
        A[4, 8] = x544 + x618 + x624 + x625
        A[5, 0] = -x671
        A[5, 1] = -x158*x672
        A[5, 2] = -x675
        A[5, 3] = x688
        A[5, 4] = x698
        A[5, 5] = -x699
        A[5, 6] = -x701
        A[5, 7] = x158*x702
        A[5, 8] = -x703
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x704
        B[0, 2] = x704
        B[0, 3] = -x343
        B[0, 4] = -x695
        B[0, 5] = x343
        B[0, 6] = x243
        B[0, 8] = x243
        B[1, 0] = x705 + x710
        B[1, 1] = x711
        B[1, 2] = x710 + x712
        B[1, 3] = -x713
        B[1, 4] = -x714
        B[1, 5] = x713
        B[1, 6] = x705 + x715
        B[1, 7] = x711
        B[1, 8] = x712 + x715
        B[2, 0] = -x730 - x741 - x749
        B[2, 1] = x750 + x752 - x753 - x756 - x757
        B[2, 2] = x741 + x758 + x759
        B[2, 3] = -x760 - x761 - x762 + x763 + x764 + x765 + x770
        B[2, 4] = 8*Py*r_13*r_21 + 8*Pz*r_13*r_31 + 8*d_4*r_11 - x59*x773 - x771 - x772
        B[2, 5] = x760 + x761 + x762 - x763 - x764 - x765 + x770
        B[2, 6] = x749 + x759 + x774
        B[2, 7] = -x750 - x752 + x753 + x756 - x757
        B[2, 8] = -x730 - x758 - x774
        B[3, 0] = x831 + x840 + x853
        B[3, 1] = x209*x713 + x23*x714 + x5*x859 - x854 - x855 - x856 - x857 - x858 + x874
        B[3, 2] = -x840 - x875 - x876
        B[3, 3] = x878 + x879 + x881 + x882 - x883 - x884 - x892
        B[3, 4] = x466*(2*Py*r_13*r_22 + 2*Pz*r_13*r_32 + 2*d_4*r_12 - r_11*x209 - 2*x722 - 2*x723)
        B[3, 5] = -x878 - x879 - x881 - x882 + x883 + x884 - x892
        B[3, 6] = -x853 - x876 - x893
        B[3, 7] = 8*Py*d_2*r_12*r_23 + 8*Pz*d_2*r_12*r_33 + 4*d_2*r_11*x209 - x854 - x855 - x856 - x857 - x858 - x874
        B[3, 8] = x831 + x875 + x893
        B[4, 0] = -x902 - x911
        B[4, 1] = -x913 + x914 - x915 + x916 - x917 - x918 + x920 + x921 + x928
        B[4, 2] = x911 + x929
        B[4, 3] = 8*Py*d_2*r_11*r_22 + 8*Pz*d_2*r_11*r_32 + x767*x912 - x930 - x931 - x932 - x933 - x934
        B[4, 4] = x773*(x501 + x64)
        B[4, 5] = x177*x742 + x177*x744 + x767*x912 - x930 - x931 - x932 - x933 + x934
        B[4, 6] = -x902 - x935
        B[4, 7] = x913 - x914 + x915 - x916 + x917 + x918 - x920 - x921 + x928
        B[4, 8] = x929 + x935
        B[5, 0] = x936 + x937 + x938 - x939 + x940 + x941 + x942 + x943 - x944 - x945 - x946 - x947 - x949 - x950 - x951 + x952 + x954 + x955 + x956 + x958 + x959 + x960 + x961 + x962 + x963 - x965 - x966 - x968 - x969 - x970 - x971 + x972 + x973 + x974 + x975 + x976 - x977 - x978 - x987 - x998
        B[5, 1] = x466*(-x724 - x754)
        B[5, 2] = d_4*x769 + x987 - x989 - x990 - x991 - x992 - x993 - x994 - x995 - x996 - x997 + x999
        B[5, 3] = Py*x4*x889 + a_3*x872 + x1000*x23 + x1000*x43 - x1001 + x1002*x101 + x1002*x103 + x1002*x99 + x1003 + x19*x773 + x28*x859
        B[5, 4] = 16*Py*r_12*x12 + 16*r_11*x814 + r_12*x1007*x21 + x100*x695 - x1004*x720 - x1004*x721 + x1005*x719 + x1005*x720 - x1005*x722 + x1006*x742 - x1006*x766 + x1007*x797 + x1008*x101 + x1008*x103 + x1009*x797 + x101*x1010 + x1010*x103 + x1011*x696 + x1011*x869 + x104*x695 - x106*x695 + x107*x695 - x108*x695 - x111*x695 + x113*x695 - x114*x695 + 16*x247*x870 + 16*x40*x744 + 16*x43*x721 + x695*x84 + x695*x85 - x695*x86 + 8*x861*x98
        B[5, 5] = 8*Px*a_3*x101 + 8*Px*a_3*x103 + 8*Px*a_3*x99 + 8*Py*a_3*r_11*r_21 + 8*Py*a_3*r_12*r_22 + 8*Py*a_3*r_13*r_23 + 8*Pz*a_3*r_11*r_31 + 8*Pz*a_3*r_12*r_32 + 8*Pz*a_3*r_13*r_33 - x1001 - x1003
        B[5, 6] = -x1012 - x979 + x980 - x981 - x983 + x985 + x986
        B[5, 7] = x466*(x719 + x720 + x721 - x722 - x723 - x754)
        B[5, 8] = -x1012 + x979 - x980 - x982 - x984 - x985 - x986
        C = np.zeros(shape=(6, 9))
        C[0, 0] = r_23
        C[0, 2] = r_23
        C[0, 3] = x1
        C[0, 4] = x3
        C[0, 5] = x2
        C[0, 6] = x0
        C[0, 8] = x0
        C[1, 0] = -x27
        C[1, 1] = x1013
        C[1, 2] = -x31
        C[1, 3] = x32
        C[1, 4] = x34
        C[1, 5] = x33
        C[1, 6] = -x36
        C[1, 7] = x1013
        C[1, 8] = -x37
        C[2, 0] = x1016 + x148 + x183
        C[2, 1] = -x1017 - x127 - x187
        C[2, 2] = -x1014 - x143 - x185 - x189 - x54 - x57 - x60
        C[2, 3] = -x162 + x163 - x164 - x165 + x166 + x167 - x168 - x180
        C[2, 4] = -x170 - x171 - x172 + x173 + x174 + x176 - x179
        C[2, 5] = -x155 - x156 + x157 - x159 + x161 - x181
        C[2, 6] = x1015 + x149 + x182 - x28 + x67 - x69 - x76 - x77
        C[2, 7] = x1017 + x133 + x188
        C[2, 8] = x1016 + x151 + x186
        C[3, 0] = -x1019 - x1020 - x296 - x477
        C[3, 1] = -x1021 - x373 - x481
        C[3, 2] = x1022 + x1023 + x284 + x478
        C[3, 3] = -x400 - x401 - x403 - x405 - x406 + x408 + x410 - x411 - x412 - x413 - x414 - x416 - x418 - x420 - x422 + x424 + x426 + x428 + x430 - x431 - x432 - x434 - x435 - x436 - x438 + x439 + x440 + x441 + x442 + x443 + x444 + x446 + x448 + x450 + x452 - x453 - x454 + x455 + x457 + x458 + x460 + x461 - x475
        C[3, 4] = -x464 + x465 - x467 - x468 + x470 + x471 - x474
        C[3, 5] = x300*x44 + x395 - x397 - x398 + x399 + x476
        C[3, 6] = x1019 + x1022 + x387 + x480
        C[3, 7] = x1021 + x363 + x482
        C[3, 8] = -x1020 - x1023 - x274 - x479
        C[4, 0] = x1024 + x561 + x621 + x625
        C[4, 1] = -x1025 - x591 - x622
        C[4, 2] = -x1026 - x560 - x620 - x624
        C[4, 3] = x1027 + x608 + x616
        C[4, 4] = 8*d_2*x117*x488 + x117*x614 - x615
        C[4, 5] = x1027 + x611 + x617
        C[4, 6] = x1026 + x597 + x619
        C[4, 7] = -x1025 - x575 - x623
        C[4, 8] = -x1024 - x533 - x594 - x618
        C[5, 0] = x671
        C[5, 1] = x158*x672
        C[5, 2] = x675
        C[5, 3] = -x688
        C[5, 4] = -x698
        C[5, 5] = x699
        C[5, 6] = x701
        C[5, 7] = -x158*x702
        C[5, 8] = x703
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[0] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor()
    # Finish code for explicit solution node 0

    # Code for equation all-zero dispatcher node 2
    def EquationAllZeroDispatcherNode_node_2_processor():
        this_node_input_index: List[int] = node_input_index[2]
        this_input_valid: bool = node_input_validity[2]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            checked_result: bool = (((abs(a_3*r_11*math.sin(th_0) - a_3*r_21*math.cos(th_0)) <= 1.0e-6)) and ((abs(a_3*r_12*math.sin(th_0) - a_3*r_22*math.cos(th_0)) <= 1.0e-6)) and ((abs(Px*math.sin(th_0) - Py*math.cos(th_0) - d_4*r_13*math.sin(th_0) + d_4*r_23*math.cos(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2

    # Code for explicit solution node 3, solved variable is th_5
    def ExplicitSolutionNode_node_3_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[3]
        this_input_valid: bool = node_input_validity[3]
        if not this_input_valid:
            return

        # The solution of non-root node 3
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (((abs(a_3*r_11*math.sin(th_0) - a_3*r_21*math.cos(th_0)) >= zero_tolerance)) or ((abs(a_3*r_12*math.sin(th_0) - a_3*r_22*math.cos(th_0)) >= zero_tolerance)) or ((abs(Px*math.sin(th_0) - Py*math.cos(th_0) - d_4*r_13*math.sin(th_0) + d_4*r_23*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_0)
                x1 = a_3*x0
                x2 = math.cos(th_0)
                x3 = -a_3*r_22*x2 + r_12*x1
                x4 = a_3*r_21*x2 - r_11*x1
                x5 = math.atan2(x3, x4)
                x6 = -Px*x0 + Py*x2 + d_4*r_13*x0 - d_4*r_23*x2
                x7 = safe_sqrt(x3**2 + x4**2 - x6**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x5 + math.atan2(x7, x6)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[7] = x5 + math.atan2(-x7, x6)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_5_processor()
    # Finish code for explicit solution node 3

    # Code for non-branch dispatcher node 4
    # Actually, there is no code

    # Code for explicit solution node 5, solved variable is th_3
    def ExplicitSolutionNode_node_5_solve_th_3_processor():
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs((r_11*math.sin(th_0) - r_21*math.cos(th_0))*math.sin(th_5) + (r_12*math.sin(th_0) - r_22*math.cos(th_0))*math.cos(th_5)) <= 70368744177665/70368744177664)
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = safe_acos((-r_11*x0 + r_21*x1)*math.sin(th_5) + (-r_12*x0 + r_22*x1)*math.cos(th_5))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                this_solution[5] = -x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_3_processor()
    # Finish code for explicit solution node 4

    # Code for non-branch dispatcher node 6
    # Actually, there is no code

    # Code for explicit solution node 7, solved variable is th_2
    def ExplicitSolutionNode_node_7_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return

        # The solution of non-root node 7
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (((2*abs(a_0*a_1) >= zero_tolerance)) or ((2*abs(a_0*d_2) >= zero_tolerance)) or ((abs(Px**2 - 2*Px*a_3*r_11*math.cos(th_5) + 2*Px*a_3*r_12*math.sin(th_5) - 2*Px*d_4*r_13 + Py**2 - 2*Py*a_3*r_21*math.cos(th_5) + 2*Py*a_3*r_22*math.sin(th_5) - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*a_3*r_31*math.cos(th_5) + 2*Pz*a_3*r_32*math.sin(th_5) - 2*Pz*d_4*r_33 - a_0**2 - a_1**2 + a_3**2*r_11**2*math.cos(th_5)**2 - a_3**2*r_11*r_12*math.sin(2*th_5) - a_3**2*r_12**2*math.cos(th_5)**2 + a_3**2*r_12**2 + a_3**2*r_21**2*math.cos(th_5)**2 - a_3**2*r_21*r_22*math.sin(2*th_5) - a_3**2*r_22**2*math.cos(th_5)**2 + a_3**2*r_22**2 + a_3**2*r_31**2*math.cos(th_5)**2 - a_3**2*r_31*r_32*math.sin(2*th_5) - a_3**2*r_32**2*math.cos(th_5)**2 + a_3**2*r_32**2 + 2*a_3*d_4*r_11*r_13*math.cos(th_5) - 2*a_3*d_4*r_12*r_13*math.sin(th_5) + 2*a_3*d_4*r_21*r_23*math.cos(th_5) - 2*a_3*d_4*r_22*r_23*math.sin(th_5) + 2*a_3*d_4*r_31*r_33*math.cos(th_5) - 2*a_3*d_4*r_32*r_33*math.sin(th_5) - d_2**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)))
            if condition_0:
                x0 = 2*a_0
                x1 = math.atan2(-d_2*x0, a_1*x0)
                x2 = a_1**2
                x3 = a_0**2
                x4 = 4*x3
                x5 = d_2**2
                x6 = 2*d_4
                x7 = r_13*x6
                x8 = r_23*x6
                x9 = r_33*x6
                x10 = a_3**2
                x11 = r_12**2*x10
                x12 = r_22**2*x10
                x13 = r_32**2*x10
                x14 = d_4**2
                x15 = math.cos(th_5)
                x16 = 2*a_3
                x17 = x15*x16
                x18 = math.sin(th_5)
                x19 = x16*x18
                x20 = a_3*x15
                x21 = a_3*x18
                x22 = x15**2
                x23 = x10*x22
                x24 = x10*math.sin(2*th_5)
                x25 = Px**2 - Px*r_11*x17 + Px*r_12*x19 - Px*x7 + Py**2 - Py*r_21*x17 + Py*r_22*x19 - Py*x8 + Pz**2 - Pz*r_31*x17 + Pz*r_32*x19 - Pz*x9 + r_11**2*x23 - r_11*r_12*x24 + r_11*x20*x7 - r_12*x21*x7 + r_13**2*x14 + r_21**2*x23 - r_21*r_22*x24 + r_21*x20*x8 - r_22*x21*x8 + r_23**2*x14 + r_31**2*x23 - r_31*r_32*x24 + r_31*x20*x9 - r_32*x21*x9 + r_33**2*x14 - x11*x22 + x11 - x12*x22 + x12 - x13*x22 + x13 - x2 - x3 - x5
                x26 = safe_sqrt(x2*x4 - x25**2 + x4*x5)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x1 + math.atan2(x26, x25)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                this_solution[4] = x1 + math.atan2(-x26, x25)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_2_processor()
    # Finish code for explicit solution node 6

    # Code for solved_variable dispatcher node 8
    def SolvedVariableDispatcherNode_node_8_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            forwarded_to_child = False
            th_3 = this_solution[5]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(14, branch_input_index)

            th_3 = this_solution[5]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(17, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8

    # Code for explicit solution node 17, solved variable is th_1th_2th_4_soa
    def ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor():
        this_node_input_index: List[int] = node_input_index[17]
        this_input_valid: bool = node_input_validity[17]
        if not this_input_valid:
            return

        # The solution of non-root node 17
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_33) >= zero_tolerance)) or ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= zero_tolerance)))
            if condition_0:
                this_solution[3] = math.atan2(r_13*math.cos(th_0) + r_23*math.sin(th_0), r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 17

    # Code for solved_variable dispatcher node 18
    def SolvedVariableDispatcherNode_node_18_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            forwarded_to_child = False
            th_2 = this_solution[4]
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.34089189866299) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(22, branch_input_index)

            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.34089189866299) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(25, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(19, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_18_processor()
    # Finish code for solved_variable dispatcher node 18

    # Code for explicit solution node 25, solved variable is th_1
    def ExplicitSolutionNode_node_25_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[25]
        this_input_valid: bool = node_input_validity[25]
        if not this_input_valid:
            return

        # The solution of non-root node 25
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(0.973688178796424*a_1 - 0.227884467377887*d_2) >= 1.0e-6)) or ((abs(a_0 + 0.227884467377887*a_1 + 0.973688178796424*d_2) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 + 0.227884467377887*a_1 + 0.973688178796424*d_2
                x1 = a_3*math.sin(th_5)
                x2 = a_3*math.cos(th_5)
                x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1
                x4 = 0.973688178796424*a_1 - 0.227884467377887*d_2
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6
                this_solution[1] = math.atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(26, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_1_processor()
    # Finish code for explicit solution node 25

    # Code for non-branch dispatcher node 26
    # Actually, there is no code

    # Code for explicit solution node 27, solved variable is th_4
    def ExplicitSolutionNode_node_27_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[26]
        this_input_valid: bool = node_input_validity[26]
        if not this_input_valid:
            return

        # The solution of non-root node 27
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(-r_13*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.cos(th_0) - r_23*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.sin(th_0) + r_33*(0.973688178796424*math.sin(th_1) + 0.227884467377887*math.cos(th_1))) >= zero_tolerance)) or ((abs(r_13*((0.973688178796424*math.sin(th_1) + 0.227884467377886*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) + r_23*((0.973688178796424*math.sin(th_1) + 0.227884467377886*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.227884467377886*math.sin(th_1) - 0.973688178796424*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = -0.973688178796424*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.973688178796424*x1
                x8 = x0*(0.227884467377886*x2 + x7)
                x9 = -0.227884467377887*x1 - x3
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(-0.227884467377886*x1 - x3), -r_13*x6*x9 - r_23*x4*x9 + r_33*(0.227884467377887*x2 + x7))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_4_processor()
    # Finish code for explicit solution node 26

    # Code for explicit solution node 22, solved variable is th_1
    def ExplicitSolutionNode_node_22_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return

        # The solution of non-root node 22
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(0.973688178796424*a_1 - 0.227884467377887*d_2) >= 1.0e-6)) or ((abs(-a_0 + 0.227884467377887*a_1 + 0.973688178796424*d_2) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 - 0.227884467377887*a_1 - 0.973688178796424*d_2
                x1 = a_3*math.sin(th_5)
                x2 = a_3*math.cos(th_5)
                x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1
                x4 = -0.973688178796424*a_1 + 0.227884467377887*d_2
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6
                this_solution[1] = math.atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(23, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_1_processor()
    # Finish code for explicit solution node 22

    # Code for non-branch dispatcher node 23
    # Actually, there is no code

    # Code for explicit solution node 24, solved variable is th_4
    def ExplicitSolutionNode_node_24_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[23]
        this_input_valid: bool = node_input_validity[23]
        if not this_input_valid:
            return

        # The solution of non-root node 24
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(-r_13*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.cos(th_0) - r_23*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.sin(th_0) + r_33*(0.973688178796424*math.sin(th_1) + 0.227884467377887*math.cos(th_1))) >= zero_tolerance)) or ((abs(-r_13*((-0.973688178796424*math.sin(th_1) - 0.227884467377886*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.973688178796424*math.sin(th_1) - 0.227884467377886*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.227884467377886*math.sin(th_1) - 0.973688178796424*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = -0.973688178796424*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.973688178796424*x1
                x8 = x0*(-0.227884467377886*x2 - x7)
                x9 = -0.227884467377887*x1 - x3
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(0.227884467377886*x1 + x3), r_13*x6*x9 + r_23*x4*x9 - r_33*(0.227884467377887*x2 + x7))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_4_processor()
    # Finish code for explicit solution node 23

    # Code for explicit solution node 19, solved variable is th_1
    def ExplicitSolutionNode_node_19_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_1th_2th_4_soa = this_solution[3]
            th_2 = this_solution[4]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(a_1*math.sin(th_2) + d_2*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 + a_1*math.cos(th_2) - d_2*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = math.sin(th_2)
                x1 = math.cos(th_2)
                x2 = -a_1*x0 - d_2*x1
                x3 = Px*math.cos(th_0) + Py*math.sin(th_0) + a_3*math.cos(th_1th_2th_4_soa) - d_4*math.sin(th_1th_2th_4_soa)
                x4 = a_0 + a_1*x1 - d_2*x0
                x5 = Pz - a_3*r_31*math.cos(th_5) + a_3*r_32*math.sin(th_5) - d_4*r_33
                this_solution[1] = math.atan2(x2*x3 - x4*x5, x2*x5 + x3*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1_processor()
    # Finish code for explicit solution node 19

    # Code for non-branch dispatcher node 20
    # Actually, there is no code

    # Code for explicit solution node 21, solved variable is th_4
    def ExplicitSolutionNode_node_21_solve_th_4_processor():
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
            th_1 = this_solution[1]
            th_1th_2th_4_soa = this_solution[3]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[6] = -th_1 + th_1th_2th_4_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_4_processor()
    # Finish code for explicit solution node 20

    # Code for explicit solution node 14, solved variable is th_1
    def ExplicitSolutionNode_node_14_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[14]
        this_input_valid: bool = node_input_validity[14]
        if not this_input_valid:
            return

        # The solution of non-root node 14
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_2 = this_solution[4]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(a_1*math.sin(th_2) + d_2*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 + a_1*math.cos(th_2) - d_2*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = math.cos(th_2)
                x1 = math.sin(th_2)
                x2 = a_0 + a_1*x0 - d_2*x1
                x3 = a_3*math.sin(th_5)
                x4 = a_3*math.cos(th_5)
                x5 = Pz - d_4*r_33 - r_31*x4 + r_32*x3
                x6 = -a_1*x1 - d_2*x0
                x7 = math.cos(th_0)
                x8 = math.sin(th_0)
                x9 = Px*x7 + Py*x8 - d_4*r_13*x7 - d_4*r_23*x8 - r_11*x4*x7 + r_12*x3*x7 - r_21*x4*x8 + r_22*x3*x8
                this_solution[1] = math.atan2(-x2*x5 + x6*x9, x2*x9 + x5*x6)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_1_processor()
    # Finish code for explicit solution node 14

    # Code for non-branch dispatcher node 15
    # Actually, there is no code

    # Code for explicit solution node 16, solved variable is th_4
    def ExplicitSolutionNode_node_16_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[15]
        this_input_valid: bool = node_input_validity[15]
        if not this_input_valid:
            return

        # The solution of non-root node 16
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_2 = this_solution[4]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance)) or ((abs(-r_13*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - r_23*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + r_33*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = x0*x1 + x2*x3
                x5 = -x0*x2 + x1*x3
                x6 = r_13*math.cos(th_0)
                x7 = r_23*math.sin(th_0)
                this_solution[6] = math.atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor()
    # Finish code for explicit solution node 15

    # Code for explicit solution node 9, solved variable is th_4
    def ExplicitSolutionNode_node_9_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[9]
        this_input_valid: bool = node_input_validity[9]
        if not this_input_valid:
            return

        # The solution of non-root node 9
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[5]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(math.sin(th_3)) >= zero_tolerance)) or ((abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance)) or ((abs((r_11*math.sin(th_0) - r_21*math.cos(th_0))*math.cos(th_5) + (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*math.sin(th_5)) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_3)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                this_solution[6] = math.atan2(x0*(-r_13*x1 + r_23*x2), x0*(-(-r_11*x1 + r_21*x2)*math.cos(th_5) + (-r_12*x1 + r_22*x2)*math.sin(th_5)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 9

    # Code for non-branch dispatcher node 10
    # Actually, there is no code

    # Code for explicit solution node 11, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor():
        this_node_input_index: List[int] = node_input_index[10]
        this_input_valid: bool = node_input_validity[10]
        if not this_input_valid:
            return

        # The solution of non-root node 11
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[5]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(math.sin(th_3)) >= zero_tolerance)) or ((abs(r_31*math.sin(th_5) + r_32*math.cos(th_5)) >= zero_tolerance)) or ((abs((r_11*math.cos(th_0) + r_21*math.sin(th_0))*math.sin(th_5) + (r_12*math.cos(th_0) + r_22*math.sin(th_0))*math.cos(th_5)) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_3)
                x1 = math.sin(th_5)
                x2 = math.cos(th_5)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                this_solution[2] = math.atan2(x0*(-r_31*x1 - r_32*x2), x0*(-x1*(-r_11*x3 - r_21*x4) + x2*(r_12*x3 + r_22*x4)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 10

    # Code for non-branch dispatcher node 12
    # Actually, there is no code

    # Code for explicit solution node 13, solved variable is th_1
    def ExplicitSolutionNode_node_13_solve_th_1_processor():
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
            th_1th_2_soa = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[1] = th_1th_2_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_1_processor()
    # Finish code for explicit solution node 12

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
        value_at_2 = ik_out_i[4]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[5]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[6]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[7]  # th_5
        new_ik_i[5] = value_at_5
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def abb_crb_15000_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = abb_crb_15000_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    strict_raw_witness = False
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw
        ee_pose_i = abb_crb_15000_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                strict_raw_witness = True
                _yaik_seed_candidates.clear()
            elif not strict_raw_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -1.5707963267948966, -3.141592653589793, -6.283185307179586, -6.283185307179586, -3.141592653589793], [3.141592653589793, 4.71238898038469, 3.141592653589793, 0.0, 0.0, 3.141592653589793], [True, True, True, True, True, True], [False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = abb_crb_15000_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
    return ik_output


def abb_crb_15000_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = []
    for transform in (abb_crb_15000_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in abb_crb_15000_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: abb_crb_15000_fk(q)
    if refinement_seeds and not any(np.max(np.abs(refinement_fk(q)-T_ee)) < 1e-7 for q in result):
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                break
    return result


def test_ik_solve_abb_crb_15000():
    theta_in = np.random.random(size=(6, ))
    ee_pose = abb_crb_15000_fk(theta_in)
    ik_output = abb_crb_15000_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = abb_crb_15000_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_abb_crb_15000()
