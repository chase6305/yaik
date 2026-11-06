import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 35
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.165
a_2: float = 0.012
d_1: float = 0.02
d_3: float = 0.1775
d_4: float = -0.0445
d_5: float = 0.045
pre_transform_special_symbol_23: float = 0.18

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
def dense_cvr_038_ik_target_original_to_raw(T_ee: np.ndarray):
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


def dense_cvr_038_ik_target_raw_to_original(T_ee: np.ndarray):
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


def dense_cvr_038_fk(theta_input: np.ndarray):
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
    x22 = 1.0*d_1
    x23 = 1.0*a_0
    x24 = x23*x6
    x25 = 1.0*a_2
    x26 = 1.0*d_3
    x27 = 1.0*d_5
    x28 = -x11*x4 - x11*x7
    x29 = x0*x28
    x30 = -x11*x14 + x11*x3*x6
    x31 = -x1*x12 + x13*x30
    x32 = x10*x31
    x33 = -1.0*x1*x13 - 1.0*x12*x30
    x34 = 1.0*x0*x31 - 1.0*x10*x28
    x35 = x14 - x3*x6
    x36 = x0*x35
    x37 = -x4 - x7
    x38 = x10*x13*x37
    x39 = 1.0*x12*x37
    x40 = 1.0*x0*x13*x37 - 1.0*x10*x35
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x17 - 1.0*x9
    ee_pose[0, 1] = x18*x19 + x20*x21
    ee_pose[0, 2] = x18*x21 - x19*x20
    ee_pose[0, 3] = d_4*x19 + x1*x24 - x11*x22 + x15*x25 + x26*x8 + x27*(-x17 - x9)
    ee_pose[1, 0] = -1.0*x29 - 1.0*x32
    ee_pose[1, 1] = x18*x33 + x20*x34
    ee_pose[1, 2] = x18*x34 - x20*x33
    ee_pose[1, 3] = d_4*x33 + x1*x22 + x11*x24 + x25*x30 + x26*x28 + x27*(-x29 - x32)
    ee_pose[2, 0] = -1.0*x36 - 1.0*x38
    ee_pose[2, 1] = -x18*x39 + x20*x40
    ee_pose[2, 2] = x18*x40 + x20*x39
    ee_pose[2, 3] = -d_4*x39 + 1.0*pre_transform_special_symbol_23 - x2*x23 + x25*x37 + x26*x35 + x27*(-x36 - x38)
    return ee_pose


def dense_cvr_038_twist_jacobian(theta_input: np.ndarray):
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
    x18 = -x11*x16 - x17*(x1*x13 + x12*x14)
    x19 = x1*x4
    x20 = x1*x9
    x21 = -x19*x3 - x20*x8
    x22 = 1.0*x0*x3*x9 - x19*x8
    x23 = -x12*x6 - x13*x22
    x24 = -x16*x21 - x17*(x12*x22 - x13*x6)
    x25 = 1.0*x8
    x26 = 1.0*x3
    x27 = x25*x4 - x26*x9
    x28 = -x25*x9 - x26*x4
    x29 = x13*x28
    x30 = -x12*x17*x28 - x16*x27
    x31 = 1.0*a_0*x4
    x32 = pre_transform_special_symbol_23 - x31
    x33 = a_2*x28 + d_3*x27 + pre_transform_special_symbol_23 - x31
    x34 = a_0*x20 + d_1*x6
    x35 = a_2*x22 + d_3*x21 + x34
    x36 = -d_4*x29 + x33
    x37 = d_4*x23 + x35
    x38 = d_5*x30 + x36
    x39 = d_5*x24 + x37
    x40 = a_0*x10 - d_1*x1
    x41 = a_2*x14 + d_3*x11 + x40
    x42 = d_4*x15 + x41
    x43 = d_5*x18 + x42
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x11
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x18
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x6
    jacobian_output[1, 3] = x21
    jacobian_output[1, 4] = x23
    jacobian_output[1, 5] = x24
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x27
    jacobian_output[2, 4] = -x29
    jacobian_output[2, 5] = x30
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x6
    jacobian_output[3, 2] = -x32*x6
    jacobian_output[3, 3] = -x21*x33 + x27*x35
    jacobian_output[3, 4] = -x23*x36 - x29*x37
    jacobian_output[3, 5] = -x24*x38 + x30*x39
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x32
    jacobian_output[4, 3] = x11*x33 - x27*x41
    jacobian_output[4, 4] = x15*x36 + x29*x42
    jacobian_output[4, 5] = x18*x38 - x30*x43
    jacobian_output[5, 2] = x1*x34 + x40*x6
    jacobian_output[5, 3] = -x11*x35 + x21*x41
    jacobian_output[5, 4] = -x15*x37 + x23*x42
    jacobian_output[5, 5] = -x18*x39 + x24*x43
    return jacobian_output


def dense_cvr_038_angular_velocity_jacobian(theta_input: np.ndarray):
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


def dense_cvr_038_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x19 = a_2*x18 + d_3*x13 + pre_transform_special_symbol_23 - x6
    x20 = 1.0*x10*x11*x14 - x14*x9
    x21 = a_0*x12*x14 + d_1*x2
    x22 = a_2*x20 + d_3*x17 + x21
    x23 = math.sin(th_3)
    x24 = x18*x23
    x25 = math.cos(th_3)
    x26 = -x2*x25 - x20*x23
    x27 = -d_4*x24 + x19
    x28 = d_4*x26 + x22
    x29 = math.cos(th_4)
    x30 = math.sin(th_4)
    x31 = -x13*x29 - x18*x25*x30
    x32 = -x17*x29 - x30*(-x2*x23 + x20*x25)
    x33 = d_5*x31 + x27
    x34 = d_5*x32 + x28
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x14
    x37 = p_on_ee_z*x36
    x38 = x2*x4
    x39 = x11*x2
    x40 = -x10*x38 - x39*x8
    x41 = 1.0*x1*x10*x11 - x38*x8
    x42 = a_0*x39 - d_1*x36
    x43 = a_2*x41 + d_3*x40 + x42
    x44 = 1.0*x14*x25 - x23*x41
    x45 = d_4*x44 + x43
    x46 = -x29*x40 - x30*(x23*x36 + x25*x41)
    x47 = d_5*x46 + x45
    x48 = x0*x14 + x1*x35
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x7 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x26 - x24*x28 - x26*x27
    jacobian_output[0, 5] = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = -x36*x7 + x37
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x43 + x19*x40
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x44 + x18*x23*x45 + x27*x44
    jacobian_output[1, 5] = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46
    jacobian_output[2, 1] = -x48
    jacobian_output[2, 2] = 1.0*x1*x42 + 1.0*x14*x21 - x48
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x43 - x22*x40
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x44 + x26*x45 - x28*x44
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x46 + x32*x47 - x34*x46
    return jacobian_output


def dense_cvr_038_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(35):
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
        R_l[0, 6] = d_1
        R_l[0, 7] = -a_2
        R_l[1, 2] = -a_0
        R_l[1, 6] = -a_2
        R_l[1, 7] = -d_1
        R_l[2, 4] = a_0
        R_l[3, 6] = -1
        R_l[4, 7] = 1
        R_l[5, 5] = 2*a_0*a_2
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
        x4 = d_4*r_22
        x5 = -x4
        x6 = d_3*r_23
        x7 = -x6
        x8 = r_21**2
        x9 = Py*x8
        x10 = r_22**2
        x11 = Py*x10
        x12 = r_23**2
        x13 = Py*x12
        x14 = d_5*r_23
        x15 = Px*r_11
        x16 = r_21*x15
        x17 = Px*r_12
        x18 = r_22*x17
        x19 = Px*r_13
        x20 = r_23*x19
        x21 = Pz*r_31
        x22 = r_21*x21
        x23 = Pz*r_32
        x24 = r_22*x23
        x25 = Pz*r_33
        x26 = r_23*x25
        x27 = x11 + x13 - x14 + x16 + x18 + x20 + x22 + x24 + x26 + x9
        x28 = d_1 + x27
        x29 = x28 + x7
        x30 = d_4*x1
        x31 = d_3*x1
        x32 = -x31
        x33 = d_3*x3
        x34 = x28 + x6
        x35 = Py*r_21
        x36 = x15 + x21 + x35
        x37 = R_l_inv_50*a_0
        x38 = x36*x37
        x39 = Py*r_23
        x40 = -d_5 + x19 + x25 + x39
        x41 = R_l_inv_52*a_0
        x42 = x40*x41
        x43 = d_5*r_21
        x44 = r_23*x15
        x45 = r_23*x21
        x46 = r_21*x19
        x47 = r_21*x25
        x48 = x43 + x44 + x45 - x46 - x47
        x49 = R_l_inv_57*a_0
        x50 = x48*x49
        x51 = a_0*r_22
        x52 = R_l_inv_54*x51
        x53 = -x52
        x54 = R_l_inv_56*d_3*x51
        x55 = R_l_inv_55*a_0
        x56 = 2*d_3
        x57 = x40*x56
        x58 = x55*x57
        x59 = x38 + x42 + x50 + x53 + x54 + x58
        x60 = d_4*r_21
        x61 = a_0*r_21
        x62 = R_l_inv_53*x61
        x63 = d_5*r_22
        x64 = r_23*x17
        x65 = r_23*x23
        x66 = r_22*x19
        x67 = r_22*x25
        x68 = x63 + x64 + x65 - x66 - x67
        x69 = R_l_inv_56*a_0
        x70 = x68*x69
        x71 = Py*r_22
        x72 = x17 + x23 + x71
        x73 = R_l_inv_51*a_0
        x74 = x72*x73
        x75 = d_3*x61
        x76 = R_l_inv_57*x75
        x77 = 2*d_4
        x78 = x55*x72
        x79 = x77*x78
        x80 = x60 + x62 + x70 - x74 + x76 + x79
        x81 = d_4*r_23
        x82 = x69*x81
        x83 = r_22*x15
        x84 = r_22*x21
        x85 = r_21*x17
        x86 = r_21*x23
        x87 = x83 + x84 - x85 - x86
        x88 = -x82 + x87
        x89 = d_4*x73
        x90 = d_3**2
        x91 = d_4**2
        x92 = d_5**2
        x93 = a_0**2
        x94 = a_2**2
        x95 = -d_1**2
        x96 = 2*d_5
        x97 = Py*x14
        x98 = Py*x1
        x99 = 2*x17
        x100 = 2*x19
        x101 = 2*x15
        x102 = 2*x71
        x103 = 2*x39
        x104 = Px**2
        x105 = r_11**2
        x106 = x104*x105
        x107 = r_12**2
        x108 = x104*x107
        x109 = r_13**2
        x110 = x104*x109
        x111 = Py**2
        x112 = x111*x8
        x113 = x10*x111
        x114 = x111*x12
        x115 = Pz**2
        x116 = r_31**2
        x117 = x115*x116
        x118 = r_32**2
        x119 = x115*x118
        x120 = r_33**2*x115
        x121 = x100*x25 + x100*x39 + x101*x21 + x102*x23 + x103*x25 + x106 + x108 + x110 + x112 + x113 + x114 + x117 + x119 + x120 + x15*x98 - x19*x96 + x21*x98 + x23*x99 - x25*x96 + x71*x99 + x90 + x91 + x92 - x93 - x94 + x95 - 2*x97
        x122 = -R_l_inv_52*a_0*d_3 - R_l_inv_55*a_0*x121 - a_2 + x89
        x123 = 2*x36
        x124 = x123*x73
        x125 = 2*x37
        x126 = x125*x72
        x127 = R_l_inv_54*a_0
        x128 = x1*x127
        x129 = 2*x49
        x130 = x129*x68
        x131 = x31*x69
        x132 = -x131
        x133 = x126 + x128 + x130 + x132
        x134 = 2*x4
        x135 = 2*x69
        x136 = x135*x48
        x137 = 2*R_l_inv_53
        x138 = x137*x51
        x139 = x51*x56
        x140 = R_l_inv_57*x139
        x141 = x134 - x136 + x138 + x140
        x142 = d_3*x41
        x143 = x121*x55
        x144 = -x89
        x145 = a_2 + x142 + x143 + x144 + x38
        x146 = -x83
        x147 = -x84
        x148 = -x42
        x149 = -x58
        x150 = x146 + x147 + x148 + x149 + x82 + x85 + x86
        x151 = x50 + x53 + x54
        x152 = 2*x81
        x153 = a_0*r_23
        x154 = x137*x153
        x155 = x135*x87
        x156 = x129*x6
        x157 = x125*x40 + x152 + x154 + x155 + x156
        x158 = x123*x41
        x159 = 4*d_3
        x160 = x36*x55
        x161 = x159*x160
        x162 = -x158 - x161
        x163 = 2*x63
        x164 = 2*x64
        x165 = 2*x65
        x166 = 2*x66
        x167 = 2*x67
        x168 = x30*x69
        x169 = -x163 - x164 - x165 + x166 + x167 + x168
        x170 = 4*x72
        x171 = x170*x41
        x172 = 8*d_3
        x173 = x172*x78
        x174 = 4*x43
        x175 = 4*x46
        x176 = 4*x47
        x177 = 4*x44
        x178 = d_4*x3
        x179 = x174 - x175 - x176 + x177 + x178*x69 + 4*x45
        x180 = x158 + x161
        x181 = x163 + x164 + x165 - x166 - x167 - x168
        x182 = -x50
        x183 = -x54
        x184 = x182 + x183 + x42 + x52 + x58
        x185 = x74 - x79
        x186 = x185 + x60 + x62 + x70 + x76 + x88
        x187 = 4*d_4
        x188 = -x124 + x160*x187
        x189 = x141 + x188
        x190 = x122 + x185 + x60 + x62 + x70 + x76
        x191 = 2*a_0
        x192 = a_2*x191
        x193 = d_1*x191
        x194 = a_2*d_1
        x195 = 2*x194
        x196 = x93 + x94 + x95
        x197 = R_l_inv_22*x192 + R_l_inv_32*x193 + R_l_inv_62*x196 + R_l_inv_72*x195
        x198 = d_3*x197
        x199 = r_21**3*x111
        x200 = R_l_inv_25*x192 + R_l_inv_35*x193 + R_l_inv_65*x196 + R_l_inv_75*x195
        x201 = x121*x200
        x202 = R_l_inv_21*x192 + R_l_inv_31*x193 + R_l_inv_61*x196 + R_l_inv_71*x195
        x203 = d_4*x202
        x204 = -x203
        x205 = -r_21*x90
        x206 = -r_21*x91
        x207 = -r_21*x92
        x208 = R_l_inv_23*x192 + R_l_inv_33*x193 + R_l_inv_63*x196 + R_l_inv_73*x195
        x209 = -r_21*x208
        x210 = R_l_inv_20*x192 + R_l_inv_30*x193 + R_l_inv_60*x196 + R_l_inv_70*x195
        x211 = x210*x36
        x212 = -x211
        x213 = R_l_inv_66*x196
        x214 = R_l_inv_26*x192 + R_l_inv_36*x193 + R_l_inv_76*x195 + x213
        x215 = -x214*x68
        x216 = r_21*x106
        x217 = r_21*x113
        x218 = r_21*x114
        x219 = r_21*x117
        x220 = R_l_inv_67*x196
        x221 = R_l_inv_27*x192 + R_l_inv_37*x193 + R_l_inv_77*x195 + x220
        x222 = d_3*r_21
        x223 = -x221*x222
        x224 = -r_21*x108
        x225 = -r_21*x110
        x226 = -r_21*x119
        x227 = -r_21*x120
        x228 = -x101*x14
        x229 = 2*x21
        x230 = -x14*x229
        x231 = x101*x9
        x232 = x101*x11
        x233 = x101*x13
        x234 = d_5*x1
        x235 = x19*x234
        x236 = x229*x9
        x237 = x11*x229
        x238 = x13*x229
        x239 = x234*x25
        x240 = 2*r_11
        x241 = r_12*x104
        x242 = r_22*x241
        x243 = x240*x242
        x244 = 2*r_13
        x245 = r_23*x104
        x246 = r_11*x244*x245
        x247 = 2*r_31
        x248 = r_32*x115
        x249 = r_22*x248
        x250 = x247*x249
        x251 = r_23*r_33
        x252 = x115*x251
        x253 = x247*x252
        x254 = x1*x17
        x255 = -x23*x254
        x256 = x19*x25
        x257 = -x1*x256
        x258 = x15*x21
        x259 = x1*x258
        x260 = x101*x24
        x261 = x101*x26
        x262 = x18*x229
        x263 = x20*x229
        x264 = x198 + x199 + x201 + x204 + x205 + x206 + x207 + x209 + x212 + x215 + x216 + x217 + x218 + x219 + x223 + x224 + x225 + x226 + x227 + x228 + x230 + x231 + x232 + x233 + x235 + x236 + x237 + x238 + x239 + x243 + x246 + x250 + x253 + x255 + x257 + x259 + x260 + x261 + x262 + x263
        x265 = x202*x72
        x266 = x197*x40
        x267 = -x266
        x268 = x200*x57
        x269 = -x268
        x270 = x200*x72
        x271 = x270*x77
        x272 = -x271
        x273 = x265 + x267 + x269 + x272
        x274 = x214*x81
        x275 = x134*x15
        x276 = x134*x21
        x277 = x17*x30
        x278 = x23*x30
        x279 = x274 - x275 - x276 + x277 + x278
        x280 = R_l_inv_24*x192 + R_l_inv_34*x193 + R_l_inv_64*x196 + R_l_inv_74*x195
        x281 = r_22*x280
        x282 = x221*x48
        x283 = d_3*r_22
        x284 = x214*x283
        x285 = d_5*x31
        x286 = x101*x6
        x287 = x229*x6
        x288 = x19*x31
        x289 = x25*x31
        x290 = x281 - x282 - x284 - x285 - x286 - x287 + x288 + x289
        x291 = x279 + x290
        x292 = 4*a_0
        x293 = a_2*x292
        x294 = d_1*x292
        x295 = 4*x194
        x296 = R_l_inv_26*x293 + R_l_inv_36*x294 + R_l_inv_76*x295 + 2*x213
        x297 = x296*x48
        x298 = 2*x23
        x299 = x102 + x298 + x99
        x300 = x210*x299
        x301 = r_22*x90
        x302 = -2*x301
        x303 = r_22*x91
        x304 = -2*x303
        x305 = r_22*x92
        x306 = -2*x305
        x307 = 2*x208
        x308 = -r_22*x307
        x309 = r_22**3*x111
        x310 = 2*x309
        x311 = r_22*x56
        x312 = -x221*x311
        x313 = r_22*x106
        x314 = -2*x313
        x315 = r_22*x110
        x316 = -2*x315
        x317 = r_22*x117
        x318 = -2*x317
        x319 = r_22*x120
        x320 = -2*x319
        x321 = r_22*x108
        x322 = 2*x321
        x323 = r_22*x112
        x324 = 2*x323
        x325 = r_22*x114
        x326 = 2*x325
        x327 = r_22*x119
        x328 = 2*x327
        x329 = 4*x17
        x330 = -x14*x329
        x331 = 4*x23
        x332 = -x14*x331
        x333 = x329*x9
        x334 = x11*x329
        x335 = x13*x329
        x336 = d_5*x3
        x337 = x19*x336
        x338 = x331*x9
        x339 = x11*x331
        x340 = x13*x331
        x341 = x25*x336
        x342 = 4*r_11
        x343 = r_21*x342
        x344 = x241*x343
        x345 = 4*r_12
        x346 = r_13*x245
        x347 = x345*x346
        x348 = 4*x248
        x349 = r_21*r_31
        x350 = x348*x349
        x351 = x251*x348
        x352 = -x258*x3
        x353 = -x256*x3
        x354 = x16*x331
        x355 = x22*x329
        x356 = x17*x3
        x357 = x23*x356
        x358 = x26*x329
        x359 = x20*x331
        x360 = x297 - x300 + x302 + x304 + x306 + x308 + x310 + x312 + x314 + x316 + x318 + x320 + x322 + x324 + x326 + x328 + x330 + x332 + x333 + x334 + x335 + x337 + x338 + x339 + x340 + x341 + x344 + x347 + x350 + x351 + x352 + x353 + x354 + x355 + x357 + x358 + x359
        x361 = x101 + x229 + x98
        x362 = x202*x361
        x363 = x200*x36
        x364 = x187*x363 - x362
        x365 = R_l_inv_27*x293 + R_l_inv_37*x294 + R_l_inv_77*x295 + 2*x220
        x366 = x365*x68
        x367 = x1*x280
        x368 = d_5*x33
        x369 = x214*x31
        x370 = x329*x6
        x371 = x331*x6
        x372 = x19*x33
        x373 = x25*x33
        x374 = -x366 - x367 - x368 + x369 - x370 - x371 + x372 + x373
        x375 = -x274
        x376 = -x277
        x377 = -x278
        x378 = x266 + x268 + x275 + x276 + x375 + x376 + x377
        x379 = -d_3*x197
        x380 = -x121*x200
        x381 = x199 + x203 + x205 + x206 + x207 + x209 + x212 + x215 + x216 + x217 + x218 + x219 + x223 + x224 + x225 + x226 + x227 + x228 + x230 + x231 + x232 + x233 + x235 + x236 + x237 + x238 + x239 + x243 + x246 + x250 + x253 + x255 + x257 + x259 + x260 + x261 + x262 + x263 + x379 + x380
        x382 = x265 + x272
        x383 = x197*x361
        x384 = 2*x25
        x385 = x100 + x103 + x384 - x96
        x386 = x159*x363
        x387 = -x210*x385 + x383 + x386
        x388 = x214*x30
        x389 = d_5*x178
        x390 = x178*x19
        x391 = x178*x25
        x392 = x187*x64
        x393 = x187*x65
        x394 = -x388 + x389 - x390 - x391 + x392 + x393
        x395 = x1*x23
        x396 = -x254 - x395 + 2*x83 + 2*x84
        x397 = x214*x396
        x398 = r_23*x90
        x399 = 2*x398
        x400 = r_23*x91
        x401 = 2*x400
        x402 = r_23*x307
        x403 = r_23*x92
        x404 = 2*x403
        x405 = r_23**3*x111
        x406 = 2*x405
        x407 = 4*d_5
        x408 = x407*x9
        x409 = x11*x407
        x410 = x13*x407
        x411 = 2*x6
        x412 = x221*x411
        x413 = r_23*x106
        x414 = 2*x413
        x415 = r_23*x108
        x416 = 2*x415
        x417 = r_23*x117
        x418 = 2*x417
        x419 = r_23*x119
        x420 = 2*x419
        x421 = r_23*x110
        x422 = 2*x421
        x423 = r_23*x112
        x424 = 2*x423
        x425 = r_23*x113
        x426 = 2*x425
        x427 = r_23*x120
        x428 = 2*x427
        x429 = x15*x174
        x430 = x17*x336
        x431 = 4*x19
        x432 = x14*x431
        x433 = x174*x21
        x434 = x23*x336
        x435 = 4*x25
        x436 = x14*x435
        x437 = x431*x9
        x438 = x11*x431
        x439 = x13*x431
        x440 = x435*x9
        x441 = x11*x435
        x442 = x13*x435
        x443 = r_13*x104
        x444 = x343*x443
        x445 = x241*x3
        x446 = r_13*x445
        x447 = r_33*x115
        x448 = 4*x349*x447
        x449 = x248*x3
        x450 = r_33*x449
        x451 = x177*x21
        x452 = x331*x64
        x453 = x16*x435
        x454 = x25*x356
        x455 = x22*x431
        x456 = x23*x3
        x457 = x19*x456
        x458 = x20*x435
        x459 = -x397 - x399 - x401 - x402 + x404 + x406 - x408 - x409 - x410 - x412 - x414 - x416 - x418 - x420 + x422 + x424 + x426 + x428 - x429 - x430 - x432 - x433 - x434 - x436 + x437 + x438 + x439 + x440 + x441 + x442 + x444 + x446 + x448 + x450 - x451 - x452 + x453 + x454 + x455 + x457 + x458
        x460 = Py*x3
        x461 = x329 + x331 + x460
        x462 = x197*x461
        x463 = x172*x270
        x464 = 8*d_5
        x465 = 8*x60
        x466 = 8*d_4
        x467 = x178*x214 - x19*x465 - x25*x465 + x44*x466 + x45*x466 + x464*x60
        x468 = x210*x385 + x383 + x386
        x469 = x397 + x399 + x401 + x402 - x404 - x406 + x408 + x409 + x410 + x412 + x414 + x416 + x418 + x420 - x422 - x424 - x426 - x428 + x429 + x430 + x432 + x433 + x434 + x436 - x437 - x438 - x439 - x440 - x441 - x442 - x444 - x446 - x448 - x450 + x451 + x452 - x453 - x454 - x455 - x457 - x458
        x470 = -x265 + x271
        x471 = x267 + x269 + x470
        x472 = -x281 + x282 + x284 + x285 + x286 + x287 - x288 - x289
        x473 = x279 + x472
        x474 = -4*d_4*x200*x36 + x362
        x475 = x366 + x367 + x368 - x369 + x370 + x371 - x372 - x373
        x476 = R_l_inv_23*x193 - R_l_inv_33*x192 + R_l_inv_63*x195 - R_l_inv_73*x196
        x477 = r_21*x476
        x478 = R_l_inv_20*x193 - R_l_inv_30*x192 + R_l_inv_60*x195 - R_l_inv_70*x196
        x479 = x36*x478
        x480 = R_l_inv_22*x193 - R_l_inv_32*x192 + R_l_inv_62*x195 - R_l_inv_72*x196
        x481 = x40*x480
        x482 = R_l_inv_36*x192
        x483 = R_l_inv_76*x196
        x484 = R_l_inv_26*x193 + R_l_inv_66*x195 - x482 - x483
        x485 = x484*x68
        x486 = R_l_inv_77*x196
        x487 = R_l_inv_27*x193 - R_l_inv_37*x192 + R_l_inv_67*x195 - x486
        x488 = x222*x487
        x489 = x484*x81
        x490 = -x489
        x491 = x56*x63
        x492 = -x491
        x493 = x6*x77
        x494 = R_l_inv_25*x193 - R_l_inv_35*x192 + R_l_inv_65*x195 - R_l_inv_75*x196
        x495 = x494*x57
        x496 = x6*x99
        x497 = -x496
        x498 = x298*x6
        x499 = -x498
        x500 = x56*x66
        x501 = x56*x67
        x502 = x477 + x479 + x481 + x485 + x488 + x490 + x492 + x493 + x495 + x497 + x499 + x500 + x501
        x503 = x48*x487
        x504 = -x301
        x505 = -x305
        x506 = R_l_inv_24*x193 - R_l_inv_34*x192 + R_l_inv_64*x195 - R_l_inv_74*x196
        x507 = r_22*x506
        x508 = -x507
        x509 = R_l_inv_21*x193 - R_l_inv_31*x192 + R_l_inv_61*x195 - R_l_inv_71*x196
        x510 = x509*x72
        x511 = x283*x484
        x512 = -x313
        x513 = -x315
        x514 = -x317
        x515 = -x319
        x516 = x494*x72
        x517 = x516*x77
        x518 = x14*x99
        x519 = -x518
        x520 = x14*x298
        x521 = -x520
        x522 = x9*x99
        x523 = x11*x99
        x524 = x13*x99
        x525 = x163*x19
        x526 = x298*x9
        x527 = x11*x298
        x528 = x13*x298
        x529 = x163*x25
        x530 = r_11*x1
        x531 = x241*x530
        x532 = r_23*x241*x244
        x533 = r_31*x1
        x534 = x248*x533
        x535 = x248*x251
        x536 = 2*x535
        x537 = x229*x83
        x538 = -x537
        x539 = x384*x66
        x540 = -x539
        x541 = x15*x395
        x542 = x21*x254
        x543 = x18*x298
        x544 = x26*x99
        x545 = x20*x298
        x546 = x303 + x309 + x321 + x323 + x325 + x327 + x503 + x504 + x505 + x508 - x510 + x511 + x512 + x513 + x514 + x515 + x517 + x519 + x521 + x522 + x523 + x524 + x525 + x526 + x527 + x528 + x529 + x531 + x532 + x534 + x536 + x538 + x540 + x541 + x542 + x543 + x544 + x545
        x547 = d_4*x509
        x548 = -d_3*x480 - x121*x494 + x547
        x549 = x77*x9
        x550 = x11*x77
        x551 = x13*x77
        x552 = x14*x77
        x553 = x15*x30
        x554 = x134*x17
        x555 = x20*x77
        x556 = x21*x30
        x557 = x134*x23
        x558 = x26*x77
        x559 = -x549 - x550 - x551 + x552 - x553 - x554 - x555 - x556 - x557 - x558
        x560 = x299*x478
        x561 = R_l_inv_27*x294 - R_l_inv_37*x293 + R_l_inv_67*x295 - 2*x486
        x562 = -x561*x68
        x563 = x1*x90
        x564 = -x563
        x565 = -x1*x506
        x566 = x1*x91
        x567 = x31*x484
        x568 = 4*x15
        x569 = 4*x21
        x570 = r_11*x445 + 4*r_31*x252 + r_31*x449 + x1*x106 - x1*x108 - x1*x110 + x1*x113 + x1*x114 + x1*x117 - x1*x119 - x1*x120 - x1*x92 + x11*x568 + x11*x569 + x13*x568 + x13*x569 - x14*x568 - x14*x569 + x15*x456 + x16*x569 + x174*x19 + x174*x25 - x175*x25 + 2*x199 + x20*x569 + x21*x356 + x26*x568 - x331*x85 + x342*x346 + x568*x9 + x569*x9
        x571 = -x560 + x562 + x564 + x565 + x566 + x567 + x570
        x572 = x361*x509
        x573 = x36*x494
        x574 = x187*x573 - x572
        x575 = R_l_inv_26*x294 - R_l_inv_36*x293 + R_l_inv_66*x295 - 2*x483
        x576 = x48*x575
        x577 = 2*x476
        x578 = r_22*x577
        x579 = d_3*x174
        x580 = x311*x487
        x581 = x568*x6
        x582 = x569*x6
        x583 = d_3*x175
        x584 = d_3*x176
        x585 = x576 - x578 - x579 - x580 - x581 - x582 + x583 + x584
        x586 = d_3*x480
        x587 = x121*x494
        x588 = -x547
        x589 = -x493
        x590 = x479 + x489 + x586 + x587 + x588 + x589
        x591 = -x481
        x592 = -x495
        x593 = x477 + x485 + x488 + x492 + x497 + x499 + x500 + x501 + x591 + x592
        x594 = x549 + x550 + x551 - x552 + x553 + x554 + x555 + x556 + x557 + x558
        x595 = x546 + x594
        x596 = x361*x480
        x597 = x159*x573
        x598 = x385*x478
        x599 = x159*x60
        x600 = -x599
        x601 = x30*x484
        x602 = x598 + x600 + x601
        x603 = x396*x484
        x604 = r_23*x577
        x605 = x411*x487
        x606 = x15*x33
        x607 = x21*x33
        x608 = x159*x85
        x609 = x159*x86
        x610 = x603 + x604 + x605 - x606 - x607 + x608 + x609
        x611 = x178*x484
        x612 = x172*x4
        x613 = x172*x516 + x461*x480
        x614 = x596 + x597
        x615 = x610 + x614
        x616 = x586 + x587 + x588
        x617 = x510 - x517
        x618 = x594 + x617
        x619 = x560 + x562 + x564 + x565 + x566 + x567 + x570
        x620 = -x576 + x578 + x579 + x580 + x581 + x582 - x583 - x584
        x621 = -x479
        x622 = x481 + x495 + x621
        x623 = x490 + x493 + x616
        x624 = -x477 - x485 - x488 + x491 + x496 + x498 - x500 - x501
        x625 = x4*x56
        x626 = x4*x96
        x627 = x134*x19
        x628 = -x627
        x629 = x134*x25
        x630 = -x629
        x631 = R_l_inv_41*x193*x72
        x632 = -x631
        x633 = R_l_inv_44*d_1
        x634 = 2*x51*x633
        x635 = -x634
        x636 = x64*x77
        x637 = x65*x77
        x638 = a_0*d_1
        x639 = R_l_inv_40*x638
        x640 = x123*x639
        x641 = R_l_inv_43*x638
        x642 = x1*x641
        x643 = R_l_inv_46*x193*x68
        x644 = R_l_inv_47*x193*x48
        x645 = R_l_inv_46*d_1*x139
        x646 = R_l_inv_47*x638
        x647 = x31*x646
        x648 = R_l_inv_45*x638
        x649 = x187*x648*x72
        x650 = x625 + x626 + x628 + x630 + x632 + x635 + x636 + x637 + x640 + x642 + x643 + x644 + x645 + x647 + x649
        x651 = R_l_inv_42*x56*x638
        x652 = -x651
        x653 = R_l_inv_45*x121*x193
        x654 = -x653
        x655 = R_l_inv_41*x638*x77
        x656 = R_l_inv_42*x193*x40
        x657 = x294*x40
        x658 = R_l_inv_45*d_3
        x659 = x657*x658
        x660 = x652 + x654 + x655 + x656 + x659
        x661 = x56*x9
        x662 = x11*x56
        x663 = x13*x56
        x664 = x6*x96
        x665 = x15*x31
        x666 = x18*x56
        x667 = x100*x6
        x668 = x21*x31
        x669 = x24*x56
        x670 = x384*x6
        x671 = x9*x96
        x672 = x11*x96
        x673 = x13*x96
        x674 = x15*x234
        x675 = x163*x17
        x676 = x100*x14
        x677 = x21*x234
        x678 = x163*x23
        x679 = x14*x384
        x680 = x100*x9
        x681 = x100*x11
        x682 = x100*x13
        x683 = x384*x9
        x684 = x11*x384
        x685 = x13*x384
        x686 = x443*x530
        x687 = x242*x244
        x688 = x447*x533
        x689 = 2*r_33*x249
        x690 = x229*x44
        x691 = x298*x64
        x692 = R_l_inv_46*x638
        x693 = x152*x692
        x694 = x1*x15*x25
        x695 = x18*x384
        x696 = x1*x19*x21
        x697 = x100*x24
        x698 = x20*x384
        x699 = x398 - x400 + x403 + x405 - x413 - x415 - x417 - x419 + x421 + x423 + x425 + x427 - x671 - x672 - x673 - x674 - x675 - x676 - x677 - x678 - x679 + x680 + x681 + x682 + x683 + x684 + x685 + x686 + x687 + x688 + x689 - x690 - x691 - x693 + x694 + x695 + x696 + x697 + x698
        x700 = -x661 - x662 - x663 + x664 - x665 - x666 - x667 - x668 - x669 - x670 + x699
        x701 = x170*x639
        x702 = x294*x36
        x703 = x36*x648
        x704 = x466*x703
        x705 = R_l_inv_41*x702 + x701 - x704
        x706 = 4*x61*x633
        x707 = R_l_inv_47*x294
        x708 = x68*x707
        x709 = 4*d_1
        x710 = R_l_inv_46*x709*x75
        x711 = x600 + x706 + x708 - x710
        x712 = x407*x60
        x713 = x187*x44
        x714 = x187*x45
        x715 = R_l_inv_46*x294
        x716 = x48*x715
        x717 = x431*x60
        x718 = x435*x60
        x719 = x3*x641
        x720 = x33*x646
        x721 = -x712 - x713 - x714 - x716 + x717 + x718 + x719 + x720
        x722 = -2*R_l_inv_41*a_0*d_1*d_4
        x723 = -x656
        x724 = -x659
        x725 = x651 + x653 + x722 + x723 + x724
        x726 = x661 + x662 + x663 - x664 + x665 + x666 + x667 + x668 + x669 + x670
        x727 = -x398 + x400 - x403 - x405 + x413 + x415 + x417 + x419 - x421 - x423 - x425 - x427 + x671 + x672 + x673 + x674 + x675 + x676 + x677 + x678 + x679 - x680 - x681 - x682 - x683 - x684 - x685 - x686 - x687 - x688 - x689 + x690 + x691 + x693 - x694 - x695 - x696 - x697 - x698 + x726
        x728 = -x566
        x729 = R_l_inv_40*x657
        x730 = -x60*x715
        x731 = x563 + x570 + x728 - x729 + x730
        x732 = R_l_inv_42*x702
        x733 = x172*x703
        x734 = x732 + x733
        x735 = x15*x178
        x736 = x178*x21
        x737 = R_l_inv_43*x153*x709
        x738 = x715*x87
        x739 = x329*x60
        x740 = x331*x60
        x741 = x6*x707
        x742 = -x735 - x736 - x737 - x738 + x739 + x740 - x741
        x743 = 8*a_0
        x744 = d_1*x743
        x745 = 8*x14
        x746 = 8*x17
        x747 = 8*x63
        x748 = 8*x23
        x749 = 8*r_12
        x750 = r_11*r_21
        x751 = 8*x248
        x752 = 8*x21
        x753 = 8*x25
        x754 = R_l_inv_46*x744
        x755 = x104*x749*x750 - x106*x3 + x108*x3 + x11*x746 + x11*x748 - x110*x3 + x112*x3 + x114*x3 - x117*x3 + x119*x3 - x120*x3 + x13*x746 + x13*x748 + x16*x748 - x17*x745 + x18*x748 + x19*x747 + x20*x748 + x22*x746 - x23*x745 + x25*x747 + x26*x746 + x3*x90 - x3*x91 - x3*x92 + 4*x309 + x346*x749 + x349*x751 - x4*x754 + 8*x535 - x66*x753 + x746*x9 + x748*x9 - x752*x83
        x756 = x563 + x570 + x728 + x729 + x730
        x757 = x735 + x736 + x737 + x738 - x739 - x740 + x741
        x758 = -x649
        x759 = x631 + x640 + x699 + x726 + x758
        x760 = x651 + x653 + x656 + x659 + x722
        x761 = x626 + x628 + x630 + x636 + x637 + x642 + x643 + x647
        x762 = -x625 + x634 - x644 - x645 + x761
        x763 = -4*R_l_inv_41*a_0*d_1*x36 + x701 + x704
        x764 = x599 - x706 - x708 + x710
        x765 = -x640
        x766 = x625 + x635 + x644 + x645 + x765
        x767 = x632 + x649 + x699 + x726
        x768 = -x626 + x627 + x629 - x636 - x637 - x642 - x643 - x647
        x769 = -x244
        x770 = r_12*x77
        x771 = -x770
        x772 = d_3*x244
        x773 = 2*Px
        x774 = r_11*x229 + r_11*x98 + r_12*x102 + r_12*x298 - r_13*x96 + x105*x773 + x107*x773 + x109*x773 + x244*x25 + x244*x39
        x775 = -x772 + x774
        x776 = d_4*x342
        x777 = d_3*x342
        x778 = d_3*x749
        x779 = x772 + x774
        x780 = 2*r_12
        x781 = x127*x780
        x782 = d_5*r_11
        x783 = r_13*x35
        x784 = r_13*x21
        x785 = r_11*x39
        x786 = r_11*x25
        x787 = x782 + x783 + x784 - x785 - x786
        x788 = x129*x787
        x789 = r_12*x56
        x790 = x69*x789
        x791 = R_l_inv_53*a_0
        x792 = d_5*r_12
        x793 = r_13*x71
        x794 = r_13*x23
        x795 = r_12*x39
        x796 = r_12*x25
        x797 = x792 + x793 + x794 - x795 - x796
        x798 = r_11*x56
        x799 = r_11*x77 + x135*x797 + x240*x791 + x49*x798
        x800 = -x781 + x788 + x790 + x799
        x801 = r_11*x102
        x802 = r_11*x298
        x803 = r_12*x98
        x804 = r_12*x229
        x805 = d_4*x244
        x806 = x69*x805
        x807 = -x801 - x802 + x803 + x804 - x806
        x808 = x127*x342
        x809 = 4*x49*x797
        x810 = r_12*x187
        x811 = 4*x69
        x812 = d_3*x345
        x813 = x345*x791 + x49*x812 - x787*x811 + x810
        x814 = x801 + x802 - x803 - x804 + x806
        x815 = 4*x792
        x816 = r_13*x460
        x817 = r_13*x331
        x818 = 4*x795
        x819 = 4*x796
        x820 = x69*x776
        x821 = r_13*x187
        x822 = r_11*x71
        x823 = r_11*x23
        x824 = r_12*x35
        x825 = r_12*x21
        x826 = x822 + x823 - x824 - x825
        x827 = 4*r_13
        x828 = d_3*x827
        x829 = x49*x828 + x791*x827 - x811*x826 + x821
        x830 = 8*x782
        x831 = 8*x785
        x832 = 8*x783
        x833 = 8*x786
        x834 = 8*x784
        x835 = d_4*x749
        x836 = x781 - x788 - x790 + x799
        x837 = x214*x805
        x838 = Py*x345
        x839 = x60*x838
        x840 = x21*x810
        x841 = r_11*x460
        x842 = d_4*x841
        x843 = x23*x776
        x844 = r_11**3
        x845 = 2*x104
        x846 = Px*x105
        x847 = 4*x35
        x848 = Px*x847
        x849 = Px*x569
        x850 = d_5*x25
        x851 = r_21*x111
        x852 = r_12*x3
        x853 = r_31*x248
        x854 = r_23*x851
        x855 = r_31*x447
        x856 = x25*x39
        x857 = x21*x35
        x858 = 4*x39
        x859 = -r_11*x307 + x107*x848 + x107*x849 + x108*x240 + x109*x848 + x109*x849 + x110*x240 + x112*x240 - x113*x240 - x114*x240 + x117*x240 - x119*x240 - x120*x240 - x221*x798 - x240*x90 - x240*x91 - x240*x92 - x296*x797 + x331*x824 + x342*x850 - x342*x856 + x342*x857 + x342*x97 + x345*x853 - x407*x783 - x407*x784 + x435*x783 - x460*x823 + x460*x825 + x569*x846 + x784*x858 + x827*x854 + x827*x855 + x844*x845 + x846*x847 + x851*x852
        x860 = x837 - x839 - x840 + x842 + x843 + x859
        x861 = x365*x787
        x862 = x280*x780
        x863 = d_5*x777
        x864 = r_12*x214
        x865 = x56*x864
        x866 = x159*x783
        x867 = x159*x784
        x868 = Py*x6
        x869 = x342*x868
        x870 = x25*x777
        x871 = -x861 + x862 - x863 - x865 - x866 - x867 + x869 + x870
        x872 = a_2*x743
        x873 = 8*x194
        x874 = x797*(R_l_inv_27*x872 + R_l_inv_37*x744 + R_l_inv_77*x873 + 4*x220)
        x875 = x280*x342
        x876 = d_5*x778
        x877 = x172*x793
        x878 = x172*x794
        x879 = r_12**3
        x880 = 4*x104
        x881 = 8*x71
        x882 = Px*x881
        x883 = Px*x748
        x884 = r_22*x111
        x885 = 8*x884
        x886 = r_13*r_23
        x887 = r_13*r_33
        x888 = 8*x35
        x889 = x71*x749
        x890 = 8*x39
        x891 = -d_3*x221*x345 + r_11*r_31*x751 + x106*x345 + x107*x882 + x107*x883 + x109*x882 + x109*x883 + x110*x345 - x112*x345 + x113*x345 - x114*x345 - x117*x345 + x119*x345 - x120*x345 - x208*x345 + x23*x889 - x345*x90 - x345*x91 - x345*x92 - x464*x793 - x464*x794 + x748*x846 + x749*x850 - x749*x856 - x749*x857 + x749*x97 + x750*x885 + x751*x887 + x752*x822 + x753*x793 + x787*(R_l_inv_26*x872 + R_l_inv_36*x744 + R_l_inv_76*x873 + 4*x213) + x794*x890 + x823*x888 + x846*x881 + x879*x880 + x885*x886
        x892 = -x837 + x839 + x840 - x842 - x843 + x859
        x893 = x214*x776
        x894 = d_5*x835
        x895 = x39*x835
        x896 = x25*x835
        x897 = 8*r_13
        x898 = Py*x4
        x899 = x897*x898
        x900 = x466*x794
        x901 = 4*x824
        x902 = x23*x342
        x903 = -4*x825 + x841 - x901 + x902
        x904 = r_13**3
        x905 = Px*x890
        x906 = Px*x753
        x907 = x111*x750
        x908 = 8*r_11
        x909 = r_33*x248
        x910 = x23*x749
        x911 = -8*Px*d_5*x105 - 8*Px*d_5*x107 - 8*Px*d_5*x109 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_5*r_11*r_21 - 8*Py*d_5*r_12*r_22 - 8*Py*d_5*r_13*r_23 - 8*Pz*d_5*r_11*r_31 - 8*Pz*d_5*r_12*r_32 - 8*Pz*d_5*r_13*r_33 - 4*d_3*r_13*x221 - 4*r_13*x10*x111 - 4*r_13*x111*x8 - 4*r_13*x115*x116 - 4*r_13*x115*x118 - 4*r_13*x208 + r_13*x25*x890 - 4*r_13*x90 - 4*r_13*x91 + r_23*x749*x884 + 8*r_23*x907 + x106*x827 + x107*x905 + x107*x906 + x108*x827 + x109*x905 + x109*x906 + x114*x827 + x120*x827 + x21*x831 + x214*x903 + x25*x889 + x35*x833 + x39*x910 + x749*x909 + x753*x846 + x827*x92 + x846*x890 + x855*x908 + x880*x904
        x912 = x861 - x862 + x863 + x865 + x866 + x867 - x869 - x870
        x913 = x561*x787
        x914 = -x780*x90
        x915 = -x780*x92
        x916 = -x506*x780
        x917 = x780*x91
        x918 = x845*x879
        x919 = x484*x805
        x920 = -x112*x780
        x921 = -x114*x780
        x922 = -x117*x780
        x923 = -x120*x780
        x924 = x484*x789
        x925 = x106*x780
        x926 = x110*x780
        x927 = x113*x780
        x928 = x119*x780
        x929 = d_3*x821
        x930 = -d_5*x816
        x931 = -d_5*x817
        x932 = x460*x846
        x933 = Px*x460
        x934 = x107*x933
        x935 = x109*x933
        x936 = x331*x846
        x937 = Px*x331
        x938 = x107*x937
        x939 = x109*x937
        x940 = x345*x97
        x941 = x25*x815
        x942 = x3*x907
        x943 = x342*x853
        x944 = x111*x3*x886
        x945 = x827*x909
        x946 = -x21*x901
        x947 = -x25*x818
        x948 = x35*x902
        x949 = x21*x841
        x950 = r_12*x23
        x951 = x460*x950
        x952 = x25*x816
        x953 = x39*x817
        x954 = x913 + x914 + x915 + x916 + x917 + x918 - x919 + x920 + x921 + x922 + x923 + x924 + x925 + x926 + x927 + x928 + x929 + x930 + x931 + x932 + x934 + x935 + x936 + x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x946 + x947 + x948 + x949 + x951 + x952 + x953
        x955 = x575*x797
        x956 = r_11*x577
        x957 = d_3*x815
        x958 = x487*x798
        x959 = Py*x33
        x960 = r_13*x959
        x961 = d_3*x817
        x962 = x6*x838
        x963 = x159*x796
        x964 = x955 + x956 - x957 + x958 - x960 - x961 + x962 + x963
        x965 = d_5*x821
        x966 = -4*Px*d_4*x105 - 4*Px*d_4*x107 - 4*Px*d_4*x109 - 4*Py*d_4*r_11*r_21 - 4*Py*d_4*r_12*r_22 - 4*Py*d_4*r_13*r_23 - 4*Pz*d_4*r_11*r_31 - 4*Pz*d_4*r_12*r_32 - 4*Pz*d_4*r_13*r_33 + x965
        x967 = x787*(R_l_inv_26*x744 - R_l_inv_36*x872 + R_l_inv_66*x873 - 4*x483)
        x968 = x345*x476
        x969 = d_3*x830
        x970 = d_3*x487
        x971 = x345*x970
        x972 = x172*x783
        x973 = x172*x784
        x974 = x868*x908
        x975 = x172*x786
        x976 = x342*x90
        x977 = x342*x91
        x978 = Px*x888
        x979 = Px*x752
        x980 = r_22*x749*x851 + x107*x978 + x107*x979 + x108*x342 + x109*x978 + x109*x979 + x110*x342 + x112*x342 - x113*x342 - x114*x342 + x117*x342 - x119*x342 - x120*x342 + x21*x889 + x25*x830 - x25*x831 + x25*x832 - x342*x92 + x35*x910 + x39*x834 - x464*x783 - x464*x784 - x748*x822 + x749*x853 + x752*x846 + x844*x880 + x846*x888 + x854*x897 + x855*x897 + x857*x908 + x908*x97
        x981 = -x342*x506 + x484*x777 - x797*(R_l_inv_27*x744 - R_l_inv_37*x872 + R_l_inv_67*x873 - 4*x486) - x976 + x977 + x980
        x982 = Px*x187
        x983 = Py*x60
        x984 = d_4*r_12
        x985 = x105*x982 + x107*x982 + x109*x982 + x21*x776 + x25*x821 + x331*x984 + x342*x983 + x39*x821 + x460*x984 - x965
        x986 = x913 + x914 + x915 + x916 + x917 + x918 + x919 + x920 + x921 + x922 + x923 + x924 + x925 + x926 + x927 + x928 - x929 + x930 + x931 + x932 + x934 + x935 + x936 + x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x946 + x947 + x948 + x949 + x951 + x952 + x953
        x987 = r_11*x466
        x988 = d_3*x987
        x989 = -x988
        x990 = x484*x776
        x991 = x172*x822 + x172*x823 - x21*x778 - x35*x778 + x476*x827 - x484*x903 + x827*x970
        x992 = -x955 - x956 + x957 - x958 + x960 + x961 - x962 - x963
        x993 = Px*x159
        x994 = x105*x993
        x995 = x107*x993
        x996 = x109*x993
        x997 = x35*x777
        x998 = r_12*x959
        x999 = x827*x868
        x1000 = x21*x777
        x1001 = d_3*r_12*x331
        x1002 = x25*x828
        x1003 = d_3*x810
        x1004 = x187*x792
        x1005 = x187*x795
        x1006 = -x1005
        x1007 = x187*x796
        x1008 = -x1007
        x1009 = R_l_inv_44*x345*x638
        x1010 = -x1009
        x1011 = d_4*x816
        x1012 = d_4*x817
        x1013 = x342*x641
        x1014 = x715*x797
        x1015 = x707*x787
        x1016 = x692*x812
        x1017 = x646*x777
        x1018 = x1003 + x1004 + x1006 + x1008 + x1010 + x1011 + x1012 + x1013 + x1014 + x1015 + x1016 + x1017
        x1019 = x244*x91
        x1020 = x244*x90
        x1021 = x244*x92
        x1022 = x845*x904
        x1023 = Px*x407
        x1024 = x1023*x105
        x1025 = x1023*x107
        x1026 = x1023*x109
        x1027 = x112*x244
        x1028 = x113*x244
        x1029 = x117*x244
        x1030 = x119*x244
        x1031 = x106*x244
        x1032 = x108*x244
        x1033 = x114*x244
        x1034 = x120*x244
        x1035 = d_5*x342
        x1036 = x1035*x35
        x1037 = x460*x792
        x1038 = x827*x97
        x1039 = x1035*x21
        x1040 = x331*x792
        x1041 = x827*x850
        x1042 = x846*x858
        x1043 = Px*x858
        x1044 = x1043*x107
        x1045 = x1043*x109
        x1046 = x435*x846
        x1047 = Px*x435
        x1048 = x1047*x107
        x1049 = x1047*x109
        x1050 = r_23*x111
        x1051 = x1050*x343
        x1052 = x342*x855
        x1053 = x1050*x852
        x1054 = x345*x909
        x1055 = x569*x783
        x1056 = x460*x794
        x1057 = x692*x821
        x1058 = x25*x342*x35
        x1059 = x21*x342*x39
        x1060 = x460*x796
        x1061 = x331*x795
        x1062 = x827*x856
        x1063 = -x1019 + x1020 + x1021 + x1022 - x1024 - x1025 - x1026 - x1027 - x1028 - x1029 - x1030 + x1031 + x1032 + x1033 + x1034 - x1036 - x1037 - x1038 - x1039 - x1040 - x1041 + x1042 + x1044 + x1045 + x1046 + x1048 + x1049 + x1051 + x1052 + x1053 + x1054 - x1055 - x1056 - x1057 + x1058 + x1059 + x1060 + x1061 + x1062
        x1064 = r_11*x172*x692
        x1065 = -Py*r_13*x465 - d_4*x830 - x466*x784 + x466*x785 + x466*x786 + x641*x749 + x646*x778 - x754*x787
        x1066 = -4*d_3*d_5*r_13 + x1000 + x1001 + x1002 + x994 + x995 + x996 + x997 + x998 + x999
        x1067 = x749*x983
        x1068 = x21*x835
        x1069 = R_l_inv_43*r_13*x744
        x1070 = r_13*x172*x646
        x1071 = -x692*x987 + x976 - x977 + x980
        x1072 = 16*d_5
        x1073 = 16*x25
        x1074 = 16*x21
        x1075 = 16*x71
        x1076 = Px*x1075
        x1077 = 16*x23
        x1078 = Px*x1077
        x1079 = 16*x884
        x1080 = x1063 + x1066
        x1081 = -d_1 + x27
        x1082 = x1081 + x7
        x1083 = -x30
        x1084 = x1081 + x6
        x1085 = -x38
        x1086 = x1085 + x151
        x1087 = a_2 + x142 + x143 + x144
        x1088 = x146 + x147 + x82 + x85 + x86
        x1089 = -2*R_l_inv_50*a_0*x40 + x152 + x154 + x155 + x156
        x1090 = x1087 + x1088
        x1091 = x199 + x205 + x206 + x207 + x209 + x211 + x215 + x216 + x217 + x218 + x219 + x223 + x224 + x225 + x226 + x227 + x228 + x230 + x231 + x232 + x233 + x235 + x236 + x237 + x238 + x239 + x243 + x246 + x250 + x253 + x255 + x257 + x259 + x260 + x261 + x262 + x263
        x1092 = x1091 + x266 + x268
        x1093 = x203 + x379 + x380
        x1094 = x297 + x300 + x302 + x304 + x306 + x308 + x310 + x312 + x314 + x316 + x318 + x320 + x322 + x324 + x326 + x328 + x330 + x332 + x333 + x334 + x335 + x337 + x338 + x339 + x340 + x341 + x344 + x347 + x350 + x351 + x352 + x353 + x354 + x355 + x357 + x358 + x359
        x1095 = x198 + x201 + x204
        x1096 = x1091 + x275 + x276 + x375 + x376 + x377
        x1097 = x388 - x389 + x390 + x391 - x392 - x393
        x1098 = x303 + x309 + x321 + x323 + x325 + x327 + x503 + x504 + x505 + x508 + x511 + x512 + x513 + x514 + x515 + x519 + x521 + x522 + x523 + x524 + x525 + x526 + x527 + x528 + x529 + x531 + x532 + x534 + x536 + x538 + x540 + x541 + x542 + x543 + x544 + x545
        x1099 = x1098 + x559 + x617
        x1100 = -4*d_4*x36*x494 + x572
        x1101 = x1098 + x548 + x618
        x1102 = x631 + x758 + x761 + x766
        x1103 = x712 + x713 + x714 + x716 - x717 - x718 - x719 - x720
        x1104 = -x732 - x733
        x1105 = x652 + x654 + x655 + x723 + x724
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x0
        A[0, 2] = x0
        A[0, 3] = x2
        A[0, 4] = -x3
        A[0, 5] = x1
        A[0, 6] = r_23
        A[0, 8] = r_23
        A[1, 0] = x29 + x5
        A[1, 1] = x30
        A[1, 2] = x29 + x4
        A[1, 3] = x32
        A[1, 4] = -x33
        A[1, 5] = x31
        A[1, 6] = x34 + x5
        A[1, 7] = x30
        A[1, 8] = x34 + x4
        A[2, 0] = -x122 - x59 - x80 - x88
        A[2, 1] = 4*R_l_inv_55*a_0*d_4*x36 - x124 - x133 - x141
        A[2, 2] = x145 + x150 + x151 + x80
        A[2, 3] = x157 + x162 + x169
        A[2, 4] = -x171 - x173 + x179
        A[2, 5] = x157 + x180 + x181
        A[2, 6] = x145 + x184 + x186
        A[2, 7] = x126 - x128 - x130 + x131 + x189
        A[2, 8] = -x150 - x182 - x183 - x190 - x38 - x52
        A[3, 0] = x264 + x273 + x291
        A[3, 1] = x360 + x364 + x374
        A[3, 2] = -x290 - x378 - x381 - x382
        A[3, 3] = -x387 - x394 - x459
        A[3, 4] = -x462 - x463 + x467
        A[3, 5] = x394 + x468 + x469
        A[3, 6] = -x381 - x471 - x473
        A[3, 7] = -x360 - x474 - x475
        A[3, 8] = x264 + x378 + x470 + x472
        A[4, 0] = -x502 - x546 - x548 - x559
        A[4, 1] = x571 + x574 + x585
        A[4, 2] = x590 + x593 + x595
        A[4, 3] = -x596 - x597 + x602 + x610
        A[4, 4] = x611 - x612 - x613
        A[4, 5] = x598 + x599 - x601 + x615
        A[4, 6] = x301 - x303 + x305 - x309 + x313 + x315 + x317 + x319 - x321 - x323 - x325 - x327 + x502 - x503 + x507 - x511 + x518 + x520 - x522 - x523 - x524 - x525 - x526 - x527 - x528 - x529 - x531 - x532 - x534 - x536 + x537 + x539 - x541 - x542 - x543 - x544 - x545 + x616 + x618
        A[4, 7] = x574 + x619 + x620
        A[4, 8] = x595 + x622 + x623 + x624
        A[5, 0] = x650 + x660 + x700
        A[5, 1] = x705 + x711 + x721
        A[5, 2] = -x650 - x725 - x727
        A[5, 3] = x731 + x734 + x742
        A[5, 4] = R_l_inv_42*x72*x744 + 16*x638*x658*x72 + x755
        A[5, 5] = -x734 - x756 - x757
        A[5, 6] = -x759 - x760 - x762
        A[5, 7] = -x721 - x763 - x764
        A[5, 8] = -x760 - x766 - x767 - x768
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x769
        B[0, 2] = x769
        B[0, 3] = -x342
        B[0, 4] = -x749
        B[0, 5] = x342
        B[0, 6] = x244
        B[0, 8] = x244
        B[1, 0] = x771 + x775
        B[1, 1] = x776
        B[1, 2] = x770 + x775
        B[1, 3] = -x777
        B[1, 4] = -x778
        B[1, 5] = x777
        B[1, 6] = x771 + x779
        B[1, 7] = x776
        B[1, 8] = x770 + x779
        B[2, 0] = -x800 - x807
        B[2, 1] = 4*R_l_inv_56*a_0*d_3*r_11 - x808 - x809 - x813
        B[2, 2] = x800 + x814
        B[2, 3] = -x815 - x816 - x817 + x818 + x819 + x820 + x829
        B[2, 4] = x69*x835 + x830 - x831 + x832 - x833 + x834
        B[2, 5] = x815 + x816 + x817 - x818 - x819 - x820 + x829
        B[2, 6] = x807 + x836
        B[2, 7] = x69*x777 - x808 - x809 + x813
        B[2, 8] = -x814 - x836
        B[3, 0] = x860 + x871
        B[3, 1] = x214*x777 + x25*x778 + x749*x868 - x874 - x875 - x876 - x877 - x878 + x891
        B[3, 2] = -x871 - x892
        B[3, 3] = x893 - x894 + x895 + x896 - x899 - x900 - x911
        B[3, 4] = x466*(-r_11*x103 - r_11*x384 + r_11*x96 + r_13*x98 + x21*x244 + x864)
        B[3, 5] = -x893 + x894 - x895 - x896 + x899 + x900 - x911
        B[3, 6] = -x860 - x912
        B[3, 7] = 8*Py*d_3*r_12*r_23 + 8*Pz*d_3*r_12*r_33 + 4*d_3*r_11*x214 - x874 - x875 - x876 - x877 - x878 - x891
        B[3, 8] = x892 + x912
        B[4, 0] = -x954 - x964 - x966
        B[4, 1] = x967 - x968 - x969 - x971 - x972 - x973 + x974 + x975 + x981
        B[4, 2] = x964 + x985 + x986
        B[4, 3] = x989 + x990 + x991
        B[4, 4] = x835*(2*R_l_inv_26*a_0*d_1 + 2*R_l_inv_66*a_2*d_1 - x482 - x483 - x56)
        B[4, 5] = x988 - x990 + x991
        B[4, 6] = -x966 - x986 - x992
        B[4, 7] = -x967 + x968 + x969 + x971 + x972 + x973 - x974 - x975 + x981
        B[4, 8] = x954 + x985 + x992
        B[5, 0] = d_5*x828 - x1000 - x1001 - x1002 + x1018 + x1063 - x994 - x995 - x996 - x997 - x998 - x999
        B[5, 1] = R_l_inv_44*r_11*x744 + R_l_inv_47*x744*x797 - x1064 + x1065 + x989
        B[5, 2] = -x1018 - x1019 + x1020 + x1021 + x1022 - x1024 - x1025 - x1026 - x1027 - x1028 - x1029 - x1030 + x1031 + x1032 + x1033 + x1034 - x1036 - x1037 - x1038 - x1039 - x1040 - x1041 + x1042 + x1044 + x1045 + x1046 + x1048 + x1049 + x1051 + x1052 + x1053 + x1054 - x1055 - x1056 - x1057 + x1058 + x1059 + x1060 + x1061 + x1062 - x1066
        B[5, 3] = -x1067 - x1068 - x1069 - x1070 + x1071 + x466*x823 + x754*x826 + x898*x908
        B[5, 4] = 16*r_11*x853 + 16*r_12*x97 + 8*x104*x879 + x106*x749 + x107*x1076 + x107*x1078 - x1072*x793 - x1072*x794 + x1073*x792 + x1073*x793 - x1073*x795 + x1074*x822 - x1074*x824 + x1075*x846 + x1075*x950 + x1076*x109 + x1077*x846 + x1078*x109 + x1079*x750 + x1079*x886 + x110*x749 - x112*x749 + x113*x749 - x114*x749 - x117*x749 + x119*x749 - x120*x749 + 16*x248*x887 + 16*x35*x823 + 16*x39*x794 - 16*x692*x984 + x749*x90 - x749*x91 - x749*x92
        B[5, 5] = 8*Py*d_4*r_11*r_22 + 8*Pz*d_4*r_11*r_32 + 8*R_l_inv_46*a_0*d_1*x826 - x1067 - x1068 - x1069 - x1070 - x1071
        B[5, 6] = x1003 - x1004 - x1006 - x1008 - x1009 - x1011 - x1012 - x1013 - x1014 + x1015 + x1016 - x1017 - x1080
        B[5, 7] = 8*R_l_inv_44*a_0*d_1*r_11 + 8*R_l_inv_47*a_0*d_1*x797 - x1064 - x1065 - x988
        B[5, 8] = -x1003 + x1004 - x1005 - x1007 - x1010 + x1011 + x1012 + x1013 + x1014 - x1015 - x1016 + x1017 - x1080
        C = np.zeros(shape=(6, 9))
        C[0, 0] = r_23
        C[0, 2] = r_23
        C[0, 3] = x1
        C[0, 4] = x3
        C[0, 5] = x2
        C[0, 6] = x0
        C[0, 8] = x0
        C[1, 0] = -x1082 - x5
        C[1, 1] = x1083
        C[1, 2] = -x1082 - x4
        C[1, 3] = x31
        C[1, 4] = x33
        C[1, 5] = x32
        C[1, 6] = -x1084 - x5
        C[1, 7] = x1083
        C[1, 8] = -x1084 - x4
        C[2, 0] = x1086 + x1087 + x148 + x149 + x186
        C[2, 1] = -x126 + x128 + x130 + x132 + x189
        C[2, 2] = -x1086 - x1088 - x190 - x42 - x58
        C[2, 3] = -x1089 - x169 - x180
        C[2, 4] = -x171 - x173 - x179
        C[2, 5] = -x1089 - x162 - x181
        C[2, 6] = x1090 + x185 + x59 - x60 - x62 - x70 - x76
        C[2, 7] = x133 - x134 + x136 - x138 - x140 + x188
        C[2, 8] = x1085 + x1090 + x184 + x80
        C[3, 0] = -x1092 - x1093 - x291 - x470
        C[3, 1] = -x1094 - x374 - x474
        C[3, 2] = x1095 + x1096 + x290 + x471
        C[3, 3] = -x1097 - x387 - x469
        C[3, 4] = -x462 - x463 - x467
        C[3, 5] = x1097 + x459 + x468
        C[3, 6] = x1092 + x1095 + x382 + x473
        C[3, 7] = x1094 + x364 + x475
        C[3, 8] = -x1093 - x1096 - x273 - x472
        C[4, 0] = x1099 + x593 + x621 + x623
        C[4, 1] = -x1100 - x585 - x619
        C[4, 2] = -x1101 - x477 - x485 - x488 - x489 - x492 - x497 - x499 - x500 - x501 - x589 - x622
        C[4, 3] = x385*x478 - x600 - x601 - x615
        C[4, 4] = -x611 + x612 - x613
        C[4, 5] = x602 - x603 - x604 - x605 + x606 + x607 - x608 - x609 + x614
        C[4, 6] = x1099 + x481 + x495 + x590 + x624
        C[4, 7] = -x1100 - x571 - x620
        C[4, 8] = -x1101 - x479 - x490 - x493 - x591 - x592 - x624
        C[5, 0] = -x1102 - x700 - x725
        C[5, 1] = x1103 + x705 + x764
        C[5, 2] = x1102 + x660 + x727
        C[5, 3] = -x1104 - x742 - x756
        C[5, 4] = 8*R_l_inv_42*a_0*d_1*x72 + 16*R_l_inv_45*a_0*d_1*d_3*x72 - x755
        C[5, 5] = x1104 + x731 + x757
        C[5, 6] = x1105 + x762 + x765 + x767
        C[5, 7] = -x1103 - x711 - x763
        C[5, 8] = x1105 + x625 + x635 + x644 + x645 + x759 + x768
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[0] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor()
    # Finish code for explicit solution node 0

    # Code for non-branch dispatcher node 2
    # Actually, there is no code

    # Code for explicit solution node 3, solved variable is th_3
    def ExplicitSolutionNode_node_3_solve_th_3_processor():
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
            th_0 = this_solution[0]
            condition_0: bool = (((d_4 != 0)) and ((abs((Px*math.sin(th_0) - Py*math.cos(th_0) + d_1 - d_5*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_4) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = safe_acos((Px*x0 - Py*x1 + d_1 + d_5*(-r_13*x0 + r_23*x1))/d_4)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[5] = -x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
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
            th_3 = this_solution[5]
            condition_0: bool = (((2*abs(a_0*d_3) >= zero_tolerance)) or ((abs(2*a_0*a_2 - 2*a_0*d_4*math.sin(th_3)) >= zero_tolerance)) or ((abs(Px**2 - 2*Px*d_5*r_13 + Py**2 - 2*Py*d_5*r_23 + Pz**2 - 2*Pz*d_5*r_33 - a_0**2 - a_2**2 + 2*a_2*d_4*math.sin(th_3) - d_1**2 + 2*d_1*d_4*math.cos(th_3) - d_3**2 - d_4**2 + d_5**2*r_13**2 + d_5**2*r_23**2 + d_5**2*r_33**2) >= zero_tolerance)))
            if condition_0:
                x0 = 2*a_0
                x1 = d_4*math.sin(th_3)
                x2 = a_2*x0 - x0*x1
                x3 = math.atan2(-d_3*x0, x2)
                x4 = a_0**2
                x5 = d_3**2
                x6 = 2*d_5
                x7 = d_5**2
                x8 = Px**2 - Px*r_13*x6 + Py**2 - Py*r_23*x6 + Pz**2 - Pz*r_33*x6 - a_2**2 + 2*a_2*x1 - d_1**2 + 2*d_1*d_4*math.cos(th_3) - d_4**2 + r_13**2*x7 + r_23**2*x7 + r_33**2*x7 - x4 - x5
                x9 = safe_sqrt(x2**2 + 4*x4*x5 - x8**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x3 + math.atan2(x9, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                this_solution[4] = x3 + math.atan2(-x9, x8)
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
                add_input_index_to(19, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6

    # Code for explicit solution node 19, solved variable is th_1th_2th_4_soa
    def ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor():
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
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_33) >= zero_tolerance)) or ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= zero_tolerance)))
            if condition_0:
                this_solution[3] = math.atan2(r_13*math.cos(th_0) + r_23*math.sin(th_0), r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 19

    # Code for non-branch dispatcher node 20
    # Actually, there is no code

    # Code for explicit solution node 21, solved variable is th_5
    def ExplicitSolutionNode_node_21_solve_th_5_processor():
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
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance)) or ((abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[7] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_5_processor()
    # Finish code for explicit solution node 20

    # Code for solved_variable dispatcher node 22
    def SolvedVariableDispatcherNode_node_22_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[4]
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(32, node_input_i_idx_in_queue)

            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True

            if not taken_by_degenerate:
                add_input_index_to(23, node_input_i_idx_in_queue)
            elif not (degenerate_valid_0):
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    SolvedVariableDispatcherNode_node_22_processor()
    # Finish code for solved_variable dispatcher node 22

    # Code for explicit solution node 32, solved variable is th_1
    def ExplicitSolutionNode_node_32_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[32]
        this_input_valid: bool = node_input_validity[32]
        if not this_input_valid:
            return

        # The solution of non-root node 32
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance)) or ((abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = safe_sqrt(x4**2 + x5**2 - 0.995450272904637*(-a_2 + 0.067605633802817*d_3)**2)
                x8 = -0.997722543047233*a_2 + 0.0674516648820665*d_3
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x6 + math.atan2(x7, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(33, appended_idx)
                this_solution[1] = x6 + math.atan2(-x7, x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(33, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_1_processor()
    # Finish code for explicit solution node 32

    # Code for non-branch dispatcher node 33
    # Actually, there is no code

    # Code for explicit solution node 34, solved variable is th_4
    def ExplicitSolutionNode_node_34_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[33]
        this_input_valid: bool = node_input_validity[33]
        if not this_input_valid:
            return

        # The solution of non-root node 34
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(-r_13*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_0) - r_23*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.sin(th_0) + r_33*(0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))) >= zero_tolerance)) or ((abs(-r_13*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.0674516648820665*math.sin(th_1) - 0.997722543047233*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.0674516648820665*x1 - 0.997722543047233*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.997722543047233*x1 + 0.0674516648820665*x2
                x8 = -x0*x7
                x9 = -x3
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_34_solve_th_4_processor()
    # Finish code for explicit solution node 33

    # Code for explicit solution node 23, solved variable is th_1
    def ExplicitSolutionNode_node_23_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_1th_2th_4_soa = this_solution[3]
            th_2 = this_solution[4]
            condition_0: bool = (((abs(a_2*math.sin(th_2) + d_3*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 + a_2*math.cos(th_2) - d_3*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_0 + a_2*x1 - d_3*x2
                x4 = -a_2*x2 - d_3*x1
                x5 = Px*math.cos(th_0) + Py*math.sin(th_0) - d_5*math.sin(th_1th_2th_4_soa)
                this_solution[1] = math.atan2(-x0*x3 + x4*x5, x0*x4 + x3*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor()
    # Finish code for explicit solution node 23

    # Code for non-branch dispatcher node 24
    # Actually, there is no code

    # Code for explicit solution node 25, solved variable is th_4
    def ExplicitSolutionNode_node_25_solve_th_4_processor():
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
    ExplicitSolutionNode_node_25_solve_th_4_processor()
    # Finish code for explicit solution node 24

    # Code for explicit solution node 14, solved variable is th_5
    def ExplicitSolutionNode_node_14_solve_th_5_processor():
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
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance)) or ((abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[7] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor()
    # Finish code for explicit solution node 14

    # Code for solved_variable dispatcher node 15
    def SolvedVariableDispatcherNode_node_15_processor():
        this_node_input_index: List[int] = node_input_index[15]
        this_input_valid: bool = node_input_validity[15]
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
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(26, branch_input_index)

            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(29, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(16, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor()
    # Finish code for solved_variable dispatcher node 15

    # Code for explicit solution node 29, solved variable is th_1
    def ExplicitSolutionNode_node_29_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[29]
        this_input_valid: bool = node_input_validity[29]
        if not this_input_valid:
            return

        # The solution of non-root node 29
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance)) or ((abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = safe_sqrt(x4**2 + x5**2 - 0.995450272904637*(a_2 - 0.067605633802817*d_3)**2)
                x8 = 0.997722543047233*a_2 - 0.0674516648820665*d_3
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x6 + math.atan2(x7, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(30, appended_idx)
                this_solution[1] = x6 + math.atan2(-x7, x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(30, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_1_processor()
    # Finish code for explicit solution node 29

    # Code for non-branch dispatcher node 30
    # Actually, there is no code

    # Code for explicit solution node 31, solved variable is th_4
    def ExplicitSolutionNode_node_31_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return

        # The solution of non-root node 31
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(-r_13*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_0) - r_23*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.sin(th_0) + r_33*(0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))) >= zero_tolerance)) or ((abs(r_13*((0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) + r_23*((0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.0674516648820665*math.sin(th_1) - 0.997722543047233*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = -0.0674516648820665*x1 + 0.997722543047233*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.997722543047233*x1 + 0.0674516648820665*x2
                x8 = x0*x7
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, -r_13*x3*x6 - r_23*x3*x4 + r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_4_processor()
    # Finish code for explicit solution node 30

    # Code for explicit solution node 26, solved variable is th_1
    def ExplicitSolutionNode_node_26_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[26]
        this_input_valid: bool = node_input_validity[26]
        if not this_input_valid:
            return

        # The solution of non-root node 26
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (((abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance)) or ((abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = safe_sqrt(x4**2 + x5**2 - 0.995450272904637*(-a_2 + 0.067605633802817*d_3)**2)
                x8 = -0.997722543047233*a_2 + 0.0674516648820665*d_3
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x6 + math.atan2(x7, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(27, appended_idx)
                this_solution[1] = x6 + math.atan2(-x7, x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(27, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_1_processor()
    # Finish code for explicit solution node 26

    # Code for non-branch dispatcher node 27
    # Actually, there is no code

    # Code for explicit solution node 28, solved variable is th_4
    def ExplicitSolutionNode_node_28_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[27]
        this_input_valid: bool = node_input_validity[27]
        if not this_input_valid:
            return

        # The solution of non-root node 28
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(-r_13*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_0) - r_23*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.sin(th_0) + r_33*(0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))) >= zero_tolerance)) or ((abs(-r_13*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.0674516648820665*math.sin(th_1) - 0.997722543047233*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.0674516648820665*x1 - 0.997722543047233*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.997722543047233*x1 + 0.0674516648820665*x2
                x8 = -x0*x7
                x9 = -x3
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_4_processor()
    # Finish code for explicit solution node 27

    # Code for explicit solution node 16, solved variable is th_1
    def ExplicitSolutionNode_node_16_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[16]
        this_input_valid: bool = node_input_validity[16]
        if not this_input_valid:
            return

        # The solution of non-root node 16
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_2 = this_solution[4]
            condition_0: bool = (((abs(a_2*math.sin(th_2) + d_3*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 + a_2*math.cos(th_2) - d_3*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_0 + a_2*x1 - d_3*x2
                x4 = -a_2*x2 - d_3*x1
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6
                this_solution[1] = math.atan2(-x0*x3 + x4*x7, x0*x4 + x3*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1_processor()
    # Finish code for explicit solution node 16

    # Code for non-branch dispatcher node 17
    # Actually, there is no code

    # Code for explicit solution node 18, solved variable is th_4
    def ExplicitSolutionNode_node_18_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[17]
        this_input_valid: bool = node_input_validity[17]
        if not this_input_valid:
            return

        # The solution of non-root node 18
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
    ExplicitSolutionNode_node_18_solve_th_4_processor()
    # Finish code for explicit solution node 17

    # Code for explicit solution node 7, solved variable is th_4
    def ExplicitSolutionNode_node_7_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_3 = this_solution[5]
            condition_0: bool = (((math.sin(th_3) != 0)) and ((abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                this_solution[6] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_4_processor()
    # Finish code for explicit solution node 7

    # Code for equation all-zero dispatcher node 8
    def EquationAllZeroDispatcherNode_node_8_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            checked_result: bool = (((abs(Pz) <= 1.0e-6)) and ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8

    # Code for explicit solution node 9, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor():
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
            th_2 = this_solution[4]
            th_3 = this_solution[5]
            th_4 = this_solution[6]
            condition_0: bool = (((abs(Pz) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)))
            if condition_0:
                x0 = -a_0*math.cos(th_2) - a_2 + d_4*math.sin(th_3) + d_5*math.sin(th_4)*math.cos(th_3)
                x1 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x2 = a_0*math.sin(th_2) - d_3 + d_5*math.cos(th_4)
                this_solution[2] = math.atan2(Pz*x0 - x1*x2, Pz*x2 + x0*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 9

    # Code for non-branch dispatcher node 10
    # Actually, there is no code

    # Code for explicit solution node 11, solved variable is th_1
    def ExplicitSolutionNode_node_11_solve_th_1_processor():
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
            th_1th_2_soa = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[1] = th_1th_2_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 10

    # Code for non-branch dispatcher node 12
    # Actually, there is no code

    # Code for explicit solution node 13, solved variable is th_5
    def ExplicitSolutionNode_node_13_solve_th_5_processor():
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
            th_1th_2_soa = this_solution[2]
            th_3 = this_solution[5]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*(math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)*math.cos(th_1th_2_soa)) - r_21*(math.sin(th_0)*math.sin(th_3)*math.cos(th_1th_2_soa) + math.cos(th_0)*math.cos(th_3)) + r_31*math.sin(th_1th_2_soa)*math.sin(th_3)) >= zero_tolerance)) or ((abs(r_12*(math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)*math.cos(th_1th_2_soa)) - r_22*(math.sin(th_0)*math.sin(th_3)*math.cos(th_1th_2_soa) + math.cos(th_0)*math.cos(th_3)) + r_32*math.sin(th_1th_2_soa)*math.sin(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_3)
                x1 = x0*math.sin(th_1th_2_soa)
                x2 = math.sin(th_0)
                x3 = math.cos(th_3)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1th_2_soa)
                x6 = x2*x3 - x4*x5
                x7 = x2*x5 + x3*x4
                this_solution[7] = math.atan2(-r_11*x6 + r_21*x7 - r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_5_processor()
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


def dense_cvr_038_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = dense_cvr_038_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw
        ee_pose_i = dense_cvr_038_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                _yaik_seed_candidates.strict_witness = True
                _yaik_seed_candidates.clear()
            elif not _yaik_seed_candidates.strict_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -1.5707963267948966, -3.141592653589793, -6.283185307179586, -6.283185307179586, -3.141592653589793], [3.141592653589793, 4.71238898038469, 3.141592653589793, 0.0, 0.0, 3.141592653589793], [True, True, True, True, True, True], [False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = dense_cvr_038_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
                if _yaik_seed_candidates is not None and np.max(np.abs(snapped_pose-T_ee)) < 1e-7:
                    _yaik_seed_candidates.strict_witness = True
                    _yaik_seed_candidates.clear()
    return ik_output


def dense_cvr_038_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = RefinementSeeds()
    for transform in (dense_cvr_038_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in dense_cvr_038_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: dense_cvr_038_fk(q)
    if refinement_seeds and not refinement_seeds.strict_witness:
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                refinement_seeds.strict_witness = True
                break
    if not refinement_seeds.strict_witness:
        refinement_seeds.clear()
        seed_parameters = ()
        for shifted in nearby_ik_targets(T_ee):
            for transform in (dense_cvr_038_ik_target_original_to_raw,):
                raw_target = transform(shifted)
                for raw_candidate in dense_cvr_038_ik_solve_raw(raw_target, *seed_parameters):
                    seed = raw_candidate.reshape(-1).copy()
                    seed[0] = seed[0] - th_0_offset_original2raw
                    seed[1] = add_revolute_offset(seed[1], -th_1_offset_original2raw)
                    seed[2] = seed[2] - th_2_offset_original2raw
                    seed[3] = add_revolute_offset(seed[3], -th_3_offset_original2raw)
                    seed[4] = add_revolute_offset(seed[4], -th_4_offset_original2raw)
                    seed[5] = seed[5] - th_5_offset_original2raw
                    retain_ik_refinement_seed(refinement_seeds, refinement_fk, T_ee, seed)
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                refinement_seeds.strict_witness = True
                break
    return result


def test_ik_solve_dense_cvr_038():
    theta_in = np.random.random(size=(6, ))
    ee_pose = dense_cvr_038_fk(theta_in)
    ik_output = dense_cvr_038_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = dense_cvr_038_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_dense_cvr_038()
