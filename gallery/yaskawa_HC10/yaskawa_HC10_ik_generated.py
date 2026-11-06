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
a_0: float = 0.7
d_1: float = -0.5
d_2: float = -0.162
d_3: float = -0.13
pre_transform_special_symbol_23: float = 0.275

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def yaskawa_HC10_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = -1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_23
    ee_transformed[1, 1] = 1.0*r_22
    ee_transformed[1, 2] = -1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = 1.0*r_33
    ee_transformed[2, 1] = 1.0*r_32
    ee_transformed[2, 2] = -1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def yaskawa_HC10_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -1.0*r_13
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = 1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = -1.0*r_23
    ee_transformed[1, 1] = 1.0*r_22
    ee_transformed[1, 2] = 1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = -1.0*r_33
    ee_transformed[2, 1] = 1.0*r_32
    ee_transformed[2, 2] = 1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def yaskawa_HC10_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.cos(th_4)
    x1 = math.cos(th_0)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x5*x6
    x8 = x1*x4 - x1*x7
    x9 = x0*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_0)
    x12 = math.sin(th_3)
    x13 = math.cos(th_3)
    x14 = x2*x5
    x15 = x3*x6
    x16 = x1*x14 + x1*x15
    x17 = -x11*x12 + x13*x16
    x18 = x10*x17
    x19 = math.cos(th_5)
    x20 = -1.0*x11*x13 - 1.0*x12*x16
    x21 = math.sin(th_5)
    x22 = 1.0*x0*x17 - 1.0*x10*x8
    x23 = 1.0*a_0
    x24 = x23*x6
    x25 = 1.0*d_1
    x26 = 1.0*d_3
    x27 = x11*x4 - x11*x7
    x28 = x0*x27
    x29 = x11*x14 + x11*x15
    x30 = x1*x12 + x13*x29
    x31 = x10*x30
    x32 = 1.0*x1*x13 - 1.0*x12*x29
    x33 = 1.0*x0*x30 - 1.0*x10*x27
    x34 = x14 + x15
    x35 = x0*x34
    x36 = -x4 + x5*x6
    x37 = x10*x13*x36
    x38 = 1.0*x0*x13*x36 - 1.0*x10*x34
    x39 = 1.0*x12*x36
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = 1.0*x18 + 1.0*x9
    ee_pose[0, 1] = -x19*x20 - x21*x22
    ee_pose[0, 2] = x19*x22 - x20*x21
    ee_pose[0, 3] = d_2*x20 + x1*x24 + x25*x8 + x26*(-x18 - x9)
    ee_pose[1, 0] = 1.0*x28 + 1.0*x31
    ee_pose[1, 1] = -x19*x32 - x21*x33
    ee_pose[1, 2] = x19*x33 - x21*x32
    ee_pose[1, 3] = d_2*x32 + x11*x24 + x25*x27 + x26*(-x28 - x31)
    ee_pose[2, 0] = 1.0*x35 + 1.0*x37
    ee_pose[2, 1] = 1.0*x12*x19*x36 - x21*x38
    ee_pose[2, 2] = x19*x38 + x21*x39
    ee_pose[2, 3] = -d_2*x39 + 1.0*pre_transform_special_symbol_23 - x2*x23 + x25*x34 + x26*(-x35 - x37)
    return ee_pose


def yaskawa_HC10_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = math.cos(th_0)
    x5 = 1.0*x4
    x6 = x3*x5
    x7 = math.cos(th_1)
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = x2*x6 - x7*x9
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = x2*x9 + x6*x7
    x14 = -x1*x11 - x12*x13
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x10*x15 - x16*(-x1*x12 + x11*x13)
    x18 = x1*x3
    x19 = x1*x8
    x20 = x18*x2 - x19*x7
    x21 = x18*x7 + x19*x2
    x22 = 1.0*x11*x4 - x12*x21
    x23 = -x15*x20 - x16*(x11*x21 + x12*x5)
    x24 = 1.0*x2
    x25 = 1.0*x7
    x26 = x24*x8 + x25*x3
    x27 = -x24*x3 + 1.0*x7*x8
    x28 = x12*x27
    x29 = -x11*x16*x27 - x15*x26
    x30 = a_0*x24
    x31 = -pre_transform_special_symbol_23 + x30
    x32 = -x31
    x33 = a_0*x7
    x34 = d_1*x20 + x1*x33
    x35 = d_1*x26 + pre_transform_special_symbol_23 - x30
    x36 = d_2*x28
    x37 = d_1*x26 - x31 - x36
    x38 = d_2*x22 + x34
    x39 = d_3*x29 + x35 - x36
    x40 = d_3*x23 + x38
    x41 = d_1*x10 + x33*x5
    x42 = d_2*x14 + x41
    x43 = d_3*x17 + x42
    x44 = a_0*x25
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = -x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = x17
    jacobian_output[1, 1] = x5
    jacobian_output[1, 2] = -x5
    jacobian_output[1, 3] = x20
    jacobian_output[1, 4] = x22
    jacobian_output[1, 5] = x23
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x26
    jacobian_output[2, 4] = -x28
    jacobian_output[2, 5] = x29
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x5
    jacobian_output[3, 2] = x32*x5
    jacobian_output[3, 3] = -x20*x35 + x26*x34
    jacobian_output[3, 4] = -x22*x37 - x28*x38
    jacobian_output[3, 5] = -x23*x39 + x29*x40
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = x1*x32
    jacobian_output[4, 3] = x10*x35 - x26*x41
    jacobian_output[4, 4] = x14*x37 + x28*x42
    jacobian_output[4, 5] = x17*x39 - x29*x43
    jacobian_output[5, 2] = -x0**2*x44 - x4**2*x44
    jacobian_output[5, 3] = -x10*x34 + x20*x41
    jacobian_output[5, 4] = -x14*x38 + x22*x42
    jacobian_output[5, 5] = -x17*x40 + x23*x43
    return jacobian_output


def yaskawa_HC10_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_2)
    x3 = math.cos(th_0)
    x4 = 1.0*x3
    x5 = x2*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x4*x7
    x9 = x1*x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x1*x8 + x5*x6
    x13 = math.cos(th_4)
    x14 = math.sin(th_4)
    x15 = x0*x2
    x16 = x0*x7
    x17 = x1*x15 - x16*x6
    x18 = x1*x16 + x15*x6
    x19 = 1.0*x1
    x20 = x19*x7 + 1.0*x2*x6
    x21 = -x19*x2 + 1.0*x6*x7
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = -x0*x10 - x11*x12
    jacobian_output[0, 5] = -x13*x9 - x14*(-x0*x11 + x10*x12)
    jacobian_output[1, 1] = x4
    jacobian_output[1, 2] = -x4
    jacobian_output[1, 3] = x17
    jacobian_output[1, 4] = 1.0*x10*x3 - x11*x18
    jacobian_output[1, 5] = -x13*x17 - x14*(x10*x18 + x11*x4)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x20
    jacobian_output[2, 4] = -x11*x21
    jacobian_output[2, 5] = -x10*x14*x21 - x13*x20
    return jacobian_output


def yaskawa_HC10_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
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
    x7 = -pre_transform_special_symbol_23 + x6
    x8 = -x7
    x9 = math.sin(th_2)
    x10 = x5*x9
    x11 = math.cos(th_2)
    x12 = math.cos(th_1)
    x13 = 1.0*x12
    x14 = x11*x13
    x15 = x10 + x14
    x16 = math.sin(th_0)
    x17 = x11*x5
    x18 = -x13*x16*x9 + x16*x17
    x19 = d_1*x15 + pre_transform_special_symbol_23 - x6
    x20 = a_0*x13
    x21 = d_1*x18 + x16*x20
    x22 = math.sin(th_3)
    x23 = 1.0*x12*x9 - x17
    x24 = x22*x23
    x25 = math.cos(th_3)
    x26 = x10*x16 + x14*x16
    x27 = 1.0*x1*x25 - x22*x26
    x28 = d_2*x24
    x29 = d_1*x15 - x28 - x7
    x30 = d_2*x27 + x21
    x31 = math.cos(th_4)
    x32 = math.sin(th_4)
    x33 = -x15*x31 - x23*x25*x32
    x34 = -x18*x31 - x32*(x2*x22 + x25*x26)
    x35 = d_3*x33 + x19 - x28
    x36 = d_3*x34 + x30
    x37 = 1.0*p_on_ee_x
    x38 = 1.0*x16
    x39 = p_on_ee_z*x38
    x40 = x11*x2
    x41 = x12*x2
    x42 = x4*x40 - x41*x9
    x43 = a_0*x41 + d_1*x42
    x44 = x12*x40 + x2*x4*x9
    x45 = -x22*x44 - x25*x38
    x46 = d_2*x45 + x43
    x47 = -x31*x42 - x32*(-x22*x38 + x25*x44)
    x48 = d_3*x47 + x46
    x49 = x1*x37
    x50 = x0*x16
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = 1.0*x1*x8 - x3
    jacobian_output[0, 3] = -p_on_ee_y*x15 + p_on_ee_z*x18 + x15*x21 - x18*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x27 - x24*x30 - x27*x29
    jacobian_output[0, 5] = -p_on_ee_y*x33 + p_on_ee_z*x34 + x33*x36 - x34*x35
    jacobian_output[1, 0] = x37
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x38 + x39
    jacobian_output[1, 2] = 1.0*x16*x8 - x39
    jacobian_output[1, 3] = p_on_ee_x*x15 - p_on_ee_z*x42 - x15*x43 + x19*x42
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x45 + x22*x23*x46 + x29*x45
    jacobian_output[1, 5] = p_on_ee_x*x33 - p_on_ee_z*x47 - x33*x48 + x35*x47
    jacobian_output[2, 1] = -x49 - x50
    jacobian_output[2, 2] = -x1**2*x20 - x16**2*x20 + x49 + x50
    jacobian_output[2, 3] = -p_on_ee_x*x18 + p_on_ee_y*x42 + x18*x43 - x21*x42
    jacobian_output[2, 4] = -p_on_ee_x*x27 + p_on_ee_y*x45 + x27*x46 - x30*x45
    jacobian_output[2, 5] = -p_on_ee_x*x34 + p_on_ee_y*x47 + x34*x48 - x36*x47
    return jacobian_output


def yaskawa_HC10_ik_solve_raw(T_ee: np.ndarray):
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
        return IkSolution([math.nan] * 6)

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

    # Code for explicit solution node 1, solved variable is th_2
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 0] = d_2*r_21
        R_l[0, 1] = d_2*r_22
        R_l[0, 2] = d_2*r_11
        R_l[0, 3] = d_2*r_12
        R_l[0, 4] = Py - d_3*r_23
        R_l[0, 5] = Px - d_3*r_13
        R_l[1, 0] = d_2*r_11
        R_l[1, 1] = d_2*r_12
        R_l[1, 2] = -d_2*r_21
        R_l[1, 3] = -d_2*r_22
        R_l[1, 4] = Px - d_3*r_13
        R_l[1, 5] = -Py + d_3*r_23
        R_l[2, 6] = d_2*r_31
        R_l[2, 7] = d_2*r_32
        R_l[3, 0] = -r_21
        R_l[3, 1] = -r_22
        R_l[3, 2] = -r_11
        R_l[3, 3] = -r_12
        R_l[4, 0] = -r_11
        R_l[4, 1] = -r_12
        R_l[4, 2] = r_21
        R_l[4, 3] = r_22
        R_l[5, 6] = 2*Px*d_2*r_11 + 2*Py*d_2*r_21 + 2*Pz*d_2*r_31 - 2*d_2*d_3*r_11*r_13 - 2*d_2*d_3*r_21*r_23 - 2*d_2*d_3*r_31*r_33
        R_l[5, 7] = 2*Px*d_2*r_12 + 2*Py*d_2*r_22 + 2*Pz*d_2*r_32 - 2*d_2*d_3*r_12*r_13 - 2*d_2*d_3*r_22*r_23 - 2*d_2*d_3*r_32*r_33
        R_l[6, 0] = -Px*r_31 + Pz*r_11 - d_3*r_11*r_33 + d_3*r_13*r_31
        R_l[6, 1] = -Px*r_32 + Pz*r_12 - d_3*r_12*r_33 + d_3*r_13*r_32
        R_l[6, 2] = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31
        R_l[6, 3] = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32
        R_l[7, 0] = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31
        R_l[7, 1] = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32
        R_l[7, 2] = Px*r_31 - Pz*r_11 + d_3*r_11*r_33 - d_3*r_13*r_31
        R_l[7, 3] = Px*r_32 - Pz*r_12 + d_3*r_12*r_33 - d_3*r_13*r_32
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
        x0 = R_l_inv_60*r_31 + R_l_inv_70*r_32
        x1 = a_0*x0
        x2 = R_l_inv_66*r_31
        x3 = R_l_inv_76*r_32
        x4 = x2 + x3
        x5 = d_1*x4
        x6 = R_l_inv_62*r_31 + R_l_inv_72*r_32
        x7 = d_1*x6
        x8 = -x7
        x9 = x1 + x5 + x8
        x10 = R_l_inv_64*r_31
        x11 = R_l_inv_74*r_32
        x12 = d_3*r_33
        x13 = Pz - x12
        x14 = x13*x6
        x15 = R_l_inv_65*r_31 + R_l_inv_75*r_32
        x16 = Px**2
        x17 = Py**2
        x18 = Pz**2
        x19 = d_2**2
        x20 = r_11**2
        x21 = x19*x20
        x22 = r_21**2
        x23 = x19*x22
        x24 = r_31**2
        x25 = x19*x24
        x26 = d_3**2
        x27 = r_13**2*x26
        x28 = r_23**2*x26
        x29 = r_33**2*x26
        x30 = d_3*r_13
        x31 = 2*Px
        x32 = x30*x31
        x33 = d_3*r_23
        x34 = 2*Py
        x35 = x33*x34
        x36 = 2*Pz
        x37 = x12*x36
        x38 = a_0**2
        x39 = d_1**2
        x40 = -x38 - x39
        x41 = x16 + x17 + x18 + x21 + x23 + x25 + x27 + x28 + x29 - x32 - x35 - x37 + x40
        x42 = x15*x41
        x43 = x14 + x42
        x44 = -x10 - x11 + x43
        x45 = 2*a_0
        x46 = x4*x45
        x47 = x45*x6
        x48 = 2*d_1
        x49 = x0*x48
        x50 = x47 + x49
        x51 = x10 + x11
        x52 = -x14 - x42 + x51
        x53 = R_l_inv_67*r_31 + R_l_inv_77*r_32
        x54 = x48*x53
        x55 = 2*R_l_inv_63
        x56 = 2*R_l_inv_73
        x57 = r_31*x55 + r_32*x56
        x58 = -x54 - x57
        x59 = -x54 + x57
        x60 = -x5
        x61 = x1 + x60
        x62 = x43 + x51
        x63 = x62 + x8
        x64 = -x1
        x65 = x5 + x62 + x7
        x66 = Px*r_11
        x67 = Py*r_21
        x68 = Pz*r_31
        x69 = r_11*x30
        x70 = r_21*x33
        x71 = r_31*x12
        x72 = x66 + x67 + x68 - x69 - x70 - x71
        x73 = Px*r_12
        x74 = Py*r_22
        x75 = Pz*r_32
        x76 = r_12*x30
        x77 = r_22*x33
        x78 = r_32*x12
        x79 = x73 + x74 + x75 - x76 - x77 - x78
        x80 = R_l_inv_62*x72 + R_l_inv_72*x79
        x81 = d_1*x80
        x82 = R_l_inv_60*x72 + R_l_inv_70*x79
        x83 = a_0*x82
        x84 = -x83
        x85 = x81 + x84
        x86 = R_l_inv_64*x72
        x87 = R_l_inv_74*x79
        x88 = R_l_inv_66*x72
        x89 = R_l_inv_76*x79
        x90 = x88 + x89
        x91 = d_1*x90
        x92 = x86 + x87 - x91
        x93 = d_2*x20
        x94 = d_2*x22
        x95 = d_2*x24
        x96 = x13*x80
        x97 = R_l_inv_65*x72 + R_l_inv_75*x79
        x98 = x41*x97
        x99 = x93 + x94 + x95 - x96 - x98
        x100 = x92 + x99
        x101 = x45*x90
        x102 = x45*x80
        x103 = x48*x82
        x104 = x102 + x103
        x105 = -x81
        x106 = x105 + x83
        x107 = x86 + x87 + x91
        x108 = x107 + x99
        x109 = R_l_inv_67*x72 + R_l_inv_77*x79
        x110 = x109*x48
        x111 = -x45
        x112 = x55*x72 + x56*x79
        x113 = x111 + x112
        x114 = -x110
        x115 = x112 + x45
        x116 = -d_2*x20 - d_2*x22 - d_2*x24 + x96 + x98
        x117 = x107 + x116
        x118 = Px*r_21
        x119 = Py*r_11
        x120 = r_11*x33 - r_21*x30 + x118 - x119
        x121 = R_l_inv_64*x120
        x122 = Px*r_22
        x123 = Py*r_12
        x124 = r_12*x33 - r_22*x30 + x122 - x123
        x125 = R_l_inv_74*x124
        x126 = R_l_inv_62*x120 + R_l_inv_72*x124
        x127 = R_l_inv_65*x120 + R_l_inv_75*x124
        x128 = x121 + x125 - x126*x13 - x127*x41
        x129 = R_l_inv_66*x120 + R_l_inv_76*x124
        x130 = d_1*x129
        x131 = -x130
        x132 = a_0 + x131
        x133 = d_1*x126
        x134 = R_l_inv_60*x120 + R_l_inv_70*x124
        x135 = a_0*x134
        x136 = -x135
        x137 = x133 + x136
        x138 = -x48
        x139 = x129*x45
        x140 = -x139
        x141 = x126*x45
        x142 = x134*x48
        x143 = x141 + x142
        x144 = -x133
        x145 = x135 + x144
        x146 = -a_0
        x147 = x130 + x146
        x148 = R_l_inv_67*x120 + R_l_inv_77*x124
        x149 = x120*x55 + x124*x56
        x150 = x148*x48 + x149
        x151 = 2*d_1*x148 - x149
        x152 = x121 + x125 + x126*x13 + x127*x41
        x153 = Py*x20 + Py*x22 + Py*x24 - x20*x33 - x22*x33 - x24*x33
        x154 = 2*d_2
        x155 = x153*x154
        x156 = Px*x20 + Px*x22 + Px*x24 - x20*x30 - x22*x30 - x24*x30
        x157 = x154*x156
        x158 = 2*x33
        x159 = 2*x30
        x160 = 2*r_11
        x161 = r_23*x26
        x162 = r_13*x161
        x163 = 2*r_31
        x164 = r_33*x161
        x165 = r_21**3*x19 - r_21*x16 + r_21*x17 - r_21*x18 + r_21*x21 + r_21*x25 - r_21*x27 + r_21*x28 - r_21*x29 + r_21*x37 + x118*x159 - x119*x159 - x158*x66 - x158*x67 - x158*x68 + x160*x162 + x163*x164 + x34*x66 + x34*x68 - x34*x71
        x166 = 2*x162
        x167 = r_13*r_33*x26
        x168 = r_11**3*x19 + r_11*x16 - r_11*x17 - r_11*x18 + r_11*x23 + r_11*x25 + r_11*x27 - r_11*x28 - r_11*x29 + r_11*x37 + r_21*x166 - x118*x158 + x119*x158 - x159*x66 - x159*x67 - x159*x68 + x163*x167 + x31*x67 + x31*x68 - x31*x71
        x169 = r_21*x19
        x170 = x160*x169
        x171 = x163*x169
        x172 = 2*r_32
        x173 = r_12*x166 + r_12*x170 - r_22*x16 + r_22*x17 - r_22*x18 + r_22*x21 + 3*r_22*x23 + r_22*x25 - r_22*x27 + r_22*x28 - r_22*x29 + r_22*x37 + r_32*x171 + x122*x159 - x123*x159 - x158*x73 - x158*x74 - x158*x75 + x164*x172 + x34*x73 + x34*x75 - x34*x78
        x174 = r_31*x160*x19
        x175 = r_12*x16 - r_12*x17 - r_12*x18 + 3*r_12*x21 + r_12*x23 + r_12*x25 + r_12*x27 - r_12*x28 - r_12*x29 + r_12*x37 + r_22*x166 + r_22*x170 + r_32*x174 - x122*x158 + x123*x158 - x159*x73 - x159*x74 - x159*x75 + x167*x172 + x31*x74 + x31*x75 - x31*x78
        x176 = R_l_inv_02*x165 + R_l_inv_12*x173 + R_l_inv_22*x168 + R_l_inv_32*x175 + R_l_inv_42*x155 + R_l_inv_52*x157
        x177 = d_1*x176
        x178 = R_l_inv_00*x165 + R_l_inv_10*x173 + R_l_inv_20*x168 + R_l_inv_30*x175 + R_l_inv_40*x155 + R_l_inv_50*x157
        x179 = a_0*x178
        x180 = -x179
        x181 = x177 + x180
        x182 = R_l_inv_04*x165
        x183 = R_l_inv_14*x173
        x184 = R_l_inv_24*x168
        x185 = R_l_inv_34*x175
        x186 = R_l_inv_06*x165 + R_l_inv_16*x173 + R_l_inv_26*x168 + R_l_inv_36*x175 + R_l_inv_46*x155 + R_l_inv_56*x157
        x187 = d_1*x186
        x188 = R_l_inv_44*x155
        x189 = R_l_inv_54*x157
        x190 = x182 + x183 + x184 + x185 - x187 + x188 + x189
        x191 = R_l_inv_05*x165 + R_l_inv_15*x173 + R_l_inv_25*x168 + R_l_inv_35*x175 + R_l_inv_45*x155 + R_l_inv_55*x157
        x192 = -x13*x176 - x191*x41
        x193 = x190 + x192
        x194 = x186*x45
        x195 = x176*x45
        x196 = x178*x48
        x197 = x195 + x196
        x198 = -x177
        x199 = x179 + x198
        x200 = x182 + x183 + x184 + x185 + x187 + x188 + x189
        x201 = x192 + x200
        x202 = R_l_inv_07*x165 + R_l_inv_17*x173 + R_l_inv_27*x168 + R_l_inv_37*x175 + R_l_inv_47*x155 + R_l_inv_57*x157
        x203 = 2*x165
        x204 = 2*x173
        x205 = 2*x168
        x206 = 2*x175
        x207 = 4*d_2
        x208 = x153*x207
        x209 = x156*x207
        x210 = R_l_inv_03*x203 + R_l_inv_13*x204 + R_l_inv_23*x205 + R_l_inv_33*x206 + R_l_inv_43*x208 + R_l_inv_53*x209 + 2*x38 - 2*x39
        x211 = x202*x48 + x210
        x212 = a_0*d_1
        x213 = 8*x212
        x214 = 2*d_1*x202 - x210
        x215 = x13*x176 + x191*x41
        x216 = x190 + x215
        x217 = x200 + x215
        x218 = R_l_inv_40*x157
        x219 = R_l_inv_50*x155
        x220 = R_l_inv_00*x168
        x221 = R_l_inv_20*x165
        x222 = R_l_inv_10*x175
        x223 = R_l_inv_30*x173
        x224 = x218 - x219 + x220 - x221 + x222 - x223
        x225 = a_0*x224
        x226 = R_l_inv_46*x157
        x227 = R_l_inv_56*x155
        x228 = R_l_inv_06*x168
        x229 = R_l_inv_26*x165
        x230 = R_l_inv_16*x175
        x231 = R_l_inv_36*x173
        x232 = x226 - x227 + x228 - x229 + x230 - x231
        x233 = d_1*x232
        x234 = R_l_inv_02*x168 + R_l_inv_12*x175 - R_l_inv_22*x165 - R_l_inv_32*x173 + R_l_inv_42*x157 - R_l_inv_52*x155
        x235 = x13*x234
        x236 = R_l_inv_05*x168 + R_l_inv_15*x175 - R_l_inv_25*x165 - R_l_inv_35*x173 + R_l_inv_45*x157 - R_l_inv_55*x155
        x237 = x236*x41
        x238 = x233 + x235 + x237
        x239 = x225 + x238
        x240 = d_1*x234
        x241 = R_l_inv_24*x165
        x242 = R_l_inv_34*x173
        x243 = R_l_inv_04*x168
        x244 = R_l_inv_14*x175
        x245 = R_l_inv_44*x157
        x246 = R_l_inv_54*x155
        x247 = x241 + x242 - x243 - x244 - x245 + x246 + x38 + x39
        x248 = -x240 + x247
        x249 = x232*x45
        x250 = x234*x45
        x251 = x224*x48
        x252 = x250 + x251
        x253 = -x225
        x254 = x240 + x253
        x255 = x247 + x254
        x256 = -x233 + x235 + x237
        x257 = R_l_inv_43*x209
        x258 = R_l_inv_53*x208
        x259 = R_l_inv_03*x205
        x260 = R_l_inv_23*x203
        x261 = R_l_inv_13*x206
        x262 = R_l_inv_33*x204
        x263 = R_l_inv_07*x168 + R_l_inv_17*x175 - R_l_inv_27*x165 - R_l_inv_37*x173 + R_l_inv_47*x157 - R_l_inv_57*x155
        x264 = x263*x48
        x265 = x257 - x258 + x259 - x260 + x261 - x262 + x264
        x266 = -x257 + x258 - x259 + x260 - x261 + x262 + x264
        x267 = -x13*x234 + x233 - x236*x41
        x268 = -x241 - x242 + x243 + x244 + x245 - x246 + x40
        x269 = 2*x12
        x270 = 2*x164
        x271 = r_21*x270 + r_31**3*x19 - r_31*x16 - r_31*x17 + r_31*x18 + r_31*x21 + r_31*x23 - r_31*x27 - r_31*x28 + r_31*x29 + r_31*x32 + r_31*x35 + x160*x167 - x269*x66 - x269*x67 - x269*x68 + x36*x66 + x36*x67 - x36*x69 - x36*x70
        x272 = 2*r_12*x167 + r_12*x174 + r_22*x171 + r_22*x270 - r_32*x16 - r_32*x17 + r_32*x18 + r_32*x21 + r_32*x23 + 3*r_32*x25 - r_32*x27 - r_32*x28 + r_32*x29 + r_32*x32 + r_32*x35 - x269*x73 - x269*x74 - x269*x75 + x36*x73 + x36*x74 - x36*x76 - x36*x77
        x273 = R_l_inv_60*x271 + R_l_inv_70*x272
        x274 = a_0*x273
        x275 = R_l_inv_66*x271 + R_l_inv_76*x272
        x276 = d_1*x275
        x277 = R_l_inv_62*x271 + R_l_inv_72*x272
        x278 = d_1*x277
        x279 = -x278
        x280 = x274 + x276 + x279
        x281 = R_l_inv_64*x271
        x282 = R_l_inv_74*x272
        x283 = R_l_inv_65*x271 + R_l_inv_75*x272
        x284 = x36*x93
        x285 = x36*x94
        x286 = x36*x95
        x287 = x13*x277 + x269*x93 + x269*x94 + x269*x95 + x283*x41 - x284 - x285 - x286
        x288 = -x281 - x282 + x287
        x289 = x275*x45
        x290 = x277*x45
        x291 = x273*x48
        x292 = x290 + x291
        x293 = x281 + x282
        x294 = -2*d_2*d_3*r_33*x20 - 2*d_2*d_3*r_33*x22 - 2*d_2*d_3*r_33*x24 - x13*x277 - x283*x41 + x284 + x285 + x286 + x293
        x295 = R_l_inv_67*x271 + R_l_inv_77*x272
        x296 = x295*x48
        x297 = 4*x212
        x298 = x271*x55 + x272*x56
        x299 = -x297 + x298
        x300 = 4*x38
        x301 = 4*x39
        x302 = x300 - x301
        x303 = -2*d_1*x295
        x304 = -x276
        x305 = x274 + x304
        x306 = x287 + x293
        x307 = x279 + x306
        x308 = -x274
        x309 = x276 + x278 + x306
        x310 = x15*x45
        x311 = x48*(x0 - x310)
        x312 = 4*d_1
        x313 = -x48*(x0 + x310)
        x314 = 4*a_0
        x315 = x314*x53
        x316 = 8*R_l_inv_63
        x317 = 8*R_l_inv_73
        x318 = x45*x97
        x319 = x48*(-x318 + x82)
        x320 = -x48*(x318 + x82)
        x321 = x109*x314
        x322 = -2*a_0*x127
        x323 = x134 - 1
        x324 = x127*x45
        x325 = -x148*x314
        x326 = x134 + 1
        x327 = x48*(2*a_0*x191 - x178)
        x328 = x48*(x178 + x191*x45)
        x329 = -x202*x314
        x330 = 16*d_2
        x331 = x153*x330
        x332 = x156*x330
        x333 = 8*x165
        x334 = 8*x168
        x335 = 8*x173
        x336 = 8*x175
        x337 = x224 + x45
        x338 = x236*x45
        x339 = -x263*x314
        x340 = x48*(2*a_0*x283 - x273)
        x341 = x48*(x273 + x283*x45)
        x342 = x295*x314
        x343 = x300 + x301
        x344 = x61 + x7
        x345 = x81 + x83
        x346 = a_0 + x130
        x347 = x136 + x144
        x348 = x141 - x142
        x349 = x133 + x135
        x350 = x131 + x146
        x351 = x180 + x198
        x352 = x177 + x179
        x353 = x225 + x240 + x247
        x354 = x278 + x305
        x355 = x297 + x298
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = -x44 - x9
        A[0, 1] = x46 - x50
        A[0, 2] = x52 + x9
        A[0, 3] = x58
        A[0, 4] = -4
        A[0, 5] = x59
        A[0, 6] = -x61 - x63
        A[0, 7] = -x46 - x50
        A[0, 8] = -x64 - x65
        A[1, 0] = x100 + x85
        A[1, 1] = x101 - x104
        A[1, 2] = x106 + x108
        A[1, 3] = -x110 - x113
        A[1, 5] = x114 + x115
        A[1, 6] = -x106 - x116 - x92
        A[1, 7] = -x101 - x104
        A[1, 8] = -x117 - x85
        A[2, 0] = -x128 - x132 - x137
        A[2, 1] = x138 + x140 + x143
        A[2, 2] = -x128 - x145 - x147
        A[2, 3] = x150
        A[2, 5] = x151
        A[2, 6] = x132 + x145 + x152
        A[2, 7] = x139 + x143 + x48
        A[2, 8] = x137 + x147 + x152
        A[3, 0] = -x181 - x193
        A[3, 1] = -x194 + x197
        A[3, 2] = -x199 - x201
        A[3, 3] = x211
        A[3, 4] = x213
        A[3, 5] = x214
        A[3, 6] = x199 + x216
        A[3, 7] = x194 + x197
        A[3, 8] = x181 + x217
        A[4, 0] = x239 + x248
        A[4, 1] = -x249 + x252
        A[4, 2] = x255 + x256
        A[4, 3] = x265
        A[4, 5] = x266
        A[4, 6] = -x255 - x267
        A[4, 7] = x249 + x252
        A[4, 8] = x238 + x254 + x268
        A[5, 0] = x280 + x288
        A[5, 1] = -x289 + x292
        A[5, 2] = -x280 - x294
        A[5, 3] = x296 + x299
        A[5, 4] = x302
        A[5, 5] = -x299 - x303
        A[5, 6] = x305 + x307
        A[5, 7] = x289 + x292
        A[5, 8] = x308 + x309
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x311
        B[0, 1] = x312*(-x2 - x3 + x6)
        B[0, 2] = x313
        B[0, 3] = x315 + 4
        B[0, 4] = -r_31*x316 - r_32*x317
        B[0, 5] = x315 - 4
        B[0, 6] = x311
        B[0, 7] = x312*(x4 + x6)
        B[0, 8] = x313
        B[1, 0] = x319
        B[1, 1] = x312*(x80 - x88 - x89)
        B[1, 2] = x320
        B[1, 3] = x321
        B[1, 4] = -x316*x72 - x317*x79
        B[1, 5] = x321
        B[1, 6] = x319
        B[1, 7] = x312*(x80 + x90)
        B[1, 8] = x320
        B[2, 0] = x48*(-x322 - x323)
        B[2, 1] = x312*(R_l_inv_66*x120 + R_l_inv_76*x124 - x126)
        B[2, 2] = x48*(x323 + x324)
        B[2, 3] = x325
        B[2, 4] = x120*x316 + x124*x317
        B[2, 5] = x325
        B[2, 6] = x48*(-x322 - x326)
        B[2, 7] = -x312*(x126 + x129)
        B[2, 8] = x48*(x324 + x326)
        B[3, 0] = x327
        B[3, 1] = x312*(R_l_inv_06*x165 + R_l_inv_16*x173 + R_l_inv_26*x168 + R_l_inv_36*x175 + 2*R_l_inv_46*d_2*x153 + 2*R_l_inv_56*d_2*x156 - x176)
        B[3, 2] = x328
        B[3, 3] = x329
        B[3, 4] = R_l_inv_03*x333 + R_l_inv_13*x335 + R_l_inv_23*x334 + R_l_inv_33*x336 + R_l_inv_43*x331 + R_l_inv_53*x332 - 8*x38 - 8*x39
        B[3, 5] = x329
        B[3, 6] = x327
        B[3, 7] = -x312*(x176 + x186)
        B[3, 8] = x328
        B[4, 0] = x48*(2*a_0*x236 - x337)
        B[4, 1] = x312*(x226 - x227 + x228 - x229 + x230 - x231 - x234)
        B[4, 2] = x48*(x111 + x224 + x338)
        B[4, 3] = x339
        B[4, 4] = R_l_inv_03*x334 + R_l_inv_13*x336 - R_l_inv_23*x333 - R_l_inv_33*x335 + R_l_inv_43*x332 - R_l_inv_53*x331
        B[4, 5] = x339
        B[4, 6] = x48*(-x218 + x219 - x220 + x221 - x222 + x223 + x338 + x45)
        B[4, 7] = x312*(-x232 - x234)
        B[4, 8] = x48*(x337 + x338)
        B[5, 0] = x340
        B[5, 1] = x312*(R_l_inv_66*x271 + R_l_inv_76*x272 - x277)
        B[5, 2] = x341
        B[5, 3] = -x342 + x343
        B[5, 4] = x271*x316 + x272*x317
        B[5, 5] = -x342 - x343
        B[5, 6] = x340
        B[5, 7] = -x312*(x275 + x277)
        B[5, 8] = x341
        C = np.zeros(shape=(6, 9))
        C[0, 0] = -x344 - x44
        C[0, 1] = x46 - x47 + x49
        C[0, 2] = x344 + x52
        C[0, 3] = x59
        C[0, 4] = 4
        C[0, 5] = x58
        C[0, 6] = -x1 - x65
        C[0, 7] = -x46 - x47 + x49
        C[0, 8] = -x60 - x63 - x64
        C[1, 0] = x105 + x108 + x84
        C[1, 1] = x101 - x102 + x103
        C[1, 2] = x100 + x345
        C[1, 3] = x113 + x114
        C[1, 5] = -x110 - x115
        C[1, 6] = -x117 - x345
        C[1, 7] = -x101 - x102 + x103
        C[1, 8] = x345 - x86 - x87 + x91 + x99
        C[2, 0] = -x128 - x346 - x347
        C[2, 1] = x140 + x348 + x48
        C[2, 2] = -x128 - x349 - x350
        C[2, 3] = x151
        C[2, 5] = x150
        C[2, 6] = x152 + x346 + x349
        C[2, 7] = x138 + x139 + x348
        C[2, 8] = x152 + x347 + x350
        C[3, 0] = -x201 - x351
        C[3, 1] = 2*a_0*x176 - x194 - x196
        C[3, 2] = -x193 - x352
        C[3, 3] = x214
        C[3, 4] = x213
        C[3, 5] = x211
        C[3, 6] = x217 + x352
        C[3, 7] = x194 + x195 - x196
        C[3, 8] = x216 + x351
        C[4, 0] = x256 + x353
        C[4, 1] = 2*a_0*x234 - x249 - x251
        C[4, 2] = x238 + x248 + x253
        C[4, 3] = x266
        C[4, 5] = x265
        C[4, 6] = x239 + x240 + x268
        C[4, 7] = x249 + x250 - x251
        C[4, 8] = -x267 - x353
        C[5, 0] = x288 + x354
        C[5, 1] = 2*a_0*x277 - x289 - x291
        C[5, 2] = -x294 - x354
        C[5, 3] = -x303 - x355
        C[5, 4] = -x302
        C[5, 5] = x296 + x355
        C[5, 6] = x274 + x309
        C[5, 7] = x289 + x290 - x291
        C[5, 8] = x304 + x307 + x308
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[2] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor()
    # Finish code for explicit solution node 0

    # Code for solved_variable dispatcher node 2
    def SolvedVariableDispatcherNode_node_2_processor():
        this_node_input_index: List[int] = node_input_index[2]
        this_input_valid: bool = node_input_validity[2]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[2]
            degenerate_valid_0 = (abs(th_2 - 1/2*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True

            th_2 = this_solution[2]
            degenerate_valid_1 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True

            if not taken_by_degenerate:
                add_input_index_to(3, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor()
    # Finish code for solved_variable dispatcher node 2

    # Code for explicit solution node 3, solved variable is th_3
    def ExplicitSolutionNode_node_3_solve_th_3_processor():
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
            th_2 = this_solution[2]
            condition_0: bool = (((a_0 != 0)) and ((d_2 != 0)) and ((math.cos(th_2) != 0)) and (((1/2)*abs((Px**2 - 2*Px*d_3*r_13 + Py**2 - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 - a_0**2 + 2*a_0*d_1*math.sin(th_2) - d_1**2 - d_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_0*d_2*math.cos(th_2))) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = 2*d_3
                x1 = d_3**2
                x2 = safe_asin((1/2)*(Px**2 - Px*r_13*x0 + Py**2 - Py*r_23*x0 + Pz**2 - Pz*r_33*x0 - a_0**2 + 2*a_0*d_1*math.sin(th_2) - d_1**2 - d_2**2 + r_13**2*x1 + r_23**2*x1 + r_33**2*x1)/(a_0*d_2*math.cos(th_2)))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = -x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[3] = x2 + math.pi
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
    # Finish code for explicit solution node 3

    # Code for equation all-zero dispatcher node 4
    def EquationAllZeroDispatcherNode_node_4_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_3 = this_solution[3]
            checked_result: bool = (((abs(d_2*math.cos(th_3)) <= 1.0e-6)) and ((abs(Px - d_3*r_13) <= 1.0e-6)) and ((abs(Py - d_3*r_23) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4

    # Code for explicit solution node 5, solved variable is th_0
    def ExplicitSolutionNode_node_5_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[5]
        this_input_valid: bool = node_input_validity[5]
        if not this_input_valid:
            return

        # The solution of non-root node 5
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_3 = this_solution[3]
            condition_0: bool = (((abs(d_2*math.cos(th_3)) >= zero_tolerance)) or ((abs(Px - d_3*r_13) >= zero_tolerance)) or ((abs(Py - d_3*r_23) >= zero_tolerance)))
            if condition_0:
                x0 = -Px + d_3*r_13
                x1 = Py - d_3*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.cos(th_3)
                x4 = safe_sqrt(-d_2**2*x3**2 + x0**2 + x1**2)
                x5 = d_2*x3
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x2 + math.atan2(x4, x5)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                this_solution[0] = x2 + math.atan2(-x4, x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_0_processor()
    # Finish code for explicit solution node 5

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
            th_3 = this_solution[3]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(12, branch_input_index)

            th_3 = this_solution[3]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(17, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6

    # Code for explicit solution node 17, solved variable is th_5
    def ExplicitSolutionNode_node_17_solve_th_5_processor():
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
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance)) or ((abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                this_solution[5] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_5_processor()
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
            th_2 = this_solution[2]
            degenerate_valid_0 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(25, node_input_i_idx_in_queue)

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
            th_3 = this_solution[3]
            condition_0: bool = (((abs(a_0 + d_1) >= 1.0e-6)) or ((abs(d_2*math.sin(th_3)) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 + d_1
                x1 = Pz - d_3*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_2*math.sin(th_3)
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
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
            th_3 = this_solution[3]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) - r_23*(-math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_0)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                this_solution[4] = math.atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_4_processor()
    # Finish code for explicit solution node 26

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
            th_2 = this_solution[2]
            condition_0: bool = (((abs(d_1*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 - d_1*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_3*r_33
                x1 = a_0 - d_1*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_1*math.cos(th_2)
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4)
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + r_23*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - r_33*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = x0*x1 - x2*x3
                x5 = x0*x2 + x1*x3
                x6 = r_13*math.cos(th_0)
                x7 = r_23*math.sin(th_0)
                this_solution[4] = math.atan2(-r_33*x4 + x5*x6 + x5*x7, -r_33*x5 - x4*x6 - x4*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_4_processor()
    # Finish code for explicit solution node 20

    # Code for explicit solution node 12, solved variable is th_5
    def ExplicitSolutionNode_node_12_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[12]
        this_input_valid: bool = node_input_validity[12]
        if not this_input_valid:
            return

        # The solution of non-root node 12
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
                this_solution[5] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_5_processor()
    # Finish code for explicit solution node 12

    # Code for solved_variable dispatcher node 13
    def SolvedVariableDispatcherNode_node_13_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[2]
            degenerate_valid_0 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(22, node_input_i_idx_in_queue)

            if not taken_by_degenerate:
                add_input_index_to(14, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor()
    # Finish code for solved_variable dispatcher node 13

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
            th_3 = this_solution[3]
            condition_0: bool = (((abs(a_0 + d_1) >= 1.0e-6)) or ((abs(d_2*math.sin(th_3)) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 + d_1
                x1 = Pz - d_3*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_2*math.sin(th_3)
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
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
            th_3 = this_solution[3]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) - r_23*(-math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_0)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                this_solution[4] = math.atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_4_processor()
    # Finish code for explicit solution node 23

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
            th_2 = this_solution[2]
            condition_0: bool = (((abs(d_1*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 - d_1*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_3*r_33
                x1 = a_0 - d_1*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_1*math.cos(th_2)
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4)
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
            th_2 = this_solution[2]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + r_23*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - r_33*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = x0*x1 - x2*x3
                x5 = x0*x2 + x1*x3
                x6 = r_13*math.cos(th_0)
                x7 = r_23*math.sin(th_0)
                this_solution[4] = math.atan2(r_33*x4 - x5*x6 - x5*x7, -r_33*x5 - x4*x6 - x4*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor()
    # Finish code for explicit solution node 15

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
            th_3 = this_solution[3]
            condition_0: bool = (((math.sin(th_3) != 0)) and ((abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_asin((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                this_solution[4] = math.pi - x0
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            checked_result: bool = (((abs(d_1*math.cos(th_2) - d_2*math.sin(th_2)*math.sin(th_3)) <= 1.0e-6)) and ((abs(-a_0 + d_1*math.sin(th_2) + d_2*math.sin(th_3)*math.cos(th_2)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8

    # Code for explicit solution node 9, solved variable is th_1
    def ExplicitSolutionNode_node_9_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (((abs(d_1*math.cos(th_2) - d_2*math.sin(th_2)*math.sin(th_3)) >= 1.0e-6)) or ((abs(-a_0 + d_1*math.sin(th_2) + d_2*math.sin(th_3)*math.cos(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_3*r_33
                x1 = math.sin(th_2)
                x2 = math.cos(th_2)
                x3 = d_2*math.sin(th_3)
                x4 = a_0 - d_1*x1 - x2*x3
                x5 = d_1*x2 - x1*x3
                x6 = math.cos(th_0)
                x7 = math.sin(th_0)
                x8 = Px*x6 + Py*x7 - d_3*r_13*x6 - d_3*r_23*x7
                this_solution[1] = math.atan2(-x0*x4 + x5*x8, x0*x5 + x4*x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1_processor()
    # Finish code for explicit solution node 9

    # Code for non-branch dispatcher node 10
    # Actually, there is no code

    # Code for explicit solution node 11, solved variable is th_5
    def ExplicitSolutionNode_node_11_solve_th_5_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*((math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) - r_21*((-math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) + math.cos(th_0)*math.cos(th_3)) - r_31*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance)) or ((abs(r_12*((math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) - r_22*((-math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) + math.cos(th_0)*math.cos(th_3)) - r_32*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_2)
                x3 = math.sin(th_2)
                x4 = math.cos(th_1)
                x5 = x0*(x1*x2 - x3*x4)
                x6 = math.sin(th_0)
                x7 = math.cos(th_3)
                x8 = math.cos(th_0)
                x9 = x1*x3 + x2*x4
                x10 = x0*x8*x9 + x6*x7
                x11 = -x0*x6*x9 + x7*x8
                this_solution[5] = math.atan2(r_11*x10 - r_21*x11 - r_31*x5, r_12*x10 - r_22*x11 - r_32*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_5_processor()
    # Finish code for explicit solution node 10

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
        value_at_2 = ik_out_i[2]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[3]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[4]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[5]  # th_5
        new_ik_i[5] = value_at_5
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def yaskawa_HC10_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = yaskawa_HC10_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    strict_raw_witness = False
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = add_revolute_offset(ik_out_i[5], -th_5_offset_original2raw)
        ee_pose_i = yaskawa_HC10_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                strict_raw_witness = True
                _yaik_seed_candidates.clear()
            elif not strict_raw_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -1.5707963267948966, -3.141592653589793, -6.283185307179586, -6.283185307179586, -6.283185307179586], [3.141592653589793, 4.71238898038469, 3.141592653589793, 0.0, 0.0, 0.0], [True, True, True, True, True, True], [False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = yaskawa_HC10_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
    return ik_output


def yaskawa_HC10_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = []
    for transform in (yaskawa_HC10_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in yaskawa_HC10_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: yaskawa_HC10_fk(q)
    if refinement_seeds and not any(np.max(np.abs(refinement_fk(q)-T_ee)) < 1e-7 for q in result):
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                break
    return result


def test_ik_solve_yaskawa_HC10():
    theta_in = np.random.random(size=(6, ))
    ee_pose = yaskawa_HC10_fk(theta_in)
    ik_output = yaskawa_HC10_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = yaskawa_HC10_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_yaskawa_HC10()
