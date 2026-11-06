import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 32
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.612
a_3: float = 0.5723
d_0: float = 0.220941
d_2: float = -0.1719
d_4: float = 0.11485899999999999
d_5: float = 0.1157
d_6: float = 0.0922
pre_transform_special_symbol_23: float = 0.1273

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -0.0
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = -0.0
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def ur_10_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12 + 6.12323399573677e-17*r_13
    ee_transformed[0, 2] = 6.12323399573677e-17*r_12 - 1.0*r_13
    ee_transformed[0, 3] = -1.0*Px
    ee_transformed[1, 0] = 1.0*r_21
    ee_transformed[1, 1] = 1.0*r_22 + 6.12323399573677e-17*r_23
    ee_transformed[1, 2] = 6.12323399573677e-17*r_22 - 1.0*r_23
    ee_transformed[1, 3] = -1.0*Py
    ee_transformed[2, 0] = -1.0*r_31
    ee_transformed[2, 1] = -1.0*r_32 - 6.12323399573677e-17*r_33
    ee_transformed[2, 2] = -6.12323399573677e-17*r_32 + 1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def ur_10_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12 + 6.12323399573677e-17*r_13
    ee_transformed[0, 2] = 6.12323399573677e-17*r_12 - 1.0*r_13
    ee_transformed[0, 3] = -1.0*Px
    ee_transformed[1, 0] = 1.0*r_21
    ee_transformed[1, 1] = 1.0*r_22 + 6.12323399573677e-17*r_23
    ee_transformed[1, 2] = 6.12323399573677e-17*r_22 - 1.0*r_23
    ee_transformed[1, 3] = -1.0*Py
    ee_transformed[2, 0] = -1.0*r_31
    ee_transformed[2, 1] = -1.0*r_32 - 6.12323399573677e-17*r_33
    ee_transformed[2, 2] = -6.12323399573677e-17*r_32 + 1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def ur_10_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_5)
    x1 = math.sin(th_3)
    x2 = math.cos(th_0)
    x3 = math.sin(th_1)
    x4 = math.sin(th_2)
    x5 = x3*x4
    x6 = math.cos(th_1)
    x7 = math.cos(th_2)
    x8 = -x2*x5 + x2*x6*x7
    x9 = math.cos(th_3)
    x10 = x3*x7
    x11 = x4*x6
    x12 = -x10*x2 - x11*x2
    x13 = -x1*x8 + x12*x9
    x14 = 1.0*x13
    x15 = math.cos(th_5)
    x16 = math.sin(th_0)
    x17 = math.sin(th_4)
    x18 = math.cos(th_4)
    x19 = x1*x12 + x8*x9
    x20 = x16*x17 + x18*x19
    x21 = 1.0*x20
    x22 = x17*x19
    x23 = 6.12323399573677e-17*x15
    x24 = 6.12323399573677e-17*x0
    x25 = 1.0*x2
    x26 = a_1*x6
    x27 = 1.0*a_3
    x28 = 1.0*d_6
    x29 = -x16*x5 + x16*x6*x7
    x30 = -x10*x16 - x11*x16
    x31 = -x1*x29 + x30*x9
    x32 = 1.0*x31
    x33 = x1*x30 + x29*x9
    x34 = -x17*x2 + x18*x33
    x35 = 1.0*x34
    x36 = x18*x2
    x37 = x17*x33
    x38 = x5 - x6*x7
    x39 = -x10 - x11
    x40 = -x1*x39 + x38*x9
    x41 = x1*x38 + x39*x9
    x42 = x18*x41
    x43 = 1.0*x42
    x44 = x17*x41
    x45 = 1.0*x40
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x14 + x15*x21
    ee_pose[0, 1] = -x0*x21 - x14*x15 + 6.12323399573677e-17*x16*x18 - 6.12323399573677e-17*x22
    ee_pose[0, 2] = -x13*x23 - 1.0*x16*x18 + 1.0*x17*x19 - x20*x24
    ee_pose[0, 3] = 1.0*d_0*x16 + 1.0*d_2*x16 + 1.0*d_4*x16 - d_5*x14 - x25*x26 - x27*x8 - x28*(x16*x18 - x22)
    ee_pose[1, 0] = -x0*x32 + x15*x35
    ee_pose[1, 1] = -x0*x35 - x15*x32 - 6.12323399573677e-17*x36 - 6.12323399573677e-17*x37
    ee_pose[1, 2] = 1.0*x17*x33 + 1.0*x18*x2 - x23*x31 - x24*x34
    ee_pose[1, 3] = -d_0*x25 - d_2*x25 - d_4*x25 - d_5*x32 - 1.0*x16*x26 - x27*x29 - x28*(-x36 - x37)
    ee_pose[2, 0] = 1.0*x0*x40 - x15*x43
    ee_pose[2, 1] = x0*x43 + x15*x45 + 6.12323399573677e-17*x44
    ee_pose[2, 2] = x23*x40 + x24*x42 - 1.0*x44
    ee_pose[2, 3] = -1.0*a_1*x3 + d_5*x45 + 1.0*pre_transform_special_symbol_23 + x27*x39 - x28*x44
    return ee_pose


def ur_10_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.sin(th_3)
    x2 = math.sin(th_1)
    x3 = math.sin(th_2)
    x4 = math.cos(th_0)
    x5 = 1.0*x4
    x6 = x3*x5
    x7 = math.cos(th_1)
    x8 = math.cos(th_2)
    x9 = x5*x8
    x10 = x2*x6 - x7*x9
    x11 = math.cos(th_3)
    x12 = x2*x9 + x6*x7
    x13 = -x1*x10 + x11*x12
    x14 = math.cos(th_4)
    x15 = math.sin(th_4)
    x16 = -x0*x14 - x15*(x1*x12 + x10*x11)
    x17 = -x5
    x18 = x0*x3
    x19 = x0*x8
    x20 = x18*x2 - x19*x7
    x21 = x18*x7 + x19*x2
    x22 = -x1*x20 + x11*x21
    x23 = 1.0*x14*x4 - x15*(x1*x21 + x11*x20)
    x24 = 1.0*x3
    x25 = 1.0*x8
    x26 = x2*x24 - x25*x7
    x27 = -x2*x25 - x24*x7
    x28 = -x1*x27 + x11*x26
    x29 = x15*(x1*x26 + x11*x27)
    x30 = 1.0*a_1*x2
    x31 = pre_transform_special_symbol_23 - x30
    x32 = a_3*x27 + pre_transform_special_symbol_23 - x30
    x33 = d_5*x28 + x32
    x34 = a_1*x7
    x35 = d_0*x5 + d_2*x5 + x0*x34
    x36 = -a_3*x20 + d_4*x5 + x35
    x37 = -d_5*x22 + x36
    x38 = -x37
    x39 = -d_6*x29 + x33
    x40 = d_6*x23 - x37
    x41 = d_0*x0 + d_2*x0 - x34*x5
    x42 = a_3*x10 + d_4*x0 + x41
    x43 = d_5*x13 + x42
    x44 = d_6*x16 + x43
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x0
    jacobian_output[0, 4] = x13
    jacobian_output[0, 5] = x16
    jacobian_output[1, 1] = x17
    jacobian_output[1, 2] = x17
    jacobian_output[1, 3] = x17
    jacobian_output[1, 4] = x22
    jacobian_output[1, 5] = x23
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 4] = x28
    jacobian_output[2, 5] = -x29
    jacobian_output[3, 1] = pre_transform_special_symbol_23*x5
    jacobian_output[3, 2] = x31*x5
    jacobian_output[3, 3] = x32*x5
    jacobian_output[3, 4] = -x22*x33 + x28*x38
    jacobian_output[3, 5] = -x23*x39 - x29*x40
    jacobian_output[4, 1] = pre_transform_special_symbol_23*x0
    jacobian_output[4, 2] = x0*x31
    jacobian_output[4, 3] = x0*x32
    jacobian_output[4, 4] = x13*x33 - x28*x43
    jacobian_output[4, 5] = x16*x39 + x29*x44
    jacobian_output[5, 2] = x0*x35 - x41*x5
    jacobian_output[5, 3] = x0*x36 - x42*x5
    jacobian_output[5, 4] = -x13*x38 + x22*x43
    jacobian_output[5, 5] = -x16*x40 + x23*x44
    return jacobian_output


def ur_10_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.sin(th_3)
    x2 = math.sin(th_1)
    x3 = math.sin(th_2)
    x4 = math.cos(th_0)
    x5 = 1.0*x4
    x6 = x3*x5
    x7 = math.cos(th_1)
    x8 = math.cos(th_2)
    x9 = x5*x8
    x10 = x2*x6 - x7*x9
    x11 = math.cos(th_3)
    x12 = x2*x9 + x6*x7
    x13 = math.cos(th_4)
    x14 = math.sin(th_4)
    x15 = -x5
    x16 = x0*x3
    x17 = x0*x8
    x18 = x16*x2 - x17*x7
    x19 = x16*x7 + x17*x2
    x20 = 1.0*x3
    x21 = 1.0*x8
    x22 = x2*x20 - x21*x7
    x23 = -x2*x21 - x20*x7
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x0
    jacobian_output[0, 4] = -x1*x10 + x11*x12
    jacobian_output[0, 5] = -x0*x13 - x14*(x1*x12 + x10*x11)
    jacobian_output[1, 1] = x15
    jacobian_output[1, 2] = x15
    jacobian_output[1, 3] = x15
    jacobian_output[1, 4] = -x1*x18 + x11*x19
    jacobian_output[1, 5] = 1.0*x13*x4 - x14*(x1*x19 + x11*x18)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 4] = -x1*x23 + x11*x22
    jacobian_output[2, 5] = -x14*(x1*x22 + x11*x23)
    return jacobian_output


def ur_10_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
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
    x6 = a_1*x5
    x7 = pre_transform_special_symbol_23 - x6
    x8 = math.cos(th_2)
    x9 = x5*x8
    x10 = math.sin(th_2)
    x11 = math.cos(th_1)
    x12 = 1.0*x11
    x13 = x10*x12
    x14 = -x13 - x9
    x15 = a_3*x14 + pre_transform_special_symbol_23 - x6
    x16 = math.cos(th_3)
    x17 = x10*x5
    x18 = x12*x8
    x19 = x17 - x18
    x20 = math.sin(th_3)
    x21 = -x14*x20 + x16*x19
    x22 = math.sin(th_0)
    x23 = x17*x22 - x18*x22
    x24 = x13*x22 + x22*x9
    x25 = x16*x24 - x20*x23
    x26 = d_5*x21 + x15
    x27 = a_1*x12*x22 + d_0*x2 + d_2*x2
    x28 = -a_3*x23 + d_4*x2 + x27
    x29 = -d_5*x25 + x28
    x30 = -x29
    x31 = math.sin(th_4)
    x32 = x14*x16 + x19*x20
    x33 = x31*x32
    x34 = math.cos(th_4)
    x35 = 1.0*x1*x34 - x31*(x16*x23 + x20*x24)
    x36 = -d_6*x33 + x26
    x37 = d_6*x35 - x29
    x38 = 1.0*p_on_ee_x
    x39 = 1.0*x22
    x40 = p_on_ee_z*x39
    x41 = x10*x2
    x42 = x2*x8
    x43 = -x11*x42 + x4*x41
    x44 = x11*x41 + x4*x42
    x45 = x16*x44 - x20*x43
    x46 = -a_1*x11*x2 + d_0*x39 + d_2*x39
    x47 = a_3*x43 + d_4*x39 + x46
    x48 = d_5*x45 + x47
    x49 = -x31*(x16*x43 + x20*x44) - x34*x39
    x50 = d_6*x49 + x48
    x51 = x0*x22 + x1*x38
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = 1.0*pre_transform_special_symbol_23*x1 - x3
    jacobian_output[0, 2] = 1.0*x1*x7 - x3
    jacobian_output[0, 3] = 1.0*x1*x15 - x3
    jacobian_output[0, 4] = -p_on_ee_y*x21 + p_on_ee_z*x25 + x21*x30 - x25*x26
    jacobian_output[0, 5] = p_on_ee_y*x33 + p_on_ee_z*x35 - x33*x37 - x35*x36
    jacobian_output[1, 0] = x38
    jacobian_output[1, 1] = 1.0*pre_transform_special_symbol_23*x22 - x40
    jacobian_output[1, 2] = 1.0*x22*x7 - x40
    jacobian_output[1, 3] = 1.0*x15*x22 - x40
    jacobian_output[1, 4] = p_on_ee_x*x21 - p_on_ee_z*x45 - x21*x48 + x26*x45
    jacobian_output[1, 5] = -p_on_ee_x*x33 - p_on_ee_z*x49 + x31*x32*x50 + x36*x49
    jacobian_output[2, 1] = x51
    jacobian_output[2, 2] = -x2*x46 + x27*x39 + x51
    jacobian_output[2, 3] = -x2*x47 + x28*x39 + x51
    jacobian_output[2, 4] = -p_on_ee_x*x25 + p_on_ee_y*x45 + x25*x48 - x30*x45
    jacobian_output[2, 5] = -p_on_ee_x*x35 + p_on_ee_y*x49 + x35*x50 - x37*x49
    return jacobian_output


def ur_10_ik_solve_raw(T_ee: np.ndarray):
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
        return IkSolution([math.nan] * 9)

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
    for i in range(32):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for non-branch dispatcher node 0
    # Actually, there is no code

    # Code for explicit solution node 1, solved variable is th_0
    def ExplicitSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The explicit solution of root node
        condition_0: bool = (((abs(d_0 + d_2 + d_4) >= zero_tolerance)) or ((abs(Px - d_6*r_13) >= zero_tolerance)) or ((abs(Py - d_6*r_23) >= zero_tolerance)))
        if condition_0:
            x0 = Px - d_6*r_13
            x1 = -Py + d_6*r_23
            x2 = math.atan2(x0, x1)
            x3 = -d_0 - d_2 - d_4
            x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
            solution_0: IkSolution = make_ik_solution()
            solution_0[0] = x2 + math.atan2(x4, x3)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)

            solution_1: IkSolution = make_ik_solution()
            solution_1[0] = x2 + math.atan2(-x4, x3)
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)

    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor()
    # Finish code for explicit solution node 0

    # Code for non-branch dispatcher node 2
    # Actually, there is no code

    # Code for explicit solution node 3, solved variable is th_4
    def ExplicitSolutionNode_node_3_solve_th_4_processor():
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
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) <= 70368744177665/70368744177664)
            if condition_0:
                x0 = safe_acos(r_13*math.sin(th_0) - r_23*math.cos(th_0))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[7] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_4_processor()
    # Finish code for explicit solution node 2

    # Code for solved_variable dispatcher node 4
    def SolvedVariableDispatcherNode_node_4_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            forwarded_to_child = False
            th_4 = this_solution[7]
            degenerate_valid_0 = (abs(th_4) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(18, branch_input_index)

            th_4 = this_solution[7]
            degenerate_valid_1 = (abs(th_4 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(25, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(5, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_4_processor()
    # Finish code for solved_variable dispatcher node 4

    # Code for explicit solution node 25, solved variable is th_5
    def ExplicitSolutionNode_node_25_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[25]
        this_input_valid: bool = node_input_validity[25]
        if not this_input_valid:
            return
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_0 = this_solution[0]
            family_values = []
            if (r_31**2 + (r_11*math.cos(th_0) + r_21*math.sin(th_0))**2 > 1.0e-12):
                family_values.append(-3.14159265358979)
                family_values.append(-2.94524311274043)
                family_values.append(-2.74889357189107)
                family_values.append(-2.55254403104171)
                family_values.append(-2.35619449019234)
                family_values.append(-2.15984494934298)
                family_values.append(-1.96349540849362)
                family_values.append(-1.76714586764426)
                family_values.append(-1.57079632679490)
                family_values.append(-1.37444678594553)
                family_values.append(-1.17809724509617)
                family_values.append(-0.981747704246811)
                family_values.append(-0.785398163397448)
                family_values.append(-0.589048622548086)
                family_values.append(-0.392699081698724)
                family_values.append(-0.196349540849362)
                family_values.append(0.0)
                family_values.append(0.196349540849362)
                family_values.append(0.392699081698724)
                family_values.append(0.589048622548086)
                family_values.append(0.785398163397448)
                family_values.append(0.981747704246811)
                family_values.append(1.17809724509617)
                family_values.append(1.37444678594553)
                family_values.append(1.57079632679490)
                family_values.append(1.76714586764426)
                family_values.append(1.96349540849362)
                family_values.append(2.15984494934298)
                family_values.append(2.35619449019234)
                family_values.append(2.55254403104171)
                family_values.append(2.74889357189107)
                family_values.append(2.94524311274043)
                family_values.append(3.14159265358979)
            if (r_31**2 + (r_11*math.cos(th_0) + r_21*math.sin(th_0))**2 > 1.0e-12):
                x0 = -Pz + d_6*r_33
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                x3 = r_11*x1 + r_21*x2
                x4 = 1/safe_sqrt(r_31**2 + x3**2)
                x5 = -x3*x4
                x6 = x1*(Px - d_6*r_13) + x2*(Py - d_6*r_23)
                x7 = r_31*x4
                x8 = 2*d_5
                x9 = x8*(-x0*x5 + x6*x7)
                x10 = -x8*(x0*x7 + x5*x6)
                x11 = d_5**2 + x0**2 + x6**2
                family_values = trig_family_representatives([(x9, x10, x11 - (a_1 - a_3)**2), (x9, x10, x11 - (a_1 + a_3)**2)], -3.141592653589793, 3.141592653589793, family_values)
            for family_index, family_value in enumerate(family_values):
                candidate = copy.copy(this_solution)
                candidate[8] = family_value
                if family_index == len(family_values)-1:
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(26, candidate_index)
            if not family_values:
                queue_element_validity[queue_index] = False
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_5_processor()
    # Finish code for explicit solution node 25

    # Code for non-branch dispatcher node 26
    # Actually, there is no code

    # Code for explicit solution node 27, solved variable is th_2
    def ExplicitSolutionNode_node_27_solve_th_2_processor():
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
            th_5 = this_solution[8]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = th_5 - math.atan2(-r_31, r_11*x0 + r_21*x1)
                x3 = safe_acos((1/2)*(-a_1**2 - a_3**2 + (-Pz + d_5*math.cos(x2) + d_6*r_33)**2 + (d_5*math.sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23))**2)/(a_1*a_3))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x3
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(28, appended_idx)
                this_solution[4] = -x3
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_2_processor()
    # Finish code for explicit solution node 26

    # Code for non-branch dispatcher node 28
    # Actually, there is no code

    # Code for explicit solution node 29, solved variable is th_1
    def ExplicitSolutionNode_node_29_solve_th_1_processor():
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
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = th_5 - math.atan2(-r_31, r_11*x0 + r_21*x1)
                this_solution[1] = -math.atan2(a_3*math.sin(th_2), a_1 + a_3*math.cos(th_2)) + math.atan2(-Pz + d_5*math.cos(x2) + d_6*r_33, d_5*math.sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(30, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_1_processor()
    # Finish code for explicit solution node 28

    # Code for non-branch dispatcher node 30
    # Actually, there is no code

    # Code for explicit solution node 31, solved variable is th_3
    def ExplicitSolutionNode_node_31_solve_th_3_processor():
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
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = True
            if condition_0:
                this_solution[6] = -th_1 - th_2 - th_5 + math.atan2(-r_31, r_11*math.cos(th_0) + r_21*math.sin(th_0)) - math.pi
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_3_processor()
    # Finish code for explicit solution node 30

    # Code for explicit solution node 18, solved variable is th_5
    def ExplicitSolutionNode_node_18_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_0 = this_solution[0]
            family_values = []
            if (r_31**2 + (r_11*math.cos(th_0) + r_21*math.sin(th_0))**2 > 1.0e-12):
                family_values.append(-3.14159265358979)
                family_values.append(-2.94524311274043)
                family_values.append(-2.74889357189107)
                family_values.append(-2.55254403104171)
                family_values.append(-2.35619449019234)
                family_values.append(-2.15984494934298)
                family_values.append(-1.96349540849362)
                family_values.append(-1.76714586764426)
                family_values.append(-1.57079632679490)
                family_values.append(-1.37444678594553)
                family_values.append(-1.17809724509617)
                family_values.append(-0.981747704246811)
                family_values.append(-0.785398163397448)
                family_values.append(-0.589048622548086)
                family_values.append(-0.392699081698724)
                family_values.append(-0.196349540849362)
                family_values.append(0.0)
                family_values.append(0.196349540849362)
                family_values.append(0.392699081698724)
                family_values.append(0.589048622548086)
                family_values.append(0.785398163397448)
                family_values.append(0.981747704246811)
                family_values.append(1.17809724509617)
                family_values.append(1.37444678594553)
                family_values.append(1.57079632679490)
                family_values.append(1.76714586764426)
                family_values.append(1.96349540849362)
                family_values.append(2.15984494934298)
                family_values.append(2.35619449019234)
                family_values.append(2.55254403104171)
                family_values.append(2.74889357189107)
                family_values.append(2.94524311274043)
                family_values.append(3.14159265358979)
            if (r_31**2 + (r_11*math.cos(th_0) + r_21*math.sin(th_0))**2 > 1.0e-12):
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = r_11*x0 + r_21*x1
                x3 = 1/safe_sqrt(r_31**2 + x2**2)
                x4 = -Pz + d_6*r_33
                x5 = x3*x4
                x6 = x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23)
                x7 = 2*d_5
                x8 = x7*(-r_31*x3*x6 - x2*x5)
                x9 = x7*(-r_31*x5 + x2*x3*x6)
                x10 = d_5**2 + x4**2 + x6**2
                family_values = trig_family_representatives([(x8, x9, x10 - (a_1 - a_3)**2), (x8, x9, x10 - (a_1 + a_3)**2)], -3.141592653589793, 3.141592653589793, family_values)
            for family_index, family_value in enumerate(family_values):
                candidate = copy.copy(this_solution)
                candidate[8] = family_value
                if family_index == len(family_values)-1:
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(19, candidate_index)
            if not family_values:
                queue_element_validity[queue_index] = False
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_5_processor()
    # Finish code for explicit solution node 18

    # Code for non-branch dispatcher node 19
    # Actually, there is no code

    # Code for explicit solution node 20, solved variable is th_2
    def ExplicitSolutionNode_node_20_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[19]
        this_input_valid: bool = node_input_validity[19]
        if not this_input_valid:
            return

        # The solution of non-root node 20
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[8]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = th_5 + math.atan2(-r_31, r_11*x0 + r_21*x1)
                x3 = safe_acos((1/2)*(-a_1**2 - a_3**2 + (-Pz - d_5*math.cos(x2) + d_6*r_33)**2 + (d_5*math.sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23))**2)/(a_1*a_3))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x3
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(21, appended_idx)
                this_solution[4] = -x3
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(21, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_2_processor()
    # Finish code for explicit solution node 19

    # Code for non-branch dispatcher node 21
    # Actually, there is no code

    # Code for explicit solution node 22, solved variable is th_1
    def ExplicitSolutionNode_node_22_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[21]
        this_input_valid: bool = node_input_validity[21]
        if not this_input_valid:
            return

        # The solution of non-root node 22
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = th_5 + math.atan2(-r_31, r_11*x0 + r_21*x1)
                this_solution[1] = -math.atan2(a_3*math.sin(th_2), a_1 + a_3*math.cos(th_2)) + math.atan2(-Pz - d_5*math.cos(x2) + d_6*r_33, d_5*math.sin(x2) + x0*(Px - d_6*r_13) + x1*(Py - d_6*r_23))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(23, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_1_processor()
    # Finish code for explicit solution node 21

    # Code for non-branch dispatcher node 23
    # Actually, there is no code

    # Code for explicit solution node 24, solved variable is th_3
    def ExplicitSolutionNode_node_24_solve_th_3_processor():
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
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = True
            if condition_0:
                this_solution[6] = -th_1 - th_2 + th_5 + math.atan2(-r_31, r_11*math.cos(th_0) + r_21*math.sin(th_0))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_3_processor()
    # Finish code for explicit solution node 23

    # Code for explicit solution node 5, solved variable is th_1th_2th_3_soa
    def ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor():
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
            th_0 = this_solution[0]
            th_4 = this_solution[7]
            condition_0: bool = (((abs(r_33) >= zero_tolerance)) or ((abs(math.sin(th_4)) >= zero_tolerance)) or ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_4)
                this_solution[3] = math.atan2(r_33*x0, x0*(-r_13*math.cos(th_0) - r_23*math.sin(th_0)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor()
    # Finish code for explicit solution node 5

    # Code for non-branch dispatcher node 6
    # Actually, there is no code

    # Code for explicit solution node 7, solved variable is th_5
    def ExplicitSolutionNode_node_7_solve_th_5_processor():
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
            th_0 = this_solution[0]
            th_4 = this_solution[7]
            condition_0: bool = (((abs(math.sin(th_4)) >= zero_tolerance)) or ((abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance)) or ((abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_4)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                this_solution[8] = math.atan2(x0*(-r_12*x1 + r_22*x2), x0*(r_11*x1 - r_21*x2))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor()
    # Finish code for explicit solution node 6

    # Code for non-branch dispatcher node 8
    # Actually, there is no code

    # Code for explicit solution node 9, solved variable is th_2
    def ExplicitSolutionNode_node_9_solve_th_2_processor():
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
            th_5 = this_solution[8]
            condition_0: bool = (((a_1 != 0)) and ((a_3 != 0)) and (((1/2)*abs((a_1**2 + a_3**2 + d_0**2 + 2*d_0*d_2 + 2*d_0*d_4 + d_2**2 + 2*d_2*d_4 + d_4**2 - d_5**2 + 2*d_5*inv_Px*math.sin(th_5) + 2*d_5*inv_Py*math.cos(th_5) - d_6**2 - 2*d_6*inv_Pz - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(a_1*a_3)) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = 2*d_0
                x1 = 2*d_5
                x2 = safe_acos((1/2)*(-a_1**2 - a_3**2 - d_0**2 - d_2**2 - 2*d_2*d_4 - d_2*x0 - d_4**2 - d_4*x0 + d_5**2 + d_6**2 + 2*d_6*inv_Pz + inv_Px**2 - inv_Px*x1*math.sin(th_5) + inv_Py**2 - inv_Py*x1*math.cos(th_5) + inv_Pz**2)/(a_1*a_3))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                this_solution[4] = -x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_2_processor()
    # Finish code for explicit solution node 8

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
            th_0 = this_solution[0]
            th_5 = this_solution[8]
            checked_result: bool = (((abs(Pz + d_5*r_31*math.sin(th_5) + d_5*r_32*math.cos(th_5) - d_6*r_33) <= 1.0e-6)) and ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_5*r_11*math.sin(th_5)*math.cos(th_0) + d_5*r_12*math.cos(th_0)*math.cos(th_5) + d_5*r_21*math.sin(th_0)*math.sin(th_5) + d_5*r_22*math.sin(th_0)*math.cos(th_5) - d_6*r_13*math.cos(th_0) - d_6*r_23*math.sin(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor()
    # Finish code for equation all-zero dispatcher node 10

    # Code for explicit solution node 11, solved variable is th_1
    def ExplicitSolutionNode_node_11_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = (((abs(Pz + d_5*r_31*math.sin(th_5) + d_5*r_32*math.cos(th_5) - d_6*r_33) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_5*r_11*math.sin(th_5)*math.cos(th_0) + d_5*r_12*math.cos(th_0)*math.cos(th_5) + d_5*r_21*math.sin(th_0)*math.sin(th_5) + d_5*r_22*math.sin(th_0)*math.cos(th_5) - d_6*r_13*math.cos(th_0) - d_6*r_23*math.sin(th_0)) >= 1.0e-6)))
            if condition_0:
                x0 = -a_1 - a_3*math.cos(th_2)
                x1 = d_5*math.sin(th_5)
                x2 = d_5*math.cos(th_5)
                x3 = Pz - d_6*r_33 + r_31*x1 + r_32*x2
                x4 = math.cos(th_0)
                x5 = math.sin(th_0)
                x6 = -Px*x4 - Py*x5 + d_6*r_13*x4 + d_6*r_23*x5 - r_11*x1*x4 - r_12*x2*x4 - r_21*x1*x5 - r_22*x2*x5
                x7 = a_3*math.sin(th_2)
                this_solution[1] = math.atan2(x0*x3 + x6*x7, x0*x6 - x3*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 11

    # Code for non-branch dispatcher node 12
    # Actually, there is no code

    # Code for explicit solution node 13, solved variable is th_2th_3_soa
    def ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor():
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
            th_1 = this_solution[1]
            th_1th_2th_3_soa = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[5] = -th_1 + th_1th_2th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(14, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor()
    # Finish code for explicit solution node 12

    # Code for non-branch dispatcher node 14
    # Actually, there is no code

    # Code for explicit solution node 15, solved variable is th_3
    def ExplicitSolutionNode_node_15_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[14]
        this_input_valid: bool = node_input_validity[14]
        if not this_input_valid:
            return

        # The solution of non-root node 15
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_2th_3_soa = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[6] = -th_2 + th_2th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(16, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_3_processor()
    # Finish code for explicit solution node 14

    # Code for non-branch dispatcher node 16
    # Actually, there is no code

    # Code for explicit solution node 17, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[2] = th_1 + th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 16

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
        value_at_3 = ik_out_i[6]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[7]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[8]  # th_5
        new_ik_i[5] = value_at_5
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def ur_10_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = ur_10_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = ik_out_i[1] - th_1_offset_original2raw
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = ik_out_i[3] - th_3_offset_original2raw
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw
        ee_pose_i = ur_10_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                _yaik_seed_candidates.strict_witness = True
                _yaik_seed_candidates.clear()
            elif not _yaik_seed_candidates.strict_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793, -6.283185307179586, -3.141592653589793], [3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 0.0, 3.141592653589793], [True, True, True, True, True, True], [False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = ur_10_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
                if _yaik_seed_candidates is not None and np.max(np.abs(snapped_pose-T_ee)) < 1e-7:
                    _yaik_seed_candidates.strict_witness = True
                    _yaik_seed_candidates.clear()
    return ik_output


def ur_10_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = RefinementSeeds()
    for transform in (ur_10_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in ur_10_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: ur_10_fk(q)
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
            for transform in (ur_10_ik_target_original_to_raw,):
                raw_target = transform(shifted)
                for raw_candidate in ur_10_ik_solve_raw(raw_target, *seed_parameters):
                    seed = raw_candidate.reshape(-1).copy()
                    seed[0] = seed[0] - th_0_offset_original2raw
                    seed[1] = seed[1] - th_1_offset_original2raw
                    seed[2] = seed[2] - th_2_offset_original2raw
                    seed[3] = seed[3] - th_3_offset_original2raw
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


def test_ik_solve_ur_10():
    theta_in = np.random.random(size=(6, ))
    ee_pose = ur_10_fk(theta_in)
    ik_output = ur_10_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = ur_10_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_ur_10()
