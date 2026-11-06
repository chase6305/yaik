import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 36
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.71
d_1: float = -0.05
d_2: float = 0.59
d_3: float = 0.1
d_4: float = 0.16
pre_transform_special_symbol_23: float = 0.22

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
def denso_cvrb_ik_target_original_to_raw(T_ee: np.ndarray):
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


def denso_cvrb_ik_target_raw_to_original(T_ee: np.ndarray):
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


def denso_cvrb_fk(theta_input: np.ndarray):
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
    ee_pose[0, 3] = d_3*x19 + x1*x24 - x11*x22 + x25*x8 + x26*(-x17 - x9)
    ee_pose[1, 0] = -1.0*x28 - 1.0*x31
    ee_pose[1, 1] = x18*x32 + x20*x33
    ee_pose[1, 2] = x18*x33 - x20*x32
    ee_pose[1, 3] = d_3*x32 + x1*x22 + x11*x24 + x25*x27 + x26*(-x28 - x31)
    ee_pose[2, 0] = -1.0*x35 - 1.0*x37
    ee_pose[2, 1] = -x18*x38 + x20*x39
    ee_pose[2, 2] = x18*x39 + x20*x38
    ee_pose[2, 3] = -d_3*x38 + 1.0*pre_transform_special_symbol_23 - x2*x23 + x25*x34 + x26*(-x35 - x37)
    return ee_pose


def denso_cvrb_twist_jacobian(theta_input: np.ndarray):
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
    x32 = -pre_transform_special_symbol_23 + x31
    x33 = -x32
    x34 = d_2*x27 + pre_transform_special_symbol_23 - x31
    x35 = a_0*x20 + d_1*x6
    x36 = d_2*x21 + x35
    x37 = d_3*x29
    x38 = d_2*x27 - x32 - x37
    x39 = d_3*x23 + x36
    x40 = d_4*x30 + x34 - x37
    x41 = d_4*x24 + x39
    x42 = a_0*x10 - d_1*x1
    x43 = d_2*x11 + x42
    x44 = d_3*x15 + x43
    x45 = d_4*x18 + x44
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
    jacobian_output[3, 2] = -x33*x6
    jacobian_output[3, 3] = -x21*x34 + x27*x36
    jacobian_output[3, 4] = -x23*x38 - x29*x39
    jacobian_output[3, 5] = -x24*x40 + x30*x41
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x33
    jacobian_output[4, 3] = x11*x34 - x27*x43
    jacobian_output[4, 4] = x15*x38 + x29*x44
    jacobian_output[4, 5] = x18*x40 - x30*x45
    jacobian_output[5, 2] = x1*x35 + x42*x6
    jacobian_output[5, 3] = -x11*x36 + x21*x43
    jacobian_output[5, 4] = -x15*x39 + x23*x44
    jacobian_output[5, 5] = -x18*x41 + x24*x45
    return jacobian_output


def denso_cvrb_angular_velocity_jacobian(theta_input: np.ndarray):
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


def denso_cvrb_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x7 = -pre_transform_special_symbol_23 + x6
    x8 = -x7
    x9 = math.sin(th_2)
    x10 = x5*x9
    x11 = math.cos(th_2)
    x12 = math.cos(th_1)
    x13 = 1.0*x12
    x14 = x10 - x11*x13
    x15 = math.sin(th_0)
    x16 = x11*x5
    x17 = x13*x9
    x18 = -x15*x16 - x15*x17
    x19 = d_2*x14 + pre_transform_special_symbol_23 - x6
    x20 = a_0*x13*x15 + d_1*x2
    x21 = d_2*x18 + x20
    x22 = math.sin(th_3)
    x23 = -x16 - x17
    x24 = x22*x23
    x25 = math.cos(th_3)
    x26 = -x10*x15 + 1.0*x11*x12*x15
    x27 = -x2*x25 - x22*x26
    x28 = d_3*x24
    x29 = d_2*x14 - x28 - x7
    x30 = d_3*x27 + x21
    x31 = math.cos(th_4)
    x32 = math.sin(th_4)
    x33 = -x14*x31 - x23*x25*x32
    x34 = -x18*x31 - x32*(-x2*x22 + x25*x26)
    x35 = d_4*x33 + x19 - x28
    x36 = d_4*x34 + x30
    x37 = 1.0*p_on_ee_x
    x38 = 1.0*x15
    x39 = p_on_ee_z*x38
    x40 = x2*x4
    x41 = x12*x2
    x42 = -x11*x40 - x41*x9
    x43 = a_0*x41 - d_1*x38
    x44 = d_2*x42 + x43
    x45 = 1.0*x1*x11*x12 - x40*x9
    x46 = 1.0*x15*x25 - x22*x45
    x47 = d_3*x46 + x44
    x48 = -x31*x42 - x32*(x22*x38 + x25*x45)
    x49 = d_4*x48 + x47
    x50 = x0*x15 + x1*x37
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x8 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x14 + p_on_ee_z*x18 + x14*x21 - x18*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x27 - x24*x30 - x27*x29
    jacobian_output[0, 5] = -p_on_ee_y*x33 + p_on_ee_z*x34 + x33*x36 - x34*x35
    jacobian_output[1, 0] = x37
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x38 + x39
    jacobian_output[1, 2] = -x38*x8 + x39
    jacobian_output[1, 3] = p_on_ee_x*x14 - p_on_ee_z*x42 - x14*x44 + x19*x42
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x46 + x22*x23*x47 + x29*x46
    jacobian_output[1, 5] = p_on_ee_x*x33 - p_on_ee_z*x48 - x33*x49 + x35*x48
    jacobian_output[2, 1] = -x50
    jacobian_output[2, 2] = 1.0*x1*x43 + 1.0*x15*x20 - x50
    jacobian_output[2, 3] = -p_on_ee_x*x18 + p_on_ee_y*x42 + x18*x44 - x21*x42
    jacobian_output[2, 4] = -p_on_ee_x*x27 + p_on_ee_y*x46 + x27*x47 - x30*x46
    jacobian_output[2, 5] = -p_on_ee_x*x34 + p_on_ee_y*x48 + x34*x49 - x36*x48
    return jacobian_output


def denso_cvrb_ik_solve_raw(T_ee: np.ndarray):
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
        return IkSolution([math.nan] * 7)

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
    for i in range(36):
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
        R_l[0, 0] = d_3*r_21
        R_l[0, 1] = d_3*r_22
        R_l[0, 2] = d_3*r_11
        R_l[0, 3] = d_3*r_12
        R_l[0, 4] = Py - d_4*r_23
        R_l[0, 5] = Px - d_4*r_13
        R_l[1, 0] = d_3*r_11
        R_l[1, 1] = d_3*r_12
        R_l[1, 2] = -d_3*r_21
        R_l[1, 3] = -d_3*r_22
        R_l[1, 4] = Px - d_4*r_13
        R_l[1, 5] = -Py + d_4*r_23
        R_l[2, 6] = d_3*r_31
        R_l[2, 7] = d_3*r_32
        R_l[3, 0] = -r_21
        R_l[3, 1] = -r_22
        R_l[3, 2] = -r_11
        R_l[3, 3] = -r_12
        R_l[4, 0] = -r_11
        R_l[4, 1] = -r_12
        R_l[4, 2] = r_21
        R_l[4, 3] = r_22
        R_l[5, 6] = 2*Px*d_3*r_11 + 2*Py*d_3*r_21 + 2*Pz*d_3*r_31 - 2*d_3*d_4*r_11*r_13 - 2*d_3*d_4*r_21*r_23 - 2*d_3*d_4*r_31*r_33
        R_l[5, 7] = 2*Px*d_3*r_12 + 2*Py*d_3*r_22 + 2*Pz*d_3*r_32 - 2*d_3*d_4*r_12*r_13 - 2*d_3*d_4*r_22*r_23 - 2*d_3*d_4*r_32*r_33
        R_l[6, 0] = -Px*r_31 + Pz*r_11 - d_4*r_11*r_33 + d_4*r_13*r_31
        R_l[6, 1] = -Px*r_32 + Pz*r_12 - d_4*r_12*r_33 + d_4*r_13*r_32
        R_l[6, 2] = Py*r_31 - Pz*r_21 + d_4*r_21*r_33 - d_4*r_23*r_31
        R_l[6, 3] = Py*r_32 - Pz*r_22 + d_4*r_22*r_33 - d_4*r_23*r_32
        R_l[7, 0] = Py*r_31 - Pz*r_21 + d_4*r_21*r_33 - d_4*r_23*r_31
        R_l[7, 1] = Py*r_32 - Pz*r_22 + d_4*r_22*r_33 - d_4*r_23*r_32
        R_l[7, 2] = Px*r_31 - Pz*r_11 + d_4*r_11*r_33 - d_4*r_13*r_31
        R_l[7, 3] = Px*r_32 - Pz*r_12 + d_4*r_12*r_33 - d_4*r_13*r_32
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
        x0 = R_l_inv_64*r_31
        x1 = R_l_inv_74*r_32
        x2 = R_l_inv_60*r_31 + R_l_inv_70*r_32
        x3 = a_0*x2
        x4 = R_l_inv_62*r_31 + R_l_inv_72*r_32
        x5 = d_2*x4
        x6 = x0 + x1 + x3 + x5
        x7 = R_l_inv_66*r_31
        x8 = R_l_inv_76*r_32
        x9 = x7 + x8
        x10 = d_2*x9
        x11 = d_1*(R_l_inv_61*r_31 + R_l_inv_71*r_32)
        x12 = d_4*r_33
        x13 = Pz - x12
        x14 = x13*x4
        x15 = R_l_inv_65*r_31 + R_l_inv_75*r_32
        x16 = Px**2
        x17 = Py**2
        x18 = Pz**2
        x19 = d_1**2
        x20 = -x19
        x21 = d_3**2
        x22 = r_11**2
        x23 = x21*x22
        x24 = r_21**2
        x25 = x21*x24
        x26 = r_31**2
        x27 = x21*x26
        x28 = d_4**2
        x29 = r_13**2*x28
        x30 = r_23**2*x28
        x31 = r_33**2*x28
        x32 = d_4*r_13
        x33 = 2*Px
        x34 = x32*x33
        x35 = d_4*r_23
        x36 = 2*Py
        x37 = x35*x36
        x38 = 2*Pz
        x39 = x12*x38
        x40 = a_0**2
        x41 = d_2**2
        x42 = -x40 - x41
        x43 = x16 + x17 + x18 + x20 + x23 + x25 + x27 + x29 + x30 + x31 - x34 - x37 - x39 + x42
        x44 = x15*x43
        x45 = x11 + x14 + x44
        x46 = x10 + x45
        x47 = 2*d_2
        x48 = x2*x47
        x49 = 2*a_0
        x50 = x4*x49
        x51 = x49*x9
        x52 = x50 + x51
        x53 = x0 + x1 - x3
        x54 = -x5
        x55 = -x10
        x56 = x45 + x55
        x57 = x54 + x56
        x58 = R_l_inv_67*r_31 + R_l_inv_77*r_32
        x59 = 2*R_l_inv_63
        x60 = 2*R_l_inv_73
        x61 = r_31*x59 + r_32*x60
        x62 = 2*d_2*x58 - x61
        x63 = 4*d_1
        x64 = x63*x9
        x65 = x64 - 4
        x66 = x47*x58 + x61
        x67 = -x0 - x1 + x3
        x68 = -x11 - x14 - x44
        x69 = Px*r_11
        x70 = Py*r_21
        x71 = Pz*r_31
        x72 = r_11*x32
        x73 = r_21*x35
        x74 = r_31*x12
        x75 = x69 + x70 + x71 - x72 - x73 - x74
        x76 = Px*r_12
        x77 = Py*r_22
        x78 = Pz*r_32
        x79 = r_12*x32
        x80 = r_22*x35
        x81 = r_32*x12
        x82 = x76 + x77 + x78 - x79 - x80 - x81
        x83 = R_l_inv_60*x75 + R_l_inv_70*x82
        x84 = a_0*x83
        x85 = R_l_inv_62*x75 + R_l_inv_72*x82
        x86 = d_2*x85
        x87 = R_l_inv_66*x75
        x88 = R_l_inv_76*x82
        x89 = x87 + x88
        x90 = d_2*x89
        x91 = x84 + x86 + x90
        x92 = d_1*(R_l_inv_61*x75 + R_l_inv_71*x82)
        x93 = x13*x85
        x94 = R_l_inv_65*x75 + R_l_inv_75*x82
        x95 = x43*x94
        x96 = R_l_inv_64*x75
        x97 = R_l_inv_74*x82
        x98 = -d_1 + x96 + x97
        x99 = -d_3*x22 - d_3*x24 - d_3*x26 + x92 + x93 + x95 + x98
        x100 = x47*x83
        x101 = x49*x85
        x102 = x49*x89
        x103 = x101 + x102
        x104 = d_3*x22
        x105 = d_3*x24
        x106 = d_3*x26
        x107 = x104 + x105 + x106 - x92 - x93 - x95
        x108 = R_l_inv_67*x75 + R_l_inv_77*x82
        x109 = -2*d_2*x108
        x110 = -x49
        x111 = x59*x75 + x60*x82
        x112 = x110 + x111
        x113 = x63*x89
        x114 = x108*x47
        x115 = x111 + x49
        x116 = -x84
        x117 = x116 + x90
        x118 = -x86
        x119 = x107 + x98
        x120 = x118 + x119
        x121 = -x90
        x122 = x121 + x84
        x123 = x119 + x86
        x124 = Px*r_21
        x125 = Py*r_11
        x126 = r_11*x35 - r_21*x32 + x124 - x125
        x127 = R_l_inv_64*x126
        x128 = Px*r_22
        x129 = Py*r_12
        x130 = r_12*x35 - r_22*x32 + x128 - x129
        x131 = R_l_inv_74*x130
        x132 = d_1*(R_l_inv_61*x126 + R_l_inv_71*x130)
        x133 = R_l_inv_62*x126 + R_l_inv_72*x130
        x134 = x13*x133
        x135 = R_l_inv_65*x126 + R_l_inv_75*x130
        x136 = x135*x43
        x137 = x127 + x131 + x132 + x134 + x136
        x138 = R_l_inv_60*x126 + R_l_inv_70*x130
        x139 = a_0*x138
        x140 = d_2*x133
        x141 = x139 + x140
        x142 = R_l_inv_66*x126 + R_l_inv_76*x130
        x143 = d_2*x142
        x144 = a_0 + x143
        x145 = -x47
        x146 = x142*x49
        x147 = x138*x47
        x148 = x133*x49
        x149 = -x147 + x148
        x150 = -a_0
        x151 = -x143
        x152 = x150 + x151
        x153 = -x139
        x154 = -x140
        x155 = x153 + x154
        x156 = R_l_inv_67*x126 + R_l_inv_77*x130
        x157 = x156*x47
        x158 = 2*d_1
        x159 = -x158
        x160 = x126*x59 + x130*x60 + x159
        x161 = -x157 + x160
        x162 = x142*x63
        x163 = -x162
        x164 = -x157 - x160
        x165 = -x127 - x131 + x132 + x134 + x136
        x166 = -x146
        x167 = 2*x35
        x168 = 2*x32
        x169 = 2*r_11
        x170 = r_23*x28
        x171 = r_13*x170
        x172 = 2*r_31
        x173 = r_33*x170
        x174 = r_21**3*x21 - r_21*x16 + r_21*x17 - r_21*x18 + r_21*x23 + r_21*x27 - r_21*x29 + r_21*x30 - r_21*x31 + r_21*x39 + x124*x168 - x125*x168 - x167*x69 - x167*x70 - x167*x71 + x169*x171 + x172*x173 + x36*x69 + x36*x71 - x36*x74
        x175 = R_l_inv_04*x174
        x176 = r_21*x21
        x177 = x169*x176
        x178 = x172*x176
        x179 = 2*x171
        x180 = 2*r_32
        x181 = r_12*x177 + r_12*x179 - r_22*x16 + r_22*x17 - r_22*x18 + r_22*x23 + 3*r_22*x25 + r_22*x27 - r_22*x29 + r_22*x30 - r_22*x31 + r_22*x39 + r_32*x178 + x128*x168 - x129*x168 - x167*x76 - x167*x77 - x167*x78 + x173*x180 + x36*x76 + x36*x78 - x36*x81
        x182 = R_l_inv_14*x181
        x183 = r_13*r_33*x28
        x184 = r_11**3*x21 + r_11*x16 - r_11*x17 - r_11*x18 + r_11*x25 + r_11*x27 + r_11*x29 - r_11*x30 - r_11*x31 + r_11*x39 + r_21*x179 - x124*x167 + x125*x167 - x168*x69 - x168*x70 - x168*x71 + x172*x183 + x33*x70 + x33*x71 - x33*x74
        x185 = R_l_inv_24*x184
        x186 = r_31*x169*x21
        x187 = r_12*x16 - r_12*x17 - r_12*x18 + 3*r_12*x23 + r_12*x25 + r_12*x27 + r_12*x29 - r_12*x30 - r_12*x31 + r_12*x39 + r_22*x177 + r_22*x179 + r_32*x186 - x128*x167 + x129*x167 - x168*x76 - x168*x77 - x168*x78 + x180*x183 + x33*x77 + x33*x78 - x33*x81
        x188 = R_l_inv_34*x187
        x189 = Py*x22 + Py*x24 + Py*x26 - x22*x35 - x24*x35 - x26*x35
        x190 = 2*d_3
        x191 = x189*x190
        x192 = Px*x22 + Px*x24 + Px*x26 - x22*x32 - x24*x32 - x26*x32
        x193 = x190*x192
        x194 = R_l_inv_06*x174 + R_l_inv_16*x181 + R_l_inv_26*x184 + R_l_inv_36*x187 + R_l_inv_46*x191 + R_l_inv_56*x193
        x195 = d_2*x194
        x196 = d_1*x49
        x197 = R_l_inv_44*x191
        x198 = R_l_inv_54*x193
        x199 = x175 + x182 + x185 + x188 + x195 + x196 + x197 + x198
        x200 = R_l_inv_00*x174 + R_l_inv_10*x181 + R_l_inv_20*x184 + R_l_inv_30*x187 + R_l_inv_40*x191 + R_l_inv_50*x193
        x201 = a_0*x200
        x202 = R_l_inv_02*x174 + R_l_inv_12*x181 + R_l_inv_22*x184 + R_l_inv_32*x187 + R_l_inv_42*x191 + R_l_inv_52*x193
        x203 = d_2*x202
        x204 = x201 + x203
        x205 = R_l_inv_01*x174 + R_l_inv_11*x181 + R_l_inv_21*x184 + R_l_inv_31*x187 + R_l_inv_41*x191 + R_l_inv_51*x193
        x206 = R_l_inv_05*x174 + R_l_inv_15*x181 + R_l_inv_25*x184 + R_l_inv_35*x187 + R_l_inv_45*x191 + R_l_inv_55*x193
        x207 = d_1*x205 + x13*x202 + x206*x43
        x208 = d_2*x63
        x209 = -x208
        x210 = x194*x49
        x211 = x200*x47
        x212 = x202*x49
        x213 = -x211 + x212
        x214 = -x201
        x215 = -x203
        x216 = x214 + x215
        x217 = -x195
        x218 = -x196
        x219 = x175 + x182 + x185 + x188 + x197 + x198 + x217 + x218
        x220 = R_l_inv_07*x174 + R_l_inv_17*x181 + R_l_inv_27*x184 + R_l_inv_37*x187 + R_l_inv_47*x191 + R_l_inv_57*x193
        x221 = x220*x47
        x222 = 2*x174
        x223 = 2*x181
        x224 = 2*x184
        x225 = 2*x187
        x226 = 4*d_3
        x227 = x189*x226
        x228 = x192*x226
        x229 = R_l_inv_03*x222 + R_l_inv_13*x223 + R_l_inv_23*x224 + R_l_inv_33*x225 + R_l_inv_43*x227 + R_l_inv_53*x228 - 2*x19 + 2*x40 - 2*x41
        x230 = -x221 + x229
        x231 = a_0*d_2
        x232 = 8*x231
        x233 = x194*x63
        x234 = -x221 - x229
        x235 = -d_1*x205 - x13*x202 - x206*x43
        x236 = -x210
        x237 = R_l_inv_40*x193
        x238 = R_l_inv_50*x191
        x239 = R_l_inv_00*x184
        x240 = R_l_inv_20*x174
        x241 = R_l_inv_10*x187
        x242 = R_l_inv_30*x181
        x243 = x237 - x238 + x239 - x240 + x241 - x242
        x244 = a_0*x243
        x245 = R_l_inv_02*x184 + R_l_inv_12*x187 - R_l_inv_22*x174 - R_l_inv_32*x181 + R_l_inv_42*x193 - R_l_inv_52*x191
        x246 = d_2*x245
        x247 = R_l_inv_46*x193
        x248 = R_l_inv_56*x191
        x249 = R_l_inv_06*x184
        x250 = R_l_inv_26*x174
        x251 = R_l_inv_16*x187
        x252 = R_l_inv_36*x181
        x253 = x247 - x248 + x249 - x250 + x251 - x252
        x254 = d_2*x253
        x255 = x244 + x246 + x254
        x256 = R_l_inv_04*x184
        x257 = R_l_inv_14*x187
        x258 = R_l_inv_24*x174
        x259 = R_l_inv_34*x181
        x260 = R_l_inv_54*x191
        x261 = R_l_inv_44*x193
        x262 = R_l_inv_01*x184 + R_l_inv_11*x187 - R_l_inv_21*x174 - R_l_inv_31*x181 + R_l_inv_41*x193 - R_l_inv_51*x191
        x263 = R_l_inv_05*x184 + R_l_inv_15*x187 - R_l_inv_25*x174 - R_l_inv_35*x181 + R_l_inv_45*x193 - R_l_inv_55*x191
        x264 = d_1*x262 + x13*x245 + x263*x43
        x265 = x19 + x256 + x257 - x258 - x259 - x260 + x261 + x264 + x42
        x266 = x243*x47
        x267 = x245*x49
        x268 = x253*x49
        x269 = x267 + x268
        x270 = x20 - x256 - x257 + x258 + x259 + x260 - x261 + x40 + x41
        x271 = R_l_inv_07*x184 + R_l_inv_17*x187 - R_l_inv_27*x174 - R_l_inv_37*x181 + R_l_inv_47*x193 - R_l_inv_57*x191
        x272 = x271*x47
        x273 = a_0*x63
        x274 = R_l_inv_23*x222
        x275 = R_l_inv_33*x223
        x276 = R_l_inv_03*x224
        x277 = R_l_inv_13*x225
        x278 = R_l_inv_53*x227
        x279 = R_l_inv_43*x228
        x280 = -x274 - x275 + x276 + x277 - x278 + x279
        x281 = x273 + x280
        x282 = x253*x63
        x283 = -x282
        x284 = -x254
        x285 = x244 + x284
        x286 = x264 + x270
        x287 = x246 + x286
        x288 = -x244
        x289 = x254 + x288
        x290 = -x246
        x291 = x286 + x290
        x292 = 2*x12
        x293 = 2*x173
        x294 = r_21*x293 + r_31**3*x21 - r_31*x16 - r_31*x17 + r_31*x18 + r_31*x23 + r_31*x25 - r_31*x29 - r_31*x30 + r_31*x31 + r_31*x34 + r_31*x37 + x169*x183 - x292*x69 - x292*x70 - x292*x71 + x38*x69 + x38*x70 - x38*x72 - x38*x73
        x295 = R_l_inv_64*x294
        x296 = 2*r_12*x183 + r_12*x186 + r_22*x178 + r_22*x293 - r_32*x16 - r_32*x17 + r_32*x18 + r_32*x23 + r_32*x25 + 3*r_32*x27 - r_32*x29 - r_32*x30 + r_32*x31 + r_32*x34 + r_32*x37 - x292*x76 - x292*x77 - x292*x78 + x38*x76 + x38*x77 - x38*x79 - x38*x80
        x297 = R_l_inv_74*x296
        x298 = R_l_inv_60*x294 + R_l_inv_70*x296
        x299 = a_0*x298
        x300 = R_l_inv_66*x294 + R_l_inv_76*x296
        x301 = d_2*x300
        x302 = d_1*x47
        x303 = x295 + x297 + x299 + x301 + x302
        x304 = R_l_inv_62*x294
        x305 = R_l_inv_72*x296
        x306 = x304 + x305
        x307 = d_2*x306
        x308 = R_l_inv_61*x294 + R_l_inv_71*x296
        x309 = R_l_inv_65*x294 + R_l_inv_75*x296
        x310 = x104*x38
        x311 = x105*x38
        x312 = x106*x38
        x313 = d_1*x308 + x104*x292 + x105*x292 + x106*x292 + x13*x306 + x309*x43 - x310 - x311 - x312
        x314 = x307 + x313
        x315 = x298*x47
        x316 = -x315
        x317 = x273 + x300*x49
        x318 = x306*x49 + x317
        x319 = -x299
        x320 = -x307
        x321 = x313 + x320
        x322 = x319 + x321
        x323 = -x301 - x302
        x324 = x295 + x297
        x325 = x323 + x324
        x326 = R_l_inv_67*x294 + R_l_inv_77*x296
        x327 = x326*x47
        x328 = -x327
        x329 = 4*x231
        x330 = x294*x59 + x296*x60
        x331 = x329 + x330
        x332 = 4*x40
        x333 = 4*x19 + x300*x63 + 4*x41
        x334 = -x332 + x333
        x335 = -x295 - x297
        x336 = x299 + x335
        x337 = -2*a_0*x306 + x317
        x338 = x301 + x302
        x339 = x15*x49
        x340 = x47*(x2 - x339)
        x341 = 4*d_2
        x342 = -x47*(x2 + x339)
        x343 = 4*a_0
        x344 = x343*x58
        x345 = 8*R_l_inv_63
        x346 = 8*R_l_inv_73
        x347 = x49*x94
        x348 = x47*(-x347 + x83)
        x349 = -x47*(x347 + x83)
        x350 = x108*x343
        x351 = -x113
        x352 = -2*a_0*x135
        x353 = x138 + 1
        x354 = x135*x49
        x355 = x156*x343
        x356 = x138 - 1
        x357 = -2*a_0*x206
        x358 = x158 + x200
        x359 = x206*x49
        x360 = x220*x343
        x361 = -x233
        x362 = 16*d_3
        x363 = R_l_inv_43*x362
        x364 = 8*R_l_inv_03
        x365 = 8*R_l_inv_13
        x366 = x159 + x200
        x367 = x263*x49
        x368 = x367 + x49
        x369 = x271*x343
        x370 = x47*(2*a_0*x309 - x298)
        x371 = x158 + x300
        x372 = x47*(x298 + x309*x49)
        x373 = x332 + x333
        x374 = x0 + x1 + x3
        x375 = x139 + x154
        x376 = a_0 + x151
        x377 = x147 + x148
        x378 = x143 + x150
        x379 = x140 + x153
        x380 = x175 + x182 + x185 + x188 + x197 + x198 + x207
        x381 = x196 + x217
        x382 = x201 + x215
        x383 = x211 + x212
        x384 = x203 + x214
        x385 = x195 + x218
        x386 = x175 + x182 + x185 + x188 + x197 + x198 + x235
        x387 = -x329 + x330
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = -x46 - x6
        A[0, 1] = x48 - x52
        A[0, 2] = -x53 - x57
        A[0, 3] = x62
        A[0, 4] = x65
        A[0, 5] = x66
        A[0, 6] = -x5 - x56 - x67
        A[0, 7] = x48 - x50 + x51
        A[0, 8] = x55 + x6 + x68
        A[1, 0] = -x91 - x99
        A[1, 1] = x100 - x103
        A[1, 2] = d_1 + x107 + x91 - x96 - x97
        A[1, 3] = -x109 - x112
        A[1, 4] = x113
        A[1, 5] = x114 + x115
        A[1, 6] = x117 + x120
        A[1, 7] = x100 - x101 + x102
        A[1, 8] = x122 + x123
        A[2, 0] = x137 + x141 + x144
        A[2, 1] = x145 + x146 + x149
        A[2, 2] = x137 + x152 + x155
        A[2, 3] = x161
        A[2, 4] = x163
        A[2, 5] = x164
        A[2, 6] = x141 + x152 + x165
        A[2, 7] = x149 + x166 + x47
        A[2, 8] = x144 + x155 + x165
        A[3, 0] = x199 + x204 + x207
        A[3, 1] = x209 + x210 + x213
        A[3, 2] = x207 + x216 + x219
        A[3, 3] = x230
        A[3, 4] = -x232 - x233
        A[3, 5] = x234
        A[3, 6] = -x199 - x216 - x235
        A[3, 7] = x208 + x213 + x236
        A[3, 8] = -x204 - x219 - x235
        A[4, 0] = x255 + x265
        A[4, 1] = -x266 + x269
        A[4, 2] = d_1*x262 + x13*x245 - x255 + x263*x43 - x270
        A[4, 3] = -x272 + x281
        A[4, 4] = x283
        A[4, 5] = -x272 + x273 - x280
        A[4, 6] = x285 + x287
        A[4, 7] = 2*a_0*x245 - x266 - x268
        A[4, 8] = x289 + x291
        A[5, 0] = x303 + x314
        A[5, 1] = x316 + x318
        A[5, 2] = x322 + x325
        A[5, 3] = x328 + x331
        A[5, 4] = -x334
        A[5, 5] = -x327 - x331
        A[5, 6] = x314 + x323 + x336
        A[5, 7] = -x315 - x337
        A[5, 8] = x322 + x335 + x338
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x340
        B[0, 1] = x341*(x4 + x9)
        B[0, 2] = x342
        B[0, 3] = -x344 + x64 - 4
        B[0, 4] = r_31*x345 + r_32*x346
        B[0, 5] = -x344 - x65
        B[0, 6] = x340
        B[0, 7] = x341*(x4 - x7 - x8)
        B[0, 8] = x342
        B[1, 0] = x348
        B[1, 1] = x341*(x85 + x89)
        B[1, 2] = x349
        B[1, 3] = -x350 - x351
        B[1, 4] = x345*x75 + x346*x82
        B[1, 5] = -x113 - x350
        B[1, 6] = x348
        B[1, 7] = x341*(x85 - x87 - x88)
        B[1, 8] = x349
        B[2, 0] = x47*(-x352 - x353)
        B[2, 1] = -x341*(x133 + x142)
        B[2, 2] = x47*(x353 + x354)
        B[2, 3] = x163 + x355
        B[2, 4] = 8*d_1 - x126*x345 - x130*x346
        B[2, 5] = x162 + x355
        B[2, 6] = x47*(-x352 - x356)
        B[2, 7] = x341*(R_l_inv_66*x126 + R_l_inv_76*x130 - x133)
        B[2, 8] = x47*(x354 + x356)
        B[3, 0] = x47*(-x357 - x358)
        B[3, 1] = -x341*(x194 + x202)
        B[3, 2] = x47*(x358 + x359)
        B[3, 3] = x360 + x361
        B[3, 4] = -8*R_l_inv_23*x184 - 8*R_l_inv_33*x187 - R_l_inv_53*x192*x362 - x174*x364 - x181*x365 - x189*x363 + 8*x19 + 8*x40 + 8*x41
        B[3, 5] = x233 + x360
        B[3, 6] = x47*(-x357 - x366)
        B[3, 7] = x341*(R_l_inv_06*x174 + R_l_inv_16*x181 + R_l_inv_26*x184 + R_l_inv_36*x187 + 2*R_l_inv_46*d_3*x189 + 2*R_l_inv_56*d_3*x192 - x202)
        B[3, 8] = x47*(x359 + x366)
        B[4, 0] = x47*(-x237 + x238 - x239 + x240 - x241 + x242 + x368)
        B[4, 1] = x341*(-x245 - x253)
        B[4, 2] = x47*(x243 + x368)
        B[4, 3] = x283 + x369
        B[4, 4] = 8*R_l_inv_23*x174 + 8*R_l_inv_33*x181 + 16*R_l_inv_53*d_3*x189 - x184*x364 - x187*x365 - x192*x363
        B[4, 5] = x282 + x369
        B[4, 6] = x47*(2*a_0*x263 - x243 - x49)
        B[4, 7] = x341*(-x245 + x247 - x248 + x249 - x250 + x251 - x252)
        B[4, 8] = x47*(x110 + x243 + x367)
        B[5, 0] = x370
        B[5, 1] = -x341*(x306 + x371)
        B[5, 2] = x372
        B[5, 3] = 4*a_0*x326 - x373
        B[5, 4] = -x294*x345 - x296*x346
        B[5, 5] = x326*x343 + x373
        B[5, 6] = x370
        B[5, 7] = x341*(-x304 - x305 + x371)
        B[5, 8] = x372
        C = np.zeros(shape=(6, 9))
        C[0, 0] = -x374 - x57
        C[0, 1] = -x48 - x52
        C[0, 2] = -x46 - x5 - x53
        C[0, 3] = x66
        C[0, 4] = -x65
        C[0, 5] = x62
        C[0, 6] = -x46 - x54 - x67
        C[0, 7] = -x48 - x50 + x51
        C[0, 8] = x10 + x374 + x54 + x68
        C[1, 0] = -x118 - x122 - x99
        C[1, 1] = -x100 - x103
        C[1, 2] = -x117 - x86 - x99
        C[1, 3] = x112 + x114
        C[1, 4] = x351
        C[1, 5] = -x109 - x115
        C[1, 6] = x116 + x121 + x123
        C[1, 7] = -x100 - x101 + x102
        C[1, 8] = x120 + x84 + x90
        C[2, 0] = x137 + x375 + x376
        C[2, 1] = x146 + x377 + x47
        C[2, 2] = x137 + x378 + x379
        C[2, 3] = x164
        C[2, 4] = x162
        C[2, 5] = x161
        C[2, 6] = x165 + x375 + x378
        C[2, 7] = x145 + x166 + x377
        C[2, 8] = x165 + x376 + x379
        C[3, 0] = x380 + x381 + x382
        C[3, 1] = x208 + x210 + x383
        C[3, 2] = x380 + x384 + x385
        C[3, 3] = x234
        C[3, 4] = -x232 - x361
        C[3, 5] = x230
        C[3, 6] = -x381 - x384 - x386
        C[3, 7] = x209 + x236 + x383
        C[3, 8] = -x382 - x385 - x386
        C[4, 0] = x265 + x285 + x290
        C[4, 1] = x266 + x269
        C[4, 2] = x246 + x265 + x289
        C[4, 3] = -x272 - x281
        C[4, 4] = x282
        C[4, 5] = -x272 - x273 - x274 - x275 + x276 + x277 - x278 + x279
        C[4, 6] = x244 + x254 + x291
        C[4, 7] = x266 + x267 - x268
        C[4, 8] = x284 + x287 + x288
        C[5, 0] = x299 + x321 + x325
        C[5, 1] = x315 + x318
        C[5, 2] = x314 + x319 + x324 + x338
        C[5, 3] = -x327 - x387
        C[5, 4] = x334
        C[5, 5] = x328 + x387
        C[5, 6] = x321 + x336 + x338
        C[5, 7] = -x316 - x337
        C[5, 8] = d_1*x308 + 2*d_3*d_4*r_33*x22 + 2*d_3*d_4*r_33*x24 + 2*d_3*d_4*r_33*x26 + x13*x306 - x303 + x309*x43 - x310 - x311 - x312 - x320
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[3] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor()
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
            th_2 = this_solution[3]
            condition_0: bool = (((2*abs(d_1*d_3) >= zero_tolerance)) or ((2*abs(a_0*d_3*math.cos(th_2)) >= zero_tolerance)) or ((abs(Px**2 - 2*Px*d_4*r_13 + Py**2 - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*d_4*r_33 - a_0**2 + 2*a_0*d_2*math.sin(th_2) - d_1**2 - d_2**2 - d_3**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_2)
                x1 = 2*d_3
                x2 = math.atan2(-a_0*x0*x1, -d_1*x1)
                x3 = d_1**2
                x4 = d_3**2
                x5 = 4*x4
                x6 = a_0**2
                x7 = 2*d_4
                x8 = d_4**2
                x9 = Px**2 - Px*r_13*x7 + Py**2 - Py*r_23*x7 + Pz**2 - Pz*r_33*x7 + 2*a_0*d_2*math.sin(th_2) - d_2**2 + r_13**2*x8 + r_23**2*x8 + r_33**2*x8 - x3 - x4 - x6
                x10 = safe_sqrt(x0**2*x5*x6 + x3*x5 - x9**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2 + math.atan2(x10, x9)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[4] = x2 + math.atan2(-x10, x9)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
    # Finish code for explicit solution node 2

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
            th_3 = this_solution[4]
            checked_result: bool = (((abs(Px - d_4*r_13) <= 1.0e-6)) and ((abs(Py - d_4*r_23) <= 1.0e-6)) and ((abs(d_1 - d_3*math.cos(th_3)) <= 1.0e-6)))
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
            th_3 = this_solution[4]
            condition_0: bool = (((abs(Px - d_4*r_13) >= zero_tolerance)) or ((abs(Py - d_4*r_23) >= zero_tolerance)) or ((abs(d_1 - d_3*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = Px - d_4*r_13
                x1 = -Py + d_4*r_23
                x2 = math.atan2(x0, x1)
                x3 = -d_1 + d_3*math.cos(th_3)
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                this_solution[0] = x2 + math.atan2(-x4, x3)
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
            th_3 = this_solution[4]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(12, branch_input_index)

            th_3 = this_solution[4]
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
                this_solution[2] = math.atan2(r_13*math.cos(th_0) + r_23*math.sin(th_0), r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 17

    # Code for non-branch dispatcher node 18
    # Actually, there is no code

    # Code for explicit solution node 19, solved variable is th_5
    def ExplicitSolutionNode_node_19_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return

        # The solution of non-root node 19
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
                this_solution[6] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_5_processor()
    # Finish code for explicit solution node 18

    # Code for solved_variable dispatcher node 20
    def SolvedVariableDispatcherNode_node_20_processor():
        this_node_input_index: List[int] = node_input_index[20]
        this_input_valid: bool = node_input_validity[20]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            forwarded_to_child = False
            th_2 = this_solution[3]
            degenerate_valid_0 = (abs(th_2 - 1/2*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(30, branch_input_index)

            th_2 = this_solution[3]
            degenerate_valid_1 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(33, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(21, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_20_processor()
    # Finish code for solved_variable dispatcher node 20

    # Code for explicit solution node 33, solved variable is th_1
    def ExplicitSolutionNode_node_33_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[33]
        this_input_valid: bool = node_input_validity[33]
        if not this_input_valid:
            return

        # The solution of non-root node 33
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(a_0 + d_2) >= 1.0e-6)) or ((abs(d_3*math.sin(th_3)) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 + d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                this_solution[1] = math.atan2(-x0*x1 - x4*x5, x0*x4 - x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(34, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_1_processor()
    # Finish code for explicit solution node 33

    # Code for non-branch dispatcher node 34
    # Actually, there is no code

    # Code for explicit solution node 35, solved variable is th_4
    def ExplicitSolutionNode_node_35_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[34]
        this_input_valid: bool = node_input_validity[34]
        if not this_input_valid:
            return

        # The solution of non-root node 35
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_0)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                this_solution[5] = math.atan2(-r_13*(x2*x3 + x4*x6) - r_23*(x2*x6 - x3*x4) - r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_4_processor()
    # Finish code for explicit solution node 34

    # Code for explicit solution node 30, solved variable is th_1
    def ExplicitSolutionNode_node_30_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return

        # The solution of non-root node 30
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(d_3*math.sin(th_3)) >= 1.0e-6)) or ((abs(a_0 - d_2) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 - d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(31, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_30_solve_th_1_processor()
    # Finish code for explicit solution node 30

    # Code for non-branch dispatcher node 31
    # Actually, there is no code

    # Code for explicit solution node 32, solved variable is th_4
    def ExplicitSolutionNode_node_32_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[31]
        this_input_valid: bool = node_input_validity[31]
        if not this_input_valid:
            return

        # The solution of non-root node 32
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)) or ((abs(-r_13*(math.sin(th_0)*math.sin(th_3) - math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_3)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                this_solution[5] = math.atan2(-r_13*(x2*x4 - x3*x6) + r_23*(x2*x3 + x4*x6) + r_33*x0*x1, r_13*x0*x3 + r_23*x0*x4 - r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_4_processor()
    # Finish code for explicit solution node 31

    # Code for explicit solution node 21, solved variable is th_1
    def ExplicitSolutionNode_node_21_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[21]
        this_input_valid: bool = node_input_validity[21]
        if not this_input_valid:
            return

        # The solution of non-root node 21
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1th_2th_4_soa = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (((abs(d_2*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 - d_2*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_4*r_33
                x1 = a_0 - d_2*math.sin(th_2)
                x2 = Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*math.sin(th_1th_2th_4_soa)
                x3 = d_2*math.cos(th_2)
                this_solution[1] = math.atan2(-x0*x1 - x2*x3, -x0*x3 + x1*x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_1_processor()
    # Finish code for explicit solution node 21

    # Code for non-branch dispatcher node 22
    # Actually, there is no code

    # Code for explicit solution node 23, solved variable is th_4
    def ExplicitSolutionNode_node_23_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return

        # The solution of non-root node 23
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_1th_2th_4_soa = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[5] = -th_1 + th_1th_2th_4_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_4_processor()
    # Finish code for explicit solution node 22

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
                this_solution[6] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
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
            forwarded_to_child = False
            th_2 = this_solution[3]
            degenerate_valid_0 = (abs(th_2 - 1/2*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(24, branch_input_index)

            th_2 = this_solution[3]
            degenerate_valid_1 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(27, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(14, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor()
    # Finish code for solved_variable dispatcher node 13

    # Code for explicit solution node 27, solved variable is th_1
    def ExplicitSolutionNode_node_27_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(a_0 + d_2) >= 1.0e-6)) or ((abs(d_3*math.sin(th_3)) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 + d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                this_solution[1] = math.atan2(-x0*x1 - x4*x5, x0*x4 - x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_1_processor()
    # Finish code for explicit solution node 27

    # Code for non-branch dispatcher node 28
    # Actually, there is no code

    # Code for explicit solution node 29, solved variable is th_4
    def ExplicitSolutionNode_node_29_solve_th_4_processor():
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
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_0)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                this_solution[5] = math.atan2(-r_13*(x2*x3 + x4*x6) - r_23*(x2*x6 - x3*x4) - r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_4_processor()
    # Finish code for explicit solution node 28

    # Code for explicit solution node 24, solved variable is th_1
    def ExplicitSolutionNode_node_24_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[24]
        this_input_valid: bool = node_input_validity[24]
        if not this_input_valid:
            return

        # The solution of non-root node 24
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(d_3*math.sin(th_3)) >= 1.0e-6)) or ((abs(a_0 - d_2) >= 1.0e-6)))
            if condition_0:
                x0 = a_0 - d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(25, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_1_processor()
    # Finish code for explicit solution node 24

    # Code for non-branch dispatcher node 25
    # Actually, there is no code

    # Code for explicit solution node 26, solved variable is th_4
    def ExplicitSolutionNode_node_26_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[25]
        this_input_valid: bool = node_input_validity[25]
        if not this_input_valid:
            return

        # The solution of non-root node 26
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)) or ((abs(-r_13*(math.sin(th_0)*math.sin(th_3) - math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_3)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                this_solution[5] = math.atan2(-r_13*(x2*x4 - x3*x6) + r_23*(x2*x3 + x4*x6) + r_33*x0*x1, r_13*x0*x3 + r_23*x0*x4 - r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_4_processor()
    # Finish code for explicit solution node 25

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
            th_2 = this_solution[3]
            condition_0: bool = (((abs(d_2*math.cos(th_2)) >= 1.0e-6)) or ((abs(a_0 - d_2*math.sin(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_4*r_33
                x1 = a_0 - d_2*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_2*math.cos(th_2)
                this_solution[1] = math.atan2(-x0*x1 - x4*x5, -x0*x5 + x1*x4)
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
            th_2 = this_solution[3]
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
                this_solution[5] = math.atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7)
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
            th_3 = this_solution[4]
            condition_0: bool = (((math.sin(th_3) != 0)) and ((abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                this_solution[5] = math.pi - x0
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
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            checked_result: bool = (((abs(d_2*math.cos(th_2) - d_3*math.sin(th_2)*math.sin(th_3)) <= 1.0e-6)) and ((abs(-a_0 + d_2*math.sin(th_2) + d_3*math.sin(th_3)*math.cos(th_2)) <= 1.0e-6)))
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
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(d_2*math.cos(th_2) - d_3*math.sin(th_2)*math.sin(th_3)) >= 1.0e-6)) or ((abs(-a_0 + d_2*math.sin(th_2) + d_3*math.sin(th_3)*math.cos(th_2)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_4*r_33
                x1 = math.sin(th_2)
                x2 = math.cos(th_2)
                x3 = math.sin(th_3)
                x4 = a_0 - d_2*x1 - d_3*x2*x3
                x5 = -d_2*x2 + d_3*x1*x3
                x6 = math.cos(th_0)
                x7 = math.sin(th_0)
                x8 = Px*x6 + Py*x7 - d_4*r_13*x6 - d_4*r_23*x7
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
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*((math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) + r_21*((math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) - math.cos(th_0)*math.cos(th_3)) + r_31*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance)) or ((abs(r_12*((math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) + r_22*((math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) - math.cos(th_0)*math.cos(th_3)) + r_32*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_2)
                x3 = math.sin(th_2)
                x4 = math.cos(th_1)
                x5 = x0*(x1*x2 + x3*x4)
                x6 = math.sin(th_0)
                x7 = math.cos(th_3)
                x8 = math.cos(th_0)
                x9 = x1*x3 - x2*x4
                x10 = x0*x8*x9 + x6*x7
                x11 = -x0*x6*x9 + x7*x8
                this_solution[6] = math.atan2(-r_11*x10 + r_21*x11 - r_31*x5, -r_12*x10 + r_22*x11 - r_32*x5)
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
        value_at_2 = ik_out_i[3]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[4]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[5]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[6]  # th_5
        new_ik_i[5] = value_at_5
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def denso_cvrb_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = denso_cvrb_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = ik_out_i[5] - th_5_offset_original2raw
        ee_pose_i = denso_cvrb_fk(ik_out_i)
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
            snapped_pose = denso_cvrb_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
                if _yaik_seed_candidates is not None and np.max(np.abs(snapped_pose-T_ee)) < 1e-7:
                    _yaik_seed_candidates.strict_witness = True
                    _yaik_seed_candidates.clear()
    return ik_output


def denso_cvrb_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = RefinementSeeds()
    for transform in (denso_cvrb_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in denso_cvrb_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: denso_cvrb_fk(q)
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
            for transform in (denso_cvrb_ik_target_original_to_raw,):
                raw_target = transform(shifted)
                for raw_candidate in denso_cvrb_ik_solve_raw(raw_target, *seed_parameters):
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


def test_ik_solve_denso_cvrb():
    theta_in = np.random.random(size=(6, ))
    ee_pose = denso_cvrb_fk(theta_in)
    ik_output = denso_cvrb_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = denso_cvrb_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_denso_cvrb()
