import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 12
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.018
a_2: float = 0.018
d_0: float = 0.287
d_3: float = 0.314
post_transform_special_symbol_13: float = -0.095
pre_transform_special_symbol_03: float = 0.2215
pre_transform_special_symbol_13: float = 2.9142e-05
pre_transform_special_symbol_23: float = 1.121

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -3.141592653589793
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 0.0
th_5_offset_original2raw: float = 1.5707963267948966
th_6_offset_original2raw: float = -1.5707963267948966


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def marvin_m6_left_arm_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -r_22
    ee_transformed[0, 1] = -r_23
    ee_transformed[0, 2] = r_21
    ee_transformed[0, 3] = Py + post_transform_special_symbol_13*r_23 - pre_transform_special_symbol_13
    ee_transformed[1, 0] = -r_32
    ee_transformed[1, 1] = -r_33
    ee_transformed[1, 2] = r_31
    ee_transformed[1, 3] = Pz + post_transform_special_symbol_13*r_33 - pre_transform_special_symbol_23
    ee_transformed[2, 0] = -r_12
    ee_transformed[2, 1] = -r_13
    ee_transformed[2, 2] = r_11
    ee_transformed[2, 3] = Px + post_transform_special_symbol_13*r_13 - pre_transform_special_symbol_03
    return ee_transformed


def marvin_m6_left_arm_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_33
    ee_transformed[0, 1] = -r_31
    ee_transformed[0, 2] = -r_32
    ee_transformed[0, 3] = Pz + post_transform_special_symbol_13*r_32 + pre_transform_special_symbol_03
    ee_transformed[1, 0] = r_13
    ee_transformed[1, 1] = -r_11
    ee_transformed[1, 2] = -r_12
    ee_transformed[1, 3] = Px + post_transform_special_symbol_13*r_12 + pre_transform_special_symbol_13
    ee_transformed[2, 0] = r_23
    ee_transformed[2, 1] = -r_21
    ee_transformed[2, 2] = -r_22
    ee_transformed[2, 3] = Py + post_transform_special_symbol_13*r_22 + pre_transform_special_symbol_23
    return ee_transformed


def marvin_m6_left_arm_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_5)
    x1 = math.cos(th_1)
    x2 = math.cos(th_3)
    x3 = math.sin(th_3)
    x4 = math.cos(th_2)
    x5 = math.sin(th_1)
    x6 = x4*x5
    x7 = -x1*x2 + x3*x6
    x8 = -x7
    x9 = math.sin(th_5)
    x10 = math.sin(th_4)
    x11 = math.sin(th_2)
    x12 = x11*x5
    x13 = math.cos(th_4)
    x14 = -x1*x3 - x2*x6
    x15 = -x10*x12 + x13*x14
    x16 = math.sin(th_6)
    x17 = -x10*x14 - x12*x13
    x18 = math.cos(th_6)
    x19 = x0*x15 - x8*x9
    x20 = x16*x19 + x17*x18
    x21 = math.cos(th_0)
    x22 = x21*x5
    x23 = math.sin(th_0)
    x24 = x11*x23
    x25 = x21*x4
    x26 = x1*x25 + x24
    x27 = x2*x22 + x26*x3
    x28 = x23*x4
    x29 = x1*x11*x21 - x28
    x30 = x2*x26 - x22*x3
    x31 = x10*x29 + x13*x30
    x32 = -x10*x30 + x13*x29
    x33 = x0*x31 - x27*x9
    x34 = x16*x33 + x18*x32
    x35 = x23*x5
    x36 = x1*x28 - x11*x21
    x37 = x2*x35 + x3*x36
    x38 = x1*x24 + x25
    x39 = x2*x36 - x3*x35
    x40 = x10*x38 + x13*x39
    x41 = -x10*x39 + x13*x38
    x42 = x0*x40 - x37*x9
    x43 = x16*x42 + x18*x41
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x8 - x15*x9
    ee_pose[0, 1] = x16*x17 - x18*x19
    ee_pose[0, 2] = x20
    ee_pose[0, 3] = -a_1*x6 + a_2*x14 - d_0*x1 - d_3*x7 - post_transform_special_symbol_13*x20 + pre_transform_special_symbol_03
    ee_pose[1, 0] = -x0*x27 - x31*x9
    ee_pose[1, 1] = x16*x32 - x18*x33
    ee_pose[1, 2] = x34
    ee_pose[1, 3] = a_1*x26 + a_2*x30 - d_0*x22 + d_3*x27 - post_transform_special_symbol_13*x34 + pre_transform_special_symbol_13
    ee_pose[2, 0] = -x0*x37 - x40*x9
    ee_pose[2, 1] = x16*x41 - x18*x42
    ee_pose[2, 2] = x43
    ee_pose[2, 3] = a_1*x36 + a_2*x39 - d_0*x35 + d_3*x37 - post_transform_special_symbol_13*x43 + pre_transform_special_symbol_23
    return ee_pose


def marvin_m6_left_arm_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_1)
    x1 = math.sin(th_1)
    x2 = math.sin(th_2)
    x3 = x1*x2
    x4 = math.cos(th_3)
    x5 = math.sin(th_3)
    x6 = math.cos(th_2)
    x7 = x1*x6
    x8 = -x0*x4 + x5*x7
    x9 = -x8
    x10 = math.cos(th_4)
    x11 = math.sin(th_4)
    x12 = -x0*x5 - x4*x7
    x13 = -x10*x3 - x11*x12
    x14 = math.cos(th_5)
    x15 = math.sin(th_5)
    x16 = -x14*x9 - x15*(x10*x12 - x11*x3)
    x17 = math.sin(th_0)
    x18 = math.cos(th_0)
    x19 = x1*x18
    x20 = x17*x6
    x21 = x0*x18*x2 - x20
    x22 = x17*x2
    x23 = x18*x6
    x24 = x0*x23 + x22
    x25 = x19*x4 + x24*x5
    x26 = -x19*x5 + x24*x4
    x27 = x10*x21 - x11*x26
    x28 = -x14*x25 - x15*(x10*x26 + x11*x21)
    x29 = x1*x17
    x30 = x0*x22 + x23
    x31 = x0*x20 - x18*x2
    x32 = x29*x4 + x31*x5
    x33 = -x29*x5 + x31*x4
    x34 = x10*x30 - x11*x33
    x35 = -x14*x32 - x15*(x10*x33 + x11*x30)
    x36 = d_0*x29
    x37 = pre_transform_special_symbol_23 - x36
    x38 = -pre_transform_special_symbol_13
    x39 = d_0*x19
    x40 = -x38 - x39
    x41 = a_1*x24 + pre_transform_special_symbol_13 - x39
    x42 = a_1*x31 + pre_transform_special_symbol_23 - x36
    x43 = a_2*x26 + d_3*x25 + x41
    x44 = a_2*x33 + d_3*x32 + x42
    x45 = d_0*x0 - pre_transform_special_symbol_03
    x46 = -x45
    x47 = a_1*x7 + x45
    x48 = -x47
    x49 = a_2*x12 - d_3*x8 - x47
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 0] = 1
    jacobian_output[0, 2] = -x0
    jacobian_output[0, 3] = -x3
    jacobian_output[0, 4] = x9
    jacobian_output[0, 5] = x13
    jacobian_output[0, 6] = x16
    jacobian_output[1, 1] = -x17
    jacobian_output[1, 2] = -x19
    jacobian_output[1, 3] = x21
    jacobian_output[1, 4] = x25
    jacobian_output[1, 5] = x27
    jacobian_output[1, 6] = x28
    jacobian_output[2, 1] = x18
    jacobian_output[2, 2] = -x29
    jacobian_output[2, 3] = x30
    jacobian_output[2, 4] = x32
    jacobian_output[2, 5] = x34
    jacobian_output[2, 6] = x35
    jacobian_output[3, 1] = pre_transform_special_symbol_13*x18 + pre_transform_special_symbol_23*x17
    jacobian_output[3, 2] = x19*x37 - x29*x40
    jacobian_output[3, 3] = -x21*x42 + x30*x41
    jacobian_output[3, 4] = -x25*x44 + x32*x43
    jacobian_output[3, 5] = -x27*x44 + x34*x43
    jacobian_output[3, 6] = -x28*x44 + x35*x43
    jacobian_output[4, 0] = pre_transform_special_symbol_23
    jacobian_output[4, 1] = -pre_transform_special_symbol_03*x18
    jacobian_output[4, 2] = -x0*x37 + x29*x46
    jacobian_output[4, 3] = -x3*x42 - x30*x48
    jacobian_output[4, 4] = -x32*x49 + x44*x9
    jacobian_output[4, 5] = x13*x44 - x34*x49
    jacobian_output[4, 6] = x16*x44 - x35*x49
    jacobian_output[5, 0] = x38
    jacobian_output[5, 1] = -pre_transform_special_symbol_03*x17
    jacobian_output[5, 2] = x0*x40 - x19*x46
    jacobian_output[5, 3] = x21*x48 + x3*x41
    jacobian_output[5, 4] = x25*x49 - x43*x9
    jacobian_output[5, 5] = -x13*x43 + x27*x49
    jacobian_output[5, 6] = -x16*x43 + x28*x49
    return jacobian_output


def marvin_m6_left_arm_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_1)
    x1 = math.sin(th_1)
    x2 = math.sin(th_2)
    x3 = x1*x2
    x4 = math.cos(th_3)
    x5 = math.sin(th_3)
    x6 = math.cos(th_2)
    x7 = x1*x6
    x8 = x0*x4 - x5*x7
    x9 = math.cos(th_4)
    x10 = math.sin(th_4)
    x11 = -x0*x5 - x4*x7
    x12 = math.cos(th_5)
    x13 = math.sin(th_5)
    x14 = math.sin(th_0)
    x15 = math.cos(th_0)
    x16 = x1*x15
    x17 = x14*x6
    x18 = x0*x15*x2 - x17
    x19 = x14*x2
    x20 = x15*x6
    x21 = x0*x20 + x19
    x22 = x16*x4 + x21*x5
    x23 = -x16*x5 + x21*x4
    x24 = x1*x14
    x25 = x0*x19 + x20
    x26 = x0*x17 - x15*x2
    x27 = x24*x4 + x26*x5
    x28 = -x24*x5 + x26*x4
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = 1
    jacobian_output[0, 2] = -x0
    jacobian_output[0, 3] = -x3
    jacobian_output[0, 4] = x8
    jacobian_output[0, 5] = -x10*x11 - x3*x9
    jacobian_output[0, 6] = -x12*x8 - x13*(-x10*x3 + x11*x9)
    jacobian_output[1, 1] = -x14
    jacobian_output[1, 2] = -x16
    jacobian_output[1, 3] = x18
    jacobian_output[1, 4] = x22
    jacobian_output[1, 5] = -x10*x23 + x18*x9
    jacobian_output[1, 6] = -x12*x22 - x13*(x10*x18 + x23*x9)
    jacobian_output[2, 1] = x15
    jacobian_output[2, 2] = -x24
    jacobian_output[2, 3] = x25
    jacobian_output[2, 4] = x27
    jacobian_output[2, 5] = -x10*x28 + x25*x9
    jacobian_output[2, 6] = -x12*x27 - x13*(x10*x25 + x28*x9)
    return jacobian_output


def marvin_m6_left_arm_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = x0*x2
    x5 = -pre_transform_special_symbol_23
    x6 = d_0*x3
    x7 = -x5 - x6
    x8 = -pre_transform_special_symbol_13
    x9 = d_0*x4
    x10 = -x8 - x9
    x11 = math.cos(th_2)
    x12 = x0*x11
    x13 = math.cos(th_1)
    x14 = math.sin(th_2)
    x15 = x1*x14
    x16 = x12 + x13*x15
    x17 = x1*x11
    x18 = x0*x13*x14 - x17
    x19 = x12*x13 + x15
    x20 = a_1*x19 + pre_transform_special_symbol_13 - x9
    x21 = -x0*x14 + x13*x17
    x22 = a_1*x21 + pre_transform_special_symbol_23 - x6
    x23 = math.cos(th_3)
    x24 = math.sin(th_3)
    x25 = x21*x24 + x23*x3
    x26 = x19*x24 + x23*x4
    x27 = x19*x23 - x24*x4
    x28 = a_2*x27 + d_3*x26 + x20
    x29 = x21*x23 - x24*x3
    x30 = a_2*x29 + d_3*x25 + x22
    x31 = math.cos(th_4)
    x32 = math.sin(th_4)
    x33 = x16*x31 - x29*x32
    x34 = x18*x31 - x27*x32
    x35 = math.cos(th_5)
    x36 = math.sin(th_5)
    x37 = -x25*x35 - x36*(x16*x32 + x29*x31)
    x38 = -x26*x35 - x36*(x18*x32 + x27*x31)
    x39 = p_on_ee_x*x0
    x40 = p_on_ee_x*x1
    x41 = d_0*x13 - pre_transform_special_symbol_03
    x42 = -x41
    x43 = x14*x2
    x44 = x11*x2
    x45 = a_1*x44 + x41
    x46 = -x45
    x47 = -x13*x23 + x24*x44
    x48 = -x47
    x49 = -x13*x24 - x23*x44
    x50 = a_2*x49 - d_3*x47 - x45
    x51 = -x31*x43 - x32*x49
    x52 = -x35*x48 - x36*(x31*x49 - x32*x43)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = -p_on_ee_y*x0 - p_on_ee_z*x1 + pre_transform_special_symbol_13*x0 + pre_transform_special_symbol_23*x1
    jacobian_output[0, 2] = p_on_ee_y*x3 - p_on_ee_z*x4 - x10*x3 + x4*x7
    jacobian_output[0, 3] = -p_on_ee_y*x16 + p_on_ee_z*x18 + x16*x20 - x18*x22
    jacobian_output[0, 4] = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x28 - x26*x30
    jacobian_output[0, 5] = -p_on_ee_y*x33 + p_on_ee_z*x34 + x28*x33 - x30*x34
    jacobian_output[0, 6] = -p_on_ee_y*x37 + p_on_ee_z*x38 + x28*x37 - x30*x38
    jacobian_output[1, 0] = -p_on_ee_z - x5
    jacobian_output[1, 1] = -pre_transform_special_symbol_03*x0 + x39
    jacobian_output[1, 2] = p_on_ee_z*x13 + x1*x2*x42 - x13*x7 - x2*x40
    jacobian_output[1, 3] = p_on_ee_x*x16 + p_on_ee_z*x43 - x16*x46 - x22*x43
    jacobian_output[1, 4] = p_on_ee_x*x25 - p_on_ee_z*x48 - x25*x50 + x30*x48
    jacobian_output[1, 5] = p_on_ee_x*x33 - p_on_ee_z*x51 + x30*x51 - x33*x50
    jacobian_output[1, 6] = p_on_ee_x*x37 - p_on_ee_z*x52 + x30*x52 - x37*x50
    jacobian_output[2, 0] = p_on_ee_y + x8
    jacobian_output[2, 1] = -pre_transform_special_symbol_03*x1 + x40
    jacobian_output[2, 2] = -p_on_ee_y*x13 + x10*x13 + x2*x39 - x4*x42
    jacobian_output[2, 3] = -p_on_ee_x*x18 - p_on_ee_y*x43 + x14*x2*x20 + x18*x46
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x48 + x26*x50 - x28*x48
    jacobian_output[2, 5] = -p_on_ee_x*x34 + p_on_ee_y*x51 - x28*x51 + x34*x50
    jacobian_output[2, 6] = -p_on_ee_x*x38 + p_on_ee_y*x52 - x28*x52 + x38*x50
    return jacobian_output


def marvin_m6_left_arm_ik_solve_raw(T_ee: np.ndarray, th_4):
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
        return IkSolution([100000.0] * 6)

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
    for i in range(12):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for non-branch dispatcher node 0
    # Actually, there is no code

    # Code for explicit solution node 1, solved variable is th_3
    def ExplicitSolutionNode_node_1_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The explicit solution of root node
        condition_0: bool = (abs(2*a_1*a_2 - 2*d_0*d_3) >= zero_tolerance) or (abs(2*a_1*d_3 + 2*a_2*d_0) >= zero_tolerance) or (abs(a_1**2 + a_2**2 + d_0**2 + d_3**2 - inv_Px**2 - inv_Py**2 - inv_Pz**2) >= zero_tolerance)
        if condition_0:
            x0 = 2*d_3
            x1 = 2*a_2
            x2 = a_1*x0 + d_0*x1
            x3 = a_1*x1 - d_0*x0
            x4 = math.atan2(x2, x3)
            x5 = -a_1**2 - a_2**2 - d_0**2 - d_3**2 + inv_Px**2 + inv_Py**2 + inv_Pz**2
            x6 = safe_sqrt(x2**2 + x3**2 - x5**2)
            solution_0: IkSolution = make_ik_solution()
            solution_0[3] = x4 + math.atan2(x6, x5)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)

            solution_1: IkSolution = make_ik_solution()
            solution_1[3] = x4 + math.atan2(-x6, x5)
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)

    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor()
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
            th_3 = this_solution[3]
            checked_result: bool = (abs(inv_Px) <= 1.0e-6) and (abs(inv_Py) <= 1.0e-6) and (abs((a_1*math.cos(th_3) + a_2 + d_0*math.sin(th_3))*math.sin(th_4)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2

    # Code for explicit solution node 3, solved variable is th_6
    def ExplicitSolutionNode_node_3_solve_th_6_processor():
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
            th_3 = this_solution[3]
            condition_0: bool = (abs(inv_Px) >= zero_tolerance) or (abs(inv_Py) >= zero_tolerance) or (abs((a_1*math.cos(th_3) + a_2 + d_0*math.sin(th_3))*math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                x0 = math.atan2(inv_Px, inv_Py)
                x1 = math.sin(th_4)
                x2 = a_1*math.cos(th_3) + a_2 + d_0*math.sin(th_3)
                x3 = safe_sqrt(inv_Px**2 + inv_Py**2 - x1**2*x2**2)
                x4 = -x1*x2
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x0 + math.atan2(x3, x4)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[5] = x0 + math.atan2(-x3, x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_6_processor()
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
            th_6 = this_solution[5]
            checked_result: bool = (abs(inv_Pz) <= 1.0e-6) and (abs(inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4

    # Code for explicit solution node 5, solved variable is th_5
    def ExplicitSolutionNode_node_5_solve_th_5_processor():
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
            th_6 = this_solution[5]
            condition_0: bool = (abs(inv_Pz) >= 1.0e-6) or (abs(inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) >= 1.0e-6)
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_3)
                x2 = (a_1*x0 + a_2 + d_0*x1)*math.cos(th_4)
                x3 = -inv_Px*math.cos(th_6) + inv_Py*math.sin(th_6)
                x4 = a_1*x1 - d_0*x0 + d_3
                this_solution[4] = math.atan2(inv_Pz*x2 - x3*x4, inv_Pz*x4 + x2*x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_5_processor()
    # Finish code for explicit solution node 5

    # Code for non-branch dispatcher node 6
    # Actually, there is no code

    # Code for linear solution type2 node 7, solved variable is th_2
    def LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            # Temp variable for efficiency
            x0 = math.sin(th_6)
            x1 = math.sin(th_4)
            x2 = math.cos(th_3)
            x3 = x1*x2
            x4 = math.sin(th_3)
            x5 = math.sin(th_5)
            x6 = math.cos(th_6)
            x7 = math.cos(th_4)
            x8 = x6*x7
            x9 = math.cos(th_5)
            x10 = x0*x3 + x2*x8*x9 - x4*x5*x6
            x11 = x1*x9
            x12 = -x0*x7 + x11*x6
            x13 = -x0*x2*x7*x9 + x0*x4*x5 + x3*x6
            x14 = x0*x11 + x8
            x15 = x2*x5*x7 + x4*x9
            x16 = x1*x5
            x17 = a_1 + a_2*x2 + d_3*x4
            # End of temp variables
            A_matrix = np.zeros(shape=(8, 4))
            A_matrix[0, 0] = r_11
            A_matrix[0, 1] = -r_21
            A_matrix[0, 2] = -x10
            A_matrix[0, 3] = x12
            A_matrix[1, 0] = r_12
            A_matrix[1, 1] = -r_22
            A_matrix[1, 2] = -x13
            A_matrix[1, 3] = -x14
            A_matrix[2, 0] = r_13
            A_matrix[2, 1] = -r_23
            A_matrix[2, 2] = x15
            A_matrix[2, 3] = -x16
            A_matrix[3, 0] = Px
            A_matrix[3, 1] = -Py
            A_matrix[3, 2] = -x17
            A_matrix[4, 0] = -r_11
            A_matrix[4, 1] = r_21
            A_matrix[4, 2] = x10
            A_matrix[4, 3] = -x12
            A_matrix[5, 0] = -r_12
            A_matrix[5, 1] = r_22
            A_matrix[5, 2] = x13
            A_matrix[5, 3] = x14
            A_matrix[6, 0] = -r_13
            A_matrix[6, 1] = r_23
            A_matrix[6, 2] = -x15
            A_matrix[6, 3] = x16
            A_matrix[7, 0] = -Px
            A_matrix[7, 1] = Py
            A_matrix[7, 2] = x17
            solution_tuple_0 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 2)
            if solution_tuple_0 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_0[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)

                this_solution[2] = solution_tuple_0[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue

            solution_tuple_1 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 3)
            if solution_tuple_1 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_1[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)

                this_solution[2] = solution_tuple_1[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue

            solution_tuple_2 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 6)
            if solution_tuple_2 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_2[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)

                this_solution[2] = solution_tuple_2[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue

            solution_tuple_3 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 7)
            if solution_tuple_3 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_3[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)

                this_solution[2] = solution_tuple_3[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue

            solution_tuple_4 = try_solve_linear_type2_specific_rows(A_matrix, 0, 2, 3)
            if solution_tuple_4 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_4[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)

                this_solution[2] = solution_tuple_4[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue

    # Invoke the processor
    LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor()
    # Finish code for explicit solution node 6

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
            checked_result: bool = (abs(Px) <= 1.0e-6) and (abs(Py) <= 1.0e-6) and (abs((a_1 + a_2*math.cos(th_3) + d_3*math.sin(th_3))*math.sin(th_2)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8

    # Code for explicit solution node 9, solved variable is th_0
    def ExplicitSolutionNode_node_9_solve_th_0_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs((a_1 + a_2*math.cos(th_3) + d_3*math.sin(th_3))*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                x0 = math.atan2(Px, -Py)
                x1 = math.sin(th_2)
                x2 = a_1 + a_2*math.cos(th_3) + d_3*math.sin(th_3)
                x3 = safe_sqrt(Px**2 + Py**2 - x1**2*x2**2)
                x4 = x1*x2
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x0 + math.atan2(x3, x4)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                this_solution[0] = x0 + math.atan2(-x3, x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor()
    # Finish code for explicit solution node 9

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
            checked_result: bool = (abs(Pz) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0)) <= 1.0e-6)
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(Pz) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_3)
                x2 = (a_1 + a_2*x0 + d_3*x1)*math.cos(th_2)
                x3 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x4 = -a_2*x1 - d_0 + d_3*x0
                this_solution[1] = math.atan2(-Pz*x2 - x3*x4, Pz*x4 - x2*x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 11

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
        value_at_4 = th_4  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[4]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[5]  # th_6
        new_ik_i[6] = value_at_6
        ik_out.append(new_ik_i)
    return ik_out


def marvin_m6_left_arm_ik_solve(T_ee: np.ndarray, th_4):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_4)):
        return []
    T_ee_raw_in = marvin_m6_left_arm_ik_target_original_to_raw(T_ee)
    ik_output_raw = marvin_m6_left_arm_ik_solve_raw(T_ee_raw_in, th_4 + th_4_offset_original2raw)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ik_out_i[6] -= th_6_offset_original2raw
        ee_pose_i = marvin_m6_left_arm_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_marvin_m6_left_arm():
    theta_in = np.random.random(size=(7, ))
    ee_pose = marvin_m6_left_arm_fk(theta_in)
    ik_output = marvin_m6_left_arm_ik_solve(ee_pose, th_4=theta_in[4])
    for i in range(len(ik_output)):
        ee_pose_i = marvin_m6_left_arm_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_marvin_m6_left_arm()
