import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 28
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.018
a_2: float = 0.018
d_0: float = 0.287
d_3: float = 0.314
post_transform_special_symbol_13: float = -0.095
pre_transform_special_symbol_03: float = -0.2215
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
def marvin_m6_right_arm_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[1, 0] = r_32
    ee_transformed[1, 1] = r_33
    ee_transformed[1, 2] = -r_31
    ee_transformed[1, 3] = -Pz - post_transform_special_symbol_13*r_33 + pre_transform_special_symbol_23
    ee_transformed[2, 0] = r_12
    ee_transformed[2, 1] = r_13
    ee_transformed[2, 2] = -r_11
    ee_transformed[2, 3] = -Px - post_transform_special_symbol_13*r_13 + pre_transform_special_symbol_03
    return ee_transformed


def marvin_m6_right_arm_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -r_33
    ee_transformed[0, 1] = r_31
    ee_transformed[0, 2] = r_32
    ee_transformed[0, 3] = -Pz - post_transform_special_symbol_13*r_32 + pre_transform_special_symbol_03
    ee_transformed[1, 0] = r_13
    ee_transformed[1, 1] = -r_11
    ee_transformed[1, 2] = -r_12
    ee_transformed[1, 3] = Px + post_transform_special_symbol_13*r_12 + pre_transform_special_symbol_13
    ee_transformed[2, 0] = -r_23
    ee_transformed[2, 1] = r_21
    ee_transformed[2, 2] = r_22
    ee_transformed[2, 3] = -Py - post_transform_special_symbol_13*r_22 + pre_transform_special_symbol_23
    return ee_transformed


def marvin_m6_right_arm_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)

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
    ee_pose[0, 0] = x0*x8 + x15*x9
    ee_pose[0, 1] = -x16*x17 + x18*x19
    ee_pose[0, 2] = -x20
    ee_pose[0, 3] = a_1*x6 - a_2*x14 + d_0*x1 + d_3*x7 + post_transform_special_symbol_13*x20 + pre_transform_special_symbol_03
    ee_pose[1, 0] = -x0*x27 - x31*x9
    ee_pose[1, 1] = x16*x32 - x18*x33
    ee_pose[1, 2] = x34
    ee_pose[1, 3] = a_1*x26 + a_2*x30 - d_0*x22 + d_3*x27 - post_transform_special_symbol_13*x34 + pre_transform_special_symbol_13
    ee_pose[2, 0] = x0*x37 + x40*x9
    ee_pose[2, 1] = -x16*x41 + x18*x42
    ee_pose[2, 2] = -x43
    ee_pose[2, 3] = -a_1*x36 - a_2*x39 + d_0*x35 - d_3*x37 + post_transform_special_symbol_13*x43 + pre_transform_special_symbol_23
    return ee_pose


def marvin_m6_right_arm_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)

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
    x9 = math.cos(th_4)
    x10 = math.sin(th_4)
    x11 = x0*x5 + x4*x7
    x12 = x1*x2*x9 - x10*x11
    x13 = math.cos(th_5)
    x14 = math.sin(th_5)
    x15 = -x13*x8 - x14*(x10*x3 + x11*x9)
    x16 = math.sin(th_0)
    x17 = math.cos(th_0)
    x18 = x1*x17
    x19 = x16*x6
    x20 = x0*x17*x2 - x19
    x21 = x16*x2
    x22 = x17*x6
    x23 = x0*x22 + x21
    x24 = x18*x4 + x23*x5
    x25 = -x18*x5 + x23*x4
    x26 = -x10*x25 + x20*x9
    x27 = -x13*x24 - x14*(x10*x20 + x25*x9)
    x28 = x1*x16
    x29 = -x0*x21 - x22
    x30 = -x0*x19 + x17*x2
    x31 = -x28*x4 + x30*x5
    x32 = x28*x5 + x30*x4
    x33 = -x10*x32 + x29*x9
    x34 = -x13*x31 - x14*(x10*x29 + x32*x9)
    x35 = d_0*x28 + pre_transform_special_symbol_23
    x36 = d_0*x18
    x37 = pre_transform_special_symbol_13 - x36
    x38 = a_1*x30 + x35
    x39 = a_1*x23 + pre_transform_special_symbol_13 - x36
    x40 = a_2*x32 + d_3*x31 + x38
    x41 = a_2*x25 + d_3*x24 + x39
    x42 = d_0*x0 + pre_transform_special_symbol_03
    x43 = a_1*x7 + x42
    x44 = a_2*x11 + d_3*x8 + x43
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 0] = -1
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x3
    jacobian_output[0, 4] = x8
    jacobian_output[0, 5] = x12
    jacobian_output[0, 6] = x15
    jacobian_output[1, 1] = -x16
    jacobian_output[1, 2] = -x18
    jacobian_output[1, 3] = x20
    jacobian_output[1, 4] = x24
    jacobian_output[1, 5] = x26
    jacobian_output[1, 6] = x27
    jacobian_output[2, 1] = -x17
    jacobian_output[2, 2] = x28
    jacobian_output[2, 3] = x29
    jacobian_output[2, 4] = x31
    jacobian_output[2, 5] = x33
    jacobian_output[2, 6] = x34
    jacobian_output[3, 1] = -pre_transform_special_symbol_13*x17 + pre_transform_special_symbol_23*x16
    jacobian_output[3, 2] = x18*x35 + x28*x37
    jacobian_output[3, 3] = -x20*x38 + x29*x39
    jacobian_output[3, 4] = -x24*x40 + x31*x41
    jacobian_output[3, 5] = -x26*x40 + x33*x41
    jacobian_output[3, 6] = -x27*x40 + x34*x41
    jacobian_output[4, 0] = -pre_transform_special_symbol_23
    jacobian_output[4, 1] = pre_transform_special_symbol_03*x17
    jacobian_output[4, 2] = x0*x35 - x28*x42
    jacobian_output[4, 3] = x1*x2*x38 - x29*x43
    jacobian_output[4, 4] = -x31*x44 + x40*x8
    jacobian_output[4, 5] = x12*x40 - x33*x44
    jacobian_output[4, 6] = x15*x40 - x34*x44
    jacobian_output[5, 0] = pre_transform_special_symbol_13
    jacobian_output[5, 1] = -pre_transform_special_symbol_03*x16
    jacobian_output[5, 2] = -x0*x37 - x18*x42
    jacobian_output[5, 3] = x20*x43 - x3*x39
    jacobian_output[5, 4] = x24*x44 - x41*x8
    jacobian_output[5, 5] = -x12*x41 + x26*x44
    jacobian_output[5, 6] = -x15*x41 + x27*x44
    return jacobian_output


def marvin_m6_right_arm_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)

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
    x9 = math.cos(th_4)
    x10 = math.sin(th_4)
    x11 = x0*x5 + x4*x7
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
    x25 = -x0*x19 - x20
    x26 = -x0*x17 + x15*x2
    x27 = -x24*x4 + x26*x5
    x28 = x24*x5 + x26*x4
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -1
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x3
    jacobian_output[0, 4] = x8
    jacobian_output[0, 5] = x1*x2*x9 - x10*x11
    jacobian_output[0, 6] = -x12*x8 - x13*(x10*x3 + x11*x9)
    jacobian_output[1, 1] = -x14
    jacobian_output[1, 2] = -x16
    jacobian_output[1, 3] = x18
    jacobian_output[1, 4] = x22
    jacobian_output[1, 5] = -x10*x23 + x18*x9
    jacobian_output[1, 6] = -x12*x22 - x13*(x10*x18 + x23*x9)
    jacobian_output[2, 1] = -x15
    jacobian_output[2, 2] = x24
    jacobian_output[2, 3] = x25
    jacobian_output[2, 4] = x27
    jacobian_output[2, 5] = -x10*x28 + x25*x9
    jacobian_output[2, 6] = -x12*x27 - x13*(x10*x25 + x28*x9)
    return jacobian_output


def marvin_m6_right_arm_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = x0*x2
    x5 = d_0*x2
    x6 = pre_transform_special_symbol_23 + x1*x5
    x7 = -pre_transform_special_symbol_13
    x8 = x0*x5
    x9 = -x7 - x8
    x10 = math.cos(th_2)
    x11 = x0*x10
    x12 = math.cos(th_1)
    x13 = math.sin(th_2)
    x14 = x1*x13
    x15 = -x11 - x12*x14
    x16 = x1*x10
    x17 = x0*x12*x13 - x16
    x18 = x0*x13 - x12*x16
    x19 = a_1*x18 + x6
    x20 = x11*x12 + x14
    x21 = a_1*x20 + pre_transform_special_symbol_13 - x8
    x22 = math.cos(th_3)
    x23 = math.sin(th_3)
    x24 = x20*x23 + x22*x4
    x25 = x18*x23 - x22*x3
    x26 = x18*x22 + x23*x3
    x27 = a_2*x26 + d_3*x25 + x19
    x28 = x20*x22 - x23*x4
    x29 = a_2*x28 + d_3*x24 + x21
    x30 = math.cos(th_4)
    x31 = math.sin(th_4)
    x32 = x15*x30 - x26*x31
    x33 = x17*x30 - x28*x31
    x34 = math.cos(th_5)
    x35 = math.sin(th_5)
    x36 = -x24*x34 - x35*(x17*x31 + x28*x30)
    x37 = -x25*x34 - x35*(x15*x31 + x26*x30)
    x38 = p_on_ee_x*x0
    x39 = p_on_ee_x*x1
    x40 = d_0*x12 + pre_transform_special_symbol_03
    x41 = x13*x2
    x42 = x10*x2
    x43 = a_1*x42 + x40
    x44 = -x12*x22 + x23*x42
    x45 = x12*x23 + x22*x42
    x46 = a_2*x45 + d_3*x44 + x43
    x47 = x13*x2*x30 - x31*x45
    x48 = -x34*x44 - x35*(x30*x45 + x31*x41)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = p_on_ee_y*x0 - p_on_ee_z*x1 - pre_transform_special_symbol_13*x0 + pre_transform_special_symbol_23*x1
    jacobian_output[0, 2] = -p_on_ee_y*x3 - p_on_ee_z*x4 + x0*x2*x6 + x1*x2*x9
    jacobian_output[0, 3] = -p_on_ee_y*x15 + p_on_ee_z*x17 + x15*x21 - x17*x19
    jacobian_output[0, 4] = -p_on_ee_y*x25 + p_on_ee_z*x24 - x24*x27 + x25*x29
    jacobian_output[0, 5] = -p_on_ee_y*x32 + p_on_ee_z*x33 - x27*x33 + x29*x32
    jacobian_output[0, 6] = -p_on_ee_y*x37 + p_on_ee_z*x36 - x27*x36 + x29*x37
    jacobian_output[1, 0] = p_on_ee_z - pre_transform_special_symbol_23
    jacobian_output[1, 1] = pre_transform_special_symbol_03*x0 - x38
    jacobian_output[1, 2] = -p_on_ee_z*x12 + x12*x6 + x2*x39 - x3*x40
    jacobian_output[1, 3] = p_on_ee_x*x15 - p_on_ee_z*x41 - x15*x43 + x19*x41
    jacobian_output[1, 4] = p_on_ee_x*x25 - p_on_ee_z*x44 - x25*x46 + x27*x44
    jacobian_output[1, 5] = p_on_ee_x*x32 - p_on_ee_z*x47 + x27*x47 - x32*x46
    jacobian_output[1, 6] = p_on_ee_x*x37 - p_on_ee_z*x48 + x27*x48 - x37*x46
    jacobian_output[2, 0] = -p_on_ee_y - x7
    jacobian_output[2, 1] = -pre_transform_special_symbol_03*x1 + x39
    jacobian_output[2, 2] = p_on_ee_y*x12 - x12*x9 + x2*x38 - x4*x40
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x13*x2 + x17*x43 - x21*x41
    jacobian_output[2, 4] = -p_on_ee_x*x24 + p_on_ee_y*x44 + x24*x46 - x29*x44
    jacobian_output[2, 5] = -p_on_ee_x*x33 + p_on_ee_y*x47 - x29*x47 + x33*x46
    jacobian_output[2, 6] = -p_on_ee_x*x36 + p_on_ee_y*x48 - x29*x48 + x36*x46
    return jacobian_output


def marvin_m6_right_arm_ik_solve_raw(T_ee: np.ndarray, th_4):
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

    # Code for explicit solution node 1, solved variable is th_3
    def ExplicitSolutionNode_node_1_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The explicit solution of root node
        condition_0: bool = (((abs(2*a_1*a_2 - 2*d_0*d_3) >= zero_tolerance)) or ((abs(2*a_1*d_3 + 2*a_2*d_0) >= zero_tolerance)) or ((abs(a_1**2 + a_2**2 + d_0**2 + d_3**2 - inv_Px**2 - inv_Py**2 - inv_Pz**2) >= zero_tolerance)))
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
            checked_result: bool = (((abs(inv_Px) <= 1.0e-6)) and ((abs(inv_Py) <= 1.0e-6)) and ((abs((a_1*math.cos(th_3) + a_2 + d_0*math.sin(th_3))*math.sin(th_4)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
            else:
                add_input_index_to(12, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2

    # Code for explicit solution node 12, solved variable is th_6
    def ExplicitSolutionNode_node_12_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[12]
        this_input_valid: bool = node_input_validity[12]
        if not this_input_valid:
            return
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_3 = this_solution[3]
            family_values = []
            if True:
                family_values.append(-3.14159265358979)
                family_values.append(-3.04341788316511)
                family_values.append(-2.94524311274043)
                family_values.append(-2.84706834231575)
                family_values.append(-2.74889357189107)
                family_values.append(-2.65071880146639)
                family_values.append(-2.55254403104171)
                family_values.append(-2.45436926061703)
                family_values.append(-2.35619449019234)
                family_values.append(-2.25801971976766)
                family_values.append(-2.15984494934298)
                family_values.append(-2.06167017891830)
                family_values.append(-1.96349540849362)
                family_values.append(-1.86532063806894)
                family_values.append(-1.76714586764426)
                family_values.append(-1.66897109721958)
                family_values.append(-1.57079632679490)
                family_values.append(-1.47262155637022)
                family_values.append(-1.37444678594553)
                family_values.append(-1.27627201552085)
                family_values.append(-1.17809724509617)
                family_values.append(-1.07992247467149)
                family_values.append(-0.981747704246811)
                family_values.append(-0.883572933822129)
                family_values.append(-0.785398163397448)
                family_values.append(-0.687223392972767)
                family_values.append(-0.589048622548086)
                family_values.append(-0.490873852123405)
                family_values.append(-0.392699081698724)
                family_values.append(-0.294524311274043)
                family_values.append(-0.196349540849362)
                family_values.append(-0.0981747704246811)
                family_values.append(0.0)
            if (abs(inv_Pz) >= 1.0e-6):
                x0 = math.sin(th_3)
                x1 = math.sin(th_4)
                x2 = math.cos(th_4)
                x3 = math.cos(th_3)
                x4 = x3**2
                x5 = inv_Pz**2
                x6 = a_1*x0
                x7 = d_3*x3
                x8 = 2*d_0
                x9 = x5*x8
                x10 = x0**2
                x11 = a_1**2*x5
                x12 = x2**2
                x13 = x12*x5
                x14 = d_0**2
                x15 = x3*x6
                x16 = a_2*x13
                x17 = a_1*x3
                x18 = x12*x4
                x19 = 1/safe_sqrt(a_2**2*x13 + d_3**2*x5 + 2*d_3*x5*x6 + x0*x16*x8 + x10*x11 + x10*x13*x14 + x11*x18 + x13*x15*x8 + x14*x4*x5 - x15*x9 + 2*x16*x17 - x7*x9)
                x20 = x0*x1
                x21 = 0.0348994967025011*x20
                x22 = a_1*x4
                x23 = inv_Pz*x19
                x24 = x2*x23
                x25 = r_11*x24
                x26 = 0.0348994967025011*x25
                x27 = a_1*x10
                x28 = 0.0348994967025011*x3
                x29 = a_2*x28
                x30 = d_3*x0
                x31 = r_12*x21 + x22*x26 + x25*x29 + x26*x27 + x26*x30
                x32 = 0.999390827019096*x20
                x33 = 0.999390827019096*x2
                x34 = x23*x33
                x35 = r_22*x34
                x36 = x22*x35
                x37 = x27*x35
                x38 = a_2*x3
                x39 = x35*x38
                x40 = x30*x35
                x41 = r_12*x24
                x42 = 0.0348994967025011*x41
                x43 = -r_11*x21 + x22*x42 + x27*x42 + x29*x41 + x30*x42
                x44 = 0.999390827019096*x23
                x45 = r_23*x44
                x46 = d_0*x4
                x47 = x45*x46
                x48 = x15*x45
                x49 = a_2*x0
                x50 = x12*x45
                x51 = x49*x50
                x52 = d_0*x10
                x53 = x50*x52
                x54 = x12*x48
                x55 = r_13*x23
                x56 = 0.0348994967025011*x55
                x57 = x28*x55
                x58 = x12*x56
                x59 = x57*x6
                x60 = -d_3*x57 + x12*x59 + x46*x56 + x49*x58 + x52*x58 - x59
                x61 = r_21*x34
                x62 = r_32*x33
                x63 = r_31*x1
                x64 = d_3*x44
                x65 = r_31*x20
                x66 = a_1*x44
                x67 = d_0*x3*x44
                x68 = x63*x67
                x69 = r_32*x1
                x70 = r_31*x24
                x71 = 0.0348994967025011*x70
                x72 = d_3*x28
                x73 = -x28*x69 + x46*x71 + x49*x71 + x52*x71 - x70*x72
                x74 = r_31*x33
                x75 = x64*x69
                x76 = r_32*x20
                x77 = x66*x76
                x78 = r_32*x24
                x79 = 0.0348994967025011*x78
                x80 = x28*x63 + x46*x79 + x49*x79 + x52*x79 - x72*x78
                x81 = r_33*x23
                x82 = x1*x33*x81
                x83 = a_2*x82
                x84 = x17*x82
                x85 = d_0*x81
                x86 = x20*x33*x85
                x87 = 0.0348994967025011*x81
                x88 = x12*x81
                x89 = x0*x28
                x90 = a_1*x18*x87 + d_0*x88*x89 + x27*x87 + x29*x88 + x30*x87 - x85*x89
                x91 = -x22*x70 - x27*x70 - x30*x70 - x38*x70 - x76
                x92 = x22*x78 + x27*x78 + x30*x78 + x38*x78 - x65
                x93 = -x15*x81 + x15*x88 + x46*x81 + x49*x88 + x52*x88 - x7*x81
                family_values = trig_family_representatives([(0.999390827019096*a_1*inv_Pz*r_21*x10*x19*x2 + 0.999390827019096*a_1*inv_Pz*r_21*x19*x2*x4 + 0.999390827019096*a_2*inv_Pz*r_21*x19*x2*x3 + 0.999390827019096*d_3*inv_Pz*r_21*x0*x19*x2 + 0.999390827019096*r_22*x0*x1 - x31, r_21*x32 - x36 - x37 - x39 - x40 + x43, x45*x7 - x47 + x48 - x51 - x53 - x54 + x60), (r_22*x32 + x22*x61 + x27*x61 + x30*x61 + x31 + x38*x61, 0.999390827019096*r_21*x0*x1 - x36 - x37 - x39 - x40 - x43, 0.999390827019096*a_1*inv_Pz*r_23*x0*x19*x3 + 0.999390827019096*d_3*inv_Pz*r_23*x19*x3 - x47 - x51 - x53 - x54 - x60), (-x62 + x63*x64 + x65*x66 - x68 + x73, 0.999390827019096*d_0*inv_Pz*r_32*x1*x19*x3 - x74 - x75 - x77 - x80, -x83 - x84 - x86 + x90), (0.999390827019096*a_1*inv_Pz*r_31*x0*x1*x19 + 0.999390827019096*d_3*inv_Pz*r_31*x1*x19 - x62 - x68 - x73, x67*x69 - x74 - x75 - x77 + x80, -x83 - x84 - x86 - x90), (x91, x92, x93 - 0.499999999999999), (x91, x92, x93 - 0.5), (x91, x92, x93 - 1), (x91, x92, x93 + 1)], -3.141592653589793, 0.0, family_values)
            for family_index, family_value in enumerate(family_values):
                candidate = copy.copy(this_solution)
                candidate[5] = family_value
                if family_index == len(family_values)-1:
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(13, candidate_index)
            if not family_values:
                queue_element_validity[queue_index] = False
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_6_processor()
    # Finish code for explicit solution node 12

    # Code for equation all-zero dispatcher node 13
    def EquationAllZeroDispatcherNode_node_13_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_6 = this_solution[5]
            checked_result: bool = (((abs(inv_Pz) <= 1.0e-6)) and ((abs(inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(14, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_13_processor()
    # Finish code for equation all-zero dispatcher node 13

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
            th_3 = this_solution[3]
            th_6 = this_solution[5]
            condition_0: bool = (((abs(inv_Pz) >= 1.0e-6)) or ((abs(inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) >= 1.0e-6)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_3)
                x2 = (a_1*x0 + a_2 + d_0*x1)*math.cos(th_4)
                x3 = -inv_Px*math.cos(th_6) + inv_Py*math.sin(th_6)
                x4 = a_1*x1 - d_0*x0 + d_3
                this_solution[4] = math.atan2(inv_Pz*x2 - x3*x4, inv_Pz*x4 + x2*x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor()
    # Finish code for explicit solution node 14

    # Code for non-branch dispatcher node 15
    # Actually, there is no code

    # Code for explicit solution node 16, solved variable is th_2
    def ExplicitSolutionNode_node_16_solve_th_2_processor():
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
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = ((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 > 1.0e-24)
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = r_31*(x2*x4 - x3*x6) + r_32*(x2*x3 + x4*x6) + r_33*x0*x1
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x0*x9
                x11 = x1*x8 - x2*x5*x9
                x12 = r_31*(x10*x4 - x11*x3) + r_32*(x10*x3 + x11*x4) + r_33*(-x1*x2*x9 - x5*x8)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = -math.atan2(-x7, -x12)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(17, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[2] = -math.atan2(x7, x12)
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(17, appended_idx)
            condition_2: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))))
            if condition_2:
                x0 = 6*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(17, appended_idx)
            condition_3: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))))
            if condition_3:
                x0 = 4*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(17, appended_idx)
            condition_4: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))))))
            if condition_4:
                x0 = 2*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(17, appended_idx)
            condition_5: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_5:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(17, appended_idx)
            condition_6: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_6:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 2*math.pi
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(17, appended_idx)
            condition_7: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 4*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 4*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_7:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 4*math.pi
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(17, appended_idx)
            condition_8: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 6*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 6*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_8:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 6*math.pi
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(17, appended_idx)
            condition_9: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))
            if condition_9:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 6*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 6*math.pi)
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(17, appended_idx)
            condition_10: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))
            if condition_10:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 4*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 4*math.pi)
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(17, appended_idx)
            condition_11: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_11:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 2*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 2*math.pi)
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(17, appended_idx)
            condition_12: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) <= min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_12:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(17, appended_idx)
            condition_13: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_13:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*math.pi
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(17, appended_idx)
            condition_14: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))))))
            if condition_14:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*math.pi
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(17, appended_idx)
            condition_15: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))))))
            if condition_15:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*math.pi
                this_solution[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_2_processor()
    # Finish code for explicit solution node 15

    # Code for non-branch dispatcher node 17
    # Actually, there is no code

    # Code for explicit solution node 18, solved variable is th_1
    def ExplicitSolutionNode_node_18_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = True
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = math.sin(th_3)
                x8 = x5*x7
                x9 = math.cos(th_3)
                x10 = x1*x9
                x11 = x0*x9
                x12 = x1*x7
                x13 = x12 - x2*x5*x9
                x14 = x0*x7
                x15 = x10 + x2*x8
                this_solution[1] = math.atan2((-r_31*(x11*x4 - x13*x3) - r_32*(x11*x3 + x13*x4) - r_33*(-x10*x2 - x8))*math.cos(th_2) - (-r_31*(x2*x4 - x3*x6) - r_32*(x2*x3 + x4*x6) - r_33*x0*x1)*math.sin(th_2), -r_31*(x14*x4 + x15*x3) - r_32*(x14*x3 - x15*x4) - r_33*(-x12*x2 + x5*x9))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(19, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_1_processor()
    # Finish code for explicit solution node 17

    # Code for non-branch dispatcher node 19
    # Actually, there is no code

    # Code for explicit solution node 20, solved variable is th_0
    def ExplicitSolutionNode_node_20_solve_th_0_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_2)
                x1 = math.sin(th_5)
                x2 = math.sin(th_4)
                x3 = x1*x2
                x4 = math.cos(th_4)
                x5 = math.cos(th_6)
                x6 = math.sin(th_6)
                x7 = math.cos(th_5)
                x8 = x2*x7
                x9 = x4*x5 + x6*x8
                x10 = x4*x6 - x5*x8
                x11 = r_11*x10 + r_12*x9 + r_13*x3
                x12 = r_21*x10 + r_22*x9 + r_23*x3
                x13 = math.sin(th_2)
                x14 = math.cos(th_1)
                x15 = x13*x14
                x16 = math.sin(th_3)
                x17 = x16*x7
                x18 = math.cos(th_3)
                x19 = x1*x18
                x20 = -x17 - x19*x4
                x21 = x18*x2
                x22 = x1*x16
                x23 = -x18*x4*x7 + x22
                x24 = x21*x6 - x23*x5
                x25 = x21*x5 + x23*x6
                x26 = r_11*x24 + r_12*x25 + r_13*x20
                x27 = math.sin(th_1)
                x28 = x18*x7 - x22*x4
                x29 = x16*x2
                x30 = x17*x4 + x19
                x31 = x29*x6 + x30*x5
                x32 = x29*x5 - x30*x6
                x33 = r_21*x24 + r_22*x25 + r_23*x20
                this_solution[0] = math.atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_0_processor()
    # Finish code for explicit solution node 19

    # Code for explicit solution node 3, solved variable is th_6
    def ExplicitSolutionNode_node_3_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[3]
        this_input_valid: bool = node_input_validity[3]
        if not this_input_valid:
            return
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_3 = this_solution[3]
            family_values = []
            if (((abs(inv_Px) >= zero_tolerance)) or ((abs(inv_Py) >= zero_tolerance)) or ((abs((a_1*math.cos(th_3) + a_2 + d_0*math.sin(th_3))*math.sin(th_4)) >= zero_tolerance))):
                x0 = math.atan2(inv_Px, inv_Py)
                x1 = math.sin(th_4)
                x2 = a_1*math.cos(th_3) + a_2 + d_0*math.sin(th_3)
                x3 = safe_sqrt(inv_Px**2 + inv_Py**2 - x1**2*x2**2)
                x4 = -x1*x2
                family_values.append(x0 + math.atan2(x3, x4))
                family_values.append(x0 + math.atan2(-x3, x4))
            if (((abs(inv_Pz) >= 1.0e-6)) and ((abs(inv_Px**2 + inv_Py**2 - (-a_1*math.cos(th_3) - a_2 - d_0*math.sin(th_3))**2*math.sin(th_4)**2) <= 2.8421709430404e-14*inv_Px**2 + 2.8421709430404e-14*inv_Py**2 + 2.8421709430404e-14*inv_Pz**2))):
                x0 = math.sin(th_3)
                x1 = math.sin(th_4)
                x2 = math.cos(th_4)
                x3 = math.cos(th_3)
                x4 = x3**2
                x5 = inv_Pz**2
                x6 = a_1*x0
                x7 = d_3*x3
                x8 = 2*d_0
                x9 = x5*x8
                x10 = x0**2
                x11 = a_1**2*x5
                x12 = x2**2
                x13 = x12*x5
                x14 = d_0**2
                x15 = x3*x6
                x16 = a_2*x13
                x17 = a_1*x3
                x18 = x12*x4
                x19 = 1/safe_sqrt(a_2**2*x13 + d_3**2*x5 + 2*d_3*x5*x6 + x0*x16*x8 + x10*x11 + x10*x13*x14 + x11*x18 + x13*x15*x8 + x14*x4*x5 - x15*x9 + 2*x16*x17 - x7*x9)
                x20 = x0*x1
                x21 = 0.0348994967025011*x20
                x22 = a_1*x4
                x23 = inv_Pz*x19
                x24 = x2*x23
                x25 = r_11*x24
                x26 = 0.0348994967025011*x25
                x27 = a_1*x10
                x28 = 0.0348994967025011*x3
                x29 = a_2*x28
                x30 = d_3*x0
                x31 = r_12*x21 + x22*x26 + x25*x29 + x26*x27 + x26*x30
                x32 = 0.999390827019096*x20
                x33 = 0.999390827019096*x2
                x34 = x23*x33
                x35 = r_22*x34
                x36 = x22*x35
                x37 = x27*x35
                x38 = a_2*x3
                x39 = x35*x38
                x40 = x30*x35
                x41 = r_12*x24
                x42 = 0.0348994967025011*x41
                x43 = -r_11*x21 + x22*x42 + x27*x42 + x29*x41 + x30*x42
                x44 = 0.999390827019096*x23
                x45 = r_23*x44
                x46 = d_0*x4
                x47 = x45*x46
                x48 = x15*x45
                x49 = a_2*x0
                x50 = x12*x45
                x51 = x49*x50
                x52 = d_0*x10
                x53 = x50*x52
                x54 = x12*x48
                x55 = r_13*x23
                x56 = 0.0348994967025011*x55
                x57 = x28*x55
                x58 = x12*x56
                x59 = x57*x6
                x60 = -d_3*x57 + x12*x59 + x46*x56 + x49*x58 + x52*x58 - x59
                x61 = r_21*x34
                x62 = r_32*x33
                x63 = r_31*x1
                x64 = d_3*x44
                x65 = r_31*x20
                x66 = a_1*x44
                x67 = d_0*x3*x44
                x68 = x63*x67
                x69 = r_32*x1
                x70 = r_31*x24
                x71 = 0.0348994967025011*x70
                x72 = d_3*x28
                x73 = -x28*x69 + x46*x71 + x49*x71 + x52*x71 - x70*x72
                x74 = r_31*x33
                x75 = x64*x69
                x76 = r_32*x20
                x77 = x66*x76
                x78 = r_32*x24
                x79 = 0.0348994967025011*x78
                x80 = x28*x63 + x46*x79 + x49*x79 + x52*x79 - x72*x78
                x81 = r_33*x23
                x82 = x1*x33*x81
                x83 = a_2*x82
                x84 = x17*x82
                x85 = d_0*x81
                x86 = x20*x33*x85
                x87 = 0.0348994967025011*x81
                x88 = x12*x81
                x89 = x0*x28
                x90 = a_1*x18*x87 + d_0*x88*x89 + x27*x87 + x29*x88 + x30*x87 - x85*x89
                x91 = -x22*x70 - x27*x70 - x30*x70 - x38*x70 - x76
                x92 = x22*x78 + x27*x78 + x30*x78 + x38*x78 - x65
                x93 = -x15*x81 + x15*x88 + x46*x81 + x49*x88 + x52*x88 - x7*x81
                family_values = trig_family_representatives([(0.999390827019096*a_1*inv_Pz*r_21*x10*x19*x2 + 0.999390827019096*a_1*inv_Pz*r_21*x19*x2*x4 + 0.999390827019096*a_2*inv_Pz*r_21*x19*x2*x3 + 0.999390827019096*d_3*inv_Pz*r_21*x0*x19*x2 + 0.999390827019096*r_22*x0*x1 - x31, r_21*x32 - x36 - x37 - x39 - x40 + x43, x45*x7 - x47 + x48 - x51 - x53 - x54 + x60), (r_22*x32 + x22*x61 + x27*x61 + x30*x61 + x31 + x38*x61, 0.999390827019096*r_21*x0*x1 - x36 - x37 - x39 - x40 - x43, 0.999390827019096*a_1*inv_Pz*r_23*x0*x19*x3 + 0.999390827019096*d_3*inv_Pz*r_23*x19*x3 - x47 - x51 - x53 - x54 - x60), (-x62 + x63*x64 + x65*x66 - x68 + x73, 0.999390827019096*d_0*inv_Pz*r_32*x1*x19*x3 - x74 - x75 - x77 - x80, -x83 - x84 - x86 + x90), (0.999390827019096*a_1*inv_Pz*r_31*x0*x1*x19 + 0.999390827019096*d_3*inv_Pz*r_31*x1*x19 - x62 - x68 - x73, x67*x69 - x74 - x75 - x77 + x80, -x83 - x84 - x86 - x90), (x91, x92, x93 - 0.499999999999999), (x91, x92, x93 - 0.5), (x91, x92, x93 - 1), (x91, x92, x93 + 1)], -3.141592653589793, 0.0, family_values)
            for family_index, family_value in enumerate(family_values):
                candidate = copy.copy(this_solution)
                candidate[5] = family_value
                if family_index == len(family_values)-1:
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(4, candidate_index)
            if not family_values:
                queue_element_validity[queue_index] = False
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
            checked_result: bool = (((abs(inv_Pz) <= 1.0e-6)) and ((abs(inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
            else:
                add_input_index_to(21, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4

    # Code for explicit solution node 21, solved variable is th_5
    def ExplicitSolutionNode_node_21_solve_th_5_processor():
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
            condition_0: bool = True
            if condition_0:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = 0.523598775598299
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(22, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[4] = 0.589048622548086
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(22, appended_idx)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[4] = 0.654498469497874
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(22, appended_idx)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[4] = 0.719948316447661
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(22, appended_idx)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[4] = 0.785398163397448
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(22, appended_idx)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[4] = 0.850848010347236
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(22, appended_idx)
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[4] = 0.916297857297023
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(22, appended_idx)
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[4] = 0.981747704246810
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(22, appended_idx)
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[4] = 1.04719755119660
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(22, appended_idx)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[4] = 1.11264739814638
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(22, appended_idx)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[4] = 1.17809724509617
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(22, appended_idx)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[4] = 1.24354709204596
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(22, appended_idx)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[4] = 1.30899693899575
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(22, appended_idx)
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[4] = 1.37444678594553
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(22, appended_idx)
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[4] = 1.43989663289532
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(22, appended_idx)
                solution_15: IkSolution = copy.copy(this_solution)
                solution_15[4] = 1.50534647984511
                appended_idx = append_solution_to_queue(solution_15)
                add_input_index_to(22, appended_idx)
                solution_16: IkSolution = copy.copy(this_solution)
                solution_16[4] = 1.57079632679490
                appended_idx = append_solution_to_queue(solution_16)
                add_input_index_to(22, appended_idx)
                solution_17: IkSolution = copy.copy(this_solution)
                solution_17[4] = 1.63624617374468
                appended_idx = append_solution_to_queue(solution_17)
                add_input_index_to(22, appended_idx)
                solution_18: IkSolution = copy.copy(this_solution)
                solution_18[4] = 1.70169602069447
                appended_idx = append_solution_to_queue(solution_18)
                add_input_index_to(22, appended_idx)
                solution_19: IkSolution = copy.copy(this_solution)
                solution_19[4] = 1.76714586764426
                appended_idx = append_solution_to_queue(solution_19)
                add_input_index_to(22, appended_idx)
                solution_20: IkSolution = copy.copy(this_solution)
                solution_20[4] = 1.83259571459405
                appended_idx = append_solution_to_queue(solution_20)
                add_input_index_to(22, appended_idx)
                solution_21: IkSolution = copy.copy(this_solution)
                solution_21[4] = 1.89804556154383
                appended_idx = append_solution_to_queue(solution_21)
                add_input_index_to(22, appended_idx)
                solution_22: IkSolution = copy.copy(this_solution)
                solution_22[4] = 1.96349540849362
                appended_idx = append_solution_to_queue(solution_22)
                add_input_index_to(22, appended_idx)
                solution_23: IkSolution = copy.copy(this_solution)
                solution_23[4] = 2.02894525544341
                appended_idx = append_solution_to_queue(solution_23)
                add_input_index_to(22, appended_idx)
                solution_24: IkSolution = copy.copy(this_solution)
                solution_24[4] = 2.09439510239319
                appended_idx = append_solution_to_queue(solution_24)
                add_input_index_to(22, appended_idx)
                solution_25: IkSolution = copy.copy(this_solution)
                solution_25[4] = 2.15984494934298
                appended_idx = append_solution_to_queue(solution_25)
                add_input_index_to(22, appended_idx)
                solution_26: IkSolution = copy.copy(this_solution)
                solution_26[4] = 2.22529479629277
                appended_idx = append_solution_to_queue(solution_26)
                add_input_index_to(22, appended_idx)
                solution_27: IkSolution = copy.copy(this_solution)
                solution_27[4] = 2.29074464324256
                appended_idx = append_solution_to_queue(solution_27)
                add_input_index_to(22, appended_idx)
                solution_28: IkSolution = copy.copy(this_solution)
                solution_28[4] = 2.35619449019234
                appended_idx = append_solution_to_queue(solution_28)
                add_input_index_to(22, appended_idx)
                solution_29: IkSolution = copy.copy(this_solution)
                solution_29[4] = 2.42164433714213
                appended_idx = append_solution_to_queue(solution_29)
                add_input_index_to(22, appended_idx)
                solution_30: IkSolution = copy.copy(this_solution)
                solution_30[4] = 2.48709418409192
                appended_idx = append_solution_to_queue(solution_30)
                add_input_index_to(22, appended_idx)
                solution_31: IkSolution = copy.copy(this_solution)
                solution_31[4] = 2.55254403104171
                appended_idx = append_solution_to_queue(solution_31)
                add_input_index_to(22, appended_idx)
                this_solution[4] = 2.61799387799149
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_5_processor()
    # Finish code for explicit solution node 21

    # Code for non-branch dispatcher node 22
    # Actually, there is no code

    # Code for explicit solution node 23, solved variable is th_2
    def ExplicitSolutionNode_node_23_solve_th_2_processor():
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
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = ((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 > 1.0e-24)
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = r_31*(x2*x4 - x3*x6) + r_32*(x2*x3 + x4*x6) + r_33*x0*x1
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x0*x9
                x11 = x1*x8 - x2*x5*x9
                x12 = r_31*(x10*x4 - x11*x3) + r_32*(x10*x3 + x11*x4) + r_33*(-x1*x2*x9 - x5*x8)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = -math.atan2(-x7, -x12)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(24, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[2] = -math.atan2(x7, x12)
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(24, appended_idx)
            condition_2: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))))
            if condition_2:
                x0 = 6*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(24, appended_idx)
            condition_3: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))))
            if condition_3:
                x0 = 4*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(24, appended_idx)
            condition_4: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))))))
            if condition_4:
                x0 = 2*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(24, appended_idx)
            condition_5: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_5:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(24, appended_idx)
            condition_6: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_6:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 2*math.pi
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(24, appended_idx)
            condition_7: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 4*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 4*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_7:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 4*math.pi
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(24, appended_idx)
            condition_8: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 6*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 6*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_8:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 6*math.pi
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(24, appended_idx)
            condition_9: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))
            if condition_9:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 6*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 6*math.pi)
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(24, appended_idx)
            condition_10: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))
            if condition_10:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 4*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 4*math.pi)
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(24, appended_idx)
            condition_11: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_11:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 2*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 2*math.pi)
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(24, appended_idx)
            condition_12: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) <= min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_12:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(24, appended_idx)
            condition_13: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_13:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*math.pi
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(24, appended_idx)
            condition_14: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))))))
            if condition_14:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*math.pi
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(24, appended_idx)
            condition_15: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))))))
            if condition_15:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*math.pi
                this_solution[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_2_processor()
    # Finish code for explicit solution node 22

    # Code for non-branch dispatcher node 24
    # Actually, there is no code

    # Code for explicit solution node 25, solved variable is th_1
    def ExplicitSolutionNode_node_25_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = True
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = math.sin(th_3)
                x8 = x5*x7
                x9 = math.cos(th_3)
                x10 = x1*x9
                x11 = x0*x9
                x12 = x1*x7
                x13 = x12 - x2*x5*x9
                x14 = x0*x7
                x15 = x10 + x2*x8
                this_solution[1] = math.atan2((-r_31*(x11*x4 - x13*x3) - r_32*(x11*x3 + x13*x4) - r_33*(-x10*x2 - x8))*math.cos(th_2) - (-r_31*(x2*x4 - x3*x6) - r_32*(x2*x3 + x4*x6) - r_33*x0*x1)*math.sin(th_2), -r_31*(x14*x4 + x15*x3) - r_32*(x14*x3 - x15*x4) - r_33*(-x12*x2 + x5*x9))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(26, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_1_processor()
    # Finish code for explicit solution node 24

    # Code for non-branch dispatcher node 26
    # Actually, there is no code

    # Code for explicit solution node 27, solved variable is th_0
    def ExplicitSolutionNode_node_27_solve_th_0_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_2)
                x1 = math.sin(th_5)
                x2 = math.sin(th_4)
                x3 = x1*x2
                x4 = math.cos(th_4)
                x5 = math.cos(th_6)
                x6 = math.sin(th_6)
                x7 = math.cos(th_5)
                x8 = x2*x7
                x9 = x4*x5 + x6*x8
                x10 = x4*x6 - x5*x8
                x11 = r_11*x10 + r_12*x9 + r_13*x3
                x12 = r_21*x10 + r_22*x9 + r_23*x3
                x13 = math.sin(th_2)
                x14 = math.cos(th_1)
                x15 = x13*x14
                x16 = math.sin(th_3)
                x17 = x16*x7
                x18 = math.cos(th_3)
                x19 = x1*x18
                x20 = -x17 - x19*x4
                x21 = x18*x2
                x22 = x1*x16
                x23 = -x18*x4*x7 + x22
                x24 = x21*x6 - x23*x5
                x25 = x21*x5 + x23*x6
                x26 = r_11*x24 + r_12*x25 + r_13*x20
                x27 = math.sin(th_1)
                x28 = x18*x7 - x22*x4
                x29 = x16*x2
                x30 = x17*x4 + x19
                x31 = x29*x6 + x30*x5
                x32 = x29*x5 - x30*x6
                x33 = r_21*x24 + r_22*x25 + r_23*x20
                this_solution[0] = math.atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_0_processor()
    # Finish code for explicit solution node 26

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
            condition_0: bool = (((abs(inv_Pz) >= 1.0e-6)) or ((abs(inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) >= 1.0e-6)))
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
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = ((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 > 1.0e-24)
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = r_31*(x2*x4 - x3*x6) + r_32*(x2*x3 + x4*x6) + r_33*x0*x1
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x0*x9
                x11 = x1*x8 - x2*x5*x9
                x12 = r_31*(x10*x4 - x11*x3) + r_32*(x10*x3 + x11*x4) + r_33*(-x1*x2*x9 - x5*x8)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = -math.atan2(-x7, -x12)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[2] = -math.atan2(x7, x12)
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(8, appended_idx)
            condition_2: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))))
            if condition_2:
                x0 = 6*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(8, appended_idx)
            condition_3: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))))
            if condition_3:
                x0 = 4*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(8, appended_idx)
            condition_4: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))))))
            if condition_4:
                x0 = 2*math.pi
                x1 = math.sin(th_4)
                x2 = x1*math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = x3*x4 + x5*x6
                x8 = x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 + r_13*x2, -r_21*x8 - r_22*x7 - r_23*x2)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[2] = (1/2)*max(-3.10668606854991, -x0 + x9 - 3.10668606854991) + (1/2)*min(3.10668606854991, -x0 + x9 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(8, appended_idx)
            condition_5: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_5:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(8, appended_idx)
            condition_6: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_6:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 2*math.pi
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(8, appended_idx)
            condition_7: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 4*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 4*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_7:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 4*math.pi
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(8, appended_idx)
            condition_8: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 6*math.pi) <= min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 6*math.pi))) and ((((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) + r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.10668606854991))))))
            if condition_8:
                x0 = math.sin(th_4)
                x1 = x0*math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = x2*x3 + x4*x5
                x7 = x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 + r_13*x1, -r_21*x7 - r_22*x6 - r_23*x1) + 6*math.pi
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[2] = (1/2)*max(-3.10668606854991, x8 - 3.10668606854991) + (1/2)*min(3.10668606854991, x8 + 3.10668606854991)
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(8, appended_idx)
            condition_9: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))
            if condition_9:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 6*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 6*math.pi)
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(8, appended_idx)
            condition_10: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))
            if condition_10:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 4*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 4*math.pi)
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(8, appended_idx)
            condition_11: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_11:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991 + 2*math.pi) + (1/2)*min(3.10668606854991, -x12 + 3.10668606854991 + 2*math.pi)
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(8, appended_idx)
            condition_12: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) <= min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_12:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(8, appended_idx)
            condition_13: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))))))
            if condition_13:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*math.pi
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(8, appended_idx)
            condition_14: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))))))
            if condition_14:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*math.pi
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(8, appended_idx)
            condition_15: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) + math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) + math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(-math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) + math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) + math.cos(th_4)*math.cos(th_6)) - r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi - 3.10668606854991) <= min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi + 3.10668606854991))) and ((((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991) > min(3.10668606854991, 3.10668606854991 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 2*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 2*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 4*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 4*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.10668606854991 + 6*math.pi) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.10668606854991 + 6*math.pi))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.10668606854991))) and ((max(-3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.10668606854991) > min(3.10668606854991, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) + math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) + math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(-math.sin(th_3)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.10668606854991))))))
            if condition_15:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = -x0*x1 - x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = x6*x7 - x8*x9
                x11 = x6*x9 + x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*math.pi
                this_solution[2] = (1/2)*max(-3.10668606854991, -x12 - 3.10668606854991) + (1/2)*min(3.10668606854991, 3.10668606854991 - x12)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_2_processor()
    # Finish code for explicit solution node 6

    # Code for non-branch dispatcher node 8
    # Actually, there is no code

    # Code for explicit solution node 9, solved variable is th_1
    def ExplicitSolutionNode_node_9_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = True
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = math.sin(th_3)
                x8 = x5*x7
                x9 = math.cos(th_3)
                x10 = x1*x9
                x11 = x0*x9
                x12 = x1*x7
                x13 = x12 - x2*x5*x9
                x14 = x0*x7
                x15 = x10 + x2*x8
                this_solution[1] = math.atan2((-r_31*(x11*x4 - x13*x3) - r_32*(x11*x3 + x13*x4) - r_33*(-x10*x2 - x8))*math.cos(th_2) - (-r_31*(x2*x4 - x3*x6) - r_32*(x2*x3 + x4*x6) - r_33*x0*x1)*math.sin(th_2), -r_31*(x14*x4 + x15*x3) - r_32*(x14*x3 - x15*x4) - r_33*(-x12*x2 + x5*x9))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1_processor()
    # Finish code for explicit solution node 8

    # Code for non-branch dispatcher node 10
    # Actually, there is no code

    # Code for explicit solution node 11, solved variable is th_0
    def ExplicitSolutionNode_node_11_solve_th_0_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = True
            if condition_0:
                x0 = math.cos(th_2)
                x1 = math.sin(th_5)
                x2 = math.sin(th_4)
                x3 = x1*x2
                x4 = math.cos(th_4)
                x5 = math.cos(th_6)
                x6 = math.sin(th_6)
                x7 = math.cos(th_5)
                x8 = x2*x7
                x9 = x4*x5 + x6*x8
                x10 = x4*x6 - x5*x8
                x11 = r_11*x10 + r_12*x9 + r_13*x3
                x12 = r_21*x10 + r_22*x9 + r_23*x3
                x13 = math.sin(th_2)
                x14 = math.cos(th_1)
                x15 = x13*x14
                x16 = math.sin(th_3)
                x17 = x16*x7
                x18 = math.cos(th_3)
                x19 = x1*x18
                x20 = -x17 - x19*x4
                x21 = x18*x2
                x22 = x1*x16
                x23 = -x18*x4*x7 + x22
                x24 = x21*x6 - x23*x5
                x25 = x21*x5 + x23*x6
                x26 = r_11*x24 + r_12*x25 + r_13*x20
                x27 = math.sin(th_1)
                x28 = x18*x7 - x22*x4
                x29 = x16*x2
                x30 = x17*x4 + x19
                x31 = x29*x6 + x30*x5
                x32 = x29*x5 - x30*x6
                x33 = r_21*x24 + r_22*x25 + r_23*x20
                this_solution[0] = math.atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_0_processor()
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
        value_at_4 = th_4  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[4]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[5]  # th_6
        new_ik_i[6] = value_at_6
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def marvin_m6_right_arm_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, th_4, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_4)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = marvin_m6_right_arm_ik_solve_raw(T_ee_raw_in, th_4 + th_4_offset_original2raw)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = th_4
        ik_out_i[5] = add_revolute_offset(ik_out_i[5], -th_5_offset_original2raw)
        ik_out_i[6] = add_revolute_offset(ik_out_i[6], -th_6_offset_original2raw)
        ee_pose_i = marvin_m6_right_arm_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                _yaik_seed_candidates.strict_witness = True
                _yaik_seed_candidates.clear()
            elif not _yaik_seed_candidates.strict_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.1066860685499065, -2.094395102393195, -3.1066860685499065, -2.530727415391778, -3.1066860685499065, -1.0471975511965976, -1.5707963267948966], [3.1066860685499065, 2.0943951023931953, 3.1066860685499065, 1.0471975511965974, 3.1066860685499065, 1.0471975511965974, 1.5707963267948966], [True, True, True, True, True, True, True], [False, False, False, False, True, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = marvin_m6_right_arm_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
                if _yaik_seed_candidates is not None and np.max(np.abs(snapped_pose-T_ee)) < 1e-7:
                    _yaik_seed_candidates.strict_witness = True
                    _yaik_seed_candidates.clear()
    return ik_output


def marvin_m6_right_arm_ik_solve(T_ee: np.ndarray, th_4):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_4)):
        return []
    result, seen = [], set()
    refinement_seeds = RefinementSeeds()
    for transform in (marvin_m6_right_arm_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in marvin_m6_right_arm_ik_solve_for_raw_target(T_ee, raw_target, th_4, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: marvin_m6_right_arm_fk(q)
    if refinement_seeds and not refinement_seeds.strict_witness:
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, True, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                refinement_seeds.strict_witness = True
                break
    if not refinement_seeds.strict_witness:
        refinement_seeds.clear()
        seed_parameters = (th_4 + th_4_offset_original2raw,)
        for shifted in nearby_ik_targets(T_ee):
            for transform in (marvin_m6_right_arm_ik_target_original_to_raw,):
                raw_target = transform(shifted)
                for raw_candidate in marvin_m6_right_arm_ik_solve_raw(raw_target, *seed_parameters):
                    seed = raw_candidate.reshape(-1).copy()
                    seed[0] = seed[0] - th_0_offset_original2raw
                    seed[1] = add_revolute_offset(seed[1], -th_1_offset_original2raw)
                    seed[2] = seed[2] - th_2_offset_original2raw
                    seed[3] = add_revolute_offset(seed[3], -th_3_offset_original2raw)
                    seed[4] = th_4
                    seed[5] = add_revolute_offset(seed[5], -th_5_offset_original2raw)
                    seed[6] = add_revolute_offset(seed[6], -th_6_offset_original2raw)
                    retain_ik_refinement_seed(refinement_seeds, refinement_fk, T_ee, seed)
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, True, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                refinement_seeds.strict_witness = True
                break
    return result


def test_ik_solve_marvin_m6_right_arm():
    theta_in = np.random.random(size=(7, ))
    ee_pose = marvin_m6_right_arm_fk(theta_in)
    ik_output = marvin_m6_right_arm_ik_solve(ee_pose, th_4=theta_in[4])
    for i in range(len(ik_output)):
        ee_pose_i = marvin_m6_right_arm_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_marvin_m6_right_arm()
