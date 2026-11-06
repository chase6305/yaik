import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 22
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.11
a_2: float = 0.016
a_3: float = 0.0092
a_4: float = 0.00921
d_2: float = -0.306
d_4: float = -0.29955
pre_transform_s0: float = 0.1406
pre_transform_s1: float = -0.2256
pre_transform_s2: float = 0.2326

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = 1.5707963267948966
th_2_offset_original2raw: float = 3.141592653589793
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793
th_6_offset_original2raw: float = 1.5707963267948966


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def atlas_r_hand_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_21
    ee_transformed[0, 1] = -r_23
    ee_transformed[0, 2] = -r_22
    ee_transformed[0, 3] = -Py + pre_transform_s1
    ee_transformed[1, 0] = -r_11
    ee_transformed[1, 1] = r_13
    ee_transformed[1, 2] = r_12
    ee_transformed[1, 3] = Px - pre_transform_s0
    ee_transformed[2, 0] = -r_31
    ee_transformed[2, 1] = r_33
    ee_transformed[2, 2] = r_32
    ee_transformed[2, 3] = Pz - pre_transform_s2
    return ee_transformed


def atlas_r_hand_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -r_21
    ee_transformed[0, 1] = r_23
    ee_transformed[0, 2] = r_22
    ee_transformed[0, 3] = Py + pre_transform_s0
    ee_transformed[1, 0] = r_11
    ee_transformed[1, 1] = -r_13
    ee_transformed[1, 2] = -r_12
    ee_transformed[1, 3] = -Px + pre_transform_s1
    ee_transformed[2, 0] = -r_31
    ee_transformed[2, 1] = r_33
    ee_transformed[2, 2] = r_32
    ee_transformed[2, 3] = Pz + pre_transform_s2
    return ee_transformed


def atlas_r_hand_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.sin(th_6)
    x1 = math.cos(th_4)
    x2 = math.cos(th_0)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.cos(th_1)
    x6 = math.sin(th_0)
    x7 = math.sin(th_2)
    x8 = x6*x7
    x9 = -x4 - x5*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_3)
    x12 = math.sin(th_1)
    x13 = x12*x6
    x14 = math.cos(th_3)
    x15 = x2*x7
    x16 = x3*x6
    x17 = -x15 + x16*x5
    x18 = x11*x13 + x14*x17
    x19 = -x1*x9 - x10*x18
    x20 = math.cos(th_6)
    x21 = math.sin(th_5)
    x22 = -x11*x17 + x12*x14*x6
    x23 = math.cos(th_5)
    x24 = x1*x18 - x10*x9
    x25 = -x21*x22 + x23*x24
    x26 = a_2*x5
    x27 = -x15*x5 + x16
    x28 = x4*x5 + x8
    x29 = x11*x12*x2 + x14*x28
    x30 = -x1*x27 - x10*x29
    x31 = -x11*x28 + x12*x14*x2
    x32 = x1*x29 - x10*x27
    x33 = -x21*x31 + x23*x32
    x34 = x12*x7
    x35 = x12*x3
    x36 = x11*x5 - x14*x35
    x37 = -x1*x34 - x10*x36
    x38 = x11*x35 + x14*x5
    x39 = x1*x36 - x10*x34
    x40 = -x21*x38 + x23*x39
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = x0*x19 - x20*x25
    ee_pose[0, 1] = -x21*x24 - x22*x23
    ee_pose[0, 2] = -x0*x25 - x19*x20
    ee_pose[0, 3] = a_1*x6 + a_3*x17 + a_4*x18 - d_2*x13 + d_4*x22 + pre_transform_s0 + x26*x6
    ee_pose[1, 0] = -x0*x30 + x20*x33
    ee_pose[1, 1] = x21*x32 + x23*x31
    ee_pose[1, 2] = x0*x33 + x20*x30
    ee_pose[1, 3] = -a_1*x2 - a_3*x28 - a_4*x29 + d_2*x12*x2 - d_4*x31 + pre_transform_s1 - x2*x26
    ee_pose[2, 0] = x0*x37 - x20*x40
    ee_pose[2, 1] = -x21*x39 - x23*x38
    ee_pose[2, 2] = -x0*x40 - x20*x37
    ee_pose[2, 3] = -a_2*x12 - a_3*x35 + a_4*x36 - d_2*x5 + d_4*x38 + pre_transform_s2
    return ee_pose


def atlas_r_hand_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x1*x7
    x9 = -x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x1*x4
    x13 = -x0*x7 + x12*x6
    x14 = x1*x10*x2 - x11*x13
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = x10*x13 + x11*x3
    x18 = -x15*x9 - x16*x17
    x19 = math.cos(th_5)
    x20 = math.sin(th_5)
    x21 = -x14*x19 - x20*(x15*x17 - x16*x9)
    x22 = x0*x2
    x23 = x0*x6*x7 - x12
    x24 = -x5*x6 - x8
    x25 = -x10*x22 - x11*x24
    x26 = x10*x24 - x11*x22
    x27 = -x15*x23 - x16*x26
    x28 = -x19*x25 - x20*(x15*x26 - x16*x23)
    x29 = x2*x7
    x30 = x2*x4
    x31 = x10*x6 + x11*x30
    x32 = -x10*x30 + x11*x6
    x33 = -x15*x29 - x16*x32
    x34 = -x19*x31 - x20*(x15*x32 - x16*x29)
    x35 = a_2*x2 + d_2*x6 - pre_transform_s2
    x36 = -x35
    x37 = a_2*x6
    x38 = x0*x37
    x39 = a_1*x0
    x40 = -pre_transform_s1 + x39
    x41 = d_2*x0*x2 - x38 - x40
    x42 = a_3*x30 + x35
    x43 = -x42
    x44 = a_3*x24 + d_2*x22 + pre_transform_s1 - x38 - x39
    x45 = a_4*x32 + d_4*x31 - x42
    x46 = a_4*x26 + d_4*x25 + x44
    x47 = a_1*x1 + pre_transform_s0
    x48 = -d_2*x3 + x1*x37 + x47
    x49 = a_3*x13 + x48
    x50 = a_4*x17 + d_4*x14 + x49
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = x18
    jacobian_output[0, 6] = x21
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = x22
    jacobian_output[1, 3] = x23
    jacobian_output[1, 4] = x25
    jacobian_output[1, 5] = x27
    jacobian_output[1, 6] = x28
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = -x6
    jacobian_output[2, 3] = x29
    jacobian_output[2, 4] = x31
    jacobian_output[2, 5] = x33
    jacobian_output[2, 6] = x34
    jacobian_output[3, 0] = pre_transform_s1
    jacobian_output[3, 1] = -pre_transform_s2*x1
    jacobian_output[3, 2] = -x22*x36 - x41*x6
    jacobian_output[3, 3] = x2*x44*x7 - x23*x43
    jacobian_output[3, 4] = -x25*x45 + x31*x46
    jacobian_output[3, 5] = -x27*x45 + x33*x46
    jacobian_output[3, 6] = -x28*x45 + x34*x46
    jacobian_output[4, 0] = -pre_transform_s0
    jacobian_output[4, 1] = pre_transform_s2*x0
    jacobian_output[4, 2] = -x3*x36 + x48*x6
    jacobian_output[4, 3] = -x29*x49 + x43*x9
    jacobian_output[4, 4] = x14*x45 - x31*x50
    jacobian_output[4, 5] = x18*x45 - x33*x50
    jacobian_output[4, 6] = x21*x45 - x34*x50
    jacobian_output[5, 1] = x0*x40 + x1*x47
    jacobian_output[5, 2] = x22*x48 + x3*x41
    jacobian_output[5, 3] = x23*x49 - x44*x9
    jacobian_output[5, 4] = -x14*x46 + x25*x50
    jacobian_output[5, 5] = -x18*x46 + x27*x50
    jacobian_output[5, 6] = -x21*x46 + x28*x50
    return jacobian_output


def atlas_r_hand_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x1*x7
    x9 = -x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x1*x4
    x13 = -x0*x7 + x12*x6
    x14 = x1*x10*x2 - x11*x13
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = x10*x13 + x11*x3
    x18 = math.cos(th_5)
    x19 = math.sin(th_5)
    x20 = x0*x2
    x21 = x0*x6*x7 - x12
    x22 = -x5*x6 - x8
    x23 = -x10*x20 - x11*x22
    x24 = x10*x22 - x11*x20
    x25 = x2*x7
    x26 = x2*x4
    x27 = x10*x6 + x11*x26
    x28 = -x10*x26 + x11*x6
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = -x15*x9 - x16*x17
    jacobian_output[0, 6] = -x14*x18 - x19*(x15*x17 - x16*x9)
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = x20
    jacobian_output[1, 3] = x21
    jacobian_output[1, 4] = x23
    jacobian_output[1, 5] = -x15*x21 - x16*x24
    jacobian_output[1, 6] = -x18*x23 - x19*(x15*x24 - x16*x21)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = -x6
    jacobian_output[2, 3] = x25
    jacobian_output[2, 4] = x27
    jacobian_output[2, 5] = -x15*x25 - x16*x28
    jacobian_output[2, 6] = -x18*x27 - x19*(x15*x28 - x16*x25)
    return jacobian_output


def atlas_r_hand_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = add_revolute_offset(theta_input[6], th_6_offset_original2raw)
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = -pre_transform_s1
    x1 = math.sin(th_0)
    x2 = math.cos(th_1)
    x3 = math.sin(th_1)
    x4 = math.cos(th_0)
    x5 = p_on_ee_z*x4
    x6 = a_2*x3 + d_2*x2 - pre_transform_s2
    x7 = -x6
    x8 = x3*x4
    x9 = a_2*x2
    x10 = x4*x9
    x11 = a_1*x4
    x12 = x0 + x11
    x13 = d_2*x3*x4 - x10 - x12
    x14 = math.sin(th_2)
    x15 = x14*x3
    x16 = math.cos(th_2)
    x17 = x1*x16
    x18 = x14*x2*x4 - x17
    x19 = x16*x3
    x20 = a_3*x19 + x6
    x21 = -x20
    x22 = x1*x14
    x23 = x16*x4
    x24 = -x2*x23 - x22
    x25 = a_3*x24 + d_2*x8 + pre_transform_s1 - x10 - x11
    x26 = math.cos(th_3)
    x27 = math.sin(th_3)
    x28 = x19*x27 + x2*x26
    x29 = -x24*x27 - x26*x8
    x30 = -x19*x26 + x2*x27
    x31 = a_4*x30 + d_4*x28 - x20
    x32 = x24*x26 - x27*x8
    x33 = a_4*x32 + d_4*x29 + x25
    x34 = math.cos(th_4)
    x35 = math.sin(th_4)
    x36 = -x15*x34 - x30*x35
    x37 = -x18*x34 - x32*x35
    x38 = math.cos(th_5)
    x39 = math.sin(th_5)
    x40 = -x28*x38 - x39*(-x15*x35 + x30*x34)
    x41 = -x29*x38 - x39*(-x18*x35 + x32*x34)
    x42 = x1*x3
    x43 = a_1*x1 + pre_transform_s0
    x44 = -d_2*x42 + x1*x9 + x43
    x45 = -x2*x22 - x23
    x46 = -x14*x4 + x17*x2
    x47 = a_3*x46 + x44
    x48 = x1*x26*x3 - x27*x46
    x49 = x26*x46 + x27*x42
    x50 = a_4*x49 + d_4*x48 + x47
    x51 = -x34*x45 - x35*x49
    x52 = -x38*x48 - x39*(x34*x49 - x35*x45)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -p_on_ee_y - x0
    jacobian_output[0, 1] = p_on_ee_z*x1 - pre_transform_s2*x1
    jacobian_output[0, 2] = p_on_ee_y*x2 - x13*x2 + x3*x5 - x7*x8
    jacobian_output[0, 3] = -p_on_ee_y*x15 + p_on_ee_z*x18 + x14*x25*x3 - x18*x21
    jacobian_output[0, 4] = -p_on_ee_y*x28 + p_on_ee_z*x29 + x28*x33 - x29*x31
    jacobian_output[0, 5] = -p_on_ee_y*x36 + p_on_ee_z*x37 - x31*x37 + x33*x36
    jacobian_output[0, 6] = -p_on_ee_y*x40 + p_on_ee_z*x41 - x31*x41 + x33*x40
    jacobian_output[1, 0] = p_on_ee_x - pre_transform_s0
    jacobian_output[1, 1] = pre_transform_s2*x4 - x5
    jacobian_output[1, 2] = -p_on_ee_x*x2 + p_on_ee_z*x1*x3 + x2*x44 - x42*x7
    jacobian_output[1, 3] = p_on_ee_x*x15 - p_on_ee_z*x45 - x15*x47 + x21*x45
    jacobian_output[1, 4] = p_on_ee_x*x28 - p_on_ee_z*x48 - x28*x50 + x31*x48
    jacobian_output[1, 5] = p_on_ee_x*x36 - p_on_ee_z*x51 + x31*x51 - x36*x50
    jacobian_output[1, 6] = p_on_ee_x*x40 - p_on_ee_z*x52 + x31*x52 - x40*x50
    jacobian_output[2, 1] = -p_on_ee_x*x1 + p_on_ee_y*x4 + x1*x43 + x12*x4
    jacobian_output[2, 2] = -p_on_ee_x*x8 - p_on_ee_y*x42 + x1*x13*x3 + x3*x4*x44
    jacobian_output[2, 3] = -p_on_ee_x*x18 + p_on_ee_y*x45 + x18*x47 - x25*x45
    jacobian_output[2, 4] = -p_on_ee_x*x29 + p_on_ee_y*x48 + x29*x50 - x33*x48
    jacobian_output[2, 5] = -p_on_ee_x*x37 + p_on_ee_y*x51 - x33*x51 + x37*x50
    jacobian_output[2, 6] = -p_on_ee_x*x41 + p_on_ee_y*x52 - x33*x52 + x41*x50
    return jacobian_output


def atlas_r_hand_ik_solve_raw(T_ee: np.ndarray, th_0):
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
    for i in range(22):
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
        R_l[0, 4] = -Pz
        R_l[0, 5] = Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2
        R_l[1, 4] = Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2
        R_l[1, 5] = Pz
        R_l[2, 0] = r_31
        R_l[2, 1] = r_32
        R_l[2, 2] = -r_11*math.cos(th_0) - r_21*math.sin(th_0)
        R_l[2, 3] = -r_12*math.cos(th_0) - r_22*math.sin(th_0)
        R_l[3, 0] = -r_11*math.cos(th_0) - r_21*math.sin(th_0)
        R_l[3, 1] = -r_12*math.cos(th_0) - r_22*math.sin(th_0)
        R_l[3, 2] = -r_31
        R_l[3, 3] = -r_32
        R_l[4, 6] = r_11*math.sin(th_0) - r_21*math.cos(th_0)
        R_l[4, 7] = r_12*math.sin(th_0) - r_22*math.cos(th_0)
        R_l[5, 6] = -Pz*r_31 - (-Px*math.sin(th_0) + Py*math.cos(th_0))*(-r_11*math.sin(th_0) + r_21*math.cos(th_0)) - (r_11*math.cos(th_0) + r_21*math.sin(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[5, 7] = -Pz*r_32 - (-Px*math.sin(th_0) + Py*math.cos(th_0))*(-r_12*math.sin(th_0) + r_22*math.cos(th_0)) - (r_12*math.cos(th_0) + r_22*math.sin(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[6, 0] = (-Px*math.sin(th_0) + Py*math.cos(th_0))*(r_11*math.cos(th_0) + r_21*math.sin(th_0)) - (-r_11*math.sin(th_0) + r_21*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[6, 1] = (-Px*math.sin(th_0) + Py*math.cos(th_0))*(r_12*math.cos(th_0) + r_22*math.sin(th_0)) - (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[6, 2] = -Pz*(-r_11*math.sin(th_0) + r_21*math.cos(th_0)) + r_31*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[6, 3] = -Pz*(-r_12*math.sin(th_0) + r_22*math.cos(th_0)) + r_32*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[7, 0] = -Pz*(-r_11*math.sin(th_0) + r_21*math.cos(th_0)) + r_31*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[7, 1] = -Pz*(-r_12*math.sin(th_0) + r_22*math.cos(th_0)) + r_32*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[7, 2] = (Px*math.sin(th_0) - Py*math.cos(th_0))*(r_11*math.cos(th_0) + r_21*math.sin(th_0)) + (-r_11*math.sin(th_0) + r_21*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[7, 3] = (Px*math.sin(th_0) - Py*math.cos(th_0))*(r_12*math.cos(th_0) + r_22*math.sin(th_0)) + (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
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
        x0 = math.sin(th_0)
        x1 = Px*x0
        x2 = math.cos(th_0)
        x3 = -Py*x2 + x1
        x4 = Pz**2
        x5 = -x3
        x6 = x5**2
        x7 = Px*x2 + Py*x0 - a_1*x0**2 - a_1*x2**2
        x8 = x7**2
        x9 = -x4 - x6 - x8
        x10 = 2*a_3
        x11 = a_4*x10
        x12 = -x11
        x13 = 2*a_4
        x14 = a_2*x13
        x15 = 2*d_2
        x16 = d_4*x15
        x17 = x12 + x14 + x16
        x18 = a_2**2
        x19 = a_3**2
        x20 = a_4**2
        x21 = d_2**2
        x22 = d_4**2
        x23 = a_2*x10
        x24 = x18 + x19 + x20 + x21 + x22 - x23
        x25 = x17 + x24
        x26 = x25 + x9
        x27 = 4*a_2
        x28 = d_4*x27
        x29 = -x28
        x30 = 4*a_3
        x31 = d_4*x30
        x32 = 4*a_4
        x33 = d_2*x32
        x34 = x31 + x33
        x35 = x29 + x34
        x36 = -x35
        x37 = x17 - x18 - x19 - x20 - x21 - x22 + x23
        x38 = -x37 - x4 - x6 - x8
        x39 = r_11*x2 + r_21*x0
        x40 = Pz*x39
        x41 = r_31*x7
        x42 = x40 - x41
        x43 = r_12*x2 + r_22*x0
        x44 = Pz*x43
        x45 = r_32*x7
        x46 = x44 - x45
        x47 = R_l_inv_67*x42 + R_l_inv_77*x46
        x48 = a_4*x47
        x49 = R_l_inv_61*x42 + R_l_inv_71*x46
        x50 = d_4*x49
        x51 = R_l_inv_66*x42 + R_l_inv_76*x46
        x52 = d_4*x51
        x53 = R_l_inv_60*x42 + R_l_inv_70*x46
        x54 = a_4*x53
        x55 = -x54
        x56 = x48 + x50 + x52 + x55
        x57 = a_3*x53
        x58 = a_3*x47
        x59 = -x58
        x60 = x57 + x59
        x61 = a_2*x53
        x62 = d_2*x49 - x61
        x63 = a_2*x47
        x64 = d_2*x51
        x65 = R_l_inv_64*x42
        x66 = R_l_inv_74*x46
        x67 = x63 + x64 - x65 - x66
        x68 = x62 + x67
        x69 = R_l_inv_65*x42 + R_l_inv_75*x46
        x70 = x10*x69
        x71 = -x70
        x72 = x13*x69
        x73 = 2*d_4
        x74 = 2*R_l_inv_62
        x75 = 2*R_l_inv_72
        x76 = 2*a_2
        x77 = -x15 + x42*x74 + x46*x75 + x69*x76
        x78 = -x73 + x77
        x79 = -x48
        x80 = x50 + x58 + x79
        x81 = -x52
        x82 = x57 + x81
        x83 = -x63 - x64 + x65 + x66
        x84 = x62 + x83
        x85 = x47*x73
        x86 = x13*x51
        x87 = x13*x49
        x88 = x53*x73
        x89 = x87 + x88
        x90 = 4*R_l_inv_63
        x91 = 4*R_l_inv_73
        x92 = 4*d_2
        x93 = x69*x92
        x94 = -x57
        x95 = -d_2*x49
        x96 = x58 + x61 + x95
        x97 = -2*a_4*x69
        x98 = x73 + x77
        x99 = x52 + x54
        x100 = x48 + x84
        x101 = -r_11*x0 + r_21*x2
        x102 = 2*Pz
        x103 = x102*x5
        x104 = 2*x7
        x105 = -r_31*x4 + r_31*x6 + r_31*x8 - x101*x103 - x104*x40
        x106 = -r_12*x0 + r_22*x2
        x107 = -r_32*x4 + r_32*x6 + r_32*x8 - x103*x106 - x104*x44
        x108 = x7*(2*Py*x2 - 2*x1)
        x109 = x101*x108 + x102*x41 - x39*x4 - x39*x6 + x39*x8
        x110 = x102*x45 + x106*x108 - x4*x43 - x43*x6 + x43*x8
        x111 = R_l_inv_00*x105 + R_l_inv_10*x107 + R_l_inv_20*x109 + R_l_inv_30*x110
        x112 = a_3*x111
        x113 = -x112
        x114 = R_l_inv_01*x105 + R_l_inv_11*x107 + R_l_inv_21*x109 + R_l_inv_31*x110
        x115 = d_4*x114
        x116 = -x115
        x117 = a_2*x111
        x118 = R_l_inv_07*x105 + R_l_inv_17*x107 + R_l_inv_27*x109 + R_l_inv_37*x110
        x119 = a_3*x118
        x120 = a_4*x111
        x121 = -d_2*x114
        x122 = R_l_inv_06*x105 + R_l_inv_16*x107 + R_l_inv_26*x109 + R_l_inv_36*x110
        x123 = d_4*x122
        x124 = -x123
        x125 = x117 + x119 + x120 + x121 + x124
        x126 = R_l_inv_04*x105
        x127 = R_l_inv_14*x107
        x128 = R_l_inv_24*x109
        x129 = R_l_inv_34*x110
        x130 = a_2*x118
        x131 = -x130
        x132 = a_4*x118
        x133 = d_2*x122
        x134 = -x133
        x135 = x126 + x127 + x128 + x129 + x131 - x132 + x134
        x136 = 2*x19
        x137 = 2*x20
        x138 = a_3*x27
        x139 = R_l_inv_05*x105 + R_l_inv_15*x107 + R_l_inv_25*x109 + R_l_inv_35*x110
        x140 = x10*x139
        x141 = x136 + x137 - x138 - x140
        x142 = a_4*x30
        x143 = -x142
        x144 = -2*x21
        x145 = -2*x22
        x146 = 2*x18
        x147 = 4*d_4
        x148 = d_2*x147
        x149 = 2*x105
        x150 = R_l_inv_02*x149
        x151 = 2*x107
        x152 = R_l_inv_12*x151
        x153 = 2*x109
        x154 = R_l_inv_22*x153
        x155 = 2*x110
        x156 = R_l_inv_32*x155
        x157 = x139*x76
        x158 = x144 + x145 + x146 - x148 + x150 + x152 + x154 + x156 + x157
        x159 = x143 + x158
        x160 = a_4*x27 + x13*x139
        x161 = -x120
        x162 = x119 + x161
        x163 = x112 + x115
        x164 = x135 + x163
        x165 = d_2*x114 - x117
        x166 = x124 + x165
        x167 = x118*x73
        x168 = x122*x13
        x169 = x114*x13
        x170 = x111*x73
        x171 = x169 + x170
        x172 = 8*d_2
        x173 = a_3*x172
        x174 = 8*d_4
        x175 = a_4*x174
        x176 = 4*x105
        x177 = 4*x107
        x178 = R_l_inv_03*x176 + R_l_inv_13*x177 + 4*R_l_inv_23*x109 + 4*R_l_inv_33*x110 - a_2*x172 - x139*x92
        x179 = x117 + x121 + x123
        x180 = x126 + x127 + x128 + x129 + x131 + x132 + x134
        x181 = x113 + x115 + x180
        x182 = -4*a_2*a_4 - 2*a_4*x139
        x183 = x144 + x145 + x146 + x148 + x150 + x152 + x154 + x156 + x157
        x184 = x142 + x183
        x185 = x112 + x116 + x180
        x186 = R_l_inv_07*x109 + R_l_inv_17*x110 - R_l_inv_27*x105 - R_l_inv_37*x107
        x187 = a_4*x186
        x188 = R_l_inv_01*x109 + R_l_inv_11*x110 - R_l_inv_21*x105 - R_l_inv_31*x107
        x189 = d_4*x188
        x190 = R_l_inv_06*x109 + R_l_inv_16*x110 - R_l_inv_26*x105 - R_l_inv_36*x107
        x191 = d_4*x190
        x192 = R_l_inv_00*x109 + R_l_inv_10*x110 - R_l_inv_20*x105 - R_l_inv_30*x107
        x193 = a_4*x192
        x194 = -x193
        x195 = x187 + x189 + x191 + x194
        x196 = a_3*x192
        x197 = a_3*x186
        x198 = -x197
        x199 = x196 + x198
        x200 = a_2*x192
        x201 = d_2*x188 - x200
        x202 = R_l_inv_24*x105
        x203 = R_l_inv_34*x107
        x204 = a_2*x186
        x205 = d_2*x190
        x206 = R_l_inv_04*x109
        x207 = R_l_inv_14*x110
        x208 = x202 + x203 + x204 + x205 - x206 - x207
        x209 = x201 + x208
        x210 = R_l_inv_05*x109 + R_l_inv_15*x110 - R_l_inv_25*x105 - R_l_inv_35*x107
        x211 = x28 - x31
        x212 = R_l_inv_02*x153
        x213 = R_l_inv_12*x155
        x214 = x210*x76
        x215 = R_l_inv_22*x149
        x216 = R_l_inv_32*x151
        x217 = d_2*x27
        x218 = -x212 - x213 - x214 + x215 + x216 + x217
        x219 = d_2*x30
        x220 = x10*x210
        x221 = a_4*x147
        x222 = -x219 + x220 + x221
        x223 = x218 + x222
        x224 = -x196
        x225 = -d_2*x188
        x226 = x200 + x224 + x225
        x227 = -x189
        x228 = x191 + x193
        x229 = x187 + x227 + x228
        x230 = x186*x73
        x231 = x13*x190
        x232 = x13*x188
        x233 = x192*x73
        x234 = x232 + x233
        x235 = 8*a_3
        x236 = a_2*x235
        x237 = 4*x21
        x238 = 4*x19
        x239 = -x237 + x238
        x240 = 4*x20
        x241 = 4*x18
        x242 = 4*x22
        x243 = -x240 + x241 + x242
        x244 = -4*R_l_inv_03*x109 - 4*R_l_inv_13*x110 + R_l_inv_23*x176 + R_l_inv_33*x177 + x210*x92 + x239 + x243
        x245 = -x202 - x203 - x204 - x205 + x206 + x207
        x246 = x197 + x245
        x247 = -x33
        x248 = x13*x210 + x247 + x31
        x249 = r_31*x103 - x101*x4 + x101*x6 - x101*x8 + x108*x39
        x250 = r_32*x103 - x106*x4 + x106*x6 - x106*x8 + x108*x43
        x251 = R_l_inv_61*x249 + R_l_inv_71*x250
        x252 = d_4*x251
        x253 = R_l_inv_60*x249 + R_l_inv_70*x250
        x254 = a_3*x253
        x255 = R_l_inv_67*x249 + R_l_inv_77*x250
        x256 = a_4*x255
        x257 = d_2*x251
        x258 = R_l_inv_66*x249 + R_l_inv_76*x250
        x259 = d_4*x258
        x260 = a_2*x253
        x261 = -x260
        x262 = x254 + x256 + x257 + x259 + x261
        x263 = a_2*x255
        x264 = d_2*x258
        x265 = R_l_inv_64*x249
        x266 = -x265
        x267 = R_l_inv_74*x250
        x268 = -x267
        x269 = a_4*x253
        x270 = x263 + x264 + x266 + x268 - x269
        x271 = a_3*x255
        x272 = -x271
        x273 = x25 + x272
        x274 = R_l_inv_65*x249 + R_l_inv_75*x250
        x275 = x13*x274
        x276 = x10*x274
        x277 = x249*x74 + x250*x75 + x274*x76
        x278 = -x276 + x277
        x279 = -x254
        x280 = x256 + x279
        x281 = -d_2*x251
        x282 = x259 + x260 + x281
        x283 = -x252
        x284 = x269 + x283
        x285 = x263 + x264 + x266 + x268 + x284
        x286 = x255*x73
        x287 = x13*x258
        x288 = x13*x251
        x289 = x253*x73
        x290 = x288 + x289
        x291 = x249*x90 + x250*x91 - x274*x92
        x292 = -x14
        x293 = x11 - x16
        x294 = -x256
        x295 = x254 + x294
        x296 = -x259
        x297 = x257 + x261
        x298 = x296 + x297
        x299 = -2*a_4*x274
        x300 = -x10 + x13
        x301 = -x10 - x13
        x302 = 4*x65
        x303 = 4*x66
        x304 = 4*x63
        x305 = 4*x64
        x306 = 4*x52
        x307 = -x136 - x137
        x308 = x142 + x158 + x307
        x309 = 4*x123
        x310 = -x118*x27 + 4*x126 + 4*x127 + 4*x128 + 4*x129 - 4*x133
        x311 = a_3*x174 + a_4*x172
        x312 = x143 + x183 + x307
        x313 = x218 + x28
        x314 = 4*x191
        x315 = x186*x27 + 4*x202 + 4*x203 + 4*x205 - 4*x206 - 4*x207
        x316 = 8*a_2*a_4
        x317 = x212 + x213 + x214 - x215 - x216 - x217 + x28
        x318 = 4*x265
        x319 = 4*x267
        x320 = x255*x27
        x321 = 4*x264
        x322 = a_4*x235 + d_4*x172 + 4*x259
        x323 = -x277
        x324 = x18 + x19 + x20 + x21 + x22 + x23
        x325 = x324 + x9
        x326 = x12 + x16 + x292
        x327 = x325 + x326
        x328 = x28 + x34
        x329 = -x328
        x330 = x14 + x293
        x331 = x325 + x330
        x332 = x50 + x54
        x333 = x85 + x86
        x334 = -x119
        x335 = x120 + x334
        x336 = x136 + x137 + x138 + x140
        x337 = x167 + x168
        x338 = -x187
        x339 = x197 + x338
        x340 = x200 + x225
        x341 = x196 + x208 + x340
        x342 = x189 + x228
        x343 = x230 + x231
        x344 = x271 + x324
        x345 = x326 + x344
        x346 = x270 + x283
        x347 = x276 + x277
        x348 = x252 + x263 + x264 + x266 + x268 + x269
        x349 = x286 + x287 + x328
        x350 = x330 + x344
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x3
        A[0, 2] = x3
        A[0, 6] = x3
        A[0, 8] = x3
        A[1, 0] = x26
        A[1, 2] = x26
        A[1, 3] = x36
        A[1, 5] = x36
        A[1, 6] = x38
        A[1, 8] = x38
        A[2, 0] = x56 + x60 + x68
        A[2, 1] = x71 + x72 + x78
        A[2, 2] = x55 + x80 + x82 + x84
        A[2, 3] = x85 - x86 - x89
        A[2, 4] = -x27 + x30 + x42*x90 + x46*x91 - x93
        A[2, 5] = -x85 + x86 - x89
        A[2, 6] = -x56 - x83 - x94 - x96
        A[2, 7] = -x71 - x97 - x98
        A[2, 8] = x100 - x50 + x57 + x58 + x99
        A[3, 0] = -x113 - x116 - x125 - x135
        A[3, 1] = x141 + x159 + x160
        A[3, 2] = x162 + x164 + x166
        A[3, 3] = x167 - x168 - x171
        A[3, 4] = x173 + x175 + x178
        A[3, 5] = -x167 + x168 - x171
        A[3, 6] = -x162 - x179 - x181
        A[3, 7] = -x141 - x182 - x184
        A[3, 8] = x119 + x120 + x123 + x165 + x185
        A[4, 0] = x195 + x199 + x209
        A[4, 1] = 2*a_4*x210 - x211 - x223 - x33
        A[4, 2] = -x198 - x208 - x226 - x229
        A[4, 3] = x230 - x231 - x234
        A[4, 4] = x236 - x244
        A[4, 5] = -x230 + x231 - x234
        A[4, 6] = -x195 - x226 - x246
        A[4, 7] = x223 + x248 + x29
        A[4, 8] = x196 + x201 + x229 + x246
        A[5, 0] = x252 + x262 + x270 + x273
        A[5, 1] = x275 + x278
        A[5, 2] = -x273 - x280 - x282 - x285
        A[5, 3] = x286 - x287 - x290 - x35
        A[5, 4] = x291
        A[5, 5] = -x211 - x247 - x286 + x287 - x290
        A[5, 6] = x24 + x272 + x285 + x292 + x293 + x295 + x298
        A[5, 7] = -x278 - x299
        A[5, 8] = x262 - x263 - x264 + x265 + x267 + x271 + x284 + x37
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x300
        B[0, 2] = x300
        B[0, 3] = x147
        B[0, 5] = x147
        B[0, 6] = x301
        B[0, 8] = x301
        B[2, 0] = x78
        B[2, 1] = x302 + x303 - x304 - x305 - x306
        B[2, 2] = -x78
        B[2, 3] = x32
        B[2, 4] = x235*x51
        B[2, 5] = -x32
        B[2, 6] = x98
        B[2, 7] = -x302 - x303 + x304 + x305 - x306
        B[2, 8] = -x98
        B[3, 0] = x308
        B[3, 1] = -x309 + x310
        B[3, 2] = -x308
        B[3, 3] = x311
        B[3, 4] = x122*x235
        B[3, 5] = -x311
        B[3, 6] = x312
        B[3, 7] = -x309 - x310
        B[3, 8] = -x312
        B[4, 0] = -x313
        B[4, 1] = -x314 - x315
        B[4, 2] = x313
        B[4, 3] = x316
        B[4, 4] = x190*x235
        B[4, 5] = -x316
        B[4, 6] = x317
        B[4, 7] = -x314 + x315
        B[4, 8] = -x317
        B[5, 0] = x277
        B[5, 1] = -x237 + x238 - x243 + x318 + x319 - x320 - x321 - x322
        B[5, 2] = x323
        B[5, 4] = 16*a_3*d_2 + 16*a_4*d_4 + x235*x258
        B[5, 6] = x277
        B[5, 7] = -x239 - x240 + x241 + x242 - x318 - x319 + x320 + x321 - x322
        B[5, 8] = x323
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x3
        C[0, 2] = x3
        C[0, 6] = x3
        C[0, 8] = x3
        C[1, 0] = x327
        C[1, 2] = x327
        C[1, 3] = x329
        C[1, 5] = x329
        C[1, 6] = x331
        C[1, 8] = x331
        C[2, 0] = x100 + x332 + x59 + x81 + x94
        C[2, 1] = -x70 - x78 - x97
        C[2, 2] = x68 + x80 + x94 + x99
        C[2, 3] = x333 - x87 + x88
        C[2, 4] = 4*R_l_inv_63*x42 + 4*R_l_inv_73*x46 - x27 - x30 - x93
        C[2, 5] = -x333 - x87 + x88
        C[2, 6] = -x332 - x48 - x67 - x82 - x96
        C[2, 7] = x70 + x72 + x98
        C[2, 8] = -x50 - x60 - x61 - x79 - x83 - x95 - x99
        C[3, 0] = x166 + x181 + x335
        C[3, 1] = -x159 - x182 - x336
        C[3, 2] = -x117 - x121 - x124 - x161 - x185 - x334
        C[3, 3] = -x169 + x170 + x337
        C[3, 4] = -x173 - x175 + x178
        C[3, 5] = -x169 + x170 - x337
        C[3, 6] = -x125 + x126 + x127 + x128 + x129 - x130 - x132 - x133 - x163
        C[3, 7] = x160 + x184 + x336
        C[3, 8] = -x164 - x179 - x335
        C[4, 0] = -x191 - x194 - x227 - x339 - x341
        C[4, 1] = x219 - x220 - x221 + x248 + x313
        C[4, 2] = x209 + x224 + x339 + x342
        C[4, 3] = -x232 + x233 + x343
        C[4, 4] = -x236 - x244
        C[4, 5] = -x232 + x233 - x343
        C[4, 6] = -x187 - x189 + x191 - x193 - x197 - x341
        C[4, 7] = x222 + x248 + x317
        C[4, 8] = -x199 - x245 - x338 - x340 - x342
        C[5, 0] = -x282 - x295 - x345 - x346
        C[5, 1] = -x299 - x347
        C[5, 2] = x259 + x279 + x294 + x297 + x345 + x348
        C[5, 3] = -x288 + x289 + x349
        C[5, 4] = x291
        C[5, 5] = -x288 + x289 - x349
        C[5, 6] = -x254 - x256 - x260 - x281 - x296 - x348 - x350
        C[5, 7] = x275 + x347
        C[5, 8] = x280 + x298 + x346 + x350
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[2] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor()
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
            th_2 = this_solution[2]
            checked_result: bool = (((abs(2*a_2*a_4*math.cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) <= 1.0e-6)) and ((abs(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) <= 1.0e-6)) and ((abs(-Px**2 + 2*Px*a_1*math.cos(th_0) - Py**2 + 2*Py*a_1*math.sin(th_0) - Pz**2 - a_1**2 + a_2**2 + 2*a_2*a_3*math.cos(th_2) + a_3**2 + a_4**2 + d_2**2 + d_4**2) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2

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
            condition_0: bool = (((abs(2*a_2*a_4*math.cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) >= zero_tolerance)) or ((abs(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) >= zero_tolerance)) or ((abs(-Px**2 + 2*Px*a_1*math.cos(th_0) - Py**2 + 2*Py*a_1*math.sin(th_0) - Pz**2 - a_1**2 + a_2**2 + 2*a_2*a_3*math.cos(th_2) + a_3**2 + a_4**2 + d_2**2 + d_4**2) >= zero_tolerance)))
            if condition_0:
                x0 = 2*d_4
                x1 = 2*a_4
                x2 = a_2*math.cos(th_2)
                x3 = -a_3*x0 - d_2*x1 - x0*x2
                x4 = a_3*x1 - d_2*x0 + x1*x2
                x5 = math.atan2(x3, x4)
                x6 = 2*a_1
                x7 = Px**2 - Px*x6*math.cos(th_0) + Py**2 - Py*x6*math.sin(th_0) + Pz**2 + a_1**2 - a_2**2 - a_3**2 - 2*a_3*x2 - a_4**2 - d_2**2 - d_4**2
                x8 = safe_sqrt(x3**2 + x4**2 - x7**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x5 + math.atan2(x8, x7)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[3] = x5 + math.atan2(-x8, x7)
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
            checked_result: bool = (((abs(Pz) <= 1.0e-6)) and ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4

    # Code for explicit solution node 5, solved variable is th_1
    def ExplicitSolutionNode_node_5_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (((abs(Pz) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1) >= 1.0e-6)))
            if condition_0:
                x0 = math.cos(th_2)
                x1 = math.cos(th_3)
                x2 = math.sin(th_3)
                x3 = -a_2 - a_3*x0 - a_4*x0*x1 + d_4*x0*x2
                x4 = -Px*math.cos(th_0) - Py*math.sin(th_0) + a_1
                x5 = a_4*x2 - d_2 + d_4*x1
                this_solution[1] = math.atan2(Pz*x3 - x4*x5, Pz*x5 + x3*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1_processor()
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(r_13*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_23*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) + r_33*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) <= 70368744177665/70368744177664)
            if condition_0:
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_1)
                x3 = math.cos(th_2)
                x4 = math.sin(th_3)
                x5 = math.sin(th_0)
                x6 = x4*math.sin(th_2)
                x7 = math.cos(th_0)
                x8 = x0*x3*x4 - x1*x2
                x9 = safe_acos(r_13*(x5*x6 + x7*x8) + r_23*(x5*x8 - x6*x7) - r_33*(x0*x1 + x2*x3*x4))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x9
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                this_solution[6] = -x9
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor()
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
            th_5 = this_solution[6]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(12, branch_input_index)

            th_5 = this_solution[6]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
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

    # Code for explicit solution node 17, solved variable is th_4th_6_soa
    def ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)) or ((abs(r_12*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.sin(th_0)
                x3 = math.cos(th_2)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 - x4*x5
                x7 = x2*x5 + x3*x4
                this_solution[5] = math.atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor()
    # Finish code for explicit solution node 17

    # Code for non-branch dispatcher node 18
    # Actually, there is no code

    # Code for explicit solution node 19, solved variable is th_4
    def ExplicitSolutionNode_node_19_solve_th_4_processor():
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
            condition_0: bool = True
            if condition_0:
                this_solution[4] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_4_processor()
    # Finish code for explicit solution node 18

    # Code for non-branch dispatcher node 20
    # Actually, there is no code

    # Code for explicit solution node 21, solved variable is th_6
    def ExplicitSolutionNode_node_21_solve_th_6_processor():
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
            th_4 = this_solution[4]
            th_4th_6_soa = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[7] = -th_4 + th_4th_6_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_6_processor()
    # Finish code for explicit solution node 20

    # Code for explicit solution node 12, solved variable is negative_th_6_positive_th_4__soa
    def ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_11*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)) or ((abs(r_12*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.sin(th_0)
                x3 = math.cos(th_2)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 - x4*x5
                x7 = x2*x5 + x3*x4
                this_solution[0] = math.atan2(-r_11*x6 + r_21*x7 - r_31*x1, r_12*x6 - r_22*x7 + r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor()
    # Finish code for explicit solution node 12

    # Code for non-branch dispatcher node 13
    # Actually, there is no code

    # Code for explicit solution node 14, solved variable is th_4
    def ExplicitSolutionNode_node_14_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return

        # The solution of non-root node 14
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                this_solution[4] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_4_processor()
    # Finish code for explicit solution node 13

    # Code for non-branch dispatcher node 15
    # Actually, there is no code

    # Code for explicit solution node 16, solved variable is th_6
    def ExplicitSolutionNode_node_16_solve_th_6_processor():
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
            negative_th_6_positive_th_4__soa = this_solution[0]
            th_4 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[7] = -negative_th_6_positive_th_4__soa + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_6_processor()
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[6]
            condition_0: bool = (((abs(math.sin(th_5)) >= zero_tolerance)) or ((abs(r_13*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_23*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_33*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)) or ((abs(r_13*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.cos(th_3)) + r_23*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.sin(th_0) - math.sin(th_2)*math.cos(th_0)*math.cos(th_3)) - r_33*(math.sin(th_1)*math.cos(th_2)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1))) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_5)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.sin(th_0)
                x4 = math.cos(th_2)
                x5 = math.cos(th_0)
                x6 = math.cos(th_1)
                x7 = x2*x6
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x4*x9
                x11 = x2*x9
                x12 = x1*x8 + x10*x6
                this_solution[4] = math.atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(-r_13*(x11*x3 + x12*x5) - r_23*(-x11*x5 + x12*x3) - r_33*(-x1*x10 + x6*x8)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 9

    # Code for non-branch dispatcher node 10
    # Actually, there is no code

    # Code for explicit solution node 11, solved variable is th_6
    def ExplicitSolutionNode_node_11_solve_th_6_processor():
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
            th_5 = this_solution[6]
            condition_0: bool = (((abs(math.sin(th_5)) >= zero_tolerance)) or ((abs(r_12*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_22*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) + r_32*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) >= zero_tolerance)) or ((abs(-r_11*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) - r_21*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) + r_31*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_5)
                x1 = math.cos(th_1)
                x2 = math.cos(th_3)
                x3 = math.sin(th_1)
                x4 = math.sin(th_3)
                x5 = x4*math.cos(th_2)
                x6 = x1*x2 + x3*x5
                x7 = math.cos(th_0)
                x8 = x4*math.sin(th_2)
                x9 = x7*x8
                x10 = math.sin(th_0)
                x11 = -x1*x5 + x2*x3
                x12 = x10*x8
                x13 = -x11
                this_solution[7] = math.atan2(x0*(r_12*(x11*x7 - x12) + r_22*(x10*x11 + x9) + r_32*x6), x0*(r_11*(x12 + x13*x7) + r_21*(x10*x13 - x9) - r_31*x6))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_6_processor()
    # Finish code for explicit solution node 10

    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = th_0  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[1]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[2]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[3]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[4]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[6]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[7]  # th_6
        new_ik_i[6] = value_at_6
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def atlas_r_hand_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, th_0, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_0)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = atlas_r_hand_ik_solve_raw(T_ee_raw_in, th_0 + th_0_offset_original2raw)
    ik_output = list()
    strict_raw_witness = False
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = th_0
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = add_revolute_offset(ik_out_i[2], -th_2_offset_original2raw)
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = add_revolute_offset(ik_out_i[5], -th_5_offset_original2raw)
        ik_out_i[6] = add_revolute_offset(ik_out_i[6], -th_6_offset_original2raw)
        ee_pose_i = atlas_r_hand_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                strict_raw_witness = True
                _yaik_seed_candidates.clear()
            elif not strict_raw_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -4.71238898038469, -6.283185307179586, -6.283185307179586, -6.283185307179586, -6.283185307179586, -4.71238898038469], [3.141592653589793, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966], [True, True, True, True, True, True, True], [True, False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = atlas_r_hand_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
    return ik_output


def atlas_r_hand_ik_solve(T_ee: np.ndarray, th_0):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_0)):
        return []
    result, seen = [], set()
    refinement_seeds = []
    for transform in (atlas_r_hand_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in atlas_r_hand_ik_solve_for_raw_target(T_ee, raw_target, th_0, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: atlas_r_hand_fk(q)
    if refinement_seeds and not any(np.max(np.abs(refinement_fk(q)-T_ee)) < 1e-7 for q in result):
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [True, False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                break
    return result


def test_ik_solve_atlas_r_hand():
    theta_in = np.random.random(size=(7, ))
    ee_pose = atlas_r_hand_fk(theta_in)
    ik_output = atlas_r_hand_ik_solve(ee_pose, th_0=theta_in[0])
    for i in range(len(ik_output)):
        ee_pose_i = atlas_r_hand_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_atlas_r_hand()
