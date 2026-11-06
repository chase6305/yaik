import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 14
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 1.3
alpha_3: float = -1.0470795620393143
alpha_5: float = -1.0470795620393143
d_1: float = -1.4
d_2: float = -0.11397670814688408
d_4: float = -0.12300000000000003
pre_transform_special_symbol_23: float = 0.8

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = -0.0
th_3_offset_original2raw: float = -1.5707963267948966
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = -1.5707963267948966


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def yaskawa_mpx3500_ik_target_original_to_raw(T_ee: np.ndarray):
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


def yaskawa_mpx3500_ik_target_raw_to_original(T_ee: np.ndarray):
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


def yaskawa_mpx3500_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = add_revolute_offset(theta_input[4], th_4_offset_original2raw)
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)

    # Temp variable for efficiency
    x0 = math.cos(alpha_3)
    x1 = math.cos(th_0)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x5*x6
    x8 = x1*x4 - x1*x7
    x9 = math.sin(alpha_3)
    x10 = math.sin(th_0)
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = x2*x5
    x14 = x3*x6
    x15 = x1*x13 + x1*x14
    x16 = -x10*x11 - x12*x15
    x17 = x16*x9
    x18 = x0*x8 - x17
    x19 = 1.0*math.cos(alpha_5)
    x20 = x18*x19
    x21 = math.sin(alpha_5)
    x22 = math.cos(th_4)
    x23 = x8*x9
    x24 = math.sin(th_4)
    x25 = -x10*x12 + x11*x15
    x26 = x0*x16
    x27 = x22*x23 + x22*x26 - x24*x25
    x28 = math.cos(th_5)
    x29 = 1.0*x21
    x30 = x18*x29
    x31 = math.sin(th_5)
    x32 = 1.0*x22*x25 + 1.0*x23*x24 + 1.0*x24*x26
    x33 = x19*x27
    x34 = 1.0*a_0
    x35 = x34*x6
    x36 = 1.0*d_1
    x37 = 1.0*d_2
    x38 = x10*x4 - x10*x7
    x39 = x10*x13 + x10*x14
    x40 = x1*x11 - x12*x39
    x41 = x40*x9
    x42 = x0*x38 - x41
    x43 = x19*x42
    x44 = x38*x9
    x45 = x1*x12 + x11*x39
    x46 = x0*x40
    x47 = x22*x44 + x22*x46 - x24*x45
    x48 = x29*x42
    x49 = 1.0*x22*x45 + 1.0*x24*x44 + 1.0*x24*x46
    x50 = x19*x47
    x51 = x13 + x14
    x52 = x0*x51
    x53 = -x4 + x5*x6
    x54 = x12*x53
    x55 = x54*x9
    x56 = x52 + x55
    x57 = x19*x56
    x58 = x11*x53
    x59 = x0*x54
    x60 = x22*x51*x9 - x22*x59 - x24*x58
    x61 = x29*x56
    x62 = 1.0*x22*x58 + 1.0*x24*x51*x9 - 1.0*x24*x59
    x63 = x19*x60
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x20 + 1.0*x21*x27
    ee_pose[0, 1] = x28*x30 + x28*x33 - x31*x32
    ee_pose[0, 2] = x28*x32 + x30*x31 + x31*x33
    ee_pose[0, 3] = d_4*x20 - d_4*x27*x29 + x0*x37*x8 + x1*x35 - x17*x37 + x36*x8
    ee_pose[1, 0] = 1.0*x21*x47 - x43
    ee_pose[1, 1] = x28*x48 + x28*x50 - x31*x49
    ee_pose[1, 2] = x28*x49 + x31*x48 + x31*x50
    ee_pose[1, 3] = -d_4*x29*x47 + d_4*x43 + x0*x37*x38 + x10*x35 + x36*x38 - x37*x41
    ee_pose[2, 0] = 1.0*x21*x60 - x57
    ee_pose[2, 1] = x28*x61 + x28*x63 - x31*x62
    ee_pose[2, 2] = x28*x62 + x31*x61 + x31*x63
    ee_pose[2, 3] = -d_4*x29*x60 + d_4*x57 + 1.0*pre_transform_special_symbol_23 - x2*x34 + x36*x51 + x37*x52 + x37*x55
    return ee_pose


def yaskawa_mpx3500_twist_jacobian(theta_input: np.ndarray):
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
    x11 = math.cos(alpha_3)
    x12 = math.sin(alpha_3)
    x13 = math.cos(th_3)
    x14 = math.sin(th_3)
    x15 = x2*x9 + x6*x7
    x16 = -x1*x13 - x14*x15
    x17 = x12*x16
    x18 = x10*x11 - x17
    x19 = math.cos(alpha_5)
    x20 = x18*x19
    x21 = math.sin(alpha_5)
    x22 = math.cos(th_4)
    x23 = x12*x22
    x24 = math.sin(th_4)
    x25 = x11*x22
    x26 = x21*(x10*x23 + x16*x25 - x24*(-x1*x14 + x13*x15))
    x27 = x20 - x26
    x28 = x1*x3
    x29 = x1*x8
    x30 = x2*x28 - x29*x7
    x31 = x2*x29 + x28*x7
    x32 = 1.0*x13*x4 - x14*x31
    x33 = x12*x32
    x34 = x11*x30 - x33
    x35 = x19*x34
    x36 = x21*(x23*x30 - x24*(x13*x31 + x14*x5) + x25*x32)
    x37 = x35 - x36
    x38 = 1.0*x2
    x39 = 1.0*x7
    x40 = x3*x39 + x38*x8
    x41 = x11*x40
    x42 = -x3*x38 + 1.0*x7*x8
    x43 = x14*x42
    x44 = x12*x43
    x45 = x41 + x44
    x46 = x19*x45
    x47 = x21*(x12*x22*x40 - x13*x24*x42 - x25*x43)
    x48 = x46 - x47
    x49 = a_0*x38
    x50 = pre_transform_special_symbol_23 - x49
    x51 = a_0*x7
    x52 = d_1*x30 + x1*x51
    x53 = d_1*x40 + pre_transform_special_symbol_23 - x49
    x54 = d_2*x41 + d_2*x44 + x53
    x55 = d_2*x11*x30 - d_2*x33 + x52
    x56 = d_4*x46 - d_4*x47 + x54
    x57 = d_4*x35 - d_4*x36 + x55
    x58 = d_1*x10 + x5*x51
    x59 = d_2*x10*x11 - d_2*x17 + x58
    x60 = d_4*x20 - d_4*x26 + x59
    x61 = a_0*x39
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = -x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = x18
    jacobian_output[0, 5] = x27
    jacobian_output[1, 1] = x5
    jacobian_output[1, 2] = -x5
    jacobian_output[1, 3] = x30
    jacobian_output[1, 4] = x34
    jacobian_output[1, 5] = x37
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x40
    jacobian_output[2, 4] = x45
    jacobian_output[2, 5] = x48
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x5
    jacobian_output[3, 2] = x5*x50
    jacobian_output[3, 3] = -x30*x53 + x40*x52
    jacobian_output[3, 4] = -x34*x54 + x45*x55
    jacobian_output[3, 5] = -x37*x56 + x48*x57
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = x1*x50
    jacobian_output[4, 3] = x10*x53 - x40*x58
    jacobian_output[4, 4] = x18*x54 - x45*x59
    jacobian_output[4, 5] = x27*x56 - x48*x60
    jacobian_output[5, 2] = -x0**2*x61 - x4**2*x61
    jacobian_output[5, 3] = -x10*x52 + x30*x58
    jacobian_output[5, 4] = -x18*x55 + x34*x59
    jacobian_output[5, 5] = -x27*x57 + x37*x60
    return jacobian_output


def yaskawa_mpx3500_angular_velocity_jacobian(theta_input: np.ndarray):
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
    x10 = math.cos(alpha_3)
    x11 = math.sin(alpha_3)
    x12 = math.cos(th_3)
    x13 = math.sin(th_3)
    x14 = x1*x8 + x5*x6
    x15 = -x0*x12 - x13*x14
    x16 = x10*x9 - x11*x15
    x17 = math.cos(alpha_5)
    x18 = math.sin(alpha_5)
    x19 = math.cos(th_4)
    x20 = x11*x19
    x21 = math.sin(th_4)
    x22 = x10*x19
    x23 = x0*x2
    x24 = x0*x7
    x25 = x1*x23 - x24*x6
    x26 = x1*x24 + x23*x6
    x27 = 1.0*x12*x3 - x13*x26
    x28 = x10*x25 - x11*x27
    x29 = 1.0*x1
    x30 = 1.0*x2*x6 + x29*x7
    x31 = -x2*x29 + 1.0*x6*x7
    x32 = x13*x31
    x33 = x10*x30 + x11*x32
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x16
    jacobian_output[0, 5] = x16*x17 - x18*(x15*x22 + x20*x9 - x21*(-x0*x13 + x12*x14))
    jacobian_output[1, 1] = x4
    jacobian_output[1, 2] = -x4
    jacobian_output[1, 3] = x25
    jacobian_output[1, 4] = x28
    jacobian_output[1, 5] = x17*x28 - x18*(x20*x25 - x21*(x12*x26 + x13*x4) + x22*x27)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x30
    jacobian_output[2, 4] = x33
    jacobian_output[2, 5] = x17*x33 - x18*(x11*x19*x30 - x12*x21*x31 - x22*x32)
    return jacobian_output


def yaskawa_mpx3500_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x7 = pre_transform_special_symbol_23 - x6
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = math.cos(th_2)
    x11 = math.cos(th_1)
    x12 = 1.0*x11
    x13 = x10*x12
    x14 = x13 + x9
    x15 = math.sin(th_0)
    x16 = x10*x5
    x17 = -x12*x15*x8 + x15*x16
    x18 = d_1*x14 + pre_transform_special_symbol_23 - x6
    x19 = a_0*x12
    x20 = d_1*x17 + x15*x19
    x21 = math.cos(alpha_3)
    x22 = x14*x21
    x23 = math.sin(alpha_3)
    x24 = math.sin(th_3)
    x25 = 1.0*x11*x8 - x16
    x26 = x24*x25
    x27 = x23*x26
    x28 = x22 + x27
    x29 = math.cos(th_3)
    x30 = x13*x15 + x15*x9
    x31 = 1.0*x1*x29 - x24*x30
    x32 = x23*x31
    x33 = x17*x21 - x32
    x34 = d_2*x22 + d_2*x27 + x18
    x35 = d_2*x17*x21 - d_2*x32 + x20
    x36 = math.cos(alpha_5)
    x37 = x28*x36
    x38 = math.sin(alpha_5)
    x39 = math.sin(th_4)
    x40 = math.cos(th_4)
    x41 = x21*x40
    x42 = x38*(x14*x23*x40 - x25*x29*x39 - x26*x41)
    x43 = x37 - x42
    x44 = x33*x36
    x45 = x23*x40
    x46 = x38*(x17*x45 + x31*x41 - x39*(x2*x24 + x29*x30))
    x47 = x44 - x46
    x48 = d_4*x37 - d_4*x42 + x34
    x49 = d_4*x44 - d_4*x46 + x35
    x50 = 1.0*p_on_ee_x
    x51 = 1.0*x15
    x52 = p_on_ee_z*x51
    x53 = x10*x2
    x54 = x11*x2
    x55 = x4*x53 - x54*x8
    x56 = a_0*x54 + d_1*x55
    x57 = x11*x53 + x2*x4*x8
    x58 = -x24*x57 - x29*x51
    x59 = x23*x58
    x60 = x21*x55 - x59
    x61 = d_2*x21*x55 - d_2*x59 + x56
    x62 = x36*x60
    x63 = x38*(-x39*(-x24*x51 + x29*x57) + x41*x58 + x45*x55)
    x64 = x62 - x63
    x65 = d_4*x62 - d_4*x63 + x61
    x66 = x1*x50
    x67 = x0*x15
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = 1.0*x1*x7 - x3
    jacobian_output[0, 3] = -p_on_ee_y*x14 + p_on_ee_z*x17 + x14*x20 - x17*x18
    jacobian_output[0, 4] = -p_on_ee_y*x28 + p_on_ee_z*x33 + x28*x35 - x33*x34
    jacobian_output[0, 5] = -p_on_ee_y*x43 + p_on_ee_z*x47 + x43*x49 - x47*x48
    jacobian_output[1, 0] = x50
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x51 + x52
    jacobian_output[1, 2] = 1.0*x15*x7 - x52
    jacobian_output[1, 3] = p_on_ee_x*x14 - p_on_ee_z*x55 - x14*x56 + x18*x55
    jacobian_output[1, 4] = p_on_ee_x*x28 - p_on_ee_z*x60 - x28*x61 + x34*x60
    jacobian_output[1, 5] = p_on_ee_x*x43 - p_on_ee_z*x64 - x43*x65 + x48*x64
    jacobian_output[2, 1] = -x66 - x67
    jacobian_output[2, 2] = -x1**2*x19 - x15**2*x19 + x66 + x67
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x55 + x17*x56 - x20*x55
    jacobian_output[2, 4] = -p_on_ee_x*x33 + p_on_ee_y*x60 + x33*x61 - x35*x60
    jacobian_output[2, 5] = -p_on_ee_x*x47 + p_on_ee_y*x64 + x47*x65 - x49*x64
    return jacobian_output


def yaskawa_mpx3500_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(14):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for non-branch dispatcher node 0
    # Actually, there is no code

    # Code for explicit solution node 1, solved variable is th_3
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 0] = -d_2*r_21*math.sin(alpha_5)
        R_l[0, 1] = -d_2*r_22*math.sin(alpha_5)
        R_l[0, 2] = -d_2*r_11*math.sin(alpha_5)
        R_l[0, 3] = -d_2*r_12*math.sin(alpha_5)
        R_l[0, 4] = Py - d_2*r_23*math.cos(alpha_5) - d_4*r_23
        R_l[0, 5] = Px - d_2*r_13*math.cos(alpha_5) - d_4*r_13
        R_l[1, 0] = -d_2*r_11*math.sin(alpha_5)
        R_l[1, 1] = -d_2*r_12*math.sin(alpha_5)
        R_l[1, 2] = d_2*r_21*math.sin(alpha_5)
        R_l[1, 3] = d_2*r_22*math.sin(alpha_5)
        R_l[1, 4] = Px - d_2*r_13*math.cos(alpha_5) - d_4*r_13
        R_l[1, 5] = -Py + d_2*r_23*math.cos(alpha_5) + d_4*r_23
        R_l[2, 6] = -d_2*r_31*math.sin(alpha_5)
        R_l[2, 7] = -d_2*r_32*math.sin(alpha_5)
        R_l[3, 0] = r_21*math.sin(alpha_5)
        R_l[3, 1] = r_22*math.sin(alpha_5)
        R_l[3, 2] = r_11*math.sin(alpha_5)
        R_l[3, 3] = r_12*math.sin(alpha_5)
        R_l[3, 4] = r_23*math.cos(alpha_5)
        R_l[3, 5] = r_13*math.cos(alpha_5)
        R_l[4, 0] = r_11*math.sin(alpha_5)
        R_l[4, 1] = r_12*math.sin(alpha_5)
        R_l[4, 2] = -r_21*math.sin(alpha_5)
        R_l[4, 3] = -r_22*math.sin(alpha_5)
        R_l[4, 4] = r_13*math.cos(alpha_5)
        R_l[4, 5] = -r_23*math.cos(alpha_5)
        R_l[5, 6] = -2*Px*d_2*r_11*math.sin(alpha_5) - 2*Py*d_2*r_21*math.sin(alpha_5) - 2*Pz*d_2*r_31*math.sin(alpha_5) + 2*d_2**2*r_11*r_13*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_21*r_23*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_31*r_33*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2*d_4*r_11*r_13*math.sin(alpha_5) + 2*d_2*d_4*r_21*r_23*math.sin(alpha_5) + 2*d_2*d_4*r_31*r_33*math.sin(alpha_5)
        R_l[5, 7] = -2*Px*d_2*r_12*math.sin(alpha_5) - 2*Py*d_2*r_22*math.sin(alpha_5) - 2*Pz*d_2*r_32*math.sin(alpha_5) + 2*d_2**2*r_12*r_13*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_22*r_23*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_32*r_33*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2*d_4*r_12*r_13*math.sin(alpha_5) + 2*d_2*d_4*r_22*r_23*math.sin(alpha_5) + 2*d_2*d_4*r_32*r_33*math.sin(alpha_5)
        R_l[6, 0] = Px*r_31*math.sin(alpha_5) - Pz*r_11*math.sin(alpha_5) + d_4*r_11*r_33*math.sin(alpha_5) - d_4*r_13*r_31*math.sin(alpha_5)
        R_l[6, 1] = Px*r_32*math.sin(alpha_5) - Pz*r_12*math.sin(alpha_5) + d_4*r_12*r_33*math.sin(alpha_5) - d_4*r_13*r_32*math.sin(alpha_5)
        R_l[6, 2] = -Py*r_31*math.sin(alpha_5) + Pz*r_21*math.sin(alpha_5) - d_4*r_21*r_33*math.sin(alpha_5) + d_4*r_23*r_31*math.sin(alpha_5)
        R_l[6, 3] = -Py*r_32*math.sin(alpha_5) + Pz*r_22*math.sin(alpha_5) - d_4*r_22*r_33*math.sin(alpha_5) + d_4*r_23*r_32*math.sin(alpha_5)
        R_l[6, 4] = Px*r_33*math.cos(alpha_5) - Pz*r_13*math.cos(alpha_5)
        R_l[6, 5] = -Py*r_33*math.cos(alpha_5) + Pz*r_23*math.cos(alpha_5)
        R_l[7, 0] = -Py*r_31*math.sin(alpha_5) + Pz*r_21*math.sin(alpha_5) - d_4*r_21*r_33*math.sin(alpha_5) + d_4*r_23*r_31*math.sin(alpha_5)
        R_l[7, 1] = -Py*r_32*math.sin(alpha_5) + Pz*r_22*math.sin(alpha_5) - d_4*r_22*r_33*math.sin(alpha_5) + d_4*r_23*r_32*math.sin(alpha_5)
        R_l[7, 2] = -Px*r_31*math.sin(alpha_5) + Pz*r_11*math.sin(alpha_5) - d_4*r_11*r_33*math.sin(alpha_5) + d_4*r_13*r_31*math.sin(alpha_5)
        R_l[7, 3] = -Px*r_32*math.sin(alpha_5) + Pz*r_12*math.sin(alpha_5) - d_4*r_12*r_33*math.sin(alpha_5) + d_4*r_13*r_32*math.sin(alpha_5)
        R_l[7, 4] = -Py*r_33*math.cos(alpha_5) + Pz*r_23*math.cos(alpha_5)
        R_l[7, 5] = -Px*r_33*math.cos(alpha_5) + Pz*r_13*math.cos(alpha_5)
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
        x0 = math.sin(alpha_5)
        x1 = r_31*x0
        x2 = r_32*x0
        x3 = R_l_inv_60*x1 + R_l_inv_70*x2
        x4 = a_0*x3
        x5 = -x4
        x6 = math.cos(alpha_3)
        x7 = -x6
        x8 = R_l_inv_62*x1 + R_l_inv_72*x2
        x9 = d_1*x8
        x10 = R_l_inv_67*x1 + R_l_inv_77*x2
        x11 = a_0*x6
        x12 = x10*x11
        x13 = R_l_inv_66*x1 + R_l_inv_76*x2
        x14 = math.sin(alpha_3)
        x15 = d_1*x14
        x16 = x13*x15
        x17 = x12 + x16 + x7 + x9
        x18 = math.cos(alpha_5)
        x19 = r_33*x18
        x20 = d_4*r_33
        x21 = d_2*x19
        x22 = -Pz + x20 + x21
        x23 = x22*x8
        x24 = R_l_inv_65*x1 + R_l_inv_75*x2
        x25 = Px**2
        x26 = Py**2
        x27 = Pz**2
        x28 = a_0**2
        x29 = d_1**2
        x30 = d_4**2
        x31 = r_13**2
        x32 = x30*x31
        x33 = r_23**2
        x34 = x30*x33
        x35 = r_33**2
        x36 = x30*x35
        x37 = d_4*x18
        x38 = x31*x37
        x39 = 2*d_2
        x40 = x33*x37
        x41 = x35*x37
        x42 = r_11**2
        x43 = d_2**2
        x44 = x0**2
        x45 = x43*x44
        x46 = x42*x45
        x47 = x18**2
        x48 = x43*x47
        x49 = x31*x48
        x50 = r_21**2
        x51 = x45*x50
        x52 = x33*x48
        x53 = r_31**2
        x54 = x45*x53
        x55 = x35*x48
        x56 = 2*Px*d_2*r_13*x18 + 2*Px*d_4*r_13 + 2*Py*d_2*r_23*x18 + 2*Py*d_4*r_23 + 2*Pz*d_2*r_33*x18 + 2*Pz*d_4*r_33 - x25 - x26 - x27 + x28 + x29 - x32 - x34 - x36 - x38*x39 - x39*x40 - x39*x41 - x46 - x49 - x51 - x52 - x54 - x55
        x57 = x24*x56
        x58 = x14*(R_l_inv_64*x1 + R_l_inv_74*x2)
        x59 = x19 + x23 + x57 - x58
        x60 = 2*R_l_inv_63
        x61 = 2*R_l_inv_73
        x62 = x1*x60 + x2*x61
        x63 = x6*x62
        x64 = 2*d_1
        x65 = x3*x64
        x66 = x63 + x65
        x67 = 4*a_0*d_1*x24 - x66
        x68 = x4 + x58
        x69 = 2*a_0
        x70 = x69*x8
        x71 = a_0*x14
        x72 = 2*x71
        x73 = x13*x72
        x74 = x70 + x73
        x75 = 4*x9
        x76 = 4*x14
        x77 = d_1*x76
        x78 = x13*x77
        x79 = x4 + x59
        x80 = -x16
        x81 = x12 + x80
        x82 = x6 - x9
        x83 = 4*d_1
        x84 = a_0*x83
        x85 = x24*x84 + x66
        x86 = x7 + x9
        x87 = -x12
        x88 = x16 + x87
        x89 = d_1*x6
        x90 = r_11*x0
        x91 = d_4*r_13
        x92 = x90*x91
        x93 = r_21*x0
        x94 = d_4*r_23
        x95 = x93*x94
        x96 = r_13*x18
        x97 = x39*x96
        x98 = r_23*x18
        x99 = x39*x98
        x100 = 2*x21
        x101 = Px*r_11*x0 + Py*r_21*x0 + Pz*r_31*x0 - x1*x100 - x1*x20 - x90*x97 - x92 - x93*x99 - x95
        x102 = r_12*x0
        x103 = x102*x91
        x104 = r_22*x0
        x105 = x104*x94
        x106 = Px*r_12*x0 + Py*r_22*x0 + Pz*r_32*x0 - x100*x2 - x102*x97 - x103 - x104*x99 - x105 - x2*x20
        x107 = R_l_inv_62*x101 + R_l_inv_72*x106
        x108 = d_1*x107
        x109 = -x107*x22
        x110 = R_l_inv_65*x101 + R_l_inv_75*x106
        x111 = -x110*x56
        x112 = d_2*x44
        x113 = x112*x42
        x114 = d_2*x47
        x115 = x114*x31
        x116 = x112*x50
        x117 = x114*x33
        x118 = x112*x53
        x119 = x114*x35
        x120 = Px*x18
        x121 = r_13*x120
        x122 = -x121
        x123 = Py*x18
        x124 = r_23*x123
        x125 = -x124
        x126 = Pz*x19
        x127 = -x126
        x128 = -x108 + x109 + x111 + x113 + x115 + x116 + x117 + x118 + x119 + x122 + x125 + x127 + x38 + x40 + x41 + x89
        x129 = R_l_inv_60*x101 + R_l_inv_70*x106
        x130 = a_0*x129
        x131 = x14*(R_l_inv_64*x101 + R_l_inv_74*x106)
        x132 = x130 + x131
        x133 = R_l_inv_67*x101 + R_l_inv_77*x106
        x134 = x11*x133
        x135 = -x134
        x136 = R_l_inv_66*x101 + R_l_inv_76*x106
        x137 = x136*x15
        x138 = -x137
        x139 = x135 + x138
        x140 = 2*x11
        x141 = x101*x60 + x106*x61
        x142 = x141*x6
        x143 = x129*x64
        x144 = x142 + x143
        x145 = 4*a_0*d_1*x110 - x140 - x144
        x146 = x134 + x137
        x147 = x108 + x109 + x111 + x113 + x115 + x116 + x117 + x118 + x119 + x122 + x125 + x127 + x38 + x40 + x41 + x89
        x148 = x107*x69
        x149 = x136*x72
        x150 = x148 + x149
        x151 = 4*x108
        x152 = x136*x77
        x153 = -x130
        x154 = x131 + x153
        x155 = x135 + x137
        x156 = x110*x84 - x140 + x144
        x157 = x134 + x138
        x158 = Px*x0
        x159 = r_21*x158
        x160 = Py*r_11*x0 + d_4*r_13*r_21*x0 - x159 - x90*x94
        x161 = r_22*x158
        x162 = Py*r_12*x0 + d_4*r_13*r_22*x0 - x102*x94 - x161
        x163 = R_l_inv_60*x160 + R_l_inv_70*x162
        x164 = a_0*x163
        x165 = -x164
        x166 = -x71
        x167 = R_l_inv_62*x160 + R_l_inv_72*x162
        x168 = d_1*x167
        x169 = R_l_inv_67*x160 + R_l_inv_77*x162
        x170 = x11*x169
        x171 = R_l_inv_66*x160 + R_l_inv_76*x162
        x172 = x15*x171
        x173 = x168 + x170 + x172
        x174 = x167*x22
        x175 = R_l_inv_65*x160 + R_l_inv_75*x162
        x176 = x175*x56
        x177 = x14*(R_l_inv_64*x160 + R_l_inv_74*x162)
        x178 = r_13*x123
        x179 = -Px*r_23*x18
        x180 = x174 + x176 - x177 + x178 + x179
        x181 = 2*x15
        x182 = x160*x60 + x162*x61
        x183 = x182*x6
        x184 = x163*x64
        x185 = x183 + x184
        x186 = x181 + x185
        x187 = r_23*x120
        x188 = x164 + x71
        x189 = x167*x69
        x190 = x171*x72
        x191 = 4*x168
        x192 = x171*x77
        x193 = -2*a_0*x167
        x194 = -x190
        x195 = -x168
        x196 = x180 + x188
        x197 = -x172
        x198 = x170 + x197
        x199 = x175*x84
        x200 = -x170
        x201 = x172 + x200
        x202 = r_11*x158
        x203 = 2*Py
        x204 = Pz*x1
        x205 = 2*d_4
        x206 = r_13*x205
        x207 = 2*Pz
        x208 = x20*x207
        x209 = 2*r_23
        x210 = x209*x30
        x211 = r_13*x90
        x212 = r_33*x210
        x213 = x0**3*x43
        x214 = 4*d_2
        x215 = r_23*x214*x37
        x216 = 4*x21
        x217 = x216*x94
        x218 = r_21*x213
        x219 = x214*x40
        x220 = x209*x48
        x221 = r_33*x1
        x222 = 3*x52
        x223 = -2*Px*d_4*r_11*r_23*x0 - 4*Py*d_2*r_11*r_13*x0*x18 - 4*Py*d_2*r_21*r_23*x0*x18 - 4*Py*d_2*r_31*r_33*x0*x18 - 2*Py*d_4*r_11*r_13*x0 - 2*Py*d_4*r_21*r_23*x0 - 2*Py*d_4*r_31*r_33*x0 - 2*Pz*d_4*r_23*r_31*x0 + r_21**3*x213 - r_21*x0*x25 - r_21*x0*x27 - r_21*x0*x30*x31 - r_21*x0*x30*x35 + x1*x212 + x1*x217 + x159*x206 + x202*x203 + x203*x204 + x208*x93 + x210*x211 + x211*x215 + x211*x220 + x218*x42 + x218*x53 + x219*x93 + x220*x221 + x222*x93 + x26*x93 + x34*x93 + x49*x93 + x55*x93
        x224 = -x223
        x225 = 2*Px
        x226 = Py*x0
        x227 = r_23*x205
        x228 = r_13*x210
        x229 = 2*x30
        x230 = r_13*x229
        x231 = r_13*x215
        x232 = x216*x91
        x233 = r_11*x213
        x234 = x214*x38
        x235 = r_13*x220
        x236 = 2*x48
        x237 = r_13*x236
        x238 = 3*x49
        x239 = 4*Px*d_2*r_11*r_13*x0*x18 + 4*Px*d_2*r_21*r_23*x0*x18 + 4*Px*d_2*r_31*r_33*x0*x18 + 2*Px*d_4*r_11*r_13*x0 + 2*Px*d_4*r_21*r_23*x0 + 2*Px*d_4*r_31*r_33*x0 + 2*Py*d_4*r_13*r_21*x0 + 2*Pz*d_4*r_13*r_31*x0 - r_11**3*x213 + r_11*x0*x26 + r_11*x0*x27 + r_11*x0*x30*x33 + r_11*x0*x30*x35 - r_11*x226*x227 - x1*x232 - x159*x203 - x204*x225 - x208*x90 - x221*x230 - x221*x237 - x228*x93 - x231*x93 - x233*x50 - x233*x53 - x234*x90 - x235*x93 - x238*x90 - x25*x90 - x32*x90 - x52*x90 - x55*x90
        x240 = r_12*x158
        x241 = Pz*x2
        x242 = r_22*x213
        x243 = 2*x218
        x244 = r_11*x243
        x245 = r_31*r_32
        x246 = r_33*x2
        x247 = -2*Px*d_4*r_12*r_23*x0 - 4*Py*d_2*r_12*r_13*x0*x18 - 4*Py*d_2*r_22*r_23*x0*x18 - 4*Py*d_2*r_32*r_33*x0*x18 - 2*Py*d_4*r_12*r_13*x0 - 2*Py*d_4*r_22*r_23*x0 - 2*Py*d_4*r_32*r_33*x0 - 2*Pz*d_4*r_23*r_32*x0 + r_12*x244 - r_22*x0*x25 - r_22*x0*x27 - r_22*x0*x30*x31 - r_22*x0*x30*x35 + x102*x228 + x102*x231 + x102*x235 + x104*x208 + x104*x219 + x104*x222 + x104*x26 + x104*x34 + x104*x49 + x104*x55 + x161*x206 + x2*x212 + x2*x217 + x203*x240 + x203*x241 + x220*x246 + x242*x42 + 3*x242*x50 + x242*x53 + x243*x245
        x248 = -x247
        x249 = r_33*x230
        x250 = r_12*x213
        x251 = 2*x233
        x252 = 4*Px*d_2*r_12*r_13*x0*x18 + 4*Px*d_2*r_22*r_23*x0*x18 + 4*Px*d_2*r_32*r_33*x0*x18 + 2*Px*d_4*r_12*r_13*x0 + 2*Px*d_4*r_22*r_23*x0 + 2*Px*d_4*r_32*r_33*x0 + 2*Py*d_4*r_13*r_22*x0 + 2*Pz*d_4*r_13*r_32*x0 + r_12*x0*x26 + r_12*x0*x27 + r_12*x0*x30*x33 + r_12*x0*x30*x35 - r_12*x226*x227 - r_22*x244 - x102*x208 - x102*x234 - x102*x238 - x102*x25 - x102*x32 - x102*x52 - x102*x55 - x104*x228 - x104*x231 - x104*x235 - x161*x203 - x2*x232 - x2*x249 - x225*x241 - x237*x246 - x245*x251 - 3*x250*x42 - x250*x50 - x250*x53
        x253 = r_23**3
        x254 = x18*x30
        x255 = x18**3*x43
        x256 = x114*x205
        x257 = r_23*x255
        x258 = 2*r_21*x45
        x259 = r_11*x258
        x260 = r_31*x19
        x261 = -2*Py*d_2*x31*x47 - 2*Py*d_2*x33*x47 - 2*Py*d_2*x35*x47 - 2*Py*d_2*x42*x44 - 2*Py*d_2*x44*x50 - 2*Py*d_2*x44*x53 - 2*Py*d_4*x18*x31 - 2*Py*d_4*x18*x33 - 2*Py*d_4*x18*x35 - r_23*x18*x25 - r_23*x18*x27 + x113*x227 + x115*x227 + x116*x227 + x118*x227 + x119*x227 + x121*x203 + x126*x203 + x253*x254 + x253*x255 + x253*x256 + x257*x31 + x257*x35 + x258*x260 + x259*x96 + x26*x98 + x32*x98 + x36*x98 + x46*x98 + 3*x51*x98 + x54*x98
        x262 = -x261
        x263 = r_13**3
        x264 = r_13*x255
        x265 = 2*r_11*x45
        x266 = 2*Px*d_2*x31*x47 + 2*Px*d_2*x33*x47 + 2*Px*d_2*x35*x47 + 2*Px*d_2*x42*x44 + 2*Px*d_2*x44*x50 + 2*Px*d_2*x44*x53 + 2*Px*d_4*x18*x31 + 2*Px*d_4*x18*x33 + 2*Px*d_4*x18*x35 + r_13*x18*x26 + r_13*x18*x27 - x113*x206 - x116*x206 - x117*x206 - x118*x206 - x119*x206 - x126*x225 - x187*x203 - x25*x96 - x254*x263 - x255*x263 - x256*x263 - x259*x98 - x260*x265 - x264*x33 - x264*x35 - x34*x96 - x36*x96 - 3*x46*x96 - x51*x96 - x54*x96
        x267 = R_l_inv_02*x224 + R_l_inv_12*x248 + R_l_inv_22*x239 + R_l_inv_32*x252 + R_l_inv_42*x262 + R_l_inv_52*x266
        x268 = d_1*x267
        x269 = R_l_inv_07*x224 + R_l_inv_17*x248 + R_l_inv_27*x239 + R_l_inv_37*x252 + R_l_inv_47*x262 + R_l_inv_57*x266
        x270 = x11*x269
        x271 = R_l_inv_06*x224 + R_l_inv_16*x248 + R_l_inv_26*x239 + R_l_inv_36*x252 + R_l_inv_46*x262 + R_l_inv_56*x266
        x272 = x15*x271
        x273 = x268 + x270 + x272
        x274 = x14*(R_l_inv_04*x224 + R_l_inv_14*x248 + R_l_inv_24*x239 + R_l_inv_34*x252 + R_l_inv_44*x262 + R_l_inv_54*x266)
        x275 = -x274
        x276 = x22*x267
        x277 = R_l_inv_05*x224 + R_l_inv_15*x248 + R_l_inv_25*x239 + R_l_inv_35*x252 + R_l_inv_45*x262 + R_l_inv_55*x266
        x278 = x277*x56
        x279 = x276 + x278
        x280 = x275 + x279
        x281 = R_l_inv_00*x224 + R_l_inv_10*x248 + R_l_inv_20*x239 + R_l_inv_30*x252 + R_l_inv_40*x262 + R_l_inv_50*x266
        x282 = a_0*x281
        x283 = x69*x89
        x284 = -x282 - x283
        x285 = x28*x6
        x286 = 2*x285
        x287 = 2*R_l_inv_03
        x288 = 2*R_l_inv_23
        x289 = 2*R_l_inv_13
        x290 = 2*R_l_inv_33
        x291 = 2*R_l_inv_43
        x292 = 2*R_l_inv_53
        x293 = x224*x287 + x239*x288 + x248*x289 + x252*x290 + x262*x291 + x266*x292
        x294 = x293*x6
        x295 = x281*x64
        x296 = x29*x6
        x297 = 2*x296
        x298 = x294 + x295 + x297
        x299 = x286 + x298
        x300 = 4*a_0*d_1*x277 - x299
        x301 = x282 + x283
        x302 = x274 + x301
        x303 = -x276 - x278
        x304 = x267*x69
        x305 = x271*x72
        x306 = x304 + x305
        x307 = -x286 + x298
        x308 = 4*x268
        x309 = x271*x77
        x310 = -x268
        x311 = x280 + x301
        x312 = -x272
        x313 = x270 + x312
        x314 = x277*x84 + x299
        x315 = x268 - x270
        x316 = R_l_inv_02*x239 + R_l_inv_12*x252 + R_l_inv_22*x223 + R_l_inv_32*x247 + R_l_inv_42*x266 + R_l_inv_52*x261
        x317 = d_1*x316
        x318 = R_l_inv_07*x239 + R_l_inv_17*x252 + R_l_inv_27*x223 + R_l_inv_37*x247 + R_l_inv_47*x266 + R_l_inv_57*x261
        x319 = x11*x318
        x320 = R_l_inv_06*x239 + R_l_inv_16*x252 + R_l_inv_26*x223 + R_l_inv_36*x247 + R_l_inv_46*x266 + R_l_inv_56*x261
        x321 = x15*x320
        x322 = x317 + x319 + x321
        x323 = R_l_inv_00*x239 + R_l_inv_10*x252 + R_l_inv_20*x223 + R_l_inv_30*x247 + R_l_inv_40*x266 + R_l_inv_50*x261
        x324 = a_0*x323
        x325 = x22*x316
        x326 = R_l_inv_05*x239 + R_l_inv_15*x252 + R_l_inv_25*x223 + R_l_inv_35*x247 + R_l_inv_45*x266 + R_l_inv_55*x261
        x327 = x326*x56
        x328 = x325 + x327
        x329 = -x324 + x328
        x330 = x14*x28
        x331 = x14*x29
        x332 = x14*(R_l_inv_04*x239 + R_l_inv_14*x252 + R_l_inv_24*x223 + R_l_inv_34*x247 + R_l_inv_44*x266 + R_l_inv_54*x261)
        x333 = x330 + x331 - x332
        x334 = x223*x288 + x239*x287 + x247*x290 + x252*x289 + x261*x292 + x266*x291
        x335 = x334*x6
        x336 = x323*x64
        x337 = -x335 - x336
        x338 = x71*x83
        x339 = x326*x84
        x340 = x338 + x339
        x341 = -x330 - x331 + x332
        x342 = x324 - x325 - x327
        x343 = x316*x69
        x344 = x320*x72
        x345 = x343 + x344
        x346 = x335 + x336
        x347 = 4*x317
        x348 = x320*x77
        x349 = -x317
        x350 = -x321
        x351 = x319 + x350
        x352 = x324 + x328 + x333
        x353 = x317 - x319
        x354 = Px*x206
        x355 = Py*x227
        x356 = r_33*x211
        x357 = r_31*x213
        x358 = x214*x41
        x359 = r_33*x220
        x360 = 3*x55
        x361 = 2*Px*d_4*r_11*r_33*x0 + 2*Py*d_4*r_21*r_33*x0 + 4*Pz*d_2*r_11*r_13*x0*x18 + 4*Pz*d_2*r_21*r_23*x0*x18 + 4*Pz*d_2*r_31*r_33*x0*x18 + 2*Pz*d_4*r_11*r_13*x0 + 2*Pz*d_4*r_21*r_23*x0 + 2*Pz*d_4*r_31*r_33*x0 - r_21*x207*x226 - r_31**3*x213 + r_31*x0*x25 + r_31*x0*x26 + r_31*x0*x30*x31 + r_31*x0*x30*x33 - x1*x27 - x1*x354 - x1*x355 - x1*x358 - x1*x36 - x1*x360 - x1*x49 - x1*x52 - x202*x207 - x212*x93 - x216*x92 - x216*x95 - x229*x356 - x236*x356 - x357*x42 - x357*x50 - x359*x93
        x362 = r_32*x213
        x363 = 2*Px*d_4*r_12*r_33*x0 + 2*Py*d_4*r_22*r_33*x0 + 4*Pz*d_2*r_12*r_13*x0*x18 + 4*Pz*d_2*r_22*r_23*x0*x18 + 4*Pz*d_2*r_32*r_33*x0*x18 + 2*Pz*d_4*r_12*r_13*x0 + 2*Pz*d_4*r_22*r_23*x0 + 2*Pz*d_4*r_32*r_33*x0 - r_12*r_31*x251 - r_22*r_31*x243 - r_22*x207*x226 + r_32*x0*x25 + r_32*x0*x26 + r_32*x0*x30*x31 + r_32*x0*x30*x33 - r_33*x102*x237 - x102*x249 - x103*x216 - x104*x212 - x104*x359 - x105*x216 - x2*x27 - x2*x354 - x2*x355 - x2*x358 - x2*x36 - x2*x360 - x2*x49 - x2*x52 - x207*x240 - x362*x42 - x362*x50 - 3*x362*x53
        x364 = R_l_inv_60*x361 + R_l_inv_70*x363
        x365 = a_0*x364
        x366 = x14*(R_l_inv_64*x361 + R_l_inv_74*x363)
        x367 = R_l_inv_62*x361 + R_l_inv_72*x363
        x368 = -x22*x367
        x369 = R_l_inv_65*x361 + R_l_inv_75*x363
        x370 = -x369*x56
        x371 = x19*x27
        x372 = r_33**3
        x373 = x255*x372
        x374 = x254*x372
        x375 = -x19*x25
        x376 = -x19*x26
        x377 = r_33*x255
        x378 = x31*x377
        x379 = x33*x377
        x380 = x19*x32
        x381 = x19*x34
        x382 = -x113*x207
        x383 = -x115*x207
        x384 = -x116*x207
        x385 = -x117*x207
        x386 = -x118*x207
        x387 = -x119*x207
        x388 = -x207*x38
        x389 = -x207*x40
        x390 = -x207*x41
        x391 = x121*x207
        x392 = x124*x207
        x393 = x256*x372
        x394 = x19*x46
        x395 = x19*x51
        x396 = 2*x20
        x397 = x113*x396
        x398 = x115*x396
        x399 = x116*x396
        x400 = x117*x396
        x401 = x118*x396
        x402 = 3*x19*x54
        x403 = r_31*x265*x96
        x404 = r_31*x258*x98
        x405 = x365 + x366 + x368 + x370 + x371 + x373 + x374 + x375 + x376 + x378 + x379 + x380 + x381 + x382 + x383 + x384 + x385 + x386 + x387 + x388 + x389 + x390 + x391 + x392 + x393 + x394 + x395 + x397 + x398 + x399 + x400 + x401 + x402 + x403 + x404
        x406 = R_l_inv_67*x361 + R_l_inv_77*x363
        x407 = x11*x406
        x408 = -x407
        x409 = R_l_inv_66*x361 + R_l_inv_76*x363
        x410 = x15*x409
        x411 = -x410
        x412 = x408 + x411
        x413 = d_1*x367
        x414 = x285 - x296 - x413
        x415 = x361*x60 + x363*x61
        x416 = x415*x6
        x417 = x364*x64
        x418 = x416 + x417
        x419 = 4*a_0*d_1*x369 - x418
        x420 = x407 + x410
        x421 = -x285 + x296 + x413
        x422 = x409*x72
        x423 = x418 + x422
        x424 = 4*x6
        x425 = a_0*d_1*x424 + x367*x69
        x426 = x409*x77
        x427 = x28*x424 + x29*x424 + 4*x413
        x428 = x368 + x370 + x371 + x373 + x374 + x375 + x376 + x378 + x379 + x380 + x381 + x382 + x383 + x384 + x385 + x386 + x387 + x388 + x389 + x390 + x391 + x392 + x393 + x394 + x395 + x397 + x398 + x399 + x400 + x401 + x402 + x403 + x404 + x421
        x429 = -x365
        x430 = x366 + x429
        x431 = x408 + x410
        x432 = x369*x84 + x418
        x433 = x407 + x411
        x434 = x368 + x370 + x371 + x373 + x374 + x375 + x376 + x378 + x379 + x380 + x381 + x382 + x383 + x384 + x385 + x386 + x387 + x388 + x389 + x390 + x391 + x392 + x393 + x394 + x395 + x397 + x398 + x399 + x400 + x401 + x402 + x403 + x404 + x414
        x435 = x14*x62
        x436 = x10*x181
        x437 = -x435 - x436
        x438 = -x76
        x439 = 4*x71
        x440 = x10*x439
        x441 = x435 - x436
        x442 = 8*R_l_inv_63
        x443 = 8*R_l_inv_73
        x444 = x133*x181
        x445 = x14*x141
        x446 = x445 + x72
        x447 = x133*x439
        x448 = x14*x182
        x449 = x169*x181
        x450 = -x448 - x449
        x451 = x169*x439
        x452 = x448 - x449
        x453 = x181*x269
        x454 = x14*x293 + 2*x330 - 2*x331
        x455 = -x453 - x454
        x456 = x269*x439
        x457 = -x453 + x454
        x458 = -8*d_1*x71
        x459 = 8*R_l_inv_03
        x460 = 8*R_l_inv_23
        x461 = 8*R_l_inv_13
        x462 = 8*R_l_inv_33
        x463 = 8*R_l_inv_43
        x464 = 8*R_l_inv_53
        x465 = x181*x318
        x466 = x14*x334
        x467 = -x465 - x466
        x468 = x318*x439
        x469 = -x465 + x466
        x470 = -x338
        x471 = x14*x415
        x472 = x181*x406
        x473 = x471 + x472
        x474 = 4*x330
        x475 = 4*x331
        x476 = x474 + x475
        x477 = x474 - x475
        x478 = x19 + x23 + x57
        x479 = x478 + x86
        x480 = x5 + x58
        x481 = x478 + x82
        x482 = -x131
        x483 = x130 + x482
        x484 = x153 + x482
        x485 = x174 + x176 + x177 + x178 + x179
        x486 = x168 + x485
        x487 = x165 + x71
        x488 = x195 + x485
        x489 = -x181 + x185
        x490 = x164 + x166
        x491 = x268 + x313
        x492 = x279 + x302
        x493 = x317 + x351
        x494 = x324 + x328 + x341
        x495 = -x366
        x496 = x365 + x495
        x497 = x429 + x495
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = -x17 - x5 - x59
        A[0, 1] = x67
        A[0, 2] = x17 - x19 - x23 - x57 + x68
        A[0, 3] = x66 + x74
        A[0, 4] = 4*x6 - x75 - x78
        A[0, 5] = -x63 - x65 + x74
        A[0, 6] = -x79 - x81 - x82
        A[0, 7] = x85
        A[0, 8] = -x79 - x86 - x88
        A[1, 0] = x128 + x132 + x139
        A[1, 1] = x145
        A[1, 2] = x132 + x146 + x147
        A[1, 3] = x144 + x150
        A[1, 4] = -x151 - x152
        A[1, 5] = -x142 - x143 + x150
        A[1, 6] = x147 + x154 + x155
        A[1, 7] = x156
        A[1, 8] = x128 + x154 + x157
        A[2, 0] = -x165 - x166 - x173 - x180
        A[2, 1] = 4*a_0*d_1*x175 - x186
        A[2, 2] = x173 - x174 - x176 + x177 - x178 + x187 + x188
        A[2, 3] = x186 + x189 + x190
        A[2, 4] = -x191 - x192
        A[2, 5] = -x186 - x193 - x194
        A[2, 6] = -x195 - x196 - x198
        A[2, 7] = x186 + x199
        A[2, 8] = -x168 - x196 - x201
        A[3, 0] = -x273 - x280 - x284
        A[3, 1] = x300
        A[3, 2] = x273 + x302 + x303
        A[3, 3] = x306 + x307
        A[3, 4] = -x308 - x309
        A[3, 5] = x286 - x294 - x295 - x297 + x306
        A[3, 6] = -x310 - x311 - x313
        A[3, 7] = x314
        A[3, 8] = -x272 - x311 - x315
        A[4, 0] = -x322 - x329 - x333
        A[4, 1] = x337 + x340
        A[4, 2] = x322 + x341 + x342
        A[4, 3] = x345 + x346
        A[4, 4] = -x347 - x348
        A[4, 5] = x337 + x345
        A[4, 6] = -x349 - x351 - x352
        A[4, 7] = x340 + x346
        A[4, 8] = -x321 - x352 - x353
        A[5, 0] = x405 + x412 + x414
        A[5, 1] = x419
        A[5, 2] = x405 + x420 + x421
        A[5, 3] = x423 + x425
        A[5, 4] = -x426 - x427
        A[5, 5] = -x416 - x417 + x422 + x425
        A[5, 6] = x428 + x430 + x431
        A[5, 7] = x432
        A[5, 8] = x430 + x433 + x434
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x437
        B[0, 1] = x438 + x440
        B[0, 2] = x441
        B[0, 3] = x76
        B[0, 4] = x14*(-x1*x442 - x2*x443)
        B[0, 5] = x438
        B[0, 6] = x441
        B[0, 7] = x440 + x76
        B[0, 8] = x437
        B[1, 0] = -x444 - x446
        B[1, 1] = x447
        B[1, 2] = -x444 + x446
        B[1, 4] = x14*(-x101*x442 - x106*x443)
        B[1, 6] = -x444 + x445 - x72
        B[1, 7] = x447
        B[1, 8] = -x444 - x445 + x72
        B[2, 0] = x450
        B[2, 1] = x451
        B[2, 2] = x452
        B[2, 4] = x14*(-x160*x442 - x162*x443)
        B[2, 6] = x452
        B[2, 7] = x451
        B[2, 8] = x450
        B[3, 0] = x455
        B[3, 1] = x456
        B[3, 2] = x457
        B[3, 3] = x458
        B[3, 4] = -x14*(x224*x459 + x239*x460 + x248*x461 + x252*x462 + x262*x463 + x266*x464) + 8*x330 + 8*x331
        B[3, 5] = x458
        B[3, 6] = x457
        B[3, 7] = x456
        B[3, 8] = x455
        B[4, 0] = x467
        B[4, 1] = x468
        B[4, 2] = x469
        B[4, 4] = x14*(-x223*x460 - x239*x459 - x247*x462 - x252*x461 - x261*x464 - x266*x463)
        B[4, 6] = x469
        B[4, 7] = x468
        B[4, 8] = x467
        B[5, 0] = -x470 - x473
        B[5, 1] = 4*a_0*x14*x406 - x476
        B[5, 2] = x338 + x471 - x472
        B[5, 3] = -x477
        B[5, 4] = x14*(-x361*x442 - x363*x443)
        B[5, 5] = x477
        B[5, 6] = -x338 + x471 - x472
        B[5, 7] = x406*x439 + x476
        B[5, 8] = -x338 - x473
        C = np.zeros(shape=(6, 9))
        C[0, 0] = -x479 - x480 - x81
        C[0, 1] = x67
        C[0, 2] = -x480 - x481 - x88
        C[0, 3] = x66 + x70 - x73
        C[0, 4] = x424 - x75 + x78
        C[0, 5] = 2*a_0*x8 - x66 - x73
        C[0, 6] = -x12 - x16 - x481 - x68
        C[0, 7] = x85
        C[0, 8] = -x479 - x68 - x80 - x87
        C[1, 0] = x128 + x155 + x483
        C[1, 1] = x145
        C[1, 2] = x147 + x157 + x483
        C[1, 3] = x144 + x148 - x149
        C[1, 4] = -x151 + x152
        C[1, 5] = 2*a_0*x107 - x144 - x149
        C[1, 6] = x139 + x147 + x484
        C[1, 7] = x156
        C[1, 8] = x128 + x146 + x484
        C[2, 0] = -x198 - x486 - x487
        C[2, 1] = x181 - x183 - x184 + x199
        C[2, 2] = -x201 - x487 - x488
        C[2, 3] = x189 + x194 + x489
        C[2, 4] = -x191 + x192
        C[2, 5] = -x190 - x193 - x489
        C[2, 6] = -x170 - x172 - x488 - x490
        C[2, 7] = x199 + x489
        C[2, 8] = -x197 - x200 - x486 - x490
        C[3, 0] = -x274 - x279 - x284 - x491
        C[3, 1] = x300
        C[3, 2] = x275 + x301 + x303 + x491
        C[3, 3] = x304 - x305 + x307
        C[3, 4] = -x308 + x309
        C[3, 5] = 2*a_0*x267 - x305 - x307
        C[3, 6] = -x270 - x272 - x310 - x492
        C[3, 7] = x314
        C[3, 8] = -x312 - x315 - x492
        C[4, 0] = -x329 - x341 - x493
        C[4, 1] = 4*a_0*d_1*x326 - x338 - x346
        C[4, 2] = x333 + x342 + x493
        C[4, 3] = x343 - x344 + x346
        C[4, 4] = -x347 + x348
        C[4, 5] = 2*a_0*x316 - x344 - x346
        C[4, 6] = -x319 - x321 - x349 - x494
        C[4, 7] = x339 + x346 + x470
        C[4, 8] = -x350 - x353 - x494
        C[5, 0] = x431 + x434 + x496
        C[5, 1] = x419
        C[5, 2] = x428 + x433 + x496
        C[5, 3] = x418 - x422 + x425
        C[5, 4] = x426 - x427
        C[5, 5] = 4*a_0*d_1*x6 + 2*a_0*x367 - x423
        C[5, 6] = x412 + x428 + x497
        C[5, 7] = x432
        C[5, 8] = x420 + x434 + x497
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[4] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor()
    # Finish code for explicit solution node 0

    # Code for non-branch dispatcher node 2
    # Actually, there is no code

    # Code for explicit solution node 3, solved variable is th_2
    def ExplicitSolutionNode_node_3_solve_th_2_processor():
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
            th_3 = this_solution[4]
            condition_0: bool = (((2*abs(a_0*d_2*math.sin(alpha_3)*math.sin(th_3)) >= zero_tolerance)) or ((abs(2*a_0*d_1 + 2*a_0*d_2*math.cos(alpha_3)) >= zero_tolerance)) or ((abs(-a_0**2 - d_1**2 - 2*d_1*d_2*math.cos(alpha_3) - d_2**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)))
            if condition_0:
                x0 = 2*a_0
                x1 = d_2*math.cos(alpha_3)
                x2 = -d_1*x0 - x0*x1
                x3 = math.sin(alpha_3)
                x4 = math.sin(th_3)
                x5 = math.atan2(x2, d_2*x0*x3*x4)
                x6 = a_0**2
                x7 = d_2**2
                x8 = -d_1**2 - 2*d_1*x1 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x6 - x7
                x9 = safe_sqrt(x2**2 + 4*x3**2*x4**2*x6*x7 - x8**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x5 + math.atan2(x9, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[3] = x5 + math.atan2(-x9, x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_2_processor()
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
            checked_result: bool = (((abs(Px - d_4*r_13) <= 1.0e-6)) and ((abs(Py - d_4*r_23) <= 1.0e-6)) and ((abs(d_2*math.sin(alpha_3)*math.cos(th_3)) <= 1.0e-6)))
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
            condition_0: bool = (((abs(d_2*math.sin(alpha_3)*math.cos(th_3)) >= zero_tolerance)) or ((abs(Px - d_4*r_13) >= zero_tolerance)) or ((abs(Py - d_4*r_23) >= zero_tolerance)))
            if condition_0:
                x0 = Px - d_4*r_13
                x1 = -Py + d_4*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.sin(alpha_3)
                x4 = math.cos(th_3)
                x5 = safe_sqrt(-d_2**2*x3**2*x4**2 + x0**2 + x1**2)
                x6 = d_2*x3*x4
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x2 + math.atan2(x5, x6)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                this_solution[1] = x2 + math.atan2(-x5, x6)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_0_processor()
    # Finish code for explicit solution node 5

    # Code for equation all-zero dispatcher node 6
    def EquationAllZeroDispatcherNode_node_6_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[1]
            checked_result: bool = (((abs(Pz - d_4*r_33) <= 1.0e-6)) and ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*r_13*math.cos(th_0) - d_4*r_23*math.sin(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(7, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_6_processor()
    # Finish code for equation all-zero dispatcher node 6

    # Code for explicit solution node 7, solved variable is th_1
    def ExplicitSolutionNode_node_7_solve_th_1_processor():
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
            th_0 = this_solution[1]
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(Pz - d_4*r_33) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*r_13*math.cos(th_0) - d_4*r_23*math.sin(th_0)) >= 1.0e-6)))
            if condition_0:
                x0 = Pz - d_4*r_33
                x1 = math.sin(th_2)
                x2 = math.cos(alpha_3)
                x3 = math.cos(th_2)
                x4 = d_2*math.sin(alpha_3)*math.sin(th_3)
                x5 = -a_0 + d_1*x1 + d_2*x1*x2 - x3*x4
                x6 = d_1*x3 + d_2*x2*x3 + x1*x4
                x7 = math.cos(th_0)
                x8 = math.sin(th_0)
                x9 = -Px*x7 - Py*x8 + d_4*r_13*x7 + d_4*r_23*x8
                this_solution[2] = math.atan2(x0*x5 - x6*x9, x0*x6 + x5*x9)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor()
    # Finish code for explicit solution node 7

    # Code for non-branch dispatcher node 8
    # Actually, there is no code

    # Code for explicit solution node 9, solved variable is negative_th_2_positive_th_1__soa
    def ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor():
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
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[0] = th_1 - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor()
    # Finish code for explicit solution node 8

    # Code for non-branch dispatcher node 10
    # Actually, there is no code

    # Code for explicit solution node 11, solved variable is th_4
    def ExplicitSolutionNode_node_11_solve_th_4_processor():
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            th_0 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (((abs(math.sin(alpha_5)) >= zero_tolerance)) or ((abs(math.sin(alpha_3)*math.sin(alpha_5)) >= zero_tolerance)) or ((abs(r_13*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_0) + r_23*math.sin(negative_th_2_positive_th_1__soa)*math.sin(th_0) + r_33*math.cos(negative_th_2_positive_th_1__soa) - math.cos(alpha_3)*math.cos(alpha_5)) >= zero_tolerance)) or ((abs(-r_13*(-math.sin(th_0)*math.sin(th_3) + math.cos(negative_th_2_positive_th_1__soa)*math.cos(th_0)*math.cos(th_3)) - r_23*(math.sin(th_0)*math.cos(negative_th_2_positive_th_1__soa)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_3)) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(alpha_5)
                x1 = math.sin(negative_th_2_positive_th_1__soa)
                x2 = math.cos(th_3)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_0)
                x6 = math.cos(negative_th_2_positive_th_1__soa)
                this_solution[5] = math.atan2(x0*(r_13*(x2*x4*x6 - x3*x5) + r_23*(x2*x5*x6 + x3*x4) - r_33*x1*x2), x0*(-r_13*x1*x4 - r_23*x1*x5 - r_33*x6 + math.cos(alpha_3)*math.cos(alpha_5))/math.sin(alpha_3))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_4_processor()
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            th_0 = this_solution[1]
            th_1 = this_solution[2]
            condition_0: bool = (((abs(d_2*math.sin(alpha_5)) >= zero_tolerance)) or ((abs(-a_0*(r_31*math.sin(th_1) - (r_11*math.cos(th_0) + r_21*math.sin(th_0))*math.cos(th_1)) + d_1*(r_11*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_0) + r_21*math.sin(negative_th_2_positive_th_1__soa)*math.sin(th_0) + r_31*math.cos(negative_th_2_positive_th_1__soa)) + inv_Px) >= zero_tolerance)) or ((abs(-a_0*(r_32*math.sin(th_1) - (r_12*math.cos(th_0) + r_22*math.sin(th_0))*math.cos(th_1)) + d_1*(r_12*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_0) + r_22*math.sin(negative_th_2_positive_th_1__soa)*math.sin(th_0) + r_32*math.cos(negative_th_2_positive_th_1__soa)) + inv_Py) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(negative_th_2_positive_th_1__soa)
                x1 = math.sin(negative_th_2_positive_th_1__soa)
                x2 = math.cos(th_0)
                x3 = r_11*x2
                x4 = math.sin(th_0)
                x5 = r_21*x4
                x6 = math.sin(th_1)
                x7 = math.cos(th_1)
                x8 = 1/(d_2*math.sin(alpha_5))
                x9 = r_12*x2
                x10 = r_22*x4
                this_solution[6] = math.atan2(x8*(-a_0*(-r_31*x6 + x7*(x3 + x5)) - d_1*(r_31*x0 + x1*x3 + x1*x5) - inv_Px), x8*(-a_0*(-r_32*x6 + x7*(x10 + x9)) - d_1*(r_32*x0 + x1*x10 + x1*x9) - inv_Py))
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
        value_at_0 = ik_out_i[1]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[2]  # th_1
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


def yaskawa_mpx3500_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = yaskawa_mpx3500_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = ik_out_i[2] - th_2_offset_original2raw
        ik_out_i[3] = add_revolute_offset(ik_out_i[3], -th_3_offset_original2raw)
        ik_out_i[4] = add_revolute_offset(ik_out_i[4], -th_4_offset_original2raw)
        ik_out_i[5] = add_revolute_offset(ik_out_i[5], -th_5_offset_original2raw)
        ee_pose_i = yaskawa_mpx3500_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                _yaik_seed_candidates.strict_witness = True
                _yaik_seed_candidates.clear()
            elif not _yaik_seed_candidates.strict_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -1.5707963267948966, -3.141592653589793, -1.5707963267948966, -6.283185307179586, -1.5707963267948966], [3.141592653589793, 4.71238898038469, 3.141592653589793, 4.71238898038469, 0.0, 4.71238898038469], [True, True, True, True, True, True], [False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = yaskawa_mpx3500_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
                if _yaik_seed_candidates is not None and np.max(np.abs(snapped_pose-T_ee)) < 1e-7:
                    _yaik_seed_candidates.strict_witness = True
                    _yaik_seed_candidates.clear()
    return ik_output


def yaskawa_mpx3500_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = RefinementSeeds()
    for transform in (yaskawa_mpx3500_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in yaskawa_mpx3500_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: yaskawa_mpx3500_fk(q)
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
            for transform in (yaskawa_mpx3500_ik_target_original_to_raw,):
                raw_target = transform(shifted)
                for raw_candidate in yaskawa_mpx3500_ik_solve_raw(raw_target, *seed_parameters):
                    seed = raw_candidate.reshape(-1).copy()
                    seed[0] = seed[0] - th_0_offset_original2raw
                    seed[1] = add_revolute_offset(seed[1], -th_1_offset_original2raw)
                    seed[2] = seed[2] - th_2_offset_original2raw
                    seed[3] = add_revolute_offset(seed[3], -th_3_offset_original2raw)
                    seed[4] = add_revolute_offset(seed[4], -th_4_offset_original2raw)
                    seed[5] = add_revolute_offset(seed[5], -th_5_offset_original2raw)
                    retain_ik_refinement_seed(refinement_seeds, refinement_fk, T_ee, seed)
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                refinement_seeds.strict_witness = True
                break
    return result


def test_ik_solve_yaskawa_mpx3500():
    theta_in = np.random.random(size=(6, ))
    ee_pose = yaskawa_mpx3500_fk(theta_in)
    ik_output = yaskawa_mpx3500_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = yaskawa_mpx3500_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_yaskawa_mpx3500()
