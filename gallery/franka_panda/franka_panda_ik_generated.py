import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 50
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_3: float = 0.00825
a_5: float = 0.088
d_2: float = 0.316
d_4: float = 0.384
post_transform_d5: float = 0.107
post_transform_sqrt2_over2: float = 0.707107
pre_transform_d4: float = 0.333

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = 3.141592653589793
th_2_offset_original2raw: float = 3.141592653589793
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 0.0
th_5_offset_original2raw: float = 3.141592653589793
th_6_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def franka_panda_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = (1/2)*(r_11 + r_12)/post_transform_sqrt2_over2
    ee_transformed[0, 1] = (1/2)*(-r_11 + r_12)/post_transform_sqrt2_over2
    ee_transformed[0, 2] = r_13
    ee_transformed[0, 3] = Px - post_transform_d5*r_13
    ee_transformed[1, 0] = (1/2)*(r_21 + r_22)/post_transform_sqrt2_over2
    ee_transformed[1, 1] = (1/2)*(-r_21 + r_22)/post_transform_sqrt2_over2
    ee_transformed[1, 2] = r_23
    ee_transformed[1, 3] = Py - post_transform_d5*r_23
    ee_transformed[2, 0] = (1/2)*(r_31 + r_32)/post_transform_sqrt2_over2
    ee_transformed[2, 1] = (1/2)*(-r_31 + r_32)/post_transform_sqrt2_over2
    ee_transformed[2, 2] = r_33
    ee_transformed[2, 3] = Pz - post_transform_d5*r_33 - pre_transform_d4
    return ee_transformed


def franka_panda_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = post_transform_sqrt2_over2*(r_11 - r_12)
    ee_transformed[0, 1] = post_transform_sqrt2_over2*(r_11 + r_12)
    ee_transformed[0, 2] = r_13
    ee_transformed[0, 3] = Px + post_transform_d5*r_13
    ee_transformed[1, 0] = post_transform_sqrt2_over2*(r_21 - r_22)
    ee_transformed[1, 1] = post_transform_sqrt2_over2*(r_21 + r_22)
    ee_transformed[1, 2] = r_23
    ee_transformed[1, 3] = Py + post_transform_d5*r_23
    ee_transformed[2, 0] = post_transform_sqrt2_over2*(r_31 - r_32)
    ee_transformed[2, 1] = post_transform_sqrt2_over2*(r_31 + r_32)
    ee_transformed[2, 2] = r_33
    ee_transformed[2, 3] = Pz + post_transform_d5*r_33 + pre_transform_d4
    return ee_transformed


def franka_panda_ik_target_original_to_raw_legacy_frame(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = post_transform_sqrt2_over2*(r_11 + r_12)
    ee_transformed[0, 1] = post_transform_sqrt2_over2*(-r_11 + r_12)
    ee_transformed[0, 2] = r_13
    ee_transformed[0, 3] = Px - post_transform_d5*r_13
    ee_transformed[1, 0] = post_transform_sqrt2_over2*(r_21 + r_22)
    ee_transformed[1, 1] = post_transform_sqrt2_over2*(-r_21 + r_22)
    ee_transformed[1, 2] = r_23
    ee_transformed[1, 3] = Py - post_transform_d5*r_23
    ee_transformed[2, 0] = post_transform_sqrt2_over2*(r_31 + r_32)
    ee_transformed[2, 1] = post_transform_sqrt2_over2*(-r_31 + r_32)
    ee_transformed[2, 2] = r_33
    ee_transformed[2, 3] = Pz - post_transform_d5*r_33 - pre_transform_d4
    return ee_transformed


def franka_panda_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_6)
    x1 = math.cos(th_4)
    x2 = math.sin(th_0)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.sin(th_2)
    x6 = math.cos(th_0)
    x7 = math.cos(th_1)
    x8 = -x4 + x5*x6*x7
    x9 = math.sin(th_4)
    x10 = math.sin(th_3)
    x11 = math.sin(th_1)
    x12 = x11*x6
    x13 = math.cos(th_3)
    x14 = x2*x5
    x15 = x3*x6
    x16 = x14 + x15*x7
    x17 = -x10*x12 + x13*x16
    x18 = x1*x8 - x17*x9
    x19 = math.sin(th_6)
    x20 = math.sin(th_5)
    x21 = x10*x16 + x12*x13
    x22 = math.cos(th_5)
    x23 = x1*x17 + x8*x9
    x24 = -x20*x21 + x22*x23
    x25 = post_transform_sqrt2_over2*(x0*x18 - x19*x24)
    x26 = x0*x24 + x18*x19
    x27 = x20*x23 + x21*x22
    x28 = x14*x7 + x15
    x29 = x11*x2
    x30 = x4*x7 - x5*x6
    x31 = -x10*x29 + x13*x30
    x32 = x1*x28 - x31*x9
    x33 = x10*x30 + x13*x29
    x34 = x1*x31 + x28*x9
    x35 = -x20*x33 + x22*x34
    x36 = post_transform_sqrt2_over2*(x0*x32 - x19*x35)
    x37 = x0*x35 + x19*x32
    x38 = x20*x34 + x22*x33
    x39 = x11*x5
    x40 = x11*x3
    x41 = -x10*x7 - x13*x40
    x42 = -x1*x39 - x41*x9
    x43 = x10*x40 - x13*x7
    x44 = -x43
    x45 = x1*x41 - x39*x9
    x46 = -x20*x44 + x22*x45
    x47 = post_transform_sqrt2_over2*(x0*x42 - x19*x46)
    x48 = x0*x46 + x19*x42
    x49 = x20*x45 + x22*x44
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = post_transform_sqrt2_over2*x26 - x25
    ee_pose[0, 1] = post_transform_sqrt2_over2*x26 + x25
    ee_pose[0, 2] = x27
    ee_pose[0, 3] = a_3*x16 + a_3*x17 + a_5*x24 - d_2*x12 + d_4*x21 + post_transform_d5*x27
    ee_pose[1, 0] = post_transform_sqrt2_over2*x37 - x36
    ee_pose[1, 1] = post_transform_sqrt2_over2*x37 + x36
    ee_pose[1, 2] = x38
    ee_pose[1, 3] = a_3*x30 + a_3*x31 + a_5*x35 - d_2*x29 + d_4*x33 + post_transform_d5*x38
    ee_pose[2, 0] = post_transform_sqrt2_over2*x48 - x47
    ee_pose[2, 1] = post_transform_sqrt2_over2*x48 + x47
    ee_pose[2, 2] = x49
    ee_pose[2, 3] = -a_3*x40 + a_3*x41 + a_5*x46 - d_2*x7 - d_4*x43 + post_transform_d5*x49 + pre_transform_d4
    return ee_pose


def franka_panda_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_0)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x2*x6*x7 - x5
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = x0*x6
    x12 = x2*x4
    x13 = x11 + x12*x7
    x14 = x10*x13 + x3*x9
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x10*x3 + x13*x9
    x18 = x15*x8 - x16*x17
    x19 = math.cos(th_5)
    x20 = math.sin(th_5)
    x21 = x15*x17 + x16*x8
    x22 = x14*x19 + x20*x21
    x23 = x0*x1
    x24 = x11*x7 + x12
    x25 = -x2*x6 + x5*x7
    x26 = x10*x25 + x23*x9
    x27 = -x10*x23 + x25*x9
    x28 = x15*x24 - x16*x27
    x29 = x15*x27 + x16*x24
    x30 = x19*x26 + x20*x29
    x31 = x1*x6
    x32 = x1*x4
    x33 = x10*x32 - x7*x9
    x34 = -x33
    x35 = -x10*x7 - x32*x9
    x36 = -x15*x31 - x16*x35
    x37 = x15*x35 - x16*x31
    x38 = x19*x34 + x20*x37
    x39 = d_2*x7
    x40 = -pre_transform_d4 + x39
    x41 = -x40
    x42 = a_3*x32
    x43 = x40 + x42
    x44 = -x43
    x45 = a_3*x25 - d_2*x23
    x46 = d_4*x33
    x47 = a_3*x35 - x43 - x46
    x48 = a_3*x27 + d_4*x26 + x45
    x49 = a_3*x35 + a_5*(x19*x37 - x20*x34) + pre_transform_d4 - x39 - x42 - x46
    x50 = a_5*(x19*x29 - x20*x26) + x48
    x51 = a_3*x13 - d_2*x3
    x52 = a_3*x17 + d_4*x14 + x51
    x53 = a_5*(-x14*x20 + x19*x21) + x52
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = x18
    jacobian_output[0, 6] = x22
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = -x23
    jacobian_output[1, 3] = x24
    jacobian_output[1, 4] = x26
    jacobian_output[1, 5] = x28
    jacobian_output[1, 6] = x30
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = -x7
    jacobian_output[2, 3] = -x31
    jacobian_output[2, 4] = x34
    jacobian_output[2, 5] = x36
    jacobian_output[2, 6] = x38
    jacobian_output[3, 1] = -pre_transform_d4*x2
    jacobian_output[3, 2] = x23*x39 + x23*x41
    jacobian_output[3, 3] = -x24*x44 - x31*x45
    jacobian_output[3, 4] = -x26*x47 + x34*x48
    jacobian_output[3, 5] = -x28*x47 + x36*x48
    jacobian_output[3, 6] = -x30*x49 + x38*x50
    jacobian_output[4, 1] = -pre_transform_d4*x0
    jacobian_output[4, 2] = -x3*x39 - x3*x41
    jacobian_output[4, 3] = x31*x51 + x44*x8
    jacobian_output[4, 4] = x14*x47 - x34*x52
    jacobian_output[4, 5] = x18*x47 - x36*x52
    jacobian_output[4, 6] = x22*x49 - x38*x53
    jacobian_output[5, 3] = x24*x51 - x45*x8
    jacobian_output[5, 4] = -x14*x48 + x26*x52
    jacobian_output[5, 5] = -x18*x48 + x28*x52
    jacobian_output[5, 6] = -x22*x50 + x30*x53
    return jacobian_output


def franka_panda_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_0)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x2*x6*x7 - x5
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = x0*x6
    x12 = x2*x4
    x13 = x11 + x12*x7
    x14 = x10*x13 + x3*x9
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x10*x3 + x13*x9
    x18 = math.cos(th_5)
    x19 = math.sin(th_5)
    x20 = x0*x1
    x21 = x11*x7 + x12
    x22 = -x2*x6 + x5*x7
    x23 = x10*x22 + x20*x9
    x24 = -x10*x20 + x22*x9
    x25 = x1*x6
    x26 = x1*x4
    x27 = -x10*x26 + x7*x9
    x28 = -x10*x7 - x26*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = x15*x8 - x16*x17
    jacobian_output[0, 6] = x14*x18 + x19*(x15*x17 + x16*x8)
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = -x20
    jacobian_output[1, 3] = x21
    jacobian_output[1, 4] = x23
    jacobian_output[1, 5] = x15*x21 - x16*x24
    jacobian_output[1, 6] = x18*x23 + x19*(x15*x24 + x16*x21)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = -x7
    jacobian_output[2, 3] = -x25
    jacobian_output[2, 4] = x27
    jacobian_output[2, 5] = -x15*x25 - x16*x28
    jacobian_output[2, 6] = x18*x27 + x19*(x15*x28 - x16*x25)
    return jacobian_output


def franka_panda_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = add_revolute_offset(theta_input[1], th_1_offset_original2raw)
    th_2 = add_revolute_offset(theta_input[2], th_2_offset_original2raw)
    th_3 = add_revolute_offset(theta_input[3], th_3_offset_original2raw)
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = add_revolute_offset(theta_input[5], th_5_offset_original2raw)
    th_6 = theta_input[6] + th_6_offset_original2raw
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.cos(th_1)
    x2 = math.sin(th_1)
    x3 = math.sin(th_0)
    x4 = p_on_ee_z*x3
    x5 = d_2*x1
    x6 = x2*x3
    x7 = -pre_transform_d4 + x5
    x8 = -x7
    x9 = math.sin(th_2)
    x10 = x2*x9
    x11 = math.cos(th_2)
    x12 = x0*x11
    x13 = x3*x9
    x14 = x1*x13 + x12
    x15 = x11*x2
    x16 = a_3*x15
    x17 = x16 + x7
    x18 = -x17
    x19 = x11*x3
    x20 = -x0*x9 + x1*x19
    x21 = a_3*x20 - d_2*x6
    x22 = math.cos(th_3)
    x23 = math.sin(th_3)
    x24 = -x1*x22 + x15*x23
    x25 = -x24
    x26 = x20*x23 + x22*x6
    x27 = d_4*x24
    x28 = -x1*x23 - x15*x22
    x29 = a_3*x28 - x17 - x27
    x30 = x20*x22 - x23*x6
    x31 = a_3*x30 + d_4*x26 + x21
    x32 = math.cos(th_4)
    x33 = math.sin(th_4)
    x34 = -x10*x32 - x28*x33
    x35 = x14*x32 - x30*x33
    x36 = math.cos(th_5)
    x37 = math.sin(th_5)
    x38 = -x10*x33 + x28*x32
    x39 = x25*x36 + x37*x38
    x40 = x14*x33 + x30*x32
    x41 = x26*x36 + x37*x40
    x42 = a_3*x28 + a_5*(-x25*x37 + x36*x38) + pre_transform_d4 - x16 - x27 - x5
    x43 = a_5*(-x26*x37 + x36*x40) + x31
    x44 = x0*x2
    x45 = x0*x1*x9 - x19
    x46 = x1*x12 + x13
    x47 = a_3*x46 - d_2*x44
    x48 = x22*x44 + x23*x46
    x49 = x22*x46 - x23*x44
    x50 = a_3*x49 + d_4*x48 + x47
    x51 = x32*x45 - x33*x49
    x52 = x32*x49 + x33*x45
    x53 = x36*x48 + x37*x52
    x54 = a_5*(x36*x52 - x37*x48) + x50
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -p_on_ee_y
    jacobian_output[0, 1] = p_on_ee_z*x0 - pre_transform_d4*x0
    jacobian_output[0, 2] = p_on_ee_y*x1 - x2*x4 + x5*x6 + x6*x8
    jacobian_output[0, 3] = p_on_ee_y*x10 + p_on_ee_z*x14 - x10*x21 - x14*x18
    jacobian_output[0, 4] = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x31 - x26*x29
    jacobian_output[0, 5] = -p_on_ee_y*x34 + p_on_ee_z*x35 - x29*x35 + x31*x34
    jacobian_output[0, 6] = -p_on_ee_y*x39 + p_on_ee_z*x41 + x39*x43 - x41*x42
    jacobian_output[1, 0] = p_on_ee_x
    jacobian_output[1, 1] = -pre_transform_d4*x3 + x4
    jacobian_output[1, 2] = -p_on_ee_x*x1 + p_on_ee_z*x0*x2 - x44*x5 - x44*x8
    jacobian_output[1, 3] = -p_on_ee_x*x10 - p_on_ee_z*x45 + x18*x45 + x2*x47*x9
    jacobian_output[1, 4] = p_on_ee_x*x25 - p_on_ee_z*x48 - x25*x50 + x29*x48
    jacobian_output[1, 5] = p_on_ee_x*x34 - p_on_ee_z*x51 + x29*x51 - x34*x50
    jacobian_output[1, 6] = p_on_ee_x*x39 - p_on_ee_z*x53 - x39*x54 + x42*x53
    jacobian_output[2, 1] = -p_on_ee_x*x0 - p_on_ee_y*x3
    jacobian_output[2, 2] = p_on_ee_x*x6 - p_on_ee_y*x44
    jacobian_output[2, 3] = -p_on_ee_x*x14 + p_on_ee_y*x45 + x14*x47 - x21*x45
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x48 + x26*x50 - x31*x48
    jacobian_output[2, 5] = -p_on_ee_x*x35 + p_on_ee_y*x51 - x31*x51 + x35*x50
    jacobian_output[2, 6] = -p_on_ee_x*x41 + p_on_ee_y*x53 + x41*x54 - x43*x53
    return jacobian_output


def franka_panda_ik_solve_raw(T_ee: np.ndarray, th_3):
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
    for i in range(50):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for equation all-zero dispatcher node 0
    def EquationAllZeroDispatcherNode_node_0_processor():
        if not node_input_validity[0]:
            return
        checked_result: bool = (((2*abs(a_5*inv_Px) <= 1.0e-6)) and ((2*abs(a_5*inv_Py) <= 1.0e-6)) and ((abs(2*a_3**2*math.cos(th_3) + 2*a_3**2 + 2*a_3*d_2*math.sin(th_3) + 2*a_3*d_4*math.sin(th_3) - a_5**2 + d_2**2 - 2*d_2*d_4*math.cos(th_3) + d_4**2 - inv_Px**2 - inv_Py**2 - inv_Pz**2) <= 1.0e-6)))
        if not checked_result:  # To non-degenerate node
            node_input_validity[1] = True
        else:
            node_input_validity[30] = True

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_0_processor()
    # Finish code for equation all-zero dispatcher node 0

    # Code for explicit solution node 30, solved variable is th_6
    def ExplicitSolutionNode_node_30_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return

        # The explicit solution of root node
        condition_0: bool = True
        if condition_0:
            solution_0: IkSolution = make_ik_solution()
            solution_0[5] = -3.14159265358979
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(31, appended_idx)

            solution_1: IkSolution = make_ik_solution()
            solution_1[5] = -2.94524311274043
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(31, appended_idx)

            solution_2: IkSolution = make_ik_solution()
            solution_2[5] = -2.74889357189107
            appended_idx = append_solution_to_queue(solution_2)
            add_input_index_to(31, appended_idx)

            solution_3: IkSolution = make_ik_solution()
            solution_3[5] = -2.55254403104171
            appended_idx = append_solution_to_queue(solution_3)
            add_input_index_to(31, appended_idx)

            solution_4: IkSolution = make_ik_solution()
            solution_4[5] = -2.35619449019234
            appended_idx = append_solution_to_queue(solution_4)
            add_input_index_to(31, appended_idx)

            solution_5: IkSolution = make_ik_solution()
            solution_5[5] = -2.15984494934298
            appended_idx = append_solution_to_queue(solution_5)
            add_input_index_to(31, appended_idx)

            solution_6: IkSolution = make_ik_solution()
            solution_6[5] = -1.96349540849362
            appended_idx = append_solution_to_queue(solution_6)
            add_input_index_to(31, appended_idx)

            solution_7: IkSolution = make_ik_solution()
            solution_7[5] = -1.76714586764426
            appended_idx = append_solution_to_queue(solution_7)
            add_input_index_to(31, appended_idx)

            solution_8: IkSolution = make_ik_solution()
            solution_8[5] = -1.57079632679490
            appended_idx = append_solution_to_queue(solution_8)
            add_input_index_to(31, appended_idx)

            solution_9: IkSolution = make_ik_solution()
            solution_9[5] = -1.37444678594553
            appended_idx = append_solution_to_queue(solution_9)
            add_input_index_to(31, appended_idx)

            solution_10: IkSolution = make_ik_solution()
            solution_10[5] = -1.17809724509617
            appended_idx = append_solution_to_queue(solution_10)
            add_input_index_to(31, appended_idx)

            solution_11: IkSolution = make_ik_solution()
            solution_11[5] = -0.981747704246811
            appended_idx = append_solution_to_queue(solution_11)
            add_input_index_to(31, appended_idx)

            solution_12: IkSolution = make_ik_solution()
            solution_12[5] = -0.785398163397448
            appended_idx = append_solution_to_queue(solution_12)
            add_input_index_to(31, appended_idx)

            solution_13: IkSolution = make_ik_solution()
            solution_13[5] = -0.589048622548086
            appended_idx = append_solution_to_queue(solution_13)
            add_input_index_to(31, appended_idx)

            solution_14: IkSolution = make_ik_solution()
            solution_14[5] = -0.392699081698724
            appended_idx = append_solution_to_queue(solution_14)
            add_input_index_to(31, appended_idx)

            solution_15: IkSolution = make_ik_solution()
            solution_15[5] = -0.196349540849362
            appended_idx = append_solution_to_queue(solution_15)
            add_input_index_to(31, appended_idx)

            solution_16: IkSolution = make_ik_solution()
            solution_16[5] = 0.0
            appended_idx = append_solution_to_queue(solution_16)
            add_input_index_to(31, appended_idx)

            solution_17: IkSolution = make_ik_solution()
            solution_17[5] = 0.196349540849362
            appended_idx = append_solution_to_queue(solution_17)
            add_input_index_to(31, appended_idx)

            solution_18: IkSolution = make_ik_solution()
            solution_18[5] = 0.392699081698724
            appended_idx = append_solution_to_queue(solution_18)
            add_input_index_to(31, appended_idx)

            solution_19: IkSolution = make_ik_solution()
            solution_19[5] = 0.589048622548086
            appended_idx = append_solution_to_queue(solution_19)
            add_input_index_to(31, appended_idx)

            solution_20: IkSolution = make_ik_solution()
            solution_20[5] = 0.785398163397448
            appended_idx = append_solution_to_queue(solution_20)
            add_input_index_to(31, appended_idx)

            solution_21: IkSolution = make_ik_solution()
            solution_21[5] = 0.981747704246811
            appended_idx = append_solution_to_queue(solution_21)
            add_input_index_to(31, appended_idx)

            solution_22: IkSolution = make_ik_solution()
            solution_22[5] = 1.17809724509617
            appended_idx = append_solution_to_queue(solution_22)
            add_input_index_to(31, appended_idx)

            solution_23: IkSolution = make_ik_solution()
            solution_23[5] = 1.37444678594553
            appended_idx = append_solution_to_queue(solution_23)
            add_input_index_to(31, appended_idx)

            solution_24: IkSolution = make_ik_solution()
            solution_24[5] = 1.57079632679490
            appended_idx = append_solution_to_queue(solution_24)
            add_input_index_to(31, appended_idx)

            solution_25: IkSolution = make_ik_solution()
            solution_25[5] = 1.76714586764426
            appended_idx = append_solution_to_queue(solution_25)
            add_input_index_to(31, appended_idx)

            solution_26: IkSolution = make_ik_solution()
            solution_26[5] = 1.96349540849362
            appended_idx = append_solution_to_queue(solution_26)
            add_input_index_to(31, appended_idx)

            solution_27: IkSolution = make_ik_solution()
            solution_27[5] = 2.15984494934298
            appended_idx = append_solution_to_queue(solution_27)
            add_input_index_to(31, appended_idx)

            solution_28: IkSolution = make_ik_solution()
            solution_28[5] = 2.35619449019234
            appended_idx = append_solution_to_queue(solution_28)
            add_input_index_to(31, appended_idx)

            solution_29: IkSolution = make_ik_solution()
            solution_29[5] = 2.55254403104171
            appended_idx = append_solution_to_queue(solution_29)
            add_input_index_to(31, appended_idx)

            solution_30: IkSolution = make_ik_solution()
            solution_30[5] = 2.74889357189107
            appended_idx = append_solution_to_queue(solution_30)
            add_input_index_to(31, appended_idx)

            solution_31: IkSolution = make_ik_solution()
            solution_31[5] = 2.94524311274043
            appended_idx = append_solution_to_queue(solution_31)
            add_input_index_to(31, appended_idx)

            solution_32: IkSolution = make_ik_solution()
            solution_32[5] = 3.14159265358979
            appended_idx = append_solution_to_queue(solution_32)
            add_input_index_to(31, appended_idx)

    # Invoke the processor
    ExplicitSolutionNode_node_30_solve_th_6_processor()
    # Finish code for explicit solution node 30

    # Code for solved_variable dispatcher node 31
    def SolvedVariableDispatcherNode_node_31_processor():
        this_node_input_index: List[int] = node_input_index[31]
        this_input_valid: bool = node_input_validity[31]
        if not this_input_valid:
            return

        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            degenerate_valid_0 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True

            degenerate_valid_1 = (abs(th_3 - 3.08938932227154 + math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True

            if not taken_by_degenerate:
                add_input_index_to(32, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    SolvedVariableDispatcherNode_node_31_processor()
    # Finish code for solved_variable dispatcher node 31

    # Code for explicit solution node 32, solved variable is th_4
    def ExplicitSolutionNode_node_32_solve_th_4_processor():
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
            th_6 = this_solution[5]
            condition_0: bool = (((a_3*math.cos(th_3) + a_3 + d_2*math.sin(th_3) != 0)) and ((abs((inv_Px*math.sin(th_6) + inv_Py*math.cos(th_6))/(a_3*math.cos(th_3) + a_3 + d_2*math.sin(th_3))) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_asin((inv_Px*math.sin(th_6) + inv_Py*math.cos(th_6))/(a_3*math.cos(th_3) + a_3 + d_2*math.sin(th_3)))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(33, appended_idx)
                this_solution[3] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(33, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_4_processor()
    # Finish code for explicit solution node 32

    # Code for equation all-zero dispatcher node 33
    def EquationAllZeroDispatcherNode_node_33_processor():
        this_node_input_index: List[int] = node_input_index[33]
        this_input_valid: bool = node_input_validity[33]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_6 = this_solution[5]
            checked_result: bool = (((abs(inv_Pz) <= 1.0e-6)) and ((abs(a_5 + inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(34, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_33_processor()
    # Finish code for equation all-zero dispatcher node 33

    # Code for explicit solution node 34, solved variable is th_5
    def ExplicitSolutionNode_node_34_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[34]
        this_input_valid: bool = node_input_validity[34]
        if not this_input_valid:
            return

        # The solution of non-root node 34
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_4 = this_solution[3]
            th_6 = this_solution[5]
            condition_0: bool = (((abs(inv_Pz) >= 1.0e-6)) or ((abs(a_5 + inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) >= 1.0e-6)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_3)
                x2 = a_3*x0 + a_3 + d_2*x1
                x3 = math.cos(th_4)
                x4 = a_3*x1 - d_2*x0 + d_4
                x5 = -a_5 - inv_Px*math.cos(th_6) + inv_Py*math.sin(th_6)
                this_solution[4] = math.atan2(-inv_Pz*x2*x3 - x4*x5, -inv_Pz*x4 + x2*x3*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(35, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_34_solve_th_5_processor()
    # Finish code for explicit solution node 34

    # Code for non-branch dispatcher node 35
    # Actually, there is no code

    # Code for linear solution type2 node 36, solved variable is th_2
    def LinearSinCosType_2_SolverNode_node_36_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[35]
        this_input_valid: bool = node_input_validity[35]
        if not this_input_valid:
            return
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            x0 = math.cos(th_6)
            x1 = math.sin(th_3)
            x2 = math.sin(th_5)
            x3 = x1*x2
            x4 = math.sin(th_6)
            x5 = math.sin(th_4)
            x6 = math.cos(th_3)
            x7 = x5*x6
            x8 = math.cos(th_4)
            x9 = math.cos(th_5)
            x10 = x6*x9
            x11 = -x0*x10*x8 + x0*x3 + x4*x7
            x12 = x4*x8
            x13 = x5*x9
            x14 = x0*x13 + x12
            x15 = x0*x7 + x10*x12 - x3*x4
            x16 = -x0*x8 + x13*x4
            x17 = x6*x8
            x18 = x1*x9 + x17*x2
            x19 = x2*x5
            x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1
            x21 = a_5*x13
            A_matrix = np.zeros((8, 4))
            A_matrix[0, 0] = r_11
            A_matrix[0, 1] = -r_21
            A_matrix[0, 2] = x11
            A_matrix[0, 3] = x14
            A_matrix[1, 0] = r_12
            A_matrix[1, 1] = -r_22
            A_matrix[1, 2] = x15
            A_matrix[1, 3] = -x16
            A_matrix[2, 0] = r_13
            A_matrix[2, 1] = -r_23
            A_matrix[2, 2] = -x18
            A_matrix[2, 3] = x19
            A_matrix[3, 0] = Px
            A_matrix[3, 1] = -Py
            A_matrix[3, 2] = -x20
            A_matrix[3, 3] = x21
            A_matrix[4, 0] = -r_11
            A_matrix[4, 1] = r_21
            A_matrix[4, 2] = -x11
            A_matrix[4, 3] = -x14
            A_matrix[5, 0] = -r_12
            A_matrix[5, 1] = r_22
            A_matrix[5, 2] = -x15
            A_matrix[5, 3] = x16
            A_matrix[6, 0] = -r_13
            A_matrix[6, 1] = r_23
            A_matrix[6, 2] = x18
            A_matrix[6, 3] = -x19
            A_matrix[7, 0] = -Px
            A_matrix[7, 1] = Py
            A_matrix[7, 2] = x20
            A_matrix[7, 3] = -x21
            linear_values = solve_linear_type2_candidates(A_matrix, [(0, 1, 2), (0, 1, 3), (0, 1, 6), (0, 1, 7), (0, 2, 3)])
            for linear_index, linear_value in enumerate(linear_values):
                candidate = copy.copy(this_solution)
                candidate[2] = linear_value
                if linear_index+1 == len(linear_values):
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(37, candidate_index)
            if not linear_values:
                queue_element_validity[queue_index] = False
    # Invoke the processor
    LinearSinCosType_2_SolverNode_node_36_solve_th_2_processor()
    # Finish code for explicit solution node 35

    # Code for equation all-zero dispatcher node 37
    def EquationAllZeroDispatcherNode_node_37_processor():
        this_node_input_index: List[int] = node_input_index[37]
        this_input_valid: bool = node_input_validity[37]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            checked_result: bool = (((abs(r_13) <= 1.0e-6)) and ((abs(r_23) <= 1.0e-6)) and ((abs(math.sin(th_2)*math.sin(th_3)*math.cos(th_5) + math.sin(th_2)*math.sin(th_5)*math.cos(th_3)*math.cos(th_4) - math.sin(th_4)*math.sin(th_5)*math.cos(th_2)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(38, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_37_processor()
    # Finish code for equation all-zero dispatcher node 37

    # Code for explicit solution node 38, solved variable is th_0
    def ExplicitSolutionNode_node_38_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[38]
        this_input_valid: bool = node_input_validity[38]
        if not this_input_valid:
            return

        # The solution of non-root node 38
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (((abs(r_13) >= zero_tolerance)) or ((abs(r_23) >= zero_tolerance)) or ((abs(math.sin(th_2)*math.sin(th_3)*math.cos(th_5) + math.sin(th_2)*math.sin(th_5)*math.cos(th_3)*math.cos(th_4) - math.sin(th_4)*math.sin(th_5)*math.cos(th_2)) >= zero_tolerance)))
            if condition_0:
                x0 = math.atan2(r_13, -r_23)
                x1 = math.sin(th_2)
                x2 = math.sin(th_5)
                x3 = x1*x2*math.cos(th_3)*math.cos(th_4) + x1*math.sin(th_3)*math.cos(th_5) - x2*math.sin(th_4)*math.cos(th_2)
                x4 = safe_sqrt(r_13**2 + r_23**2 - x3**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x0 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(39, appended_idx)
                this_solution[0] = x0 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(39, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_38_solve_th_0_processor()
    # Finish code for explicit solution node 38

    # Code for equation all-zero dispatcher node 39
    def EquationAllZeroDispatcherNode_node_39_processor():
        this_node_input_index: List[int] = node_input_index[39]
        this_input_valid: bool = node_input_validity[39]
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
                add_input_index_to(40, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_39_processor()
    # Finish code for equation all-zero dispatcher node 39

    # Code for explicit solution node 40, solved variable is th_1
    def ExplicitSolutionNode_node_40_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[40]
        this_input_valid: bool = node_input_validity[40]
        if not this_input_valid:
            return

        # The solution of non-root node 40
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (((abs(Pz) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)))
            if condition_0:
                x0 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x1 = math.sin(th_3)
                x2 = math.cos(th_3)
                x3 = math.sin(th_5)
                x4 = a_5*math.cos(th_5)
                x5 = x4*math.cos(th_4)
                x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5
                x7 = math.cos(th_2)
                x8 = a_3*x7
                x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*math.sin(th_2)*math.sin(th_4) - x8
                this_solution[1] = math.atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_40_solve_th_1_processor()
    # Finish code for explicit solution node 40

    # Code for explicit solution node 1, solved variable is th_6
    def ExplicitSolutionNode_node_1_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[1]
        this_input_valid: bool = node_input_validity[1]
        if not this_input_valid:
            return

        # The explicit solution of root node
        condition_0: bool = (((2*abs(a_5*inv_Px) >= zero_tolerance)) or ((2*abs(a_5*inv_Py) >= zero_tolerance)) or ((abs(2*a_3**2*math.cos(th_3) + 2*a_3**2 + 2*a_3*d_2*math.sin(th_3) + 2*a_3*d_4*math.sin(th_3) - a_5**2 + d_2**2 - 2*d_2*d_4*math.cos(th_3) + d_4**2 - inv_Px**2 - inv_Py**2 - inv_Pz**2) >= zero_tolerance)))
        if condition_0:
            x0 = 2*a_5
            x1 = math.atan2(inv_Py*x0, -inv_Px*x0)
            x2 = inv_Px**2
            x3 = a_5**2
            x4 = 4*x3
            x5 = inv_Py**2
            x6 = 2*a_3**2
            x7 = a_3*math.sin(th_3)
            x8 = math.cos(th_3)
            x9 = -d_2**2 + 2*d_2*d_4*x8 - 2*d_2*x7 - d_4**2 - 2*d_4*x7 + inv_Pz**2 + x2 + x3 + x5 - x6*x8 - x6
            x10 = safe_sqrt(x2*x4 + x4*x5 - x9**2)
            solution_0: IkSolution = make_ik_solution()
            solution_0[5] = x1 + math.atan2(x10, x9)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)

            solution_1: IkSolution = make_ik_solution()
            solution_1[5] = x1 + math.atan2(-x10, x9)
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)

    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_6_processor()
    # Finish code for explicit solution node 1

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
            forwarded_to_child = False
            degenerate_valid_0 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(12, branch_input_index)

            degenerate_valid_1 = (abs(th_3 - 3.08938932227154 + math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(13, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(3, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor()
    # Finish code for solved_variable dispatcher node 2

    # Code for explicit solution node 13, solved variable is th_4
    def ExplicitSolutionNode_node_13_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return

        # The solution of non-root node 13
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_6 = this_solution[5]
            condition_0: bool = (((1.99863771551522*a_3 - 0.0521796239019005*d_2 != 0)) and ((abs((inv_Px*math.sin(th_6) + inv_Py*math.cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2)) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_asin((inv_Px*math.sin(th_6) + inv_Py*math.cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(14, appended_idx)
                this_solution[3] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(14, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_4_processor()
    # Finish code for explicit solution node 13

    # Code for non-branch dispatcher node 14
    # Actually, there is no code

    # Code for explicit solution node 15, solved variable is th_5
    def ExplicitSolutionNode_node_15_solve_th_5_processor():
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
            th_4 = this_solution[3]
            th_6 = this_solution[5]
            condition_0: bool = (((abs(0.0521796239019005*a_3 + 0.998637715515219*d_2 - 1.0*d_4) >= 1.0e-6)) or ((abs(1.99863771551522*a_3*math.cos(th_4) - 0.0521796239019005*d_2*math.cos(th_4)) >= 1.0e-6)))
            if condition_0:
                x0 = math.cos(th_4)
                x1 = -1.99863771551522*a_3*x0 + 0.0521796239019005*d_2*x0
                x2 = 0.0521796239019005*a_3 + 0.998637715515219*d_2 - 1.0*d_4
                x3 = 1.0*a_5 + 1.0*inv_Px*math.cos(th_6) - 1.0*inv_Py*math.sin(th_6)
                this_solution[4] = math.atan2(inv_Pz*x1 - x2*x3, inv_Pz*x2 + x1*x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_5_processor()
    # Finish code for explicit solution node 14

    # Code for non-branch dispatcher node 24
    # Actually, there is no code

    # Code for explicit solution node 25, solved variable is th_2
    def ExplicitSolutionNode_node_25_solve_th_2_processor():
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
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = ((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 > 1.0e-24)
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = r_31*(-x2*x4 - x3*x6) + r_32*(-x2*x3 + x4*x6) - r_33*x0*x1
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x0*x9
                x11 = x1*x8 - x2*x5*x9
                x12 = r_31*(-x10*x4 - x11*x3) + r_32*(-x10*x3 + x11*x4) + r_33*(x1*x2*x9 + x5*x8)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = -math.atan2(-x7, -x12)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(26, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[2] = -math.atan2(x7, x12)
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(26, appended_idx)
            condition_2: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))))
            if condition_2:
                x0 = 6*math.pi
                x1 = math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = -x3*x4 + x5*x6
                x8 = -x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[2] = (1/2)*max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1/2)*min(3.14159265358979, -x0 + x9 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(26, appended_idx)
            condition_3: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))))
            if condition_3:
                x0 = 4*math.pi
                x1 = math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = -x3*x4 + x5*x6
                x8 = -x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[2] = (1/2)*max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1/2)*min(3.14159265358979, -x0 + x9 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(26, appended_idx)
            condition_4: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))))))
            if condition_4:
                x0 = 2*math.pi
                x1 = math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = -x3*x4 + x5*x6
                x8 = -x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[2] = (1/2)*max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1/2)*min(3.14159265358979, -x0 + x9 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(26, appended_idx)
            condition_5: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_5:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(26, appended_idx)
            condition_6: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 2*math.pi) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 2*math.pi))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_6:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 2*math.pi
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(26, appended_idx)
            condition_7: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 4*math.pi) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 4*math.pi))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_7:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 4*math.pi
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(26, appended_idx)
            condition_8: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 6*math.pi) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 6*math.pi))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_8:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 6*math.pi
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(26, appended_idx)
            condition_9: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))
            if condition_9:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979 + 6*math.pi) + (1/2)*min(3.14159265358979, -x12 + 3.14159265358979 + 6*math.pi)
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(26, appended_idx)
            condition_10: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))
            if condition_10:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979 + 4*math.pi) + (1/2)*min(3.14159265358979, -x12 + 3.14159265358979 + 4*math.pi)
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(26, appended_idx)
            condition_11: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))))
            if condition_11:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979 + 2*math.pi) + (1/2)*min(3.14159265358979, -x12 + 3.14159265358979 + 2*math.pi)
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(26, appended_idx)
            condition_12: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) <= min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))))
            if condition_12:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(26, appended_idx)
            condition_13: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.14159265358979) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) > min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))))
            if condition_13:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*math.pi
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(26, appended_idx)
            condition_14: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.14159265358979) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) > min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.14159265358979))))))
            if condition_14:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*math.pi
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(26, appended_idx)
            condition_15: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi - 3.14159265358979) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) > min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.14159265358979))))))
            if condition_15:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*math.pi
                this_solution[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(26, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_2_processor()
    # Finish code for explicit solution node 24

    # Code for non-branch dispatcher node 26
    # Actually, there is no code

    # Code for explicit solution node 27, solved variable is th_1
    def ExplicitSolutionNode_node_27_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_4 = this_solution[3]
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
                this_solution[1] = math.atan2((-r_31*(-x11*x4 - x13*x3) - r_32*(-x11*x3 + x13*x4) - r_33*(x10*x2 + x8))*math.cos(th_2) - (-r_31*(-x2*x4 - x3*x6) - r_32*(-x2*x3 + x4*x6) + r_33*x0*x1)*math.sin(th_2), -r_31*(-x14*x4 + x15*x3) - r_32*(-x14*x3 - x15*x4) - r_33*(x12*x2 - x5*x9))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_1_processor()
    # Finish code for explicit solution node 26

    # Code for non-branch dispatcher node 28
    # Actually, there is no code

    # Code for explicit solution node 29, solved variable is th_0
    def ExplicitSolutionNode_node_29_solve_th_0_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_4 = this_solution[3]
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
                x9 = -x4*x5 + x6*x8
                x10 = -x4*x6 - x5*x8
                x11 = r_11*x10 + r_12*x9 - r_13*x3
                x12 = r_21*x10 + r_22*x9 - r_23*x3
                x13 = math.sin(th_2)
                x14 = math.cos(th_1)
                x15 = x13*x14
                x16 = math.sin(th_3)
                x17 = x16*x7
                x18 = math.cos(th_3)
                x19 = x1*x18
                x20 = x17 + x19*x4
                x21 = x18*x2
                x22 = x1*x16
                x23 = -x18*x4*x7 + x22
                x24 = -x21*x6 - x23*x5
                x25 = -x21*x5 + x23*x6
                x26 = r_11*x24 + r_12*x25 + r_13*x20
                x27 = math.sin(th_1)
                x28 = -x18*x7 + x22*x4
                x29 = x16*x2
                x30 = x17*x4 + x19
                x31 = -x29*x6 + x30*x5
                x32 = -x29*x5 - x30*x6
                x33 = r_21*x24 + r_22*x25 + r_23*x20
                this_solution[0] = math.atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_0_processor()
    # Finish code for explicit solution node 28

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
            th_6 = this_solution[5]
            condition_0: bool = (((abs(inv_Pz) >= zero_tolerance)) or ((abs(d_2 + d_4) >= zero_tolerance)) or ((abs(a_5 + inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) >= zero_tolerance)))
            if condition_0:
                x0 = d_2 + d_4
                this_solution[4] = math.atan2((a_5 + inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6))/x0, -inv_Pz/x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(16, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_5_processor()
    # Finish code for explicit solution node 12

    # Code for non-branch dispatcher node 16
    # Actually, there is no code

    # Code for explicit solution node 17, solved variable is th_4
    def ExplicitSolutionNode_node_17_solve_th_4_processor():
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
            condition_0: bool = True
            if condition_0:
                this_solution[3] = 0.0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_4_processor()
    # Finish code for explicit solution node 16

    # Code for non-branch dispatcher node 18
    # Actually, there is no code

    # Code for explicit solution node 19, solved variable is th_2
    def ExplicitSolutionNode_node_19_solve_th_2_processor():
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
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = ((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 > 1.0e-24)
            if condition_0:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = math.cos(th_5)
                x6 = x0*x5
                x7 = r_31*(-x2*x4 - x3*x6) + r_32*(-x2*x3 + x4*x6) - r_33*x0*x1
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x0*x9
                x11 = x1*x8 - x2*x5*x9
                x12 = r_31*(-x10*x4 - x11*x3) + r_32*(-x10*x3 + x11*x4) + r_33*(x1*x2*x9 + x5*x8)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = -math.atan2(-x7, -x12)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(20, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[2] = -math.atan2(x7, x12)
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(20, appended_idx)
            condition_2: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))))
            if condition_2:
                x0 = 6*math.pi
                x1 = math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = -x3*x4 + x5*x6
                x8 = -x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[2] = (1/2)*max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1/2)*min(3.14159265358979, -x0 + x9 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(20, appended_idx)
            condition_3: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))))
            if condition_3:
                x0 = 4*math.pi
                x1 = math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = -x3*x4 + x5*x6
                x8 = -x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[2] = (1/2)*max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1/2)*min(3.14159265358979, -x0 + x9 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(20, appended_idx)
            condition_4: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))))))
            if condition_4:
                x0 = 2*math.pi
                x1 = math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = math.cos(th_4)
                x4 = math.cos(th_6)
                x5 = math.sin(th_6)
                x6 = x1*math.cos(th_5)
                x7 = -x3*x4 + x5*x6
                x8 = -x3*x5 - x4*x6
                x9 = math.atan2(r_11*x8 + r_12*x7 - r_13*x1*x2, -r_21*x8 - r_22*x7 + r_23*x1*x2)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[2] = (1/2)*max(-3.14159265358979, -x0 + x9 - 3.14159265358979) + (1/2)*min(3.14159265358979, -x0 + x9 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(20, appended_idx)
            condition_5: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_5:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(20, appended_idx)
            condition_6: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 2*math.pi) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 2*math.pi))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_6:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 2*math.pi
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(20, appended_idx)
            condition_7: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 4*math.pi) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 4*math.pi))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_7:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 4*math.pi
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(20, appended_idx)
            condition_8: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) < 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 6*math.pi) <= min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 6*math.pi))) and ((((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 6*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, math.atan2(r_11*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) + r_12*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) - r_13*math.sin(th_4)*math.sin(th_5), -r_21*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_22*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_23*math.sin(th_4)*math.sin(th_5)) - 2*math.pi + 3.14159265358979))))))
            if condition_8:
                x0 = math.sin(th_4)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_6)
                x4 = math.sin(th_6)
                x5 = x0*math.cos(th_5)
                x6 = -x2*x3 + x4*x5
                x7 = -x2*x4 - x3*x5
                x8 = math.atan2(r_11*x7 + r_12*x6 - r_13*x0*x1, -r_21*x7 - r_22*x6 + r_23*x0*x1) + 6*math.pi
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[2] = (1/2)*max(-3.14159265358979, x8 - 3.14159265358979) + (1/2)*min(3.14159265358979, x8 + 3.14159265358979)
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(20, appended_idx)
            condition_9: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))
            if condition_9:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979 + 6*math.pi) + (1/2)*min(3.14159265358979, -x12 + 3.14159265358979 + 6*math.pi)
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(20, appended_idx)
            condition_10: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))
            if condition_10:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979 + 4*math.pi) + (1/2)*min(3.14159265358979, -x12 + 3.14159265358979 + 4*math.pi)
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(20, appended_idx)
            condition_11: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))))
            if condition_11:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979 + 2*math.pi) + (1/2)*min(3.14159265358979, -x12 + 3.14159265358979 + 2*math.pi)
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(20, appended_idx)
            condition_12: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) <= min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))))
            if condition_12:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(20, appended_idx)
            condition_13: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.14159265358979) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) > min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))))))
            if condition_13:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 2*math.pi
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(20, appended_idx)
            condition_14: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.14159265358979) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) > min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.14159265358979))))))
            if condition_14:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 4*math.pi
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(20, appended_idx)
            condition_15: bool = (((-r_31*((math.sin(th_3)*math.cos(th_4)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3))*math.cos(th_6) - math.sin(th_3)*math.sin(th_4)*math.sin(th_6)) - r_32*((-math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - math.sin(th_5)*math.cos(th_3))*math.sin(th_6) - math.sin(th_3)*math.sin(th_4)*math.cos(th_6)) - r_33*(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) - math.cos(th_3)*math.cos(th_5)) >= 0)) and (((r_31*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_32*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_33*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))**2 + (-r_31*(-math.sin(th_4)*math.cos(th_5)*math.cos(th_6) - math.sin(th_6)*math.cos(th_4)) - r_32*(math.sin(th_4)*math.sin(th_6)*math.cos(th_5) - math.cos(th_4)*math.cos(th_6)) + r_33*math.sin(th_4)*math.sin(th_5))**2 <= 1.0e-14)) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi - 3.14159265358979) <= min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 6*math.pi + 3.14159265358979))) and ((((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979) > min(3.14159265358979, 3.14159265358979 - math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)))))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 2*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 2*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 4*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 4*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 3.14159265358979 + 6*math.pi) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) + 3.14159265358979 + 6*math.pi))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi - 3.14159265358979) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 4*math.pi + 3.14159265358979))) and ((max(-3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi - 3.14159265358979) > min(3.14159265358979, -math.atan2(r_21*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_22*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_23*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4)), r_11*((-math.sin(th_3)*math.sin(th_5) + math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.cos(th_6) - math.sin(th_4)*math.sin(th_6)*math.cos(th_3)) + r_12*((math.sin(th_3)*math.sin(th_5) - math.cos(th_3)*math.cos(th_4)*math.cos(th_5))*math.sin(th_6) - math.sin(th_4)*math.cos(th_3)*math.cos(th_6)) + r_13*(math.sin(th_3)*math.cos(th_5) + math.sin(th_5)*math.cos(th_3)*math.cos(th_4))) - 2*math.pi + 3.14159265358979))))))
            if condition_15:
                x0 = math.sin(th_3)
                x1 = math.cos(th_5)
                x2 = math.sin(th_5)
                x3 = math.cos(th_3)
                x4 = math.cos(th_4)
                x5 = x0*x1 + x2*x3*x4
                x6 = math.sin(th_6)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_6)
                x9 = x0*x2 - x1*x3*x4
                x10 = -x6*x7 - x8*x9
                x11 = x6*x9 - x7*x8
                x12 = math.atan2(r_21*x10 + r_22*x11 + r_23*x5, r_11*x10 + r_12*x11 + r_13*x5) + 6*math.pi
                this_solution[2] = (1/2)*max(-3.14159265358979, -x12 - 3.14159265358979) + (1/2)*min(3.14159265358979, 3.14159265358979 - x12)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_2_processor()
    # Finish code for explicit solution node 18

    # Code for non-branch dispatcher node 20
    # Actually, there is no code

    # Code for explicit solution node 21, solved variable is th_1
    def ExplicitSolutionNode_node_21_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_4 = this_solution[3]
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
                this_solution[1] = math.atan2((-r_31*(-x11*x4 - x13*x3) - r_32*(-x11*x3 + x13*x4) - r_33*(x10*x2 + x8))*math.cos(th_2) - (-r_31*(-x2*x4 - x3*x6) - r_32*(-x2*x3 + x4*x6) + r_33*x0*x1)*math.sin(th_2), -r_31*(-x14*x4 + x15*x3) - r_32*(-x14*x3 - x15*x4) - r_33*(x12*x2 - x5*x9))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_1_processor()
    # Finish code for explicit solution node 20

    # Code for non-branch dispatcher node 22
    # Actually, there is no code

    # Code for explicit solution node 23, solved variable is th_0
    def ExplicitSolutionNode_node_23_solve_th_0_processor():
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
            th_2 = this_solution[2]
            th_4 = this_solution[3]
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
                x9 = -x4*x5 + x6*x8
                x10 = -x4*x6 - x5*x8
                x11 = r_11*x10 + r_12*x9 - r_13*x3
                x12 = r_21*x10 + r_22*x9 - r_23*x3
                x13 = math.sin(th_2)
                x14 = math.cos(th_1)
                x15 = x13*x14
                x16 = math.sin(th_3)
                x17 = x16*x7
                x18 = math.cos(th_3)
                x19 = x1*x18
                x20 = x17 + x19*x4
                x21 = x18*x2
                x22 = x1*x16
                x23 = -x18*x4*x7 + x22
                x24 = -x21*x6 - x23*x5
                x25 = -x21*x5 + x23*x6
                x26 = r_11*x24 + r_12*x25 + r_13*x20
                x27 = math.sin(th_1)
                x28 = -x18*x7 + x22*x4
                x29 = x16*x2
                x30 = x17*x4 + x19
                x31 = -x29*x6 + x30*x5
                x32 = -x29*x5 - x30*x6
                x33 = r_21*x24 + r_22*x25 + r_23*x20
                this_solution[0] = math.atan2(x0*x11 + x0*x14*x33 - x12*x15 + x13*x26 - x27*(r_21*x31 + r_22*x32 + r_23*x28), -x0*x12 + x0*x14*x26 - x11*x15 - x13*x33 - x27*(r_11*x31 + r_12*x32 + r_13*x28))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_0_processor()
    # Finish code for explicit solution node 22

    # Code for explicit solution node 3, solved variable is th_4
    def ExplicitSolutionNode_node_3_solve_th_4_processor():
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
            th_6 = this_solution[5]
            condition_0: bool = (((a_3*math.cos(th_3) + a_3 + d_2*math.sin(th_3) != 0)) and ((abs((inv_Px*math.sin(th_6) + inv_Py*math.cos(th_6))/(a_3*math.cos(th_3) + a_3 + d_2*math.sin(th_3))) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_asin((inv_Px*math.sin(th_6) + inv_Py*math.cos(th_6))/(a_3*math.cos(th_3) + a_3 + d_2*math.sin(th_3)))
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                this_solution[3] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_4_processor()
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
            checked_result: bool = (((abs(inv_Pz) <= 1.0e-6)) and ((abs(a_5 + inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
            else:
                add_input_index_to(41, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4

    # Code for explicit solution node 41, solved variable is th_5
    def ExplicitSolutionNode_node_41_solve_th_5_processor():
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
            condition_0: bool = True
            if condition_0:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = -3.14159265358979
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(42, appended_idx)
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[4] = -2.94524311274043
                appended_idx = append_solution_to_queue(solution_1)
                add_input_index_to(42, appended_idx)
                solution_2: IkSolution = copy.copy(this_solution)
                solution_2[4] = -2.74889357189107
                appended_idx = append_solution_to_queue(solution_2)
                add_input_index_to(42, appended_idx)
                solution_3: IkSolution = copy.copy(this_solution)
                solution_3[4] = -2.55254403104171
                appended_idx = append_solution_to_queue(solution_3)
                add_input_index_to(42, appended_idx)
                solution_4: IkSolution = copy.copy(this_solution)
                solution_4[4] = -2.35619449019234
                appended_idx = append_solution_to_queue(solution_4)
                add_input_index_to(42, appended_idx)
                solution_5: IkSolution = copy.copy(this_solution)
                solution_5[4] = -2.15984494934298
                appended_idx = append_solution_to_queue(solution_5)
                add_input_index_to(42, appended_idx)
                solution_6: IkSolution = copy.copy(this_solution)
                solution_6[4] = -1.96349540849362
                appended_idx = append_solution_to_queue(solution_6)
                add_input_index_to(42, appended_idx)
                solution_7: IkSolution = copy.copy(this_solution)
                solution_7[4] = -1.76714586764426
                appended_idx = append_solution_to_queue(solution_7)
                add_input_index_to(42, appended_idx)
                solution_8: IkSolution = copy.copy(this_solution)
                solution_8[4] = -1.57079632679490
                appended_idx = append_solution_to_queue(solution_8)
                add_input_index_to(42, appended_idx)
                solution_9: IkSolution = copy.copy(this_solution)
                solution_9[4] = -1.37444678594553
                appended_idx = append_solution_to_queue(solution_9)
                add_input_index_to(42, appended_idx)
                solution_10: IkSolution = copy.copy(this_solution)
                solution_10[4] = -1.17809724509617
                appended_idx = append_solution_to_queue(solution_10)
                add_input_index_to(42, appended_idx)
                solution_11: IkSolution = copy.copy(this_solution)
                solution_11[4] = -0.981747704246811
                appended_idx = append_solution_to_queue(solution_11)
                add_input_index_to(42, appended_idx)
                solution_12: IkSolution = copy.copy(this_solution)
                solution_12[4] = -0.785398163397448
                appended_idx = append_solution_to_queue(solution_12)
                add_input_index_to(42, appended_idx)
                solution_13: IkSolution = copy.copy(this_solution)
                solution_13[4] = -0.589048622548086
                appended_idx = append_solution_to_queue(solution_13)
                add_input_index_to(42, appended_idx)
                solution_14: IkSolution = copy.copy(this_solution)
                solution_14[4] = -0.392699081698724
                appended_idx = append_solution_to_queue(solution_14)
                add_input_index_to(42, appended_idx)
                solution_15: IkSolution = copy.copy(this_solution)
                solution_15[4] = -0.196349540849362
                appended_idx = append_solution_to_queue(solution_15)
                add_input_index_to(42, appended_idx)
                solution_16: IkSolution = copy.copy(this_solution)
                solution_16[4] = 0.0
                appended_idx = append_solution_to_queue(solution_16)
                add_input_index_to(42, appended_idx)
                solution_17: IkSolution = copy.copy(this_solution)
                solution_17[4] = 0.196349540849362
                appended_idx = append_solution_to_queue(solution_17)
                add_input_index_to(42, appended_idx)
                solution_18: IkSolution = copy.copy(this_solution)
                solution_18[4] = 0.392699081698724
                appended_idx = append_solution_to_queue(solution_18)
                add_input_index_to(42, appended_idx)
                solution_19: IkSolution = copy.copy(this_solution)
                solution_19[4] = 0.589048622548086
                appended_idx = append_solution_to_queue(solution_19)
                add_input_index_to(42, appended_idx)
                solution_20: IkSolution = copy.copy(this_solution)
                solution_20[4] = 0.785398163397448
                appended_idx = append_solution_to_queue(solution_20)
                add_input_index_to(42, appended_idx)
                solution_21: IkSolution = copy.copy(this_solution)
                solution_21[4] = 0.981747704246811
                appended_idx = append_solution_to_queue(solution_21)
                add_input_index_to(42, appended_idx)
                solution_22: IkSolution = copy.copy(this_solution)
                solution_22[4] = 1.17809724509617
                appended_idx = append_solution_to_queue(solution_22)
                add_input_index_to(42, appended_idx)
                solution_23: IkSolution = copy.copy(this_solution)
                solution_23[4] = 1.37444678594553
                appended_idx = append_solution_to_queue(solution_23)
                add_input_index_to(42, appended_idx)
                solution_24: IkSolution = copy.copy(this_solution)
                solution_24[4] = 1.57079632679490
                appended_idx = append_solution_to_queue(solution_24)
                add_input_index_to(42, appended_idx)
                solution_25: IkSolution = copy.copy(this_solution)
                solution_25[4] = 1.76714586764426
                appended_idx = append_solution_to_queue(solution_25)
                add_input_index_to(42, appended_idx)
                solution_26: IkSolution = copy.copy(this_solution)
                solution_26[4] = 1.96349540849362
                appended_idx = append_solution_to_queue(solution_26)
                add_input_index_to(42, appended_idx)
                solution_27: IkSolution = copy.copy(this_solution)
                solution_27[4] = 2.15984494934298
                appended_idx = append_solution_to_queue(solution_27)
                add_input_index_to(42, appended_idx)
                solution_28: IkSolution = copy.copy(this_solution)
                solution_28[4] = 2.35619449019234
                appended_idx = append_solution_to_queue(solution_28)
                add_input_index_to(42, appended_idx)
                solution_29: IkSolution = copy.copy(this_solution)
                solution_29[4] = 2.55254403104171
                appended_idx = append_solution_to_queue(solution_29)
                add_input_index_to(42, appended_idx)
                solution_30: IkSolution = copy.copy(this_solution)
                solution_30[4] = 2.74889357189107
                appended_idx = append_solution_to_queue(solution_30)
                add_input_index_to(42, appended_idx)
                solution_31: IkSolution = copy.copy(this_solution)
                solution_31[4] = 2.94524311274043
                appended_idx = append_solution_to_queue(solution_31)
                add_input_index_to(42, appended_idx)
                this_solution[4] = 3.14159265358979
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(42, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_5_processor()
    # Finish code for explicit solution node 41

    # Code for non-branch dispatcher node 42
    # Actually, there is no code

    # Code for linear solution type2 node 43, solved variable is th_2
    def LinearSinCosType_2_SolverNode_node_43_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[42]
        this_input_valid: bool = node_input_validity[42]
        if not this_input_valid:
            return
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            x0 = math.cos(th_6)
            x1 = math.sin(th_3)
            x2 = math.sin(th_5)
            x3 = x1*x2
            x4 = math.sin(th_6)
            x5 = math.sin(th_4)
            x6 = math.cos(th_3)
            x7 = x5*x6
            x8 = math.cos(th_4)
            x9 = math.cos(th_5)
            x10 = x6*x9
            x11 = -x0*x10*x8 + x0*x3 + x4*x7
            x12 = x4*x8
            x13 = x5*x9
            x14 = x0*x13 + x12
            x15 = x0*x7 + x10*x12 - x3*x4
            x16 = -x0*x8 + x13*x4
            x17 = x6*x8
            x18 = x1*x9 + x17*x2
            x19 = x2*x5
            x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1
            x21 = a_5*x13
            A_matrix = np.zeros((8, 4))
            A_matrix[0, 0] = r_11
            A_matrix[0, 1] = -r_21
            A_matrix[0, 2] = x11
            A_matrix[0, 3] = x14
            A_matrix[1, 0] = r_12
            A_matrix[1, 1] = -r_22
            A_matrix[1, 2] = x15
            A_matrix[1, 3] = -x16
            A_matrix[2, 0] = r_13
            A_matrix[2, 1] = -r_23
            A_matrix[2, 2] = -x18
            A_matrix[2, 3] = x19
            A_matrix[3, 0] = Px
            A_matrix[3, 1] = -Py
            A_matrix[3, 2] = -x20
            A_matrix[3, 3] = x21
            A_matrix[4, 0] = -r_11
            A_matrix[4, 1] = r_21
            A_matrix[4, 2] = -x11
            A_matrix[4, 3] = -x14
            A_matrix[5, 0] = -r_12
            A_matrix[5, 1] = r_22
            A_matrix[5, 2] = -x15
            A_matrix[5, 3] = x16
            A_matrix[6, 0] = -r_13
            A_matrix[6, 1] = r_23
            A_matrix[6, 2] = x18
            A_matrix[6, 3] = -x19
            A_matrix[7, 0] = -Px
            A_matrix[7, 1] = Py
            A_matrix[7, 2] = x20
            A_matrix[7, 3] = -x21
            linear_values = solve_linear_type2_candidates(A_matrix, [(0, 1, 2), (0, 1, 3), (0, 1, 6), (0, 1, 7), (0, 2, 3)])
            for linear_index, linear_value in enumerate(linear_values):
                candidate = copy.copy(this_solution)
                candidate[2] = linear_value
                if linear_index+1 == len(linear_values):
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(44, candidate_index)
            if not linear_values:
                queue_element_validity[queue_index] = False
    # Invoke the processor
    LinearSinCosType_2_SolverNode_node_43_solve_th_2_processor()
    # Finish code for explicit solution node 42

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
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            checked_result: bool = (((abs(r_13) <= 1.0e-6)) and ((abs(r_23) <= 1.0e-6)) and ((abs(math.sin(th_2)*math.sin(th_3)*math.cos(th_5) + math.sin(th_2)*math.sin(th_5)*math.cos(th_3)*math.cos(th_4) - math.sin(th_4)*math.sin(th_5)*math.cos(th_2)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(45, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_44_processor()
    # Finish code for equation all-zero dispatcher node 44

    # Code for explicit solution node 45, solved variable is th_0
    def ExplicitSolutionNode_node_45_solve_th_0_processor():
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
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (((abs(r_13) >= zero_tolerance)) or ((abs(r_23) >= zero_tolerance)) or ((abs(math.sin(th_2)*math.sin(th_3)*math.cos(th_5) + math.sin(th_2)*math.sin(th_5)*math.cos(th_3)*math.cos(th_4) - math.sin(th_4)*math.sin(th_5)*math.cos(th_2)) >= zero_tolerance)))
            if condition_0:
                x0 = math.atan2(r_13, -r_23)
                x1 = math.sin(th_2)
                x2 = math.sin(th_5)
                x3 = x1*x2*math.cos(th_3)*math.cos(th_4) + x1*math.sin(th_3)*math.cos(th_5) - x2*math.sin(th_4)*math.cos(th_2)
                x4 = safe_sqrt(r_13**2 + r_23**2 - x3**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x0 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(46, appended_idx)
                this_solution[0] = x0 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(46, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_45_solve_th_0_processor()
    # Finish code for explicit solution node 45

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
            th_0 = this_solution[0]
            checked_result: bool = (((abs(Pz) <= 1.0e-6)) and ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(47, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_46_processor()
    # Finish code for equation all-zero dispatcher node 46

    # Code for explicit solution node 47, solved variable is th_1
    def ExplicitSolutionNode_node_47_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (((abs(Pz) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)))
            if condition_0:
                x0 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x1 = math.sin(th_3)
                x2 = math.cos(th_3)
                x3 = math.sin(th_5)
                x4 = a_5*math.cos(th_5)
                x5 = x4*math.cos(th_4)
                x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5
                x7 = math.cos(th_2)
                x8 = a_3*x7
                x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*math.sin(th_2)*math.sin(th_4) - x8
                this_solution[1] = math.atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_47_solve_th_1_processor()
    # Finish code for explicit solution node 47

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
            th_4 = this_solution[3]
            th_6 = this_solution[5]
            condition_0: bool = (((abs(inv_Pz) >= 1.0e-6)) or ((abs(a_5 + inv_Px*math.cos(th_6) - inv_Py*math.sin(th_6)) >= 1.0e-6)))
            if condition_0:
                x0 = math.cos(th_3)
                x1 = math.sin(th_3)
                x2 = a_3*x0 + a_3 + d_2*x1
                x3 = math.cos(th_4)
                x4 = a_3*x1 - d_2*x0 + d_4
                x5 = -a_5 - inv_Px*math.cos(th_6) + inv_Py*math.sin(th_6)
                this_solution[4] = math.atan2(-inv_Pz*x2*x3 - x4*x5, -inv_Pz*x4 + x2*x3*x5)
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
        for queue_index in this_node_input_index:
            if not queue_element_validity[queue_index]:
                continue
            this_solution = solution_queue[queue_index]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            x0 = math.cos(th_6)
            x1 = math.sin(th_3)
            x2 = math.sin(th_5)
            x3 = x1*x2
            x4 = math.sin(th_6)
            x5 = math.sin(th_4)
            x6 = math.cos(th_3)
            x7 = x5*x6
            x8 = math.cos(th_4)
            x9 = math.cos(th_5)
            x10 = x6*x9
            x11 = -x0*x10*x8 + x0*x3 + x4*x7
            x12 = x4*x8
            x13 = x5*x9
            x14 = x0*x13 + x12
            x15 = x0*x7 + x10*x12 - x3*x4
            x16 = -x0*x8 + x13*x4
            x17 = x6*x8
            x18 = x1*x9 + x17*x2
            x19 = x2*x5
            x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1
            x21 = a_5*x13
            A_matrix = np.zeros((8, 4))
            A_matrix[0, 0] = r_11
            A_matrix[0, 1] = -r_21
            A_matrix[0, 2] = x11
            A_matrix[0, 3] = x14
            A_matrix[1, 0] = r_12
            A_matrix[1, 1] = -r_22
            A_matrix[1, 2] = x15
            A_matrix[1, 3] = -x16
            A_matrix[2, 0] = r_13
            A_matrix[2, 1] = -r_23
            A_matrix[2, 2] = -x18
            A_matrix[2, 3] = x19
            A_matrix[3, 0] = Px
            A_matrix[3, 1] = -Py
            A_matrix[3, 2] = -x20
            A_matrix[3, 3] = x21
            A_matrix[4, 0] = -r_11
            A_matrix[4, 1] = r_21
            A_matrix[4, 2] = -x11
            A_matrix[4, 3] = -x14
            A_matrix[5, 0] = -r_12
            A_matrix[5, 1] = r_22
            A_matrix[5, 2] = -x15
            A_matrix[5, 3] = x16
            A_matrix[6, 0] = -r_13
            A_matrix[6, 1] = r_23
            A_matrix[6, 2] = x18
            A_matrix[6, 3] = -x19
            A_matrix[7, 0] = -Px
            A_matrix[7, 1] = Py
            A_matrix[7, 2] = x20
            A_matrix[7, 3] = -x21
            linear_values = solve_linear_type2_candidates(A_matrix, [(0, 1, 2), (0, 1, 3), (0, 1, 6), (0, 1, 7), (0, 2, 3)])
            for linear_index, linear_value in enumerate(linear_values):
                candidate = copy.copy(this_solution)
                candidate[2] = linear_value
                if linear_index+1 == len(linear_values):
                    solution_queue[queue_index] = candidate
                    candidate_index = queue_index
                else:
                    candidate_index = append_solution_to_queue(candidate)
                add_input_index_to(8, candidate_index)
            if not linear_values:
                queue_element_validity[queue_index] = False
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
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            checked_result: bool = (((abs(r_13) <= 1.0e-6)) and ((abs(r_23) <= 1.0e-6)) and ((abs(math.sin(th_2)*math.sin(th_3)*math.cos(th_5) + math.sin(th_2)*math.sin(th_5)*math.cos(th_3)*math.cos(th_4) - math.sin(th_4)*math.sin(th_5)*math.cos(th_2)) <= 1.0e-6)))
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
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (((abs(r_13) >= zero_tolerance)) or ((abs(r_23) >= zero_tolerance)) or ((abs(math.sin(th_2)*math.sin(th_3)*math.cos(th_5) + math.sin(th_2)*math.sin(th_5)*math.cos(th_3)*math.cos(th_4) - math.sin(th_4)*math.sin(th_5)*math.cos(th_2)) >= zero_tolerance)))
            if condition_0:
                x0 = math.atan2(r_13, -r_23)
                x1 = math.sin(th_2)
                x2 = math.sin(th_5)
                x3 = x1*x2*math.cos(th_3)*math.cos(th_4) + x1*math.sin(th_3)*math.cos(th_5) - x2*math.sin(th_4)*math.cos(th_2)
                x4 = safe_sqrt(r_13**2 + r_23**2 - x3**2)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x0 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                this_solution[0] = x0 + math.atan2(-x4, x3)
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
            checked_result: bool = (((abs(Pz) <= 1.0e-6)) and ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue)
            else:
                add_input_index_to(48, node_input_i_idx_in_queue)

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor()
    # Finish code for equation all-zero dispatcher node 10

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
            th_0 = this_solution[0]
            checked_result: bool = (((abs(r_31) <= 1.0e-6)) and ((abs(r_32) <= 1.0e-6)) and ((abs(r_33) <= 1.0e-6)) and ((abs(r_11*math.cos(th_0) + r_21*math.sin(th_0)) <= 1.0e-6)) and ((abs(r_12*math.cos(th_0) + r_22*math.sin(th_0)) <= 1.0e-6)) and ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) <= 1.0e-6)))
            if not checked_result:  # To non-degenerate node
                add_input_index_to(49, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False

    # Invoke the processor
    EquationAllZeroDispatcherNode_node_48_processor()
    # Finish code for equation all-zero dispatcher node 48

    # Code for explicit solution node 49, solved variable is th_1
    def ExplicitSolutionNode_node_49_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = (((((abs(r_33) >= 1.0e-6)) or ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= 1.0e-6)))) and ((((((abs(r_31) < 1.0e-6)) and ((abs(r_11*math.cos(th_0) + r_21*math.sin(th_0)) < 1.0e-6)))) or ((max(abs(r_33), abs(r_13*math.cos(th_0) + r_23*math.sin(th_0))) >= max(abs(r_31), abs(r_11*math.cos(th_0) + r_21*math.sin(th_0))))))) and ((((((abs(r_32) < 1.0e-6)) and ((abs(r_12*math.cos(th_0) + r_22*math.sin(th_0)) < 1.0e-6)))) or ((max(abs(r_33), abs(r_13*math.cos(th_0) + r_23*math.sin(th_0))) >= max(abs(r_32), abs(r_12*math.cos(th_0) + r_22*math.sin(th_0))))))))
            if condition_0:
                x0 = -r_13*math.cos(th_0) - r_23*math.sin(th_0)
                x1 = math.cos(th_3)
                x2 = math.cos(th_5)
                x3 = math.sin(th_3)
                x4 = math.sin(th_5)
                x5 = x4*math.cos(th_4)
                x6 = x1*x2 - x3*x5
                x7 = math.cos(th_2)
                x8 = -x1*x5*x7 - x2*x3*x7 - x4*math.sin(th_2)*math.sin(th_4)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = math.atan2(r_33*x8 - x0*x6, r_33*x6 + x0*x8)
                appended_idx = append_solution_to_queue(solution_0)
            condition_1: bool = (((((abs(r_31) >= 1.0e-6)) or ((abs(r_11*math.cos(th_0) + r_21*math.sin(th_0)) >= 1.0e-6)))) and ((((((abs(r_32) < 1.0e-6)) and ((abs(r_12*math.cos(th_0) + r_22*math.sin(th_0)) < 1.0e-6)))) or ((max(abs(r_31), abs(r_11*math.cos(th_0) + r_21*math.sin(th_0))) >= max(abs(r_32), abs(r_12*math.cos(th_0) + r_22*math.sin(th_0))))))) and ((((((abs(r_33) < 1.0e-6)) and ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) < 1.0e-6)))) or ((max(abs(r_31), abs(r_11*math.cos(th_0) + r_21*math.sin(th_0))) > max(abs(r_33), abs(r_13*math.cos(th_0) + r_23*math.sin(th_0))))))))
            if condition_1:
                x0 = -r_11*math.cos(th_0) - r_21*math.sin(th_0)
                x1 = math.cos(th_3)
                x2 = math.cos(th_6)
                x3 = math.sin(th_5)
                x4 = math.sin(th_3)
                x5 = math.sin(th_4)
                x6 = math.sin(th_6)
                x7 = math.cos(th_4)
                x8 = x2*math.cos(th_5)
                x9 = x7*x8
                x10 = -x1*x2*x3 + x4*x5*x6 - x4*x9
                x11 = math.sin(th_2)
                x12 = math.cos(th_2)
                x13 = x1*x12*x5*x6 - x1*x12*x9 - x11*x5*x8 - x11*x6*x7 + x12*x2*x3*x4
                solution_1: IkSolution = copy.copy(this_solution)
                solution_1[1] = math.atan2(r_31*x13 - x0*x10, r_31*x10 + x0*x13)
                appended_idx = append_solution_to_queue(solution_1)
            condition_2: bool = (((((abs(r_32) >= 1.0e-6)) or ((abs(r_12*math.cos(th_0) + r_22*math.sin(th_0)) >= 1.0e-6)))) and ((((((abs(r_31) < 1.0e-6)) and ((abs(r_11*math.cos(th_0) + r_21*math.sin(th_0)) < 1.0e-6)))) or ((max(abs(r_32), abs(r_12*math.cos(th_0) + r_22*math.sin(th_0))) > max(abs(r_31), abs(r_11*math.cos(th_0) + r_21*math.sin(th_0))))))) and ((((((abs(r_33) < 1.0e-6)) and ((abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) < 1.0e-6)))) or ((max(abs(r_32), abs(r_12*math.cos(th_0) + r_22*math.sin(th_0))) > max(abs(r_33), abs(r_13*math.cos(th_0) + r_23*math.sin(th_0))))))))
            if condition_2:
                x0 = -r_12*math.cos(th_0) - r_22*math.sin(th_0)
                x1 = math.sin(th_3)
                x2 = math.cos(th_6)
                x3 = math.sin(th_4)
                x4 = x2*x3
                x5 = math.cos(th_3)
                x6 = math.sin(th_6)
                x7 = x6*math.sin(th_5)
                x8 = math.cos(th_4)
                x9 = x6*math.cos(th_5)
                x10 = x8*x9
                x11 = x1*x10 + x1*x4 + x5*x7
                x12 = math.sin(th_2)
                x13 = math.cos(th_2)
                x14 = x13*x5
                x15 = -x1*x13*x7 + x10*x14 - x12*x2*x8 + x12*x3*x9 + x14*x4
                this_solution[1] = math.atan2(r_32*x15 - x0*x11, r_32*x11 + x0*x15)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_1_processor()
    # Finish code for explicit solution node 49

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
            th_4 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (((abs(Pz) >= 1.0e-6)) or ((abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)))
            if condition_0:
                x0 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x1 = math.sin(th_3)
                x2 = math.cos(th_3)
                x3 = math.sin(th_5)
                x4 = a_5*math.cos(th_5)
                x5 = x4*math.cos(th_4)
                x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5
                x7 = math.cos(th_2)
                x8 = a_3*x7
                x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*math.sin(th_2)*math.sin(th_4) - x8
                this_solution[1] = math.atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9)
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
        value_at_3 = th_3  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[3]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[4]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[5]  # th_6
        new_ik_i[6] = value_at_6
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def franka_panda_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, th_3):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_3)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = franka_panda_ik_solve_raw(T_ee_raw_in, add_revolute_offset(th_3, th_3_offset_original2raw))
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] = ik_out_i[0] - th_0_offset_original2raw
        ik_out_i[1] = add_revolute_offset(ik_out_i[1], -th_1_offset_original2raw)
        ik_out_i[2] = add_revolute_offset(ik_out_i[2], -th_2_offset_original2raw)
        ik_out_i[3] = th_3
        ik_out_i[4] = ik_out_i[4] - th_4_offset_original2raw
        ik_out_i[5] = add_revolute_offset(ik_out_i[5], -th_5_offset_original2raw)
        ik_out_i[6] = ik_out_i[6] - th_6_offset_original2raw
        ee_pose_i = franka_panda_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -6.283185307179586, -6.283185307179586, -6.283185307179586, -3.141592653589793, -6.283185307179586, -3.141592653589793], [3.141592653589793, 0.0, 0.0, 0.0, 3.141592653589793, 0.0, 3.141592653589793], [True, True, True, True, True, True, True], [False, False, False, True, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = franka_panda_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
    return ik_output


def franka_panda_ik_solve(T_ee: np.ndarray, th_3):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not (math.isfinite(th_3)):
        return []
    result, seen = [], set()
    for transform in (franka_panda_ik_target_original_to_raw_legacy_frame, franka_panda_ik_target_original_to_raw):
        raw_target = transform(T_ee)
        for candidate in franka_panda_ik_solve_for_raw_target(T_ee, raw_target, th_3):
            key = tuple(float(value) for value in candidate.ravel())
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result


def test_ik_solve_franka_panda():
    theta_in = np.random.random(size=(7, ))
    ee_pose = franka_panda_fk(theta_in)
    ik_output = franka_panda_ik_solve(ee_pose, th_3=theta_in[3])
    for i in range(len(ik_output)):
        ee_pose_i = franka_panda_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_franka_panda()
