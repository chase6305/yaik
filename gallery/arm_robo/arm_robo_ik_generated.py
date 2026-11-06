import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 26
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
l_1: float = 0.19681
l_2: float = 0.251
l_3: float = 0.145423

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def arm_robo_ik_target_original_to_raw(T_ee: np.ndarray):
    return T_ee


def arm_robo_ik_target_raw_to_original(T_ee: np.ndarray):
    return T_ee


def arm_robo_fk(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_4)
    x1 = math.sin(th_2)
    x2 = math.sin(th_0)
    x3 = math.sin(th_1)
    x4 = math.cos(th_0)
    x5 = math.cos(th_1)
    x6 = x2*x3 - x4*x5
    x7 = -x6
    x8 = x1*x7
    x9 = math.cos(th_4)
    x10 = math.sin(th_3)
    x11 = x2*x5 + x3*x4
    x12 = x10*x11
    x13 = math.cos(th_3)
    x14 = math.cos(th_2)
    x15 = x14*x7
    x16 = -x12 + x13*x15
    x17 = x10*x15 + x11*x13
    x18 = x1*x13
    x19 = x1*x10
    x20 = x1*x11
    x21 = -x10*x6 + x11*x13*x14
    x22 = x12*x14 + x13*x6
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x8 + x16*x9
    ee_pose[0, 1] = -x0*x16 - x8*x9
    ee_pose[0, 2] = x17
    ee_pose[0, 3] = d_6*x17 + l_1*x4 - l_3*x8
    ee_pose[1, 0] = -x0*x14 - x18*x9
    ee_pose[1, 1] = x0*x18 - x14*x9
    ee_pose[1, 2] = -x19
    ee_pose[1, 3] = -d_6*x19 - l_2 - l_3*x14
    ee_pose[2, 0] = -x0*x20 + x21*x9
    ee_pose[2, 1] = -x0*x21 - x20*x9
    ee_pose[2, 2] = x22
    ee_pose[2, 3] = d_6*x22 + l_1*x2 - l_3*x20
    return ee_pose


def arm_robo_twist_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_1)
    x2 = math.sin(th_1)
    x3 = math.cos(th_0)
    x4 = x0*x1 + x2*x3
    x5 = math.sin(th_2)
    x6 = x0*x2 - x1*x3
    x7 = -x6
    x8 = x5*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = math.cos(th_2)
    x12 = x10*x11
    x13 = x12*x7 + x4*x9
    x14 = x10*x5
    x15 = -x14
    x16 = x4*x5
    x17 = x12*x4 + x6*x9
    x18 = l_1*x0
    x19 = -l_2 - l_3*x11
    x20 = -l_3*x16 + x18
    x21 = l_1*x3
    x22 = -l_3*x8 + x21
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 2] = x4
    jacobian_output[0, 3] = -x8
    jacobian_output[0, 4] = x13
    jacobian_output[1, 0] = -1
    jacobian_output[1, 1] = -1
    jacobian_output[1, 3] = -x11
    jacobian_output[1, 4] = x15
    jacobian_output[2, 2] = x6
    jacobian_output[2, 3] = -x16
    jacobian_output[2, 4] = x17
    jacobian_output[3, 1] = x18
    jacobian_output[3, 2] = -l_2*x6
    jacobian_output[3, 3] = x11*x20 - x16*x19
    jacobian_output[3, 4] = x14*x20 + x17*x19
    jacobian_output[3, 5] = x13
    jacobian_output[4, 2] = l_1*x0*x4 - x21*x6
    jacobian_output[4, 3] = -x20*x8 + x22*x4*x5
    jacobian_output[4, 4] = x13*x20 - x17*x22
    jacobian_output[4, 5] = x15
    jacobian_output[5, 1] = -x21
    jacobian_output[5, 2] = l_2*x4
    jacobian_output[5, 3] = -x11*x22 + x19*x8
    jacobian_output[5, 4] = -x13*x19 - x14*x22
    jacobian_output[5, 5] = x17
    return jacobian_output


def arm_robo_angular_velocity_jacobian(theta_input: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_1)
    x2 = math.sin(th_1)
    x3 = math.cos(th_0)
    x4 = x0*x1 + x2*x3
    x5 = math.sin(th_2)
    x6 = x0*x2 - x1*x3
    x7 = -x6
    x8 = math.cos(th_3)
    x9 = math.sin(th_3)
    x10 = math.cos(th_2)
    x11 = x10*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 2] = x4
    jacobian_output[0, 3] = -x5*x7
    jacobian_output[0, 4] = x11*x7 + x4*x8
    jacobian_output[1, 0] = -1
    jacobian_output[1, 1] = -1
    jacobian_output[1, 3] = -x10
    jacobian_output[1, 4] = -x5*x9
    jacobian_output[2, 2] = x6
    jacobian_output[2, 3] = -x4*x5
    jacobian_output[2, 4] = x11*x4 + x6*x8
    return jacobian_output


def arm_robo_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    theta_input = np.asarray(theta_input).reshape(-1)
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = -p_on_ee_z
    x1 = math.sin(th_0)
    x2 = l_1*x1
    x3 = math.sin(th_1)
    x4 = math.cos(th_0)
    x5 = math.cos(th_1)
    x6 = x1*x3 - x4*x5
    x7 = math.cos(th_2)
    x8 = math.sin(th_2)
    x9 = x1*x5 + x3*x4
    x10 = p_on_ee_y*x9
    x11 = x8*x9
    x12 = -l_3*x11 + x2
    x13 = -l_2 - l_3*x7
    x14 = math.sin(th_3)
    x15 = x14*x8
    x16 = math.cos(th_3)
    x17 = x14*x7
    x18 = x16*x6 + x17*x9
    x19 = -x6
    x20 = x16*x9 + x17*x19
    x21 = l_1*x4
    x22 = x19*x8
    x23 = -l_3*x22 + x21
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = x0
    jacobian_output[0, 1] = x0 + x2
    jacobian_output[0, 2] = -l_2*x6 - p_on_ee_y*x6
    jacobian_output[0, 3] = -p_on_ee_z*x7 + x10*x8 - x11*x13 + x12*x7
    jacobian_output[0, 4] = -p_on_ee_y*x18 - p_on_ee_z*x15 + x12*x14*x8 + x13*x18
    jacobian_output[0, 5] = x20
    jacobian_output[1, 2] = l_1*x1*x9 + p_on_ee_x*x6 - p_on_ee_z*x9 - x21*x6
    jacobian_output[1, 3] = -p_on_ee_x*x11 + p_on_ee_z*x19*x8 - x12*x22 + x23*x8*x9
    jacobian_output[1, 4] = p_on_ee_x*x18 - p_on_ee_z*x20 + x12*x20 - x18*x23
    jacobian_output[1, 5] = -x15
    jacobian_output[2, 0] = p_on_ee_x
    jacobian_output[2, 1] = p_on_ee_x - x21
    jacobian_output[2, 2] = l_2*x9 + x10
    jacobian_output[2, 3] = p_on_ee_x*x7 - p_on_ee_y*x22 + x13*x22 - x23*x7
    jacobian_output[2, 4] = p_on_ee_x*x15 + p_on_ee_y*x20 - x13*x20 - x15*x23
    jacobian_output[2, 5] = x18
    return jacobian_output


def arm_robo_ik_solve_raw(T_ee: np.ndarray):
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
        return IkSolution([math.nan] * 10)

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
    for i in range(26):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True

    # Code for non-branch dispatcher node 0
    # Actually, there is no code

    # Code for polynomial solution node 0, solved variable is th_0
    def PolynomialSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return

        # The polynomial solution of root node
        poly_coefficient_0_num = (inv_Px**2 - 2*inv_Px*l_1*r_11 - 2*inv_Px*l_2*r_21 + inv_Py**2 - 2*inv_Py*l_1*r_12 - 2*inv_Py*l_2*r_22 + l_1**2*r_11**2 + l_1**2*r_12**2 + 2*l_1*l_2*r_11*r_21 + 2*l_1*l_2*r_12*r_22 + l_2**2*r_21**2 + l_2**2*r_22**2 - l_3**2)*(inv_Px**2 + 2*inv_Px*l_1*r_11 - 2*inv_Px*l_2*r_21 + inv_Py**2 + 2*inv_Py*l_1*r_12 - 2*inv_Py*l_2*r_22 + l_1**2*r_11**2 + l_1**2*r_12**2 - 2*l_1*l_2*r_11*r_21 - 2*l_1*l_2*r_12*r_22 + l_2**2*r_21**2 + l_2**2*r_22**2 - l_3**2)
        poly_coefficient_0_denom = 1
        if (not isinstance(poly_coefficient_0_num, int) and not np.isfinite(poly_coefficient_0_num)) or (not isinstance(poly_coefficient_0_denom, int) and not np.isfinite(poly_coefficient_0_denom)) or poly_coefficient_0_denom == 0:
            return
        try:
            poly_coefficient_0 = poly_coefficient_0_num / poly_coefficient_0_denom
        except OverflowError:
            return
        if not np.isfinite(poly_coefficient_0):
            return
        poly_coefficient_1_num = 4*l_1*(inv_Px**3*r_31 + inv_Px**2*inv_Py*r_32 - 3*inv_Px**2*l_2*r_21*r_31 - inv_Px**2*l_2*r_22*r_32 + inv_Px*inv_Py**2*r_31 - 2*inv_Px*inv_Py*l_2*r_21*r_32 - 2*inv_Px*inv_Py*l_2*r_22*r_31 - inv_Px*l_1**2*r_11**2*r_31 - 2*inv_Px*l_1**2*r_11*r_12*r_32 + inv_Px*l_1**2*r_12**2*r_31 + 3*inv_Px*l_2**2*r_21**2*r_31 + 2*inv_Px*l_2**2*r_21*r_22*r_32 + inv_Px*l_2**2*r_22**2*r_31 - inv_Px*l_3**2*r_31 + inv_Py**3*r_32 - inv_Py**2*l_2*r_21*r_31 - 3*inv_Py**2*l_2*r_22*r_32 + inv_Py*l_1**2*r_11**2*r_32 - 2*inv_Py*l_1**2*r_11*r_12*r_31 - inv_Py*l_1**2*r_12**2*r_32 + inv_Py*l_2**2*r_21**2*r_32 + 2*inv_Py*l_2**2*r_21*r_22*r_31 + 3*inv_Py*l_2**2*r_22**2*r_32 - inv_Py*l_3**2*r_32 + l_1**2*l_2*r_11**2*r_21*r_31 - l_1**2*l_2*r_11**2*r_22*r_32 + 2*l_1**2*l_2*r_11*r_12*r_21*r_32 + 2*l_1**2*l_2*r_11*r_12*r_22*r_31 - l_1**2*l_2*r_12**2*r_21*r_31 + l_1**2*l_2*r_12**2*r_22*r_32 - l_2**3*r_21**3*r_31 - l_2**3*r_21**2*r_22*r_32 - l_2**3*r_21*r_22**2*r_31 - l_2**3*r_22**3*r_32 + l_2*l_3**2*r_21*r_31 + l_2*l_3**2*r_22*r_32)
        poly_coefficient_1_denom = 1
        if (not isinstance(poly_coefficient_1_num, int) and not np.isfinite(poly_coefficient_1_num)) or (not isinstance(poly_coefficient_1_denom, int) and not np.isfinite(poly_coefficient_1_denom)) or poly_coefficient_1_denom == 0:
            return
        try:
            poly_coefficient_1 = poly_coefficient_1_num / poly_coefficient_1_denom
        except OverflowError:
            return
        if not np.isfinite(poly_coefficient_1):
            return
        poly_coefficient_2_num = 2*l_1**2*(inv_Px**2*r_11**2 - inv_Px**2*r_12**2 + 3*inv_Px**2*r_31**2 + inv_Px**2*r_32**2 + 4*inv_Px*inv_Py*r_11*r_12 + 4*inv_Px*inv_Py*r_31*r_32 - 2*inv_Px*l_2*r_11**2*r_21 - 4*inv_Px*l_2*r_11*r_12*r_22 + 2*inv_Px*l_2*r_12**2*r_21 - 6*inv_Px*l_2*r_21*r_31**2 - 2*inv_Px*l_2*r_21*r_32**2 - 4*inv_Px*l_2*r_22*r_31*r_32 - inv_Py**2*r_11**2 + inv_Py**2*r_12**2 + inv_Py**2*r_31**2 + 3*inv_Py**2*r_32**2 + 2*inv_Py*l_2*r_11**2*r_22 - 4*inv_Py*l_2*r_11*r_12*r_21 - 2*inv_Py*l_2*r_12**2*r_22 - 4*inv_Py*l_2*r_21*r_31*r_32 - 2*inv_Py*l_2*r_22*r_31**2 - 6*inv_Py*l_2*r_22*r_32**2 - l_1**2*r_11**4 - 2*l_1**2*r_11**2*r_12**2 - l_1**2*r_11**2*r_31**2 + l_1**2*r_11**2*r_32**2 - 4*l_1**2*r_11*r_12*r_31*r_32 - l_1**2*r_12**4 + l_1**2*r_12**2*r_31**2 - l_1**2*r_12**2*r_32**2 + l_2**2*r_11**2*r_21**2 - l_2**2*r_11**2*r_22**2 + 4*l_2**2*r_11*r_12*r_21*r_22 - l_2**2*r_12**2*r_21**2 + l_2**2*r_12**2*r_22**2 + 3*l_2**2*r_21**2*r_31**2 + l_2**2*r_21**2*r_32**2 + 4*l_2**2*r_21*r_22*r_31*r_32 + l_2**2*r_22**2*r_31**2 + 3*l_2**2*r_22**2*r_32**2 + l_3**2*r_11**2 + l_3**2*r_12**2 - l_3**2*r_31**2 - l_3**2*r_32**2)
        poly_coefficient_2_denom = 1
        if (not isinstance(poly_coefficient_2_num, int) and not np.isfinite(poly_coefficient_2_num)) or (not isinstance(poly_coefficient_2_denom, int) and not np.isfinite(poly_coefficient_2_denom)) or poly_coefficient_2_denom == 0:
            return
        try:
            poly_coefficient_2 = poly_coefficient_2_num / poly_coefficient_2_denom
        except OverflowError:
            return
        if not np.isfinite(poly_coefficient_2):
            return
        poly_coefficient_3_num = 4*l_1**3*(inv_Px*r_11**2*r_31 + 2*inv_Px*r_11*r_12*r_32 - inv_Px*r_12**2*r_31 + inv_Px*r_31**3 + inv_Px*r_31*r_32**2 - inv_Py*r_11**2*r_32 + 2*inv_Py*r_11*r_12*r_31 + inv_Py*r_12**2*r_32 + inv_Py*r_31**2*r_32 + inv_Py*r_32**3 - l_2*r_11**2*r_21*r_31 + l_2*r_11**2*r_22*r_32 - 2*l_2*r_11*r_12*r_21*r_32 - 2*l_2*r_11*r_12*r_22*r_31 + l_2*r_12**2*r_21*r_31 - l_2*r_12**2*r_22*r_32 - l_2*r_21*r_31**3 - l_2*r_21*r_31*r_32**2 - l_2*r_22*r_31**2*r_32 - l_2*r_22*r_32**3)
        poly_coefficient_3_denom = 1
        if (not isinstance(poly_coefficient_3_num, int) and not np.isfinite(poly_coefficient_3_num)) or (not isinstance(poly_coefficient_3_denom, int) and not np.isfinite(poly_coefficient_3_denom)) or poly_coefficient_3_denom == 0:
            return
        try:
            poly_coefficient_3 = poly_coefficient_3_num / poly_coefficient_3_denom
        except OverflowError:
            return
        if not np.isfinite(poly_coefficient_3):
            return
        poly_coefficient_4_num = l_1**4*(r_11**2 - 2*r_11*r_32 + r_12**2 + 2*r_12*r_31 + r_31**2 + r_32**2)*(r_11**2 + 2*r_11*r_32 + r_12**2 - 2*r_12*r_31 + r_31**2 + r_32**2)
        poly_coefficient_4_denom = 1
        if (not isinstance(poly_coefficient_4_num, int) and not np.isfinite(poly_coefficient_4_num)) or (not isinstance(poly_coefficient_4_denom, int) and not np.isfinite(poly_coefficient_4_denom)) or poly_coefficient_4_denom == 0:
            return
        try:
            poly_coefficient_4 = poly_coefficient_4_num / poly_coefficient_4_denom
        except OverflowError:
            return
        if not np.isfinite(poly_coefficient_4):
            return
        p_coefficients = np.zeros(shape=(4 + 1,))
        p_coefficients[4] = poly_coefficient_0
        p_coefficients[3] = poly_coefficient_1
        p_coefficients[2] = poly_coefficient_2
        p_coefficients[1] = poly_coefficient_3
        p_coefficients[0] = poly_coefficient_4

        # Note that in np.roots, p_coefficient[0] is the highest order
        poly_roots = compute_polynomial_real_roots(p_coefficients)

        # Result collection
        for root_idx in range(len(poly_roots)):
            this_root = poly_roots[root_idx]
            if abs(this_root) > 1 + zero_tolerance:
                continue
            this_root = min(1.0, max(-1.0, this_root))
            first_angle = np.arcsin(this_root)
            second_angle = np.pi - np.arcsin(this_root)
            solution_0: IkSolution = make_ik_solution()
            solution_0[2] = first_angle
            solution_1: IkSolution = make_ik_solution()
            solution_1[2] = second_angle
            appended_idx_0 = append_solution_to_queue(solution_0)
            appended_idx_1 = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx_0)
            add_input_index_to(2, appended_idx_1)

    # Invoke the processor
    PolynomialSolutionNode_node_1_solve_th_0_processor()
    # Finish code for polynomial solution node 0

    # Code for non-branch dispatcher node 2
    # Actually, there is no code

    # Code for explicit solution node 3, solved variable is d_6
    def ExplicitSolutionNode_node_3_solve_d_6_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[0] = -inv_Pz - l_1*r_13*math.cos(th_0) - l_1*r_33*math.sin(th_0) + l_2*r_23
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_d_6_processor()
    # Finish code for explicit solution node 2

    # Code for non-branch dispatcher node 4
    # Actually, there is no code

    # Code for explicit solution node 5, solved variable is th_4
    def ExplicitSolutionNode_node_5_solve_th_4_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (((abs(l_3) >= zero_tolerance)) or ((abs(inv_Px + l_1*(r_11*math.cos(th_0) + r_31*math.sin(th_0)) - l_2*r_21) >= zero_tolerance)) or ((abs(inv_Py + l_1*(r_12*math.cos(th_0) + r_32*math.sin(th_0)) - l_2*r_22) >= zero_tolerance)))
            if condition_0:
                x0 = 1/l_3
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                this_solution[9] = math.atan2(x0*(-inv_Px - l_1*(r_11*x1 + r_31*x2) + l_2*r_21), x0*(-inv_Py - l_1*(r_12*x1 + r_32*x2) + l_2*r_22))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_4_processor()
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
            d_6 = this_solution[0]
            condition_0: bool = (((l_3 != 0)) and ((abs((Py - d_6*r_23 + l_2)/l_3) <= 70368744177665/70368744177664)))
            if condition_0:
                x0 = safe_acos((-Py + d_6*r_23 - l_2)/l_3)
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                this_solution[7] = -x0
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
            th_2 = this_solution[7]
            degenerate_valid_0 = (abs(th_2) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(14, branch_input_index)

            th_2 = this_solution[7]
            degenerate_valid_1 = (abs(th_2 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                branch_input_index = node_input_i_idx_in_queue
                if forwarded_to_child:
                    branch_input_index = append_solution_to_queue(copy.copy(this_solution))
                forwarded_to_child = True
                add_input_index_to(21, branch_input_index)

            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)

    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8

    # Code for explicit solution node 21, solved variable is negative_th_3_positive_th_1__soa
    def ExplicitSolutionNode_node_21_solve_negative_th_3_positive_th_1__soa_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0) + r_33*math.sin(th_0)) >= zero_tolerance)) or ((abs(r_13*math.sin(th_0) - r_33*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                this_solution[1] = math.atan2(r_13*x0 + r_33*x1, r_13*x1 - r_33*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_negative_th_3_positive_th_1__soa_processor()
    # Finish code for explicit solution node 21

    # Code for non-branch dispatcher node 22
    # Actually, there is no code

    # Code for explicit solution node 23, solved variable is th_1
    def ExplicitSolutionNode_node_23_solve_th_1_processor():
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
            condition_0: bool = True
            if condition_0:
                this_solution[5] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor()
    # Finish code for explicit solution node 22

    # Code for non-branch dispatcher node 24
    # Actually, there is no code

    # Code for explicit solution node 25, solved variable is th_3
    def ExplicitSolutionNode_node_25_solve_th_3_processor():
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
            negative_th_3_positive_th_1__soa = this_solution[1]
            th_1 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[8] = -negative_th_3_positive_th_1__soa + th_1
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_3_processor()
    # Finish code for explicit solution node 24

    # Code for explicit solution node 14, solved variable is th_0th_1th_3_soa
    def ExplicitSolutionNode_node_14_solve_th_0th_1th_3_soa_processor():
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
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13) >= zero_tolerance)) or ((abs(r_33) >= zero_tolerance)))
            if condition_0:
                this_solution[4] = math.atan2(r_13, -r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_0th_1th_3_soa_processor()
    # Finish code for explicit solution node 14

    # Code for non-branch dispatcher node 15
    # Actually, there is no code

    # Code for explicit solution node 16, solved variable is th_1th_3_soa
    def ExplicitSolutionNode_node_16_solve_th_1th_3_soa_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (((1 >= zero_tolerance)) or ((abs(r_13*math.cos(th_0) + r_33*math.sin(th_0)) >= zero_tolerance)) or ((abs(r_13*math.sin(th_0) - r_33*math.cos(th_0)) >= zero_tolerance)))
            if condition_0:
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                this_solution[6] = math.atan2(r_13*x0 + r_33*x1, r_13*x1 - r_33*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1th_3_soa_processor()
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
            condition_0: bool = True
            if condition_0:
                this_solution[5] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(19, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_1_processor()
    # Finish code for explicit solution node 17

    # Code for non-branch dispatcher node 19
    # Actually, there is no code

    # Code for explicit solution node 20, solved variable is th_3
    def ExplicitSolutionNode_node_20_solve_th_3_processor():
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
            th_1 = this_solution[5]
            th_1th_3_soa = this_solution[6]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[8] = -th_1 + th_1th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_3_processor()
    # Finish code for explicit solution node 19

    # Code for explicit solution node 9, solved variable is th_3
    def ExplicitSolutionNode_node_9_solve_th_3_processor():
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
            th_2 = this_solution[7]
            th_4 = this_solution[9]
            condition_0: bool = (((abs(r_23) >= zero_tolerance)) or ((abs(math.sin(th_2)) >= zero_tolerance)) or ((abs(r_21*math.cos(th_4) - r_22*math.sin(th_4)) >= zero_tolerance)))
            if condition_0:
                x0 = 1/math.sin(th_2)
                this_solution[8] = math.atan2(-r_23*x0, x0*(-r_21*math.cos(th_4) + r_22*math.sin(th_4)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_3_processor()
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
            d_6 = this_solution[0]
            th_0 = this_solution[2]
            th_2 = this_solution[7]
            condition_0: bool = (((abs(l_3*math.sin(th_2)) >= zero_tolerance)) or ((abs(-Px*math.sin(th_0) + Pz*math.cos(th_0) + d_6*(r_13*math.sin(th_0) - r_33*math.cos(th_0))) >= zero_tolerance)) or ((abs(Px*math.cos(th_0) + Pz*math.sin(th_0) - d_6*(r_13*math.cos(th_0) + r_33*math.sin(th_0)) - l_1) >= zero_tolerance)))
            if condition_0:
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = 1/(l_3*math.sin(th_2))
                this_solution[5] = math.atan2(x2*(Px*x0 - Pz*x1 + d_6*(-r_13*x0 + r_33*x1)), x2*(-Px*x1 - Pz*x0 + d_6*(r_13*x1 + r_33*x0) + l_1))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 10

    # Code for non-branch dispatcher node 12
    # Actually, there is no code

    # Code for explicit solution node 13, solved variable is th_0th_1_soa
    def ExplicitSolutionNode_node_13_solve_th_0th_1_soa_processor():
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
            th_0 = this_solution[2]
            th_1 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                this_solution[3] = th_0 + th_1
                solution_queue[node_input_i_idx_in_queue] = this_solution
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_0th_1_soa_processor()
    # Finish code for explicit solution node 12

    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = ik_out_i[2]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[5]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[7]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[8]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[9]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[0]  # d_6
        new_ik_i[5] = value_at_5
        if np.all(np.isfinite(new_ik_i)):
            ik_out.append(new_ik_i)
    return ik_out


def arm_robo_ik_solve_for_raw_target(T_ee: np.ndarray, T_ee_raw_in: np.ndarray, _yaik_seed_candidates=None):
    if not np.all(np.isfinite(T_ee)):
        return []
    if not np.all(np.isfinite(T_ee_raw_in)):
        return []
    ik_output_raw = arm_robo_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ee_pose_i = arm_robo_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if _yaik_seed_candidates is not None:
            if ee_pose_diff < 1e-7:
                _yaik_seed_candidates.strict_witness = True
                _yaik_seed_candidates.clear()
            elif not _yaik_seed_candidates.strict_witness:
                retain_refinement_seed(_yaik_seed_candidates, ik_out_i, ee_pose_diff)
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
        snapped = snap_near_joint_limits(ik_out_i, [-3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793, -3.141592653589793, -1.0], [3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 1.0], [True, True, True, True, True, False], [False, False, False, False, False, False])
        if snapped is not None:
            snapped = snapped.reshape(ik_out_i.shape)
            snapped_pose = arm_robo_fk(snapped)
            if (np.linalg.norm(snapped_pose[:3, 3]-T_ee[:3, 3]) <= 1e-7 and np.max(np.abs(snapped_pose[:3, :3]-T_ee[:3, :3])) <= 1e-7):
                ik_output.append(snapped)
                if _yaik_seed_candidates is not None and np.max(np.abs(snapped_pose-T_ee)) < 1e-7:
                    _yaik_seed_candidates.strict_witness = True
                    _yaik_seed_candidates.clear()
    return ik_output


def arm_robo_ik_solve(T_ee: np.ndarray):
    if not np.all(np.isfinite(T_ee)):
        return []
    result, seen = [], set()
    refinement_seeds = RefinementSeeds()
    for transform in (arm_robo_ik_target_original_to_raw,):
        raw_target = transform(T_ee)
        for candidate in arm_robo_ik_solve_for_raw_target(T_ee, raw_target, _yaik_seed_candidates=refinement_seeds):
            result.append(candidate)
    refinement_fk = lambda q: arm_robo_fk(q)
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
            for transform in (arm_robo_ik_target_original_to_raw,):
                raw_target = transform(shifted)
                for raw_candidate in arm_robo_ik_solve_raw(raw_target, *seed_parameters):
                    seed = raw_candidate.reshape(-1).copy()
                    retain_ik_refinement_seed(refinement_seeds, refinement_fk, T_ee, seed)
        for _, seed in refinement_seeds:
            refined = refine_ik_seed(refinement_fk, T_ee, seed, [False, False, False, False, False, False])
            if refined is not None:
                result.append(refined.reshape((-1, 1)))
                refinement_seeds.strict_witness = True
                break
    return result


def test_ik_solve_arm_robo():
    theta_in = np.random.random(size=(6, ))
    ee_pose = arm_robo_fk(theta_in)
    ik_output = arm_robo_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = arm_robo_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_arm_robo()
