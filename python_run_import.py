import numpy as np
import math
from typing import Tuple, Optional, List


def trig_family_representatives(coefficients, lower, upper, seeds=()):
    """Cover every cell cut by a*cos(t)+b*sin(t)+c=0 on one revolution.

    Endpoints and every analytic boundary are retained, plus one midpoint per
    open cell. No distance-based deduplication is used: a feasible cell can be
    arbitrarily narrow. Floating-point roots are not certified interval roots.
    """
    lower, upper = float(lower), float(upper)
    period = 2*math.pi
    if not (math.isfinite(lower) and math.isfinite(upper) and
            lower <= upper and upper-lower <= period):
        raise ValueError('family bounds must span at most one finite revolution')
    points = [lower, upper]
    margin = 64*np.finfo(float).eps*(1+max(abs(lower), abs(upper)))
    for a, b, c in coefficients:
        a, b, c = float(a), float(b), float(c)
        if not all(math.isfinite(value) for value in (a, b, c)):
            continue
        scale = max(abs(a), abs(b), abs(c))
        if not math.isfinite(scale) or scale == 0:
            continue
        a, b, c = a/scale, b/scale, c/scale
        radius = math.hypot(a, b)
        if radius == 0 or abs(c) > radius + 64*np.finfo(float).eps:
            continue
        center = math.atan2(b, a)
        angle = math.acos(max(-1., min(1., -c/radius)))
        for root in (center-angle, center+angle):
            first = math.ceil((lower-root)/period)
            for turns in (first-1, first, first+1):
                value = root+turns*period
                if lower-margin <= value <= upper+margin:
                    points.append(min(upper, max(lower, value)))
    points = sorted(set(points))
    values = points + [a+.5*(b-a) for a, b in zip(points, points[1:])]
    values.extend(float(value) for value in seeds
                  if math.isfinite(float(value)) and lower <= float(value) <= upper)
    return sorted(set(values))


# General utility
def snap_near_joint_limits(values, lower, upper, periodic, fixed, tolerance=1e-6):
    result = np.asarray(values, dtype=float).reshape(-1).copy()
    changed = False
    for index, value in enumerate(result):
        lo, hi = lower[index], upper[index]
        if not math.isfinite(value):
            return None
        if fixed[index]:
            if value < lo or value > hi:
                return None
            continue
        if periodic[index] and (value < lo or value > hi):
            turns = math.ceil((lo-tolerance-value)/(2*math.pi))
            value += turns*2*math.pi
        if value < lo:
            if lo-value > tolerance:
                return None
            value = lo
            changed = True
        elif value > hi:
            if value-hi > tolerance:
                return None
            value = hi
            changed = True
        result[index] = value
    return result if changed else None


def safe_sqrt(a):
    return math.sqrt(a) if a >= 0 else 0


def safe_asin(a):
    if a > 1:
        return 0.5 * math.pi
    elif a < -1:
        return -0.5 * math.pi
    else:
        return math.asin(a)


def safe_acos(a):
    if a > 1:
        return 0.0
    elif a < -1:
        return math.pi
    else:
        return math.acos(a)


def compute_polynomial_real_roots(coefficients, tolerance: float = 1e-6) -> List[float]:
    """Return finite real roots, allowing roundoff in the imaginary part."""
    coefficients = np.asarray(coefficients, dtype=float)
    if not np.all(np.isfinite(coefficients)):
        return []
    coefficients = np.trim_zeros(coefficients, 'f')
    if coefficients.size < 2:
        return []
    coefficients = coefficients / np.max(np.abs(coefficients))
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            candidates = np.roots(coefficients)
    except np.linalg.LinAlgError:
        return []
    roots = []
    for root in candidates:
        if abs(root.imag) > tolerance * (1 + abs(root.real)):
            continue
        value = float(root.real)
        with np.errstate(over='ignore', invalid='ignore'):
            residual = np.polyval(coefficients, value)
            scale = np.polyval(np.abs(coefficients), abs(value))
        if np.isfinite(residual) and abs(residual) <= tolerance * max(1.0, scale):
            roots.append(value)
    return roots


# For general 6-dof
def compute_solution_from_tanhalf_LME(
        A_x2: np.ndarray,
        B_x: np.ndarray,
        C: np.ndarray,
        return_coupled_3_solution: bool = False):
    """
    The solver in Equation 15 of Efficient Inverse Kinematics for General 6R Manipulators.
    """
    # To 12x12 matrix
    A_12x12 = np.zeros(shape=(12, 12))
    A_12x12[0:6, 0:9] = A_x2
    A_12x12[6:12, 3:12] = A_x2
    A_inv = None
    try:
        A_inv = np.linalg.inv(A_12x12)
    except:
        # print('Cannot inverse the matrix A', print(np.linalg.det(A_12x12)))
        return None
    assert A_inv is not None
    B_12x12 = np.zeros(shape=(12, 12))
    B_12x12[0:6, 0:9] = B_x
    B_12x12[6:12, 3:12] = B_x
    C_12x12 = np.zeros(shape=(12, 12))
    C_12x12[0:6, 0:9] = C
    C_12x12[6:12, 3:12] = C

    M = np.zeros(shape=(24, 24))
    M[0:12, 12:24] = np.eye(12)
    M[12:24, 0:12] = - A_inv.dot(C_12x12)
    M[12:24, 12:24] = - A_inv.dot(B_12x12)

    # The case only need one solution
    if not return_coupled_3_solution:
        eigen_values = np.linalg.eigvals(M)
        imag_part = np.imag(eigen_values)
        real_part = np.real(eigen_values)
        solution: List[float] = list()
        for i in range(eigen_values.size):
            if abs(imag_part[i]) < 1e-6:
                atan_value = np.arctan(real_part[i])
                solution.append(2 * atan_value)
        return solution

    # We need all three coupled solution
    solution: List[Tuple[float]] = list()
    eigen_values, eigen_vectors = np.linalg.eig(M)
    for i in range(eigen_values.size):
        if abs(np.imag(eigen_values[i])) > 1e-6:
            continue
        tanhalf_x3 = np.real(eigen_values[i])
        theta3_value = 2.0 * np.arctan(tanhalf_x3)

        # The remaining part
        full_ev_i = eigen_vectors[:, i]
        ev_i = full_ev_i[0:12]
        tanhalf_x4 = ev_i[3] / ev_i[0]
        tanhalf_x5 = ev_i[1] / ev_i[0]
        if np.imag(tanhalf_x4) > 1e-6 or np.imag(tanhalf_x5) > 1e-6:
            continue
        theta4_value = 2.0 * np.arctan(np.real(tanhalf_x4))
        theta5_value = 2.0 * np.arctan(np.real(tanhalf_x5))
        solution.append((float(theta3_value), float(theta4_value), float(theta5_value)))
    return solution


# For linear solver type2
def try_solve_linear_type2(A: np.ndarray, allow_family: bool = True) -> Optional[Tuple[float, float]]:
    """
    Solve y from three linear equations in the form
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    Each row of matrix corresponds to A, B, C, D
    """
    assert A.shape[0] == 3
    assert A.shape[1] == 4
    if not np.all(np.isfinite(A)):
        return None
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    ad_minus_bc = a * d - b * c
    zero_tolerance = 1e-10
    if abs(ad_minus_bc) < zero_tolerance:
        return None
    A_inv = np.zeros(shape=(2, 2))
    A_inv[0, 0] = d
    A_inv[0, 1] = -b
    A_inv[1, 0] = -c
    A_inv[1, 1] = a
    A_inv /= ad_minus_bc
    # [sin(x), cos(x)].T = B * [sin(y), cos(y)].T
    B = - A_inv.dot(A[0:2, 2:4])

    # e sin(y) + f * cos(y) == 0
    e = A[2, 2] + A[2, 0] * B[0, 0] + A[2, 1] * B[1, 0]
    f = A[2, 3] + A[2, 0] * B[0, 1] + A[2, 1] * B[1, 1]
    if abs(e) < zero_tolerance and abs(f) < zero_tolerance:
        # Orthogonal B leaves a continuous unit-circle family; retain representatives.
        if allow_family and np.max(np.abs(B.T @ B - np.eye(2))) <= 1e-10:
            return 0.0, math.pi
        return None
    solution_0 = math.atan2(- f, e)
    solution_1 = solution_0 + np.pi
    if solution_1 > np.pi:
        solution_1 -= 2 * np.pi
    return solution_0, solution_1


def try_solve_linear_type2_specific_rows(
        A_combined: np.ndarray, row_0: int, row_1: int, row_2: int) -> Optional[Tuple[float, float]]:
    """
    For a m x 4 matrix A, each row of whom corresponds to a linear equation in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    Select three equations from row_0/1/2 and solve it
    """
    # A rank-three homogeneous system has a one-dimensional nullspace.
    # Recover it from all rows instead of subtracting nearly equal elimination
    # residuals from a preselected triple near a singular configuration.
    if np.all(np.isfinite(A_combined)):
        scale = np.max(np.abs(A_combined))
        if scale > 0:
            try:
                _, singular, vectors = np.linalg.svd(A_combined/scale, full_matrices=len(A_combined) < 4)
                if (len(singular) >= 3 and singular[2] > 1e-10*singular[0] and
                        (len(singular) == 3 or singular[3] <= 1e-8*singular[0])):
                    direction = vectors[-1, 2:]
                    if np.linalg.norm(direction) > 1e-10:
                        angle = math.atan2(float(direction[0]), float(direction[1]))
                        other = angle+math.pi
                        if other > math.pi:
                            other -= 2*math.pi
                        return angle, other
            except np.linalg.LinAlgError:
                pass
    A_tmp = np.zeros(shape=(3, 4))
    A_tmp[0, :] = A_combined[row_0, :]
    A_tmp[1, :] = A_combined[row_1, :]
    A_tmp[2, :] = A_combined[row_2, :]
    # Do not choose arbitrary family representatives while other rows may constrain y.
    A_tmp_sol = try_solve_linear_type2(A_tmp, allow_family=False)
    if A_tmp_sol is not None:
        return A_tmp_sol
    if not np.all(np.isfinite(A_combined)):
        return None
    best_pair = None
    best_det = 1e-10
    for i in range(len(A_combined)):
        for j in range(i + 1, len(A_combined)):
            det = abs(A_combined[i, 0] * A_combined[j, 1] - A_combined[i, 1] * A_combined[j, 0])
            if det > best_det:
                best_det, best_pair = det, (i, j)
    if best_pair is None:
        return None
    top = A_combined[list(best_pair)]
    try:
        B = -np.linalg.solve(top[:, :2], top[:, 2:])
    except np.linalg.LinAlgError:
        return None
    residual = A_combined[:, 2:] + A_combined[:, :2] @ B
    remaining = [i for i in range(len(A_combined)) if i not in best_pair]
    third = max(remaining, key=lambda i: float(residual[i] @ residual[i]))
    return try_solve_linear_type2(A_combined[[best_pair[0], best_pair[1], third]])
