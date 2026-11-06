import numpy as np
import math
from typing import Tuple, Optional, List


def add_revolute_offset(value, offset):
    """Add an angular offset without losing it to a large angle's float ulp."""
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(offset, np.ndarray):
        offset = offset.item()
    if offset == 0 or (abs(value) <= 1048576. and abs(offset) <= 1048576.):
        return value+offset
    phase = math.atan2(math.sin(value), math.cos(value))
    shift = math.atan2(math.sin(offset), math.cos(offset)) if abs(offset) > math.pi else offset
    return phase+shift


def fixed_joint_phase_aliases(arguments, centers, tolerance=1e-6):
    """Try equivalent representatives near saved branch centers after failure."""
    original = tuple(float(value) for value in arguments)
    aliases, seen, combined, changed = [], set(), list(original), set()
    for index, center in centers:
        value = original[index]
        if not (math.isfinite(value) and math.isfinite(center)) or abs(value-center) <= tolerance:
            continue
        sv, cv, sc, cc = math.sin(value), math.cos(value), math.sin(center), math.cos(center)
        phase = math.atan2(sv*cc-cv*sc, cv*cc+sv*sc)
        if abs(phase) > tolerance:
            continue
        replacement = center+phase
        alias = list(original)
        alias[index] = replacement
        alias = tuple(alias)
        if alias not in seen:
            seen.add(alias)
            aliases.append(alias)
        if index not in changed:
            combined[index] = replacement
            changed.add(index)
    combined = tuple(combined)
    if len(changed) > 1 and combined not in seen:
        aliases.append(combined)
    return aliases


class RefinementSeeds(list):
    def __init__(self):
        super().__init__()
        self.strict_witness = False


def nearby_ik_targets(target):
    """Six deterministic small target shifts, used only to construct seeds."""
    targets = []
    for direction in ((1., 2., 3.), (3., 1., -2.), (-2., 3., 1.)):
        axis = np.array(direction)/math.sqrt(14.)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        for sign in (-1., 1.):
            angle = sign*1e-4
            perturb = np.eye(4)
            perturb[:3, :3] += math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)
            perturb[:3, 3] = sign*1e-4*axis
            with np.errstate(over='ignore', invalid='ignore'):
                shifted = perturb@target
            if np.isfinite(shifted).all():
                targets.append(shifted)
    return targets


def retain_refinement_seed(seeds, candidate, error):
    """Keep a few distinct finite raw candidates, ordered by original FK error."""
    if not math.isfinite(error) or (len(seeds) >= 4 and error >= seeds[-1][0]):
        return
    candidate = np.asarray(candidate, dtype=float).reshape(-1)
    if not np.isfinite(candidate).all():
        return
    if any(np.array_equal(candidate, entry[1]) for entry in seeds):
        return
    if len(seeds) < 4 or error < seeds[-1][0]:
        seeds.append((float(error), candidate.copy()))
        seeds.sort(key=lambda entry: entry[0])
        del seeds[4:]


def retain_ik_refinement_seed(seeds, fk, target, candidate):
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            error = float(np.max(np.abs(fk(candidate)-target)))
    except (ArithmeticError, ValueError):
        return
    retain_refinement_seed(seeds, candidate, error)


def refine_ik_seed(fk, target, seed, fixed_joints=()):
    """Refine one finite seed with bounded steps; return only a strict FK witness.

    This local search does not establish reachability or completeness. Fixed
    coordinates are never perturbed. Failure leaves the caller's seed untouched.
    """
    q = np.asarray(seed, dtype=float).reshape(-1).copy()
    target = np.asarray(target, dtype=float)
    fixed = np.asarray(fixed_joints if len(fixed_joints) else [False]*len(q), dtype=bool)
    if target.shape != (4, 4) or fixed.shape != q.shape or not np.isfinite(target).all() or not np.isfinite(q).all():
        return None
    active = np.flatnonzero(~fixed)

    def residual(values):
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                pose = np.asarray(fk(values), dtype=float)
                error = pose-target
            if pose.shape == target.shape and np.isfinite(error).all():
                return error.reshape(-1)
        except (ArithmeticError, ValueError):
            pass
        return None

    error = residual(q)
    if error is None:
        return None
    for _ in range(40):
        if np.max(np.abs(error)) < 1e-8:
            return q
        if not len(active):
            return None
        with np.errstate(over='ignore', invalid='ignore'):
            previous_norm = float(np.dot(error, error))
        if not math.isfinite(previous_norm):
            return None
        columns = []
        for joint in active:
            plus, minus = q.copy(), q.copy()
            plus[joint] += 1e-6
            minus[joint] -= 1e-6
            upper, lower = residual(plus), residual(minus)
            if upper is None or lower is None:
                return None
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                columns.append((upper-lower)/(2e-6))
        jacobian = np.asarray(columns).T
        if not np.isfinite(jacobian).all():
            return None
        improved = False
        for damping in (1e-10, 1e-8, 1e-6, 1e-4, .01):
            try:
                augmented = np.vstack((jacobian, math.sqrt(damping)*np.eye(len(active))))
                step = np.linalg.lstsq(augmented, np.r_[-error, np.zeros(len(active))], rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(step).all():
                continue
            step *= min(1., .35/max(1e-30, float(np.max(np.abs(step)))))
            for scale in (1., .5, .25, .125):
                trial = q.copy()
                trial[active] += scale*step
                next_error = residual(trial)
                with np.errstate(over='ignore', invalid='ignore'):
                    next_norm = math.inf if next_error is None else float(np.dot(next_error, next_error))
                if next_norm < previous_norm:
                    q, error, improved = trial, next_error, True
                    break
            if improved:
                break
        if not improved:
            break
    return q if np.max(np.abs(error)) < 1e-7 else None


def _trig_root_offset(a, b, c):
    """atan2(sqrt(a*a+b*b-c*c), -c) for scaled finite coefficients.

    acos(-c/hypot(a,b)) can round two distinct near-tangent roots together.
    Error-free square products and fsum retain the small discriminant instead.
    The exact equal-magnitude case also avoids underflow of the small square.
    """
    if c == 0:
        return math.pi/2
    large, small = max(abs(a), abs(b)), min(abs(a), abs(b))
    if large == abs(c):
        root = small
    else:
        terms = []
        for value, sign in ((a, 1.), (b, 1.), (c, -1.)):
            product = value*value
            split = 134217729.*value
            high = split-(split-value)
            low = value-high
            error = product-high*high
            error -= low*high
            error -= high*low
            error = low*low-error
            terms.extend((sign*product, sign*error))
        root = math.sqrt(max(0., math.fsum(terms)))
    return math.atan2(root, -c)


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
    # Multiplying a huge turn count by a rounded 2*pi loses the target phase.
    # The library sin/cos reduction retains it; only small offsets are added.
    origin_phase = math.atan2(math.sin(lower), math.cos(lower)) if abs(lower) > period else None
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
        angle = _trig_root_offset(a, b, c)
        for root in (center-angle, center+angle):
            if origin_phase is None:
                first = math.ceil((lower-root)/period)
                candidates = (root+turns*period for turns in (first-1, first, first+1))
            else:
                offset = math.remainder(root-origin_phase, period)
                candidates = (lower+(offset+turns*period) for turns in (-1, 0, 1))
            for value in candidates:
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

    Each 6x9 coefficient matrix multiplies
    [y*y*z*z, y*y*z, y*y, y*z*z, y*z, y, z*z, z, 1].
    Return angles for x=tan(theta/2), or (theta, 2*atan(y), 2*atan(z))
    when coupled output is requested. At most two 24-value eigensystems are
    evaluated; candidates still require the remaining equations and FK checks.
    Empty output does not establish infeasibility.
    """
    inputs = (A_x2, B_x, C)
    if any(np.iscomplexobj(value) for value in inputs):
        return []
    arrays = [np.asarray(value, dtype=float) for value in inputs]
    if any(value.shape != (6, 9) or not np.isfinite(value).all() for value in arrays):
        return []
    scale = max(float(np.max(np.abs(value))) for value in arrays)
    if scale == 0:
        return []
    a, b, c = (value/scale for value in arrays)

    def expand(value):
        matrix = np.zeros((12, 12))
        matrix[:6, :9], matrix[6:, 3:] = value, value
        return matrix

    def companion(leading, linear, constant, legacy=False):
        matrix = np.zeros((24, 24))
        matrix[:12, 12:] = np.eye(12)
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                if legacy:
                    inverse = np.linalg.inv(leading)
                    matrix[12:, :12] = -inverse.dot(constant)
                    matrix[12:, 12:] = -inverse.dot(linear)
                else:
                    matrix[12:, :] = -np.linalg.solve(leading, np.concatenate((constant, linear), axis=1))
        except np.linalg.LinAlgError:
            return None
        return matrix if np.isfinite(matrix).all() else None

    # Keep the established candidates: switching charts can improve one nearly
    # singular root while perturbing another. Their final FK checks decide which
    # representatives are acceptable, rather than a condition estimate alone.
    systems = []
    legacy = companion(*(expand(value) for value in arrays), legacy=True)
    if legacy is not None:
        systems.append((legacy, 0.))

    # x=(y+t)/(1-t*y) is an exact change of angular chart. A singular leading
    # coefficient can hide finite angles (including x=infinity at theta=pi).
    # Add at most one stable chart, retaining a 48-candidate bound across both
    # eigensystems. Structurally singular polynomials still need other methods.
    best = None
    for t in (0., 1., -1., .5, -.5, 2., -2.):
        leading = expand(a-t*b+t*t*c)
        try:
            condition = np.linalg.cond(leading)
        except np.linalg.LinAlgError:
            continue
        if math.isfinite(condition) and (best is None or condition < best[0]):
            best = (condition, t, leading)
        if t == 0. and condition < 1e8 and np.max(np.abs(a)) > 1e-8:
            break
    ordinary_stable = (best is not None and best[1] == 0. and best[0] < 1e8
                       and np.max(np.abs(a)) > 1e-8)
    if (not ordinary_stable or not systems) and best is not None and best[0] <= 1e14:
        _, t, leading = best
        shifted = companion(leading, expand(2*t*a+(1-t*t)*b-2*t*c),
                            expand(t*t*a+t*b+c))
        if shifted is not None:
            systems.append((shifted, t))

    solution = []

    def coupled_angle(vector, pairs):
        # The lifted vector is [y^3*z^2,y^3*z,y^3,...,z^2,z,1]. Ratios
        # numerator/denominator give y or z, not their reciprocals. Use a
        # projective pair with the largest magnitude so zeros and infinite
        # tangent values do not require division by a vanishing component.
        numerator, denominator = max(((vector[i], vector[j]) for i, j in pairs),
                                     key=lambda pair: max(abs(pair[0]), abs(pair[1])))
        scale = max(abs(numerator), abs(denominator))
        if scale == 0:
            return None
        anchor = numerator if abs(numerator) >= abs(denominator) else denominator
        phase = anchor.conjugate()/abs(anchor)
        numerator, denominator = (numerator/scale)*phase, (denominator/scale)*phase
        if abs(numerator.imag) > 1e-6 or abs(denominator.imag) > 1e-6:
            return None
        angle = 2*math.atan2(numerator.real, denominator.real)
        return math.atan2(math.sin(angle), math.cos(angle))

    for matrix, t in systems:
        try:
            if return_coupled_3_solution:
                eigen_values, eigen_vectors = np.linalg.eig(matrix)
            else:
                eigen_values = np.linalg.eigvals(matrix)
        except np.linalg.LinAlgError:
            continue
        for i, value in enumerate(eigen_values):
            if not np.isfinite(value) or abs(value.imag) > 1e-6:
                continue
            angle = 2*np.arctan(value.real)+2*math.atan(t)
            theta3 = math.atan2(math.sin(angle), math.cos(angle)) if t else float(angle)
            if not return_coupled_3_solution:
                solution.append(theta3)
                continue
            vector = eigen_vectors[:12, i]
            if not np.isfinite(vector).all():
                continue
            theta4 = coupled_angle(vector, [(j, j+3) for j in range(9)])
            theta5 = coupled_angle(vector, [(j, j+1) for j in range(12) if j % 3 != 2])
            if theta4 is None or theta5 is None:
                continue
            sy, cy = math.sin(theta4/2), math.cos(theta4/2)
            sz, cz = math.sin(theta5/2), math.cos(theta5/2)
            monomials = np.kron([sy*sy, sy*cy, cy*cy], [sz*sz, sz*cz, cz*cz])
            sx, cx = math.sin(theta3/2), math.cos(theta3/2)
            polynomial = a*sx*sx+b*sx*cx+c*cx*cx
            residual = polynomial@monomials
            bound = np.abs(polynomial)@np.abs(monomials)
            if np.max(np.abs(residual)/(1+bound)) > 1e-6:
                continue
            solution.append((theta3, theta4, theta5))
    return solution


# For linear solver type2
def solve_linear_type2_all(matrix):
    """Return up to four candidate y angles for two coupled unit circles.

    Rows multiply (sin(x), cos(x), sin(y), cos(y)). A rank-two nullspace
    intersects the equal-circle-norm quadratic in up to two projective lines,
    each giving an antipodal pair. Rank-three systems give at most one pair.
    Continuous families receive finite representatives. Numerical rank tests
    and empty output are not completeness or infeasibility certificates.
    Legacy two-root helpers below retain their existing APIs.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError('linear type2 requires a matrix with four columns')
    if not matrix.size or not np.all(np.isfinite(matrix)):
        return []
    scale = np.max(np.abs(matrix))
    if scale == 0:
        return []
    try:
        _, singular, vectors = np.linalg.svd(matrix/scale, full_matrices=len(matrix) < 4)
    except np.linalg.LinAlgError:
        return []
    tolerance = 64*np.finfo(float).eps*max(matrix.shape)*singular[0]
    if len(singular) < 2 or singular[1] <= tolerance:
        return []
    if len(singular) >= 3 and singular[2] > tolerance:
        if len(singular) >= 4 and singular[3] > 1e-8*singular[0]:
            return []
        directions = [vectors[-1]]
    else:
        basis = vectors[-2:].T
        quadratic = basis[:2].T@basis[:2]-basis[2:].T@basis[2:]
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
        except np.linalg.LinAlgError:
            return []
        low, high = eigenvalues
        if low < 0 < high:
            negative = math.sqrt(high)*eigenvectors[:, 0]
            positive = math.sqrt(-low)*eigenvectors[:, 1]
            directions = [basis@(negative+positive), basis@(negative-positive)]
        elif max(abs(low), abs(high)) <= 64*np.finfo(float).eps:
            return [0., math.pi]
        else:
            zero = int(np.argmin(np.abs(eigenvalues)))
            if abs(eigenvalues[zero]) > 64*np.finfo(float).eps:
                return []
            directions = [basis@eigenvectors[:, zero]]
    angles = []
    for direction in directions:
        if math.hypot(float(direction[2]), float(direction[3])) <= 1e-12:
            continue
        angle = math.atan2(float(direction[2]), float(direction[3]))
        other = angle+math.pi
        if other > math.pi:
            other -= 2*math.pi
        for value in (angle, other):
            if value not in angles:
                angles.append(value)
    return angles


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


def solve_linear_type2_candidates(matrix, rows_to_try=()):
    """Preserve the legacy pair and add up to four coupled-circle candidates.

    Approximate legacy models can have a full-rank matrix within the public
    FK tolerance. Their existing row-selection fallback must remain available.
    Exact equality is the only deduplication here, preserving nearby branches.
    """
    matrix = np.asarray(matrix, dtype=float)
    conic = solve_linear_type2_all(matrix)
    if not matrix.size or not np.all(np.isfinite(matrix)):
        return []
    scale = np.max(np.abs(matrix))
    if scale == 0:
        return []
    legacy_matrix = matrix/scale if scale < 1e-100 or scale > 1e100 else matrix
    angles = []
    for rows in rows_to_try:
        pair = try_solve_linear_type2_specific_rows(legacy_matrix, *rows)
        if pair is not None:
            angles.extend(value for value in pair if math.isfinite(value))
            break
    for value in conic:
        if value not in angles:
            angles.append(value)
    return angles
