"""Deterministic analytical IK for the exact Marvin M6 S CCS 696 V4.0 arms.

J5 (q[4], wrist_pitch) is fixed, in original URDF radians. The generated
modified-DH models in this directory specify these constants and transforms.
This kernel uses no numerical refinement or target perturbation. Singular
families have finite representatives, with no workspace completeness guarantee.
"""

import math
import numpy as np

PI = math.pi
LOWER = np.array([-3.1066860685499065, -2.0943951023931953,
                  -3.1066860685499065, -2.530727415391778,
                  -3.1066860685499065, -1.0471975511965976, -PI / 2])
UPPER = np.array([3.1066860685499065, 2.0943951023931953,
                  3.1066860685499065, 1.0471975511965976,
                  3.1066860685499065, 1.0471975511965976, PI / 2])
OFFSETS = np.array([0, -PI, 0, PI, 0, PI / 2, -PI / 2])
RX_PLUS = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
RX_MINUS = RX_PLUS.T
RX_PI = np.diag([1., -1, -1])
POST_R = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])
POST_P = np.array([0., -0.095, 0])
A = 0.018
UPPER_ARM = 0.287
FOREARM = 0.314


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0], [0, 0, 1]])


def _distal_rotation(t3, t4, t5, t6):
    return RX_PLUS @ _rz(t3) @ RX_PLUS @ _rz(t4) @ RX_MINUS @ _rz(t5) @ RX_MINUS @ _rz(t6)


def _raw_fk(t):
    shoulder = _rz(t[0]) @ RX_MINUS @ _rz(t[1]) @ RX_MINUS @ _rz(t[2])
    distal = _distal_rotation(*t[3:])
    p = np.array([A + A * math.cos(t[3]) + FOREARM * math.sin(t[3]),
                  0, UPPER_ARM + A * math.sin(t[3]) - FOREARM * math.cos(t[3])])
    return shoulder @ distal, shoulder @ p


def _root_pair(a, b, c, discriminant=None):
    """Solve a*cos(t)+b*sin(t)=c, retaining roundoff at repeated roots."""
    squared = a * a + b * b
    d = squared - c * c if discriminant is None else discriminant
    if d < -64 * np.finfo(float).eps * max(squared, c * c, 1.0):
        return ()
    root = math.sqrt(max(d, 0.0))
    center = math.atan2(b, a)
    return center + math.atan2(root, c), center + math.atan2(-root, c)


def _shoulder_candidates(rotation):
    # S Rx(pi) = Rz(t0) Ry(t1) Rz(-t2): use the entire rotation.
    x = rotation @ RX_PI
    sine = math.hypot(x[0, 2], x[1, 2])
    if sine > 1e-10:
        for sign in (1, -1):
            t1 = math.atan2(sign * sine, x[2, 2])
            t2 = -math.atan2(sign * x[2, 1], -sign * x[2, 0])
            # Recover the correlated base angle from all columns. Independent
            # Euler atan2 values lose their correlation near shoulder alignment.
            zero = RX_MINUS @ _rz(t1) @ RX_MINUS @ _rz(t2)
            delta = rotation @ zero.T
            t0 = math.atan2(delta[1, 0] - delta[0, 1], delta[0, 0] + delta[1, 1])
            yield t0, t1, t2
    elif x[2, 2] < 0:
        # t1=pi is URDF J2=0; intersect the family t0+t2=delta
        # with both joint limits instead of choosing an invalid t0=0.
        delta = math.atan2(-x[0, 1], x[1, 1])
        for turns in (-1, 0, 1):
            total = delta + turns * 2 * PI
            lo = max(LOWER[0], total - UPPER[2])
            hi = min(UPPER[0], total - LOWER[2])
            if lo <= hi:
                t0 = min(max(0., lo), hi)
                yield t0, PI, total - t0
    # t1=0 would give URDF J2=pi, outside this model's limits.


class Solver:
    """Exact standalone arm-base -> *_ee FK and constrained analytical IK."""

    def __init__(self, side='left'):
        if side not in ('left', 'right'):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        sign = 1 if side == 'left' else -1
        self.pre_r = np.array([[0., 0, sign], [1, 0, 0], [0, sign, 0]])
        self.pre_p = np.array([sign * .2215, .000029142, 1.121])

    def fk(self, q):
        if np.iscomplexobj(q):
            raise ValueError('q must contain seven real joint angles')
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('q must contain seven finite joint angles')
        r, p = _raw_fk(q + OFFSETS)
        target = np.eye(4)
        target[:3, :3] = self.pre_r @ r @ POST_R
        target[:3, 3] = self.pre_r @ (p + r @ POST_P) + self.pre_p
        return target

    def ik(self, target, q5):
        """Return unique, in-limit solutions with J5 exactly q5.

        Invalid SE(3) targets / J5 values raise ValueError. Empty output means
        no accepted representative was found for the specified fixed J5.
        Inverse-wrist-axis families use 33 deterministic bounded J7 samples;
        an empty result does not prove infeasibility.
        """
        if np.iscomplexobj(target):
            raise ValueError('target must be a real SE(3) transform')
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError('target must be a finite 4x4 SE(3) transform')
        rotation = target[:3, :3]
        if (np.max(np.abs(rotation)) > 1 + 1e-8 or
                np.max(np.abs(target[3] - [0, 0, 0, 1])) > 1e-9 or
                np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-8 or
                abs(np.linalg.det(rotation) - 1) > 1e-8):
            raise ValueError('target must be an SE(3) transform')
        try:
            if not np.isscalar(q5) or np.iscomplexobj(q5):
                raise ValueError
            q5 = float(q5)
            if not math.isfinite(q5) or not LOWER[4] <= q5 <= UPPER[4]:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError('q5 must be finite and inside the URDF J5 limits') from None
        # A generous physical reach bound rejects huge finite positions before
        # squaring them; the exact maximum wrist-to-shoulder reach is < 0.7 m.
        if np.max(np.abs(target[:3, 3] - self.pre_p)) > 1.0:
            return []
        r = self.pre_r.T @ rotation @ POST_R.T
        p = self.pre_r.T @ (target[:3, 3] - self.pre_p) - r @ POST_P
        v = -r.T @ p
        c = float(v @ v) - 2 * A * A - UPPER_ARM**2 - FOREARM**2
        solutions = []
        for t3 in _root_pair(2 * A * A - 2 * UPPER_ARM * FOREARM,
                             2 * A * (FOREARM + UPPER_ARM), c):
            h = A * math.cos(t3) + A + UPPER_ARM * math.sin(t3)
            k = A * math.sin(t3) - UPPER_ARM * math.cos(t3) + FOREARM
            hsin = h * math.sin(q5)
            # XY length avoids subtracting similar squared lengths near the axis.
            discriminant = v[0]**2 + v[1]**2 - hsin**2
            for t5 in _root_pair(k, h * math.cos(q5), v[2], discriminant):
                axial = h * math.cos(q5) * math.cos(t5) - k * math.sin(t5)
                if math.hypot(v[0], v[1]) <= 1e-10:
                    wrists = np.linspace(-PI, 0, 33)
                else:
                    wrists = [math.atan2(-v[0] * hsin + v[1] * axial,
                                         -v[0] * axial - v[1] * hsin)]
                    # Admit an exact J7 endpoint only when its inverse-position
                    # equation meets the position tolerance. This recovers an
                    # ill-conditioned boundary without a heuristic axis radius.
                    # The full original-target FK gate remains mandatory.
                    for boundary in (-PI, 0.):
                        expected_x = -hsin * math.sin(boundary) - axial * math.cos(boundary)
                        expected_y = axial * math.sin(boundary) - hsin * math.cos(boundary)
                        if math.hypot(expected_x - v[0], expected_y - v[1]) <= 1e-7:
                            wrists.append(boundary)
                for t6 in wrists:
                    distal = _distal_rotation(t3, q5, t5, t6)
                    for t0, t1, t2 in _shoulder_candidates(r @ distal.T):
                        raw = np.array([t0, t1, t2, t3, q5, t5, t6])
                        q = (raw - OFFSETS + PI) % (2 * PI) - PI
                        q[4] = q5
                        if np.any(q < LOWER - 1e-9) or np.any(q > UPPER + 1e-9):
                            continue
                        q = np.clip(q, LOWER, UPPER)
                        actual = self.fk(q)
                        delta = actual[:3, :3] @ rotation.T
                        sine = .5 * np.linalg.norm([delta[2, 1] - delta[1, 2],
                                                   delta[0, 2] - delta[2, 0],
                                                   delta[1, 0] - delta[0, 1]])
                        angle = math.atan2(sine, (np.trace(delta) - 1) * .5)
                        if np.linalg.norm(actual[:3, 3] - target[:3, 3]) > 1e-7 or angle > 1e-6:
                            continue
                        if not any(np.max(np.abs(q - existing)) < 1e-7 for existing in solutions):
                            solutions.append(q)
        return solutions


def solve_ik(target, q5, side='left'):
    return Solver(side).ik(target, q5)
