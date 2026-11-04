#pragma once

// Exact Marvin M6 S CCS 696 V4.0 arm-base -> *_ee geometry. All lengths are
// metres and original URDF joint angles are radians. Pure analytical kernel:
// no random perturbation, iterative refinement, or change to fixed URDF J5.
// Singular continua have finite representatives, not complete enumeration.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace marvin_m6 {
enum class Side { Left, Right };
using JointVector = std::array<double, 7>;
inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double a = .018, upper_arm = .287, forearm = .314;
inline constexpr JointVector lower{{-3.1066860685499065, -2.0943951023931953,
    -3.1066860685499065, -2.530727415391778, -3.1066860685499065,
    -1.0471975511965976, -pi / 2}};
inline constexpr JointVector upper{{3.1066860685499065, 2.0943951023931953,
    3.1066860685499065, 1.0471975511965976, 3.1066860685499065,
    1.0471975511965976, pi / 2}};
inline constexpr JointVector offsets{{0, -pi, 0, pi, 0, pi / 2, -pi / 2}};

namespace detail {
inline Eigen::Matrix3d rxPlus() {
    Eigen::Matrix3d r; r << 1, 0, 0, 0, 0, -1, 0, 1, 0; return r;
}
inline Eigen::Matrix3d postR() {
    Eigen::Matrix3d r; r << 0, -1, 0, 0, 0, -1, 1, 0, 0; return r;
}
inline Eigen::Matrix3d preR(Side side) {
    const double sign = side == Side::Left ? 1 : -1;
    Eigen::Matrix3d r; r << 0, 0, sign, 1, 0, 0, 0, sign, 0; return r;
}
inline Eigen::Vector3d preP(Side side) {
    return {side == Side::Left ? .2215 : -.2215, .000029142, 1.121};
}
inline Eigen::Matrix3d rz(double t) {
    const double c = std::cos(t), s = std::sin(t);
    Eigen::Matrix3d r; r << c, -s, 0, s, c, 0, 0, 0, 1; return r;
}
inline Eigen::Matrix3d distal(double t3, double t4, double t5, double t6) {
    const Eigen::Matrix3d plus = rxPlus(), minus = plus.transpose();
    return plus * rz(t3) * plus * rz(t4) * minus * rz(t5) * minus * rz(t6);
}
inline double wrap(double t) {
    t = std::fmod(t + pi, 2 * pi);
    if (t < 0) t += 2 * pi;
    return t - pi;
}
inline std::vector<double> roots(double av, double b, double c, double d) {
    const double squared = av * av + b * b;
    if (d < -64 * std::numeric_limits<double>::epsilon() *
            std::max({squared, c * c, 1.0})) return {};
    const double root = std::sqrt(std::max(d, 0.0)), center = std::atan2(b, av);
    return {center + std::atan2(root, c), center + std::atan2(-root, c)};
}
inline std::vector<std::array<double, 3>> shoulders(const Eigen::Matrix3d& s) {
    // S Rx(pi) = Rz(t0) Ry(t1) Rz(-t2), using all rotation columns.
    const Eigen::Matrix3d x = s * Eigen::Vector3d(1, -1, -1).asDiagonal();
    const double sine = std::hypot(x(0, 2), x(1, 2));
    std::vector<std::array<double, 3>> result;
    if (sine > 1e-10) {
        for (int sign : {1, -1}) {
            const double t1 = std::atan2(sign * sine, x(2, 2));
            const double t2 = -std::atan2(sign * x(2, 1), -sign * x(2, 0));
            // Full-rotation projection preserves correlated Euler angles near
            // alignment, where their independent atan2 values lose precision.
            const Eigen::Matrix3d minus = rxPlus().transpose();
            const Eigen::Matrix3d zero = minus * rz(t1) * minus * rz(t2);
            const Eigen::Matrix3d delta = s * zero.transpose();
            const double t0 = std::atan2(delta(1, 0) - delta(0, 1),
                                         delta(0, 0) + delta(1, 1));
            result.push_back({{t0, t1, t2}});
        }
    } else if (x(2, 2) < 0) {
        // t1=pi gives URDF J2=0 and t0+t2=delta. Intersect limits.
        const double delta = std::atan2(-x(0, 1), x(1, 1));
        for (int turns : {-1, 0, 1}) {
            const double total = delta + turns * 2 * pi;
            const double lo = std::max(lower[0], total - upper[2]);
            const double hi = std::min(upper[0], total - lower[2]);
            if (lo <= hi) {
                const double t0 = std::clamp(0.0, lo, hi);
                result.push_back({{t0, pi, total - t0}});
            }
        }
    }
    // t1=0 would give URDF J2=pi, outside this model's limits.
    return result;
}
inline void validateSide(Side side) {
    if (side != Side::Left && side != Side::Right)
        throw std::invalid_argument("side must be Left or Right");
}
}  // namespace detail

inline Eigen::Matrix4d computeFK(const JointVector& q, Side side = Side::Left) {
    detail::validateSide(side);
    JointVector t;
    for (int i = 0; i < 7; ++i) {
        if (!std::isfinite(q[i])) throw std::invalid_argument("q must be finite");
        t[i] = q[i] + offsets[i];
    }
    const Eigen::Matrix3d minus = detail::rxPlus().transpose();
    const Eigen::Matrix3d shoulder = detail::rz(t[0]) * minus *
        detail::rz(t[1]) * minus * detail::rz(t[2]);
    const Eigen::Matrix3d r = shoulder * detail::distal(t[3], t[4], t[5], t[6]);
    const Eigen::Vector3d distal_p(a + a * std::cos(t[3]) + forearm * std::sin(t[3]),
        0, upper_arm + a * std::sin(t[3]) - forearm * std::cos(t[3]));
    const Eigen::Matrix3d pre = detail::preR(side);
    Eigen::Matrix4d target = Eigen::Matrix4d::Identity();
    target.block<3, 3>(0, 0) = pre * r * detail::postR();
    target.block<3, 1>(0, 3) = pre * (shoulder * distal_p + r *
        Eigen::Vector3d(0, -.095, 0)) + detail::preP(side);
    return target;
}

// Invalid SE(3) / fixed-J5 values throw invalid_argument. Empty output means
// no accepted solution representative was found for this fixed J5. Wrist-axis
// continua use 33 deterministic bounded J7 samples, without completeness claims.
inline std::vector<JointVector> solveIK(const Eigen::Matrix4d& target, double q5,
                                      Side side = Side::Left) {
    detail::validateSide(side);
    const Eigen::Matrix3d rotation = target.block<3, 3>(0, 0);
    if (!target.allFinite() || rotation.cwiseAbs().maxCoeff() > 1 + 1e-8 ||
        (target.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).cwiseAbs().maxCoeff() > 1e-9 ||
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-8 ||
        std::abs(rotation.determinant() - 1) > 1e-8)
        throw std::invalid_argument("target must be a finite SE(3) transform");
    if (!std::isfinite(q5) || q5 < lower[4] || q5 > upper[4])
        throw std::invalid_argument("q5 must be finite and inside URDF J5 limits");
    // Conservative reach check before any squared-distance arithmetic.
    if ((target.block<3, 1>(0, 3) - detail::preP(side)).cwiseAbs().maxCoeff() > 1.0)
        return {};
    const Eigen::Matrix3d pre = detail::preR(side);
    const Eigen::Matrix3d r = pre.transpose() * rotation * detail::postR().transpose();
    const Eigen::Vector3d p = pre.transpose() * (target.block<3, 1>(0, 3) -
        detail::preP(side)) - r * Eigen::Vector3d(0, -.095, 0);
    const Eigen::Vector3d v = -r.transpose() * p;
    const double c = v.squaredNorm() - 2 * a * a - upper_arm * upper_arm - forearm * forearm;
    const double av = 2 * a * a - 2 * upper_arm * forearm, b = 2 * a * (forearm + upper_arm);
    std::vector<JointVector> solutions;
    for (double t3 : detail::roots(av, b, c, av * av + b * b - c * c)) {
        const double h = a * std::cos(t3) + a + upper_arm * std::sin(t3);
        const double k = a * std::sin(t3) - upper_arm * std::cos(t3) + forearm;
        const double hsin = h * std::sin(q5);
        // XY length retains precision at inverse-wrist-axis alignment.
        const double d = v.x() * v.x() + v.y() * v.y() - hsin * hsin;
        for (double t5 : detail::roots(k, h * std::cos(q5), v.z(), d)) {
            const double axial = h * std::cos(q5) * std::cos(t5) - k * std::sin(t5);
            std::vector<double> wrists;
            if (std::hypot(v.x(), v.y()) <= 1e-10)
                for (int i = 0; i <= 32; ++i) wrists.push_back(-pi + pi * i / 32);
            else {
                wrists.push_back(std::atan2(-v.x() * hsin + v.y() * axial,
                                           -v.x() * axial - v.y() * hsin));
                // Admit exact J7 endpoints through their inverse-position
                // residual, avoiding a heuristic definition of near alignment.
                // The complete original-target FK gate remains mandatory.
                for (double boundary : {-pi, 0.0}) {
                    const double expected_x = -hsin * std::sin(boundary) - axial * std::cos(boundary);
                    const double expected_y = axial * std::sin(boundary) - hsin * std::cos(boundary);
                    if (std::hypot(expected_x - v.x(), expected_y - v.y()) <= 1e-7)
                        wrists.push_back(boundary);
                }
            }
            for (double t6 : wrists) {
                const Eigen::Matrix3d distal = detail::distal(t3, q5, t5, t6);
                for (const auto& shoulder : detail::shoulders(r * distal.transpose())) {
                    JointVector raw{{shoulder[0], shoulder[1], shoulder[2], t3, q5, t5, t6}}, q;
                    bool valid = true;
                    for (int i = 0; i < 7; ++i) {
                        q[i] = i == 4 ? q5 : detail::wrap(raw[i] - offsets[i]);
                        if (q[i] < lower[i] - 1e-9 || q[i] > upper[i] + 1e-9) valid = false;
                        q[i] = std::clamp(q[i], lower[i], upper[i]);
                    }
                    if (!valid) continue;
                    const Eigen::Matrix4d actual = computeFK(q, side);
                    const Eigen::Matrix3d delta = actual.block<3, 3>(0, 0) * rotation.transpose();
                    const double sine = .5 * Eigen::Vector3d(delta(2, 1) - delta(1, 2),
                        delta(0, 2) - delta(2, 0), delta(1, 0) - delta(0, 1)).norm();
                    if ((actual.block<3, 1>(0, 3) - target.block<3, 1>(0, 3)).norm() > 1e-7 ||
                        std::atan2(sine, (delta.trace() - 1) * .5) > 1e-6) continue;
                    bool duplicate = false;
                    for (const auto& existing : solutions) {
                        double maximum = 0;
                        for (int i = 0; i < 7; ++i)
                            maximum = std::max(maximum, std::abs(q[i] - existing[i]));
                        if (maximum < 1e-7) { duplicate = true; break; }
                    }
                    if (!duplicate) solutions.push_back(q);
                }
            }
        }
    }
    return solutions;
}
}  // namespace marvin_m6
