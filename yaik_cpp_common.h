#pragma once

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigen>
#include <unsupported/Eigen/Polynomials>

namespace yaik_cpp {

// Indicate the value is invalid
constexpr double invalid_value = std::numeric_limits<double>::max();

inline double addRevoluteOffset(double value, double offset) {
  if (offset == 0 || (std::abs(value) <= 1048576. && std::abs(offset) <= 1048576.))
    return value+offset;
  const double phase=std::atan2(std::sin(value), std::cos(value));
  const double shift=std::abs(offset) > M_PI ? std::atan2(std::sin(offset), std::cos(offset)) : offset;
  return phase+shift;
}

template <int N>
bool snapNearJointLimits(const std::array<double, N>& source, std::array<double, N>& result,
                        const std::array<double, N>& lower, const std::array<double, N>& upper,
                        const std::array<bool, N>& periodic, const std::array<bool, N>& fixed) {
  result = source;
  bool changed = false;
  for (int i = 0; i < N; ++i) {
    double value = result[i];
    if (!std::isfinite(value)) return false;
    if (fixed[i]) { if (value < lower[i] || value > upper[i]) return false; continue; }
    if (periodic[i] && (value < lower[i] || value > upper[i]))
      value += std::ceil((lower[i]-1e-6-value)/(2*M_PI))*2*M_PI;
    if (value < lower[i]) {
      if (lower[i]-value > 1e-6) return false;
      value = lower[i]; changed = true;
    } else if (value > upper[i]) {
      if (value-upper[i] > 1e-6) return false;
      value = upper[i]; changed = true;
    }
    result[i] = value;
  }
  return changed;
}

// Compensated square products keep distinct near-tangent roots from collapsing
// through a rounded acos argument. Inputs have already been scaled to [-1,1].
inline double trigRootOffset(double a, double b, double c) {
  if (c == 0) return 1.57079632679489661923;
  const double large = std::max(std::abs(a), std::abs(b));
  const double small = std::min(std::abs(a), std::abs(b));
  double root;
  if (large == std::abs(c)) {
    root = small;  // Also avoid underflow in small*small.
  } else {
    std::array<double, 6> partials{};
    std::size_t count = 0;
    for (const auto& term : std::array<std::array<double, 2>, 3>{{{{a, 1}}, {{b, 1}}, {{c, -1}}}}) {
      const double square = term[0]*term[0];
      for (double value : {term[1]*square, term[1]*std::fma(term[0], term[0], -square)}) {
        std::size_t kept = 0;
        for (std::size_t i = 0; i < count; ++i) {
          double previous = partials[i];
          if (std::abs(value) < std::abs(previous)) std::swap(value, previous);
          const double high = value+previous, low = previous-(high-value);
          if (low != 0) partials[kept++] = low;
          value = high;
        }
        partials[kept++] = value;
        count = kept;
      }
    }
    double discriminant = 0;
    while (count) discriminant += partials[--count];
    root = std::sqrt(std::max(0.0, discriminant));
  }
  return std::atan2(root, -c);
}

// Analytic cuts of a one-dimensional angular family, followed by one midpoint
// per open cell. Preserve distinct nearby cuts; feasible intervals may be tiny.
inline std::vector<double> trigFamilyRepresentatives(
    const std::vector<std::array<double, 3>>& coefficients,
    double lower, double upper, const std::vector<double>& seeds = {}) {
  const double period = 2*M_PI;
  if (!(std::isfinite(lower) && std::isfinite(upper) && lower <= upper && upper-lower <= period))
    throw std::invalid_argument("family bounds must span at most one finite revolution");
  const double epsilon = 64*std::numeric_limits<double>::epsilon();
  const double margin = epsilon*(1+std::max(std::abs(lower), std::abs(upper)));
  const bool shifted = std::abs(lower) > period;
  const double origin_phase = shifted ? std::atan2(std::sin(lower), std::cos(lower)) : 0;
  std::vector<double> points{lower, upper};
  for (const auto& row : coefficients) {
    double a = row[0], b = row[1], c = row[2];
    if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c))) continue;
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
    if (!std::isfinite(scale) || scale == 0) continue;
    a /= scale; b /= scale; c /= scale;
    const double radius = std::hypot(a, b);
    if (radius == 0 || std::abs(c) > radius+epsilon) continue;
    const double center = std::atan2(b, a);
    const double angle = trigRootOffset(a, b, c);
    for (double root : {center-angle, center+angle}) {
      const double first = shifted ? 0 : std::ceil((lower-root)/period);
      const double relative = shifted ? std::remainder(root-origin_phase, period) : 0;
      for (int offset : {-1, 0, 1}) {
        const double value = shifted ? lower+(relative+offset*period) : root+(first+offset)*period;
        if (value >= lower-margin && value <= upper+margin)
          points.push_back(std::max(lower, std::min(upper, value)));
      }
    }
  }
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  auto result = points;
  for (std::size_t i = 1; i < points.size(); ++i)
    result.push_back(points[i-1]+.5*(points[i]-points[i-1]));
  for (double value : seeds)
    if (std::isfinite(value) && value >= lower && value <= upper) result.push_back(value);
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

// Safe operations
template <typename T>
T safe_sqrt(T a) {
  return a >= 0 ? std::sqrt(a) : 0.0;
}

template <typename T>
T safe_asin(T a) {
  if (a > 1)
    return 0.5 * M_PI;
  else if (a < -1)
    return -0.5 * M_PI;
  else
    return std::asin(a);
}

template <typename T>
T safe_acos(T a) {
  if (a > 1)
    return 0.0;
  else if (a < -1)
    return M_PI;
  else
    return std::acos(a);
}

// Periodic comparison is opt-in; select limit-compatible representatives first.
template <int N>
void removeDuplicateImpl(std::vector<std::array<double, N>>& ik_output,
                      double tolerance,
                      const std::array<bool, N>& revolute_joints,
                      const std::vector<int>* classes = nullptr) {
  if (classes && classes->size() != ik_output.size())
    throw std::invalid_argument("duplicate classes must match the candidate count");
  std::vector<int> kept_classes = classes ? *classes : std::vector<int>{};
  // Do nothing if size less than 1
  if (ik_output.size() <= 1) return;

  // Functor to compare two vectors
  auto is_duplicate = [tolerance, &revolute_joints](const std::array<double, N>& a,
                                  const std::array<double, N>& b) -> bool {
    for (std::size_t j = 0; j < a.size(); j++) {
      if (revolute_joints[j]) {
        if (!std::isfinite(a[j]) || !std::isfinite(b[j])) return false;
        if (std::signbit(a[j]) != std::signbit(b[j]) &&
            std::abs(a[j]) > std::numeric_limits<double>::max() - std::abs(b[j])) return false;
      }
      double difference = std::abs(a[j] - b[j]);
      if (revolute_joints[j]) {
        if (!std::isfinite(difference)) return false;
        if (difference > tolerance) {
          // Generated angles are normalized; reduce unusual inputs individually.
          if (difference > 2 * M_PI) {
            const double angle_a = std::atan2(std::sin(a[j]), std::cos(a[j]));
            const double angle_b = std::atan2(std::sin(b[j]), std::cos(b[j]));
            difference = std::abs(angle_a - angle_b);
          }
          difference = std::min(difference, std::abs(difference - 2 * M_PI));
        }
      }
      if (difference > tolerance) return false;
    }
    return true;
  };

  // There is no duplication in [0, no_duplicate_until]
  std::size_t no_duplicate_until = 0;
  std::size_t next_to_try = 1;

  // Processing loop
  while (next_to_try < ik_output.size()) {
    // Check duplication in [0, no_duplicate_until]
    bool contains_duplicate = false;
    assert(no_duplicate_until + 1 <= next_to_try);
    for (std::size_t i = 0; i <= no_duplicate_until; i++) {
      if ((!classes || kept_classes[i] == (*classes)[next_to_try]) &&
          is_duplicate(ik_output[i], ik_output[next_to_try])) {
        contains_duplicate = true;
        break;
      }
    }

    // Update index
    if (contains_duplicate) {
      next_to_try += 1;
    } else {
      // Copy the result to no_duplicate_until + 1
      std::size_t copied_to = no_duplicate_until + 1;
      assert(copied_to < ik_output.size());
      if (copied_to != next_to_try) {
        ik_output[copied_to] = ik_output[next_to_try];
      }
      if (classes) {
        kept_classes[copied_to] = (*classes)[next_to_try];
      }

      // Update the index
      no_duplicate_until = copied_to;
      next_to_try += 1;
    }
  }

  // [0, no_duplicate_until] is the range
  assert(no_duplicate_until < ik_output.size());
  ik_output.resize(no_duplicate_until + 1);
}

template <int N>
void removeDuplicate(std::vector<std::array<double, N>>& ik_output,
                      double tolerance, const std::array<bool, N>& revolute_joints) {
  removeDuplicateImpl<N>(ik_output, tolerance, revolute_joints);
}

template <int N>
void removeDuplicateWithinClasses(std::vector<std::array<double, N>>& ik_output,
                                 double tolerance, const std::vector<int>& classes) {
  removeDuplicateImpl<N>(ik_output, tolerance, {}, &classes);
}

// Distinguish candidates that admit a bounded periodic representation before
// deduplication. A nearby out-of-limit root must not erase a legal witness.
template <int N>
std::vector<int> jointLimitClasses(const std::vector<std::array<double, N>>& candidates,
                                 const std::array<double, N>& lower,
                                 const std::array<double, N>& upper,
                                 const std::array<bool, N>& periodic) {
  std::vector<int> result;
  for (const auto& candidate : candidates) {
    bool bounded = true;
    for (int joint = 0; joint < N; ++joint) {
      const double value = candidate[joint];
      if (!std::isfinite(value)) { bounded = false; break; }
      if (periodic[joint]) {
        const double first = std::ceil((lower[joint]-1e-9-value)/(2*M_PI));
        const double last = std::floor((upper[joint]+1e-9-value)/(2*M_PI));
        if (first > last) { bounded = false; break; }
      } else if (value < lower[joint]-1e-9 || value > upper[joint]+1e-9) {
        bounded = false; break;
      }
    }
    result.push_back(bounded ? 1 : 0);
  }
  return result;
}

// A coarse FK-accepted candidate must not erase a nearby strict-pose witness.
// Keep both feasibility and pose-quality classes; the legacy acceptance
// thresholds and candidate ordering remain unchanged.
template <int N, typename FK>
std::vector<int> ikCandidateClasses(const std::vector<std::array<double, N>>& candidates,
                                  const std::array<double, N>& lower,
                                  const std::array<double, N>& upper,
                                  const std::array<bool, N>& periodic,
                                  FK&& fk, const Eigen::Matrix4d& target,
                                  double position_tolerance = 1e-7,
                                  double rotation_tolerance = 1e-6) {
  auto classes = jointLimitClasses<N>(candidates, lower, upper, periodic);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    classes[i] *= 2;
    const Eigen::Matrix4d actual = fk(candidates[i]);
    if (!actual.allFinite() || !target.allFinite() ||
        actual.template block<3, 3>(0, 0).cwiseAbs().maxCoeff() > 2 ||
        target.template block<3, 3>(0, 0).cwiseAbs().maxCoeff() > 2) continue;
    if ((actual.template block<3, 1>(0, 3)-target.template block<3, 1>(0, 3)).norm() > position_tolerance)
      continue;
    const Eigen::Matrix3d delta = actual.template block<3, 3>(0, 0)*target.template block<3, 3>(0, 0).transpose();
    const Eigen::Vector3d skew(delta(2, 1)-delta(1, 2), delta(0, 2)-delta(2, 0), delta(1, 0)-delta(0, 1));
    const double angle = std::atan2(.5*skew.norm(), .5*(delta.trace()-1));
    if (angle <= rotation_tolerance) ++classes[i];
  }
  return classes;
}

template <int N>
void removeDuplicate(std::vector<std::array<double, N>>& ik_output,
                      double tolerance) {
  removeDuplicate<N>(ik_output, tolerance, {});
}

// Fixe-size array to hold ik solution
template <int N>
using IntermediateSolution = std::array<double, N>;

// Construct an ik solution with invalid valid
template <int N>
IntermediateSolution<N> make_intermediate_solution() {
  IntermediateSolution<N> solution;
  std::fill(solution.begin(), solution.end(), invalid_value);
  return solution;
}

// A queue with fixed capacity
template <typename T, int Capacity>
class FixedBufferQueue {
 private:
  std::array<T, Capacity> queue_element_;
  int top_index_;

 public:
  explicit FixedBufferQueue() : top_index_(0) {
    static_assert(Capacity >= 1, "Zero capacity not allowed");
  }
  void reset() { top_index_ = 0; }

  /// Push an element into the queue.
  /// If push succeed, then return the appended idx
  /// Else, return -1.
  int push(T element) {
    if (top_index_ >= queue_element_.size()) return -1;
    int appended_idx = top_index_;
    queue_element_[top_index_] = std::move(element);
    top_index_ += 1;
    return appended_idx;
  }

  /// Element access by index
  const T& operator[](int index) const { return queue_element_[index]; }
  T& operator[](int index) { return queue_element_[index]; }
  int size() const { return top_index_; }
};

// The queue to hold the solution
template <int Nq, int Capacity = 16>
struct SolutionQueue {
  FixedBufferQueue<IntermediateSolution<Nq>, Capacity> solutions;
  std::array<bool, Capacity> solutions_validity;

  explicit SolutionQueue() = default;
  void reset() {
    solutions.reset();
    std::fill(solutions_validity.begin(), solutions_validity.end(), false);
  }

  /// Push an solution into the queue
  int push_solution(IntermediateSolution<Nq> solution) {
    int appended_idx = solutions.push(std::move(solution));
    if (appended_idx >= 0) solutions_validity[appended_idx] = true;
    return appended_idx;
  }

  /// Interface for obtain solution and check its validity
  // clang-format off
  const IntermediateSolution<Nq>& get_solution(int solution_idx) const { return solutions[solution_idx]; }
  IntermediateSolution<Nq>& get_solution(int solution_idx) { return solutions[solution_idx]; }
  bool is_solution_valid(int solution_idx) const { return solutions_validity[solution_idx]; }
  int size() const { return solutions.size(); }
  // clang-format on
};

// The index workspace for different nodes
template <int max_n_solution = 32>
struct NodeIndexWorkspace {
  using IndexQueue = FixedBufferQueue<int, max_n_solution>;
  std::vector<IndexQueue> node_input_indices_vector;
  std::vector<bool> node_input_validity_vector;

  // Constructors and mutators
  NodeIndexWorkspace() = default;
  void reset(int n_node) {
    node_input_indices_vector.resize(n_node);
    node_input_validity_vector.resize(n_node);
    for (auto i = 0; i < n_node; i++) {
      node_input_indices_vector[i].reset();
      node_input_validity_vector[i] = false;
    }

    // Set the root node validity to true
    node_input_validity_vector[0] = true;
  }

  /// Add a index to a node
  bool append_index_to_node(int node_idx, int index_to_append) {
    auto appended_idx =
        node_input_indices_vector.at(node_idx).push(index_to_append);
    if (appended_idx >= 0) node_input_validity_vector[node_idx] = true;
    return appended_idx >= 0;
  }

  /// Access method
  // clang-format off
  const IndexQueue& get_input_indices_for_node(int node_idx) const { return node_input_indices_vector.at(node_idx); }
  bool is_input_indices_valid_for_node(int node_idx) const {return node_input_validity_vector.at(node_idx); }
  int num_nodes() const { return node_input_indices_vector.size(); }
  // clang-format on
};

/// A struct to hold all the buffers
template <int intermediate_solution_size, int max_n_solutions, int robot_nq>
struct RawIkWorkspaceTemplate {
  SolutionQueue<intermediate_solution_size, max_n_solutions> solution_queue;
  NodeIndexWorkspace<max_n_solutions> node_index_workspace;
  std::vector<std::array<double, robot_nq>> raw_ik_out;
};

/// The polynomial solver
struct PolynomialRealRoot {
  bool is_valid{false};
  double value{0.0};
};

template <int poly_order>
std::array<PolynomialRealRoot, poly_order> computePolynomialRealRoots(
    const std::array<double, poly_order + 1>& coefficient_high_to_low_order) {
  std::array<PolynomialRealRoot, poly_order> poly_roots{};
  double coefficient_scale = 0;
  for (double coefficient : coefficient_high_to_low_order) {
    if (!std::isfinite(coefficient)) return poly_roots;
    coefficient_scale = std::max(coefficient_scale, std::abs(coefficient));
  }
  int first_nonzero = 0;
  while (first_nonzero <= poly_order && coefficient_high_to_low_order[first_nonzero] == 0)
    ++first_nonzero;
  const int degree = poly_order - first_nonzero;
  if (degree <= 0) return poly_roots;

  // Convert to eigen, note the reversed order
  Eigen::VectorXd eigen_coefficient(degree + 1);
  for (auto i = 0; i <= degree; i++) {
    eigen_coefficient[i] = coefficient_high_to_low_order[poly_order - i] / coefficient_scale;
  }
  // Scaling can underflow a nonzero leading coefficient.
  if (eigen_coefficient[degree] == 0) return poly_roots;

  // Call eigen solver
  Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
  solver.compute(eigen_coefficient);

  // The result
  const auto& roots = solver.roots();
  for (auto i = 0; i < roots.size(); i++) {
    const std::complex<double>& root_i = roots[i];
    if (std::abs(root_i.imag()) <= 1e-6 * (1 + std::abs(root_i.real()))) {
      const double real_root_i = root_i.real();
      double residual = eigen_coefficient[degree];
      double scale = std::abs(residual);
      for (int j = degree - 1; j >= 0; --j) {
        residual = residual * real_root_i + eigen_coefficient[j];
        scale = scale * std::abs(real_root_i) + std::abs(eigen_coefficient[j]);
      }
      if (!std::isfinite(residual) || std::abs(residual) > 1e-6 * std::max(1.0, scale)) continue;
      PolynomialRealRoot root_i_return;
      root_i_return.is_valid = true;
      root_i_return.value = real_root_i;
      poly_roots[i] = root_i_return;
    } else {
      PolynomialRealRoot invalid_root;
      invalid_root.is_valid = false;
      poly_roots[i] = invalid_root;
    }
  }

  // Should be ok
  return poly_roots;
}

// For linear solver type2
namespace linear_solver {

template <typename T>
bool trySolveLinearType2(const Eigen::Ref<const Eigen::Matrix<T, 3, 4>>& A,
                         T& solution_0, T& solution_1, bool allow_family = true) {
  if (!A.allFinite()) return false;
  const T a = A(0, 0);
  const T b = A(0, 1);
  const T c = A(1, 0);
  const T d = A(1, 1);
  const T ad_minus_bc = a * d - b * c;
  constexpr T zero_tolerance = 1e-10;
  if (std::abs(ad_minus_bc) < zero_tolerance) return false;

  // Compute the inverse to top 2x2 matrix
  Eigen::Matrix<T, 2, 2> A_top2x2_inv;
  A_top2x2_inv(0, 0) = d / ad_minus_bc;
  A_top2x2_inv(0, 1) = -b / ad_minus_bc;
  A_top2x2_inv(1, 0) = -c / ad_minus_bc;
  A_top2x2_inv(1, 1) = a / ad_minus_bc;

  // [sin(x), cos(x)].T = B * [sin(y), cos(y)].T
  const Eigen::Matrix<T, 2, 2> B = -A_top2x2_inv * A.template block<2, 2>(0, 2);
  const T e = A(2, 2) + A(2, 0) * B(0, 0) + A(2, 1) * B(1, 0);
  const T f = A(2, 3) + A(2, 0) * B(0, 1) + A(2, 1) * B(1, 1);
  if (std::abs(e) < zero_tolerance && std::abs(f) < zero_tolerance) {
    if (allow_family && (B.transpose() * B - Eigen::Matrix<T, 2, 2>::Identity())
                            .cwiseAbs().maxCoeff() <= T(1e-10)) {
      solution_0 = T(0);
      solution_1 = std::acos(T(-1));
      return true;
    }
    return false;
  }

  // The solution should be OK
  constexpr T local_pi = T(3.141592653589793238462643383279502884L);
  solution_0 = std::atan2(-f, e);
  solution_1 = solution_0 + local_pi;
  if (solution_1 > local_pi) solution_1 -= 2 * local_pi;
  return true;
}

template <int n_rows>
bool trySolveLinearType2SpecificRows(
    const Eigen::Ref<const Eigen::Matrix<double, n_rows, 4>>& A, int row_0, int row_1,
    int row_2, double& solution_0, double& solution_1) {
  if (A.allFinite()) {
    const double scale = A.cwiseAbs().maxCoeff();
    if (scale > 0) {
      const Eigen::Matrix<double, n_rows, 4> normalized = A/scale;
      Eigen::JacobiSVD<Eigen::Matrix<double, n_rows, 4>> svd(normalized, Eigen::ComputeFullV);
      const auto& singular = svd.singularValues();
      if (singular.size() >= 3 && singular[2] > 1e-10*singular[0] &&
          (singular.size() == 3 || singular[3] <= 1e-8*singular[0])) {
        const Eigen::Vector2d direction = svd.matrixV().col(3).template tail<2>();
        if (direction.norm() > 1e-10) {
          solution_0 = std::atan2(direction[0], direction[1]);
          solution_1 = solution_0+M_PI;
          if (solution_1 > M_PI) solution_1 -= 2*M_PI;
          return true;
        }
      }
    }
  }
  Eigen::Matrix<double, 3, 4> A_for_given_row;
  A_for_given_row.row(0) = A.row(row_0);
  A_for_given_row.row(1) = A.row(row_1);
  A_for_given_row.row(2) = A.row(row_2);
  if (trySolveLinearType2<double>(A_for_given_row, solution_0, solution_1, false)) return true;
  if (!A.allFinite()) return false;
  int best_i = -1, best_j = -1;
  double best_det = 1e-10;
  for (int i = 0; i < n_rows; ++i) {
    for (int j = i + 1; j < n_rows; ++j) {
      const double det = std::abs(A(i, 0) * A(j, 1) - A(i, 1) * A(j, 0));
      if (det > best_det) { best_det = det; best_i = i; best_j = j; }
    }
  }
  if (best_i < 0) return false;
  A_for_given_row.row(0) = A.row(best_i);
  A_for_given_row.row(1) = A.row(best_j);
  const Eigen::Matrix2d B = -A_for_given_row.block<2, 2>(0, 0).inverse() * A_for_given_row.block<2, 2>(0, 2);
  int best_k = -1;
  double best_residual = -1;
  for (int k = 0; k < n_rows; ++k) {
    if (k == best_i || k == best_j) continue;
    const Eigen::RowVector2d residual = A.row(k).template tail<2>() + A.row(k).template head<2>() * B;
    if (residual.squaredNorm() > best_residual) {
      best_residual = residual.squaredNorm(); best_k = k;
    }
  }
  if (best_k < 0) return false;
  A_for_given_row.row(2) = A.row(best_k);
  return trySolveLinearType2<double>(A_for_given_row, solution_0, solution_1);
}

// Up to four candidates from the intersection of a rank-two nullspace with
// ||(sin x, cos x)||^2 = ||(sin y, cos y)||^2. Families use representatives;
// numerical rank/empty results are not exact completeness certificates.
template <int n_rows>
std::vector<double> solveLinearType2All(
    const Eigen::Ref<const Eigen::Matrix<double, n_rows, 4>>& A) {
  std::vector<double> angles;
  if (!A.allFinite()) return angles;
  const double scale = A.cwiseAbs().maxCoeff();
  if (!(scale > 0)) return angles;
  const Eigen::Matrix<double, n_rows, 4> normalized = A/scale;
  Eigen::JacobiSVD<Eigen::Matrix<double, n_rows, 4>> svd(normalized, Eigen::ComputeFullV);
  const auto& singular = svd.singularValues();
  const double epsilon = 64*std::numeric_limits<double>::epsilon();
  const double tolerance = epsilon*std::max(n_rows, 4)*singular[0];
  if (singular.size() < 2 || singular[1] <= tolerance) return angles;
  std::vector<Eigen::Vector4d> directions;
  if (singular.size() >= 3 && singular[2] > tolerance) {
    if (singular.size() >= 4 && singular[3] > 1e-8*singular[0]) return angles;
    directions.push_back(svd.matrixV().col(3));
  } else {
    const Eigen::Matrix<double, 4, 2> basis = svd.matrixV().template rightCols<2>();
    const Eigen::Matrix2d quadratic = basis.template topRows<2>().transpose()*basis.template topRows<2>()-
        basis.template bottomRows<2>().transpose()*basis.template bottomRows<2>();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(quadratic);
    if (eigen.info() != Eigen::Success) return angles;
    const double low = eigen.eigenvalues()[0], high = eigen.eigenvalues()[1];
    if (low < 0 && high > 0) {
      const Eigen::Vector2d negative = std::sqrt(high)*eigen.eigenvectors().col(0);
      const Eigen::Vector2d positive = std::sqrt(-low)*eigen.eigenvectors().col(1);
      directions.push_back(basis*(negative+positive));
      directions.push_back(basis*(negative-positive));
    } else if (std::max(std::abs(low), std::abs(high)) <= epsilon) {
      return {0.0, M_PI};
    } else {
      const int zero = std::abs(low) < std::abs(high) ? 0 : 1;
      if (std::abs(eigen.eigenvalues()[zero]) > epsilon) return angles;
      directions.push_back(basis*eigen.eigenvectors().col(zero));
    }
  }
  for (const auto& direction : directions) {
    if (direction.template tail<2>().norm() <= 1e-12) continue;
    const double angle = std::atan2(direction[2], direction[3]);
    double other = angle+M_PI;
    if (other > M_PI) other -= 2*M_PI;
    for (double value : {angle, other})
      if (std::find(angles.begin(), angles.end(), value) == angles.end()) angles.push_back(value);
  }
  return angles;
}

template <int n_rows>
std::vector<double> solveLinearType2Candidates(
    const Eigen::Ref<const Eigen::Matrix<double, n_rows, 4>>& A,
    const std::vector<std::array<int, 3>>& rows_to_try = {}) {
  std::vector<double> angles;
  if (!A.allFinite()) return angles;
  const double scale = A.cwiseAbs().maxCoeff();
  if (!(scale > 0)) return angles;
  Eigen::Matrix<double, n_rows, 4> legacy = A;
  if (scale < 1e-100 || scale > 1e100) legacy /= scale;
  for (const auto& rows : rows_to_try) {
    double first, second;
    if (trySolveLinearType2SpecificRows<n_rows>(legacy, rows[0], rows[1], rows[2], first, second)) {
      for (double value : {first, second}) if (std::isfinite(value)) angles.push_back(value);
      break;
    }
  }
  for (double value : solveLinearType2All<n_rows>(A))
    if (std::find(angles.begin(), angles.end(), value) == angles.end()) angles.push_back(value);
  return angles;
}

}  // namespace linear_solver

// For general-6dof solver
namespace general_6dof_internal {

template <int n_equations = 14, int n_lhs_unknowns = 9, int n_rhs_unknowns = 8>
bool numericalReduce(
    const Eigen::Matrix<double, n_equations, n_lhs_unknowns>& A_sin,
    const Eigen::Matrix<double, n_equations, n_lhs_unknowns>& A_cos,
    const Eigen::Matrix<double, n_equations, n_lhs_unknowns>& C_const,
    const Eigen::Matrix<double, n_equations, n_rhs_unknowns>& rhs_matrix,
    const std::array<int, n_rhs_unknowns>& lines_to_reduce,
    const std::array<int, n_equations - n_rhs_unknowns>& remaining_lines,
    Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns>*
        tau_sin,
    Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns>*
        tau_cos,
    Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns>*
        tau_const) {
  constexpr int remaining_rows = n_equations - n_rhs_unknowns;
  Eigen::Matrix<double, remaining_rows, n_rhs_unknowns> R_u;
  Eigen::Matrix<double, n_rhs_unknowns, n_rhs_unknowns> R_l;
  Eigen::Matrix<double, remaining_rows, n_lhs_unknowns> P_u_sin, P_u_cos,
      P_u_const;
  Eigen::Matrix<double, n_rhs_unknowns, n_lhs_unknowns> P_l_sin, P_l_cos,
      P_l_const;

  // Fill in the matrix for reduction
  int reduced_row_counter = 0;
  for (auto i = 0; i < lines_to_reduce.size(); i++) {
    auto row_idx = lines_to_reduce[i];
    R_l.row(reduced_row_counter) = rhs_matrix.row(row_idx);
    P_l_sin.row(reduced_row_counter) = A_sin.row(row_idx);
    P_l_cos.row(reduced_row_counter) = A_cos.row(row_idx);
    P_l_const.row(reduced_row_counter) = C_const.row(row_idx);
    reduced_row_counter += 1;
  }

  // Fill in the matrix for remaining rows
  int remaining_row_counter = 0;
  for (auto i = 0; i < remaining_lines.size(); i++) {
    auto row_idx = remaining_lines[i];
    R_u.row(remaining_row_counter) = rhs_matrix.row(row_idx);
    P_u_sin.row(remaining_row_counter) = A_sin.row(row_idx);
    P_u_cos.row(remaining_row_counter) = A_cos.row(row_idx);
    P_u_const.row(remaining_row_counter) = C_const.row(row_idx);
    remaining_row_counter += 1;
  }

  // Inverse and check nan
  const Eigen::Matrix<double, n_rhs_unknowns, n_rhs_unknowns> inv_R_l =
      R_l.inverse();
  for (auto r = 0; r < inv_R_l.rows(); r++) {
    for (auto c = 0; c < inv_R_l.cols(); c++) {
      if (std::isnan(inv_R_l(r, c)) || (!std::isfinite(inv_R_l(r, c)))) {
        return false;
      }
    }
  }

  // Compute the reduction
  const Eigen::Matrix<double, remaining_rows, n_rhs_unknowns> R_u_dot_R_l_inv =
      R_u * inv_R_l;
  *tau_sin = P_u_sin - R_u_dot_R_l_inv * P_l_sin;
  *tau_cos = P_u_cos - R_u_dot_R_l_inv * P_l_cos;
  *tau_const = P_u_const - R_u_dot_R_l_inv * P_l_const;
  return true;
}

template <int mat_rows = 14>
void sincosCoefficientToTanhalfCoefficient(
    const Eigen::Matrix<double, mat_rows, 9>& A_sincos,
    Eigen::Matrix<double, mat_rows, 9>* A_tanhalf) {
  A_tanhalf->col(0) =
      A_sincos.col(3) - A_sincos.col(5) - A_sincos.col(7) + A_sincos.col(8);
  A_tanhalf->col(1) = (-2.0) * (A_sincos.col(2) - A_sincos.col(6));
  A_tanhalf->col(2) = (-1.0 * A_sincos.col(3)) - A_sincos.col(5) +
                      A_sincos.col(7) + A_sincos.col(8);
  A_tanhalf->col(3) = (-2.0) * (A_sincos.col(1) - A_sincos.col(4));
  A_tanhalf->col(4) = 4 * A_sincos.col(0);
  A_tanhalf->col(5) = (2.0) * (A_sincos.col(1) + A_sincos.col(4));
  A_tanhalf->col(6) = (-1.0 * A_sincos.col(3)) + A_sincos.col(5) -
                      A_sincos.col(7) + A_sincos.col(8);
  A_tanhalf->col(7) = (2.0) * (A_sincos.col(2) + A_sincos.col(6));
  A_tanhalf->col(8) =
      A_sincos.col(3) + A_sincos.col(5) + A_sincos.col(7) + A_sincos.col(8);
}

template <typename T, std::size_t Capacity>
int computeSolutionFromTanhalfLME(const Eigen::Matrix<T, 6, 9>& A_x2,
                                  const Eigen::Matrix<T, 6, 9>& B_x,
                                  const Eigen::Matrix<T, 6, 9>& C,
                                  std::array<T, Capacity>* solution_buffer) {
  if (!A_x2.allFinite() || !B_x.allFinite() || !C.allFinite()) return 0;
  const T scale = std::max({A_x2.cwiseAbs().maxCoeff(), B_x.cwiseAbs().maxCoeff(),
                            C.cwiseAbs().maxCoeff()});
  if (!(scale > T(0))) return 0;
  using Matrix6 = Eigen::Matrix<T, 6, 9>;
  using Matrix12 = Eigen::Matrix<T, 12, 12>;
  const Matrix6 a=A_x2/scale, b=B_x/scale, c=C/scale;
  auto expand = [](const Matrix6& value) -> Matrix12 {
    Matrix12 matrix=Matrix12::Zero();
    matrix.template block<6, 9>(0, 0)=value;
    matrix.template block<6, 9>(6, 3)=value;
    return matrix;
  };
  using Matrix24 = Eigen::Matrix<T, 24, 24>;
  int solution_counter=0;
  auto append_eigenvalues = [&](const Matrix24& matrix, T chart) {
    if (!matrix.allFinite()) return;
    Eigen::EigenSolver<Matrix24> solver;
    solver.compute(matrix, false);
    if (solver.info() != Eigen::Success) return;
    const auto& eigenvalues=solver.eigenvalues();
    for (Eigen::Index i=0; i<eigenvalues.size(); ++i) {
      const auto& value=eigenvalues[i];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
          std::abs(value.imag()) >= 1e-6 || solution_counter >= static_cast<int>(Capacity)) continue;
      const T angle=T(2)*std::atan(value.real())+T(2)*std::atan(chart);
      solution_buffer->at(solution_counter++) = chart == T(0) ? angle :
          std::atan2(std::sin(angle), std::cos(angle));
    }
  };
  // Preserve established candidates before adding a better conditioned chart.
  // An estimate of conditioning alone must not discard an existing FK witness.
  const Matrix12 original_leading=expand(A_x2);
  const Matrix12 inverse=original_leading.inverse();
  bool legacy_usable=false;
  if (inverse.allFinite()) {
    Matrix24 legacy=Matrix24::Zero();
    legacy.template block<12, 12>(0, 12).setIdentity();
    legacy.template block<12, 12>(12, 0)=-(inverse*expand(C));
    legacy.template block<12, 12>(12, 12)=-(inverse*expand(B_x));
    legacy_usable=legacy.allFinite();
    append_eigenvalues(legacy, T(0));
  }
  // Change the tangent-half-angle chart: x=(y+t)/(1-t*y). At most one
  // alternative chart is added, keeping the combined candidate bound at 48.
  T best_reciprocal=T(0), chart=T(0);
  bool ordinary_stable=false;
  Matrix12 leading;
  for (const T t : {T(0), T(1), T(-1), T(.5), T(-.5), T(2), T(-2)}) {
    const Matrix12 candidate=expand(a-t*b+t*t*c);
    const Eigen::FullPivLU<Matrix12> lu(candidate);
    // rcond() estimates an inverse through solve(); on a rank-deficient
    // FullPivLU it can report a deceptively large reciprocal condition.
    const T reciprocal=lu.isInvertible() ? lu.rcond() : T(0);
    if (std::isfinite(reciprocal) && reciprocal > best_reciprocal) {
      best_reciprocal=reciprocal; chart=t; leading=candidate;
    }
    if (t == T(0) && reciprocal > T(1e-8) && a.cwiseAbs().maxCoeff() > T(1e-8)) {
      ordinary_stable=true; break;
    }
  }
  if ((ordinary_stable && legacy_usable) || best_reciprocal < T(1e-14)) return solution_counter;
  const Matrix12 linear=expand(T(2)*chart*a+(T(1)-chart*chart)*b-T(2)*chart*c);
  const Matrix12 constant=expand(chart*chart*a+chart*b+c);
  const Eigen::PartialPivLU<Matrix12> factor(leading);

  Matrix24 M;
  M.setZero();
  M.template block<12, 12>(0, 12).setIdentity();
  M.template block<12, 12>(12, 0) = -factor.solve(constant);
  M.template block<12, 12>(12, 12) = -factor.solve(linear);
  append_eigenvalues(M, chart);

  return solution_counter;
}

template <int mat_rows, int mat_cols>
void sincosLME2TanhalfLME(
    const Eigen::Matrix<double, mat_rows, mat_cols>& A_sin,
    const Eigen::Matrix<double, mat_rows, mat_cols>& A_cos,
    const Eigen::Matrix<double, mat_rows, mat_cols>& C_const,
    // Output
    Eigen::Matrix<double, mat_rows, mat_cols>* A,
    Eigen::Matrix<double, mat_rows, mat_cols>* B,
    Eigen::Matrix<double, mat_rows, mat_cols>* C) {
  *A = C_const - A_cos;
  *B = 2 * A_sin;
  *C = C_const + A_cos;
}

};  // namespace general_6dof_internal

template <int n_equations = 14, int n_lhs_unknowns = 9, int n_rhs_unknowns = 8,
          std::size_t Capacity = 16>
int general6DofNumericalReduceSolve(
    const Eigen::Matrix<double, n_equations, n_lhs_unknowns>& A_sin,
    const Eigen::Matrix<double, n_equations, n_lhs_unknowns>& A_cos,
    const Eigen::Matrix<double, n_equations, n_lhs_unknowns>& C_const,
    const Eigen::Matrix<double, n_equations, n_rhs_unknowns>& rhs_matrix,
    const std::array<int, n_rhs_unknowns>& lines_to_reduce,
    const std::array<int, n_equations - n_rhs_unknowns>& remaining_lines,
    std::array<double, Capacity>* solution_buffer) {
  // First step, try reduce
  Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns> tau_sin;
  Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns> tau_cos;
  Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns> tau_const;
  bool reduced =
      general_6dof_internal::numericalReduce<n_equations, n_lhs_unknowns,
                                             n_rhs_unknowns>(
          A_sin, A_cos, C_const, rhs_matrix, lines_to_reduce, remaining_lines,
          &tau_sin, &tau_cos, &tau_const);

  // No solution if cannot reduce
  if (!reduced) return 0;

  // To tanhalf LME
  Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns>
      tau_sin_tanhalf;
  Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns>
      tau_cos_tanhalf;
  Eigen::Matrix<double, n_equations - n_rhs_unknowns, n_lhs_unknowns>
      tau_const_tanhalf;
  general_6dof_internal::sincosCoefficientToTanhalfCoefficient<n_equations -
                                                               n_rhs_unknowns>(
      tau_sin, &tau_sin_tanhalf);
  general_6dof_internal::sincosCoefficientToTanhalfCoefficient<n_equations -
                                                               n_rhs_unknowns>(
      tau_cos, &tau_cos_tanhalf);
  general_6dof_internal::sincosCoefficientToTanhalfCoefficient<n_equations -
                                                               n_rhs_unknowns>(
      tau_const, &tau_const_tanhalf);

  // Re-use the memory for A, B, C
  general_6dof_internal::sincosLME2TanhalfLME<n_equations - n_rhs_unknowns,
                                              n_lhs_unknowns>(
      tau_sin_tanhalf, tau_cos_tanhalf, tau_const_tanhalf, &tau_sin, &tau_cos,
      &tau_const);

  // Solve the equation
  int n_solution = general_6dof_internal::computeSolutionFromTanhalfLME(
      tau_sin, tau_cos, tau_const, solution_buffer);
  return n_solution;
}

template <typename T>
Eigen::Matrix<T, 4, 4> inverse_transform(
    const Eigen::Matrix<T, 4, 4>& transform) {
  Eigen::Matrix<T, 4, 4> inv;
  inv.setIdentity();
  auto rotation = transform.template block<3, 3>(0, 0).transpose();
  inv.template block<3, 3>(0, 0) = rotation;
  inv.template block<3, 1>(0, 3) =
      -rotation * transform.template block<3, 1>(0, 3);
  return inv;
}

template <typename T>
Eigen::Matrix<T, 4, 4> disturbTransform(const Eigen::Matrix<T, 4, 4>& transform,
                                        T disturb_angle = T(1e-2),
                                        T disturb_translation = T(1e-2),
                                        Eigen::Matrix<T, 4, 4>* disturbance_out = nullptr) {
  // Make angle axis
  Eigen::Matrix<T, 3, 1> axis;
  axis.setRandom();
  if (axis.squaredNorm() < 1e-4) {
    axis[0] += 0.1;
  }
  axis.normalize();
  Eigen::AngleAxis<T> angle_axis(disturb_angle, axis);

  // To matrix4 disturb
  Eigen::Matrix<T, 4, 4> disturb_matrix;
  disturb_matrix.setIdentity();
  disturb_matrix.template block<3, 3>(0, 0) = angle_axis.matrix();
  disturb_matrix(0, 3) -= disturb_translation;
  disturb_matrix(1, 3) -= disturb_translation;
  disturb_matrix(2, 3) -= disturb_translation;
  if (disturbance_out) *disturbance_out = disturb_matrix;
  return disturb_matrix * transform;
}

template <int robot_nq = 6>
void numericalRefinement(
    const std::function<Eigen::Matrix4d(const std::array<double, robot_nq>&)>&
        fk_functor,
    const std::function<void(const std::array<double, robot_nq>&,
                             Eigen::Matrix<double, 6, robot_nq>&)>&
        twist_jacobian_functor,
    const Eigen::Matrix4d& ee_target, std::array<double, robot_nq>& q,
    double diagonal_damping = 1e-4,
    const std::array<bool, robot_nq>& fixed_joints = {}) {
  if (!ee_target.allFinite() || !std::isfinite(diagonal_damping)) return;
  for (double value : q) {
    if (!std::isfinite(value)) return;
  }
  Eigen::Matrix<double, 6, robot_nq> jacobian;
  Eigen::Matrix<double, robot_nq, robot_nq> jtj;
  Eigen::Matrix<double, robot_nq, 1> delta_q, jte;
  Eigen::Matrix<double, 6, 1> twist_target;

  constexpr int n_iterations = 10;
  for (auto i = 0; i < n_iterations; i++) {
    // Compute the twist target
    Eigen::Matrix4d ee_pose_i = fk_functor(q);
    // A zero skew rotation residual alone would also occur at a pi rotation.
    const double pose_error = (ee_pose_i - ee_target).squaredNorm();
    if (!std::isfinite(pose_error) || pose_error <= 1e-20) return;
    Eigen::Matrix4d twist_target_4x4 = ee_target * inverse_transform(ee_pose_i);
    twist_target[0] = 0.5 * (twist_target_4x4(2, 1) - twist_target_4x4(1, 2));
    twist_target[1] = 0.5 * (twist_target_4x4(0, 2) - twist_target_4x4(2, 0));
    twist_target[2] = 0.5 * (twist_target_4x4(1, 0) - twist_target_4x4(0, 1));
    twist_target[3] = twist_target_4x4(0, 3);
    twist_target[4] = twist_target_4x4(1, 3);
    twist_target[5] = twist_target_4x4(2, 3);
    if (!twist_target.allFinite()) return;

    // Compute jacobian and solve it
    twist_jacobian_functor(q, jacobian);
    if (!jacobian.allFinite()) return;
    for (auto j = 0; j < robot_nq; j++) {
      if (fixed_joints[j]) jacobian.col(j).setZero();
    }
    jtj = jacobian.transpose() * jacobian;
    jte = jacobian.transpose() * twist_target;
    for (auto j = 0; j < robot_nq; j++) {
      jtj(j, j) += diagonal_damping;
    }
    if (!jtj.allFinite() || !jte.allFinite()) return;

    // Solve Ax = b and update
    delta_q = jtj.colPivHouseholderQr().solve(jte);
    if (!delta_q.allFinite()) return;
    const auto previous_q = q;
    for (auto j = 0; j < robot_nq; j++) {
      // A zero Jacobian column can still receive a tiny QR roundoff update.
      // Fixed coordinates retain the caller's exact value, including signed zero.
      if (fixed_joints[j]) continue;
      q[j] += delta_q[j];
      if (!std::isfinite(q[j])) {
        q = previous_q;
        return;
      }
    }
    if (q == previous_q) return;
  }
}

};  // namespace yaik_cpp
