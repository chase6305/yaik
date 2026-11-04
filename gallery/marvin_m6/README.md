# Marvin M6 S CCS 696 V4.0

This example provides separate analytical IK solvers for the left and right
seven-joint arms in `HumanoidAssets/Marvin_M6_S_CCS_696_V4.0`. Each arm has:

- `marvin_m6_{side}_arm.yaml`: modified DH model and URDF angle offsets.
- `marvin_m6_{side}_arm_ik.yaml`: serialized analytical solution tree.
- `marvin_m6_{side}_arm_ik_generated.py` and `.cpp`: generated FK, Jacobians,
  and IK kernels using the repository's existing interfaces.

`urdf/left_arm.urdf` and `urdf/right_arm.urdf` are exact source snapshots.
`provenance.json` records their source revision, SHA-256 hashes, generation
seed, dependencies, and generated artifact hashes. The snapshots retain mesh
references for provenance; meshes are not included or needed for kinematics.

## Frames and joint convention

Targets are 4×4 rigid transforms from `left_arm_base` to `left_ee`, or from
`right_arm_base` to `right_ee`, respectively. Both position and orientation
must use the selected arm's base frame. The generated files already include
the correct side's base transform, the shoulder's `0.000029142 m` Y offset,
and the fixed `0.095 m` wrist-to-EE transform.

Joint vectors follow the URDF order:

| Index | Joint | IK convention |
| --- | --- | --- |
| 0 | SHOULDER_PITCH_{L,R}_J1 | Solved |
| 1 | SHOULDER_ROLL_{L,R}_J2 | Solved |
| 2 | ELBOW_PITCH_{L,R}_J3 | Solved |
| 3 | ELBOW_YAW_{L,R}_J4 | Solved |
| 4 | WRIST_PITCH_{L,R}_J5 | Fixed runtime parameter `th_4` / `q5` |
| 5 | WRIST_YAW_{L,R}_J6 | Solved |
| 6 | WRIST_ROLL_{L,R}_J7 | Solved |

Angles are in radians and lengths are in metres. Joint angles need no
permutation or sign changes. `th_4` means the fifth joint, not J4. A target
reachable at one J5 value may be unreachable at another; the fixed parameter
is a hard constraint. To use a whole-robot base or a different TCP, convert
the full target transform into the corresponding standalone arm/EE frames.

## Using the bounded analytical API

`solver.py` and the standalone C++17 `solver.hpp` provide a deterministic
geometric solver for this exact model. They recover shoulder angles from
the full rotation matrix, handle aligned shoulder axes, filter joint limits,
remove duplicate solutions, and verify the final pose. They preserve the
supplied J5 exactly and use no numerical refinement or target perturbation.
Accepted candidates must have position error at most `1e-7 m` and rotation
angle error at most `1e-6 rad`. Near singularities, accepted endpoint
representatives can approach the position tolerance.

The geometric derivation uses the three intersecting shoulder axes. In raw
DH angles, the wrist position is `S @ p`, with shoulder rotation `S` and

```text
p = [0.018 + 0.018*cos(t3) + 0.314*sin(t3),
     0,
     0.287 + 0.018*sin(t3) - 0.314*cos(t3)]
```

Its squared length determines the elbow angle. The inverse target translation
then determines the remaining wrist angles for fixed J5. Removing the distal
rotation leaves the shoulder rotation for Euler decomposition. Near aligned
axes, correlated angle recovery and feasible family representatives avoid
dividing by a vanishing rotation-column projection. Wrist endpoint candidates
are also verified against the original target before acceptance.

Python, from the repository root (NumPy is the only runtime dependency):

```python
import numpy as np
from gallery.marvin_m6.solver import Solver

arm = Solver('right')                    # or 'left'
q = np.array([0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1])
target = arm.fk(q)                        # replace with your arm-base -> EE target
solutions = arm.ik(target, q5=q[4])
```

C++17, with `gallery/marvin_m6` and Eigen3 on the include path:

```cpp
#include "solver.hpp"

int main() {
    const marvin_m6::JointVector q{{0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1}};
    const auto side = marvin_m6::Side::Right;
    const auto target = marvin_m6::computeFK(q, side);
    const auto solutions = marvin_m6::solveIK(target, q[4], side);
    return solutions.empty() ? 1 : 0;
}
```

An invalid rigid transform or invalid/out-of-limit J5 raises `ValueError` in
Python or `std::invalid_argument` in C++. Empty output means no accepted
representative was found for that target and fixed J5. Inverse-wrist-axis
continua use 33 deterministic bounded J7 samples; this finite sampling does
not guarantee finding every feasible singular configuration.

The `*_ik_generated` files retain the standard gallery API and its unbounded
candidate semantics. Python entry points are `marvin_m6_{side}_arm_fk(q)` and
`marvin_m6_{side}_arm_ik_solve(target, th_4)`. C++ uses the
`marvin_m6_{side}_arm_ik` struct's `computeFK` and `computeIK` methods and
requires the repository's `yaik_cpp_common.h`. Its checked `computeIK`
retains the generator's perturbation/refinement fallback. Use the bounded
API above when you need strict limits and deterministic analytical behavior.

## Regeneration

Run from the repository root with NumPy, SymPy, PyYAML, and attrs installed:

```sh
# Validate snapshots/models and regenerate code from the saved solution trees.
python gallery/marvin_m6/generate.py --codegen-only

# Check committed artifacts without modifying the gallery.
python gallery/marvin_m6/generate.py --check

# Repeat the offline symbolic search (can take substantially longer).
python gallery/marvin_m6/generate.py --seed 42
```

The internal DH chain is shared, so the offline search runs once. The script
then saves each tree with its own base transform and original angle mapping.
It rejects code generation when the saved tree's model differs from the
input URDF. A fixed seed controls offline sampling; process deadlines and
dependency versions can still change the symbolic search result. Code
generation from a saved tree is checked separately for reproducibility.

## Validation

The validator parses the URDF independently of the model converter and
checks generated FK and returned IK candidates against that FK. It covers
random poses, joint limits, and singular configurations, with J5 fixed to
the value used to create each target. C++ validation compiles the actual
generated source and requires a C++17 compiler and Eigen3 headers.
The report separates standard generated-kernel coverage from the bounded
analytical API; the validation exit status requires the bounded API to pass
all sampled reachable targets and reject its negative-input cases.

A local run with seed `20261103` checked 1,000 random targets and 233 joint-limit,
singular, and near-singular targets per arm:

| API | Left arm | Right arm |
| --- | ---: | ---: |
| Standard generated Python | 1,133 / 1,233 | 1,133 / 1,233 |
| Standard generated C++ | 1,200 / 1,233 | 1,197 / 1,233 |
| Bounded analytical Python | 1,233 / 1,233 | 1,233 / 1,233 |
| Bounded analytical C++ | 1,233 / 1,233 | 1,233 / 1,233 |

Each count requires at least one solution passing independent URDF FK, joint
limits, and fixed-J5 checks. Standard kernels have four unimplemented degenerate
branches and retain the repository's looser matrix-error contract; their report
also records candidates outside limits or above the stricter integration pose
thresholds. The bounded APIs returned no invalid or duplicate candidates, and
rejected all 21 negative-input cases per arm/backend. Their maximum measured
position and rotation errors were `6.10e-9 m` and `9.40e-10 rad`.

```sh
python gallery/marvin_m6/validate.py --samples 1000 --seed 20261103 \
    --output /tmp/marvin-m6-validation.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
    gallery/marvin_m6 solver/test_regressions.py -q
```

Random sampling and finite representatives of singular solution families do
not establish completeness over the full workspace. These solvers evaluate
kinematics and joint position limits; collision and trajectory constraints
need separate handling.
