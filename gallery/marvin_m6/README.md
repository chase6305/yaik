# Marvin M6 S CCS 696 V4.0

[CLI overview](../../README.md) · [中文使用说明](../../README.zh-CN.md)

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

Both arms share the same internal geometry, joint axes, limits, and fixed EE
transform. Their URDF shoulder mounting transforms differ:

| First-joint origin | Left | Right |
| --- | --- | --- |
| XYZ, metres | `(0.2215, 0.000029142, 1.121)` | `(-0.2215, 0.000029142, 1.121)` |
| RPY, radians | `(-pi/2, 0, -pi/2)` | `(pi/2, 0, -pi/2)` |

The search can be shared while each saved tree retains its own base transform.
The same joint vector gives different left/right poses; users pass their original
URDF angles directly, without mirroring or changing signs.

## Command-line FK and IK

Run from the repository root. These commands use the bounded analytical API:

```sh
python yaik.py marvin fk --side right \
    --joints '[0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]' \
    --output build/right-target.json

python yaik.py marvin ik --side right \
    --target build/right-target.json --output build/right-solutions.json
```

`--joints` takes a quoted JSON array of seven URDF angles in radians. FK JSON
includes the 4x4 `target`, `q5`, `side`, `base_frame`, and `tip_frame`. IK reads
that object or a raw 4x4 JSON matrix; raw matrices need an explicit `--q5`.
An explicit `--q5` overrides a recorded value. Side/frame metadata must match
`--side`. The output contains `status`, `count`, and `solutions`.

Without `--output`, results go to stdout. `--target -` reads stdin:

```sh
python yaik.py marvin fk --side left \
    --joints '[0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]' \
    | python yaik.py marvin ik --side left --target -
```

Exit code `0` means success, `1` means no accepted representative was found,
and `2` means an input, dependency, or file error. Errors go to stderr; use
`python yaik.py --debug ...` for a traceback. Empty output does not establish
mathematical infeasibility. See [the root README](../../README.md) for installation
and generic generation commands.

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

The generic generator now also recovers coefficient-zero branches. It solves
explicit alternatives from the complete FK equations, retaining their guards,
and tries 33 bounded representatives of one revolute joint on larger singular
branches. Only complete alternatives are attached to a tree. C++ workspace
capacity follows the actual candidate and branch counts; it is no longer
limited to 16 candidates for these trees.

Recovery retains the 50-node budget and uses at most one representative joint per
path. Export still reports unresolved conditions in the expanded trees. This
does not establish full-workspace completeness, even when all regression
targets pass; the strict validator below gates the generated APIs themselves.

## Regeneration

Run from the repository root with NumPy, SymPy, PyYAML, and attrs installed:

```sh
# Export from saved trees to a new directory; no offline search.
python yaik.py marvin generate --output-dir build/marvin_m6

# Check the exported files, or omit --output-dir to check the gallery.
python yaik.py marvin check --output-dir build/marvin_m6

# Repeat the offline symbolic search explicitly (can take substantially longer).
python yaik.py marvin generate --solve --seed 42 --output-dir build/marvin-regenerated
```

The internal DH chain is shared, so the offline search runs once. The script
then saves each tree with its own base transform and original angle mapping.
For code export, existing output trees are used if present, otherwise the gallery
trees; `--tree-dir` selects a source explicitly. Both models are checked against
the supplied URDFs before output files change, and all requested artifacts are
staged before publication. Unrelated files are preserved. `--models-only` exports
just the modified-DH models. `--solve` and `--models-only` are mutually exclusive.

`--urdf-dir` selects another directory containing the two arm URDFs. A modified
snapshot records a null source revision instead of claiming the original commit.
Code export preserves a search seed only when source provenance matches both tree
hashes; otherwise it records `null`. `--seed` controls the new search with
`--solve`. A fixed seed controls offline sampling, while process deadlines and
dependency versions can still change the symbolic result. Code generation from
a saved tree is checked separately for reproducibility.
Equivalent `+pi`/`-pi` offsets use a fixed DH convention across BLAS backends;
model mismatches report the first differing field and its saved/URDF values.

The direct `generate.py` script remains available for compatibility:
`python gallery/marvin_m6/generate.py --codegen-only --output-dir build/marvin_m6`
and `python gallery/marvin_m6/generate.py --check`. With no mode flag that script
performs a full search; the unified CLI requires `--solve` for that operation.

## Validation

The validator parses the URDF independently of the model converter and
checks generated FK and returned IK candidates against that FK. It covers
random poses, joint limits, and singular configurations, with J5 fixed to
the value used to create each target. C++ validation compiles the actual
generated source and requires a C++17 compiler and Eigen3 headers.
The report separates standard generated-kernel coverage from the bounded
analytical API. The validation exit status requires both APIs to solve all
sampled reachable targets and reject their respective negative-input cases.

A local run with seed `20261103` checked 1,000 random targets and 233 joint-limit,
singular, and near-singular targets per arm:

| API | Left arm | Right arm |
| --- | ---: | ---: |
| Standard generated Python | 1,233 / 1,233 | 1,233 / 1,233 |
| Standard generated C++ | 1,233 / 1,233 | 1,233 / 1,233 |
| Bounded analytical Python | 1,233 / 1,233 | 1,233 / 1,233 |
| Bounded analytical C++ | 1,233 / 1,233 | 1,233 / 1,233 |

Each count requires at least one solution passing independent URDF FK, joint
limits, and fixed-J5 checks. The completed generated kernels recover the zero
pose, its J4 perturbations, and all 96 wrist-axis/near-axis cases in this suite.
They retain the repository's looser matrix-error contract; their report also
records candidates outside limits or above the stricter integration pose
thresholds. The bounded APIs returned no invalid or duplicate candidates, and
rejected all 21 negative-input cases per arm/backend. Their maximum measured
position and rotation errors were `6.10e-9 m` and `9.40e-10 rad`.

```sh
python yaik.py marvin validate --samples 1000 --seed 20261103 \
    --output /tmp/marvin-m6-validation.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
    test_cli.py gallery/marvin_m6 solver/test_regressions.py \
    solver/test_codegen_safety.py solver/test_equation_branches.py -q
```

Use `--python-only` without a C++ toolchain, or `--eigen-dir PATH` to select
another Eigen3 include directory. Both bounded arms share one compiled native
library with a runtime side argument; generated kernels are compiled separately.

Random sampling and finite representatives of singular solution families do
not establish completeness over the full workspace. These solvers evaluate
kinematics and joint position limits; collision and trajectory constraints
need separate handling.
