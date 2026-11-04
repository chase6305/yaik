# yaik

[简体中文](./README.zh-CN.md) · [Marvin M6 guide](./gallery/marvin_m6/README.md)

**Yet Another Analytical Inverse Kinematics generator** produces Python and C++
FK, Jacobian, and IK solvers for serial robots. It supports redundant arms,
conditional degenerate branches, polynomial solutions, and general 6-DoF methods.
The [gallery](./gallery) contains saved solutions and generated code.

## Installation

Use Python 3.10 or newer, from a checkout of this repository:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python yaik.py --help
```

The pinned requirements match the Marvin generation environment. Marvin runtime
FK/IK needs only NumPy; help and the built-in model list need only Python's
standard library. Code generation additionally uses SymPy, PyYAML, and attrs.
Native C++ validation needs `g++`, C++17, and Eigen3 headers; on Ubuntu:

```sh
sudo apt-get install g++ libeigen3-dev
```

Run the examples below from the repository root. An absolute path to `yaik.py`
also works from another directory; relative input/output paths follow your
current directory. No source-code edits are needed to choose a robot or output.

## Choose a workflow

| Task | Command |
| --- | --- |
| List built-in robot names | `python yaik.py models` |
| Generate a new analytical solution | `python yaik.py generate --builtin puma --output-dir build/puma` |
| Export an existing IK tree | `python yaik.py codegen gallery/puma/puma_ik.yaml --output-dir build/puma` |
| Use the Marvin arms | `python yaik.py marvin --help` |

`generate` performs offline symbolic search and can take minutes or longer.
`codegen` reuses a saved solution tree and writes code without repeating that
search. Both emit an artifact list as JSON; progress and diagnostics go to stderr.
Use `COMMAND --help` for options.

## Marvin M6: use the arms immediately

The [Marvin example](./gallery/marvin_m6/README.md) contains separate left/right
models, saved trees, generated Python/C++ kernels, and bounded analytical APIs.
The CLI uses the bounded API, including joint-limit filtering and pose checks.

Compute a reachable target from seven URDF joint angles, then solve it:

```sh
python yaik.py marvin fk --side right \
  --joints '[0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]' \
  --output build/target.json

python yaik.py marvin ik --side right \
  --target build/target.json --output build/solutions.json
```

The quoted joint array accepts negative and scientific-notation values.
`fk` writes the target matrix, frame names, side, and `q5`. `ik` accepts that
object or a raw 4×4 matrix in a JSON file. For a raw matrix, supply `--q5 0.4`.
An explicit `--q5` overrides the FK object's recorded value. Without `--output`,
results go to stdout; `--target -` reads stdin, so these commands also compose:

```sh
python yaik.py marvin fk --side left \
  --joints '[0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]' \
  | python yaik.py marvin ik --side left --target -
```

Angles are in radians, lengths in metres, and joint order follows the URDF.
**J5 is the fifth joint (`q[4]`, generated parameter `th_4`), held fixed during
IK.** A target reachable with one J5 value may have no solution at another.
Targets map `left_arm_base → left_ee` or `right_arm_base → right_ee`, including
the original shoulder placement and fixed 95 mm EE transform. FK JSON metadata
is checked against `--side` to catch accidental frame mixing.

Both arms have identical internal geometry and joint limits. Their shoulder
mounting transforms differ, so they share an analytical core while retaining
separate base transforms. The same joint vector produces different left/right
poses. No caller-side angle permutation or sign change is needed.

For Python integration, use `gallery.marvin_m6.solver.Solver`; for standalone
C++17, use `gallery/marvin_m6/solver.hpp`. See the guide for complete examples,
input frames, measured singular coverage, and tolerance contracts.

## Export or regenerate Marvin files

The default Marvin export is fast and writes to `build/marvin_m6`:

```sh
python yaik.py marvin generate --output-dir build/marvin_m6
python yaik.py marvin check --output-dir build/marvin_m6
```

It uses saved trees in the output directory when present, otherwise the gallery
trees. `--tree-dir PATH` selects saved trees explicitly. Both tree models must
match the input URDFs before output files are changed. Generation is staged
before publication, and unrelated output files are preserved. Export records
the verified source trees' seed, or `null` if no matching provenance is available;
`--seed` controls a new search, not code export.

To repeat the symbolic search explicitly:

```sh
python yaik.py marvin generate --solve --seed 42 \
  --output-dir build/marvin-regenerated
```

`--urdf-dir PATH` selects a directory containing `left_arm.urdf` and
`right_arm.urdf`; `--models-only` exports just the modified-DH models. The Marvin
converter follows this specific seven-joint arm topology. Its geometric bounded
APIs describe the checked-in V4.0 geometry.

## Generate for another robot

Choose a built-in model, or provide a modified-DH model YAML:

```sh
python yaik.py generate --builtin puma --seed 42 \
  --output-dir build/puma --language both

python yaik.py generate --model path/to/robot.yaml --seed 42 \
  --output-dir build/my_robot --language python
```

The [modified-DH documentation](./docs/modified_dh.md),
[built-in models](./fk/robot_models.py), and
[chain conversion examples](./fk/chain_models.py) describe the model format.
A model YAML has `name`, `dh_params`, and `unknowns`; an `*_ik.yaml` solution
tree has `robot` and `tree_nodes`. The CLI reports when these inputs are confused.

A successful search produces the model, `*_numerical_test.yaml`, intermediate
`*_main.yaml`, final `*_ik.yaml`, and requested generated files. `--test-cases`
can reuse saved numerical samples. `--language` accepts `python`, `cpp`, or
`both` (default). To export existing trees, use `codegen`; it loads a tree once
for both languages. C++ code uses safe mathematical operators.

Generated Python needs the repository's `python_run_import.py` on `PYTHONPATH`.
Generated C++ needs the repository's `yaik_cpp_common.h` and Eigen3 on its include
path. Runtime calls do not run symbolic search or require SymPy.

The library APIs remain available:

```python
from ik_solve import run_ik, RunIKOption
run_ik(robot, run_ik_option=RunIKOption(random_seed=42))
```

Seeds are integers from 0 through `2**32 - 1`. A seeded library run restores the
caller's Python and NumPy random states. Dependency versions and process deadlines
can still affect which symbolic candidates finish. Automatic special samples stay
within joint bounds and include endpoints; timed unary searches share one deadline
including algebra. Compatibility scripts also accept explicit inputs:

```sh
python ik_solve.py --model path/to/robot.yaml --output-dir build/my_robot
python py_codegen.py gallery/puma/puma_ik.yaml --output-dir build/puma
python cpp_codegen.py gallery/puma/puma_ik.yaml --output-dir build/puma
```

## Results, errors, and limits

| Exit code | Meaning |
| --- | --- |
| `0` | Success; IK returned at least one accepted representative |
| `1` | IK found no accepted representative, or validation failed; inspect JSON/report |
| `2` | Invalid arguments, input, dependency, or file access |
| `130` | Interrupted by the user |

FK/IK JSON output is kept separate from diagnostics. Output files replace the
specified destination after a complete result is available. Input errors have
short messages; `python yaik.py --debug ...` shows the traceback for diagnosis.

The bounded Marvin APIs require position error ≤ `1e-7 m` and rotation error
≤ `1e-6 rad`, preserve supplied J5, and reject invalid rigid transforms. Wrist-axis
continua use 33 deterministic bounded J7 samples; an empty result does not prove
infeasibility or workspace completeness. Collision and trajectory constraints
need application-level handling.

Standard generated wrappers keep their original contracts: Python checks maximum
matrix-entry error against `1e-4`; C++ checks squared matrix error against `1e-6`
(`1e-4` after refinement). They reject nonfinite input and verify FK, but do not
enforce URDF limits. Fixed redundant joints retain their supplied angle
representation; C++ normalizes other revolute joints before FK verification.
`yaik_cpp::removeDuplicate<N>` accepts an optional per-joint revolute mask for
periodic comparison after normalization and limit filtering. Its default keeps
distinct `-pi`/`+pi` representations, which can differ in limit feasibility.

Generation now completes coefficient-zero branches as well as variable-value
branches. It returns to the full FK equations to recover alternative formulas;
larger singular systems also try 33 bounded representatives of one revolute
joint. These candidates pass the generated wrapper's FK checks. This is finite
coverage of continuous families, with no guarantee of full-workspace completeness.
The stored Marvin trees and both generated languages include this recovery.

## Validation

```sh
python -m pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  test_cli.py gallery/marvin_m6 solver/test_regressions.py \
  solver/test_codegen_safety.py solver/test_equation_branches.py -q

python yaik.py marvin validate --samples 1000 --seed 20261103 \
  --output build/marvin-validation.json
```

Use `--python-only` without a native toolchain, or `--eigen-dir PATH` for another
Eigen installation. Validation uses independent XML FK, checks every returned
candidate, and separates standard-kernel coverage from bounded-API results.
Its exit status requires both generated and bounded APIs to retain a valid
bounded solution for every reachable case, reject their negative-input cases,
and pass all FK checks.
Both bounded C++ arms reuse one compilation with a runtime side argument.
The report records source hashes, seeds, failures, and error maxima. CI runs
artifact checks, CLI/regression tests, and native validation.

### Acknowledgement

This work was done during the author's employment at Mech-Mind Robotics. We have learned a lot from existing works, in particular [IkFast](https://github.com/rdiankov/openrave) and [IKBT](https://github.com/uw-biorobotics/IKBT).

### Contact

For an unsupported model or a failing command, include the model/URDF, exact CLI
invocation, dependency versions, and a `--debug` traceback in a repository issue.
