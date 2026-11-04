# yaik

A Python package for generating analytical inverse kinematic solvers for robotics manipulators. The generated solvers support many robots in [Mech-Mind Robotics](https://www.mech-mind.com/) and have been extensively tested in real-world deployment. The name **yaik** stands for **Y**et **A**nother **A**nalytical **I**nverse **K**inematics generator.

### Features

- Closed-form, analytical inverse kinematics for serial chain robots in python/C++
- Support robots with 7 DoF or more, such as [Kuka IIwa Arm](./gallery/kuka_iiwa), [Panda Robot](./gallery/franka_panda), or arms of the Atlas humanoid
- A new factorized polynomial solution based on *groebner reduce*
- General 6-dof solution using linear matrix equations tangent of half angles
- Analysis to avoid degenerated (or conditional degenerated) solutions

### Using the generated solver

The [gallery](./gallery) contains generated IK solvers for tens of robots. Even if your robot is not included in the library, it might include a robot with the same structure and can be used after parameter modification. The file format is

```
gallery/
    puma/
        puma_ik.yaml
        puma_ik_generated.cpp
        puma_ik_generated.py
    {robot_name}/
        {robot_name}_ik.yaml
        {robot_name}_ik_generated.cpp
        {robot_name}_ik_generated.py
```

The [Marvin M6 V4.0 example](./gallery/marvin_m6/README.md) provides separate
left and right arm models, serialized IK trees, and Python/C++ solvers for the
`Marvin_M6_S_CCS_696_V4.0` URDFs, with independent URDF validation.

To use the generated C++ solver, you need to include the [yaik_cpp_common.h](./yaik_cpp_common.h) file, which further depends on Eigen. Please refer to the [yaik_cpp](https://bitbucket.org/gaowei19951004/yaik_cpp/src/master/) repo for an example.

To use the generated Python solver, you need to import [python_run_import.py](./python_run_import.py), which further depends on numpy.

Checked IK wrappers reject nonfinite targets and scalar parameters, and verify candidates
with generated FK. Fixed redundant joints retain the supplied angle representation.
C++ normalizes other revolute joints before FK verification; prismatic joints are not wrapped.
The existing error contracts are preserved: Python checks maximum matrix-entry error
against `1e-4`, while C++ checks squared matrix error against `1e-6` (`1e-4` after refinement).
For robot integration, validate base/tool frame conversion, URDF limits, and independent
position/rotation errors as needed; these matrix tolerances do not impose joint limits.
C++ callers can pass a per-joint revolute mask as the third argument to
`yaik_cpp::removeDuplicate<N>` for periodic comparison after normalization and limit
filtering. The default comparison preserves distinct `-pi`/`+pi` representations,
which can differ in feasibility under joint limits; prismatic mask entries stay false.

### Generate a solver for your robot

To generate a new solver for a robot that is not in the gallery, you need to install python3, [pyyaml](https://github.com/yaml/pyyaml), [attrs](https://www.attrs.org/en/stable/index.html) and [sympy](https://github.com/sympy/sympy/wiki/Download-Installation) on your system. Optionally, you might install [sagemath](https://www.sagemath.org/index.html), which contains a faster groebner reduce implementation.

Then, you need to express your robot in **Modified** DH format. A documentation regarding that can be found in [here](./docs/modified_dh.md). Several examples can be found in [this file](./fk/robot_models.py). Alternatively, the modified dh model can be converted from chain model in [this file](./fk/chain_models.py), or URDF using a tool in [another library](https://bitbucket.org/gaowei19951004/multibody/src/main/examples/urdf_sdf_to_dh.cpp).

After that, you can run `ik_solve.py` to generate the solution for your robot in YAML format. This step can take a LOT of time, as we perform offline analysis to find the simplest solution and avoid degeneration. Moreover, this step might fail to find a feasible solution (although the pipeline succeeded for all the robots we tested).

Finally, you can use `cpp_codegen.py/py_codegen.py` to generate solvers in C++ and Python, similar to the ones in the gallery.

For repeatable offline sampling and candidate selection, pass a seed:

```python
from ik_solve import run_ik, RunIKOption

run_ik(robot, run_ik_option=RunIKOption(random_seed=42))
```

The seeded run restores the caller's Python and NumPy random states when it finishes.
Use the same dependency versions when comparing generated output; process deadlines can
still change which expensive candidates finish. Automatic special samples stay within
joint bounds and include their endpoints; explicitly requested branch values are preserved.
Timed unary searches include algebraic expansion and use one deadline per call; small
single-variable searches batch rules into one isolated worker.

### Acknowledgement

This work was done during the author's employment at Mech-Mind Robotics. We have learned a lot from existing works, in particular [IkFast](https://github.com/rdiankov/openrave) and [IKBT](https://github.com/uw-biorobotics/IKBT).

### Contact

If you have a new challenging robot that cannot be solved, please email me.
