# Modified DH Parameter in YAIK

The modified DH parameter is shown in `docs/modified_dh.png`, but the DH parameter is not indexed in the same way. The DH parameter in the code is

```python
class DHEntry:  
	alpha # alpha_{i-1} in the figure
  a # a_{i-1} in the figure
  theta # theta_i in the figure
  d # d_i in the figure
```

This `DHEntry` is **modified** DH parameters. When using as modified DH parameter, the transformation is
$$
T_{i}^{i-1} = \text{rotx}(\alpha_{i-1})~\text{trans\_x}(a_{i-1})~\text{rotz}(\theta_i)~\text{trans\_z}(d_i)
$$
where $T_{i}^{i-1}$ maps coordinate vectors in $i$-th frame (child frame) to the $(i-1)$-th frame parent frame. In python code, it is written as

```python
entry_i = DHEntry()
T_{i}_to_{i-1} = rotx(entry_i.alpha) * trans_x(entry_i.a) * rotz(entry_i.theta) * trans_z(entry_i.d)
```

## Base and tool transforms

`RobotAuxiliaryData` maps the raw DH pose into the application's frames:
`T_original = pre_transform_sp * T_raw * post_transform_sp`.
Joint offsets use `q_raw = q_original + unknown_offset`.
Base/tool transforms must be rigid 4×4 matrices and independent of joint variables.
Their geometric parameters can be fixed or supplied at runtime:

```python
import sympy as sp
from fk.robots import RobotAuxiliaryData

mount_angle, tool_length = sp.symbols('mount_angle tool_length')
robot.symbolic_parameters.update({mount_angle, tool_length})
robot.parameters_bound.update({mount_angle: (-1., 1.), tool_length: (0., .2)})
robot.auxiliary_data = RobotAuxiliaryData()
robot.auxiliary_data.pre_transform_sp = sp.Matrix([
    [sp.cos(mount_angle), 0, sp.sin(mount_angle), 0],
    [0, 1, 0, 0],
    [-sp.sin(mount_angle), 0, sp.cos(mount_angle), 0],
    [0, 0, 0, 1]])
robot.auxiliary_data.post_transform_sp = sp.eye(4)
robot.auxiliary_data.post_transform_sp[2, 3] = tool_length
```

Generated FK, Jacobians, and checked IK accept runtime geometry in alphabetical
parameter-name order. IK then accepts fixed redundant joints in model joint order.
Target conversion helpers accept only runtime parameters used by the base/tool
transforms, also alphabetically. Declared parameter bounds guide offline sampling;
the standard generated API checks finite runtime values and FK, rather than
enforcing geometry bounds or all joint limits.

Stored rotations sometimes contain rounded constants such as `0.707107`.
Target conversion uses the actual matrix inverse for those frames and retains
the transpose fast path for symbolically orthogonal rotations. Checked IK keeps
the previous transpose-based candidates as an additional source for approximate
models; all candidates are checked against the same original target. Constant
singular frames and non-homogeneous bottom rows are rejected during export.
Runtime frame parameters must describe nonsingular transforms.

Raw IK takes a raw DH target and raw fixed-joint values, and returns raw DH joint
values. Add offsets before passing fixed joints to raw IK and subtract offsets
from its outputs before evaluating original-frame FK. The checked API performs
these conversions automatically.
Raw outputs require an assigned finite value for every physical joint. A
partial symbolic leaf remains unresolved and produces no raw candidate.
`generate --resume-tree PATH` can extend supported saved branches and records
its input hash and continuation seed separately from the historical search.

