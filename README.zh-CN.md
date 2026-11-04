# yaik 使用说明

[English](./README.md) · [Marvin M6 详细说明](./gallery/marvin_m6/README.md)

yaik 根据机器人运动学模型离线生成解析逆运动学解，输出 Python/C++ 的
FK、Jacobian 和 IK 代码。使用现成解文件时不需要重新进行符号求解。

## 安装与入口

使用 Python 3.10 或更新版本，在仓库根目录执行：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python yaik.py --help
```

Marvin 的运行时 FK/IK 只依赖 NumPy；查看帮助和列出内置模型不加载第三方库。
生成代码还需要 SymPy、PyYAML 和 attrs，已包含在 requirements 中。
C++ 验证需要 g++、C++17 和 Eigen3；Ubuntu 可执行
`sudo apt-get install g++ libeigen3-dev`。

下列示例在仓库根目录运行。在其他目录也可通过 `yaik.py` 的绝对路径调用，
输入和输出的相对路径按当前目录解析。

## 先使用 Marvin 左右臂

从一组关节角计算目标，再求逆解：

```sh
python yaik.py marvin fk --side right \
  --joints '[0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]' \
  --output build/target.json

python yaik.py marvin ik --side right \
  --target build/target.json --output build/solutions.json
```

`--joints` 是加引号的 JSON 数组，按 URDF 顺序填写 7 个角度，支持负数和科学计数法。
单位为弧度，长度单位为米。`fk` 的 JSON 输出包含目标矩阵、侧别、坐标系和 `q5`。
`ik` 默认使用其中的 `q5`；显式传入 `--q5` 可以覆盖它。

**固定输入是第 5 关节 J5，即 `q[4]`，生成代码中的名称是 `th_4`。**
它不是 J4。目标在某个 J5 下可达，不代表换一个 J5 仍然有解。
如果目标文件只有原始的 4×4 JSON 矩阵，必须显式指定 `--q5`。

`--side left/right` 必须填写。目标变换分别为
`left_arm_base → left_ee` 和 `right_arm_base → right_ee`，已包括肩部安装变换和
95 mm 固定末端变换。完整机器人基座或其他 TCP 的目标，需要先换算到对应坐标系。
CLI 会检查 FK JSON 的侧别和坐标系信息，提示意外混用。

左右臂的内部几何、关节轴和限位相同，可以共享解析核心；肩部安装变换不同，
因此各自保留基座变换。同一组关节角在两侧的末端位姿不同，调用者无需额外改符号或交换关节。

不指定 `--output` 时输出 JSON 到 stdout，`--target -` 从 stdin 读取：

```sh
python yaik.py marvin fk --side left \
  --joints '[0.3, -0.5, 0.7, -1.0, 0.4, -0.2, 0.1]' \
  | python yaik.py marvin ik --side left --target -
```

CLI 使用带关节限位过滤的解析接口。Python 可直接调用
`gallery.marvin_m6.solver.Solver`，C++17 使用 `gallery/marvin_m6/solver.hpp`。
完整调用例子和标准生成接口见 [Marvin 指南](./gallery/marvin_m6/README.md)。

## 快速导出与重新求解

只需要解文件时，默认复用现成解树，输出到独立目录：

```sh
python yaik.py marvin generate --output-dir build/marvin_m6
python yaik.py marvin check --output-dir build/marvin_m6
```

输出目录中已有解树时使用它们，否则使用 gallery 中的解树。
`--tree-dir PATH` 可指定解树来源，`--urdf-dir PATH` 可指定包含
`left_arm.urdf` 和 `right_arm.urdf` 的目录。两侧模型均匹配后才发布生成结果，
输入不匹配或代码生成失败时保留已有解文件，也保留输出目录中的其他文件。
导出保留可验证的原始搜索种子；无匹配来源记录时为 `null`。

需要重新进行耗时的符号求解时，明确添加 `--solve`：

```sh
python yaik.py marvin generate --solve --seed 42 \
  --output-dir build/marvin-regenerated
```

`--seed` 只控制新搜索；快速导出不会重新采样。`--models-only` 只导出 Modified-DH 模型。
Marvin 的 URDF 转换流程针对这套手臂拓扑，带限位几何接口对应仓库中的 V4.0 几何。

## 其他机器人

```sh
python yaik.py models
python yaik.py generate --builtin puma --seed 42 --output-dir build/puma
python yaik.py generate --model path/to/robot.yaml --output-dir build/my_robot
python yaik.py codegen gallery/puma/puma_ik.yaml --output-dir build/puma
```

`generate` 输入机器人模型，进行离线符号求解；`codegen` 输入已有的 `*_ik.yaml` 解树，
直接生成代码。`--language python/cpp/both` 选择输出语言，默认生成两种语言。
模型格式见 [Modified-DH 说明](./docs/modified_dh.md) 和
[内置模型](./fk/robot_models.py)。模型 YAML 使用 `name/dh_params/unknowns`，
解树使用 `robot/tree_nodes`，CLI 会提示两种输入混淆。

成功搜索会输出模型、数值样本、中间主分支解树、最终解树和代码。
`--test-cases PATH` 可复用样本。种子范围为 `0` 到 `2**32 - 1`；相同依赖和种子有助于复现，
符号计算的进程超时仍可能影响最终选中的候选。

生成的 Python 代码需要仓库的 `python_run_import.py` 位于 `PYTHONPATH`；
生成的 C++ 代码需要仓库的 `yaik_cpp_common.h` 和 Eigen3 位于 include 路径。
原有 Python 库 API 保留；`ik_solve.py`、`py_codegen.py` 和 `cpp_codegen.py`
也支持显式传入模型或解树，具体例子见 [英文说明](./README.md)。

## 返回码、错误和限制

| 返回码 | 含义 |
| --- | --- |
| `0` | 操作成功；IK 至少找到一个通过检查的代表解 |
| `1` | IK 未找到通过检查的代表解，或验证失败；检查 JSON/报告 |
| `2` | 参数、输入、依赖或文件访问错误 |
| `130` | 用户中断 |

进度和错误输出到 stderr，FK/IK 的 stdout 保持 JSON 格式。
需要异常堆栈时使用 `python yaik.py --debug ...`。

带限位接口要求位置误差不超过 `1e-7 m`、旋转误差不超过 `1e-6 rad`，保持 J5 输入不变。
腕轴连续解族使用 33 个确定的 J7 采样代表，有限测试和采样不能证明全工作空间完备性，
空结果也不证明目标不可达。碰撞、轨迹与动力学约束仍需应用层处理。
标准 `*_ik_generated` 接口保留原先的较宽矩阵误差阈值和无关节限位过滤行为，
奇异覆盖率与带限位接口分开统计。

通用生成器现在补全“方程系数同时为零”的退化分支：从完整 FK 方程重新选择
解析公式；较大的退化系统还会在一个旋转关节范围内枚举 33 个代表值，再由生成接口
进行 FK 校验。左右臂保存的解树和生成的 Python/C++ 文件均包含这些分支。
连续解族仍采用有限代表值，不能据此保证全工作空间完备性。

## 验证

```sh
python -m pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  test_cli.py gallery/marvin_m6 solver/test_regressions.py \
  solver/test_codegen_safety.py solver/test_equation_branches.py -q

python yaik.py marvin validate --samples 1000 --seed 20261103 \
  --output build/marvin-validation.json
```

没有 C++ 工具链时加 `--python-only`；Eigen 位于其他位置时指定 `--eigen-dir`。
验证直接解析 URDF 计算 FK，检查每个返回解，并分别记录标准生成接口和带限位接口。
退出状态要求生成接口和带限位接口对每个已知可达目标均保留至少一个满足严格 FK、
关节限位和固定 J5 的解，并通过各自的负例检查及全部 FK 检查。JSON 报告记录输入哈希、种子、
失败案例和误差上限。左右臂的带限位 C++ 接口复用一次编译。
