# CANDID `src/`：公共组件说明

本目录存放 CANDID 的**公共仿真、数据采集与运行组件**。我们的目标是让组员新增任务时，主要在项目根目录的 `envs/` 中声明物体与生成范围，而不需要为每一个场景重新实现 ManiSkill Actor、随机布局、Settling、Disturbance 记录或数据保存。

> **当前版本：组件化数据采集 v1（含 Pose-based Settling 修复）。** 默认面向 ManiSkill 3、xArm6 + Robotiq、`pd_joint_pos`、单环境 CPU/MPlib 采集。这个目录不是训练好的 DP/Predictor 模型；它主要负责提供二者共用的原始数据与数据格式。

## 1. 整体结构

```text
src/
└── candid/
    ├── sim/                    # 场景描述、物体创建、布局与物理初始化
    │   ├── spec.py
    │   ├── actors.py
    │   ├── layout.py
    │   ├── settling.py
    │   └── environment.py
    ├── data/                   # 数据记录、指标、存储与训练样本接口
    │   ├── monitor.py
    │   ├── metrics.py
    │   ├── storage.py
    │   ├── adapter.py
    │   ├── windows.py
    │   └── validate.py
    └── runner/
        └── collect.py         # Scene × Trajectory 的采集流程和命令行参数
```

三个子模块的职责不应混在一起：

- **`sim/`**：决定场景里有什么、Actor 怎么创建、物体如何随机生成以及何时可以开始一条 Episode；**不负责把轨迹写入数据集**。
- **`data/`**：记录已经执行出来的物理状态，计算和保存 Disturbance，以及将原始数据转换成模型训练样本；**不负责生成布局或规划机械臂动作**。
- **`runner/`**：串联 Environment、Planner、Recorder 和 Data 组件；**不自己定义物体或重新实现 DR 公式**。

组员实际运行命令请看 [`../scripts/README.md`](../scripts/README.md)。

## 2. `sim/`：公共仿真组件

| 文件 | 主要内容 | 一般谁会修改 |
|---|---|---|
| `spec.py` | `XYRegion`、`ObjectSpec`、`SceneSpec`：定义场景的数据结构 | 新增场景时通常只**调用**，无需修改 |
| `actors.py` | `build_actor()` 创建 box/cylinder/sphere/YCB；`describe_actor()` 从碰撞几何提取尺寸、落地高度和占地半径 | 需要增加新的 Mesh/物体类型时由团队统一扩展 |
| `layout.py` | `sample_layout()`：基于 Seed 连续随机采样 XY/yaw，避免初始物体重叠并预留 Target/Goal 空间 | 需要统一修改随机化规则时修改 |
| `settling.py` | `SettlingConfig`、`settle_scene()`：基于 Pose 变化和无动作验证判断物体是否稳定 | 统一维护 Settling 策略 |
| `environment.py` | `CANDIDTabletopEnv`：ManiSkill 环境基类，处理 Robot、Table、Actor、Reset、Settling、Observation、Reward 和 Success | 需要增加**新的任务机制**时修改 |

### 2.1 新建场景：优先写 `envs/*.py`

假设需要一个 Target Cube，旁边有 Box 和 Cylinder。新建 `envs/candid_my_scene_env.py`：

```python
from mani_skill.utils.registration import register_env
from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion

TARGET = XYRegion((-0.145, -0.095), (-0.035, 0.035))
CLUTTER = XYRegion((-0.235, 0.145), (-0.175, 0.175))
GOAL = XYRegion((0.080, 0.120), (-0.040, 0.040))

SCENE = SceneSpec(
    name="my_scene",
    target=ObjectSpec(
        "target_cube", "box", TARGET, (-0.12, 0.0),
        size=(0.04, 0.04, 0.04), color=(1.0, 0.1, 0.1, 1.0),
    ),
    distractors=(
        ObjectSpec("box_1", "box", CLUTTER, (-0.15, 0.11),
                   size=(0.04, 0.04, 0.08)),
        ObjectSpec("cylinder_1", "cylinder", CLUTTER, (-0.02, -0.11),
                   size=(0.04, 0.04, 0.08)),
    ),
    goal_region=GOAL,
    goal_fixed_xy=(0.10, 0.0),
)

@register_env("CANDIDMyScene-v1", max_episode_steps=500)
class CANDIDMySceneEnv(CANDIDTabletopEnv):
    SCENE = SCENE
```

`ObjectSpec` 当前支持：

- `box`：`size=(长, 宽, 高)`，单位为 m。
- `cylinder`：`size=(直径, 直径, 高)`，单位为 m。
- `sphere`：`size=(直径, 直径, 直径)`，单位为 m。
- `ycb`：填写 `asset_id`（例如 `"013_apple"`）；尺寸由碰撞网格自动读取，不要求填写 `size`。

`XYRegion` 为连续均匀采样的 XY 区间；`fixed_xy` 与 `fixed_yaw` 是 `--fixed-scene` 时的固定布局。物体 ID 在一个 Scene 中必须唯一，当前 `SceneSpec` 支持 **1–6 个 Distractors**。

随机布局会检查保守的二维占地圆间距，但**不保证 MPlib 一定能到达目标**。复杂 YCB 的实际抓取也可能需要新的 Planner。新环境注册名应唯一，且别忘了运行时传入 `--env-module` 和 `--env-id`。

### 2.2 Settling 在哪里完成？

`environment.py` 的 `reset()` 内调用 `settling.py`：

1. 按当前 Seed 生成布局，放置所有动态 Actor，并初始化 Robot。
2. 在外部 Recorder/Monitor 开始记录前，运行**不计入 Episode** 的物理步。
3. 检查最近一段时间的最大平移和旋转，再额外进行无动作验证。
4. 通过后返回真实稳定状态的 Observation；不通过则抛错，拒绝记录不可靠 Scene。

默认使用最近 15 个控制步的 Pose 窗口（最大平移 0.7 mm / 旋转 1°），再验证 30 个无动作控制步（最大平移 1.0 mm / 旋转 1.5°）；相关参数在 `CANDIDTabletopEnv` 的构造函数中，可通过 Runner 的 `--env-kwargs` 传入。旧的瞬时速度阈值参数为兼容保留，**不是当前 Settling 通过与否的依据**。

同一 Scene 的第二条及后续轨迹会恢复保存的仿真状态，**不再重新随机化，也不再重复 Settling**。

## 3. `data/`：统一数据记录与整理

| 文件 | 公开接口或核心功能 | 生成/处理的数据 |
|---|---|---|
| `monitor.py` | `PhysicsTrace`，包装环境的 `reset()`/`step()` | 每一步的 Distractor Pose、线速度、角速度 |
| `metrics.py` | `disturbance_from_poses()`、`angle_between_wxyz()` | 每个物体的 Peak/Residual Translation、Rotation；DR Mean/Max |
| `storage.py` | `DatasetWriter`、`freeze_state()`、`save_state_npz()`、`load_state_npz()` | 稳定 Scene Snapshot、Trace NPZ、Scene/Trajectory JSONL 索引 |
| `adapter.py` | `canonical_observation()` | 将 HDF5 `obs` 转成 Vanilla DP / Predictor 共用的命名 State |
| `windows.py` | `windows()`、`prepare_window()` | 根据完整轨迹动态切 Observation History、Action Chunk、Predictor 标签 |
| `validate.py` | `validate_dataset()` | 检查 Scene/Trajectory 关联、文件存在、张量形状、T/T+1 对齐 |

### 3.1 `PhysicsTrace` 记录什么？

一个 Episode 有 `T` 条 Action 和 `N` 个 Distractors，Trace 中至少有：

| NPZ Key | Shape | 说明 |
|---|---|---|
| `distractor_poses` | `[T+1, N, 7]` | 世界坐标系 `[x,y,z,qw,qx,qy,qz]` |
| `distractor_linear_velocities` | `[T+1, N, 3]` | 线速度，m/s |
| `distractor_angular_velocities` | `[T+1, N, 3]` | 角速度，rad/s |
| `object_sizes_m` | `[N]` | DR 平移项的特征尺寸（碰撞网格 AABB 最大边长） |

`T+1` 包含 Settling 结束后的第一帧，以及每一条动作执行后的状态。`PhysicsTrace` 通过公共的 `distractor_actors` 列表读取 Actor，组员**不需要在 Environment 里自己编写记录逻辑**。

目前 Trace 负责保存上述 Distractor 运动数据；Robot、Target 的时序数据随 ManiSkill `RecordEpisode` 的 `obs` 保存到 HDF5，Scene 初始状态在独立 Snapshot 中保存。

### 3.2 Disturbance 如何计算？

以某个参考状态 `start` 为起点，统计 `[start, stop]`（**包括 stop**）期间的峰值位移与旋转：

\[
D_i = 0.7\frac{\max_u \|p_i^u-p_i^{start}\|_2}{s_i}
    +0.3\frac{\max_u \theta(q_i^u,q_i^{start})}{\pi}
\]

`dr_mean` 是各物体 `D_i` 的平均值；`dr_max` 是最大值。还会保存独立的 `peak_translation_m`、`residual_translation_m`、`peak_rotation_rad`、`residual_rotation_rad` 和 `per_object_dr`。

- **Episode DR**：`start=0`，`stop=T`。
- **Chunk DR**：`start=t`，`stop=t+H`；参考点是 `state[t]`，不是整个 Episode 的初始状态。

这里只计算连续扰动分数，不等同于接触次数、碰撞力或真正的发生率。Topple/Fall 的统一几何判定尚未纳入 v1 的必需标签。

### 3.3 DP / Predictor 共用的 State 接口

正式采集使用 `--obs-mode state_dict`，HDF5 中保留 `obs/agent/...` 和 `obs/extra/...` 的具名字段。`adapter.py` 会转换出以下单时刻特征：

| Key | Shape | 说明 |
|---|---|---|
| `robot_qpos` / `robot_qvel` | `[J]` | Robot 关节位置/速度 |
| `tcp_pose` | `[7]` | TCP Pose |
| `target_state` | `[16]` | Pose 7 + Linear Velocity 3 + Angular Velocity 3 + Dimensions 3 |
| `distractor_states` | `[6, 16]` | 最多 6 个物体，不足使用零 Padding |
| `distractor_mask` | `[6]` | 有效物体标记（1=有效、0=Padding） |
| `goal_xy` | `[2]` | Goal 的 XY 位置 |

`windows.py` 的当前默认值是 Observation Horizon = 2、Action Horizon = 12：

- **Vanilla DP**：输入状态历史，监督目标为未来 12 步**实际执行的 Action**。
- **Disturbance Predictor**：输入相同的状态历史及候选 12 步 Action，监督目标是该 Chunk 真实造成的连续 DR。

数据切片必须遵守 `actions[t:t+12]` 对应 `states[t:t+13]`，不能把用于计算标签的未来真实 Pose 提前放进 Predictor 输入。当前 Adapter/Window 是数据接口函数，**不是已经训练好的 Dataset/DataLoader 或模型**。

## 4. `runner/`：Scene × Trajectory 采集编排

`runner/collect.py` 提供参数解析和采集主逻辑，由根目录下的 `scripts/run_baseline.py` 调用。

```text
给定 Env ID / SceneSpec / Planner
    ↓
创建一个随机 Scene → 完成 Settling → 保存稳定初始状态
    ↓
Trajectory 0：执行 Planner → RecordEpisode(HDF5) + PhysicsTrace(NPZ)
    ↓
恢复同一份 Scene Snapshot（不再随机化）
    ↓
Trajectory 1：更换 Planner 参数 → 记录第二条轨迹
    ↓
继续下一 Scene / 下一个 Seed
```

Runner 的其他约束：

- 当前仅支持 `num_envs=1` 的采集方式；不要假定已支持 GPU 向量化并行。
- Planner 的 `solve()` **必须接受** `reset_env=False`，因为只有 Runner 可以决定何时 Reset/Restore；Planner 不应自己记录或保存数据。
- 同一 Scene 的多条 Trajectory 使用相同初始 Snapshot；Runner 会检查恢复后的关键初始状态一致性。
- Planner 失败也可以保留已执行的动作与物理后果；成功与否写入索引，而不是和 DR 混为一谈。
- 默认 `control_mode=pd_joint_pos`；记录的是控制器实际收到的 Action，不能事后当作 TCP Delta Pose。

## 5. 生成数据的位置与格式

```text
demos/<run_name>/
├── trajectories_<run_id>.h5          # ManiSkill：obs / actions / success 等
├── trajectories_<run_id>.json        # ManiSkill Recorder 元数据
├── scenes.jsonl                      # 一行一个 Scene，指向 Snapshot
├── trajectories.jsonl                # 一行一个 Trajectory，指向 HDF5 Group 和 Trace
├── scenes/
│   └── <scene_id>.npz                # Settling 后的完整模拟器初始状态
└── disturbance_traces/
    └── <trajectory_id>.npz          # 每一步 Distractor 的 Pose / Velocity
```

文件之间通过唯一的 `scene_id`、`trajectory_id` 和 `hdf5_trajectory_key` 关联。大规模数据请存到团队共享数据目录，**不要把 `demos/`、`.h5`、`.npz`、视频等提交到 Git**。

## 6. 组员修改规范

1. **新增一种物体组合**：在 `envs/` 新建一个 `SceneSpec`，无需修改 `src/`。
2. **使用新的物理形状/网格**：统一扩展 `sim/actors.py` 及必要的 `ObjectSpec` 定义，并加测试；不要在每个 Env 复制 Actor 创建代码。
3. **修改 DR 公式**：只改 `data/metrics.py` 并同步测试、更新版本说明，确保 Episode DR 和 Chunk DR 一致。
4. **修改保存格式**：只改 `data/storage.py` / `monitor.py` / `validate.py` 的统一契约，避免各组输出不同字段。
5. **增加新的 Planner**：在根目录 `planners/` 新建实现，保持 `solve(..., reset_env=False)` 接口。
6. **处理场景不可达/无法稳定**：优先调整 SceneSpec 布局或公共组件；不要把真实扰动强制写成零，也不要绕过 Settling。
7. **合并数据前**：运行 `scripts/check_scene.py` 与 `scripts/validate_dataset.py`；Train/Val/Test 应按 Scene 划分，避免同一个 Scene 的多个 Trajectory 跨集合泄漏。

已验证的 Baseline 测试说明基本采集流程可运行，但不同 Env 和新的 Planner 仍需各自验证。完整运行命令请查看 [`../scripts/README.md`](../scripts/README.md)。
