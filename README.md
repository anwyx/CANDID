# CANDID — 多人协作的组件化采集架构（v1 初稿）

> **状态：** 已完成 Python 代码与无仿真依赖的单元测试。**未在你的 ManiSkill Docker 中执行集成测试**。不要直接开始上千条轨迹的采集，先完成文末 Pilot Checklist。

## 1. 各模块职责（不要交叉）

```text
CANDID/
├── envs/                              # 👥 组员只修改场景声明
│   ├── candid_baseline_env.py        # 4 cubes
│   ├── candid_ycb_fruit_env.py       # strawberry + 4 YCB
│   └── candid_primitive_env.py       # box / cylinder / sphere
├── planners/                         # 动作策略插件；不碰记录逻辑
│   └── execute_baseline.py          # 改为支持 reset_env=False
├── scripts/                          # 仅 CLI 入口，保持简短
│   ├── run_baseline.py
│   ├── check_scene.py
│   └── validate_dataset.py
├── src/candid/
│   ├── sim/
│   │   ├── spec.py                   # SceneSpec、ObjectSpec、XYRegion
│   │   ├── actors.py                 # ActorFactory: primitives/YCB + geometry
│   │   ├── layout.py                 # seeded continuous domain randomization
│   │   └── environment.py            # 统一 ManiSkill env / Settling / State obs
│   ├── data/
│   │   ├── monitor.py                # PhysicsTrace: GT pose/linear/angular velocity
│   │   ├── metrics.py                # pure continuous DR calculations
│   │   ├── storage.py                # SceneStore + TrajectoryStore + JSONL manifest
│   │   ├── adapter.py                # named model observations + max 6 + mask
│   │   ├── windows.py                # sliding window; DP/Predictor supervision
│   │   └── validate.py               # dataset integrity check
│   └── runner/
│       └── collect.py                # CLI args, reset/restore, planner orchestration
└── tests/
    └── test_contract.py
```

**唯一职责原则：**

- `envs/*.py`：填入物体（类型、ID、真实尺寸、颜色）、固定位置、随机化位置范围。仅注册一个 `CANDIDTabletopEnv` 子类。
- `src/candid/sim/`：所有组员共享的物理环境基础设施。YCB 的真实碰撞网格尺寸自动测量；随机布局避开物体间起始穿模；先 Settling，才可开始 Episode。
- `planners/*.py`：只执行动作，不自己 reset，也不保存 GT、DR、H5。
- `src/candid/data/monitor.py`：只监听 `env.reset/step`，从标准 Actor Registry 获取真实物理状态；**不注入到 DP observation**。
- `src/candid/data/metrics.py`：连续 DR 的唯一公式来源（计算 Episode 或任意 Chunk 时直接复用）。
- `src/candid/data/storage.py`：只管保存文件、JSONL 和 Scene Snapshot。
- `src/candid/runner/collect.py`：只编排 Scene × Trajectory；不定义物体，不写指标公式。

## 2. 组员需要写什么？

示例：`envs/candid_primitive_env.py`。它只包含以下内容：

```python
SCENE = SceneSpec(
    name="primitive_mixed_shapes",
    target=ObjectSpec("target_cube", "box", TARGET_ZONE, (-.12, 0), size=(.04,.04,.04)),
    distractors=(
        ObjectSpec("tall_box", "box", CLUTTER_ZONE, (-.15,.115), size=(.035,.035,.10)),
        ObjectSpec("cylinder", "cylinder", CLUTTER_ZONE, (-.01,-.115), size=(.045,.045,.08)),
    ),
    goal_region=GOAL_ZONE, goal_fixed_xy=(.1,0),
)

@register_env("CANDIDPrimitive-v1", max_episode_steps=500)
class CANDIDPrimitiveEnv(CANDIDTabletopEnv):
    SCENE = SCENE
```

说明：

- `XYRegion` 是连续 **均匀采样**区域，`fixed_xy` 仅在 `--fixed-scene` 时使用。
- `ObjectSpec.kind` 已支持 `box`, `cylinder`, `sphere`, `ycb`；自定义 Mesh / L 形需要在 `src/candid/sim/actors.py` **扩展一次**物体工厂，之后所有组员共用。
- 每个新环境应注册**新的 Env ID**，不要覆盖别人的注册名。
- 如果 Target 不再是普通方块，**Scene 注册与记录不受影响**；但旧 xArm6 Planner 的抓取方法可能失效，应单独提供一个适合该 Target 的 Planner 插件。

## 3. Scene → 多个 Trajectory，Runner 严格控制

```text
Scene ID (seed=42)
   │
   ├─ build objects / randomize XY+yaw
   ├─ physics settling (up to a fixed timeout, must be stable)
   ├─ SAVE stable initial simulator state in scenes/<scene_id>.npz
   │
   ├─ traj_000: Plan A; execute; RecordEpisode(HDF5); PhysicsTrace(NPZ); metadata
   │
   ├─ restore EXACT saved state (not re-sample, not re-settle)
   ├─ traj_001: Plan B; execute; RecordEpisode(HDF5); PhysicsTrace(NPZ); metadata
   │
   └─ restore EXACT saved state
      traj_002: Plan C ...
```

**关键改变：** 旧版 Planner 的 `solve()` 会调用 `env.reset()`。提供的新 `planners/execute_baseline.py` 增加 `reset_env=False` 参数；Runner 先 reset，再调用 `solve(..., reset_env=False)`。新 Planner 也必须遵守这个接口，否则 Runner 会拒绝运行。

每次恢复时检查 Target、Goal、Distractor 与 Robot QPos 是否回到同一初始状态。Scene 物理状态存储使用无 pickle 的 NPZ；相同 ManiSkill / asset / 控制配置才能可靠重放。

## 4. 数据 Contract v1

### 原始数据在磁盘上的样子

```text
demos/pilot_run/
├── trajectories_<run_id>.h5         # ManiSkill RecordEpisode: obs, actions, success ...
├── trajectories_<run_id>.json       # ManiSkill 录制元数据
├── scenes.jsonl                     # 一行一个 Scene
├── trajectories.jsonl               # 一行一个 Trajectory, 含 HDF5 group/NPZ 相对路径
├── scenes/
│   └── <scene_id>.npz               # 稳定后的完整初始模拟器状态（可恢复）
└── disturbance_traces/
    ├── <trajectory_id>.npz
    └── ...
```

每个原始 Pose NPZ 保存：

| Field | Shape | 说明 |
|---|---|---|
| `distractor_poses` | `[T+1,N,7]` | `[xyz, qwxyz]`，单位 m、世界坐标 |
| `distractor_linear_velocities` | `[T+1,N,3]` | m/s |
| `distractor_angular_velocities` | `[T+1,N,3]` | rad/s |
| `object_sizes_m` | `[N]` | 碰撞网格 AABB 最大边长作为归一化尺寸 |

从 `state_0` 经 `action_0` 到 `state_1`，因此 **T 条 Action 对应 T+1 帧状态**。Scene Metadata 有 Asset IDs、真实几何尺寸、初始稳定性、初始姿态、control_dt_s 和版本信息。

**环境必须使用 `--obs-mode state_dict` 采集训练数据**。ManiSkill 的 `state` 会把整个状态打平成匿名向量，不利于多人协作维护字段；`state_dict` 在 HDF5 中保留 `obs/agent/...` 与 `obs/extra/...` 的字段名。`--obs-mode none` 仅与 `--no-record` 同用做调试。

### DP 与 Predictor 输入共用 Adapter

`src/candid/data/adapter.py::canonical_observation()`：

| 规范键名 | 形状（单步） |
|---|---|
| `robot_qpos`, `robot_qvel` | `[J]` |
| `tcp_pose` | `[7]` |
| `target_state` | `[16] = pose7 + linear3 + angular3 + dimensions3` |
| `distractor_states` | `[6,16]` |
| `distractor_mask` | `[6]` |
| `goal_xy` | `[2]` |

**Vanilla DP：** 最近 2 帧的同一输入 → 未来 12 步 `[12,A]` Action；只使用成功演示。

**Predictor：** 最近 2 帧的同一输入 + `[12,A]` Action → chunk 扰动标签（每物体 peak translation / rotation / per-object DR，整体 dr_mean、dr_max）；允许成功/失败样本。

`src/candid/data/windows.py` 生成滑动窗口。Chunk 的参考状态是 `state[t]`，标签使用 `state[t:t+12]`，**绝不把未来 GT 状态放入 Predictor 输入**。`metrics.py` 是唯一 DR 计算方法，默认权重 0.7/0.3，不使用接触的二值指标。Topple/Fall 目前仅保留原始轨迹供后续统一、形状感知的事件判定，不作为 v1 必需字段。

**注意：** `pose7` 仅描述物体位姿，不足以表示任意复杂 YCB 几何；Scene 元数据保存 `asset_id`。未来可增加 mesh/shape embedding，原始数据无需重采。

## 5. 常用命令（Docker 的 `/workspace/CANDID` 根目录）

**A. 运行一个 Baseline 场景：**

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 1 --seed 0 --output-dir demos/pilot_baseline
```

**B. 同一个 Scene 生成 3 条不同 Planner 轨迹：**

```bash
python scripts/run_baseline.py \
  --env-module envs.candid_baseline_env \
  --env-id CANDIDBaseline-v1 \
  --scenes 5 --seed 42 \
  --plan-variants '[{"lateral_offset":0.0},{"lateral_offset":0.04},{"lateral_offset":-0.04}]' \
  --output-dir demos/pilot_multi
```

默认 `--trajectories-per-scene` 根据 `--plan-variants` 的长度决定。本命令得到 5 个 Scene，每个 3 条，共 15 条 Trajectory（若 Planner 失败也保存其真实执行段）。**单独重复相同 Planner 参数不能保证不同轨迹**，请各自设计真正不同的轨迹策略。

**C. 运行组员的 Primitive Scene：**

```bash
python scripts/run_baseline.py \
  --env-module envs.candid_primitive_env --env-id CANDIDPrimitive-v1 \
  --scenes 3 --output-dir demos/pilot_primitive
```

**D. 检查某场景没有机械臂动作时是否自行漂移：**

```bash
python scripts/check_scene.py \
  --env-module envs.candid_baseline_env --env-id CANDIDBaseline-v1 \
  --seed 42 --steps 100
```

**E. 校验 Scene / Trajectory / HDF5 / NPZ 的时间对应：**

```bash
python scripts/validate_dataset.py demos/pilot_multi
```

**F. YCB 文件依赖：**

```bash
python -m mani_skill.utils.download_asset ycb -y
```

YCB 的 Scene 初始化 / 记录接口可使用新版，但默认 `execute_baseline.py` 是为方块 Target 写的 Top-down grasp **不保证能抓草莓**。若要有效演示，添加 `planners/ycb_planner.py` 并通过 `--planner ... --planner-config-class ...` 指定；Planner 仍须接受 `reset_env=False`。

## 6. 三个人开始并行前的 Pilot Checklist

1. 使用新分支，例如 `git switch -c refactor/data-contract-v1`，先备份旧文件。复制包内 `src/`, `envs/`, `scripts/`, `planners/` 到仓库。**新环境注册 ID 是 `-v1`；不要同时使用旧版 `-v0` 当作同一个实验。**
2. 在 Docker 里测试 Baseline `--scenes 1` 与 `--scenes 1 --plan-variants ...`，检查视频与物体是否出现真实碰撞，而不是穿模。
3. 用 `check_scene.py` 测 20 个不同 Seed 的无动作漂移。若有非零 DR 明显过大，调整 `max_settle_steps`, `stable_steps` 或布局；不要改写 Disturbance 为零来掩盖自然运动。
4. 用 `validate_dataset.py` 检查至少一个 Scene 的 3 条轨迹。确认 HDF5 `actions` 长 T、`obs` 和 NPZ 长 T+1。
5. 验证相同 Scene 三条轨迹的初始 GT 状态一致，但由于 Planner 设置不同，后续动作确实有所不同。
6. 确认 YCB 数据目录存在，并验证新的 YCB Scene 可初始化；之后再开发适用的抓取 Planner。
7. 先让每个组员生成 5–10 条数据，每人单独一个 `--output-dir`。**不要多人同时写同一个 HDF5 或 JSONL**。校验通过再同步到共享存储。
8. 为正式数据集锁定 ManiSkill Commit、物体资产版本、Action Space `pd_joint_pos`、Control DT、DR 权重、`state_dict` 字段规范和 Object Size 规则。按 Scene ID 而不是轨迹随机切 Train/Val/Test。

## 7. 当前明确没有解决的问题

- 本包未在用户 Docker 实机/物理仿真环境运行，只有模块语法检查、采样、时间对齐、文件读写与 Adapter 的单元测试。ManiSkill 不同版本可能要求适配少量 API。
- 该采集器先面向 **单场景 CPU + MPlib**，不支持 GPU 并行 Vector Env 或多人写同一数据目录。
- 默认 Planner 的 grasp 适用于 Target Cube；YCB Target、复杂 Mesh 等仍需专用 Planner。
- ActorFactory 支持基本 Primitive/YCB，不含通用 Mesh 的自动碰撞/惯量设置；后续仅扩展工厂一次。
- 初始布局使用保守 footprint circle 防穿模，并不保证 IK 可达，部分场景/抓取会失败；采集器如实标记任务失败。自适应 Settling 也不能保证场景在后续 100 步完全零漂移，请运行无动作测试。
- 事件级 Topple / Fall 不能用所有物体统一的 z 角度或桌面中心高度简单推断，需要下一阶段确定 shape-aware 标注标准。

**核心工程约束：** 三个组员不应该编辑 Runner、Metrics、DataStore 或模型 Input Adapter。确实需要新类型物体时，由团队合并一次 `ActorFactory` 扩展，避免分叉。
