# CANDID — 低扰动机器人操作数据采集

> **团队默认入口：** `python scripts/collect.py`。通常只需要指定 Scene 数量、随机种子和输出目录，不需要选择采样模式或人工填写 XYZ 偏移。

## 1. 项目简介

CANDID（本仓库）基于 **ManiSkill 3 / SAPIEN** 构建带有可移动干扰物（Distractors）的桌面 Pick-and-Place 环境。机器人为 **xArm6 + Robotiq**：抓取目标物体并移动至 Goal，同时记录其操作过程中对其他物体造成的物理扰动（Disturbance）。

项目关注的不是绝对无碰撞：轻微接触可以存在，但我们希望区分不同动作带来的扰动，并为后续 **Diffusion Policy（DP）** 和 **Disturbance Predictor** 准备可复用的数据。当前仓库主要负责**仿真环境、规划示范、数据采集、扰动标签和数据接口**，并不包含已经训练好的 DP 或 Predictor。

默认数据采集采用：

- 每个 Scene 随机生成一次 Target、Goal 和 Distractors 的初始布局，完成物理 Settling。
- 保存稳定状态的完整模拟器 Snapshot，同一个 Scene 的每条 Trajectory 都从这一状态恢复。
- 使用 MPlib / ManiSkill xArm6 Motion Planning Solver 执行 Pick-and-Place，自动生成 Early Approach / Lift 等阶段的不同 XYZ Waypoint。
- 前 13 条轨迹用于固定探索；之后结合已执行轨迹的成功结果和 DR 标签，自动选择 Feedback 或 Uniform 候选。
- 所有尝试独立记录成功与否、HDF5 动作/观测、NPZ 物理轨迹和 Disturbance 指标。**失败轨迹同样保留**。

目前支持单环境 CPU 采集流程（`num_envs=1`），默认 `control_mode=pd_joint_pos`、`obs_mode=state_dict`。

## 2. 快速开始：组员只需运行这几条命令

在**已配置好 ManiSkill/SAPIEN/MPlib 的项目环境**中，进入 CANDID 根目录（Docker 中通常为 `/workspace/CANDID`）。本仓库当前没有统一锁定的依赖安装清单，请团队统一现有的可运行 Docker 镜像、Python 环境及资产版本。

```bash
cd /workspace/CANDID

# 从 GitHub 更新正式版本
git pull origin main

# 可选：快速确认依赖
python -c "import mani_skill, sapien, numpy, h5py; print('Core imports OK')"

# 采集 50 个 Scene；每个 Scene 默认 24 条轨迹
python scripts/collect.py \
  --scenes 50 \
  --seed 42 \
  --output-dir demos/team_a_batch01
```

**不需要** `--trajectory-mode`、`--plan-variants`、`--lateral-offset` 或任何手写的 XYZ 参数。脚本默认自动使用已集成的 Feedback + Uniform 采样流程。

采集结束后查看结果：

```bash
# 检查 Scene / Trajectory / HDF5 / NPZ 的基本结构和时间长度
python scripts/validate_dataset.py demos/team_a_batch01

# 汇总成功率、DR 范围、Topple/Fall 和各 Scene 的覆盖度
python scripts/analyze_collection.py demos/team_a_batch01
```

采集脚本自己也会在正常完成全部 Scene 后运行一次 `validate_dataset()`，并写出 `collection_summary.csv`。独立执行验证命令仍便于组员交付数据时复核。

### 多位组员如何分配任务

每个人使用**不同的 Scene Seed 范围和输出目录**，例如：

```bash
# 组员 A
python scripts/collect.py --scenes 50 --seed 42 --output-dir demos/team_a

# 组员 B
python scripts/collect.py --scenes 50 --seed 10042 --output-dir demos/team_b

# 组员 C
python scripts/collect.py --scenes 50 --seed 20042 --output-dir demos/team_c
```

`--seed` 是第一个 Scene 的 Seed；每个后续 Scene 使用 `seed + scene_index`。相同环境与相同 Seed 有利于重现实验设置，但底层 MPlib/物理执行的完整逐位一致性不应仅凭 Seed 假定。

> **不要多个进程同时写同一个输出目录。** `scripts/collect.py` 会拒绝向已经包含 `scenes.jsonl` 或 `trajectories.jsonl` 的目录继续追加数据。一次新的采集请使用新的目录；中断后不要简单地对原目录重跑同一命令。

## 3. 常用参数

正式采集入口是 `scripts/collect.py`：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--scenes` | `1` | 生成多少个独立 Scene |
| `--trajectories-per-scene` | `24` | 每个 Scene 进行多少次独立轨迹尝试；当前最少 `14` |
| `--seed` | `42` | 第一个 Scene 的布局随机种子 |
| `--output-dir` | 自动生成唯一的 `demos/candid_*` 目录 | 数据输出目录；建议团队显式指定 |
| `--save-video` | 关闭 | 额外保存视频（依赖渲染环境） |
| `--variant-seed` | `2026` | 轨迹候选采样种子；一般不改 |
| `--variant-scale` | `1.0` | XYZ 偏移幅度系数，允许 `(0, 1.5]`；一般不改 |
| `--max-episode-steps` | `700` | 单条 Episode 的环境最大步数 |
| `--env-id` | `CANDIDBaseline-v1` | 要运行的 Gym 环境 ID |
| `--env-module` | `envs.candid_baseline_env` | 负责注册该环境的 Python 模块 |
| `--env-kwargs` | `{}` | 传入环境构造函数的 JSON 参数 |
| `--verbose` | 关闭 | 打印规划阶段详细信息 |

查看完整选项：

```bash
python scripts/collect.py --help
```

例如，只想做一轮最小 Feedback 冒烟测试：

```bash
python scripts/collect.py \
  --scenes 1 \
  --trajectories-per-scene 14 \
  --save-video \
  --output-dir demos/final_smoke
```

14 条对应 **13 条固定 Warmup + 1 条后续候选**；标准 24 条则对应 **13 Warmup + 9 Feedback + 2 Uniform**。`--save-video` 用于少量调试，批量采集通常不打开。

## 4. Scene 与 Trajectory 是如何生成的？

### 4.1 Scene 生成与初始状态固定

1. 根据 SceneSpec 在允许的区域内采样 Target、Distractors 和 Goal 的 **XY 位置与 yaw**；采样时检查初始物体间距。
2. `CANDIDTabletopEnv.reset()` 完成物理 Settling，确保开始记录前物体基本稳定。
3. 保存完整的模拟器状态 Snapshot，包括机器人及物体状态。
4. 运行第一条轨迹；之后恢复同一份 Snapshot，再执行下一条。恢复时不会重新随机化布局或额外 Settling。
5. Runner 对重要的初始位置、姿态和机器人状态进行一致性检查。

因此同一个 Scene 内的不同 Trajectory 应该具有相同起点，差异主要来自 Planner 的动作路径，而不是初始物体布局。

### 4.2 最终轨迹采样策略

`src/candid/runner/production_sampler.py` 负责调度，实际候选构造与评分由 `adaptive_sampling.py` 和 `feedback_sampler.py` 支持。

**前 13 条固定 Warmup：**

| 轨迹序号（从 0 起） | 策略 |
|---|---|
| 0 | Baseline |
| 1–2 | Repeat Baseline（衡量固定配置下的规划波动） |
| 3–8 | Early Approach 的 X/Y/Z 正负偏移，共 6 条 |
| 9–12 | Lift 的 X/Y 正负偏移，共 4 条 |

之后每次从一个由 Seed 决定的候选池中选择 Early、Lift 或少量 Combined 三维偏移。默认每 **5 条后续轨迹**中有 1 条 Uniform，其余走 Feedback；对默认的 11 条后续尝试，恰好是 **9 Feedback + 2 Uniform**。

- **Feedback：** 综合已完成的成功轨迹 DR、配置接近程度、新颖性和失败记录对候选进行排序。
- **Uniform：** 在同一候选池中均匀选择一个候选，用于保留探索能力；这不意味着 XYZ 坐标在整个工作空间中均匀采样。
- **Fallback：** 若初始 Baseline 未成功，不能可靠地计算相对 DR 目标，程序使用备用候选机制继续保留尝试，并标记选择来源，而不是假装 Feedback 已生效。

这是**两次完整 Trajectory 之间**的反馈采样，不是机器人运动到一半的在线 Replan，也不是训练好的 Predictor。每一次执行结束后才记录结果，并供同一 Scene 的后续选择参考。

### 4.3 Planner 具体改变了什么？

`planners/execute_baseline.py` 使用 MPlib 规划到若干 TCP Waypoint，然后在 ManiSkill 中执行：

- **Early Approach：** 在正式 Pre-grasp 前增加可偏移的 XYZ 中间 waypoint。
- **Lift：** 抓取后、抬升完成前增加 XYZ 中间 waypoint。
- **Transfer：** Planner 也支持搬运阶段的 XYZ/lateral 偏移，但当前默认候选池主要集中于 Early 与 Lift。
- 最终 Grasp 与 Place 目标保持不变；Planner 使用世界坐标系中的米（m）。

MPlib 不把 Distractors 作为严格的点云避障约束；场景中的真实物理接触仍然由仿真器处理。因此本项目允许出现轻微碰撞和失败，并以实际 Disturbance 为评价依据。

## 5. 目录结构与文件职责

```text
CANDID/
├── envs/
│   ├── candid_baseline_env.py       # 默认 4 个 Box Distractors
│   ├── candid_primitive_env.py      # Box / Cylinder / Sphere 混合场景
│   └── candid_ycb_fruit_env.py      # YCB 示例环境
├── planners/
│   └── execute_baseline.py         # MPlib xArm6 Pick-and-Place + XYZ Waypoints
├── scripts/
│   ├── collect.py                  # ★ 团队正式采集入口
│   ├── analyze_collection.py       # ★ 采集结果分析
│   ├── validate_dataset.py         # 数据格式验证
│   ├── check_scene.py              # 场景稳定性/空动作检查
│   └── run_baseline.py             # 旧的显式 Planner 变体调试入口
├── src/candid/
│   ├── sim/
│   │   ├── spec.py                 # SceneSpec / ObjectSpec / XYRegion
│   │   ├── actors.py               # Primitive 和 YCB Actor 构建
│   │   ├── layout.py               # 布局随机化
│   │   ├── settling.py             # 物理 Settling
│   │   └── environment.py          # 公共 ManiSkill Tabletop 环境
│   ├── runner/
│   │   ├── collect.py              # 旧 Runner 的通用工具/调试采集入口
│   │   ├── production_sampler.py   # 正式采样调度
│   │   ├── adaptive_sampling.py    # Warmup 和 XYZ 候选生成
│   │   ├── feedback_sampler.py     # 历史 DR 反馈候选排序
│   │   └── variants.py            # XYZ 参数校验和 Waypoint 工具
│   └── data/
│       ├── monitor.py             # PhysicsTrace，逐帧物体状态
│       ├── metrics.py             # DR + Topple / Fall
│       ├── storage.py             # Scene / Trajectory 持久化
│       ├── validate.py            # 数据契约校验
│       ├── adapter.py             # DP / Predictor 共用 Observation
│       └── windows.py             # Observation History / Action Chunk
└── tests/                           # 原有数据契约、事件和 Settling 测试
```

旧的 `scripts/run_baseline.py` 仍可用于单条 Baseline 或手动 Planner Variant 调试，但它**不是**正式 Feedback 采集入口；其中的旧参数和默认采集行为与 `scripts/collect.py` 不同。

## 6. 输出文件与数据契约

一次正式采集的目录大致如下：

```text
demos/team_a_batch01/
├── scenes.jsonl                         # Scene 索引、Seed、环境与采样配置
├── trajectories.jsonl                   # Trajectory 索引、任务状态、DR、Planner 配置
├── trajectories_<run_id>.h5             # RecordEpisode: Observation / Action
├── trajectories_<run_id>.json           # Recorder 元数据（如有生成）
├── collection_summary.csv               # 采集完成后的每 Scene 汇总
├── collection_analysis.csv              # 运行 analyze_collection.py 后生成
├── scenes/
│   └── <scene_id>.npz                   # 保存的初始模拟器 Snapshot
└── disturbance_traces/
    └── <trajectory_id>.npz              # 逐帧 Distractor 运动数据和事件标记
```

`scenes.jsonl` 每行是一条 Scene 记录；`trajectories.jsonl` 每行是一条尝试，包含 `scene_id`、`trajectory_id`、`task_success`、`planner_config`、`sampling_selection`、`planner_seed`、Disturbance 摘要及对应 HDF5/NPZ 的相对路径。两类文件通过 ID 关联。

### 6.1 时间对齐

假设某轨迹执行了 `T` 次 Action、有 `N` 个 Distractors：

| 数据 | 形状 | 说明 |
|---|---|---|
| HDF5 `actions` | `[T, A]` | 实际执行的 `pd_joint_pos` 控制 Action |
| HDF5 `obs` | `[T+1, ...]` | 包含动作前第一帧及每次动作后的状态 |
| NPZ `distractor_poses` | `[T+1, N, 7]` | `[x,y,z,qw,qx,qy,qz]` |
| NPZ `distractor_linear_velocities` | `[T+1, N, 3]` | m/s |
| NPZ `distractor_angular_velocities` | `[T+1, N, 3]` | rad/s |
| NPZ `object_sizes_m` | `[N]` | DR 归一化的物体特征尺寸 |
| NPZ `distractor_toppled` | `[T+1, N]` | Bool，每帧的 Topple 状态 |
| NPZ `distractor_fallen` | `[T+1, N]` | Bool，每帧的 Fall 状态 |

这里的基本约束是：**T 条 Action 对应 T+1 帧物理状态**。

### 6.2 Disturbance 标签

`metrics.py` 对每个 Distractor 计算相对于起始状态的最大位移和旋转：

\[
DR_i = 0.7\frac{\max_t\|p_i(t)-p_i(0)\|_2}{s_i} + 0.3\frac{\max_t\theta(q_i(t),q_i(0))}{\pi}
\]

整体 `dr_mean` 是所有物体 `DR_i` 的均值；`dr_max` 是最大值。还保留各物体的平移、旋转、残余位移及 `per_object_dr`。

当前 Topple/Fall 是简单可重复的几何事件标签：

- **Topple：** 相对于第一帧，物体局部向上轴的倾角超过 45°；Sphere 不计 Topple。
- **Fall：** 物体中心 Z 低于桌面平面 1 cm（默认桌面 Z=0）。
- `topple_count` / `fall_count` 统计在指定时间段内首次发生相应 False→True 变化的物体数，并非精细的接触事件次数。

这些指标描述几何后果，不等同于接触力、精确碰撞次数或复杂 YCB 物体的语义失稳判定。

### 6.3 DP / Predictor 数据接口

`src/candid/data/adapter.py` 中的 `canonical_observation()` 会从 HDF5 的 `state_dict` 读取命名状态：

| 字段 | 单步形状 |
|---|---|
| `robot_qpos`, `robot_qvel` | `[J]` |
| `tcp_pose` | `[7]` |
| `target_state` | `[16]` |
| `distractor_states` | `[6, 16]`（少于 6 个补零） |
| `distractor_mask` | `[6]`（1 为真实物体，0 为补位） |
| `goal_xy` | `[2]` |

`windows.py` 是通用滑动窗口工具，其默认参数仍是 Observation Horizon = 2、Action Horizon = 12；**Predictor 训练配置现在使用 Observation Horizon = 2、Action Horizon = 8**，并显式传入窗口长度，以匹配 Vanilla DP 输出的 8 步动作：

- **DP：** 可用成功轨迹进行行为克隆；是否包含失败样本由训练端自行决定。
- **Predictor：** 以状态历史和候选 Action Chunk 为输入，以该动作区间的真实 DR / Topple / Fall 作为监督标签；失败轨迹也可保留。

计算监督标签所用的未来物体 Pose **不能**泄漏进模型输入。模型训练入口见第 10 节。Vanilla DP 按 ManiSkill 的预测时域构造样本；Predictor 使用 2 步观测和 8 步动作窗口。已有 12 步 Predictor checkpoint 不适用于新的 8 步输入，需要重新训练。

## 7. 验证与数据质量检查

正式采集后至少执行：

```bash
python scripts/validate_dataset.py demos/team_a_batch01
python scripts/analyze_collection.py demos/team_a_batch01
```

`validate_dataset.py` 主要检查文件引用、基本字段、时间长度、数组维度、有限数值和 HDF5 与 Trace 的 T/T+1 对齐。它**不会代替逐帧数值一致性、成功率、轨迹多样性或 DR 分布分析**。

采集分析输出重点关注：

- `attempts` / `successes`：尝试数及任务成功数。
- `baseline_dr` / `min_success_dr` / `max_success_dr`：成功轨迹的扰动基线和范围。
- `meaningful_low` / `meaningful_high`：相对于 Baseline 超过分析阈值的低/高扰动轨迹数。
- `bidirectional`：是否同时覆盖相对较低与较高 DR。
- `feedback_attempts` / `uniform_attempts` / `fallback_attempts`：不同候选选择方式的计数。

这里“meaningful”是脚本中的经验阈值（结合绝对变化、Baseline 10% 和重复 Baseline 波动），**不是统计学显著性检验**。

原有测试文件仍可运行：

```bash
python -m pytest tests/ -q
```

当前远端 `main` 没有提交最终 Feedback/Production Sampler 对应的新单元测试；代码更新时应通过 Docker 实际采集回归测试，并考虑未来补充 CI。

## 8. 自定义环境与使用限制

默认环境：

```text
CANDIDBaseline-v1      envs.candid_baseline_env
CANDIDPrimitive-v1     envs.candid_primitive_env
CANDIDYCBFruit-v1      envs.candid_ycb_fruit_env
```

例如测试 Primitive 场景，可先使用：

```bash
python scripts/collect.py \
  --env-id CANDIDPrimitive-v1 \
  --env-module envs.candid_primitive_env \
  --scenes 1 \
  --output-dir demos/primitive_test
```

**注意：** 当前官方示范 Planner 优先面向固定形状的 Cube Target；复杂 YCB Target 的抓取未经过统一验证，即使能成功初始化环境，也不能保证稳定完成任务。自定义场景应先做少量运行和视觉检查，再批量采集。

环境 Settling 参数可以通过 `--env-kwargs` 调整，例如：

```bash
python scripts/collect.py \
  --scenes 1 \
  --env-kwargs '{"max_settle_steps":400}' \
  --output-dir demos/settle_test
```

需要牢记以下限制：

1. **不是在线 Replan**：当前 Feedback 在完整 Trajectory 之间生效，而不是机器人执行中途根据接触不断重新规划。
2. **不是严格无碰撞规划**：MPlib 不将 Distractors 作为不可碰撞硬障碍，故会出现不同程度的真实物理扰动。
3. **不保证各 Scene 的 DR 双向覆盖**：反馈排序基于成功历史的局部经验，不是理想的扰动预测器。
4. **不是 GPU 向量化采集**：当前默认 CPU / `num_envs=1`，多机合作建议分独立采集任务。
5. **不保证所有轨迹成功**：失败轨迹保留，但行为克隆数据通常还需要根据任务成功状态筛选。
6. **暂无统一的环境依赖锁文件和针对正式采样器的仓库 CI**：团队应统一 Docker 镜像与版本，保留 Git Commit、Seed 和采集配置。

## 9. 团队协作原则

- 组员只需要使用统一的 `scripts/collect.py`；不要在本地单独修改 DR 公式、Sampler 默认比例或动作控制模式后混入共享数据。
- 每人使用不同的 `seed` 范围和输出目录；不同环境类型分开存放，保留 `env_id`、Git Commit 和 SceneSpec。
- 先验证，再上传 HDF5、NPZ、JSONL 到团队共享存储；不要将大规模原始数据当作代码提交。
- **Train / Validation / Test 应按 Scene 划分**，不要把同一 Scene 的多条 Trajectory 随机打散到不同数据集，避免共享初始布局导致数据泄漏。
- 如果需要改变 Planner 或采样策略，先用少量 Scene 比较任务成功率和 DR 分布，并更新版本记录后再统一采集。

---

**团队建议：** 将本文件作为仓库根目录的 `README.md`。此前根目录和子目录的部分说明仍描述早期 v1 架构及手动 `--plan-variants`，以本文档与实际 `main` 代码为准。正式采集遇到错误时，优先提供 `git rev-parse HEAD`、完整命令、终端报错及对应数据目录的 `scenes.jsonl` / `trajectories.jsonl`。

## 10. 模型训练与评估

Vanilla DP 和 Disturbance Predictor 各自独立，代码在 `model/vanilla_dp` 与 `model/predictor`。删除 `baselines/` 不影响二者运行。Vanilla DP 的网络、损失和采样跟随 ManiSkill commit `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` 的 state Diffusion Policy。动作是绝对 `pd_joint_pos`，训练前会线性映射到 `[-1, 1]`，执行前再映射回来。

```bash
python scripts/train_dp.py --dataset demos/dense_random --output-dir runs/vanilla_dp --total-iters 1000
python scripts/train_predictor.py --dataset demos/dense_random --output-dir runs/predictor --action-horizon 8 --steps 10000 --log-freq 100 --eval-freq 500 --val-max-windows 4096 --device cpu
python scripts/eval_dp.py --dataset demos/dense_random --checkpoint runs/vanilla_dp/checkpoints/vanilla_dp.pt --output-dir runs/vanilla_dp_eval --safe-dr-mean-max 0.2 --no-video
```

三处都会在当前 Python 能看到 CUDA 时使用 GPU，否则使用 CPU。强制 CPU：DP 训练加 `--no-cuda`，Predictor 和评估加 `--device cpu`。物理回放仍由 ManiSkill 执行；评估时 GPU 只运行策略网络。

第一次训练会在数据集目录写入 `split.json`。评估只回放其中的 test scenes，并且需要 ManiSkill。`--total-iters` 的上游默认值是 1000000；上面的 1000 只是一次短训练。

Predictor 训练每隔 `--log-freq` 步记录训练损失、每隔 `--eval-freq` 步在固定验证窗口上评估（训练最后一步也会验证）。结果保存在 `runs/predictor/train_log.csv`、`summary.json` 和 `checkpoints/best_val.pt` / `checkpoints/last.pt`。最佳模型按归一化 MSE 选择，测试场景不参与训练和模型选择。`--steps 40` 可用于快速冒烟测试，默认正式训练为 10000 步。

### Predictor P1: lazy data, event classification and held-out evaluation

The Predictor remains independent from Vanilla DP and data collection. The shared src/candid/data/
modules are unchanged. Default horizons: observation 2, action 8.

- DR mean/max: normalized MSE regression; Topple/Fall: masked, class-weighted BCE logits.
- An event is newly triggered inside the chunk (false-to-true); it is not an episode safety decision.
- Predictor-only lazy HDF5/NPZ reads and trajectory-local shuffled batches reduce peak memory.
- Missing disturbance event flags fail by default. Pass --allow-missing-event-labels to
  mask unknown event labels while retaining each window's DR regression target.
- Checkpoint version 2 uses separate regression/event heads; earlier predictor checkpoints
  must be retrained. Inference returns dr_mean, dr_max, topple_prob and fall_prob.

Train with a fresh output directory:

```bash
python scripts/train_predictor.py \
  --dataset demos/dense_random \
  --output-dir runs/predictor_p1 \
  --action-horizon 8 \
  --steps 10000 \
  --device cpu
```

Evaluate frozen held-out test scenes (does not start ManiSkill):

```bash
python scripts/eval_predictor.py \
  --dataset demos/dense_random \
  --checkpoint runs/predictor_p1/checkpoints/best_val.pt \
  --output-dir runs/predictor_p1_eval \
  --device cpu
```

The evaluator tunes event thresholds using validation scenes only. Outputs: summary.json,
predictions.csv, scene_metrics.csv and DR scatter plots when matplotlib is installed.
Missing positive/negative classes yield unavailable AUROC/AUPRC (null).

```bash
python -m pytest tests/test_predictor_p0.py tests/test_predictor_p1.py tests/test_train_pilot.py -q
```

### Frozen DP + Predictor integration (Shadow / Rerank)

The new integration wrapper is in model/integration/. It does not import into,
train, or change either model. The shared rollout gets only optional hooks for
action bounds and per-episode diagnostics; plain Vanilla DP evaluation is unchanged.

Shadow mode samples one chunk and never changes the selected DP actions:

```bash
python scripts/eval_dp_predictor.py \
  --dataset demos/dense_random \
  --dp-checkpoint runs/vanilla_dp/checkpoints/vanilla_dp.pt \
  --predictor-checkpoint runs/predictor_p1/checkpoints/best_val.pt \
  --output-dir runs/dp_predictor_shadow \
  --mode shadow --num-candidates 1 --exec-horizon 8 \
  --safe-dr-mean-max 0.2 --device cpu --no-video
```

Rerank mode evaluates four separately sampled DP action chunks:

```bash
python scripts/eval_dp_predictor.py \
  --dataset demos/dense_random \
  --dp-checkpoint runs/vanilla_dp/checkpoints/vanilla_dp.pt \
  --predictor-checkpoint runs/predictor_p1/checkpoints/best_val.pt \
  --output-dir runs/dp_predictor_rerank \
  --mode rerank --num-candidates 4 --exec-horizon 8 \
  --min-improvement 0.01 --max-joint-deviation 0.15 \
  --safe-dr-mean-max 0.2 --device cpu --no-video
```

These commands assume the named checkpoint files already exist; adjust paths.
Each episode writes policy_diagnostics.json with raw candidate risk estimates,
the chosen action index, and any action clipping. The summary retains the same
Success/Safety/Safe Success metrics as the original rollout.

Risk scoring is local to an 8-step chunk, not an episode safety guarantee.
Topple logits are treated as uncalibrated ranking scores. Fall is disabled by
default because current held-out data has no Fall positives. Action candidates
are reranked only when predicted risk improvement and max joint deviation
pass the configured conservative gates. Candidate 0 is always retained as
fallback. Executing fewer than eight steps scores a longer horizon than is
actually executed; the first validation should use --exec-horizon 8.

```bash
python -m pytest tests/test_dp_predictor_integration.py tests/test_train_pilot.py -q
```
