# CANDID `scripts/`：运行命令与测试指南

本目录只有**命令行入口**。具体的 Environment、Runner 和数据处理逻辑分别在 `envs/`、`src/candid/` 和 `planners/` 中。请不要把新的 Disturbance 公式或文件写入逻辑塞进 `scripts/`。

下面的命令基于 **CANDID 组件化 v1（包含 Pose-based Settling）**，默认在 Docker 容器内执行。

## 1. 从哪里运行？

进入 ManiSkill Docker 容器后，先回到项目根目录：

```bash
cd /workspace/CANDID
```

全部命令都假定当前工作目录为 `/workspace/CANDID`，所以使用 `python scripts/xxx.py`。已有的 `scripts/*.py` 会自行将项目根目录和 `src/` 加入 Python 搜索路径，一般不需要手动设置 `PYTHONPATH`。

如果还没有进入 Docker，可以从 Mac 终端进入**已存在的**容器（容器名称请按本机实际情况替换）：

```bash
docker start maniskill-dev
docker exec -it maniskill-dev bash
cd /workspace/CANDID
```

## 2. 三个主要脚本

| 脚本 | 用途 | 建议何时运行 |
|---|---|---|
| `run_baseline.py` | 创建 Scene，执行一条或多条 Planner Trajectory，自动保存数据 | 采集演示轨迹 |
| `check_scene.py` | 不调用 Planner，检查环境 Reset + Settling 后有没有明显自然漂移 | 新写 Env、修改物体或随机区域后 |
| `validate_dataset.py` | 检查 Scene/Trajectory 索引、HDF5/NPZ 文件及 Action/State 时间对齐 | 每次采集完成后、上传共享数据前 |
| `generate_scenes.py` | 只生成并保存稳定后的 Scene 快照，不执行 Planner | 先准备场景，之后再采集轨迹 |
| `preview_scene.py` | 将稳定后的一个 Scene 渲染为 PNG | 查看物体大小与初始布局 |

### 2.1 最基础的 Baseline 采集

生成 1 个随机 Scene、1 条 Trajectory：

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 1 \
  --seed 0 \
  --output-dir demos/pilot_baseline
```

正常情况下会打印一条记录，包含 `success`、`steps` 和 `DR`。`success=True` 表示完成了任务，**不代表 DR 必须为零**。

### 2.2 同一 Scene 生成三条不同 Trajectory

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 1 \
  --seed 42 \
  --plan-variants '[{"lateral_offset":0.0},{"lateral_offset":0.04},{"lateral_offset":-0.04}]' \
  --output-dir demos/pilot_multi
```

这里的三个 `lateral_offset` 属于 Planner Config，单位是 m：0 表示直接搬运，正负值表示不同的侧向路径。Runner 先随机生成并稳定一个 Scene，再从同一份初始 Snapshot 恢复，记录三条不同尝试。

**注意：** `--plan-variants` 使用合法 JSON；Shell 中外层用单引号包住 JSON 列表即可。默认每个 Scene 的 Trajectory 数量等于列表长度。

### 2.3 批量生成多个 Scene

例如 5 个 Scene，每个 3 条 Trajectory：

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 5 \
  --seed 100 \
  --plan-variants '[{"lateral_offset":0.0},{"lateral_offset":0.04},{"lateral_offset":-0.04}]' \
  --output-dir demos/pilot_5scenes
```

预期尝试数为 `5 × 3 = 15`。其中某些 Planner 可能失败；失败尝试同样会保留状态及扰动信息，但若没有实际执行任何动作，Recorder 不一定会生成对应的非空 HDF5 Group。`validate_dataset.py` 会依据 Manifest 中的 `saved_hdf5` 标志检查。

### 2.4 在其他 Environment 里采集

对于组员新增的环境，需要**同时**指定 Python 模块与注册的 Env ID。例如 Primitive：

```bash
python scripts/run_baseline.py \
  --env-module envs.candid_primitive_env \
  --env-id CANDIDPrimitive-v1 \
  --scenes 3 \
  --seed 10 \
  --output-dir demos/pilot_primitive
```

YCB 初始化和无动作检查可用公共组件，但 **`execute_baseline.py` 的 Top-down Cube Planner 尚不保证能成功抓取 Strawberry**。因此不要将 YCB 采集命令的 Task Failure 直接理解为数据组件错误；正式采集 YCB 成功示范前，需要适配 Planner。

若 Docker 中尚未安装 YCB 资源，可执行：

```bash
python -m mani_skill.utils.download_asset ycb -y
```

### 2.5 固定布局调试

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --fixed-scene \
  --scenes 1 \
  --output-dir demos/debug_fixed
```

`--fixed-scene` 使 ObjectSpec 的 `fixed_xy` 和 `fixed_yaw` 生效，用于回归测试；大规模随机数据采集应使用默认的 `randomize_layout=True`。

### 2.6 控制 Settling 参数

Pose-based Settling 的参数来自公共 `CANDIDTabletopEnv`，通过 `--env-kwargs` 传入 JSON。下面只延长最大 Settling 时间，其他条件不变：

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 5 \
  --seed 100 \
  --env-kwargs '{"max_settle_steps":400}' \
  --output-dir demos/pilot_settling_400
```

常用参数还有 `settle_min_steps`、`stable_steps`（Pose 窗口长度）、`settle_translation_m`、`settle_rotation_deg`、`idle_verify_steps`、`idle_translation_m` 和 `idle_rotation_deg`。如果 Scene 一直无法稳定，应先检查初始碰撞、物体形状与持续运动，**不要为了采集通过而无限放宽阈值**。

### 2.7 录制视频与调试

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 1 \
  --save-video \
  --verbose \
  --output-dir demos/pilot_video
```

`--save-video` 需要 Docker 的 ManiSkill 渲染配置可用。在没有 CUDA 的 Mac Docker 上出现 `sapien_cuda` 回退 CPU 警告并不一定影响物理模拟，但**不能保证视频渲染正常**。

只想调试，不需要保存训练轨迹时可以使用 `--no-record`；它不会保存 HDF5 Action/Observation，但 Runner 仍会写 Scene、Disturbance Trace 和索引：

```bash
python scripts/run_baseline.py \
  --env-id CANDIDBaseline-v1 \
  --scenes 1 \
  --no-record \
  --output-dir demos/debug_no_h5
```

若使用 `--save-success-only`，未成功尝试的 HDF5 可能不保存；正式训练 Predictor 需要保留失败轨迹时，不要开启该选项。

### 2.8 先生成场景，之后再执行

以下命令只创建并稳定两个场景（seed 3 和 4），保存 `scenes.jsonl` 与 `scenes/*.npz`，不会执行抓取规划器，也不会生成动作轨迹：

```bash
python scripts/generate_scenes.py \
  --env-module envs.candid_tight_upright_env \
  --env-id CANDIDTightUpright-v1 \
  --scenes 2 --seed 3 \
  --output-dir demos/tight_scenes

python scripts/validate_dataset.py demos/tight_scenes
```

要给这批**已保存的场景快照**逐个生成对应的预览图：

```bash
python scripts/preview_scene.py --scene-source demos/tight_scenes
```

图片保存在 `demos/tight_scenes/previews/`，文件名与 `scenes.jsonl` 中的 `scene_id` 对应。渲染只恢复场景状态，不执行机械臂动作；需要当前 Docker 的 SAPIEN CPU 渲染可用。

前面的两场景示例预期校验结果为 2 个 Scene、0 个 Trajectory、0 个 HDF5。需要执行时，在**新的输出目录**中读取这些快照：

```bash
python scripts/run_baseline.py \
  --env-module envs.candid_tight_upright_env \
  --env-id CANDIDTightUpright-v1 \
  --scene-source demos/tight_scenes \
  --save-video \
  --output-dir demos/tight_rollouts

python scripts/validate_dataset.py demos/tight_rollouts
```

`--scene-source` 默认执行源目录中的全部 Scene；可加 `--scenes N` 只执行前 N 个。`--save-video` 会在轨迹目录中保存 MP4。每条输出轨迹的场景记录包含 `source_scene_id`，并会检查恢复后的目标、干扰物与机械臂初始状态是否和快照一致。重放需要相同的环境代码、`SceneSpec`、ManiSkill/资产和控制配置。采集时保留与干扰物的真实碰撞及任务失败结果；不要加 `--save-success-only` 来过滤失败轨迹。`--no-record` 仍会执行机械臂，只是不保存 HDF5；要真正只生成场景，请使用 `generate_scenes.py`。

如果需要更明显的位置变化，可使用 `candid_tight_upright_diverse_env`。它保留原场景的物体尺寸，扩大矮方块的摆放范围，并要求每个物体与最近物体按外接圆估算的间距不超过 8 cm。高物体的摆放范围较窄，以保持初始物理稳定。它使用新的环境 ID，因此不会改变旧场景快照对应的环境：

```bash
python scripts/generate_scenes.py \
  --env-module envs.candid_tight_upright_diverse_env \
  --env-id CANDIDTightUprightDiverse-v1 \
  --env-kwargs '{"num_distractors":5,"distractor_seed":4105}' \
  --scenes 20 --seed 4105 \
  --output-dir demos/test_scence_toy_diverse

python scripts/preview_scene.py --scene-source demos/test_scence_toy_diverse
python scripts/validate_dataset.py demos/test_scence_toy_diverse
```

## 3. 新环境必须运行的无动作测试

Baseline：

```bash
python scripts/check_scene.py \
  --env-module envs.candid_baseline_env \
  --env-id CANDIDBaseline-v1 \
  --seed 42 \
  --steps 100
```

YCB：

```bash
python scripts/check_scene.py \
  --env-module envs.candid_ycb_fruit_env \
  --env-id CANDIDYCBFruit-v1 \
  --seed 42 \
  --steps 100
```

批量检查不同 Seed（以 Baseline 为例）：

```bash
for seed in 10 20 30 40 50; do
  echo "=== seed ${seed} ==="
  python scripts/check_scene.py \
    --env-id CANDIDBaseline-v1 \
    --seed "$seed" \
    --steps 100
done
```

此脚本会先正常 Reset 和 Settling，再运行 100 个无动作控制步，并打印 `Idle disturbance`。默认只以 `dr_mean <= 0.01` 为粗筛标准；请同时关注 `peak_translation_m` 与 `peak_rotation_rad`，因为平均值可能掩盖单个物体的明显漂移。

**当前的 `check_scene.py` 不提供 `--env-kwargs` 参数。** 如果需要改变 Settling 配置，请通过 `run_baseline.py --env-kwargs ...` 测试，或者由团队统一扩展 `check_scene.py`，不要直接在各个 Env 复制一套检查逻辑。

## 4. 数据集完整性校验

每次采集结束后执行：

```bash
python scripts/validate_dataset.py demos/pilot_multi
```

预期类似：

```text
{'num_scenes': 1, 'num_trajectories': 3, 'num_recorded_hdf5': 3}
```

该脚本检查 Scene/Trajectory ID 是否重复、索引关联是否存在、Scene Snapshot 是否非空、Trace 的 `[T+1,N,7]` / `[T+1,N,3]` 时间结构是否正确，以及已有 HDF5 中 `actions` 的长度是不是 `T`、`obs` 的长度是不是 `T+1`。

**通过不等于任务成功，也不等于轨迹之间一定不同。** 新的 Planner 还应抽检视频或 HDF5 Actions，以验证执行的真实差异。

## 5. 运行单元测试

使用已安装的 Python 环境，在项目根目录执行：

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_settling.py' -v
```

数据契约测试：

```bash
PYTHONPATH=src:. python -m pytest tests/test_contract.py -q
```

这两类测试检查稳定性计算和数据结构的纯 Python 逻辑；**不替代 ManiSkill 的真实物理测试**。如果当前 Docker 环境没有安装 `pytest`，可先确认 `python -m pytest --version`，再使用项目维护的依赖安装方式。

## 6. 常见运行参数速查

以下参数都属于 `scripts/run_baseline.py`，实际实现位于 `src/candid/runner/collect.py`：

| 参数 | 默认值 / 用途 |
|---|---|
| `--env-id` | `CANDIDBaseline-v1`，Gym 注册 ID |
| `--env-module` | `envs.candid_baseline_env`，导入以完成注册 |
| `--scenes` | Scene 数量，默认 1 |
| `--seed` | 第一个 Scene 的 Seed；随后 Scene 使用递增 Seed |
| `--plan-variants` | JSON 列表；一个 Scene 的不同 Planner 配置 |
| `--trajectories-per-scene` | 若未指定，按 `--plan-variants` 长度决定；多轨迹时两者数量须匹配 |
| `--plan-kwargs` | 应用于所有 Trajectory 的共同 Planner 参数 JSON |
| `--planner` | `模块:函数`，默认为 `planners.execute_baseline:solve` |
| `--planner-config-class` | `模块:类`，默认为 `planners.execute_baseline:BaselinePlanConfig` |
| `--fixed-scene` | 使用固定位置和朝向而不是随机布局 |
| `--env-kwargs` | 传给 Environment 的构造参数 JSON，例如 Settling 设置 |
| `--control-mode` | 默认 `pd_joint_pos`，与当前 Planner 保持一致 |
| `--obs-mode` | 默认 `state_dict`；正式 HDF5 采集必须用它 |
| `--sim-backend` | 默认 `cpu`；当前流程仍限制 `num_envs=1` |
| `--save-video` | 另外录制视频（要求渲染可用） |
| `--save-success-only` | 只将成功尝试录入 HDF5；慎用于 Predictor 数据 |
| `--no-record` | 不写 HDF5 轨迹，仅用于调试 |
| `--translation-weight` / `--rotation-weight` | DR 权重，默认 0.7 / 0.3 |
| `--output-dir` | 输出目录，默认 `demos/candid_v1` |
| `--scene-only` | 只保存稳定 Scene，不运行 Planner；`generate_scenes.py` 会自动设置 |
| `--scene-source` | 从已有输出目录的 `scenes.jsonl` 和 NPZ 快照恢复场景并执行 |

查看最完整的参数帮助：

```bash
python scripts/run_baseline.py --help
```

## 7. 输出目录与团队协作约定

```text
demos/pilot_multi/
├── trajectories_<run_id>.h5
├── trajectories_<run_id>.json
├── scenes.jsonl
├── trajectories.jsonl
├── scenes/<scene_id>.npz
└── disturbance_traces/<trajectory_id>.npz
```

- 每个组员使用**自己的输出目录**（例如 `demos/member_a/...`），不要多人同时写一个 JSONL/HDF5 文件。
- 数据先运行 `validate_dataset.py`，再上传团队共享数据存储。
- 统一 ManiSkill 版本、资产版本、`pd_joint_pos`、`state_dict`、control timestep 和 DR 定义。
- 划分训练/验证/测试集时按 **Scene ID** 分，而不是把同一 Scene 的不同 Trajectory 随机分到多个集合。
- `demos/`、HDF5、NPZ 和视频不应提交 GitHub；代码、测试、README 和 SceneSpec 应提交。

更多关于模块职责和 DP/Predictor 字段说明，请阅读 [`../src/README.md`](../src/README.md)。

## 路线意图试采（`scripts/collect.py`）

使用 `--route-intents` 后，每四次尝试组成一组：`direct`、`left`、`right`、`over`。同组使用同一个场景快照和相同的抓取/抬起扰动，只改变夹起方块后的搬运路线。`left/right` 以实际搬运方向定义，`over` 根据附近物体顶部生成较高的中间点。路线候选点有 XY/Z 范围约束；机械臂可达性、整臂碰撞和任务结果仍由 MPlib 规划与仿真执行判断。失败尝试照常记录。

```bash
python scripts/collect.py \
  --env-module envs.candid_tight_upright_diverse_env \
  --env-id CANDIDTightUprightDiverse-v1 \
  --scenes 1 --trajectories-per-scene 4 --seed 42 \
  --route-intents --no-collision-aware --save-video \
  --output-dir demos/route_intent_pilot

python scripts/validate_dataset.py demos/route_intent_pilot
```

`--trajectories-per-scene` 在此模式下必须是 4 的倍数；设置为 24 即每场景六组、每组四条，共 24 次尝试。采集结果的 `trajectories.jsonl` 包含 `planner_config.route_intent`、`proposal.route_block`、实际使用的 `route_waypoints_xyz` 和失败阶段；`collection_summary.csv` 列出各路线的成功次数。默认模式不受影响。若不加 `--no-collision-aware`，每个场景还会额外执行一次独立的避障参考尝试，不计入指定的轨迹次数。
