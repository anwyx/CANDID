# CANDID Guidance 动作修改过程记录与可视化 —— 开发任务 README

> **本 README 是给 AI 编程助手的开发需求。请先检查仓库现有实现，再按下面要求写代码、测试，并说明修改内容。不要凭空声称实验结果，也不要重新训练模型。**

## 1. 目标与示例效果

我们的项目是 CANDID：使用一个物理后果 Predictor，在 Vanilla Diffusion Policy (DP) 的反向扩散采样中，通过风险梯度引导动作，使机器人在完成抓取放置任务时尽可能降低对周围障碍物的扰动（DR、Topple、Fall）。

**我要新增一个独立脚本**，例如 `scripts/inspect_candid_guidance.py`。给定一条冻结的测试 Scene，以及已经训练好的 DP 和 Predictor Checkpoint，能够运行 CANDID Guidance，**同时记录、比较并可视化**：

1. 在**同一 Scene、同一重规划时刻、同一观测历史、同一扩散随机数**下，Vanilla DP 原本会生成的完整 Action Chunk 是什么；
2. CANDID Guidance 后，最终生成的 Action Chunk 变成什么；
3. 这**两个完整 Action Chunk**经过同一个 Predictor 评估后的 `predicted_dr_mean` 和 `predicted_dr_max` 分别是多少（以及可选的 Topple/Fall 风险和加权 Risk）；
4. 这些动作具体在**哪些反向扩散步**被 Guidance 影响：每步预测风险、梯度范数、修改幅度等；
5. 保存便于科研分析的数据，并生成一张直观的图：以机器人操作画面为背景/左侧，上面或右侧显示**蓝色 Vanilla DP、红色 CANDID Guided** 的双条形图，标注各自的预测 DR；另外绘制关节动作差异图与扩散过程诊断图。

用户提供的参考图片是“操作画面 + 两条带数值的彩色风险条”。**仅参考视觉布局**，不要把指标叫作 `Spillage Probability`：我们的核心指标是 **`Predicted Local DR Mean` / `Predicted Local DR Max`**，它们不是概率。Topple/Fall 的 sigmoid 输出可以单独标为预测概率/风险分数，不能与 DR 混淆。不要为了作图强制令 Guided 风险小于 Vanilla 风险，必须展示真实预测结果。

## 2. 仓库上下文：必须先看这些文件

仓库：`anwyx/CANDID`。以当前工作树与实际 Checkpoint 格式为准；不要盲目用旧分支文件覆盖新代码。

- `scripts/eval_dp_guidance.py`：当前 Guidance 评估入口。
- `model/guidance/evaluate.py`：加载 DP、Predictor，构造 Guided Policy，并调用共享评估器；已有 `--diagnose`。
- `model/guidance/policy.py`：`ConsequenceGuidedPolicy`；已有同 RNG 的 Vanilla / Guided 配对诊断，但默认只保存差异的汇总统计，没有保存两份完整动作数组。
- `model/guidance/sampler.py`：`ConsequenceGuidedSampler.sample()`，真正执行逐步 DDPM 去噪及风险梯度修正；实现中先计算 epsilon、估计干净动作，再用 Predictor 计算风险梯度，DDPM 更新后施加 correction。
- `model/guidance/objective.py`：`chunk_risk()` 和 `risk_from_predictions()`；风险目标是 DR Mean/Max 与 Topple/Fall 的加权组合。
- `model/vanilla_dp/checkpoint.py`、`model/vanilla_dp/policy.py`、`model/vanilla_dp/normalizer.py`：DP 网络、动作反归一化、**可能存在的 Observation Normalization**。
- `model/predictor/checkpoint.py`、`model/predictor/inference.py`：Predictor 和其归一化、DR 输出格式；检查模型是否是 ordinal DR 版本。
- `src/candid/eval/rollout.py`：读取冻结 Scene 初始状态，执行动作，保存 `trajectory.npz`、结果、诊断 JSON 等。
- `src/candid/eval/scene_suite.py`、`src/candid/data/adapter.py`：冻结 Scene Suite 的结构与真实观测特征顺序。
- `src/candid/eval/local_dr.py`：已有局部物理后果配对重放；可参考，但本任务优先实现**扩散动作修改过程的记录和可视化**，不要自行声称预测 DR 就是真实物理 DR。

**重要的现有行为：**

- DP 用 `obs_horizon=2`，通常 `act_horizon=8`，闭环 `exec_horizon=4`；根据 Checkpoint 验证，禁止硬编码不一致的 Horizon。
- DP Action Chunk 是 7 维 `pd_joint_pos` 绝对关节位置命令，不是 Cartesian XYZ 轨迹。
- Predictor 输入使用其自己的 `predictor_normalizer`；DP 若训练启用了 `normalize_obs=True`，Guidance 的去噪网络也必须接收按**DP 自己的**观测归一化器处理的观测；不要把 DP-normalized 观测误送入 Predictor。
- `--strength 0` 必须产生等价于 Vanilla DP 的结果（在相同 RNG、同一状态下）；`--strength > 0` 才真正施加梯度修正。
- 单个反向扩散时间步上的估计动作 `x0_hat` **不等同于最终输出 Action Chunk**，不能把 `A_k` 或每一步 `x0_hat` 误标为“最终 Vanilla/Guided 动作”。

## 3. 开发范围：分两层记录（必须区分）

### A. Replan 级别：最终动作的严格配对比较（必做，优先）

每次机器人重新规划（replan）时：

1. 捕获这个时刻真正送给策略的原始观测历史 `obs_window`、Scene ID、Episode Step、Replan Index、RNG/Seed。
2. 在不改变 Guided Policy 随机数序列的前提下，从**完全相同的观测、完全相同的初始扩散噪声及 DDPM 随机采样状态**，分别生成：
   - `vanilla_chunk`: 未引导 DP 的最终动作 `[H_act, 7]`；
   - `guided_chunk`: Guidance 完整运行后的最终动作 `[H_act, 7]`。
3. 两份动作都必须先从 **DP normalized action** 转换到实际 `pd_joint_pos` 物理命令单位，再按与共享 Rollout 完全相同的 `env.action_space` 做裁剪，得到实际会执行的候选命令。**用裁剪后命令统一交给 Predictor 评分**，并把裁剪前动作另外留档。
4. 用**同一个冻结 Predictor、同一观测历史和同一风险配置**，计算两组动作的：
   - `pred_dr_mean_vanilla`、`pred_dr_mean_guided`
   - `pred_dr_max_vanilla`、`pred_dr_max_guided`
   - `pred_topple_score` / `pred_fall_score`（可选，但有就同时记录）
   - `pred_weighted_risk_vanilla` / `pred_weighted_risk_guided`
   - `delta_dr_mean`、`delta_dr_max`、`delta_risk`（定义为 Guided − Vanilla；负值代表预测风险下降）
   - `action_abs_mean`、`action_abs_max`、`action_l2`，必要时逐关节统计。
5. 保存两个**完整 8-step（或 Checkpoint 定义的 H_act）Action Chunk**，而不是只保存前 4 个实际执行动作。与此同时可另外保存前 `exec_horizon` 个动作以方便可视化。
6. 物理环境仍只执行 **Guided** 这一条真实闭环轨迹；Vanilla Chunk 是同观测下的反事实候选动作，**不能混入真实 Guided 轨迹执行**。如果日后要比较两份动作的真实物理 DR，应使用一致初始状态与相同动作前缀进行独立重放，而不能把 Predictor 输出当物理真值。

**核心科学问题：** CANDID 改了什么动作？Predictor 认为变化后的局部物理后果如何？不要把各自独立长时间 Rollout 的同一序号 replan 误认为“同一状态下的配对”。

### B. Diffusion Step 级别：真正在哪一步修改了扩散采样（必做）

对一个可选的 Replan（例如 `--trace-replan 10`），逐个记录逆扩散时间步：

- `diffusion_timestep`、是否触发 Guidance；
- 当前估计干净动作 `x0_estimate` 在可执行窗口中的值或其摘要（必须标注这是**中间估计**）；
- 此刻对该候选 Chunk 计算的 Predictor Risk（最好同时保留 `pred_dr_mean`、`pred_dr_max`）；
- `raw_gradient_l2`、`applied_correction_l2`、`max_abs_correction`；
- 可选保存实际修正向量 `correction` `[H_act, 7]` 以及局部的候选动作 `x0_action_chunk`；
- 无引导步骤填入 `guided=false` / `correction_l2=0`，不要捏造预测值。

**时序含义非常重要：**当前 `sampler.py` 中的 correction 施加在 **DDPM 更新后的下一个 noisy sample** 上，不是直接覆写当前 `x0_estimate`。所以不要拿“当前 x0 改前/改后”去假装它是两次真实完整采样。若希望得到“该步骤修正前后”的严格对比，必须实现受控的同噪声单步 counterfactual 分支，明确两条分支使用相同 DDPM step 随机状态；否则只展示该步的预测风险、梯度、更新量，并用 A 部分最终的严格配对来展示最终动作差异。

尽量只在启用 trace 时才保存逐步数据；默认评估不能因日志增加大量 GPU 同步、磁盘写入或改变 RNG 顺序。建议收集到内存后，Replan 结束一次性落盘。

## 4. 输出文件和可视化格式

默认输出到**新的**目录，如 `eval/guidance_inspect/<run_name>/`，不要覆盖以前任何 Evaluation 结果。

建议结构：

```text
eval/guidance_inspect/<run_name>/
├── metadata.json                # 场景、Checkpoint、hash/路径、参数、seed、版本
├── replans.jsonl               # 每次 Replan 的配对指标（小数据）
├── chunks.npz                  # 全部/所选 Replan 的 vanilla/guided action chunks
├── diffusion_trace.jsonl       # 每个记录的扩散 timestep 风险/梯度/更新摘要
├── diffusion_arrays.npz        # 可选完整 x0/correction 等数组
├── figures/
│   ├── replan_000.png          # 参考图风格的蓝/红 DR 对比图
│   ├── actions_replan_000.png  # 7 个关节 vanilla/guided 动作曲线
│   └── diffusion_replan_000.png # Risk、梯度范数、修正幅度 vs timestep
├── trajectory.npz             # Guided 真正执行的动作及物理轨迹（若启用闭环）
└── comparison.mp4             # 可选；渲染受环境限制
```

数据至少要能用 `np.load(..., allow_pickle=False)` 读取固定形状的数值数组：`vanilla_actions`, `guided_actions`, `vanilla_actions_clipped`, `guided_actions_clipped`, `replan_indices`, `episode_steps`。如变长需明确 padding/mask 或按 Replan 独立文件，**不要使用 Pickle/object 数组**。

每个 Replan 的 JSON 示例字段（`null` 是占位符，不是实验数据）：

```json
{
  "scene_id": "<actual_scene_id>",
  "episode_step": 40,
  "replan_index": 10,
  "paired_same_obs_and_rng": true,
  "predicted_local_dr": {
    "vanilla": {"dr_mean": null, "dr_max": null},
    "guided": {"dr_mean": null, "dr_max": null}
  },
  "delta_dr_mean_guided_minus_vanilla": null,
  "delta_dr_max_guided_minus_vanilla": null,
  "action_abs_mean": null,
  "action_abs_max": null,
  "action_l2": null,
  "guided_diffusion_steps": null
}
```

### 4.1 图片要求

**主图**应尽可能接近用户给的参考图：

- 左侧/背景为选择的 Replan 时刻操作 Scene 画面（必须是真实仿真渲染帧；不可把虚构图像冒充实验帧）。
- 右侧或画面角落覆盖清晰的标题 `Predicted Local DR Mean` 和双条形图：蓝色 `Vanilla DP`，红色 `CANDID Guided`；数字必须是实际模型预测值。可再加一组 `Predicted Local DR Max`。
- 两条 Bar 使用**相同数值刻度**；标明比例尺上限，超过上限时清晰提示，不可以分别归一化各自的值制造视觉差异。
- 下方标明 Scene ID、Episode Step、Replan Index、Guidance Strength、ΔDR 和动作变化量。
- 如果该步 Guided 风险反而更高，照实显示，不美化。

**可选 MP4：**若具备正常离屏渲染环境，可将上述 Overlay 叠加到 Guided Rollout 视频中，随 Replan 更新数值；帧要与 Episode Step 对齐。当前服务器可能缺 Vulkan 系统库，因此**不能将 MP4/渲染当成核心程序成功的前提**。`--no-render` 模式至少要稳定产出可视化数据、动作曲线图和扩散诊断图；如果没有真实 Scene 帧，则用独立图表替代背景，不要伪造机器人画面。不要要求 sudo。

### 4.2 关节动作图

以每个 Action Chunk 的时间位置 `0..H_act-1` 为横轴，画 **7 个关节的 Vanilla vs Guided 命令**（可用 7 行小图）。清楚区别完整预测 Chunk 和只执行的前 `exec_horizon` 步；最好用浅色背景标记执行区间。另给出 Δaction 的统计量。

### 4.3 扩散诊断图

横轴为**真实逆扩散顺序**或明确标注的 diffusion timestep，展示 2–3 条曲线：Predictor Risk、Gradient L2、Correction L2。标出 guidance 从哪一个 timestep 开始生效。不要将每步的中间风险当作最终动作的预测 DR。

## 5. 命令行接口（建议）

至少支持以下参数；具体命名可按当前仓库风格微调，但文档和 `--help` 要完整：

```bash
python scripts/inspect_candid_guidance.py \
  --eval-suite eval/combined250 \
  --scene-id '<从 scenes.jsonl 选出的完整 scene_id>' \
  --dp-checkpoint ckpt/<new_dp>/best_val.pt \
  --predictor-checkpoint ckpt/<predictor>/best_val.pt \
  --output-dir eval/guidance_inspect/demo_scene \
  --strength 0.005 \
  --start-timestep 30 \
  --max-steps 300 \
  --exec-horizon 4 \
  --eval-seed 2026 \
  --device cuda \
  --trace-replan 10 \
  --no-render
```

以上 Checkpoint 路径只是占位示例，不能假定文件已存在。建议额外支持 `--max-replans`、`--all-replans`、`--render`、`--save-video`、`--profile-timing`。`--scene-id` 必须匹配真实冻结测试 Scene，加载其已有的 snapshot；不允许重新随机生成替代 Scene。`--strength 0` 应用于严格一致性测试。

快速找到真实 Scene ID：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('eval/combined250/scenes.jsonl')
for line in p.read_text().splitlines()[:10]:
    print(json.loads(line)['scene_id'])
PY
```

## 6. 实现约束

1. **最小侵入式修改**：主入口新增在 `scripts/`；必要时可为 `ConsequenceGuidedPolicy` / `ConsequenceGuidedSampler.sample` 新增可选 trace/callback 参数，但默认行为必须完全不变。不要重写 DP Denoiser、DDPM Scheduler、Predictor、控制器或 Metrics。
2. **保持现有 250 Scene Evaluation 兼容**：已有 `scripts/eval_dp.py`、`scripts/eval_dp_guidance.py`、`scripts/eval_dp_predictor.py` 不应因为新增诊断功能改变结果或接口。
3. **复用现有 `--diagnose` 的 paired RNG 思路**：在 `model/guidance/policy.py` 中已有 `torch.random.fork_rng` 机制，优先扩展它保存成对 Chunk 和 DR，而不要重新发明一个可能改变采样随机性的方案。检查 CPU 和 CUDA RNG 都被正确恢复。
4. **模型归一化必须正确**：DP 的 `obs_normalizer` 若存在，要用于 DP Denoiser；Predictor 仍单独用 Predictor Normalizer；输出 DR 必须正确解码（普通回归版本和 ordinal DR 版本都应检查）。如果当前分支缺少新版 Checkpoint 兼容，先报告并正确修复，禁止忽略 `normalize_obs=True` 或丢弃统计量。
5. **匹配预测与真正执行的物理命令**：绝对角度动作先做 DP action denormalization，再做真实控制器 bounds clipping；日志同时保留 raw 和 clipped，评分用后者。禁止以 normalized action 直接计算物理 DR 风险。
6. **不改变 Guided 轨迹**：记录诊断时额外生成 Vanilla counterfactual 不能改变 Guided 策略真实 RNG、时序或动作；与未启用 Trace 的同 Seed Guided Eval 应一致（容忍明确标注的浮点数级别差异）。
7. **风险定义透明**：分别展示 `pred_dr_mean` / `pred_dr_max` 和用于优化的 weighted risk；不要误把 `Topple/Fall` 的输出或 DR ordinal 超阈值分数当作已校准的概率。
8. **仿真正确性**：使用原始 SceneSpec / snapshot 与共同 rollout 的观测、动作和 DR 定义；不修改用户已有 Scene 文件和测试集；不要把不同初始条件下的最终结果当作严格配对反事实。
9. **资源控制**：默认只选单 Scene 和一个或少量 Replan 记录完整扩散数组，以免每个 DDPM 步大量写盘；视频可选，headless 必须可运行。
10. **不进行训练**、不更改任何已有 `.pt` 权重、不依赖 sudo。

## 7. 测试与验收标准（务必实际执行并报告）

至少新增针对诊断功能的测试，例如 `tests/test_guidance_visual_trace.py`，覆盖：

- [ ] 新 CLI 可以解析 Scene ID、DP/Predictor Checkpoint、Output Dir，输入错误时清晰报错。
- [ ] 输出的 Vanilla/Guided Chunk 形状均为 `[act_horizon, 7]`，无 NaN/Inf；JSON/NPZ 字段完整且可读取。
- [ ] `strength=0`、同观测、同 RNG 时，Vanilla 与 Guided **最终执行命令**一致或满足严格数值容差，`delta_dr` 接近 0。
- [ ] `strength>0` 时能够记录实际引导步数和每步 correction，并准确报告动作是否变化；**不强求**所有场景必须风险下降。
- [ ] `--trace-replan` 记录的逆扩散 timestep 次序、风险、梯度范数和更新幅度与真实 `sampler.py` 操作一致。
- [ ] 开关诊断时，在同一初始 Scene、同一 Seed 下，实际 Guided Rollout 的执行动作保持一致。
- [ ] 若启用 `normalize_obs=True`，训练统计正确用于 DP 分支，Predictor 仍使用自己的 Normalizer；没有统计量要显式报错。
- [ ] 没有图形渲染 / Vulkan 的服务器上，`--no-render` 依然可以生成 JSON、NPZ 和动作/扩散诊断 PNG。
- [ ] 可选的渲染叠加图清晰、标签正确，数字来自真实模型输出；风险更高时照实显示。
- [ ] 现有关键测试仍通过，不修改原评估结果定义。

**最低可接受交付：**一条命令指定单 Scene 后，能拿到同 RNG 的两组完整关节动作、两组预测 `DR Mean / DR Max`、改动作的时刻与大小，并自动生成两幅数据图（Action 对比 + Diffusion Step Trace）和一个双 Bar 的 DR 比较图。真实机器人操作帧叠加作为可选增强，不能阻塞数值核心交付。

## 8. 请按以下顺序完成并回复我

1. 先阅读并解释现有 Guidance 在 `sampler.py` 中何时、如何修改 noisy action；指出现有 `--diagnose` 缺什么字段，并检查新版 DP 的 `normalize_obs` 兼容性。
2. 给一个简短的改动方案，列出需要新增/修改的文件和对应 API，不要直接大面积重构。
3. 先实现**单 Scene、单 Replan 的严格配对数值记录**，并运行相应测试。
4. 再实现逐 Diffusion Step 的 Trace，确保指标定义准确、RNG 不变。
5. 最后实现 PNG 可视化和可选视频 Overlay；没有 Vulkan 时提供可用的 headless 替代结果。
6. 给出完整运行指令、示例输出路径、测试通过情况、已知限制。**没有实际运行的测试不得声称通过。**

**最终期望得到的科研证据链：**

`同一个 Observation + 相同随机数` → `Vanilla Action Chunk 与 Guided Action Chunk` → `Predictor 预测 DR 的变化` → `Guidance 各扩散步的梯度与修正记录` → （可选另做真实物理重放）`实际 DR 变化`。

请确保每一步的含义严格、可追溯、可重复；可视化是对真实计算的呈现，不是演示性质的假数据。
