# VQE-QAS 向大规模扩展：执行协议 v3（定稿骨架）

> 自包含文档，读者无需前置对话。
> 设计原则：**冻结形式，不冻结常数**（字段名/默认值/自适应规则/记录格式/评估口径冻结；
> `target_n / K / n_trials / threshold` 等为"带默认值的输入变量"）。

---

## v2 → v3 变更摘要（看过 v2 的人读这段即可）

| 类别 | 变更 |
|------|------|
| 🔴 θ 初始化 | `zero_plus_random` → **`random_uniform_pi`**（θ_j ~ U[−π,π]）。**zero start 降级为诊断项，禁止混入最终 fair ranking** |
| 🔴 小规模预算 | 统一预算不仅统一 `n_trials`，**还要统一 optimizer budget**；主口径 per-parameter budget + 旁口径 equal-maxfev stress test |
| 🔴 optimizer budget | 冻结默认值（见 §5），最大待定项落地 |
| 🟠 Migration 判据 | 给默认阈值：`family Spearman ≥ 0.6 且 top-K overlap ≥ 0.5 → 允许弱外推；否则必须 few-shot 校正` |
| 🟠 MoE | 改为**两条件 AND 触发规则**（见 §3 硬问题 2）|
| 🟠 15q regret 表述 | 收紧：**只有 15q 全枚举才叫 regret**；否则 high-coverage validation + rejected audit |
| 🟠 B1 benchmark | 每尺度测 **3 类结构（light/middle/heavy）**，非单一代表 |
| 🟠 B3 15q 闸门 | 48h 墙钟切换线（≤48h 全枚举，否则分层抽样+audit）|
| 🟢 命名 | low-fidelity proxy `B1/B2/improvement` → **`LF50/LF400/LF_improvement`**（避免与 §6 benchmark B1/B2 混淆）|
| 🟢 basin_SR | → **`posthoc_basin_SR`**（事后统计，标明不可用于在线决策）|
| 🟢 top-K | `K = min(N, max(10, ceil(0.1N)))` |
| 🟢 reference 精度 | 对齐用 `float64/complex128`；screening `complex64` 不参与 reference validation |
| 🟢 B4 10q | 默认不加，改为异常触发项 |

---

## 0. 背景

### 0.1 项目目标
VQE-QAS 是量子架构搜索框架：对 TFIM 求基态能量，从结构化 HEA 候选空间自动找最低能量 ansatz。

### 0.2 搜索空间（声明式 HEA）
```
rotation_set × entangler × layers × pattern × final_rotation
```
mask 名例：`hea_mask_L3_rx_ry_rz_cz_linear_ry_rz`。
**关键规模特性**：候选数基本不随 qubit 数增长（6q≈108，15q 预计仍一两百）；暴涨的是单候选 VQE 成本（∝2^n）。

### 0.3 当前 pipeline
```
Stage 1  zero-cost 结构过滤 + 多样性候选池（分层配额 + expr/train 下限 + diversity）
Stage 2  Hamiltonian-aware 候选保留（Pauli profile → priority seeds，无条件进终审）
Final    multi-trial fair VQE 排名（最终裁判）
```

### 0.4 名词
| 名词 | 含义 |
|------|------|
| proxy | 不完整训练就能算的便宜评分 |
| fair VQE | 给候选公平预算、真正跑参数优化得能量 |
| n_trials | 同结构独立优化轨迹数（不同 θ 初始化）|
| theta_init_mode | **参数角 θ** 初始化方式（量子初态恒为 \|0…0⟩）|
| reference_energy | 基态能量金标准（小规模 dense diag；大规模 TFIM free-fermion 解析解）|
| delta_ref | best−reference，越小越好；同时报 mHa 与 mHa/qubit |
| LF50/LF400/LF_improvement | 低保真信号：≈50 evals / ≈400 evals / 二者之差。**非 zero-cost，计入 selection cost** |
| 优化器 | `scipy.optimize.minimize(method="COBYLA")`，`aicir/qas/vqe_hea_demo.py` |

### 0.5 字段口径冻结
- **量子初态**：恒为 `|0…0⟩`，非变量。
- **参数角初始化** `theta_init_mode = random_uniform_pi`：所有 θ_j ~ U[−π,π]。
  - **zero start 仅作诊断项**（单独 run 可用于观察"从平凡点出发"行为），**禁止混入最终 fair ranking**。
- **轨迹统一为 `n_trials`**：固定 `fair_n_starts = 1`，`n_trials` = 独立轨迹数。
  - ⚠️ 合并前置检查：现有 `fair_n_starts` 与 `repeats` 可能走不同代码路径（历史上 3 vs 1 曾致结果不一致），合并前须验证**数值等价**。

### 0.6 已确认结论（不再推翻）
1. **4q**：ref=−3.427034，best=−3.423114（`L3_rx_ry_rz_rzz_ring_ry_rz`），delta_ref≈3.9 mHa。
2. **6q**：ref=−5.522030，best≈−5.5096（`L2_rx_ry_rz_cz_linear`），delta_ref≈14 mHa；三套独立流程都收敛到 CZ+L2/L3、都卡 ~14 mHa。
3. **复杂结构不稳定**：大 n_params 同结构不同 θ 能量差 40+ mHa；根因 COBYLA 高维非凸对初始点敏感 ⇒ 单次不可信，**多轨迹、以 best 排名**。
4. **proxy 不可靠**：LF50/LF400/improvement 不能精排，LF400 偏好简单快收敛结构；当前仅 hard-fail 淘汰。
5. **最优族随规模漂移**：4q RZZ 最强，6q CZ 最强、RZZ 掉到 12 名后 ⇒ "哪种结构好""哪个 proxy 可信"都 system-size dependent。
6. **实现风险**：θ 若非真随机，`n_trials` 失真（std=0）。Phase 0 必排查。

---

## 1. 本协议解决的问题

### 1.1 核心动机（"重复求解"悖论）
```
VQE 里 fair VQE 算出的能量本身就是科学终点（基态能量）。
为找"最低能量结构"，把所有结构的基态能量都算了一遍。
最优候选一旦算完，问题已答；继续枚举 = 为"确认它最优"付指数代价。
```
→ **目标：用便宜 predictor 规避"枚举所有候选"，只对少数候选跑真实 VQE。**

### 1.2 两条 scaling 轴（汇报必须分开）
```
轴 A — selection cost：枚举随规模越来越贵 → predictor 解决。主线。
轴 B — performance ceiling：delta_ref 随 n 变差（4q 3.9→6q 14 mHa）
        → predictor 不解决，是优化器 + HEA 表达能力的独立限制。
```

### 1.3 目标规模（tier）
```
4/6/8q   校准层：全枚举 + 统一预算，训练 predictor + 验 regret
15q      外推闸门：尽量全枚举/高覆盖验证跨尺度外推
20–22q   主部署：校正后 predictor 选 top-K（主叙事）
26–28q   stretch / NPU
```
> 内存（complex128=16B）：15q≈512KB，20q≈16MB，25q≈512MB，28q≈4GB，30q≈16GB。时间每加 1 qubit 翻倍。

---

## 2. 核心方案：小规模训练/校准 → 大规模部署（方案丙）

```
┌──────────────────────────────────────────────────────────────┐
│ 4/6/8q  可全枚举                                               │
│  · 全枚举 × 统一预算(统一 n_trials + 统一 optimizer budget)     │
│    → ground-truth                                             │
│  · 产 predictor 训练标签（禁止 halving）                       │
│  · 量 regret，证明不漏好电路                                   │
└──────────────────────────────────────────────────────────────┘
                     ↓ predictor（特征含 n_qubits）
┌──────────────────────────────────────────────────────────────┐
│ 15q  外推闸门                                                  │
│  · 若全枚举 → 验 regret                                        │
│  · 若不可全枚举 → high-coverage validation + rejected audit    │
│    （不叫 regret；成本计入 selection cost）                     │
│  · drift 大 → 在此做 few-shot 校正                             │
└──────────────────────────────────────────────────────────────┘
                     ↓ 校正后 predictor
┌──────────────────────────────────────────────────────────────┐
│ 20–22q 主部署 / 26–28q stretch                                │
│  · 全候选算 proxy → predictor 排序 → top-K                     │
│  · 仅 top-K 跑 multi-trial fair VQE                            │
│  · few-shot 回流校正 + 小规模最优族作 priority seeds 注入       │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 三个硬问题

### 硬问题 1：标签"鸡生蛋"
predictor 需大规模真实标签，而那正是想避免的昂贵计算。
**出路**：few-shot（少量但严格 multi-trial），质量 > 数量。大规模标注前先 **benchmark 单候选单轨迹耗时**（§6 B1），别假设 NPU 线性加速。

### 硬问题 2：跨尺度漂移（drift）—— 约束所有 predictor 形态
drift 大时问题不是"换更复杂模型"，而是**训练分布不够**。

**Migration 判据（默认阈值，可作输入变量）**：
```
family Spearman ≥ 0.6  且  top-K family overlap ≥ 0.5
  → 允许弱外推（n_qubits 当普通特征）
否则
  → 禁止零标签外推；目标尺度必须 few-shot 校正
    （MoE/GIN/tabular 只是模型形态，不替代目标尺度标签）
```

**MoE 启用触发规则（两条件 AND）**：
```
启用 MoE 当且仅当：
  条件1（drift 信号）：
    Migration < 0.6  或  top-K family overlap < 0.5
    或 proxy Spearman 随 n 明显翻转/下降
  且
  条件2（简单模型不够）：
    普通 tabular/GIN 在 15q 闸门不稳：
      regret_15q > 2 mHa
      或 top-K audit 发现误杀
      或 bootstrap/retrain top-K overlap < 0.5
```
两条件不同时满足 → 不上 MoE（VAE 始终 future work）。

### 硬问题 3：regret（有没有漏掉好电路）
> 澄清：非"训练集必须含最优"。训练集覆盖 L1–L4×各种门、学到趋势即可。
> regret 解决"predictor 跳过枚举，怎么证明没漏真最优"。

**度量（仅可枚举尺度），符号修正**：
```
regret(n)    = E_best_in_predictor_topK(n) − E_global_best(n) ≥ 0
Overlap@K(n) = |predictor_topK ∩ GT_topK| / K
```
- 同时报 per-qubit regret = regret(n)/n。
- **大规模无全枚举 top-1 ⇒ 不写 true regret**，只做 rejected/uncertain audit 作旁证。
- **15q 表述**：`若全枚举 → 验 regret；否则 high-coverage validation + rejected audit`。
- 判据：小规模 `regret(n) ≤ 2 mHa`（主）/ `≤ 5 mHa`（宽松）且不随 n 上升。

---

## 4. 分阶段执行

> Phase 0–1 零 ML，最先做。

### Phase 0 —— 修 bug + 冻结协议 + reference 对齐
1. **排查 θ 随机性**（结论 6）：确认 `theta_init_mode=random_uniform_pi` 每轨迹不同种子；复现 std=0 则修复。
2. **冻结 fair VQE 协议**：见 §5。输出 best/mean/std/worst/SR/delta_ref/n_params/**nfev**。
3. **冻结 optimizer budget**：§5 默认值；**必记 nfev**（COBYLA 可能提前停），论文报 nfev 分布。
4. **数值精度**：final label `complex128`；screening `complex64`，**不作 mHa 级最终结论，且不参与 reference validation**。
5. **reference 对齐**（地基，用 float64/complex128）：
   ```
   assert abs(E_dense − E_free_fermion) < 1e-9   # n = 4, 6, 8
   ```
   锁死：Hamiltonian 符号、J/h、OBC/PBC 边界、常数项、Pauli 归一化；**PBC free-fermion 注意 parity sector**。通过后 20q+ 才用解析解。

### Phase 1 —— 漂移度量（零 ML，决定外推策略）
- 输入：n∈{4,6,8}（10q 仅触发时加，见 §6 B4），每候选跑 **统一预算** multi-trial fair VQE（禁 halving），得 GT(n)+全 proxy。
- 三信号：① ρ_p(n)=Spearman(proxy 排名,GT(n))；② family Migration(n_i,n_j)；③ top-K family overlap。
- 输出：proxy×size 相关性表 + family migration 矩阵 + top-K 重叠曲线。
- 判据：硬问题 2 的 Migration 默认阈值。

### Phase 2 —— 标签生产
- **小规模（4/6/8/15）**：统一预算全枚举/高覆盖，**禁 halving**。
- **大规模（20q+）few-shot/主动标注**：用 successive halving 分配预算：
  ```
  all×2 → top50%×5(total) → top25%×12(total) → top5×30(total)   # 默认可调
  ```
- 记录（§6 B5）：每 candidate/seed 独立落盘（seed/nfev/walltime/best trace），便于 NPU 多卡调度与失败恢复。

### Phase 3 —— predictor + regret 验证
- **形态（渐进）**：v1 tabular/pairwise ranker（先）→ v2 GIN（对照，小样本易过拟合，非低风险）→ MoE（仅满足 §3 两条件）→ VAE（future）。
- **标签**：multi-trial best 的**相对序**，只保留置信对 `gap > k·pooled_std`（默认 k=2，敏感性 1.5/3）。
- **特征**（先 mask 结构特征，DAG 后置）：
  ```
  结构：n_params, layers, depth, 2q_gate_count, entangler_density,
        rotation_count, pattern, entangler_type, topological_width, topological_depth
  proxy：expressibility, trainability
  低保真(可选，计入 selection cost 并 ablation)：LF50, LF400, LF_improvement
  规模：n_qubits
  ```
- **regret 协议**：小规模算 regret(n)/Overlap@K(n)/per-qubit regret。
- 判据：regret(n) ≤ 2 mHa（主）且不随 n 上升。

### Phase 4 —— 大规模部署
- 全候选算 proxy → predictor 排序 → `K = min(N, max(10, ceil(0.1N)))`（报 K=5/10/20 曲线）→ 仅 top-K 跑 multi-trial fair VQE → few-shot 回流校正 → 小规模最优族作 priority seeds 注入。
- **15q 闸门**：见 §6 B3 切换规则。
- **成功判据（三口径全报）**：
  ```
  VQE_trajectory_saving / wall_clock_saving / proxy_cost_included_saving
  + 大规模抽查 5–10 个 rejected/uncertain，确认未误杀
  ```

### Phase 5（可选，轴 B，独立汇报）
- 两阶段优化器（SPSA 粗搜 + COBYLA 精修）提升复杂结构 SR；加层/更宽 rotation 测天花板。future work。

---

## 5. 冻结协议表（默认值已填）

| 项 | 字段 / 规则 | 默认值（可输入变量覆盖）|
|----|------------|---------------------------|
| 目标规模 | tier | 4/6/8 校准 · 15 闸门 · 20–22 主 · 26–28 stretch |
| 小规模集 | calib_scales | {4,6,8}（10q 触发时加）|
| θ 初始化 | theta_init_mode | **`random_uniform_pi`**（θ_j~U[−π,π]）；zero start 仅诊断 |
| 轨迹 | fair_n_starts / n_trials | `fair_n_starts=1`；`n_trials` 见下 |
| n_trials 自适应 | by n_params | ≤20→8；20–40→12；>40→20；top-5→32 |
| **optimizer** | COBYLA budget | `screening_maxfev=max(500,80·n_params)`（保守默认；benchmark 后可下调至 `max(300,50·n_params)`）；`final_maxfev=max(1000,200·n_params)`；`rhobeg=1.0`；`tol=1e-6`；**必记 nfev** |
| **小规模预算口径** | 双轨 | 主：per-parameter budget；旁：**equal-maxfev stress test**（防"复杂结构拿了更多评估"质疑）|
| regret 阈值 | regret_tau | 主 ≤2 mHa；宽松 ≤5 mHa；报 per-qubit |
| success_rate | 双轨 | `absolute_SR_tau=P(E<ref+tau)`（同尺度，tau=10/20mHa 都报，主文 20）；`posthoc_basin_SR_margin=P(E<E_best_observed(n)+margin)`（事后统计，**不可在线决策**）|
| Migration 阈值 | extrapolation gate | `Spearman≥0.6 且 topK_overlap≥0.5 → 弱外推；否则 few-shot 校正` |
| MoE | 触发 | §3 两条件 AND |
| halving | 用途 | **仅大规模**：all×2→top50%×5→top25%×12→top5×30 |
| predictor | 路线 | tabular/pairwise → GIN → MoE(仅触发) → VAE(future) |
| 置信对 | k | 2（敏感性 1.5/3）|
| 图编码 | DAG | 后置；先 mask 结构特征 |
| proxy 集 | features | 见 Phase 3；LF50/LF400/LF_improvement 可选且计成本 |
| top-K | K | `min(N, max(10, ceil(0.1N)))`；报 5/10/20 曲线 |
| 成功口径 | savings | trajectory + wall-clock + proxy-cost-included；audit 5–10 |
| reference | source / dtype | 小规模 dense diag；大规模 TFIM free-fermion；对齐 1e-9，用 float64/complex128（screening complex64 不参与）|
| 精度 | dtype | final `complex128`；screening `complex64` |
| 并行/断点 | logging | 每 candidate/seed 独立落盘：seed/nfev/walltime/best trace |

---

## 6. Benchmark / 触发项（已拍板）

### B1 —— 单候选耗时 benchmark（必做）
对 n=4/6/8/15/20，按真实 candidate pool 各选 **3 类结构**：
```
light  : n_params ≈ P10（若 P10 无意义则用 P25）
middle : n_params ≈ P50
heavy  : n_params ≈ P90/Pmax
优先选 pipeline 会保留 / 历史表现不差的 family
```
每结构跑 1 trial，final `complex128`，COBYLA 用 `final_maxfev`。太慢则再测一遍 screening budget。
记录：`walltime / nfev / time_per_eval / best_energy`。用于估最终标注成本 + 低保真特征成本 + NPU 并行收益。

### B2 —— optimizer budget（已冻结，见 §5）
benchmark 后只允许整体系数调整；硬记录 nfev 分布。

### B3 —— 15q 闸门切换
```
if estimated_full_enum_walltime <= 48h (且不超可用预算):
    15q full enum  → 验 regret
else:
    15q stratified high-coverage validation + rejected audit
    分层至少覆盖 entangler × layers × pattern × rotation_set，
    并额外含 predictor top-K + Hamiltonian priority seeds
    + 被 predictor 判差的 5–10 个 audit 候选
```

### B4 —— 10q calib（默认不加，异常触发）
```
add 10q if:
  Migration(6,8) 与 Migration(4,6) 趋势冲突
  或 proxy Spearman 在 8q 突然翻转
  或 15q 闸门 audit 发现明显误杀
  或 8q regret > 2 mHa 或接近 5 mHa
```

### B5 —— 并行/断点（落实到记录格式）
每 candidate/seed 独立落盘，记 seed/nfev/walltime/best trace。

---

## 7. 一页纸总览

```
动机：VQE 里"找最优结构"= 把所有结构基态能量算了一遍 → predictor 规避枚举
两轴：A 选择成本(predictor，主线) / B 性能天花板(优化器+表达能力，另说)
路线(方案丙)：4/6/8 训练+统一预算GT+验regret → 15q外推闸门 → 20-22主部署 → 26-28 stretch
三硬骨头：① 标签 few-shot非全量 ② drift大则目标尺度必须few-shot校正(换模型不替代标签)
          ③ regret小规模证≤2mHa(E_topK_best − E_global_best ≥ 0)，大规模只audit
护栏：θ=random_uniform_pi(zero仅诊断)·统一optimizer budget·TFIM解析解先对齐1e-9
      ·final complex128·LF50/LF400计入成本·MoE两条件AND触发
执行序：P0修bug+冻协议+对齐reference → P1漂移度量(零ML) → P2标签(小统一/大halving)
        → P3 tabular先→GIN→MoE(仅触发)+regret → P4部署+audit →(可选)P5抬天花板
先做且零风险：Phase 0 与 Phase 1
原则：冻结形式，不冻结常数
```