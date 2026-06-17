# VQE-QAS v3 当前进度与 TODO

来源：对话 `019eaa4a-faff-7923-8cf3-1c427b6e6d47` 的线程摘要、当前仓库代码状态，以及正在运行的 NPU full enumeration 任务。

## 现在进度

1. 方法主线已冻结为 v3：先做 Phase 0 reference/随机性/预算协议，再做 Phase 1 小规模 full enumeration 与漂移度量，之后才进入 predictor/regret 和大规模 top-K 部署。

2. 已确认的实验结论不再推翻：4q 最优族偏 RZZ，delta_ref 约 3.9 mHa；6q 最优族偏 CZ L2/L3，delta_ref 约 14 mHa；复杂结构对初始参数敏感；LF50/LF400/improvement 不能做精排，只能作为筛选或消融信号。

3. 代码已经有 v3 执行入口：`demos/vqe_tfim_v3_scaling.py` 负责 Phase 0 reference 对齐和 4/6/8q uniform-budget full enumeration；`demos/run_vqe_tfim_v3_npu_full.sh` 负责按 NPU 卡数切 shard 并跑 full enumeration；`demos/vqe_tfim_v3_phase1_analysis.py` 负责合并 enumeration CSV/shard partial CSV 并输出 family migration 分析。

4. NPU full enumeration 仍在跑。当前不应改动正在跑任务的输出目录、参数或 checkpoint 格式；独立开发只做“吃 CSV 的后处理代码”和文档整理。

5. 本轮新增独立代码：`demos/vqe_tfim_v3_regret_analysis.py`。它不跑 VQE，只读取 `phase1_uniform_enum_*q*.csv` 或 shard CSV/partial CSV，计算 family-transfer、exact-name-transfer 或外部 predictor CSV 的 top-K regret/Overlap@K。等 NPU 结果落盘后可直接运行。

## 立即 TODO

1. 等 NPU full enumeration 结束后，先运行 Phase 1 合并与漂移分析：

```bash
python aicir/qas/demos/vqe_tfim_v3_phase1_analysis.py \
  --input-dir outputs/vqe_tfim_v3_scaling_npu_full \
  --output-dir outputs/vqe_tfim_v3_scaling_npu_full \
  --scales 4,6,8
```

2. 运行新增的 transfer regret 分析，先用最简单的 family-transfer baseline 做 sanity check：

```bash
python aicir/qas/demos/vqe_tfim_v3_regret_analysis.py \
  --input-dir outputs/vqe_tfim_v3_scaling_npu_full \
  --output-dir outputs/vqe_tfim_v3_scaling_npu_full \
  --scales 4,6,8 \
  --predictor family-transfer
```

3. 再跑 exact-name-transfer baseline，对比“同名 mask 跨尺度迁移”和“family 跨尺度迁移”哪个更稳：

```bash
python aicir/qas/demos/vqe_tfim_v3_regret_analysis.py \
  --input-dir outputs/vqe_tfim_v3_scaling_npu_full \
  --output-dir outputs/vqe_tfim_v3_scaling_npu_full \
  --scales 4,6,8 \
  --predictor exact-name-transfer
```

4. 根据 `phase1_migration_analysis.md` 和 `phase3_transfer_regret.md` 判定下一步：
   - 如果 family Spearman >= 0.6 且 top-K family overlap >= 0.5，同时 regret <= 2 mHa，可以先走简单 tabular/pairwise predictor。
   - 如果 drift 明显或 best candidate 的 predicted rank 很差，15q/20q 必须做 few-shot 校正，不能零标签外推。

5. Phase 3 代码的下一块独立任务：写 `tabular/pairwise` predictor 的训练与 leave-one-scale-out 验证，只读已有 CSV/features，不启动 VQE。

6. Phase 4 前的必要任务：补一个 rejected/uncertain audit 选择器，输入 predictor 排名，自动挑 5-10 个“模型判差但结构覆盖关键族”的候选用于大规模抽查。

## 暂缓 TODO

1. 不在 NPU full enumeration 期间修改 `run_vqe_tfim_v3_npu_full.sh` 的默认参数。

2. 不在 4/6/8q 结果未完整合并前训练 GIN/MoE。当前优先用 tabular 和 transfer baseline 建立 regret 口径。

3. 不把 SPSA+COBYLA 或扩展 HEA 空间混入主线。它们属于 Phase 5，用于抬高 performance ceiling，不解决 selection cost。

4. 不把 LF50/LF400 当成 zero-cost 或最终 predictor 结论；若使用，必须计入 selection cost，并做 ablation。

## 输出文件口径

`phase1_migration_analysis.md/json`：跨尺度 family drift。

`phase3_transfer_regret.md/csv/json`：跨尺度 transfer predictor 的 regret、Overlap@K、全局最优在预测排名中的位置。

`phase1_uniform_enum_*q.csv`：每个尺度的 ground-truth ranking，后续 predictor/regret 都以此为标签来源。
