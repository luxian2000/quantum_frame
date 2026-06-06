# H2O 基态能量对比结果

本次对比使用 `demos/H2O/H2O.py` 中的 6 比特 H2O 活性空间 Hamiltonian（STO-3G 基组、4 电子/3 空间轨道活性空间、Jordan-Wigner 映射），以及 `demos/H2O/H2O_cir.py` 中保存的 supernet QAS 搜索得到的线路。

## 计算结果

| 项目 | 能量 / Ha |
| --- | ---: |
| Hamiltonian 精确基态能量 | -6.159663677216 |
| 线路态能量 | -6.155271530151 |
| 线路能量 - 精确基态能量 | +0.004392147064 |
| 绝对误差 | 0.004392147064 |

结论：`H2O_cir.py` 中的线路得到的能量比精确基态能量高约 `4.39e-3 Ha`（约 4.39 mHa）。这符合变分法（Rayleigh-Ritz）的预期，即固定线路得到的能量不低于精确基态能量。

与 LiH 不同，H2O 是 6 比特、4 电子的关联基态，对从 `|000000>` 出发的浅层硬件高效 ansatz 而言显著更难。supernet 方法通过更深的层数、更密的双比特连接（最近邻 + 次近邻共 9 对）和更多超网络（`W=5`）将能量逼近到约 4.4 mHa，接近但尚未进入化学精度（1.6 mHa）。

## 搜索配置

`demos/H2O/H2O.py` 中 `h2o_vqe_qas_config` 的默认设置：

- 量子比特数：6，层数：6
- 单比特门池：`{i, h, rx, ry, rz}`
- 双比特门池：`{cx, rzz}`（`rzz` 为带可训练参数的纠缠门）
- 双比特连接：最近邻 + 次近邻，共 9 对
- 超网络数 `supernet_num=5`，搜索步数 `supernet_steps=250`，排序数 `ranking_num=80`，微调步数 `finetune_steps=250`
- 随机种子 `seed=2`

## 验证方式

线路能量通过两种方式计算，结果一致：

1. 逐门演化 statevector 后计算 `<psi|H|psi>`。
2. 使用 `Measure(..., observables={"H2O": H})` 计算 Hamiltonian 期望值。

验证命令：

```bash
PYTHONPATH=. pytest tests/test_h2o_demo.py -q
```

测试结果：

```text
3 passed in 0.51s
```

## 复现方式

```bash
# 重新搜索并生成 H2O_cir.py / H2O_cir.qasm / H2O_cir.png（约 1 分钟）
python -m demos.H2O.H2O

# 仅根据已记录的线路重新绘图
python -m demos.H2O.H2O_cir
```
