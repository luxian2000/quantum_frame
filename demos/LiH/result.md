# LiH 基态能量对比结果

本次对比使用 `demos/LiH/LiH.py` 中的 4 比特 LiH 活性空间 Hamiltonian，以及 `demos/LiH/LiH_cir.py` 中保存的固定 VQE/QAS 线路。

## 计算结果

| 项目 | 能量 / Ha |
| --- | ---: |
| Hamiltonian 精确基态能量 | -1.059311985970 |
| 线路态能量 | -1.059035062790 |
| 线路能量 - 精确基态能量 | +0.000276923180 |
| 绝对误差 | 0.000276923180 |

结论：`LiH_cir.py` 中的线路得到的能量比精确基态能量高约 `2.77e-4 Ha`。这符合变分法的预期，即固定线路得到的能量不低于精确基态能量，并且该线路已经较好地接近 LiH 活性空间模型的基态。

## 验证方式

线路能量通过两种方式计算，结果一致：

1. 逐门演化 statevector 后计算 `<psi|H|psi>`。
2. 使用 `Measure(..., observables={"LiH": H})` 计算 Hamiltonian 期望值。

验证命令：

```bash
PYTHONPATH=. pytest tests/test_lih_demo.py -q
```

测试结果：

```text
3 passed in 2.25s
```
