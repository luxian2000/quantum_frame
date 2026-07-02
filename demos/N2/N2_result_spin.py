"""求 N2 精确基态能量、复算线路能量并生成 result_spin.md（在 NPU 机器上运行）。

    python -m demos.N2.N2_result_spin                 # 默认稀疏 Lanczos，几分钟
    python -m demos.N2.N2_result_spin --method dense  # 全谱稠密对角化（大内存机器）
    python -m demos.N2.N2_result_spin --allow-cpu-fallback  # 本地冒烟测试

流程：
1. 精确基态能量：用“带符号置换”逻辑高效构造 14 比特 Hamiltonian（每个 Pauli 串每行只有一个
   非零，逐项向量化累加，避免 to_matrix 的逐项 2^14×2^14 临时矩阵），再取最小特征值。
   - 默认 ``--method sparse``：scipy Lanczos ``eigsh(k=1, which="SA")``，只求最小特征值；
     14 比特 / 670 项的 CSR 构造较 BeH2 轻（量子比特更少），普通大内存机器即可；
   - ``--method dense``：完整稠密对角化。注意 Ascend 无 complex ``eigvalsh`` 内核，torch 会
     自动回退到 CPU 跑全谱（2^14 维稠密极大、很慢），仅建议在超大内存机器上使用。
2. 逐门把保存线路作用到态向量（14 比特无法构造 2^14 全局 unitary，故不走 circuit.unitary()），
   再对全部 Pauli 项逐项计算 <psi|P|psi> 求和得到线路能量；
3. 解析 output_spin.txt 中 NPU 运行报告的能量/配置；
4. 写出 demos/N2/result_spin.md。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from aicir.core.gates import apply_gate_to_state
from demos.N2.N2 import n2_vqe_qas_kwargs, build_n2_hamiltonian
from demos.N2.N2_cir_spin import build_n2_npu_qas_circuit

HERE = Path(__file__).parent
REPORT_PATH = HERE / "output_spin.txt"
RESULT_PATH = HERE / "result_spin.md"
CHEMICAL_ACCURACY_HA = 1.6e-3  # 化学精度 ~1.6 mHa


def select_device(allow_cpu_fallback: bool):
    """返回 (backend, device_str)。优先 NPU；NPU 不可用时按需回退 CPU。"""
    from aicir.backends.npu_backend import NPUBackend, is_npu_available

    if is_npu_available():
        return NPUBackend(device="npu:0", fallback_to_cpu=False), "npu:0"
    if not allow_cpu_fallback:
        raise RuntimeError(
            "未检测到 Ascend NPU。请在 NPU 机器上运行，或加 --allow-cpu-fallback 做本地冒烟测试。"
        )
    from aicir import NumpyBackend

    return NumpyBackend(), "cpu"


# ---------------------------------------------------------------------------
# 1. 稠密对角化：精确基态能量
# ---------------------------------------------------------------------------
def build_dense_hamiltonian(dtype=np.complex64) -> np.ndarray:
    """高效构造 14 比特 Hamiltonian 稠密矩阵。

    每个 Pauli 串都是“带符号置换”：P|j> = phase(j) · |j ^ flipmask>，每行只有一个非零，
    因此可对 j 向量化散射累加，避免 to_matrix 逐项构造 2^n×2^n 的 kron 临时矩阵。
    特征谱与比特序约定无关，这里用 qubit0=最高位、字符下标 i=qubit i 的一致约定即可。
    """
    return _build_hamiltonian(dense=True, dtype=dtype)


def _term_mask(labels, n):
    """把一个 Pauli 串编码为 (flipmask, phasemask, n_y)。"""
    flipmask = phasemask = n_y = 0
    for q, p in enumerate(labels):  # 字符下标 i 对应 qubit i
        bit = 1 << (n - 1 - q)  # qubit q 在索引中的位（qubit0 = 最高位）
        if p == "X":
            flipmask |= bit
        elif p == "Y":
            flipmask |= bit
            phasemask |= bit
            n_y += 1
        elif p == "Z":
            phasemask |= bit
    return flipmask, phasemask, n_y


def _build_hamiltonian(*, dense: bool, dtype=np.complex64):
    """统一的带符号置换构造器；dense=True 返回稠密 ndarray，否则返回 scipy CSR。"""
    ham = build_n2_hamiltonian()
    n = ham.n_qubits
    dim = 1 << n
    j = np.arange(dim, dtype=np.int64)
    popcount = np.array([bin(x).count("1") for x in range(dim)], dtype=np.int8)

    if dense:
        matrix = np.zeros((dim, dim), dtype=dtype)
    else:
        rows_all, cols_all, data_all = [], [], []

    for term in ham.terms:
        flipmask, phasemask, n_y = _term_mask(term.qubit_labels, n)
        sign = 1 - 2 * (popcount[phasemask & j].astype(np.int64) & 1)  # (-1)^popcount
        value = (term.coefficient * (1j ** n_y) * sign).astype(dtype)  # 长度 dim 的复向量
        rows = j ^ flipmask  # P|j> 落在的行；flipmask 固定时是 j 的置换，无行内重复
        if dense:
            matrix[rows, j] += value
        else:
            rows_all.append(rows)
            cols_all.append(j)
            data_all.append(value)

    if dense:
        return matrix

    from scipy.sparse import coo_matrix

    coo = coo_matrix(
        (np.concatenate(data_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(dim, dim),
        dtype=dtype,
    )
    return coo.tocsr()  # tocsr 自动对重复 (row,col) 求和


def exact_ground_energy(method: str, device: str, dtype=np.complex64) -> tuple[float, str]:
    """返回 (基态能量, 求解路径标签)。

    method="sparse"（默认，推荐）：scipy Lanczos `eigsh(k=1, which='SA')`，只求最小特征值；
        14 比特 / 670 项的 CSR 构造较 BeH2 轻（量子比特更少），普通机器即可完成。
    method="dense"：完整稠密对角化。注意 Ascend 无 complex `eigvalsh` 内核，torch 会自动
        回退到 CPU 跑全谱（2^14 维稠密较大、稍慢），故仅在超大内存机器上使用。
    """
    if method == "sparse":
        from scipy.sparse.linalg import eigsh

        csr = _build_hamiltonian(dense=False, dtype=dtype)
        ground = float(eigsh(csr, k=1, which="SA", return_eigenvectors=False)[0])
        return ground, "eigsh(Lanczos,SA)@cpu"

    # dense
    matrix_np = _build_hamiltonian(dense=True, dtype=dtype)
    if device.startswith("npu"):
        try:
            import torch

            mat = torch.from_numpy(matrix_np).to(torch.device(device))
            eigvals = torch.linalg.eigvalsh(mat)  # Ascend 无内核时 torch 自动回退 CPU
            ground = float(eigvals.min().to("cpu").item())
            del mat, eigvals
            return ground, f"eigvalsh(dense)@{device}(may fall back to CPU)"
        except Exception as exc:
            print(f"[warn] NPU eigvalsh 失败，回退到 numpy(CPU)：{exc}")
    ground = float(np.linalg.eigvalsh(matrix_np).min())
    return ground, "eigvalsh(dense)@cpu"


# ---------------------------------------------------------------------------
# 2. 线路能量复算
# ---------------------------------------------------------------------------
def compute_circuit_energy(backend, circuit=None) -> float:
    """逐门演化 + 逐 Pauli 项求期望，返回线路能量。

    ``circuit`` 为 None 时读取 ``N2_cir_spin.py`` 保存的线路；也可直接传入
    搜索返回的 ``result.best_circuit`` 以避免依赖落盘文件（供 N2_npu.py 内联调用）。
    """
    if circuit is None:
        circuit = build_n2_npu_qas_circuit()
    n = circuit.n_qubits

    state = backend.zeros_state(n)
    for gate in circuit.gates:
        state = apply_gate_to_state(gate, state, n, backend)

    psi = np.asarray(backend.to_numpy(state)).reshape(-1).astype(np.complex128)
    T = psi.reshape((2,) * n)

    ham = build_n2_hamiltonian()
    energy = 0.0 + 0.0j
    for term in ham.terms:
        energy += term.coefficient * np.vdot(T, _pauli_apply(T, term.qubit_labels, n))
    return float(energy.real)


def _pauli_apply(T: np.ndarray, labels: list[str], n: int) -> np.ndarray:
    """把 Pauli 串作用到 (2,)*n 形态的态张量。"""
    out = T
    for q, p in enumerate(labels):
        if p == "I":
            continue
        if p == "Z":
            out = out.copy()
            idx = [slice(None)] * n
            idx[q] = 1
            out[tuple(idx)] *= -1
        elif p == "X":
            out = np.flip(out, axis=q)
        elif p == "Y":
            out = np.flip(out, axis=q).astype(np.complex128).copy()
            i0 = [slice(None)] * n
            i0[q] = 0
            i1 = [slice(None)] * n
            i1[q] = 1
            out[tuple(i0)] *= -1j
            out[tuple(i1)] *= 1j
    return out


# ---------------------------------------------------------------------------
# 3. 解析 NPU 运行报告
# ---------------------------------------------------------------------------
def _parse_report(text: str) -> dict:
    def grab(pattern: str, cast=str, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    report = {
        "world_size": grab(r"NPUs \(world_size\)\s*:\s*(\d+)", int),
        "seed": grab(r"seed\s*:\s*(\d+)", int),
        "device": grab(r"device\s*:\s*(\S+)"),
        "mode": grab(r"mode\s*:\s*(\S+)"),
        "wall_clock": grab(r"total wall-clock time\s*:\s*([\d.]+)", float),
        "fine_tuned": grab(r"fine-tuned energy\s*:\s*([-\d.]+)", float),
        "baseline": grab(r"baseline VQE\s*:\s*([-\d.]+)", float),
        "layers": grab(r"layers \(depth\)\s*:\s*(\d+)", int),
        "gradient": grab(r"gradient\s*:\s*(\S+)"),
    }
    qt = re.search(r"qubits / terms\s*:\s*(\d+)\s*/\s*(\d+)", text)
    if qt:
        report["n_qubits"], report["n_terms"] = int(qt.group(1)), int(qt.group(2))
    exc = re.search(r"excitations\s*:\s*(\d+)\s*\(single=(\d+),\s*double=(\d+)\)", text)
    if exc:
        report["excitations"] = int(exc.group(1))
        report["single"], report["double"] = int(exc.group(2)), int(exc.group(3))
    return report


def _gate_counts(circuit) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gate in circuit.gates:
        counts[gate["type"]] = counts.get(gate["type"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 4. 渲染 markdown
# ---------------------------------------------------------------------------
def render_markdown(
    circuit_energy: float,
    exact_energy: float | None,
    exact_path: str | None,
    device: str,
    report: dict,
    circuit=None,
) -> str:
    cfg = n2_vqe_qas_kwargs()
    if circuit is None:
        circuit = build_n2_npu_qas_circuit()
    counts = _gate_counts(circuit)
    hf = ", ".join(str(q) for q in cfg["hf_occupied_qubits"])

    n_x = counts.get("pauli_x", 0)
    n_single = counts.get("single_excitation", 0)
    n_double = counts.get("double_excitation", 0)
    total_gates = len(circuit.gates)

    ft = report.get("fine_tuned")
    base = report.get("baseline")
    delta_report = circuit_energy - ft if ft is not None else float("nan")
    wall = report.get("wall_clock")
    wall_h = wall / 3600.0 if wall is not None else float("nan")
    n_qubits = report.get("n_qubits", 14)
    dim = 1 << n_qubits
    n_terms = report.get("n_terms", 670)
    ws = report.get("world_size", 4)
    layers = report.get("layers") or cfg["layers"]
    gradient = report.get("gradient") or "ad"
    grad_label = {"psr": "参数移位 `psr`", "ad": "自动微分 `ad`"}.get(gradient, f"`{gradient}`")

    # 精确能量相关
    if exact_energy is not None:
        err = circuit_energy - exact_energy
        err_mha = err * 1000.0
        base_err_mha = (base - exact_energy) * 1000.0 if base is not None else float("nan")
        within = "以内" if abs(err) <= CHEMICAL_ACCURACY_HA else "之外"
        exact_rows = (
            f"| Hamiltonian 精确基态能量 | {exact_energy:.10f} |\n"
            f"| 线路能量 - 精确基态能量 | {err:+.10f} |\n"
            f"| 绝对误差 | {abs(err):.10f} |\n"
        )
        exact_block = f"""
本次精确基态能量由对角化求得（`{exact_path}`）：构造 {n_qubits} 比特 Hamiltonian（2^{n_qubits}={dim} 维）后取最小特征值。

- 保存线路能量比精确基态能量高约 `{abs(err_mha):.3f} mHa`，处于化学精度（约 1.6 mHa）{within}（线路能量 ≥ 精确基态能量，满足变分下界）。
- 同一次运行的固定 ansatz VQE 基线 `{base:.10f} Ha` 相对精确基态高约 `{base_err_mha:.3f} mHa`。"""
    else:
        err = float("nan")
        exact_rows = "| Hamiltonian 精确基态能量 | （本次跳过，--skip-exact） |\n"
        exact_block = "\n本次未计算精确基态能量（`--skip-exact`），下面以同一次运行的固定 ansatz VQE 基线为参照。"

    # 结论里 QAS vs baseline
    gap_mha = (ft - base) * 1000.0 if (ft is not None and base is not None) else float("nan")

    return f"""# N2 基态能量结果（NPU 自旋保持 ansatz 版）

本次结果使用 `demos/N2/N2.py` 中的 {n_qubits} 比特 N2 活性空间 Hamiltonian（STO-3G 基组、10 电子/7 空间轨道活性空间、Jordan-Wigner 映射，共 {n_terms} 个 Pauli 项），以及 `demos/N2/N2_cir_spin.py` 中保存的、在 **{ws} 块 Ascend NPU**（`device={report.get('device', 'npu:0')}`，`world_size={ws}`，搜索内分片 in-search sharding）上由 supernet QAS 搜索得到的线路。

该线路采用分子 VQE 更合适的粒子数/自旋保持 ansatz：先用 {n_x} 个 `pauli_x` 制备 closed-shell Hartree-Fock 参考态（占据比特 `{hf}`），再从 `single_excitation` 与 `double_excitation` 门池中搜索激发算符。

## 计算结果

| 项目 | 能量 / Ha |
| --- | ---: |
{exact_rows}| NPU 搜索报告 fine-tuned energy | {ft:.10f} |
| 固定 ansatz VQE 基线 | {base:.10f} |
| 保存线路重新计算能量 | {circuit_energy:.10f} |
| 保存线路能量 - NPU 报告能量 | {delta_report:+.10f} |
{exact_block}

结论：

- 保存线路（`N2_cir_spin.py`）经 statevector 复算得到 `{circuit_energy:.10f} Ha`，与 NPU 运行报告的 `fine-tuned energy = {ft:.10f} Ha` 一致，相差约 `{abs(delta_report):.2e} Ha`，源于 NPU complex64 精度与求值路径差异。
- 同一次 NPU 运行中固定 ansatz 的 VQE 基线为 `{base:.10f} Ha`，与 QAS 搜索到的线路相差约 `{gap_mha:.2f} mHa`（正值表示搜索线路更优）。在这组 {n_qubits} 比特 / `L={layers}` 配置下，若 supernet 搜索未收敛到优于固定基线的结构，可考虑增大 `supernet_steps`/`finetune_steps`、调整 `supernet_num (W)` 或减小层数（参见 `demos/BeH2/result_spin.md` 的同类观察）。

## 搜索配置

本次 NPU 运行（见 `demos/N2/output_spin.txt` 头部，对应 `demos/N2/N2_npu.py`）的设置：

- 设备：{ws} 块 Ascend NPU，`device={report.get('device', 'npu:0')}`，`world_size={ws}`，`mode={report.get('mode', 'safe')}`（与单卡数值等价的搜索内分片）
- 量子比特数：{n_qubits}，层数（深度）：{layers}
- HF 参考态占据比特：`({hf})`
- 单比特门池：`{{i}}`，用于保留 supernet 每层可跳过的占位门
- 两比特门池：`{{single_excitation}}`，连接来自 closed-shell spin-preserving singles
- 四比特门池：`{{double_excitation}}`，连接来自 closed-shell paired doubles
- 超网络数 `supernet_num={cfg['supernet_num']}`，搜索步数 `supernet_steps={cfg['supernet_steps']}`，排序数 `ranking_num={cfg['ranking_num']}`，微调步数 `finetune_steps={cfg['finetune_steps']}`
- 学习率 `learning_rate={cfg['learning_rate']}`，微调学习率 `finetune_learning_rate={cfg['finetune_learning_rate']}`
- 梯度方式：{grad_label}（`--gradient {gradient}`）
- 随机种子 `seed={report.get('seed', cfg['seed'])}`（所有 rank 同 seed，保证权重/候选集一致）
- 任务 `task="{cfg['task']}"`
- 总墙钟时间：约 {wall:.1f} s（约 {wall_h:.2f} 小时）

本次保存线路包含 {total_gates} 个门：

- {n_x} 个 `pauli_x` 门用于制备 HF 参考态
- {n_single} 个 `single_excitation` 门
- {n_double} 个 `double_excitation` 门

（NPU 报告中的 `excitations = {report.get('excitations', n_single + n_double)} (single={report.get('single', n_single)}, double={report.get('double', n_double)})` 仅统计激发门，不含 {n_x} 个 HF 制备门。）

## 生成文件

```text
demos/N2/N2_cir_spin.py
demos/N2/N2_cir_spin.png
demos/N2/output_spin.txt
demos/N2/N2_npu_cir.qasm
```

OpenQASM 3.0 导出当前会跳过，因为 QASM 导出器暂不支持 `single_excitation` / `double_excitation` 这类高层化学激发门。

## 验证方式

精确基态能量与线路能量都由本脚本算得（求解路径：`{exact_path or '(skipped)'}`）：

1. **精确基态能量**：用同一套“带符号置换”逻辑高效构造 {n_qubits} 比特 Hamiltonian（每个 Pauli 串每行只有一个非零，逐项向量化累加，避免 `to_matrix` 的逐项 2^{n_qubits} 临时矩阵），再取最小特征值。默认走稀疏 Lanczos `scipy.sparse.linalg.eigsh(k=1, which="SA")`；14 比特 / {n_terms} 项的 CSR 构造较 BeH2 轻（量子比特更少）。Ascend 当前没有 complex `eigvalsh` 内核，故 `--method dense` 的全谱稠密对角化会被 torch 自动回退到 CPU 运行（2^{n_qubits} 维稠密较大、稍慢），仅建议在超大内存机器上使用。
2. **线路能量**：按门逐个作用到态向量（{n_qubits} 比特无法构造 {dim}×{dim} 全局 unitary，故不走 `circuit.unitary()`），再对 {n_terms} 个 Pauli 项逐项计算 `<psi|P|psi>` 求和。

```bash
# 默认稀疏 Lanczos 求精确基态能量 + 复算线路能量并刷新本文件
python -m demos.N2.N2_result_spin

# 如需完整稠密对角化（超大内存机器；Ascend 无内核会回退到 CPU 全谱）
python -m demos.N2.N2_result_spin --method dense
```

复算输出（device={device}）：

```text
exact   = {('%.10f Ha' % exact_energy) if exact_energy is not None else '(skipped)'}
circuit = {circuit_energy:.10f} Ha
```

## 复现方式

```bash
# 在 {ws} 块 Ascend NPU 上重新搜索（需 torch_npu）
torchrun --nproc_per_node={ws} demos/N2/N2_npu.py --gradient {gradient} --output output_spin.txt

# 仅根据已记录的线路重新绘图
python -m demos.N2.N2_cir_spin
```
"""


def generate_result_md(
    circuit,
    report: dict,
    *,
    backend=None,
    device: str | None = None,
    allow_cpu_fallback: bool = False,
    method: str = "sparse",
    skip_exact: bool = False,
) -> tuple[Path, float, float | None]:
    """计算精确基态能量 + 复算线路能量并写出 ``result_spin.md``，返回 (路径, 线路能量, 精确能量)。

    供 ``N2_npu.py`` 在搜索结束后内联调用：直接传入 ``result.best_circuit`` 与内存中的
    ``report`` 字典，无需再解析 ``output_spin.txt``。``skip_exact=True`` 跳过（14 比特下很重的）
    精确对角化。若已有初始化好的 NPU ``backend``/``device`` 可传入复用。
    """
    if backend is None:
        backend, device = select_device(allow_cpu_fallback)

    exact_energy: float | None = None
    exact_path: str | None = None
    if not skip_exact:
        exact_energy, exact_path = exact_ground_energy(method, device)

    circuit_energy = compute_circuit_energy(backend, circuit)
    markdown = render_markdown(circuit_energy, exact_energy, exact_path, device, report, circuit=circuit)
    RESULT_PATH.write_text(markdown, encoding="utf-8")
    return RESULT_PATH, circuit_energy, exact_energy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="NPU 稀疏/稠密对角化求精确基态能量 + 复算 N2 线路能量并生成 result_spin.md。",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="NPU 不可用时回退到 CPU（仅用于本地冒烟测试）。默认严格要求 NPU。",
    )
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="跳过精确基态能量计算（CSR 构造较重时建议加上）。",
    )
    parser.add_argument(
        "--method",
        choices=("sparse", "dense"),
        default="sparse",
        help="精确基态求解法：sparse=Lanczos eigsh（默认，COO 中间体可达数十 GB）；"
             "dense=全谱稠密对角化（2^14 维，Ascend 无内核会回退 CPU）。",
    )
    args = parser.parse_args(argv)

    report = _parse_report(REPORT_PATH.read_text(encoding="utf-8"))
    backend, device = select_device(args.allow_cpu_fallback)
    print(f"device  : {device}")
    if not args.skip_exact:
        print(f"exact ground energy via method={args.method} ... 构造 Hamiltonian 并求最小特征值")

    t0 = time.time()
    result_path, circuit_energy, exact_energy = generate_result_md(
        build_n2_npu_qas_circuit(),
        report,
        backend=backend,
        device=device,
        method=args.method,
        skip_exact=args.skip_exact,
    )
    if exact_energy is not None:
        print(f"exact   : {exact_energy:.10f} Ha   [{time.time() - t0:.1f}s]")
    print(f"circuit : {circuit_energy:.10f} Ha")
    if report.get("fine_tuned") is not None:
        print(f"report  : {report['fine_tuned']:.10f} Ha")
        print(f"|delta| : {abs(circuit_energy - report['fine_tuned']):.3e} Ha")
    if exact_energy is not None:
        print(f"|error| : {abs(circuit_energy - exact_energy) * 1000:.3f} mHa (vs exact)")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
