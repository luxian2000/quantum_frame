"""MaxCut Ising 哈密顿量与对应的 VQE 基态线路（自动生成）。

本文件由 ``demos/MaxCut/maxcut.py`` 生成，请勿手工编辑。

包含：
- 随机图的节点数与边集；
- MaxCut Ising 哈密顿量项 ``MAXCUT_HAMILTONIAN_TERMS``；
- ``aicir.qas.supernet`` 搜索 + 微调得到的 VQE 线路 ``VQE_CIRCUIT_JSON``；
- 重建辅助函数 ``build_hamiltonian()`` / ``build_vqe_circuit()``。
"""

from __future__ import annotations

from aicir import Circuit
from aicir.channel.operators import Hamiltonian
from aicir.core.io.json_io import circuit_from_json

# ── 随机图 ────────────────────────────────────────────────────────────────
N_NODES = 5
# (节点 i, 节点 j, 边权 w)
EDGES = [(0, 1, 1.0),
 (0, 2, 1.0),
 (0, 4, 1.0),
 (1, 2, 1.0),
 (1, 3, 1.0),
 (1, 4, 1.0),
 (2, 3, 1.0),
 (2, 4, 1.0),
 (3, 4, 1.0)]

# ── MaxCut Ising 哈密顿量 ───────────────────────────────────────────────────
# 项格式：("ZZ", [i, j], coeff) 或 ("I"*N_NODES, const)。
MAXCUT_HAMILTONIAN_TERMS = [('ZZ', [0, 1], 0.5),
 ('ZZ', [0, 2], 0.5),
 ('ZZ', [0, 4], 0.5),
 ('ZZ', [1, 2], 0.5),
 ('ZZ', [1, 3], 0.5),
 ('ZZ', [1, 4], 0.5),
 ('ZZ', [2, 3], 0.5),
 ('ZZ', [2, 4], 0.5),
 ('ZZ', [3, 4], 0.5),
 ('IIIII', -4.5)]


def build_hamiltonian() -> Hamiltonian:
    """重建 MaxCut Ising 哈密顿量。"""
    return Hamiltonian(n_qubits=N_NODES, terms=MAXCUT_HAMILTONIAN_TERMS)


# ── VQE 基态线路（QAS supernet 搜索并微调）────────────────────────────────────
# 指标（基于归一化态重新计算）：vqe_energy 为线路能量，exact_ground_energy 为
# 精确基态能量，max_cut/achieved_cut 为最大割与求得割，approx_ratio 为近似比。
VQE_METRICS = {'vqe_energy': -5.5,
 'exact_ground_energy': -6.0,
 'max_cut': 6.0,
 'achieved_cut': 5.5,
 'approx_ratio': 0.9166666666666666,
 'n_gates': 31}

# 线路以 circuit-JSON 形式记录，可用 circuit_from_json 无损重建。
VQE_CIRCUIT_JSON = ('{\n'
 '  "format": "aicir.circuit",\n'
 '  "version": "1.0",\n'
 '  "n_qubits": 5,\n'
 '  "gates": [\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 0,\n'
 '      "parameter": -0.14245599508285522\n'
 '    },\n'
 '    {\n'
 '      "type": "hadamard",\n'
 '      "target_qubit": 1\n'
 '    },\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 2,\n'
 '      "parameter": -0.14023266732692719\n'
 '    },\n'
 '    {\n'
 '      "type": "rx",\n'
 '      "target_qubit": 3,\n'
 '      "parameter": -4.76113855256699e-05\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 2,\n'
 '      "control_qubits": [\n'
 '        1\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "rzz",\n'
 '      "qubit_1": 1,\n'
 '      "qubit_2": 3,\n'
 '      "parameter": 0.009498202241957188\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 4,\n'
 '      "control_qubits": [\n'
 '        2\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "rx",\n'
 '      "target_qubit": 0,\n'
 '      "parameter": -0.00035489312722347677\n'
 '    },\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 1,\n'
 '      "parameter": 0.028388017788529396\n'
 '    },\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 2,\n'
 '      "parameter": -0.02040782757103443\n'
 '    },\n'
 '    {\n'
 '      "type": "ry",\n'
 '      "target_qubit": 3,\n'
 '      "parameter": -1.5712645053863525\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 1,\n'
 '      "control_qubits": [\n'
 '        0\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "rzz",\n'
 '      "qubit_1": 0,\n'
 '      "qubit_2": 2,\n'
 '      "parameter": -0.01722477376461029\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 4,\n'
 '      "control_qubits": [\n'
 '        2\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 0,\n'
 '      "parameter": -0.28282079100608826\n'
 '    },\n'
 '    {\n'
 '      "type": "hadamard",\n'
 '      "target_qubit": 1\n'
 '    },\n'
 '    {\n'
 '      "type": "ry",\n'
 '      "target_qubit": 2,\n'
 '      "parameter": 1.5716181993484497\n'
 '    },\n'
 '    {\n'
 '      "type": "rzz",\n'
 '      "qubit_1": 0,\n'
 '      "qubit_2": 2,\n'
 '      "parameter": 0.031485386192798615\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 3,\n'
 '      "control_qubits": [\n'
 '        2\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 4,\n'
 '      "control_qubits": [\n'
 '        2\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 4,\n'
 '      "control_qubits": [\n'
 '        3\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 0,\n'
 '      "parameter": 0.29442617297172546\n'
 '    },\n'
 '    {\n'
 '      "type": "rz",\n'
 '      "target_qubit": 2,\n'
 '      "parameter": -0.2567024230957031\n'
 '    },\n'
 '    {\n'
 '      "type": "hadamard",\n'
 '      "target_qubit": 3\n'
 '    },\n'
 '    {\n'
 '      "type": "hadamard",\n'
 '      "target_qubit": 4\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 1,\n'
 '      "control_qubits": [\n'
 '        0\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 2,\n'
 '      "control_qubits": [\n'
 '        0\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "cx",\n'
 '      "target_qubit": 2,\n'
 '      "control_qubits": [\n'
 '        1\n'
 '      ],\n'
 '      "control_states": [\n'
 '        1\n'
 '      ]\n'
 '    },\n'
 '    {\n'
 '      "type": "rzz",\n'
 '      "qubit_1": 1,\n'
 '      "qubit_2": 3,\n'
 '      "parameter": -0.1543472409248352\n'
 '    },\n'
 '    {\n'
 '      "type": "rzz",\n'
 '      "qubit_1": 2,\n'
 '      "qubit_2": 4,\n'
 '      "parameter": -0.11353835463523865\n'
 '    },\n'
 '    {\n'
 '      "type": "rzz",\n'
 '      "qubit_1": 3,\n'
 '      "qubit_2": 4,\n'
 '      "parameter": -0.1823810338973999\n'
 '    }\n'
 '  ]\n'
 '}')


def build_vqe_circuit() -> Circuit:
    """重建 QAS 搜索得到的 VQE 基态线路。"""
    return circuit_from_json(VQE_CIRCUIT_JSON)
