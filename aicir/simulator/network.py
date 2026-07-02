"""把 Circuit 构建成带整数标签的张量网络（见 plan「局部矩阵约定」）。"""

from __future__ import annotations

import numpy as np

from ..core.gates import gate_tensors


def _basis_vector(bit: int):
    return np.array([1.0, 0.0] if int(bit) == 0 else [0.0, 1.0], dtype=np.complex64)


def build_network(circuit, backend, *, input_bits=None, output_spec=None):
    n = int(circuit.n_qubits)
    if input_bits is None:
        input_bits = [0] * n
    if output_spec is None:
        output_spec = [None] * n
    if len(input_bits) != n or len(output_spec) != n:
        raise ValueError("input_bits / output_spec 长度必须等于 n_qubits")

    tensors, indices = [], []
    counter = 0

    def fresh():
        nonlocal counter
        counter += 1
        return counter

    # 输入边界：每比特一个 |bit> 向量
    wire = [0] * n
    for q in range(n):
        vid = fresh()
        tensors.append(backend.cast(_basis_vector(input_bits[q])))
        indices.append((vid,))
        wire[q] = vid

    # 门节点：matrix reshape 成 (2,)*k(out) + (2,)*k(in)
    for gate in circuit.gates:
        for matrix, axes in gate_tensors(gate, backend):
            k = len(axes)
            node = backend.reshape(backend.cast(matrix), (2,) * (2 * k))
            out_ids = [fresh() for _ in range(k)]
            in_ids = [wire[a] for a in axes]
            tensors.append(node)
            indices.append(tuple(out_ids) + tuple(in_ids))
            for j, a in enumerate(axes):
                wire[a] = out_ids[j]

    # 输出边界：None 开放；0/1 接 <bit|
    open_indices = []
    for q in range(n):
        spec = output_spec[q]
        if spec is None:
            open_indices.append(wire[q])
        else:
            tensors.append(backend.cast(_basis_vector(int(spec))))
            indices.append((wire[q],))

    return tensors, indices, tuple(open_indices)
