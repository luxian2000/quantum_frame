from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import VariableMetadata


@dataclass(frozen=True)
class IsingExport:
    """Ising 模型的强类型导出容器，字段语义对齐 :meth:`IsingModel.named` 等既有 dict payload。

    - ``linear``：线性项系数（旧 dict 的 ``"h"``），键的类型取决于来源
      （``IsingModel.to_export()`` 保留索引键；``Model.to_ising_export()`` 是变量名键）。
    - ``quadratic``：二次项系数（旧 dict 的 ``"J"``），键为二元组，类型规则同上。
    - ``offset``：常数偏移（旧 dict 的 ``"offset"``）。
    - ``variable_names``：变量名列表，转为 tuple 以配合 ``frozen=True``；不可用时为 ``None``。
    - ``variable_metadata``：``VariableMetadata`` 附加信息（角色/来源等）；不可用时为 ``None``。
    """

    linear: Mapping[Any, float]
    quadratic: Mapping[tuple[Any, Any], float]
    offset: float
    variable_names: tuple[str, ...] | None = None
    variable_metadata: tuple[VariableMetadata, ...] | None = None


@dataclass(frozen=True)
class IsingModel:
    h: dict[int, float]
    J: dict[tuple[int, int], float]
    offset: float = 0.0
    variable_names: list[str] | None = None
    variable_metadata: list[VariableMetadata] | None = None

    def named(self) -> dict[str, object]:
        """按变量名重映射 ``h``/``J`` 键，返回裸 dict。

        已弃用（deprecated）：新代码请使用类型化的 :meth:`to_export`
        （保留索引键）或 ``Model.to_ising_export()``（变量名键，字段语义与本方法一致）。
        """
        if self.variable_names is None:
            raise ValueError("Named Ising output requires variable_names.")
        return {
            "h": {self.variable_names[index]: coeff for index, coeff in self.h.items()},
            "J": {
                (self.variable_names[left], self.variable_names[right]): coeff
                for (left, right), coeff in self.J.items()
            },
            "offset": self.offset,
        }

    def to_export(self) -> IsingExport:
        """转换为强类型 :class:`IsingExport`，键语义与 ``self.h``/``self.J`` 完全一致（仅类型包装）。"""
        return IsingExport(
            linear=dict(self.h),
            quadratic=dict(self.J),
            offset=self.offset,
            variable_names=tuple(self.variable_names) if self.variable_names is not None else None,
            variable_metadata=(
                tuple(self.variable_metadata) if self.variable_metadata is not None else None
            ),
        )

    def to_qaoa_terms(self) -> list[QAOATerm]:
        terms = [QAOATerm((index,), coeff) for index, coeff in sorted(self.h.items())]
        terms.extend(QAOATerm((left, right), coeff) for (left, right), coeff in sorted(self.J.items()))
        return terms


@dataclass(frozen=True)
class QAOATerm:
    qubits: tuple[int, ...]
    coefficient: float

    @property
    def pauli(self) -> str:
        return "Z" * len(self.qubits)


def qubo_to_ising_indices(
    qubo: dict[tuple[int, int], float],
    offset: float = 0.0,
    variable_names: list[str] | None = None,
    variable_metadata: list[VariableMetadata] | None = None,
) -> IsingModel:
    h: dict[int, float] = {}
    couplings: dict[tuple[int, int], float] = {}
    ising_offset = offset

    for (left, right), coeff in qubo.items():
        if left == right:
            h[left] = h.get(left, 0.0) + coeff / 2.0
            ising_offset += coeff / 2.0
        else:
            key = (left, right) if left <= right else (right, left)
            h[left] = h.get(left, 0.0) + coeff / 4.0
            h[right] = h.get(right, 0.0) + coeff / 4.0
            couplings[key] = couplings.get(key, 0.0) + coeff / 4.0
            ising_offset += coeff / 4.0

    return IsingModel(
        h={key: value for key, value in h.items() if abs(value) > 1e-12},
        J={key: value for key, value in couplings.items() if abs(value) > 1e-12},
        offset=ising_offset,
        variable_names=variable_names,
        variable_metadata=variable_metadata,
    )

