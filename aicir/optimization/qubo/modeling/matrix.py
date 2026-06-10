from __future__ import annotations

from dataclasses import dataclass

from .registry import VariableMetadata


@dataclass(frozen=True)
class SparseMatrixCOO:
    row: list[int]
    col: list[int]
    data: list[float]
    shape: tuple[int, int]
    offset: float = 0.0
    variable_names: list[str] | None = None
    variable_metadata: list[VariableMetadata] | None = None

    def to_dense(self) -> list[list[float]]:
        matrix = [[0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        for row, col, value in zip(self.row, self.col, self.data):
            matrix[row][col] += value
        return matrix

