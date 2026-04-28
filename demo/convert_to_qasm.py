"""
将 demo 目录中的 JSON 电路转换为 OpenQASM 文件（2.0）。

用法：
    python convert_to_qasm.py

会在同目录生成 canonical_ghz.qasm 和 best_random_ghz.qasm
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexq.circuit.io.json_io import load_circuit_json
from nexq.circuit.io.qasm import save_circuit_qasm


def convert(json_path: Path, qasm_path: Path, version: str = '2.0'):
    try:
        c = load_circuit_json(json_path)
        save_circuit_qasm(c, qasm_path, version=version)
        print(f"Converted {json_path.name} -> {qasm_path.name}")
    except Exception as e:
        print(f"Failed to convert {json_path}: {e}")


if __name__ == '__main__':
    base = Path(__file__).parent
    inputs = ['canonical_ghz.json', 'best_random_ghz.json']
    for name in inputs:
        jp = base / name
        qp = base / (Path(name).stem + '.qasm')
        if jp.exists():
            convert(jp, qp, version='2.0')
        else:
            print(f"Not found: {jp}")
