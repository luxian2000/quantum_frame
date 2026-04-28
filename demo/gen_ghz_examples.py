"""
生成与验证 3-qubit GHZ 态的示例脚本：
- 构建标准 GHZ 制备电路并保存为 JSON
- 使用随机搜索通过 `BasicThreeQubitEnv` 寻找高保真度电路并保存

在 `demo` 目录下运行：
python gen_ghz_examples.py
"""

import sys
import numpy as np
from pathlib import Path

# 确保能导入 workspace 中的 nexq 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils_nexq import get_ghz_state
from qas_envs_nexq import BasicThreeQubitEnv
from nexq.circuit.model import Circuit, hadamard, cnot
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.circuit.io.json_io import save_circuit_json


def build_canonical_ghz_and_save(out_path: str = 'canonical_ghz.json'):
    target = get_ghz_state(3)
    n_qubits = 3

    # 构建标准 GHZ 电路： H(0); CNOT(0->1); CNOT(0->2)
    ghz_circuit = Circuit(hadamard(0), cnot(1, [0]), cnot(2, [0]), n_qubits=n_qubits)

    # 计算最终态与保真度
    backend = NumpyBackend()
    U = ghz_circuit.unitary(backend=backend)
    if hasattr(U, 'numpy'):
        U = U.numpy()
    else:
        U = np.asarray(U)

    init = np.zeros(2**n_qubits, dtype=complex)
    init[0] = 1.0
    final_state = U @ init

    inner = np.vdot(final_state, target)
    fidelity = float(np.abs(inner)**2)

    print(f"Canonical GHZ circuit fidelity vs target: {fidelity:.6f}")

    save_circuit_json(ghz_circuit, out_path)
    print(f"Saved canonical GHZ circuit to {out_path}")
    return ghz_circuit, fidelity


def random_search_env(env: BasicThreeQubitEnv, episodes: int = 200, out_path: str = 'best_random_ghz.json'):
    best = {'fidelity': -1.0, 'circuit': None}

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, len(env.action_gates))
            obs, reward, done, info = env.step(action)
        if info['fidelity'] > best['fidelity']:
            best['fidelity'] = info['fidelity']
            best['circuit'] = info['circuit']

    if best['circuit'] is not None:
        save_circuit_json(best['circuit'], out_path)
        print(f"Random search best fidelity={best['fidelity']:.6f}, saved to {out_path}")
    else:
        print("Random search found no circuit")
    return best


if __name__ == '__main__':
    print('\n== Generating canonical GHZ circuit ==')
    ghz_circ, ghz_fid = build_canonical_ghz_and_save('canonical_ghz.json')

    print('\n== Random search in BasicThreeQubitEnv ==')
    env = BasicThreeQubitEnv(max_timesteps=10, fidelity_threshold=0.99)
    best = random_search_env(env, episodes=500, out_path='best_random_ghz.json')

    print('\nDone.')
