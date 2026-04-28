"""
量子架构搜索环境 (基于nexq库)

通过强化学习自动设计量子电路以制备目标量子态。
与原始的Cirq版本兼容的接口，但使用nexq库实现。
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from io import StringIO
import sys
from contextlib import closing

# nexq 库导入
sys.path.insert(0, '/Users/luxian/GitSpace/quantum_frame')
from nexq.circuit.model import Circuit
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.circuit.gates import gate_to_matrix


class QuantumArchSearchEnv:
    """
    量子架构搜索环境。
    
    代理通过选择量子门来构造电路，目标是生成接近目标量子态的电路。
    每个时步，代理接收基于可观测量的状态，采取动作添加一个量子门。
    如果保真度超过阈值，获得正奖励；否则获得负奖励。
    
    属性:
        target: 目标量子态向量
        n_qubits: 量子比特数量
        action_gates: 可选的量子门列表
        state_observables: 用于计算环境状态的观测量列表
        fidelity_threshold: 保真度阈值
        reward_penalty: 奖励惩罚系数
        max_timesteps: 最大时步数
    """
    
    def __init__(
        self,
        target: np.ndarray,
        n_qubits: int,
        state_observables: List[Dict],
        action_gates: List[Dict],
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_observables: Optional[float] = None,
        error_gates: Optional[float] = None,
        backend = None,
    ):
        """
        初始化量子架构搜索环境。
        
        Args:
            target: 目标量子态向量
            n_qubits: 量子比特数量
            state_observables: 用于计算观测值的泡利算子列表
            action_gates: 代理可选择的量子门列表
            fidelity_threshold: 保真度阈值，超过则终止
            reward_penalty: 未达到保真度时的惩罚
            max_timesteps: 最大步数
            error_observables: 观测量的误差率（可选，用于噪声模拟）
            error_gates: 量子门的误差率（可选，用于噪声模拟）
            backend: 计算后端（默认为NumpyBackend）
        """
        # 验证目标态
        assert len(target) == 2**n_qubits, \
            f"目标态大小应为 2^{n_qubits}={2**n_qubits}，得到 {len(target)}"
        
        # 归一化目标态
        target_norm = np.linalg.norm(target)
        if target_norm > 0:
            target = target / target_norm
        
        # 保存参数
        self.target = target
        self.n_qubits = n_qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates
        
        # 后端
        self.backend = backend if backend is not None else NumpyBackend()
        
        # 计算目标密度矩阵
        self.target_density = np.outer(target, np.conj(target))
        
        # 环境状态
        self.circuit_gates = []
        self.current_timestep = 0
        
        # 观测空间：每个观测量的期望值
        self.observation_space_shape = (len(state_observables),)
    
    def __str__(self) -> str:
        """环境的字符串表示"""
        desc = 'QuantumArchSearchEnv('
        desc += f'Qubits={self.n_qubits}, '
        desc += f'Target={self.target}, '
        desc += f'Gates=[{", ".join(str(g) for g in self.action_gates[:3])}...], '
        desc += f'Observables=[{", ".join(str(o) for o in self.state_observables[:3])}...])'
        return desc
    
    def seed(self, seed=None):
        """设置随机种子（保持与gym兼容）"""
        np.random.seed(seed)
        return [seed]
    
    def reset(self) -> np.ndarray:
        """
        重置环境。
        
        Returns:
            初始观测值
        """
        self.circuit_gates = []
        self.current_timestep = 0
        return self._get_obs()
    
    def _build_circuit(self) -> Circuit:
        """构建当前的量子电路"""
        circuit = Circuit(*self.circuit_gates, n_qubits=self.n_qubits)
        return circuit
    
    def _add_noise_to_circuit(self, circuit: Circuit) -> Circuit:
        """
        向电路添加噪声（模拟）。
        
        在实际应用中，可以添加更复杂的噪声模型。
        当前实现为占位符。
        
        Args:
            circuit: 原始电路
            
        Returns:
            带噪声的电路（当前直接返回原电路）
        """
        # TODO: 实现噪声模型
        return circuit
    
    def _get_obs(self) -> np.ndarray:
        """
        获取当前观测值。
        
        观测值是所有观测量在当前量子态上的期望值。
        
        Returns:
            观测值向量
        """
        # 获取当前态
        state = self.get_circuit_state()
        
        # 计算观测量期望值
        observables = []
        for obs in self.state_observables:
            # 构建观测算子矩阵
            O = gate_to_matrix(obs, cir_qubits=self.n_qubits, backend=self.backend)
            if hasattr(O, 'numpy'):
                O = O.numpy()
            else:
                O = np.asarray(O)
            
            # 期望值 = ⟨ψ|O|ψ⟩
            expectation = np.real(np.conj(state) @ O @ state)
            observables.append(float(expectation))
        
        return np.array(observables, dtype=np.float32)
    
    def _get_fidelity(self) -> float:
        """
        计算当前电路生成的态与目标态之间的保真度。
        
        保真度 = |⟨target|state⟩|²
        
        Returns:
            保真度值（0到1之间）
        """
        # 获取当前态
        state = self.get_circuit_state()
        
        # 计算保真度
        inner_product = np.conj(state) @ self.target
        fidelity = np.abs(inner_product)**2
        
        return float(np.real(fidelity))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一个环境步骤。
        
        Args:
            action: 选择的动作索引（对应action_gates中的一个门）
            
        Returns:
            observation: 新的观测值向量
            reward: 获得的奖励
            done: 是否终止
            info: 附加信息字典
        """
        # 添加选定的门到电路
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)
        self.current_timestep += 1
        
        # 获取观测值
        observation = self._get_obs()
        
        # 计算保真度
        fidelity = self._get_fidelity()
        
        # 计算奖励
        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty
        
        # 判断是否终止
        done = (fidelity > self.fidelity_threshold) or \
               (self.current_timestep >= self.max_timesteps)
        
        # 返回信息
        info = {
            'fidelity': fidelity,
            'circuit': self._build_circuit(),
            'timestep': self.current_timestep,
        }
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        渲染环境（打印当前电路）。
        
        Args:
            mode: 渲染模式 ('human' 打印到stdout, 'ansi' 返回字符串)
            
        Returns:
            如果mode='ansi'，返回电路的字符串表示
        """
        circuit = self._build_circuit()
        circuit_str = f"\n量子电路 (n_qubits={self.n_qubits}, depth={len(self.circuit_gates)}):\n"
        
        for i, gate in enumerate(self.circuit_gates):
            circuit_str += f"  步骤{i}: {gate}\n"
        
        if mode == 'human':
            print(circuit_str)
            return None
        else:
            return circuit_str
    
    def get_circuit_unitary(self) -> np.ndarray:
        """
        获取当前电路的幺正矩阵表示。
        
        Returns:
            幺正矩阵
        """
        circuit = self._build_circuit()
        U = circuit.unitary(backend=self.backend)
        if hasattr(U, 'numpy'):
            U = U.numpy()
        else:
            U = np.asarray(U)
        return U
    
    def get_circuit_state(self) -> np.ndarray:
        """
        获取当前电路作用在|0...0⟩上的最终态。
        
        Returns:
            最终量子态向量
        """
        # 计算初始态（|0...0⟩）
        initial_state = np.zeros(2**self.n_qubits, dtype=complex)
        initial_state[0] = 1.0
        
        # 如果没有门，直接返回初始态
        if not self.circuit_gates:
            return initial_state
        
        # 逐个应用电路中的门
        state = initial_state
        for gate in self.circuit_gates:
            U = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)
            if hasattr(U, 'numpy'):
                U = U.numpy()
            else:
                U = np.asarray(U)
            state = U @ state
        
        return state
