#!/usr/bin/env python3
"""
快速测试脚本 - 验证 nexq 量子架构搜索环境

测试所有核心功能是否正确工作。
"""

import sys
import numpy as np

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试 1: 导入模块")
    print("=" * 60)
    
    try:
        from utils_nexq import get_default_gates, get_bell_state, get_ghz_state
        print("✓ 工具函数导入成功")
    except Exception as e:
        print(f"✗ 工具函数导入失败: {e}")
        return False
    
    try:
        from qas_env_nexq import QuantumArchSearchEnv
        print("✓ 环境类导入成功")
    except Exception as e:
        print(f"✗ 环境类导入失败: {e}")
        return False
    
    try:
        from qas_envs_nexq import BasicTwoQubitEnv, BasicThreeQubitEnv
        print("✓ 预定义环境导入成功")
    except Exception as e:
        print(f"✗ 预定义环境导入失败: {e}")
        return False
    
    return True


def test_utils():
    """测试工具函数"""
    print("\n" + "=" * 60)
    print("测试 2: 工具函数")
    print("=" * 60)
    
    from utils_nexq import (
        get_default_gates,
        get_default_observables,
        get_bell_state,
        get_ghz_state,
        get_w_state,
    )
    
    # 测试量子门
    gates = get_default_gates(2)
    print(f"✓ 获得 2量子比特默认门: {len(gates)} 个门")
    
    # 测试观测量
    observables = get_default_observables(2)
    print(f"✓ 获得 2量子比特默认观测量: {len(observables)} 个")
    
    # 测试目标态
    bell = get_bell_state()
    print(f"✓ Bell态: 形状={bell.shape}, 范数={np.linalg.norm(bell):.6f}")
    
    ghz = get_ghz_state(3)
    print(f"✓ GHZ态 (3量子比特): 形状={ghz.shape}, 范数={np.linalg.norm(ghz):.6f}")
    
    w = get_w_state(3)
    print(f"✓ W态 (3量子比特): 形状={w.shape}, 范数={np.linalg.norm(w):.6f}")
    
    return True


def test_basic_env():
    """测试基础环境"""
    print("\n" + "=" * 60)
    print("测试 3: 基础 2量子比特环境")
    print("=" * 60)
    
    from qas_envs_nexq import BasicTwoQubitEnv
    
    try:
        env = BasicTwoQubitEnv(max_timesteps=10)
        print(f"✓ 环境创建成功: {env}")
        
        # 重置
        obs = env.reset()
        print(f"✓ 重置成功，观测形状: {obs.shape}")
        
        # 执行一步
        action = 0
        obs, reward, done, info = env.step(action)
        print(f"✓ 执行动作成功:")
        print(f"  - 奖励: {reward:.6f}")
        print(f"  - 保真度: {info['fidelity']:.6f}")
        print(f"  - 完成: {done}")
        
        # 执行多步
        for i in range(5):
            action = i % len(env.action_gates)
            obs, reward, done, info = env.step(action)
            if done:
                print(f"✓ 在步骤 {i+2} 完成，最终保真度: {info['fidelity']:.6f}")
                break
        
        return True
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_three_qubit_env():
    """测试 3量子比特环境"""
    print("\n" + "=" * 60)
    print("测试 4: 3量子比特环境")
    print("=" * 60)
    
    from qas_envs_nexq import BasicThreeQubitEnv
    
    try:
        env = BasicThreeQubitEnv(max_timesteps=15)
        print(f"✓ 环境创建成功")
        
        obs = env.reset()
        print(f"✓ 重置成功，观测形状: {obs.shape}")
        
        # 执行几步
        total_reward = 0
        for i in range(5):
            action = np.random.randint(0, len(env.action_gates))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                print(f"✓ 在步骤 {i+1} 完成")
                print(f"  - 总奖励: {total_reward:.6f}")
                print(f"  - 最终保真度: {info['fidelity']:.6f}")
                break
        
        return True
    except Exception as e:
        print(f"✗ 3量子比特环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_noisy_env():
    """测试有噪声的环境"""
    print("\n" + "=" * 60)
    print("测试 5: 有噪声的环境")
    print("=" * 60)
    
    from qas_envs_nexq import NoisyTwoQubitEnv
    
    try:
        env = NoisyTwoQubitEnv(error_rate=0.001, max_timesteps=10)
        print(f"✓ 有噪声环境创建成功 (错误率: 0.1%)")
        
        obs = env.reset()
        print(f"✓ 重置成功")
        
        # 执行动作
        obs, reward, done, info = env.step(0)
        print(f"✓ 执行动作成功，保真度: {info['fidelity']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 有噪声环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_circuit_properties():
    """测试电路属性"""
    print("\n" + "=" * 60)
    print("测试 6: 电路属性")
    print("=" * 60)
    
    from qas_envs_nexq import BasicTwoQubitEnv
    
    try:
        env = BasicTwoQubitEnv(max_timesteps=5)
        env.reset()
        
        # 添加一些门
        for i in range(3):
            env.step(i % len(env.action_gates))
        
        # 获取幺正矩阵
        U = env.get_circuit_unitary()
        print(f"✓ 获得电路幺正矩阵: 形状={U.shape}")
        
        # 验证幺正性: U†U = I
        I = U.conj().T @ U
        identity_error = np.linalg.norm(I - np.eye(4))
        print(f"✓ 幺正性检验: ||U†U - I|| = {identity_error:.2e}")
        
        # 获取最终态
        state = env.get_circuit_state()
        print(f"✓ 获得电路最终态: 形状={state.shape}, 范数={np.linalg.norm(state):.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 电路属性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_target():
    """测试自定义目标态"""
    print("\n" + "=" * 60)
    print("测试 7: 自定义目标态")
    print("=" * 60)
    
    from qas_envs_nexq import BasicNQubitEnv
    
    try:
        # 创建自定义目标态（3量子比特）
        custom_target = np.zeros(8, dtype=complex)
        custom_target[0] = 1.0
        custom_target = custom_target / np.linalg.norm(custom_target)
        
        env = BasicNQubitEnv(target=custom_target, max_timesteps=10)
        print(f"✓ 自定义环境创建成功")
        
        obs = env.reset()
        print(f"✓ 重置成功")
        
        action = 0
        obs, reward, done, info = env.step(action)
        print(f"✓ 执行动作成功，初始保真度: {info['fidelity']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 自定义目标态测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "量子架构搜索环境 - 快速测试".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    tests = [
        ("导入模块", test_imports),
        ("工具函数", test_utils),
        ("基础环境", test_basic_env),
        ("3量子比特环境", test_three_qubit_env),
        ("有噪声环境", test_noisy_env),
        ("电路属性", test_circuit_properties),
        ("自定义目标态", test_custom_target),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总体: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
