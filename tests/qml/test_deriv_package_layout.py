"""固定 ``aicir.qml.deriv`` 的可导入名称集合（Phase 4 拆包前后不变）。

拆包前该模块是单个 2660 行文件；拆包后是一个包（``__init__.py`` 聚合各子模块）。
本测试把拆包前 ``dir(aicir.qml.deriv)`` 采集到的“真实定义”名称集合硬编码下来，
保证拆包不丢失任何一个既有导入路径（包括测试代码之外没人使用、但曾经可以
``from aicir.qml.deriv import xxx`` 的私有 helper）。

刻意排除的“偶然泄漏”名称（旧模块顶层 ``import``/推导式 walrus 产生的、并非该
模块自身定义的 API）：``np``、``math``、``itertools``、``numbers``、``Any``、
``Callable``、``annotations``、``gen``（``_AD_PAULI_GENERATOR`` 字典推导式里
walrus 赋值泄漏的循环变量）、``gate_generator``、``parametric_pauli_gates``、
``circuit_instructions``、``instruction_controls``、``instruction_name``、
``instruction_parameter``、``instruction_qubits``（均是从 ``aicir.gates``/
``aicir.ir`` re-export 的外部符号，不是 deriv 自身的 API 面）。
"""

from __future__ import annotations

import aicir.qml.deriv as deriv

# 拆包前 `dir(aicir.qml.deriv)` 采集（HEAD bd5c642，2660 行单文件版本）。
PRE_SPLIT_PUBLIC_NAMES = {
    "_AD_DIFFERENTIABLE", "_AD_PAULI_GENERATOR", "_CDTYPE", "_FOUR_TERM_C1",
    "_FOUR_TERM_C2", "_SWAP_LOCAL", "_ad_apply", "_ad_gate_local_matrix_and_axes",
    "_ad_generator_local_and_axes", "_add_at_torch", "_as_numpy_scalar", "_as_scalar",
    "_as_state_vector", "_as_torch_scalar", "_as_torch_scalar_rotosolve",
    "_as_torch_state_vector", "_auto_torch_device", "_contains_torch_tensor",
    "_controlled_generator_local", "_fd_at_index", "_fd_second_at_indices", "_fd_torch",
    "_flat_to_index", "_index_to_flat", "_is_multi_index", "_is_npu_family_backend",
    "_is_torch_family_backend", "_is_torch_tensor", "_kfac_factors_from_qfim_block",
    "_kfac_factors_from_qfim_block_torch", "_normalize_kfac_blocks",
    "_normalize_kfac_factor_shapes", "_normalize_parameter_indices",
    "_normalize_qng_blocks", "_ordinary_gradient_for_dqng_torch",
    "_ordinary_gradient_for_qng", "_pauli_local", "_psr_second_at_index", "_psr_torch",
    "_qfim_blocks_from_state_fd", "_qfim_diag_from_state_fd",
    "_qfim_diag_from_state_fd_torch", "_qfim_from_derivatives",
    "_qfim_from_derivatives_torch", "_qfim_from_state_fd",
    "_qfim_kfac_factors_from_state_fd", "_qfim_kfac_factors_from_state_fd_torch",
    "_real_torch_dtype_from_backend", "_rotosolve_delta_numpy", "_rotosolve_delta_torch",
    "_rotosolve_numpy", "_rotosolve_torch", "_shifted_difference", "_solve_damped_system",
    "_solve_kfac_factor_block", "_solve_kfac_factor_block_torch", "_solve_linear_matrix",
    "_spsa_perturbation_matrix", "_spsa_torch", "_state_derivatives_fd",
    "_state_derivatives_fd_torch", "_state_tensor_and_backend", "_to_real_torch_scalar",
    "_torch_inner_product", "_torch_or_none", "_torch_real_tensor",
    "_torch_shifted_difference", "_validate_kfac_factors", "_validate_kfac_factors_torch",
    "_validate_qfim", "_validate_qfim_blocks", "_validate_qfim_diag", "_validate_shift",
    "ad", "auto", "bdqng", "dqng", "fd", "hessian", "kqng", "metric_tensor", "mpsr",
    "psr", "psr4", "qfim", "qfim_blocks", "qfim_diag", "qng", "rotosolve", "spsa", "spsr",
}


def test_pre_split_names_all_importable_post_split():
    """拆包前的每一个真实定义名称，拆包后都能从 ``aicir.qml.deriv`` 顶层拿到。"""
    missing = sorted(name for name in PRE_SPLIT_PUBLIC_NAMES if not hasattr(deriv, name))
    assert not missing, f"names dropped by the deriv package split: {missing}"


def test_pre_split_names_importable_via_from_import():
    """``from aicir.qml.deriv import <name>`` 对每个名称都必须成立。"""
    import importlib

    module = importlib.import_module("aicir.qml.deriv")
    for name in sorted(PRE_SPLIT_PUBLIC_NAMES):
        assert getattr(module, name, _MISSING) is not _MISSING, name


_MISSING = object()


def test_psr_is_same_object_via_deriv_and_qml_top_level():
    """``aicir.qml.deriv.psr`` 与 ``aicir.qml.psr`` 必须是同一个对象。"""
    import aicir.qml as qml

    assert deriv.psr is qml.psr


def test_deriv_is_now_a_package():
    """拆包后 ``aicir.qml.deriv`` 应为包（有 ``__path__``），不再是单文件模块。"""
    assert hasattr(deriv, "__path__")
