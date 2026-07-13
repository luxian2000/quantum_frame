"""Derivative and derivative-free utilities for quantum machine learning workflows.

本包是原单文件 ``aicir/qml/deriv.py``（2660 行）的拆分版本，按功能拆成子模块：

- ``_coerce.py``：跨子模块共享的返回值/参数归一化 helper（``_as_scalar``、
  ``_as_state_vector``、参数索引/QNG block 归一化、torch 探测与设备张量转换）。
- ``fn_gradient.py``：标量目标函数梯度族 ``psr``/``psr4``/``spsr``/``spsa``/
  ``fd``/``auto``/``mpsr``（含各自 torch/NPU 设备驻留私有变体）。
- ``hessian.py``：``hessian``（psr 对角自检 + fd 降级/报错路径）。
- ``adjoint.py``：``ad``（伴随微分）。
- ``qfim.py``：``qfim``/``metric_tensor``/``qfim_diag``/``qfim_blocks``。
- ``qng.py``：量子自然梯度族 ``qng``/``bdqng``/``kqng``/``dqng``。
- ``rotosolve.py``：``rotosolve``。

本文件把上述子模块重新聚合为原来的扁平命名空间——``from aicir.qml.deriv import
xxx`` 对拆分前存在的每一个名称（含下划线开头的私有 helper，例如
``aicir.qml.deriv._as_scalar``）都继续成立，这是本次拆包的硬性不变量，见
``tests/qml/test_deriv_package_layout.py`` 的钉子测试。

返回值契约（array-in/array-out 规则，现状文档化——拆包不改变任何返回行为）：

- **一律返回 NumPy**：``auto``、``psr``、``psr4``、``spsr``、``spsa``、``mpsr``、
  ``fd``、``ad``、``hessian``、``qfim``、``metric_tensor``、``qfim_diag``、
  ``qfim_blocks``、**``qng``**、**``bdqng``**。即便某些方法（``auto``、
  ``qfim``/``qfim_diag`` 在 torch 态输入下）内部用 torch 张量计算以保持设备
  驻留，最终都会 ``.detach().cpu().numpy()`` 转换后再返回；``qng``/``bdqng``
  （稠密/分块 QFIM 版本）则完全没有 torch 设备驻留分支——即便 ``state_fn``
  返回 torch/NPU 张量，或预先算好的 ``grad``/``qfim``/``qfim_blocks`` 本身是
  torch 张量，内部也会经 ``_as_scalar``/``_as_state_vector``/
  ``np.asarray(..., dtype=float)`` 统一转回 host 端 NumPy。调用方对这些方法
  永远拿到 NumPy 数组。
- **入参决定出参设备（torch 入参/backend → 同设备 torch 出参）**：只有
  ``kqng``、``dqng`` 真正实现了 torch/NPU 设备驻留分支（当 backend 是
  NPU-family，或预先算好的 ``grad``/``qfim_diag``/``kfac_factors`` 是 torch
  张量时，返回 torch 张量；否则返回 NumPy 数组），以及 ``rotosolve``（当
  ``params`` 是 torch 张量，或 ``backend`` 是 Torch/NPU 系时，返回 torch
  张量；否则返回 NumPy 数组）。
"""

from __future__ import annotations

from ._coerce import (
    _as_scalar,
    _as_state_vector,
    _as_torch_state_vector,
    _contains_torch_tensor,
    _flat_to_index,
    _index_to_flat,
    _is_multi_index,
    _is_torch_tensor,
    _normalize_parameter_indices,
    _normalize_qng_blocks,
    _real_torch_dtype_from_backend,
    _state_tensor_and_backend,
    _torch_or_none,
)
from .fn_gradient import (
    _FOUR_TERM_C1,
    _FOUR_TERM_C2,
    _as_torch_scalar,
    _auto_torch_device,
    _fd_at_index,
    _fd_torch,
    _psr_torch,
    _shifted_difference,
    _spsa_perturbation_matrix,
    _spsa_torch,
    _to_real_torch_scalar,
    _torch_shifted_difference,
    auto,
    fd,
    mpsr,
    psr,
    psr4,
    spsa,
    spsr,
)
from .hessian import _fd_second_at_indices, _psr_second_at_index, hessian
from .adjoint import (
    _AD_DIFFERENTIABLE,
    _AD_PAULI_GENERATOR,
    _CDTYPE,
    _SWAP_LOCAL,
    _ad_apply,
    _ad_gate_local_matrix_and_axes,
    _ad_generator_local_and_axes,
    _controlled_generator_local,
    _pauli_local,
    ad,
)
from .qfim import (
    _qfim_blocks_from_state_fd,
    _qfim_diag_from_state_fd,
    _qfim_diag_from_state_fd_torch,
    _qfim_from_derivatives,
    _qfim_from_derivatives_torch,
    _qfim_from_state_fd,
    _state_derivatives_fd,
    _state_derivatives_fd_torch,
    _torch_inner_product,
    metric_tensor,
    qfim,
    qfim_blocks,
    qfim_diag,
)
from .qng import (
    _is_npu_family_backend,
    _kfac_factors_from_qfim_block,
    _kfac_factors_from_qfim_block_torch,
    _normalize_kfac_blocks,
    _normalize_kfac_factor_shapes,
    _ordinary_gradient_for_dqng_torch,
    _ordinary_gradient_for_qng,
    _qfim_kfac_factors_from_state_fd,
    _qfim_kfac_factors_from_state_fd_torch,
    _solve_damped_system,
    _solve_kfac_factor_block,
    _solve_kfac_factor_block_torch,
    _solve_linear_matrix,
    _torch_real_tensor,
    _validate_kfac_factors,
    _validate_kfac_factors_torch,
    _validate_qfim,
    _validate_qfim_blocks,
    _validate_qfim_diag,
    bdqng,
    dqng,
    kqng,
    qng,
)
from .rotosolve import (
    _add_at_torch,
    _as_numpy_scalar,
    _as_torch_scalar_rotosolve,
    _is_torch_family_backend,
    _rotosolve_delta_numpy,
    _rotosolve_delta_torch,
    _rotosolve_numpy,
    _rotosolve_torch,
    _validate_shift,
    rotosolve,
)

__all__ = ["auto", "psr", "psr4", "spsr", "spsa", "mpsr", "fd", "ad", "qng", "bdqng", "kqng", "dqng", "rotosolve"]
