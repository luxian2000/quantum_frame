"""Shared algorithm-level metric implementations."""

from .expressibility import KL_Haar_divergence, KL_Haar_relative, MMD_relative
from .hardware import (
    DEFAULT_NATIVE_GATES,
    HardwareProfile,
    native_depth_twoq_efficiency,
    native_depth_twoq_efficiency_details,
    topology_mapping_efficiency,
    topology_mapping_efficiency_details,
)
from .noisy_expressibility import (
    KL_Haar_noisy,
    MMD_noisy,
    comparative_expressibility,
    expressibility_score,
)
from .trainability import (
    gradient_norm_score,
    gradient_variance_score,
    local_probe_gradient_statistics,
    local_probe_objective,
    structure_proxy,
    structure_proxy_details,
)

__all__ = [
    "KL_Haar_divergence",
    "KL_Haar_relative",
    "KL_Haar_noisy",
    "MMD_relative",
    "MMD_noisy",
    "DEFAULT_NATIVE_GATES",
    "HardwareProfile",
    "comparative_expressibility",
    "expressibility_score",
    "gradient_norm_score",
    "gradient_variance_score",
    "local_probe_gradient_statistics",
    "local_probe_objective",
    "native_depth_twoq_efficiency",
    "native_depth_twoq_efficiency_details",
    "structure_proxy",
    "structure_proxy_details",
    "topology_mapping_efficiency",
    "topology_mapping_efficiency_details",
]
