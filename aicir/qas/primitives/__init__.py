"""Reusable low-level QAS primitives.

These helpers are shared by traditional QAS, VQE-QAS closed loops, and
algorithm demos without owning a complete workflow.
"""

from .ansatz import (
    HEAMask,
    LayerwiseAnsatzGene,
    SupernetAnsatzGene,
    architecture_from_hea_mask,
    architecture_from_layerwise_gene,
    architecture_from_supernet_gene,
    enumerate_hea_masks,
    sample_layerwise_genes,
)
from .backend_utils import backend_runtime_metadata, resolve_qas_backend

__all__ = [
    "HEAMask",
    "LayerwiseAnsatzGene",
    "SupernetAnsatzGene",
    "architecture_from_hea_mask",
    "architecture_from_layerwise_gene",
    "architecture_from_supernet_gene",
    "backend_runtime_metadata",
    "enumerate_hea_masks",
    "resolve_qas_backend",
    "sample_layerwise_genes",
]
