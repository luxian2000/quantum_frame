"""nexq.algorithms.qas

Quantum architecture search and state-synthesis utilities.
"""

from .expressibility import KL_Haar_relative, MMD_relative
from .CRLQAS import AdamSPSAConfig, CRLQASConfig, CRLQASResult, crlqas, train_crlqas
from .PPR_DQL import PPRDQLConfig, PPRDQLPolicy, PPRDQLResult, ppr_dql_state_to_circuit, train_ppr_dql

try:
    from .state_qas import StateQASConfig, state_to_circuit
except ImportError:
    StateQASConfig = None
    state_to_circuit = None

__all__ = [
    "KL_Haar_relative",
    "MMD_relative",
    "AdamSPSAConfig",
    "CRLQASConfig",
    "CRLQASResult",
    "crlqas",
    "PPRDQLConfig",
    "PPRDQLPolicy",
    "PPRDQLResult",
    "ppr_dql_state_to_circuit",
    "train_crlqas",
    "train_ppr_dql",
]

if StateQASConfig is not None and state_to_circuit is not None:
    __all__.extend(["state_to_circuit", "StateQASConfig"])
