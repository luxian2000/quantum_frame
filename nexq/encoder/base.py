"""Base encoder class for quantum data encoding."""


class BaseEncoder:
    """Base class for all quantum encoders.
    
    This class defines the interface that all encoder implementations must follow.
    """
    
    def __init__(self, n_qubits: int):
        """Initialize the encoder.
        
        Args:
            n_qubits: Number of qubits to use for encoding.
        """
        self.n_qubits = n_qubits
    
    def encode(self, data):
        """Encode classical data into quantum states.
        
        Args:
            data: Classical data to encode.
            
        Returns:
            Encoded quantum representation.
        """
        raise NotImplementedError("Subclasses must implement encode method")
    
    def decode(self, quantum_state):
        """Decode quantum states back to classical data.
        
        Args:
            quantum_state: Quantum state to decode.
            
        Returns:
            Decoded classical data.
        """
        raise NotImplementedError("Subclasses must implement decode method")
