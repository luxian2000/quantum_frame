# aicir Architecture Design: Evaluating Execution Schemes

This document evaluates the main architectural schemes for quantum circuit execution, assesses their suitability for the core functions of `aicir` (Quantum Machine Learning, Quantum Architecture Search, and Quantum Error Correction), and proposes a layered integration strategy.

## 1. Main Architectural Alternatives

### 1.1 The `QNode` Abstraction (PennyLane Style)
Wraps the quantum circuit, hardware device, measurement type, and gradient rule into a single callable function (e.g., `@qnode(device="numpy", diff_method="psr")`).
*   **Advantages:**
    *   Massively reduces boilerplate code for users.
    *   Seamless integration with classical ML frameworks (PyTorch/TensorFlow) as a neural network layer.
    *   Allows the framework to automatically inject the best differentiation method (`select_diff`).
*   **Disadvantages:**
    *   Tightly couples the circuit definition to the execution engine and gradient method, making it harder to compile once and run on multiple different backends.
    *   Can introduce Python execution overhead for highly dynamic circuits.

### 1.2 The "Primitives" Pattern (Qiskit Style)
Separates the passive data structure (`Circuit`) from the execution engine (`Estimator` or `Sampler`).
*   **Advantages:**
    *   Clear separation of concerns. A circuit is compiled/optimized once and can be passed to multiple backends.
    *   Standardized interface for retrieving expectation values or probability distributions.
*   **Disadvantages:**
    *   Slightly more verbose for simple ML tasks compared to a `QNode`.
    *   Less native "feel" when embedded inside a PyTorch computation graph.

### 1.3 Pure Tensor-Network / Auto-Diff Graph (JAX / Torch Style)
Treats quantum gates strictly as tensor multiplications. There is no boundary between classical and quantum code; everything is a single auto-differentiable graph.
*   **Advantages:**
    *   **Unmatched Simulation Speed:** No translation overhead. 
    *   **Exact Gradients:** Enables exact analytical gradients ($O(1)$ memory) via backpropagation without relying on parameter-shift rules.
*   **Disadvantages:**
    *   Highly rigid. Changing the topology (adding/removing gates) requires rebuilding the entire computational graph.
    *   Only works for classical simulation; cannot be deployed to real QPUs.

### 1.4 AOT Compilation / MLIR (Catalyst / QIR Style)
Just-In-Time (JIT) or Ahead-Of-Time (AOT) compiles the entire hybrid program into machine-level instructions (like LLVM or OpenQASM 3.0) before execution.
*   **Advantages:**
    *   Extremely fast hardware execution.
    *   Crucial for dynamic circuits: allows real-time classical control flow (if/else statements based on mid-circuit measurements) to execute in microseconds on hardware controllers.
*   **Disadvantages:**
    *   High engineering complexity (requires writing compiler passes).
    *   Compilation overhead can be slow if the circuit changes structure every iteration.

---

## 2. Most Suitable Schemes per Domain

The three core domains of `aicir` have fundamentally different computational requirements:

### 2.1 Quantum Machine Learning (QML)
*   **Best Scheme:** **Pure Tensor-Graph (Auto-Diff Graph)**
*   **Why:** QML models have fixed topologies but thousands of parameters that need constant updating. Treating gates as PyTorch/NPU tensors yields the fastest forward passes and allows for exact, highly efficient backpropagation (`aicir.qml.deriv.auto`).

### 2.2 Quantum Architecture Search (QAS)
*   **Best Scheme:** **Primitives (`Estimator`) + JIT Evaluation**
*   **Why:** QAS constantly alters the topology of the circuit (adding/removing blocks). Tensor-graphs struggle with dynamic structures. Primitives allow rapid generation of lightweight `Circuit` objects, which can be quickly evaluated by a fast state-vector engine using heuristic/non-differentiable rewards (e.g., hardware efficiency, expressibility).

### 2.3 Quantum Error Correction (QEC)
*   **Best Scheme:** **AOT Compilation (MLIR / OpenQASM 3.0)**
*   **Why:** QEC relies on mid-circuit measurements and real-time classical feed-forward (e.g., "if syndrome matches X, apply Pauli Z"). Python is too slow to handle this. The circuit must be compiled down to hardware-level instructions so the classical logic executes directly on the QPU control electronics.

---

## 3. Integration Strategy: The Layered Architecture

To harness the advantages of all these schemes without compromising performance, `aicir` should adopt an **Intermediate Representation (IR) Driven Architecture**:

### Layer 1: Flexible Frontend APIs (User Experience)
*   For **QML users**: Provide the PennyLane-style `@qnode` decorator for rapid prototyping and seamless PyTorch integration.
*   For **QAS researchers**: Expose raw `Circuit` objects and `Estimator` primitives for manual hacking and topology generation.
*   For **QEC researchers**: Provide explicit dynamic control-flow APIs (e.g., `circuit.c_if()`).

### Layer 2: A Unified Typed IR (The Core)
*   Regardless of how the circuit was constructed, the frontend immediately compiles it into a strictly typed **`CircuitIR`** (already underway in `aicir.ir`). 
*   This IR supports gates, classical registers, and dynamic control flow.

### Layer 3: The Smart Dispatcher (Execution Engine)
When execution is requested, the framework inspects the `CircuitIR` and the target backend to route the computation optimally:
1.  **For QML Simulation:** The dispatcher maps the IR to the `NPUBackend` / `GPUBackend`, behaving like a pure Tensor-Graph to utilize `auto` gradients.
2.  **For QAS Simulation:** The dispatcher routes the IR to a fast, JIT-compiled C++ or NumPy engine that excels at rapidly evaluating topological heuristics without building massive computation graphs.
3.  **For QEC & Real Hardware:** The dispatcher routes the IR through `aicir.transpile` to be compiled into AOT representations like **OpenQASM 3.0**, preserving mid-circuit measurements and shipping the binary to the QPU.
