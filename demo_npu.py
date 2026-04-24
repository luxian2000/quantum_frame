"""NPU backend verification demo for nexq.

Usage examples:
	python demo_npu.py
	python demo_npu.py --mode new
	python demo_npu.py --mode old
	python demo_npu.py --shots 2048 --allow-cpu-fallback

Default behavior is strict NPU mode (no CPU fallback), which is useful
for validating remote NPU platform availability.
"""

from __future__ import annotations

import argparse

from nexq import Circuit, Measure, NPUBackend, cnot, hadamard, rz


def build_demo_circuit(backend=None) -> Circuit:
	"""Build a small 2-qubit circuit with entanglement and phase rotation."""
	return Circuit(
		hadamard(0),
		cnot(1, [0]),
		rz(0.3, 1),
		n_qubits=2,
		backend=backend,
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run a nexq circuit on NPU backend for platform verification.",
	)
	parser.add_argument("--shots", type=int, default=1024, help="Sampling shots, default=1024")
	parser.add_argument(
		"--mode",
		choices=["new", "old"],
		default="new",
		help="Execution path: new=circuit-bound backend, old=measure-time backend",
	)
	parser.add_argument(
		"--allow-cpu-fallback",
		action="store_true",
		help="Allow CPU fallback when NPU is unavailable (default: strict NPU mode)",
	)
	args = parser.parse_args()

	# Strict by default to ensure remote NPU validation is meaningful.
	backend = NPUBackend.from_distributed_env(fallback_to_cpu=args.allow_cpu_fallback)
	measure = Measure(backend)

	if args.mode == "new":
		# New path: bind backend in Circuit so front-end matrix assembly and execution stay on one XPU.
		circuit = build_demo_circuit(backend=backend)
	else:
		# Old path: keep circuit backend-agnostic and pick backend at measure time.
		circuit = build_demo_circuit(backend=None)

	result = measure.run(circuit, shots=args.shots)

	print("=== nexq NPU Backend Verification ===")
	print(f"mode: {args.mode}")
	print(f"backend: {backend.name}")
	print(f"runtime_context: {backend.runtime_context}")
	print(f"circuit.backend: {getattr(circuit, 'backend', None)}")
	print(f"result.backend_name: {result.backend_name}")
	print(f"circuit: {circuit}")
	print(f"probabilities: {result.probabilities}")
	print(f"counts({args.shots} shots): {result.counts}")
	print(f"summary: {result.summary()}")


if __name__ == "__main__":
	main()