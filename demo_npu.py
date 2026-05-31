"""Ascend NPU verification demo for nexq.

Single-process strict NPU check:
    python demo_npu.py

Allow local CPU fallback for development:
    python demo_npu.py --allow-cpu-fallback

Multi-NPU task-parallel check with torchrun:
    torchrun --nproc_per_node=2 demo_npu.py --batch-size 6

The distributed path is task parallel for batches/parameter scans. It does not
shard one state vector across devices.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np

from nexq import Circuit, Measure, NPUBackend, cnot, hadamard, ry, rz


def build_bell_phase_circuit(theta: float = 0.3, backend=None) -> Circuit:
	"""Build a 2-qubit entangling circuit that exercises local gate application."""
	return Circuit(
		hadamard(0),
		cnot(1, [0]),
		rz(theta, 1),
		n_qubits=2,
		backend=backend,
	)


def build_scan_circuit(theta: float, backend=None) -> Circuit:
	"""Build a simple parameterized circuit for batch/distributed checks."""
	return Circuit(
		hadamard(0),
		ry(theta, 1),
		cnot(1, [0]),
		rz(theta / 2.0, 1),
		n_qubits=2,
		backend=backend,
	)


def print_rank(message: str, rank: int) -> None:
	print(f"[rank {rank}] {message}", flush=True)


def format_probabilities(probs: Iterable[float], n_qubits: int) -> str:
	parts = []
	for idx, prob in enumerate(np.asarray(probs).reshape(-1)):
		if abs(float(prob)) > 1e-7:
			parts.append(f"|{idx:0{n_qubits}b}>={float(prob):.6f}")
	return ", ".join(parts) if parts else "<all zero>"


def run_single_circuit(measure: Measure, backend: NPUBackend, shots: int) -> None:
	rank = backend.distributed_rank
	circuit = build_bell_phase_circuit(backend=backend)
	result = measure.run(circuit, shots=shots, return_state=False)

	print_rank("single circuit result", rank)
	print_rank(f"  backend_name: {result.backend_name}", rank)
	print_rank(f"  probabilities: {format_probabilities(result.probabilities, result.n_qubits)}", rank)
	print_rank(f"  counts({shots}): {result.counts}", rank)
	print_rank(f"  metadata: {result.metadata}", rank)


def run_batch_circuits(measure: Measure, backend: NPUBackend, shots: int, batch_size: int) -> None:
	rank = backend.distributed_rank
	params = np.linspace(0.0, np.pi, batch_size)
	circuits = [build_scan_circuit(float(theta), backend=backend) for theta in params]
	options = [
		{
			"label": f"theta_{i}",
			"return_state": False,
		}
		for i in range(batch_size)
	]

	results = measure.run_batch(
		circuits,
		shots=shots,
		mode="state_vector",
		per_circuit_options=options,
	)

	print_rank("batch results gathered in input order", rank)
	for result in results:
		idx = int(result.metadata["batch_index"])
		owner = idx % max(1, backend.distributed_world_size)
		peak_state, peak_prob = result.most_probable()
		print_rank(
			f"  idx={idx} owner_rank={owner} label={result.metadata.get('label')} "
			f"peak={peak_state} prob={peak_prob:.6f} counts={result.counts}",
			rank,
		)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run nexq NPU backend verification on Ascend NPU.",
	)
	parser.add_argument("--shots", type=int, default=1024, help="Sampling shots per circuit.")
	parser.add_argument("--batch-size", type=int, default=4, help="Number of circuits for batch/distributed check.")
	parser.add_argument(
		"--allow-cpu-fallback",
		action="store_true",
		help="Allow CPU fallback when NPU is unavailable. Default is strict NPU mode.",
	)
	parser.add_argument(
		"--no-init-process-group",
		action="store_true",
		help="Do not initialize torch.distributed even when torchrun env variables exist.",
	)
	parser.add_argument(
		"--skip-batch",
		action="store_true",
		help="Only run the single-circuit check.",
	)
	args = parser.parse_args()

	if args.shots <= 0:
		raise ValueError("--shots must be positive")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be positive")

	backend = NPUBackend.from_distributed_env(
		fallback_to_cpu=args.allow_cpu_fallback,
		init_process_group=not args.no_init_process_group,
	)
	measure = Measure(backend)
	rank = backend.distributed_rank

	print_rank("=== nexq Ascend NPU Backend Verification ===", rank)
	print_rank(f"pid: {os.getpid()}", rank)
	print_rank(f"backend: {backend.name}", rank)
	print_rank(f"runtime_context: {backend.runtime_context}", rank)
	print_rank(f"distributed_initialized: {backend.distributed_initialized}", rank)
	print_rank(f"world_size: {backend.distributed_world_size}", rank)

	run_single_circuit(measure, backend, shots=args.shots)
	if not args.skip_batch:
		run_batch_circuits(measure, backend, shots=args.shots, batch_size=args.batch_size)


if __name__ == "__main__":
	main()
