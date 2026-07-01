"""Generate large-molecule chemistry presets with NPU-computed ground energy.

For molecules whose ``2^n × 2^n`` dense Hamiltonian is too large to diagonalize
(CH4 18q → 550 GB dense), this script computes the ground energy **matrix-free**:
the Hamiltonian is applied to a ``2^n`` state vector as a sum of Pauli-string
gather+phase operations (no dense matrix), and the lowest eigenvalue is found with
a Lanczos iteration (``scipy.sparse.linalg.eigsh``). The state-vector applies run
on the Ascend NPU (or CPU/CUDA fallback), which is what makes ≤ ~30-qubit
molecules tractable.

It then writes ``aicir/chemistry/molecules/<Formula>.py`` with the extracted Pauli
terms, provenance metadata, and the computed ground energy embedded in the docstring.

Run from the repository root:

    # NPU (fast; requires Ascend + torch_npu)
    PYTHONPATH=. python scripts/gen_molecule_preset.py --molecule ch4 --device npu:0

    # CPU dry run (small molecules, or to validate the pipeline)
    PYTHONPATH=. python scripts/gen_molecule_preset.py --molecule h2o --device cpu

    # all large presets (ch4/n2/beh2/nh3) sequentially on one NPU
    PYTHONPATH=. python scripts/gen_molecule_preset.py --all-large --device npu:0

    # same, one molecule per NPU across 4 devices (plain subprocesses, staggered
    # launch to avoid concurrent Ascend/TBE driver init races — no torchrun)
    PYTHONPATH=. python scripts/gen_molecule_preset.py --all-large --num-devices 4

Requires ``torch`` and ``scipy``. Qubit ordering is big-endian (qubit ``q`` → bit
``n-1-q``), matching ``aicir`` gate/state conventions.

Multi-device note: this deliberately does not use ``torchrun``. All-devices-at-once
launchers start every process at nearly the same instant, which can crash the
Ascend NPU/TBE op-compiler on concurrent first-touch initialization (seen in
practice as ``AclSetCompileopt`` / ``No module named 'tbe'`` errors). ``--num-devices``
instead spawns plain, staggered subprocesses per molecule (same convention as
``aicir/qas/vqe_loop/shard_scheduler.py``): independent task parallelism, no
``torch.distributed``/HCCL involved.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# demo module / build fn / formula-case filename / metadata attribute names.
SPECS = {
    "ch4": dict(mod="demos.CH4.CH4", build="build_ch4_hamiltonian", file="CH4",
                formula="CH4", cname="CH4_STO3G_JW_18Q",
                geo="CH4_GEOMETRY_ANGSTROM", basis="CH4_BASIS", mapper="CH4_QUBIT_MAPPER",
                ael="CH4_ACTIVE_ELECTRONS", aorb="CH4_ACTIVE_SPATIAL_ORBITALS"),
    "n2": dict(mod="demos.N2.N2", build="build_n2_hamiltonian", file="N2",
               formula="N2", cname="N2_STO3G_JW_14Q",
               geo="N2_GEOMETRY_ANGSTROM", basis="N2_BASIS", mapper="N2_QUBIT_MAPPER",
               ael="N2_ACTIVE_ELECTRONS", aorb="N2_ACTIVE_SPATIAL_ORBITALS"),
    "beh2": dict(mod="demos.BeH2.BeH2", build="build_beh2_hamiltonian", file="BeH2",
                 formula="BeH2", cname="BEH2_321G_JW_16Q",
                 geo="BEH2_GEOMETRY_ANGSTROM", basis="BEH2_BASIS", mapper="BEH2_QUBIT_MAPPER",
                 ael="BEH2_ACTIVE_ELECTRONS", aorb="BEH2_ACTIVE_SPATIAL_ORBITALS"),
    "nh3": dict(mod="demos.NH3.NH3", build="build_nh3_hamiltonian", file="NH3",
                formula="NH3", cname="NH3_STO3G_JW_12Q",
                geo="NH3_GEOMETRY_ANGSTROM", basis="NH3_BASIS", mapper="NH3_QUBIT_MAPPER",
                ael="NH3_ACTIVE_ELECTRONS", aorb="NH3_ACTIVE_SPATIAL_ORBITALS"),
    "h2o": dict(mod="demos.H2O.H2O", build="build_h2o_hamiltonian", file="H2O",
                formula="H2O", cname="H2O_STO3G_JW_6Q",
                geo="H2O_GEOMETRY_ANGSTROM", basis="H2O_BASIS", mapper="H2O_QUBIT_MAPPER",
                ael="H2O_ACTIVE_ELECTRONS", aorb="H2O_ACTIVE_SPATIAL_ORBITALS"),
}

MAX_QUBITS = 32

# Molecules whose dense matrix is infeasible — the matrix-free targets for --all-large.
LARGE = ("nh3", "n2", "beh2", "ch4")


def _launch_parallel(names, num_devices: int, args, stagger_seconds: float) -> None:
    """Spawn one plain subprocess per molecule, round-robin across NPUs.

    Deliberately not torchrun: launching all devices at once can crash the Ascend
    NPU/TBE op-compiler on concurrent first-touch initialization. Staggering the
    launches and using independent single-device processes (no torch.distributed)
    avoids that, matching ``aicir/qas/vqe_loop/shard_scheduler.py``.
    """
    procs = []
    for i, name in enumerate(names):
        device = f"npu:{i % num_devices}"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--molecule", name, "--device", device, "--out-dir", args.out_dir,
        ]
        if args.allow_cpu_fallback:
            command.append("--allow-cpu-fallback")
        if args.no_energy:
            command.append("--no-energy")
        print(f"launching {name} on {device}: {' '.join(command)}")
        procs.append((name, subprocess.Popen(command)))
        if i + 1 < len(names):
            time.sleep(stagger_seconds)

    failed = [name for name, proc in procs if proc.wait() != 0]
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def _resolve_backend(device: str, allow_cpu_fallback: bool):
    if str(device).lower().startswith("npu"):
        from aicir.backends.npu_backend import NPUBackend

        backend = NPUBackend(device=device)
        is_npu = type(backend).__name__ == "NPUBackend" and backend._device.type == "npu"
        if not is_npu and not allow_cpu_fallback:
            raise RuntimeError(
                "NPU not available; pass --allow-cpu-fallback to run on CPU instead."
            )
        return backend
    from aicir.backends.gpu_backend import GPUBackend

    return GPUBackend(device=device)


def _extract(spec) -> tuple[list[tuple[float, str]], int, dict]:
    module = importlib.import_module(spec["mod"])
    hamiltonian = getattr(module, spec["build"])()
    terms = []
    for ps in hamiltonian.terms:
        coeff = complex(ps.coefficient)
        assert abs(coeff.imag) < 1e-12, f"complex coefficient {coeff}"
        terms.append((float(coeff.real), "".join(ps.qubit_labels)))
    geometry = "; ".join(
        f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in getattr(module, spec["geo"])
    )
    meta = dict(
        basis=getattr(module, spec["basis"]),
        mapper=getattr(module, spec["mapper"]),
        ael=getattr(module, spec["ael"]),
        aorb=getattr(module, spec["aorb"]),
        geometry=geometry,
    )
    return terms, int(hamiltonian.n_qubits), meta


def ground_energy(terms, n_qubits: int, backend) -> float:
    """Matrix-free lowest eigenvalue: apply H as Pauli gather+phase, Lanczos via eigsh."""
    import torch
    from scipy.sparse.linalg import LinearOperator, eigsh

    dim = 1 << n_qubits
    device = backend.cast(np.zeros(1, dtype=np.complex64)).device
    idx = torch.arange(dim, device=device, dtype=torch.int64)

    # Precompute per-term (scalar coeff·i^{#Y}, X/Y flip mask, Y|Z phase mask).
    meta = []
    for coeff, pauli in terms:
        xmask = phasemask = n_y = 0
        for q, ch in enumerate(pauli):
            bit = 1 << (n_qubits - 1 - q)  # big-endian: qubit 0 is MSB
            if ch == "X":
                xmask |= bit
            elif ch == "Y":
                xmask |= bit
                phasemask |= bit
                n_y += 1
            elif ch == "Z":
                phasemask |= bit
        meta.append((complex(coeff) * (1j ** (n_y % 4)), xmask, phasemask))

    def matvec(vec: np.ndarray) -> np.ndarray:
        psi = backend.cast(np.ascontiguousarray(vec).astype(np.complex64))
        out = torch.zeros_like(psi)
        for scalar, xmask, phasemask in meta:
            src = idx ^ xmask
            parity = src & phasemask
            parity = parity ^ (parity >> 16)
            parity = parity ^ (parity >> 8)
            parity = parity ^ (parity >> 4)
            parity = parity ^ (parity >> 2)
            parity = parity ^ (parity >> 1)
            sign = (1 - 2 * (parity & 1)).to(psi.dtype)
            out = out + (sign * psi.index_select(0, src)) * scalar
        return np.asarray(backend.to_numpy(out)).astype(np.complex128)

    operator = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
    values = eigsh(operator, k=1, which="SA", return_eigenvectors=False)
    return float(np.real(values[0]))


def _emit(spec, terms, n_qubits, meta, energy, out_dir: Path) -> Path:
    lines = [
        f'"""{spec["formula"]} active-space qubit Hamiltonian preset.',
        "",
        f'PySCF/Qiskit Nature: {meta["basis"]} basis, {meta["ael"]}-electron/'
        f'{meta["aorb"]}-orbital active space, {meta["mapper"]}.',
        f"Ground energy {energy:.10f} Ha computed matrix-free (Lanczos, {n_qubits} qubits)",
        "by scripts/gen_molecule_preset.py; too large for dense diagonalization.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from ._base import MoleculeHamiltonian, register_molecule",
        "",
        "",
        f"{spec['cname']} = register_molecule(",
        "    MoleculeHamiltonian(",
        f'        name="{spec["formula"].lower()}",',
        f'        formula="{spec["formula"]}",',
        f"        n_qubits={n_qubits},",
        f'        basis="{meta["basis"]}",',
        f'        mapping="{meta["mapper"]}",',
        f'        geometry="{meta["geometry"]}",',
        f'        source="PySCF/Qiskit Nature (ActiveSpaceTransformer + {meta["mapper"]})",',
        f'        description="{spec["formula"]} {meta["ael"]}e/{meta["aorb"]}o active-space '
        f'Hamiltonian ({n_qubits} qubits); ground energy {energy:.6f} Ha.",',
        "        terms=(",
    ]
    for coeff, pauli in terms:
        lines.append(f'            ({coeff!r}, "{pauli}"),')
    lines += ["        ),", "    )", ")", ""]
    path = out_dir / f"{spec['file']}.py"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _generate_one(name: str, backend, out_dir: Path, no_energy: bool) -> None:
    spec = SPECS[name]
    terms, n_qubits, meta = _extract(spec)
    if n_qubits >= MAX_QUBITS:
        raise SystemExit(
            f"{spec['formula']} has {n_qubits} qubits; a 2^{n_qubits} state vector "
            f"exceeds the {MAX_QUBITS}-qubit limit for matrix-free computation."
        )
    print(f"{spec['formula']}: {n_qubits} qubits, {len(terms)} Pauli terms")

    if no_energy:
        energy = float("nan")
    else:
        print(f"computing ground energy matrix-free on {backend.cast(np.zeros(1)).device} ...")
        energy = ground_energy(terms, n_qubits, backend)
        print(f"ground energy = {energy:.10f} Ha")

    path = _emit(spec, terms, n_qubits, meta, energy, out_dir)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--molecule", choices=sorted(SPECS), help="molecule preset to generate")
    group.add_argument("--all-large", action="store_true",
                       help=f"generate all large presets ({', '.join(LARGE)}) in one run")
    parser.add_argument("--device", default="npu:0", help="torch device (npu:0 / cpu / cuda)")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="allow CPU when the requested NPU is unavailable")
    parser.add_argument("--no-energy", action="store_true",
                        help="skip the (matrix-free) ground-energy computation")
    parser.add_argument("--out-dir", default="aicir/chemistry/molecules",
                        help="output directory for the generated preset")
    parser.add_argument("--num-devices", type=int, default=1,
                        help="with --all-large: spawn one subprocess per molecule, "
                             "round-robin across this many npu:N devices (no torchrun)")
    parser.add_argument("--stagger-seconds", type=float, default=5.0,
                        help="delay between subprocess launches in --num-devices mode, "
                             "to avoid concurrent Ascend/TBE driver init races")
    args = parser.parse_args()

    if args.all_large and args.num_devices > 1:
        _launch_parallel(list(LARGE), args.num_devices, args, args.stagger_seconds)
        return

    names = list(LARGE) if args.all_large else [args.molecule]
    backend = None if args.no_energy else _resolve_backend(args.device, args.allow_cpu_fallback)
    out_dir = Path(args.out_dir)
    for name in names:
        _generate_one(name, backend, out_dir, args.no_energy)


if __name__ == "__main__":
    main()
