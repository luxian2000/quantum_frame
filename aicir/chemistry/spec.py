"""Hamiltonian input specifications for chemistry and VQE/QAS workflows.

This module keeps electronic-structure generation as an optional front-end:
PySCF/Qiskit Nature are imported only when a molecular specification is
resolved.  The output is plain weighted Pauli terms, so downstream VQE/QAS
execution can still use aicir backends such as numpy, torch, or NPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


PauliTerm = tuple[float, str]
GeometryAtom = tuple[str, tuple[float, float, float]]


@dataclass(frozen=True)
class PauliTermsSpec:
    """A Hamiltonian supplied directly as weighted Pauli terms."""

    terms: Sequence[PauliTerm]
    source: str = "manual"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MolecularSpec:
    """A molecular electronic-structure request."""

    geometry: str | Sequence[GeometryAtom]
    basis: str = "sto3g"
    charge: int = 0
    spin: int = 0
    unit: str = "angstrom"
    driver: str = "pyscf"
    mapping: str = "jordan_wigner"
    active_space: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedHamiltonian:
    """Plain Pauli Hamiltonian plus stable provenance metadata."""

    terms: tuple[PauliTerm, ...]
    n_qubits: int
    hamiltonian_class: str
    hamiltonian_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[: int(length)]


def _real_float(value: Any, *, field_name: str = "coefficient") -> float:
    number = complex(value)
    if abs(number.imag) > 1.0e-10:
        raise ValueError(f"{field_name} must be real-valued, got {value!r}")
    return float(number.real)


def _normalize_terms(raw_terms: Iterable[Any]) -> tuple[PauliTerm, ...]:
    terms: list[PauliTerm] = []
    width: int | None = None
    for raw in raw_terms:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"Invalid Pauli term {raw!r}; expected [coefficient, pauli]")
        coefficient, pauli = raw
        pauli_text = str(pauli).strip().upper()
        if not pauli_text or any(label not in {"I", "X", "Y", "Z"} for label in pauli_text):
            raise ValueError(f"Invalid Pauli string {pauli!r}")
        if width is None:
            width = len(pauli_text)
        elif len(pauli_text) != width:
            raise ValueError(f"Pauli strings must have one width, found {width} and {len(pauli_text)}")
        terms.append((_real_float(coefficient), pauli_text))
    if not terms:
        raise ValueError("Hamiltonian must contain at least one Pauli term")
    return tuple(terms)


def _geometry_to_qiskit_atom(geometry: str | Sequence[GeometryAtom]) -> str:
    if isinstance(geometry, str):
        text = geometry.strip()
        if not text:
            raise ValueError("molecular geometry cannot be empty")
        return text
    atoms: list[str] = []
    for atom in geometry:
        symbol, coords = atom
        if len(coords) != 3:
            raise ValueError(f"Atom coordinates must be length 3, got {atom!r}")
        x, y, z = (float(value) for value in coords)
        atoms.append(f"{str(symbol).strip()} {x:.16g} {y:.16g} {z:.16g}")
    if not atoms:
        raise ValueError("molecular geometry cannot be empty")
    return "; ".join(atoms)


def _geometry_to_jsonable(geometry: str | Sequence[GeometryAtom]) -> Any:
    if isinstance(geometry, str):
        return geometry.strip()
    return [[str(symbol), [float(x), float(y), float(z)]] for symbol, (x, y, z) in geometry]


def _normalize_mapping(mapping: str) -> str:
    key = str(mapping).strip().lower().replace("-", "_")
    aliases = {
        "jw": "jordan_wigner",
        "jordanwigner": "jordan_wigner",
        "jordan_wigner": "jordan_wigner",
        "parity": "parity",
        "bravyi_kitaev": "bravyi_kitaev",
        "bk": "bravyi_kitaev",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported qubit mapper {mapping!r}")
    return aliases[key]


def _symbols_from_formula(formula: str) -> list[str]:
    symbols: list[str] = []
    for symbol, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula).strip()):
        count = int(count_text) if count_text else 1
        symbols.extend([symbol] * count)
    if not symbols:
        raise ValueError(f"Cannot infer molecular geometry from molecule {formula!r}")
    return symbols


def _default_geometry_from_molecule(raw: Mapping[str, Any]) -> Any:
    if "geometry" in raw or "atom" in raw:
        return raw.get("geometry", raw.get("atom", ""))
    molecule = str(raw.get("molecule", raw.get("name", ""))).strip()
    if not molecule:
        return ""
    distance = raw.get("distance", raw.get("bond_length", raw.get("r", None)))
    if distance is None:
        raise ValueError("molecular shorthand requires geometry or distance")
    symbols = _symbols_from_formula(molecule)
    if len(symbols) != 2:
        raise ValueError("molecular shorthand with distance currently supports diatomic molecules only")
    return [[symbols[0], [0.0, 0.0, 0.0]], [symbols[1], [0.0, 0.0, float(distance)]]]


def _molecular_payload(spec: MolecularSpec) -> dict[str, Any]:
    return {
        "kind": "molecular",
        "geometry": _geometry_to_jsonable(spec.geometry),
        "basis": str(spec.basis).strip().lower(),
        "charge": int(spec.charge),
        "spin": int(spec.spin),
        "unit": str(spec.unit).strip().lower(),
        "driver": str(spec.driver).strip().lower(),
        "mapping": _normalize_mapping(spec.mapping),
        "active_space": dict(spec.active_space or {}),
        "metadata": dict(spec.metadata),
    }


def _pauli_payload(spec: PauliTermsSpec, terms: tuple[PauliTerm, ...]) -> dict[str, Any]:
    return {
        "kind": "pauli_terms",
        "terms": [[coefficient, pauli] for coefficient, pauli in terms],
        "source": str(spec.source),
        "metadata": dict(spec.metadata),
    }


def _generated_from_pauli_terms(spec: PauliTermsSpec) -> GeneratedHamiltonian:
    terms = _normalize_terms(spec.terms)
    n_qubits = len(terms[0][1])
    payload = _pauli_payload(spec, terms)
    metadata = {"source": str(spec.source), **dict(spec.metadata), "input_kind": "pauli_terms"}
    return GeneratedHamiltonian(
        terms=terms,
        n_qubits=n_qubits,
        hamiltonian_class="pauli_terms",
        hamiltonian_id=f"pauli_terms_{n_qubits}q_{_stable_digest(payload)}",
        metadata=metadata,
    )


def _qiskit_unit(unit: str):
    from qiskit_nature.units import DistanceUnit

    key = str(unit).strip().lower()
    if key in {"angstrom", "ang", "a"}:
        return DistanceUnit.ANGSTROM
    if key in {"bohr", "b"}:
        return DistanceUnit.BOHR
    raise ValueError(f"Unsupported molecular distance unit {unit!r}")


def _qiskit_mapper(mapping: str):
    from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper, ParityMapper

    key = _normalize_mapping(mapping)
    if key == "jordan_wigner":
        return JordanWignerMapper()
    if key == "parity":
        return ParityMapper()
    if key == "bravyi_kitaev":
        return BravyiKitaevMapper()
    raise ValueError(f"Unsupported qubit mapper {mapping!r}")


def _apply_active_space(problem: Any, active_space: Mapping[str, Any] | None) -> Any:
    if not active_space:
        return problem
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    kwargs: dict[str, Any] = {}
    if "num_electrons" in active_space:
        value = active_space["num_electrons"]
        kwargs["num_electrons"] = tuple(value) if isinstance(value, list) else value
    if "num_spatial_orbitals" in active_space:
        kwargs["num_spatial_orbitals"] = int(active_space["num_spatial_orbitals"])
    if "active_orbitals" in active_space:
        kwargs["active_orbitals"] = tuple(int(item) for item in active_space["active_orbitals"])
    transformer = ActiveSpaceTransformer(**kwargs)
    return transformer.transform(problem)


def _sparse_pauli_terms(qubit_op: Any) -> tuple[PauliTerm, ...]:
    if not hasattr(qubit_op, "to_list"):
        raise TypeError("Qiskit mapper returned an object without SparsePauliOp.to_list()")
    raw_terms = []
    for label, coefficient in qubit_op.to_list():
        coeff = _real_float(coefficient)
        if abs(coeff) > 1.0e-14:
            raw_terms.append((coeff, str(label)))
    return _normalize_terms(raw_terms)


def _generated_from_molecular_spec(spec: MolecularSpec) -> GeneratedHamiltonian:
    if str(spec.driver).strip().lower() != "pyscf":
        raise ValueError(f"Unsupported molecular driver {spec.driver!r}")
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
    except ImportError as exc:
        raise ImportError(
            "Molecular Hamiltonian generation requires optional dependencies qiskit-nature and pyscf. "
            "Install them before resolving MolecularSpec inputs."
        ) from exc

    payload = _molecular_payload(spec)
    driver = PySCFDriver(
        atom=_geometry_to_qiskit_atom(spec.geometry),
        basis=str(spec.basis),
        charge=int(spec.charge),
        spin=int(spec.spin),
        unit=_qiskit_unit(spec.unit),
    )
    problem = _apply_active_space(driver.run(), spec.active_space)
    qubit_op = _qiskit_mapper(spec.mapping).map(problem.hamiltonian.second_q_op())
    terms = _sparse_pauli_terms(qubit_op)
    n_qubits = len(terms[0][1])
    metadata = {
        **payload,
        "input_kind": "molecular",
        "qiskit_nature_driver": "PySCFDriver",
        "term_count": len(terms),
    }
    return GeneratedHamiltonian(
        terms=terms,
        n_qubits=n_qubits,
        hamiltonian_class=f"molecular_{_normalize_mapping(spec.mapping)}",
        hamiltonian_id=f"molecular_{n_qubits}q_{_stable_digest({'spec': payload, 'terms': terms})}",
        metadata=metadata,
    )


def generate_hamiltonian(spec: PauliTermsSpec | MolecularSpec) -> GeneratedHamiltonian:
    """Resolve a supported Hamiltonian specification into Pauli terms."""

    if isinstance(spec, PauliTermsSpec):
        return _generated_from_pauli_terms(spec)
    if isinstance(spec, MolecularSpec):
        return _generated_from_molecular_spec(spec)
    raise TypeError(f"Unsupported Hamiltonian spec {type(spec)!r}")


def spec_from_mapping(raw: Mapping[str, Any]) -> PauliTermsSpec | MolecularSpec:
    """Build a typed spec from a JSON-like mapping."""

    kind = str(raw.get("kind", raw.get("type", ""))).strip().lower()
    if not kind and any(key in raw for key in ("molecule", "geometry", "atom", "distance", "bond_length")):
        kind = "molecular"
    if kind in {"pauli", "pauli_terms", "literal", "terms"}:
        return PauliTermsSpec(
            terms=tuple((item[0], item[1]) for item in raw.get("terms", ())),
            source=str(raw.get("source", "manual")),
            metadata=dict(raw.get("metadata", {})),
        )
    if kind in {"molecule", "molecular"}:
        return MolecularSpec(
            geometry=_default_geometry_from_molecule(raw),
            basis=str(raw.get("basis", "sto3g")),
            charge=int(raw.get("charge", 0)),
            spin=int(raw.get("spin", 0)),
            unit=str(raw.get("unit", "angstrom")),
            driver=str(raw.get("driver", "pyscf")),
            mapping=str(raw.get("mapping", "jordan_wigner")),
            active_space=raw.get("active_space"),
            metadata=dict(raw.get("metadata", {})),
        )
    raise ValueError(f"Unsupported Hamiltonian input kind {kind!r}")


def load_hamiltonian_input(path: str | Path) -> GeneratedHamiltonian:
    """Load a Hamiltonian JSON file and resolve it to Pauli terms.

    Supported inputs are either the legacy ``[[coeff, pauli], ...]`` term list,
    or a mapping with ``kind: pauli_terms`` / ``kind: molecular``.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        return generate_hamiltonian(PauliTermsSpec(terms=tuple((item[0], item[1]) for item in raw)))
    if isinstance(raw, dict):
        return generate_hamiltonian(spec_from_mapping(raw))
    raise ValueError("Hamiltonian JSON must be a term list or a specification object")


__all__ = [
    "GeneratedHamiltonian",
    "GeometryAtom",
    "MolecularSpec",
    "PauliTerm",
    "PauliTermsSpec",
    "generate_hamiltonian",
    "load_hamiltonian_input",
    "spec_from_mapping",
]
