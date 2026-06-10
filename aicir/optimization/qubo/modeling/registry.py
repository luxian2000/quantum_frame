from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableMetadata:
    name: str
    kind: str = "binary"
    role: str = "decision"
    source: str | None = None


@dataclass
class VariableRegistry:
    """Maps external variable names to compact integer IDs."""

    name_to_id: dict[str, int] = field(default_factory=dict)
    id_to_name: list[str] = field(default_factory=list)
    metadata_by_id: list[VariableMetadata] = field(default_factory=list)

    def get_or_create(
        self,
        name: str,
        kind: str = "binary",
        role: str = "decision",
        source: str | None = None,
    ) -> int:
        if name in self.name_to_id:
            return self.name_to_id[name]
        var_id = len(self.id_to_name)
        self.name_to_id[name] = var_id
        self.id_to_name.append(name)
        self.metadata_by_id.append(VariableMetadata(name=name, kind=kind, role=role, source=source))
        return var_id

    def name(self, var_id: int) -> str:
        return self.id_to_name[var_id]

    def names(self) -> list[str]:
        return list(self.id_to_name)

    def metadata(self, var_id: int) -> VariableMetadata:
        return self.metadata_by_id[var_id]

    def variables_by_role(self, role: str) -> list[str]:
        return [metadata.name for metadata in self.metadata_by_id if metadata.role == role]

    def auxiliary_names(self) -> list[str]:
        return self.variables_by_role("auxiliary")


GLOBAL_REGISTRY = VariableRegistry()


@dataclass
class ModelContext:
    """Owns a registry for one modeling session."""

    registry: VariableRegistry = field(default_factory=VariableRegistry)

    def binary(self, name: str):
        from .polynomial import Binary

        return Binary(name, registry=self.registry)

    def auxiliary_binary(self, name: str, source: str | None = None):
        from .polynomial import Binary

        return Binary(name, registry=self.registry, role="auxiliary", source=source)

    def binary_array(self, prefix: str, shape: int | tuple[int, ...]) -> list:
        from .polynomial import binary_array

        return binary_array(prefix, shape, registry=self.registry)

    def integer(
        self,
        name: str,
        lower_bound: int = 0,
        upper_bound: int | None = None,
        encoding: str = "log",
        role: str = "decision",
        source: str | None = None,
    ):
        from .integer import Integer

        return Integer(
            name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            encoding=encoding,
            registry=self.registry,
            role=role,
            source=source,
        )

    def auxiliary_integer(
        self,
        name: str,
        lower_bound: int = 0,
        upper_bound: int | None = None,
        encoding: str = "log",
        source: str | None = None,
    ):
        return self.integer(
            name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            encoding=encoding,
            role="auxiliary",
            source=source,
        )

    def zero(self):
        from .polynomial import Polynomial

        return Polynomial.constant(0.0, registry=self.registry)

    def qubo_builder(self):
        from .builder import QuboBuilder

        return QuboBuilder(registry=self.registry)

    def decode_solution(self, assignment, integers=None, include_auxiliary: bool = False):
        from .solution import decode_solution

        return decode_solution(
            assignment,
            registry=self.registry,
            integers=integers,
            include_auxiliary=include_auxiliary,
        )

