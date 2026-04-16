"""Core data models for the V1 point-query library.

The public library should talk in terms of datasets, zones, fields, and query
results rather than raw CGNS tree nodes. These dataclasses define that
normalized model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


FieldKind = Literal["scalar", "vector"]
FieldLocation = Literal["vertex", "cell"]
FieldValue = float | tuple[float, ...]


@dataclass(frozen=True)
class Field:
    """One logical CFD field.

    ``values`` stores data on the support identified by ``location``:

    - scalar field: shape ``(n_support,)``
    - vector field: shape ``(n_support, n_components)``
    """

    name: str
    kind: FieldKind
    location: FieldLocation
    values: np.ndarray
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "values", values)

        if self.kind == "scalar":
            if values.ndim != 1:
                raise ValueError(
                    f"Scalar field {self.name!r} must be one-dimensional, got shape {values.shape}."
                )
            if self.component_names and self.component_names != (self.name,):
                raise ValueError(
                    f"Scalar field {self.name!r} must not declare component names other than itself."
                )
        elif self.kind == "vector":
            if values.ndim != 2:
                raise ValueError(
                    f"Vector field {self.name!r} must be two-dimensional, got shape {values.shape}."
                )
            if not self.component_names:
                raise ValueError(f"Vector field {self.name!r} must declare component names.")
            if values.shape[1] != len(self.component_names):
                raise ValueError(
                    f"Vector field {self.name!r} has {values.shape[1]} components but "
                    f"{len(self.component_names)} component names."
                )
        else:  # pragma: no cover - protected by Literal typing
            raise ValueError(f"Unsupported field kind {self.kind!r}.")


@dataclass(frozen=True)
class Zone:
    """Normalized V1 zone representation used by the query engine."""

    name: str
    coordinates: np.ndarray
    tetra_connectivity: np.ndarray
    fields: dict[str, Field]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        tetra_connectivity = np.asarray(self.tetra_connectivity, dtype=np.int64)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "tetra_connectivity", tetra_connectivity)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields.keys())

    def resolve_field(self, name: str) -> tuple[Field, int | None]:
        """Resolve a logical field name or one vector component name."""

        field = self.fields.get(name)
        if field is not None:
            return field, None

        for vector_field in self.fields.values():
            if vector_field.kind != "vector":
                continue
            try:
                component_index = vector_field.component_names.index(name)
            except ValueError:
                continue
            return vector_field, component_index

        available = ", ".join(self.field_names)
        raise KeyError(f"Unknown field {name!r}. Available fields: {available}")


@dataclass(frozen=True)
class LocatedTetra:
    """Containing tetrahedron plus interpolation weights for a query point."""

    tetra_index: int
    vertex_indices: np.ndarray
    barycentric_weights: np.ndarray


@dataclass(frozen=True)
class QueryResult:
    """Interpolated CFD state at a physical point."""

    x: float
    y: float
    z: float
    values: dict[str, FieldValue]

    def __getitem__(self, field: str) -> FieldValue:
        return self.values[field]


# Backward-compatible alias while the package settles on ``QueryResult``.
CFDPoint = QueryResult
