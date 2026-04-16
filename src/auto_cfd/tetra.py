"""Geometry helpers for V1 tetrahedral point queries.

This module knows nothing about CGNS. It works entirely on normalized zone and
field objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Field, FieldValue, LocatedTetra, Zone


@dataclass(frozen=True)
class TetraLocator:
    """Bounding-box filter plus barycentric point-in-tetra lookup."""

    tetra_connectivity: np.ndarray
    tetra_vertices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    tolerance: float = 1e-10

    @classmethod
    def from_zone(cls, zone: Zone, *, tolerance: float = 1e-10) -> "TetraLocator":
        """Precompute per-tetra geometry used during repeated point queries."""

        tetra_vertices = zone.coordinates[zone.tetra_connectivity]
        bbox_min = tetra_vertices.min(axis=1)
        bbox_max = tetra_vertices.max(axis=1)
        return cls(
            tetra_connectivity=zone.tetra_connectivity,
            tetra_vertices=tetra_vertices,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            tolerance=tolerance,
        )

    def locate(self, x: float, y: float, z: float) -> LocatedTetra:
        """Find the tetrahedron that contains the query point.

        The search first applies a cheap bounding-box filter and then uses a
        barycentric containment test on the remaining candidates.
        """

        point = np.asarray((x, y, z), dtype=np.float64)
        candidate_mask = np.logical_and(
            np.all(point >= (self.bbox_min - self.tolerance), axis=1),
            np.all(point <= (self.bbox_max + self.tolerance), axis=1),
        )
        candidate_indices = np.flatnonzero(candidate_mask)

        if candidate_indices.size == 0:
            raise ValueError(f"Point ({x}, {y}, {z}) is outside all tetra bounding boxes.")

        for tetra_index in candidate_indices:
            weights = barycentric_weights(
                point,
                self.tetra_vertices[tetra_index],
                tolerance=self.tolerance,
            )
            if weights is None:
                continue
            return LocatedTetra(
                tetra_index=int(tetra_index),
                vertex_indices=self.tetra_connectivity[tetra_index],
                barycentric_weights=weights,
            )

        raise ValueError(f"Point ({x}, {y}, {z}) was not found inside any tetrahedron.")


def barycentric_weights(
    point: np.ndarray,
    tetra_vertices: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray | None:
    """Return barycentric weights when the point is inside the tetrahedron.

    ``None`` means the point lies outside the tetrahedron or the tetrahedron is
    degenerate and cannot be solved robustly.
    """

    origin = tetra_vertices[0]
    matrix = np.column_stack(
        (
            tetra_vertices[1] - origin,
            tetra_vertices[2] - origin,
            tetra_vertices[3] - origin,
        )
    )

    try:
        local_coordinates = np.linalg.solve(matrix, point - origin)
    except np.linalg.LinAlgError:
        return None

    weights = np.empty(4, dtype=np.float64)
    weights[1:] = local_coordinates
    weights[0] = 1.0 - np.sum(local_coordinates)

    if np.any(weights < -tolerance) or np.any(weights > 1.0 + tolerance):
        return None

    weights[np.abs(weights) < tolerance] = 0.0
    return weights / weights.sum()


def sample_field(field: Field, located_tetra: LocatedTetra) -> FieldValue:
    """Sample one field inside the located tetrahedron.

    Vertex-based fields are interpolated linearly with barycentric weights.
    Cell-centered fields are treated as piecewise constant over the tetra.
    """

    if field.location == "vertex":
        supported_values = field.values[located_tetra.vertex_indices]
        sampled = np.dot(located_tetra.barycentric_weights, supported_values)
    elif field.location == "cell":
        sampled = field.values[located_tetra.tetra_index]
    else:  # pragma: no cover - protected by FieldLocation typing
        raise ValueError(f"Unsupported field location {field.location!r}.")

    if field.kind == "scalar":
        return float(np.asarray(sampled, dtype=np.float64))

    vector = np.asarray(sampled, dtype=np.float64)
    return tuple(float(value) for value in vector.tolist())
