"""Public dataset API for the V1 point-query library.

Most users should only need this module. It hides CGNS parsing and tetrahedral
geometry behind a small dataset-oriented interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .cgns_reader import load_zone_data
from .models import CFDPoint, Field, QueryResult, Zone
from .tetra import TetraLocator, sample_field


class CGNSDataset:
    """Happy-path V1 dataset wrapper around one unstructured tetra zone."""

    def __init__(self, path: str | Path, zone_data: Zone):
        self.path = Path(path)
        self._zone_data = zone_data
        self._locator = TetraLocator.from_zone(zone_data)

    @property
    def zone_name(self) -> str:
        return self._zone_data.name

    @property
    def field_names(self) -> tuple[str, ...]:
        return self._zone_data.field_names

    def zones(self) -> tuple[str, ...]:
        return (self.zone_name,)

    def fields(self) -> tuple[Field, ...]:
        return tuple(self._zone_data.fields.values())

    def query_point(
        self,
        x: float,
        y: float,
        z: float,
        fields: str | Iterable[str],
    ) -> QueryResult:
        """Sample the requested fields at one physical point."""

        field_names = _normalize_fields(fields)
        located_tetra = self._locator.locate(x, y, z)

        values = {}
        for field_name in field_names:
            field, component_index = self._zone_data.resolve_field(field_name)
            sampled_value = sample_field(field, located_tetra)
            if component_index is not None:
                sampled_value = sampled_value[component_index]
            values[field_name] = sampled_value

        return QueryResult(x=x, y=y, z=z, values=values)

    def query_points(
        self,
        coordinates: Iterable[tuple[float, float, float]],
        fields: str | Iterable[str],
    ) -> list[QueryResult]:
        """Sample the requested fields at multiple points."""

        field_names = _normalize_fields(fields)
        return [
            self.query_point(x, y, z, field_names)
            for x, y, z in coordinates
        ]

    @property
    def zone(self) -> Zone:
        return self._zone_data


def open_cgns(path: str | Path) -> CGNSDataset:
    """Load a CGNS file into the V1 happy-path query dataset."""

    return CGNSDataset(path=path, zone_data=load_zone_data(path))


def _normalize_fields(fields: str | Iterable[str]) -> list[str]:
    """Accept either one field name or an iterable of field names."""

    if isinstance(fields, str):
        normalized = [fields]
    else:
        normalized = list(fields)

    if not normalized:
        raise ValueError("query_point() requires at least one field name.")

    return normalized
