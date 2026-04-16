"""CGNS-specific loading helpers.

This module is the only part of the V1 library that knows how to walk the
pyCGNS tree. Its job is to validate the narrow happy-path assumptions and
convert the file into the normalized ``Zone`` / ``Field`` model used by the
query engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Field, FieldLocation, Zone

try:
    import CGNS.MAP
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise RuntimeError(
        "pyCGNS is required to load CGNS files. Install it in the project environment."
    ) from exc


TETRA_4 = 10


def load_zone_data(path: str | Path) -> Zone:
    """Load a single happy-path CGNS zone into the normalized V1 model.

    V1 intentionally supports one unstructured zone with one flow solution and
    one tetra volume-element section. The flow solution may be either vertex-
    based or cell-centered.
    """

    filename = Path(path)
    tree, _links, _skipped_paths = CGNS.MAP.load(str(filename))

    base = _require_single_child(tree, label="CGNSBase_t", context="CGNSTree")
    zone = _require_single_child(base, label="Zone_t", context=node_name(base))

    _require_unstructured_zone(zone)
    coordinates = _read_coordinates(zone)
    tetra_connectivity = _read_tetra_connectivity(zone)
    fields = _read_solution_fields(
        zone,
        vertex_count=coordinates.shape[0],
        cell_count=tetra_connectivity.shape[0],
    )

    return Zone(
        name=node_name(zone),
        coordinates=coordinates,
        tetra_connectivity=tetra_connectivity,
        fields=fields,
    )


def node_name(node) -> str:
    return node[0]


def node_value(node):
    return node[1]


def node_children(node) -> list:
    return node[2]


def node_label(node) -> str:
    return node[3]


def child_nodes(node, *, label: str | None = None, name: str | None = None) -> list:
    children = node_children(node)
    if label is not None:
        children = [child for child in children if node_label(child) == label]
    if name is not None:
        children = [child for child in children if node_name(child) == name]
    return children


def first_child(node, *, label: str | None = None, name: str | None = None):
    matches = child_nodes(node, label=label, name=name)
    return matches[0] if matches else None


def decode_text_value(value) -> str | int | float | None:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            text = "".join(value.astype("U").ravel().tolist())
            return text.replace("\x00", "").strip()
        if value.ndim == 0:
            return value.item()

    return value


def _require_single_child(node, *, label: str, context: str):
    """Return one child with the requested CGNS label or fail loudly."""

    matches = child_nodes(node, label=label)
    if len(matches) != 1:
        raise NotImplementedError(
            f"V1 expects exactly one {label} under {context}; found {len(matches)}."
        )
    return matches[0]


def _require_named_child(node, *, label: str, name: str, context: str):
    """Return one named child node or raise a descriptive error."""

    match = first_child(node, label=label, name=name)
    if match is None:
        raise ValueError(f"Missing {label} named {name!r} under {context}.")
    return match


def _require_unstructured_zone(zone) -> None:
    """Validate that the zone matches the only topology supported in V1."""

    zone_type = _require_single_child(zone, label="ZoneType_t", context=node_name(zone))
    zone_type_value = decode_text_value(node_value(zone_type))
    if zone_type_value != "Unstructured":
        raise NotImplementedError(
            f"V1 only supports unstructured zones, got {zone_type_value!r}."
        )


def _read_coordinates(zone) -> np.ndarray:
    """Extract vertex coordinates as an ``(n_vertices, 3)`` float array."""

    coords = _require_single_child(zone, label="GridCoordinates_t", context=node_name(zone))
    coordinate_x = _require_named_child(
        coords,
        label="DataArray_t",
        name="CoordinateX",
        context=node_name(coords),
    )
    coordinate_y = _require_named_child(
        coords,
        label="DataArray_t",
        name="CoordinateY",
        context=node_name(coords),
    )
    coordinate_z = _require_named_child(
        coords,
        label="DataArray_t",
        name="CoordinateZ",
        context=node_name(coords),
    )

    x = np.asarray(node_value(coordinate_x), dtype=np.float64).ravel()
    y = np.asarray(node_value(coordinate_y), dtype=np.float64).ravel()
    z = np.asarray(node_value(coordinate_z), dtype=np.float64).ravel()

    if not (x.size == y.size == z.size):
        raise ValueError("Coordinate arrays must have the same size.")

    return np.column_stack((x, y, z))


def _read_tetra_connectivity(zone) -> np.ndarray:
    """Extract tetra connectivity as zero-based ``(n_tets, 4)`` indices."""

    sections = child_nodes(zone, label="Elements_t")
    tetra_sections = [
        section for section in sections if _element_type_code(section) == TETRA_4
    ]

    if not tetra_sections:
        raise NotImplementedError("V1 requires one tetra volume-element section.")
    if len(tetra_sections) != 1:
        raise NotImplementedError(
            f"V1 supports exactly one tetra section; found {len(tetra_sections)}."
        )

    section = tetra_sections[0]
    connectivity_node = _require_named_child(
        section,
        label="DataArray_t",
        name="ElementConnectivity",
        context=node_name(section),
    )
    element_range_node = _require_named_child(
        section,
        label="IndexRange_t",
        name="ElementRange",
        context=node_name(section),
    )

    connectivity = np.asarray(node_value(connectivity_node), dtype=np.int64).ravel()
    element_range = np.asarray(node_value(element_range_node), dtype=np.int64).ravel()
    if element_range.size != 2:
        raise ValueError("ElementRange must contain exactly two indices.")

    tetra_count = int(element_range[1] - element_range[0] + 1)
    expected_size = tetra_count * 4
    if connectivity.size != expected_size:
        raise ValueError(
            "Tetra connectivity size does not match the declared element range."
        )

    return connectivity.reshape(tetra_count, 4) - 1


def _element_type_code(section) -> int:
    """Read the CGNS element-type code from an ``Elements_t`` section."""

    section_value = np.asarray(node_value(section), dtype=np.int64).ravel()
    if section_value.size < 1:
        raise ValueError(f"Elements_t section {node_name(section)!r} has no type code.")
    return int(section_value[0])


def _read_solution_fields(
    zone,
    *,
    vertex_count: int,
    cell_count: int,
) -> dict[str, Field]:
    """Extract one flow solution into logical scalar/vector fields."""

    solutions = child_nodes(zone, label="FlowSolution_t")
    if len(solutions) != 1:
        raise NotImplementedError(
            f"V1 expects exactly one FlowSolution_t node; found {len(solutions)}."
        )

    solution = solutions[0]
    location_value = "Vertex"
    grid_location = first_child(solution, label="GridLocation_t")
    if grid_location is not None:
        location_value = decode_text_value(node_value(grid_location))

    if location_value == "Vertex":
        expected_size = vertex_count
        location: FieldLocation = "vertex"
    elif location_value == "CellCenter":
        expected_size = cell_count
        location = "cell"
    else:
        raise NotImplementedError(
            f"V1 does not support flow-solution location {location_value!r}."
        )

    raw_fields: dict[str, np.ndarray] = {}
    for field_node in child_nodes(solution, label="DataArray_t"):
        field_name = node_name(field_node)
        raw_values = np.asarray(node_value(field_node))
        if raw_values.ndim != 1:
            raise ValueError(f"Field {field_name!r} must be one-dimensional in V1.")
        if raw_values.size != expected_size:
            raise ValueError(
                f"Field {field_name!r} has size {raw_values.size}, expected {expected_size}."
            )
        if raw_values.dtype.kind not in "iuf":
            raise ValueError(f"Field {field_name!r} must be numeric in V1.")
        raw_fields[field_name] = raw_values.astype(np.float64, copy=False)

    if not raw_fields:
        raise ValueError("No flow-solution fields were found in the zone.")

    return _build_fields(raw_fields, location=location)


def _build_fields(
    raw_fields: dict[str, np.ndarray],
    *,
    location: FieldLocation,
) -> dict[str, Field]:
    """Convert raw scalar arrays into logical scalar/vector fields."""

    fields: dict[str, Field] = {}
    consumed: set[str] = set()

    for field_name, values in raw_fields.items():
        if field_name in consumed:
            continue

        if field_name.endswith("X"):
            field_prefix = field_name[:-1]
            component_names = (
                f"{field_prefix}X",
                f"{field_prefix}Y",
                f"{field_prefix}Z",
            )
            if all(component_name in raw_fields for component_name in component_names):
                component_values = np.column_stack(
                    [raw_fields[component_name] for component_name in component_names]
                )
                fields[field_prefix] = Field(
                    name=field_prefix,
                    kind="vector",
                    location=location,
                    values=component_values,
                    component_names=component_names,
                )
                consumed.update(component_names)
                continue

        fields[field_name] = Field(
            name=field_name,
            kind="scalar",
            location=location,
            values=values,
            component_names=(field_name,),
        )

    return fields
