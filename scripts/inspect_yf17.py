#!/usr/bin/env python3
"""Minimal pyCGNS learning script for a real unstructured CGNS example.

This script does a read-only pass over ``data/yf17_hdf5.cgns`` and prints:

- the basic CGNS tree structure
- the first base and zone metadata
- coordinate arrays
- element sections
- flow-solution nodes and field statistics

It is intentionally a single script so the data flow is easy to follow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import CGNS.MAP
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "pyCGNS is not installed in this environment.\n"
        "Sync the project environment first, for example:\n"
        "  uv sync"
    ) from exc


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


def walk(node, path: str = "") -> Iterable[tuple[str, list]]:
    current_path = f"{path}/{node_name(node)}" if path else f"/{node_name(node)}"
    yield current_path, node
    for child in node_children(node):
        yield from walk(child, current_path)


def decode_value(value) -> str | int | float | list | None:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            text = "".join(value.astype("U").ravel().tolist())
            return text.replace("\x00", "").strip()
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    return value


def summarize_value(value) -> str:
    if value is None:
        return "None"

    if not isinstance(value, np.ndarray):
        return repr(value)

    if value.dtype.kind in {"S", "U"}:
        return repr(decode_value(value))

    if value.size <= 12:
        return f"dtype={value.dtype}, shape={value.shape}, value={value.tolist()}"

    return f"dtype={value.dtype}, shape={value.shape}"


def print_tree_preview(tree, max_depth: int) -> None:
    print("Tree preview:")
    for path, node in walk(tree):
        depth = max(path.count("/") - 1, 0)
        if depth > max_depth:
            continue
        indent = "  " * depth
        print(f"{indent}- {node_name(node)} [{node_label(node)}]")
    print()


def print_base_summary(base) -> None:
    print(f"Base: {node_name(base)}")
    print(f"  value: {summarize_value(node_value(base))}")

    zones = child_nodes(base, label="Zone_t")
    print(f"  zones: {len(zones)}")
    for zone in zones:
        print_zone_summary(zone)


def print_zone_summary(zone) -> None:
    print(f"  Zone: {node_name(zone)}")
    print(f"    value: {summarize_value(node_value(zone))}")

    zone_type = first_child(zone, label="ZoneType_t")
    print(f"    zone type: {decode_value(node_value(zone_type)) if zone_type else 'unknown'}")

    coords_nodes = child_nodes(zone, label="GridCoordinates_t")
    print(f"    grid-coordinate sets: {len(coords_nodes)}")
    for coords in coords_nodes:
        arrays = child_nodes(coords, label="DataArray_t")
        print(f"      {node_name(coords)}: {[node_name(a) for a in arrays]}")
        for array in arrays:
            arr = node_value(array)
            if isinstance(arr, np.ndarray):
                print(
                    f"        {node_name(array)}: dtype={arr.dtype}, shape={arr.shape}"
                )

    sections = child_nodes(zone, label="Elements_t")
    print(f"    element sections: {len(sections)}")
    for section in sections:
        print(f"      {node_name(section)}: {summarize_value(node_value(section))}")
        data_arrays = child_nodes(section, label="DataArray_t")
        if data_arrays:
            print(f"        data arrays: {[node_name(a) for a in data_arrays]}")
            for data_array in data_arrays:
                value = node_value(data_array)
                if isinstance(value, np.ndarray):
                    print(
                        f"          {node_name(data_array)}: "
                        f"dtype={value.dtype}, shape={value.shape}"
                    )

    solutions = child_nodes(zone, label="FlowSolution_t")
    print(f"    flow solutions: {len(solutions)}")
    for solution in solutions:
        print_solution_summary(solution)

    zone_bc = first_child(zone, label="ZoneBC_t")
    if zone_bc:
        bcs = child_nodes(zone_bc, label="BC_t")
        print(f"    boundary conditions: {[node_name(bc) for bc in bcs]}")


def print_solution_summary(solution) -> None:
    location = first_child(solution, label="GridLocation_t")
    location_value = decode_value(node_value(location)) if location else "Vertex (implicit default)"
    print(f"      {node_name(solution)}")
    print(f"        location: {location_value}")

    fields = child_nodes(solution, label="DataArray_t")
    print(f"        fields: {[node_name(field) for field in fields]}")
    numeric_fields: dict[str, np.ndarray] = {}
    for field in fields:
        arr = node_value(field)
        if not isinstance(arr, np.ndarray):
            print(f"          {node_name(field)}: {arr!r}")
            continue
        if arr.dtype.kind not in "iuf":
            print(f"          {node_name(field)}: dtype={arr.dtype}, shape={arr.shape}")
            continue

        flat = np.asarray(arr).ravel()
        numeric_fields[node_name(field)] = flat
        print(
            "          "
            f"{node_name(field)}: "
            f"dtype={arr.dtype}, shape={arr.shape}, "
            f"min={flat.min():.6g}, max={flat.max():.6g}, mean={flat.mean():.6g}"
        )

    velocity_names = ("VelocityX", "VelocityY", "VelocityZ")
    if all(name in numeric_fields for name in velocity_names):
        speed = np.sqrt(sum(numeric_fields[name] ** 2 for name in velocity_names))
        print(
            "          "
            f"SpeedMagnitude: shape={speed.shape}, "
            f"min={speed.min():.6g}, max={speed.max():.6g}, mean={speed.mean():.6g}"
        )


def inspect_file(filename: Path, max_depth: int) -> None:
    tree, links, skipped_paths = CGNS.MAP.load(str(filename))

    print(f"File: {filename}")
    print(f"Root node: {node_name(tree)} [{node_label(tree)}]")
    print(f"External links: {len(links)}")
    print(f"Deferred/omitted paths from load: {len(skipped_paths)}")
    print()

    print_tree_preview(tree, max_depth=max_depth)

    bases = child_nodes(tree, label="CGNSBase_t")
    print(f"Number of bases: {len(bases)}")
    print()
    for base in bases:
        print_base_summary(base)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filename",
        nargs="?",
        default="data/yf17_hdf5.cgns",
        help="Path to the CGNS/HDF5 file to inspect.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Tree preview depth to print before the detailed summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filename = Path(args.filename)
    if not filename.exists():
        print(f"File not found: {filename}", file=sys.stderr)
        return 1

    inspect_file(filename, max_depth=args.max_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
