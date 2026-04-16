#!/usr/bin/env python3
"""Demonstrate a happy-path query against the sample CGNS file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_cfd import open_cgns
from auto_cfd.cgns_reader import load_zone_data


DEFAULT_FILE = REPO_ROOT / "data" / "yf17_hdf5.cgns"
DEFAULT_FIELDS = ["Pressure", "Temperature", "Velocity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filename",
        nargs="?",
        default=str(DEFAULT_FILE),
        help="Path to the CGNS file to query.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Field names to interpolate at the chosen point.",
    )
    return parser.parse_args()


def first_tetra_centroid(filename: Path) -> tuple[float, float, float]:
    zone_data = load_zone_data(filename)
    tetra_vertices = zone_data.coordinates[zone_data.tetra_connectivity[0]]
    centroid = tetra_vertices.mean(axis=0)
    return tuple(float(value) for value in centroid)


def main() -> int:
    args = parse_args()
    filename = Path(args.filename)
    dataset = open_cgns(filename)

    # Compute query location and query the point
    x, y, z = first_tetra_centroid(filename)
    point = dataset.query_point(x, y, z, args.fields)

    print(f"File: {filename}")
    print(f"Zone: {dataset.zone_name}")
    print(f"Query point: ({point.x}, {point.y}, {point.z})")
    print()

    print("Logical fields available:")
    for field in dataset.fields():
        print(f"  {field.name}: kind={field.kind}, location={field.location}")
    print()

    print("Queried values:")
    for field_name, value in point.values.items():
        print(f"  {field_name}: {value}")
    print()

    print("Component alias access:")
    print(f"  Pressure:    {point['Pressure']:.8f}")
    print(f"  Temperature: {point['Temperature']:.8f}")
    vx, vy, vz = dataset.query_point(x, y, z, ["VelocityX", "VelocityY", "VelocityZ"]).values.values()
    print(f"  VelocityX:   {vx:.8f}")
    print(f"  VelocityY:   {vy:.8f}")
    print(f"  VelocityZ:   {vz:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
