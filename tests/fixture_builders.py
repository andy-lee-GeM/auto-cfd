from __future__ import annotations

from pathlib import Path

import numpy as np

import CGNS.MAP
import CGNS.PAT.cgnslib as C
import CGNS.PAT.cgnskeywords as CK


def write_single_tetra_fixture(path: str | Path) -> Path:
    """Write a minimal unstructured CGNS file with one tetrahedral cell.

    The generated file contains:
    - one base
    - one unstructured zone
    - four vertices
    - one tetra element
    - one vertex-based flow solution with one scalar and one vector field
    """

    fixture_path = Path(path)

    tree = C.newCGNSTree()
    base = C.newCGNSBase(tree, "Base", 3, 3)

    # Unstructured zsize is a 3x1 array:
    #   [number of vertices, number of cells, number of boundary vertices]
    zsize = np.array([[4], [1], [0]], dtype=np.int32)
    zone = C.newZone(base, "Zone1", zsize, CK.Unstructured_s)

    # Coordinates are stored as separate DataArray_t nodes under GridCoordinates_t.
    grid_coordinates = C.newGridCoordinates(zone, CK.GridCoordinates_s)
    C.newDataArray(
        grid_coordinates,
        CK.CoordinateX_s,
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    )
    C.newDataArray(
        grid_coordinates,
        CK.CoordinateY_s,
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )
    C.newDataArray(
        grid_coordinates,
        CK.CoordinateZ_s,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )

    # Connectivity uses 1-based vertex indices in the CGNS file.
    C.newElements(
        zone,
        "Tetras",
        CK.TETRA_4,
        np.array([1, 1], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
    )

    # FlowSolution_t groups field arrays that share the same GridLocation_t.
    flow_solution = C.newFlowSolution(zone, "FlowSolution", CK.Vertex_s)
    C.newDataArray(
        flow_solution,
        "Temperature",
        np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64),
    )
    C.newDataArray(
        flow_solution,
        "VelocityX",
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )
    C.newDataArray(
        flow_solution,
        "VelocityY",
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
    )
    C.newDataArray(
        flow_solution,
        "VelocityZ",
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    )

    CGNS.MAP.save(str(fixture_path), tree)
    return fixture_path
