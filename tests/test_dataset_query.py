from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_cfd.dataset import CGNSDataset
from auto_cfd.models import Field, Zone


class DatasetQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        coordinates = np.asarray(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            dtype=np.float64,
        )
        tetra_connectivity = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        fields = {
            "Temperature": Field(
                name="Temperature",
                kind="scalar",
                location="vertex",
                values=np.asarray([100.0, 200.0, 300.0, 400.0]),
                component_names=("Temperature",),
            ),
            "Velocity": Field(
                name="Velocity",
                kind="vector",
                location="vertex",
                values=np.asarray(
                    [
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                        (1.0, 1.0, 1.0),
                    ],
                    dtype=np.float64,
                ),
                component_names=("VelocityX", "VelocityY", "VelocityZ"),
            ),
            "Pressure": Field(
                name="Pressure",
                kind="scalar",
                location="cell",
                values=np.asarray([101325.0]),
                component_names=("Pressure",),
            ),
        }
        self.dataset = CGNSDataset(
            path="synthetic.cgns",
            zone_data=Zone(
                name="SyntheticZone",
                coordinates=coordinates,
                tetra_connectivity=tetra_connectivity,
                fields=fields,
            ),
        )

    def test_query_point_handles_vertex_vector_and_cell_scalar_fields(self) -> None:
        point = self.dataset.query_point(
            0.25,
            0.25,
            0.25,
            ["Temperature", "Velocity", "Pressure"],
        )

        self.assertAlmostEqual(point["Temperature"], 250.0)
        self.assertEqual(point["Velocity"], (0.5, 0.5, 0.5))
        self.assertAlmostEqual(point["Pressure"], 101325.0)

    def test_query_points_reuses_the_same_field_contract(self) -> None:
        points = self.dataset.query_points(
            [
                (0.25, 0.25, 0.25),
                (0.1, 0.2, 0.3),
            ],
            ["Temperature", "VelocityX"],
        )

        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0]["Temperature"], 250.0)
        self.assertAlmostEqual(points[0]["VelocityX"], 0.5)
        self.assertAlmostEqual(points[1]["Temperature"], 240.0)
        self.assertAlmostEqual(points[1]["VelocityX"], 0.7)


if __name__ == "__main__":
    unittest.main()
