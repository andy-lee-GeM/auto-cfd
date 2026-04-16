from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_cfd import open_cgns
from auto_cfd.cgns_reader import load_zone_data


SAMPLE_FILE = REPO_ROOT / "data" / "yf17_hdf5.cgns"
FIELDS = ["Pressure", "Temperature", "Velocity"]


class QueryPointIntegrationTest(unittest.TestCase):
    def test_query_point_interpolates_scalar_and_vector_fields_at_first_tetra_centroid(self) -> None:
        zone_data = load_zone_data(SAMPLE_FILE)
        dataset = open_cgns(SAMPLE_FILE)

        vertex_indices = zone_data.tetra_connectivity[0]
        centroid = zone_data.coordinates[vertex_indices].mean(axis=0)

        point = dataset.query_point(
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
            FIELDS,
        )

        self.assertEqual(point.values.keys(), dict.fromkeys(FIELDS).keys())
        field_names = tuple(field.name for field in dataset.fields())
        self.assertIn("Pressure", field_names)
        self.assertIn("Temperature", field_names)
        self.assertIn("Velocity", field_names)

        expected_pressure = float(zone_data.fields["Pressure"].values[vertex_indices].mean())
        expected_temperature = float(zone_data.fields["Temperature"].values[vertex_indices].mean())
        expected_velocity = tuple(
            float(value)
            for value in zone_data.fields["Velocity"].values[vertex_indices].mean(axis=0)
        )

        np.testing.assert_allclose(
            point["Pressure"],
            expected_pressure,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Interpolated pressure did not match the tetra-centroid average.",
        )
        np.testing.assert_allclose(
            point["Temperature"],
            expected_temperature,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Interpolated temperature did not match the tetra-centroid average.",
        )
        np.testing.assert_allclose(
            point["Velocity"],
            expected_velocity,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Interpolated velocity did not match the tetra-centroid average.",
        )

    def test_query_point_supports_vector_component_aliases(self) -> None:
        zone_data = load_zone_data(SAMPLE_FILE)
        dataset = open_cgns(SAMPLE_FILE)

        vertex_indices = zone_data.tetra_connectivity[0]
        centroid = zone_data.coordinates[vertex_indices].mean(axis=0)

        point = dataset.query_point(
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
            ["VelocityX", "VelocityY", "VelocityZ"],
        )

        expected_components = zone_data.fields["Velocity"].values[vertex_indices].mean(axis=0)
        np.testing.assert_allclose(point["VelocityX"], expected_components[0], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(point["VelocityY"], expected_components[1], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(point["VelocityZ"], expected_components[2], rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
