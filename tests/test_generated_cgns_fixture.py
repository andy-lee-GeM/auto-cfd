from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_cfd import open_cgns

from fixture_builders import write_single_tetra_fixture


class GeneratedCGNSFixtureTest(unittest.TestCase):
    def test_generated_single_tetra_file_can_be_loaded_and_queried(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = write_single_tetra_fixture(Path(tmpdir) / "single_tetra.cgns")

            dataset = open_cgns(fixture_path)
            point = dataset.query_point(0.25, 0.25, 0.25, ["Temperature", "Velocity"])

        self.assertEqual(dataset.zone_name, "Zone1")
        self.assertAlmostEqual(point["Temperature"], 250.0)
        self.assertEqual(point["Velocity"], (0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
