from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tm_ecg.modeling.triads import build_triad_memberships
from tm_ecg.ontology import map_ludb_text, map_ptbxl_labels
from tm_ecg.signal.fiducials import verify_fiducial_order
from tm_ecg.types import BeatAcceptance, BeatFiducials


class OntologyAndTriadTests(unittest.TestCase):
    def test_ptbxl_mapping_and_paced_flag(self) -> None:
        labels = map_ptbxl_labels({"scp_codes": "{'NORM': 100, 'AFIB': 80}", "pacemaker": "1"})
        self.assertIn("Normal", labels)
        self.assertIn("AF", labels)
        self.assertIn("Paced", labels)

    def test_ludb_mapping(self) -> None:
        labels = map_ludb_text("Right bundle branch block; atrial flutter")
        self.assertIn("RBBB spectrum", labels)
        self.assertIn("AFL", labels)

    def test_triad_memberships_and_fiducial_order(self) -> None:
        acceptances = [
            BeatAcceptance("b1", "r1", True, ["accepted"]),
            BeatAcceptance("b2", "r1", True, ["accepted"]),
            BeatAcceptance("b3", "r1", False, ["artifact"]),
            BeatAcceptance("b4", "r1", True, ["accepted"]),
            BeatAcceptance("b5", "r1", True, ["accepted"]),
        ]
        triads = build_triad_memberships("r1", acceptances)
        self.assertEqual(len(triads), 2)
        fiducials = BeatFiducials(
            beat_id="b1",
            record_id="r1",
            p_on=1,
            p_peak=2,
            p_off=3,
            qrs_on=4,
            r_peak=5,
            qrs_off=6,
            t_on=7,
            t_peak=8,
            t_off=9,
        )
        self.assertTrue(verify_fiducial_order(fiducials))


if __name__ == "__main__":
    unittest.main()
