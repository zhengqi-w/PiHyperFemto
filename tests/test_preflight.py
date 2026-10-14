"""Bounded input-only checks follow the actual readers; no model or pipeline runs."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import uproot

from pih3l.config import DEFAULTS
from pih3l.errors import AnalysisError
from pih3l.preflight import validate_inputs
from synthetic_data import candidate_columns


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(DEFAULTS)
        self.config["data_root"] = str(self.root)
        self.config["training"]["features"] = ["DcaHe"]
        self.config["working_point"]["candidate_cuts"] = {}
        self.config["working_point"]["event_cuts"] = {}
        self.frames = {}
        rng = np.random.default_rng(123)
        for role, folder, tree in (
            ("femto", "Femto", "O2pihypfemto"),
            ("background", "Hyper", "O2hypcands"),
            ("signal", "MCHyper", "O2mchypcands"),
        ):
            data = candidate_columns(np.full(40, 2.99131), role == "signal", rng)
            if role == "femto":
                data.update(
                    fIsMixed=np.arange(40) % 2 == 0,
                    fChargeChannel=np.zeros(40, dtype=np.int8),
                    fHadPt=np.full(40, 0.3),
                    fHadEta=np.zeros(40),
                    fHadPhi=np.full(40, 1.0),
                    fIsClosePairRejected=np.zeros(40, dtype=np.uint8),
                )
            self.frames[role] = data
            self.config["inputs"][role].update(
                root=str(self.root / folder), patterns=["AO2D*.root"]
            )
            self.write(role, tree=tree)
        self.config["inputs"]["mc_femto"]["root"] = None

    def write(self, role, *, suffix="", tree=None):
        trees = {
            "femto": "O2pihypfemto",
            "background": "O2hypcands",
            "signal": "O2mchypcands",
            "mc_femto": "O2pihypfemtomc",
        }
        path = Path(self.config["inputs"][role]["root"]) / f"AO2D{suffix}.root"
        path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(path) as archive:
            archive["DF_1/" + (tree or trees[role])] = self.frames[role]

    def check(self):
        return validate_inputs(self.config)

    def test_streaming_does_not_require_femto_candidate_variations_in_training(self):
        self.config["systematics"]["variations"] = [
            {
                "name": "pair_only",
                "overrides": {
                    "analysis": {"candidate_cuts": {"OnlyPairCandidate": {"min": 0}}}
                },
            }
        ]
        self.frames["femto"]["fOnlyPairCandidate"] = np.ones(40)
        self.write("femto")
        self.check()
        self.config["io"]["persist_prepared"] = True
        with self.assertRaisesRegex(AnalysisError, "OnlyPairCandidate"):
            self.check()

    def test_wp_mc_cut_is_required_only_when_mc_tails_are_calibrated(self):
        self.config["working_point"]["candidate_cuts"] = {"WPQuality": {"min": 0}}
        self.frames["background"]["fWPQuality"] = np.ones(40)
        self.write("background")
        with self.assertRaisesRegex(AnalysisError, "WPQuality"):
            self.check()
        self.config["working_point"]["mass_fit"]["calibrate_mc_tails"] = False
        self.check()

    def test_explicit_wp_event_cut_remains_required_when_also_in_analysis(self):
        self.config["analysis"]["event_cuts"] = {"HypTrackOccupancy": {"min": 0}}
        self.config["working_point"]["event_cuts"] = {"HypTrackOccupancy": {"min": 0}}
        self.frames["femto"]["fTrackOccupancy"] = np.ones(40)
        self.write("femto")
        with self.assertRaisesRegex(AnalysisError, "HypTrackOccupancy"):
            self.check()
        self.config["working_point"]["event_cuts"] = None
        self.check()

    def test_preserved_projection_cannot_invent_missing_prepared_wp_columns(self):
        self.config["io"]["persist_prepared"] = True
        self.config["working_point"]["candidate_cuts"] = {"WPQuality": {"min": 0}}
        for role in ("background", "signal"):
            self.frames[role]["fWPQuality"] = np.ones(40)
            self.write(role)
        with self.assertRaisesRegex(AnalysisError, "WPQuality"):
            self.check()
        self.config["io"]["root_projection"] = False
        self.check()

    def test_full_scored_schema_detects_extra_branch_when_projection_disabled(self):
        self.config["io"].update(root_projection=False, slim_scored=False)
        self.frames["femto"]["fExtraDiagnostic"] = np.ones(40)
        self.write("femto", suffix="-later")
        with self.assertRaisesRegex(AnalysisError, "inconsistent scored schemas"):
            self.check()
        self.config["io"]["slim_scored"] = True
        self.check()
        self.config["io"].update(root_projection=True, slim_scored=False)
        self.check()

    def test_required_variation_column_is_checked_in_later_file(self):
        self.config["systematics"]["variations"] = [
            {
                "name": "pion",
                "overrides": {
                    "analysis": {
                        "pair_cuts": {"PairPiTrackTpcNClsCrossedRows": {"min": 70}}
                    }
                },
            }
        ]
        self.write("femto", suffix="-later")
        self.frames["femto"]["fHadTpcNClsCrossedRows"] = np.full(40, 80)
        self.write("femto")
        with self.assertRaisesRegex(AnalysisError, "PairPiTrackTpcNClsCrossedRows"):
            self.check()

    def test_optional_schema_difference_is_detected_before_scoring(self):
        self.frames["femto"]["fTrackOccupancy"] = np.ones(40)
        self.write("femto", suffix="-later")
        with self.assertRaisesRegex(AnalysisError, "inconsistent scored schemas"):
            self.check()
        self.config["io"]["scored_optional_columns"].remove("HypTrackOccupancy")
        self.check()

    def test_mc_truth_heterogeneity_is_not_hidden_by_optional_output_settings(self):
        self.config["inputs"]["mc_femto"].update(
            root=str(self.root / "MCFemto"), patterns=["AO2D*.root"]
        )
        self.config["io"]["scored_optional_columns"] = []
        self.frames["mc_femto"] = copy.deepcopy(self.frames["femto"])
        self.write("mc_femto")
        self.frames["mc_femto"]["fHypIsSignal"] = np.ones(40, dtype=bool)
        self.write("mc_femto", suffix="-later")
        with self.assertRaisesRegex(AnalysisError, "inconsistent scored schemas"):
            self.check()
        self.write("mc_femto")
        self.check()

    def test_probe_decodes_at_most_32_rows_and_never_hashes_full_input(self):
        from pih3l.io import read_root_shard

        self.config["io"]["fingerprint"] = "sha256"
        with patch(
            "pih3l.io.sha256_file", side_effect=AssertionError("full hash")
        ), patch("pih3l.io.read_root_shard", wraps=read_root_shard) as read:
            self.check()
        self.assertEqual(len(read.call_args_list), 3)
        self.assertTrue(all(call.args[2]["stop"] <= 32 for call in read.call_args_list))


if __name__ == "__main__":
    unittest.main()
