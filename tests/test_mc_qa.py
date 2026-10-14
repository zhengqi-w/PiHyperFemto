"""Nonempty MC interface: explicit fraction denominators, SE/ME truth and resolution."""

from pih3l.scored_storage import write_storage_metadata
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS
from pih3l.io import write_batch
from pih3l.mc import run_mc_qa, wilson_interval
from pih3l.physics import derive, four_vector, kstar


class MonteCarloQATests(unittest.TestCase):
    def test_wilson_intervals_have_nonzero_uncertainty_at_boundaries(self):
        result = wilson_interval([0, 5, 10, 0], [10, 10, 10, 0])
        np.testing.assert_array_equal(result["valid"], [True, True, True, False])
        self.assertAlmostEqual(result["low"][0], 0)
        self.assertGreater(result["high"][0], 0)
        self.assertLess(result["low"][2], 1)
        self.assertAlmostEqual(result["high"][2], 1)
        self.assertTrue(np.isnan(result["value"][3]))

    def test_gen_kinematics_supply_kstar_mc_when_branch_is_absent(self):
        config = copy.deepcopy(DEFAULTS)
        config["physics"]["pair_mass"] = "nominal"
        frame = pd.DataFrame(
            {
                "PtHe3": [2.0],
                "EtaHe3": [0.0],
                "PhiHe3": [0.0],
                "PtPi": [0.2],
                "EtaPi": [0.0],
                "PhiPi": [0.0],
                "XPrimVtx": [0.0],
                "YPrimVtx": [0.0],
                "ZPrimVtx": [0.0],
                "XDecVtx": [4.0],
                "YDecVtx": [0.0],
                "ZDecVtx": [0.0],
                "PairPiTrackPt": [0.3],
                "PairPiTrackEta": [0.2],
                "PairPiTrackPhi": [2.0],
                "HypGenPt": [-2.2],
                "HypGenEta": [0.0],
                "HypGenPhi": [0.0],
                "PairPiGenPt": [0.3],
                "PairPiGenEta": [0.2],
                "PairPiGenPhi": [2.0],
            }
        )
        result = derive(frame, config["physics"], pairs=True)
        reference = kstar(
            four_vector([2.2], [0.0], [0.0], config["physics"]["mass_h3l"]),
            four_vector([0.3], [0.2], [2.0], config["physics"]["mass_pi"]),
        )
        np.testing.assert_allclose(result.KstarMC, reference)
        np.testing.assert_allclose(result.KstarMC, result.kstar)

    @staticmethod
    def frame():
        base = pd.DataFrame(
            {
                "IsMixed": [False] * 4 + [True] * 4,
                "HypCentFT0C": [5.0] * 8,
                "ChargeChannel": [0] * 8,
                "IsClosePairRejected": [0] * 8,
                "bdt_score": [0.9] * 8,
                "mass_h3l": [2.99131] * 8,
                "kstar": [0.11] * 8,
                "PairPiIsPhysicalPrimary": [True, False, True, True] * 2,
                "PairPiIsTruePion": [True, True, False, True] * 2,
                "HypIsSignal": [True, True, True, False] * 2,
                "SameMCCollision": [True, True, True, False] + [False] * 4,
                "IsTruthSelfCorrelation": [False] * 8,
                "KstarMC": [0.105] * 8,
            }
        )
        high = base.copy()
        high["kstar"], high["KstarMC"] = 0.35, 0.345
        return pd.concat([base, high], ignore_index=True)

    def test_mc_fraction_denominators_and_me_collision_truth(self):
        config = copy.deepcopy(DEFAULTS)
        config["purity"].update({"method": "fixed", "fixed": 0.8})
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            write_batch(base / "scored", 0, self.frame())
            write_storage_metadata(base / "scored", config, {"storage_threshold": None})
            run_mc_qa(base / "scored", base / "qa", config, 0.5)
            result = json.loads((base / "qa/summary.json").read_text())
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["truth_closure_status"], "available")
            self.assertEqual(result["counters"]["selected_rows"], 16)
            for label in ["se", "me"]:
                item = result["bins"]["cent_0_10__ch_0__" + label]
                index = int(np.argmax(item["total"]))
                self.assertEqual(item["primary_fraction"]["denominator"][index], 4)
                self.assertAlmostEqual(item["primary_fraction"]["value"][index], 0.75)
                self.assertAlmostEqual(item["true_pion_fraction"]["value"][index], 0.75)
                self.assertAlmostEqual(
                    item["primary_pion_fraction"]["value"][index], 0.5
                )
                self.assertEqual(sum(item["residual_count"]), 4)
                finite_bias = np.array(item["resolution_bias"], float)
                np.testing.assert_allclose(
                    finite_bias[np.isfinite(finite_bias)], 0.005, atol=1e-12
                )
            truth = json.loads((base / "qa/truth_closure/results.json").read_text())[
                "cent_0_10__ch_0"
            ]
            self.assertEqual(sum(truth["regions"]["signal"]["se"]), 2)
            self.assertEqual(sum(truth["regions"]["signal"]["me"]), 2)
            self.assertTrue(any(truth["correlations"]["signal"]["valid"]))

    def test_fractions_use_only_the_data_signal_mass_window(self):
        frame = self.frame()
        sideband = frame.copy()
        sideband["mass_h3l"] = 2.97
        sideband["PairPiIsPhysicalPrimary"] = False
        sideband["PairPiIsTruePion"] = False
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            write_batch(
                base / "scored", 0, pd.concat([frame, sideband], ignore_index=True)
            )
            write_storage_metadata(
                base / "scored", copy.deepcopy(DEFAULTS), {"storage_threshold": None}
            )
            run_mc_qa(base / "scored", base / "qa", copy.deepcopy(DEFAULTS), 0.5)
            result = json.loads((base / "qa/summary.json").read_text())
            self.assertEqual(result["counters"]["selected_rows"], 32)
            self.assertEqual(result["counters"]["selected_signal_window_rows"], 16)
            item = result["bins"]["cent_0_10__ch_0__se"]
            self.assertEqual(sum(item["total"]), 8)
            self.assertEqual(sum(item["primary_pion_fraction"]["numerator"]), 4)
            self.assertEqual(
                result["fraction_mass_window"], DEFAULTS["analysis"]["signal_window"]
            )

    def test_missing_truth_is_explicit_not_a_zero_fraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            frame = self.frame().drop(
                columns=["PairPiIsPhysicalPrimary", "PairPiIsTruePion", "KstarMC"]
            )
            write_batch(base / "scored", 0, frame)
            write_storage_metadata(
                base / "scored", copy.deepcopy(DEFAULTS), {"storage_threshold": None}
            )
            run_mc_qa(base / "scored", base / "qa", copy.deepcopy(DEFAULTS), 0.5)
            result = json.loads((base / "qa/summary.json").read_text())
            self.assertEqual(result["truth_closure_status"], "missing_truth_columns")
            item = result["bins"]["cent_0_10__ch_0__se"]
            self.assertEqual(item["primary_fraction"]["status"], "missing_truth_column")
            self.assertEqual(
                item["primary_pion_fraction"]["status"], "missing_truth_columns"
            )
            self.assertFalse((base / "qa/truth_closure").exists())


if __name__ == "__main__":
    unittest.main()
