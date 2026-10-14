"""Current compact table support and exact bin/bitmask semantics."""

import copy
import unittest

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS
from pih3l.errors import AnalysisError, SchemaError
from pih3l.schema import adapt, canonical_name
from pih3l.selection import cut_mask, in_window, pair_weights, select_pairs


def compact_pairs():
    count = 4
    data = {
        "fHyp" + name: np.ones(count)
        for name in [
            "PtHe3",
            "EtaHe3",
            "PhiHe3",
            "PtPi",
            "EtaPi",
            "PhiPi",
            "XDecVtx",
            "YDecVtx",
            "ZDecVtx",
        ]
    }
    data.update(
        {"f" + name: np.zeros(count) for name in ["XPrimVtx", "YPrimVtx", "ZPrimVtx"]}
    )
    data.update(
        {
            "fCentFT0C": [0, 10, 20, 49.99],
            "fHypIsMatter": [True, True, False, False],
            "fHadSign": [1, -1, -1, 1],
            "fHadPt": np.ones(count),
            "fHadEta": np.zeros(count),
            "fHadPhi": np.zeros(count),
            "fIsMixed": [False, True, False, True],
        }
    )
    return pd.DataFrame(data)


class SchemaAndSelectionTests(unittest.TestCase):
    def test_compact_schema_derives_all_four_channels_without_ids(self):
        result = adapt(compact_pairs(), "femto")
        np.testing.assert_array_equal(result.ChargeChannel, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.HypCentFT0C, [0, 10, 20, 49.99])
        self.assertNotIn("candidate_uid", result)
        self.assertNotIn("event_uid", result)
        self.assertNotIn("PairCentFT0C", result)

    def test_current_and_extended_aliases(self):
        names = {
            "fHypPtHe3": "PtHe3",
            "fPtHe3": "PtHe3",
            "fHypXDecVtx": "XDecVtx",
            "fXPrimVtx": "XPrimVtx",
            "fHadDcaXY": "PairPiTrackDcaXY",
            "fCentralityFT0C": "HypCentFT0C",
            "fHypCentFT0C": "HypCentFT0C",
            "fIsSignal": "HypIsSignal",
            "fIsReco": "HypIsReco",
            "fHadGenPt": "PairPiGenPt",
        }
        for name, expected in names.items():
            with self.subTest(name=name):
                self.assertEqual(canonical_name(name), expected)

    def test_ambiguous_aliases_rejected(self):
        frame = compact_pairs()
        frame["fHypCentFT0C"] = frame.fCentFT0C
        with self.assertRaises(SchemaError):
            adapt(frame, "femto")

    def test_invalid_hadron_charge_is_not_assigned_a_channel(self):
        frame = compact_pairs()
        frame.loc[0, "fHadSign"] = 0
        with self.assertRaises(SchemaError):
            adapt(frame, "femto")

    def test_requested_missing_cut_raises(self):
        with self.assertRaises(SchemaError):
            cut_mask(pd.DataFrame({"ct": [5]}), {"PairPiTrackDcaZ": {"abs_max": 2}})

    def test_nonfinite_values_fail_cuts(self):
        frame = pd.DataFrame({"ct": [1.0, 5.0, 10.0, np.nan, np.inf]})
        np.testing.assert_array_equal(
            cut_mask(frame, {"ct": [1, 10]}), [True, True, False, False, False]
        )

    def test_centrality_boundaries_are_half_open(self):
        cent = np.array([-1, 0, 9.99, 10, 49.99, 50, np.nan])
        first = in_window(cent, [0, 10])
        second = in_window(cent, [10, 50])
        np.testing.assert_array_equal(
            first, [False, True, True, False, False, False, False]
        )
        np.testing.assert_array_equal(
            second, [False, False, False, True, True, False, False]
        )
        self.assertFalse(np.any(first & second))

    def test_close_pair_unavailable_bits_are_not_rejection_bits(self):
        frame = pd.DataFrame(
            {"IsClosePairRejected": np.arange(16), "bdt_score": np.ones(16)}
        )
        mask, flow = select_pairs(
            frame, copy.deepcopy(DEFAULTS["analysis"]), threshold=0.5
        )
        np.testing.assert_array_equal(np.flatnonzero(mask), [0, 4, 8, 12])
        self.assertEqual(flow["bdt"], 4)

    def test_nan_mixing_depth_raises_instead_of_silently_losing_pairs(self):
        frame = pd.DataFrame({"IsMixed": [False, True], "MixingDepth": [0, np.nan]})
        with self.assertRaises(AnalysisError):
            pair_weights(frame, "inverse_depth")

    def test_weight_default_is_unit_and_inverse_depth_only_affects_me(self):
        frame = pd.DataFrame({"IsMixed": [False, True, True], "MixingDepth": [0, 2, 5]})
        np.testing.assert_allclose(pair_weights(frame, "unit"), [1, 1, 1])
        np.testing.assert_allclose(pair_weights(frame, "inverse_depth"), [1, 0.5, 0.2])
        frame.loc[1, "MixingDepth"] = 0
        with self.assertRaises(AnalysisError):
            pair_weights(frame, "inverse_depth")


if __name__ == "__main__":
    unittest.main()
