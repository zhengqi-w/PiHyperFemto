"""MC calibration isolation and a consistent inference-domain contract."""

import copy
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pih3l.config import DEFAULTS
from pih3l.errors import AnalysisError
from pih3l.models import score_models
from pih3l.score_cache import cached_model_scores, score_key
from pih3l.training import train, training_groups
from pih3l.training_support import mc_range_mask, prepare_mc_reference


class CountingModel:
    def __init__(self):
        self.rows = 0

    def predict(self, frame, output_margin=False):
        self.rows += len(frame)
        return np.full(len(frame), 0.7)


class TrainingSupportTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(DEFAULTS)
        self.config["training"].update(
            features=["DcaHe", "NSigmaHe", "NTPCCrossedRowsHe"], min_per_class=8
        )
        self.config["training"]["mc_range_selection"].update(
            enabled=True,
            quantiles=[0.1, 0.9],
            margin_fraction=0.05,
            reference_fraction=0.25,
            min_reference_rows=20,
        )
        rng = np.random.default_rng(917)
        self.signal = pd.DataFrame(
            {
                "DcaHe": rng.normal(0.5, 0.1, 200),
                "NSigmaHe": rng.normal(0, 1, 200),
                "NTPCCrossedRowsHe": np.full(200, 150),
                "_source_file": ["MC.root"] * 200,
                "_source_tree": ["MCHyper"] * 200,
                "XPrimVtx": np.repeat(np.arange(50), 4),
                "YPrimVtx": np.zeros(200),
                "ZPrimVtx": np.zeros(200),
            }
        )

    def test_reference_groups_are_removed_and_nonreference_outliers_cannot_move_bounds(
        self,
    ):
        original = self.signal.copy(deep=True)
        groups = training_groups(original)
        pool, reference = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.25, random_state=self.config["seed"] + 107
            ).split(original, groups=groups)
        )
        selected, definition = prepare_mc_reference(original, self.config)
        self.assertEqual(definition["reference"]["overlap_with_training_pool"], 0)
        self.assertEqual(definition["reference"]["rows"], len(reference))
        self.assertFalse(set(training_groups(selected)) & set(groups[reference]))
        self.assertFalse(
            definition["reference"]["used_for_model_or_efficiency_mapping"]
        )
        self.assertIn("NTPCCrossedRowsHe", definition["skipped_features"])
        self.assertNotIn("NTPCCrossedRowsHe", definition["ranges"])
        for name, bounds in definition["ranges"].items():
            low, high = np.quantile(original.iloc[reference][name], [0.1, 0.9])
            np.testing.assert_allclose(
                bounds, [low - 0.05 * (high - low), high + 0.05 * (high - low)]
            )
            self.assertTrue(selected[name].between(*bounds, inclusive="both").all())
        changed = original.copy(deep=True)
        changed.loc[pool[0], "DcaHe"] = 1e8
        _, other = prepare_mc_reference(changed, self.config)
        self.assertEqual(definition["ranges"], other["ranges"])
        self.assertEqual(
            definition["reference"]["group_hash_sha256"],
            other["reference"]["group_hash_sha256"],
        )
        pd.testing.assert_frame_equal(self.signal, original)

    def test_range_edges_nonfinite_and_sequential_vs_independent_rejections(self):
        frame = pd.DataFrame({"DcaHe": [0, 1, 2, np.nan], "NSigmaHe": [0, 2, 0, 0]})
        mask, counts = mc_range_mask(
            frame, {"DcaHe": [0, 1], "NSigmaHe": [0, 1]}, return_cutflow=True
        )
        np.testing.assert_array_equal(mask, [True, False, False, False])
        self.assertEqual(counts["DcaHe"]["rejected_rows"], 2)
        self.assertEqual(counts["NSigmaHe"]["input_rows"], 2)
        self.assertEqual(counts["NSigmaHe"]["retained_rows"], 1)
        self.assertEqual(counts["NSigmaHe"]["independent_rejected_rows"], 1)
        with self.assertRaises(AnalysisError):
            mc_range_mask(frame, {"DcaHe": [2, 1]})

    def test_disabled_and_insufficient_reference_do_not_change_or_borrow_samples(self):
        self.config["training"]["mc_range_selection"]["enabled"] = False
        returned, definition = prepare_mc_reference(self.signal, self.config)
        self.assertIs(returned, self.signal)
        self.assertFalse(definition["enabled"])
        self.config["training"]["mc_range_selection"].update(
            enabled=True, min_reference_rows=100
        )
        with self.assertRaisesRegex(AnalysisError, "reference has"):
            prepare_mc_reference(self.signal, self.config)

    def test_direct_training_cannot_bypass_resolved_mc_range_preselection(self):
        with self.assertRaisesRegex(AnalysisError, "outside its resolved MC ranges"):
            train(
                self.signal,
                self.signal.copy(),
                self.config,
                selection_metadata={
                    "mc_range_selection": {
                        "enabled": True,
                        "ranges": {"DcaHe": [0.3, 0.7]},
                        "reference": {"group_hash_sha256": "resolved-reference"},
                    }
                },
            )

    def test_direct_and_cached_inference_share_frozen_domain_and_cache_identity(self):
        model = CountingModel()
        frame = pd.DataFrame({"DcaHe": [0.2, 0.8, 1.0, np.nan]})
        spec = {
            "id": "global",
            "pt_range": None,
            "bundle": {
                "model": model,
                "features": ["DcaHe"],
                "mc_range_selection": {
                    "enabled": True,
                    "ranges": {"DcaHe": [0.2, 0.8]},
                },
            },
            "inference_checksum": "same-handler",
            "inference_code": "same-code",
        }
        direct = score_models(frame, {"models": [spec]})
        np.testing.assert_array_equal(direct.bdt_in_scope, [True, True, False, False])
        self.assertEqual(model.rows, 2)
        with tempfile.TemporaryDirectory() as temporary:
            for _ in range(2):
                cached = cached_model_scores(
                    frame,
                    {"models": [spec]},
                    checkpoint_root=temporary,
                    source={"input": "fixed"},
                    active_paths=set(),
                )
                np.testing.assert_array_equal(cached.bdt_in_scope, direct.bdt_in_scope)
                np.testing.assert_allclose(
                    cached.bdt_score, direct.bdt_score, equal_nan=True
                )
            self.assertEqual(model.rows, 4)
            changed = copy.deepcopy(spec)
            changed["bundle"]["mc_range_selection"]["ranges"]["DcaHe"] = [0.2, 0.7]
            self.assertNotEqual(score_key({}, spec), score_key({}, changed))


if __name__ == "__main__":
    unittest.main()
