"""Shared input cuts precede reservoir/split and are audited in training QA."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS
from pih3l.errors import AnalysisError
from pih3l.io import read_json, write_batch
from pih3l.models import model_specs, scoped_config
from pih3l.training import sample_candidates, train


class BatchSource:
    """Replayable native-like source exposing per-shard upstream counts."""

    def __init__(self, frames):
        self.frames = frames

    def iter_batches(self, **kwargs):
        for frame in self.frames:
            yield frame.copy()


def selected_sample(role, count=96):
    rng = np.random.default_rng(992 + (role == "signal"))
    feature = rng.normal(0.7 if role == "signal" else 0.3, 0.23, count)
    return pd.DataFrame(
        {
            "DcaHe": feature,
            "DcaPi": rng.normal(0, 0.2, count),
            "decay_radius": np.ones(count),
            "NSigmaHe": np.zeros(count),
            "pt_h3l": np.full(count, 3.0),
            "mass_h3l": (
                rng.normal(2.991, 0.0015, count)
                if role == "signal"
                else rng.uniform(3.021, 3.04, count)
            ),
            "XPrimVtx": np.arange(count),
            "YPrimVtx": np.zeros(count),
            "ZPrimVtx": np.zeros(count),
            "_source_file": [role + ".root"] * count,
            "_source_tree": ["DF/tree"] * count,
        }
    )


class TrainingSelectionFlowTests(unittest.TestCase):
    def test_signal_and_background_use_strict_disjoint_mass_windows(self):
        config = self.config()
        masses = np.tile([2.94, 2.95, 2.96, 3.01, 3.02, 3.03], 4)
        for role, expected in (("signal", [2.96, 3.01]), ("background", [2.94, 3.03])):
            with self.subTest(role=role):
                frame = selected_sample(role, len(masses))
                frame.mass_h3l = masses
                sample, flow = sample_candidates(
                    BatchSource([frame]), role, config, return_cutflow=True
                )
                self.assertEqual(flow["after_role_mass_selection"], 8)
                self.assertEqual(sorted(sample.mass_h3l.unique()), expected)

    def config(self):
        config = copy.deepcopy(DEFAULTS)
        config["io"]["chunk_rows"] = 32
        config["training"].update(
            features=["DcaHe", "DcaPi"],
            max_per_class=70,
            min_per_class=4,
            pt_range=[2, 8],
            candidate_cuts={"decay_radius": {"min": 0.8}, "NSigmaHe": {"abs_max": 5}},
            efficiency_min=0.5,
            efficiency_max=0.8,
            npoints_for_effi=4,
            target_signal_efficiency=0.7,
            qa_bins=12,
            qa_shap_max_per_class=6,
        )
        if "mc_range_selection" in config["training"]:
            config["training"]["mc_range_selection"]["enabled"] = False
        config["training"]["model_params"].update(
            n_estimators=8,
            max_depth=2,
            n_jobs=1,
            gamma=0,
            min_child_weight=1,
            subsample=1,
            colsample_bytree=1,
        )
        return config

    def test_shared_cuts_finite_filter_and_raw_source_counts_precede_sampling(self):
        config = self.config()
        for role in ("signal", "background"):
            with self.subTest(role=role):
                good = selected_sample(role, 12)
                outliers = good.iloc[:5].copy()
                outliers.iloc[0, outliers.columns.get_loc("decay_radius")] = 0.1
                outliers.iloc[1, outliers.columns.get_loc("NSigmaHe")] = 9
                outliers.iloc[2, outliers.columns.get_loc("pt_h3l")] = 8
                outliers.iloc[3, outliers.columns.get_loc("mass_h3l")] = 2.991
                outliers.iloc[4, outliers.columns.get_loc("DcaPi")] = np.nan
                frame = pd.concat([good, outliers], ignore_index=True)
                frame.attrs["source_counts"] = {
                    "raw_rows": 24,
                    "after_role_filter": 20,
                    "retained_rows": 17,
                    "nonfinite_kinematics": 3,
                }
                sample, flow = sample_candidates(
                    BatchSource([frame]), role, config, return_cutflow=True
                )
                self.assertEqual(flow["input_rows"], 17)
                self.assertEqual(flow["after_candidate_cuts"], 15)
                self.assertEqual(flow["after_pt_range"], 14)
                self.assertEqual(
                    flow["after_role_mass_selection"], 14 if role == "signal" else 13
                )
                self.assertEqual(
                    flow["after_finite_features"], 13 if role == "signal" else 12
                )
                self.assertEqual(flow["sampled_rows"], len(sample))
                self.assertTrue(flow["source_counts_available"])
                self.assertEqual(flow["raw_source_rows"], 24)
                self.assertEqual(flow["after_source_role_filter"], 20)
                self.assertEqual(flow["after_source_finite_kinematics"], 17)
                self.assertEqual(flow["source_nonfinite_kinematics"], 3)
                self.assertTrue((sample.decay_radius >= 0.8).all())
                self.assertTrue((abs(sample.NSigmaHe) < 5).all())
                self.assertTrue((sample.pt_h3l < 8).all())
                self.assertTrue(
                    np.isfinite(sample[config["training"]["features"]]).all().all()
                )

    def test_prepared_sampling_is_repeatable_bounded_and_does_not_invent_raw_counts(
        self,
    ):
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for index in range(4):
                frame = selected_sample("signal", 64)
                frame.XPrimVtx += 100 * index
                write_batch(source, index, frame)
            resident = []
            original_concat = pd.concat

            def measured_concat(*args, **kwargs):
                result = original_concat(*args, **kwargs)
                resident.append(len(result))
                return result

            with patch("pih3l.training.pd.concat", side_effect=measured_concat):
                first, flow = sample_candidates(
                    source, "signal", config, return_cutflow=True
                )
            second, available = sample_candidates(source, "signal", config)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(first), 70)
            self.assertEqual(available, 256)
            self.assertEqual(flow["after_finite_features"], 256)
            self.assertFalse(flow["source_counts_available"])
            for name in (
                "raw_source_rows",
                "after_source_role_filter",
                "after_source_finite_kinematics",
                "source_nonfinite_kinematics",
            ):
                self.assertIsNone(flow[name])
            self.assertNotIn("_priority", first)
            self.assertEqual(flow["reservoir_limit"], 70)
            self.assertLessEqual(max(resident), 70 + config["io"]["chunk_rows"])

    def test_pt_scoped_cuts_merge_global_rules_without_mutating_common_baseline(self):
        config = self.config()
        config["training"].update(
            pt_range=None,
            candidate_cuts={
                "DcaHe": {"min": 0.1, "max": 2},
                "NSigmaHe": {"abs_max": 5},
            },
        )
        config["training"]["pt_bins"] = [
            {"id": "low", "pt_range": [2, 4], "candidate_cuts": {"DcaHe": {"min": 0.3}}}
        ]
        spec = model_specs(config)[0]
        self.assertEqual(
            spec["candidate_cuts"],
            {"DcaHe": {"min": 0.3, "max": 2}, "NSigmaHe": {"abs_max": 5}},
        )
        scoped = scoped_config(config, spec)
        scoped["training"]["candidate_cuts"]["DcaHe"]["min"] = 0.4
        self.assertEqual(spec["candidate_cuts"]["DcaHe"]["min"], 0.3)
        self.assertEqual(config["training"]["candidate_cuts"]["DcaHe"]["min"], 0.1)

    def test_mc_support_filters_both_samples_without_clipping_surviving_values(self):
        config = self.config()
        for role in ("signal", "background"):
            with self.subTest(role=role):
                frame = selected_sample(role, 8)
                frame.DcaHe = [0, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1]
                frame.DcaPi = 0.0
                sample, flow = sample_candidates(
                    BatchSource([frame]),
                    role,
                    config,
                    support_ranges={"DcaHe": [0.2, 0.8], "DcaPi": [-0.1, 0.1]},
                    return_cutflow=True,
                )
                self.assertEqual(flow["after_finite_features"], 8)
                self.assertEqual(flow["after_mc_range"], 5)
                self.assertEqual(flow["sampled_rows"], 5)
                self.assertEqual(
                    flow["per_variable_mc_range"]["DcaHe"]["rejected_rows"], 3
                )
                self.assertEqual(
                    flow["per_variable_mc_range"]["DcaPi"]["input_rows"], 5
                )
                np.testing.assert_array_equal(sample.DcaHe, [0.2, 0.3, 0.4, 0.5, 0.8])

    def test_mc_enabled_direct_training_cannot_silently_bypass_range_resolution(self):
        config = self.config()
        config["training"]["mc_range_selection"]["enabled"] = True
        with self.assertRaisesRegex(AnalysisError, "resolved reference metadata"):
            train(
                selected_sample("signal", 12), selected_sample("background", 12), config
            )

    def test_inference_scope_includes_frozen_mc_ranges_and_loader_preserves_them(self):
        from pih3l.models import load_model_bundle, score_models
        from pih3l.training_settings import create_model_handler
        from pih3l.io import write_json

        config = self.config()
        metadata = {
            "enabled": True,
            "ranges": {"DcaHe": [0.2, 0.8]},
            "reference": {"group_hash_sha256": "reference-hash"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            handler = create_model_handler(config["training"], 12)
            handler.dump_model_handler(str(directory / "model_handler.pkl"))
            write_json(
                directory / "metrics.json",
                {
                    "features": config["training"]["features"],
                    "training_pt_range": [2, 8],
                    "model_id": "global",
                    "score_output_margin": True,
                    "score_domain": "raw_margin",
                    "training_selection": {"mc_range_selection": metadata},
                },
            )
            bundle = load_model_bundle(directory / "model_handler.pkl")
            self.assertEqual(bundle["mc_range_selection"], metadata)

        class MarginModel:
            def predict(self, frame, *, output_margin):
                if not output_margin:
                    raise AssertionError("Expected raw margins")
                return frame.DcaHe.to_numpy()

        frame = pd.DataFrame(
            {"pt_h3l": [1, 3, 3, 3], "DcaHe": [0.5, 0.2, 0.8, 1], "DcaPi": [0] * 4}
        )
        bundle["model"] = MarginModel()
        scored = score_models(
            frame, {"models": [{"id": "global", "pt_range": [2, 8], "bundle": bundle}]}
        )
        self.assertEqual(scored.bdt_in_scope.tolist(), [False, True, True, False])
        np.testing.assert_allclose(
            scored.bdt_score, [np.nan, 0.2, 0.8, np.nan], equal_nan=True
        )

    def test_actual_selected_inputs_and_metadata_reach_training_and_qa(self):
        from pih3l.root_io import load_root

        load_root()
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cutflows, samples = {}, {}
            for role in ("signal", "background"):
                good = selected_sample(role)
                bad = good.iloc[:30].copy()
                bad.decay_radius = 0.1
                bad.DcaHe = 1e6
                write_batch(base / role, 0, pd.concat([good, bad], ignore_index=True))
                samples[role], cutflows[role] = sample_candidates(
                    base / role, role, config, return_cutflow=True
                )
                self.assertEqual(cutflows[role]["after_candidate_cuts"], 96)
                self.assertEqual(cutflows[role]["after_finite_features"], 96)
            extra = {
                "mc_learned_ranges": {
                    "DcaHe": {
                        "source": "MCHyper",
                        "percentiles": [0, 100],
                        "bounds": [-1, 2],
                    }
                }
            }
            handler, metrics, validation = train(
                samples["signal"],
                samples["background"],
                config,
                qa_directory=base / "training",
                input_cutflow=cutflows,
                selection_metadata=extra,
            )
            self.assertEqual(metrics["available_signal_candidates"], 96)
            self.assertEqual(metrics["available_background_candidates"], 96)
            self.assertEqual(metrics["sampled_signal_candidates"], 70)
            self.assertEqual(metrics["sampled_background_candidates"], 70)
            self.assertTrue((validation.decay_radius >= 0.8).all())
            distributions = metrics["qa"]["feature_distributions"]["DcaHe"]
            self.assertLess(max(distributions["edges"]), 2)
            self.assertEqual(sum(distributions["signal"]), 70)
            self.assertEqual(sum(distributions["background"]), 70)
            index = read_json(base / "training/qa/index.json")
            self.assertEqual(index["selection"], metrics["training_selection"])
            self.assertEqual(index["input_cutflow"], metrics["input_cutflow"])
            self.assertEqual(
                index["selection"]["mc_learned_ranges"], extra["mc_learned_ranges"]
            )
            import ROOT

            with ROOT.TFile.Open(
                str(base / "training/qa/training.root"), "READ"
            ) as archive:
                saved = json.loads(str(archive.Get("metadata/training").GetString()))
                self.assertEqual(saved["input_cutflow"], metrics["input_cutflow"])
                self.assertEqual(saved["available_signal_candidates"], 96)
                self.assertEqual(saved["training_selection"], index["selection"])
                self.assertEqual(archive.Get("features/DcaHe/signal").GetEntries(), 70)
            # The small caller-owned model never receives the rejected 1e6 outlier.
            self.assertEqual(handler.get_training_columns(), ["DcaHe", "DcaPi"])

    def test_mismatched_supplied_sampling_counts_are_rejected_before_fit(self):
        config = self.config()
        samples = {role: selected_sample(role, 12) for role in ("signal", "background")}
        flow = {
            role: {"sampled_rows": 11, "after_finite_features": 12} for role in samples
        }
        with self.assertRaisesRegex(AnalysisError, "cutflow sampled_rows"):
            train(samples["signal"], samples["background"], config, input_cutflow=flow)


if __name__ == "__main__":
    unittest.main()
