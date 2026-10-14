"""Training isolation, application invariance, and real ROOT/Parquet round trips."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hipe4ml.model_handler import ModelHandler
import numpy as np
import pandas as pd
import uproot
from xgboost import XGBClassifier

from pih3l.config import DEFAULTS
from pih3l.errors import AnalysisError
from pih3l.io import (
    discover_files,
    iter_parquet,
    iter_root,
    tree_paths,
    write_batch,
    write_json,
)
from pih3l.training import sample_candidates, score_frame, train, training_groups
from synthetic_data import small_xgb_parameters


def training_sample(label, count=120):
    rng = np.random.default_rng(734 + label)
    return pd.DataFrame(
        {
            "DcaHe": rng.normal(1.5 if label else 0.1, 0.25, count),
            "ct": rng.uniform(2, 20, count),
            "mass_h3l": rng.normal(2.991 if label else 3.03, 0.002, count),
            "XPrimVtx": np.repeat(np.arange(count // 3), 3),
            "YPrimVtx": np.zeros(count),
            "ZPrimVtx": np.zeros(count),
            "_source_file": [f"class{label}.root"] * count,
            "_source_tree": ["DF_1/O2hypcands"] * count,
            "_entry": np.arange(count),
        }
    )


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = copy.deepcopy(DEFAULTS)
        cls.config["training"].update(
            features=["DcaHe"],
            model_params=small_xgb_parameters(),
            qa_shap_max_per_class=40,
            efficiency_min=0.5,
            efficiency_max=0.9,
            npoints_for_effi=9,
        )
        cls.signal, cls.background = training_sample(1), training_sample(0)
        with patch.object(
            ModelHandler,
            "train_test_model",
            autospec=True,
            side_effect=ModelHandler.train_test_model,
        ) as fitting:
            cls.model, cls.metrics, cls.validation = train(
                cls.signal, cls.background, cls.config
            )
        cls.fitting_calls = fitting.call_args_list

    def test_training_uses_real_handler_and_raw_margin_prediction(self):
        self.assertIsInstance(self.model, ModelHandler)
        self.assertIsInstance(self.model.get_original_model(), XGBClassifier)
        self.assertEqual(self.model.get_training_columns(), ["DcaHe"])
        self.assertEqual(len(self.fitting_calls), 1)
        call = self.fitting_calls[0]
        self.assertIs(call.args[0], self.model)
        self.assertTrue(call.kwargs["return_prediction"])
        self.assertTrue(call.kwargs["output_margin"])
        data = call.args[1]
        self.assertEqual(len(data), 4)
        weights = np.asarray(call.kwargs["sample_weight"])
        self.assertEqual(len(weights), len(data[0]))
        self.assertTrue(np.isfinite(weights).all() and (weights > 0).all())
        np.testing.assert_allclose(
            self.validation.bdt_score,
            self.model.predict(self.validation, output_margin=True),
        )
        np.testing.assert_allclose(
            self.validation.bdt_score,
            self.model.get_original_model().predict(
                self.validation[["DcaHe"]], output_margin=True
            ),
        )
        self.assertTrue((self.validation.bdt_score < 0).any())
        self.assertTrue((self.validation.bdt_score > 1).any())

    def test_training_groups_do_not_cross_validation_split(self):
        for label in ["0", "1"]:
            self.assertEqual(self.metrics["split"][label]["group_overlap"], 0)
            self.assertEqual(
                self.metrics["split"][label]["train_rows"]
                + self.metrics["split"][label]["validation_rows"],
                120,
            )
        self.assertGreater(self.metrics["auc_validation"], 0.95)
        self.assertEqual(self.metrics["threshold_source"], "held_out_signal_efficiency")

    def test_efficiency_at_float32_score_boundary_replays_exactly_after_persistence(
        self,
    ):
        from pih3l.validation_scores import (
            read_validation_scores,
            write_validation_scores,
        )

        score = self.model.predict(
            self.validation[self.validation.label == 1], output_margin=True
        )[0]
        cut = np.nextafter(np.float64(score), np.inf)

        def mapped(labels, scores, efficiencies, *, return_metadata=False):
            values = np.full(len(efficiencies), cut)
            return (values, {}) if return_metadata else values

        with patch("pih3l.training.thresholds_from_efficiencies", side_effect=mapped):
            _, metrics, validation = train(self.signal, self.background, self.config)
        self.assertEqual(validation.bdt_score.dtype, np.dtype("float64"))
        with tempfile.TemporaryDirectory() as temporary:
            write_validation_scores(temporary, validation)
            persisted = read_validation_scores(temporary)
        signal = persisted.loc[persisted.label == 1, "bdt_score"].to_numpy()
        np.testing.assert_array_equal(
            metrics["score_efficiency_array"]["actual_efficiency"],
            [
                (signal >= threshold).mean()
                for threshold in metrics["score_efficiency_array"]["threshold"]
            ],
        )

    def test_vertex_grouping_repeats_are_preserved(self):
        groups = training_groups(self.signal)
        self.assertEqual(len(np.unique(groups)), 40)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[3])

    def test_application_ignores_pair_multiplicity_and_pair_pion_variables(self):
        candidate = self.signal.iloc[[0]].copy()
        pair_rows = pd.concat([candidate] * 12, ignore_index=True)
        pair_rows["PairPiTrackPt"] = np.linspace(0.1, 5, 12)
        pair_rows["ChargeChannel"] = np.tile([0, 1, 2, 3], 3)
        pair_rows["IsMixed"] = np.arange(12) % 2 == 0
        score = score_frame(pair_rows, self.model, self.config["training"]["features"])
        np.testing.assert_array_equal(
            score.bdt_score, np.repeat(score.bdt_score.iloc[0], 12)
        )
        self.assertNotIn("bdt_score", pair_rows)

    def test_nonfinite_features_do_not_receive_finite_scores(self):
        frame = self.signal.iloc[:3].copy()
        frame.loc[frame.index[1], "DcaHe"] = np.nan
        frame.loc[frame.index[2], "DcaHe"] = np.inf
        scored = score_frame(frame, self.model, self.config["training"]["features"])
        np.testing.assert_array_equal(
            np.isfinite(scored.bdt_score), [True, False, False]
        )

    def test_insufficient_vertex_groups_rejected(self):
        sample = self.signal.copy()
        sample["XPrimVtx"] = 0
        with self.assertRaises(AnalysisError):
            train(sample, self.background, self.config)

    def test_background_sampling_uses_strict_complement_of_signal_mass_window(self):
        config = copy.deepcopy(self.config)
        config["training"].update(
            {"signal_mass_window": [2.95, 3.02], "max_per_class": 40}
        )
        data = pd.concat([self.background, self.signal], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary:
            write_batch(temporary, 0, data.iloc[:100])
            write_batch(temporary, 1, data.iloc[100:])
            sample, eligible = sample_candidates(temporary, "background", config)
            repeated, repeated_count = sample_candidates(
                temporary, "background", config
            )
            self.assertEqual(eligible, 120)
            self.assertEqual(repeated_count, eligible)
            self.assertEqual(len(sample), 40)
            self.assertTrue((sample.mass_h3l > 3.02).all())
            pd.testing.assert_frame_equal(sample, repeated)


class InputOutputTests(unittest.TestCase):
    def test_json_is_strict_and_nonfinite_values_are_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            write_json(
                path, {"a": np.array([1, np.nan, np.inf]), "integer": np.int64(4)}
            )
            result = json.loads(
                path.read_text(), parse_constant=lambda token: self.fail(token)
            )
            self.assertEqual(result, {"a": [1.0, None, None], "integer": 4})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_optional_missing_input_has_no_fabricated_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(DEFAULTS)
            config["data_root"] = temporary
            self.assertEqual(discover_files(config, "mc_femto"), [])
            with self.assertRaises(AnalysisError):
                discover_files(config, "femto")

    def test_root_chunking_canonical_names_and_provenance(self):
        count = 7
        columns = {
            "f" + name: np.zeros(count, dtype=np.float32)
            for name in [
                "PtHe3",
                "EtaHe3",
                "PhiHe3",
                "PtPi",
                "EtaPi",
                "PhiPi",
                "XPrimVtx",
                "YPrimVtx",
                "ZPrimVtx",
                "XDecVtx",
                "YDecVtx",
                "ZDecVtx",
                "CentralityFT0C",
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "Hyper").mkdir()
            path = folder / "Hyper" / "AO2D.root"
            with uproot.recreate(path) as root:
                root["DF_1/O2hypcands"] = columns
                root["DF_2/O2hypcands"] = columns
            config = copy.deepcopy(DEFAULTS)
            config["data_root"] = str(folder)
            config["io"]["chunk_rows"] = 3
            self.assertEqual(len(tree_paths(path, "background")), 2)
            chunks = list(iter_root(config, "background"))
            self.assertEqual([len(chunk) for chunk in chunks], [3, 3, 1, 3, 3, 1])
            self.assertEqual(sum(map(len, chunks)), 14)
            for chunk in chunks:
                self.assertIn("HypCentFT0C", chunk)
                self.assertIn("_source_file", chunk)
                self.assertNotIn("candidate_uid", chunk)
            np.testing.assert_array_equal(pd.concat(chunks[:3])._entry, np.arange(7))
            cache = folder / "cache"
            for index, chunk in enumerate(chunks):
                write_batch(cache, index, chunk)
            self.assertEqual(
                sum(len(chunk) for chunk in iter_parquet(cache, batch_size=2)), 14
            )


if __name__ == "__main__":
    unittest.main()
