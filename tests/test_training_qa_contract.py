"""Official HIPE4ML QA plots and their canonical numerical-artifact contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS
from pih3l.io import read_json
from pih3l.training_qa import (
    TRAINING_QA_ARTIFACTS,
    build_training_qa,
    render_training_qa,
)
from pih3l.training_settings import create_model_handler


class TrainingQAContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Wrapping official plot_utils imports SHAP/Numba. Initialize Cling
        # before that import, just as the production renderer preflight does.
        from pih3l.root_io import load_root

        load_root()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.config = copy.deepcopy(DEFAULTS)
        cls.config["training"].update(
            features=["DcaHe", "DcaPi"], qa_bins=12, qa_shap_max_per_class=6
        )
        cls.config["training"]["model_params"].update(
            n_estimators=4,
            max_depth=2,
            n_jobs=1,
            gamma=0,
            min_child_weight=1,
            subsample=1,
            colsample_bytree=1,
        )
        rng = np.random.default_rng(971)

        def candidates(count):
            label = np.arange(count) % 2
            feature = rng.normal(0.9 * label, 0.5, count)
            return pd.DataFrame(
                {
                    "label": label,
                    "DcaHe": feature,
                    "DcaPi": 0.4 * feature + rng.normal(0, 0.2, count),
                    "mass_h3l": np.where(
                        label,
                        rng.normal(2.991, 0.0015, count),
                        rng.uniform(2.96, 3.025, count),
                    ),
                }
            )

        training, validation = candidates(64), candidates(32)
        handler = create_model_handler(cls.config["training"], cls.config["seed"])
        handler.fit(training, training.label)
        train_scores = handler.predict(training, output_margin=True)
        valid_scores = handler.predict(validation, output_margin=True)
        cls.metrics = {
            "features": cls.config["training"]["features"],
            "qa": build_training_qa(
                training,
                validation,
                train_scores,
                valid_scores,
                cls.config["training"]["features"],
                12,
            ),
            "score_efficiency_array": {
                "threshold": [0.2, 0.4, 0.6],
                "efficiency": [0.9, 0.7, 0.5],
                "actual_efficiency": [
                    float((valid_scores[validation.label == 1] >= cut).mean())
                    for cut in [0.2, 0.4, 0.6]
                ],
            },
            "feature_importance": dict(
                zip(
                    cls.config["training"]["features"],
                    handler.get_original_model().feature_importances_,
                )
            ),
            "feature_importance_basis": "XGBoost native gain; separate from SHAP PDF importance",
            "optuna": {"enabled": False},
        }
        data = [
            training,
            training.label.to_numpy(),
            validation,
            validation.label.to_numpy(),
        ]
        # Verify every plot invokes the actual installed HIPE4ML implementation.
        from hipe4ml import plot_utils

        methods = [
            "plot_distr",
            "plot_corr",
            "plot_output_train_test",
            "plot_roc_train_test",
            "plot_bdt_eff",
            "plot_feature_imp",
        ]
        cls.calls = {}
        state = np.random.get_state()
        # Exercise removal of the former name for a caller replacing QA in place.
        (cls.directory / "qa").mkdir()
        (cls.directory / "qa/bdt_train_test.pdf").write_bytes(b"obsolete")
        from contextlib import ExitStack

        with ExitStack() as stack:
            spies = {
                method: stack.enter_context(
                    patch.object(plot_utils, method, wraps=getattr(plot_utils, method))
                )
                for method in methods
            }
            cls.paths = render_training_qa(
                cls.directory, handler, data, cls.metrics, cls.config
            )
            cls.calls = {
                method: list(spy.call_args_list) for method, spy in spies.items()
            }
        restored = np.random.get_state()
        cls.rng_unchanged = (
            state[0] == restored[0]
            and np.array_equal(state[1], restored[1])
            and state[2:] == restored[2:]
        )
        cls.index = read_json(cls.directory / "qa/index.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_canonical_files_and_index_cover_every_requested_qa(self):
        expected = {
            "bdt_output",
            "bdt_test_score",
            "correlations",
            "efficiency_vs_score",
            "feature_distributions",
            "feature_importance",
            "roc_train_test",
        }
        self.assertEqual({entry["id"] for entry in self.index["plots"]}, expected)
        self.assertEqual(self.index["class_labels"], {"0": "Background", "1": "Signal"})
        self.assertEqual(self.index["score_domain"]["output_margin"], True)
        self.assertEqual(self.index["features"], ["DcaHe", "DcaPi"])
        self.assertEqual(self.index["histogram_bins"], 12)
        self.assertEqual(self.index["shap_sample_limit_per_class"], 6)
        files = {name for entry in self.index["plots"] for name in entry["files"]}
        self.assertEqual(len(files), 9)
        self.assertEqual(
            {path.name for path in self.paths}, files | {"training.root", "index.json"}
        )
        self.assertTrue((self.directory / "qa/bdt_output.pdf").is_file())
        self.assertFalse((self.directory / "qa/bdt_train_test.pdf").exists())
        self.assertFalse(list(self.directory.rglob("*.png")))
        for file in files:
            self.assertTrue(
                (self.directory / "qa" / file).read_bytes().startswith(b"%PDF-")
            )
        for artifact in TRAINING_QA_ARTIFACTS:
            self.assertTrue((self.directory / artifact).is_file(), artifact)

    def test_official_api_calls_preserve_raw_margin_and_label_semantics(self):
        self.assertEqual(len(self.calls["plot_distr"]), 2)
        for method in (
            "plot_corr",
            "plot_output_train_test",
            "plot_roc_train_test",
            "plot_bdt_eff",
            "plot_feature_imp",
        ):
            self.assertEqual(len(self.calls[method]), 1, method)
        output = self.calls["plot_output_train_test"][0]
        self.assertEqual(output.kwargs["output_margin"], True)
        self.assertEqual(output.kwargs["labels"], ["Background", "Signal"])
        self.assertEqual(output.kwargs["bins"], 12)
        distributions = self.calls["plot_distr"][0]
        self.assertEqual(distributions.kwargs["labels"], ["Signal", "Background"])
        self.assertEqual(distributions.args[1], ["DcaHe", "DcaPi", "mass_h3l"])
        self.assertEqual(self.index["qa_only_columns"], ["mass_h3l"])
        self.assertEqual(self.calls["plot_corr"][0].args[1], ["DcaHe", "DcaPi"])
        self.assertEqual(
            self.calls["plot_corr"][0].kwargs["labels"], ["Signal", "Background"]
        )
        importance = self.calls["plot_feature_imp"][0]
        self.assertEqual(importance.kwargs["n_sample"], 6)
        self.assertTrue(importance.kwargs["approximate"])
        self.assertTrue(self.rng_unchanged)
        plots = {entry["id"]: entry for entry in self.index["plots"]}
        for entry in plots.values():
            self.assertTrue(entry["purpose"])
            self.assertTrue(entry["sample_scope"])
            self.assertTrue(entry["provider"].startswith("hipe4ml.plot_utils."))
            self.assertTrue(entry["api_parameters"])
        self.assertIn("separate", plots["feature_importance"]["purpose"])
        self.assertIn(
            "actual model input features", plots["feature_distributions"]["purpose"]
        )

    def test_root_archive_contains_indexed_numerical_objects_without_canvases(self):
        import ROOT

        with ROOT.TFile.Open(
            str(self.directory / "qa/training.root"), "READ"
        ) as archive:
            self.assertFalse(archive.IsZombie())
            for entry in self.index["plots"]:
                for key in entry["root_objects"]:
                    self.assertTrue(archive.Get(key), f"{entry['id']}: {key}")
            for key in self.index["root_archive"]["metadata_objects"]:
                self.assertTrue(archive.Get(key), key)
            self.assertFalse(archive.GetDirectory("canvases"))
            stored = json.loads(str(archive.Get("metadata/qa_index").GetString()))
            self.assertEqual(stored, self.index)
            for split in ("train", "test"):
                roc = archive.Get("roc/" + split)
                self.assertTrue(roc.InheritsFrom("TGraph"))
                self.assertGreater(roc.GetN(), 1)
            self.assertTrue(archive.Get("correlations/signal").InheritsFrom("TH2"))
            self.assertTrue(archive.Get("efficiency/efficiency").InheritsFrom("TGraph"))


if __name__ == "__main__":
    unittest.main()
