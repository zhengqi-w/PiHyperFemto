"""Real small Optuna searches and strict configuration/model round trips."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hipe4ml.model_handler import ModelHandler
import numpy as np
from xgboost import XGBClassifier

from pih3l.config import DEFAULTS, validate
from pih3l.errors import AnalysisError
from pih3l.io import read_json, write_batch
from pih3l.models import load_model_registry, score_models, train_models
from pih3l.training import train, training_groups
from pih3l.training_optimization import optimize_parameters
from pih3l.training_settings import model_parameters
from synthetic_data import small_xgb_parameters
from test_model_routing import toy_candidates


class TrainingOptimizationTests(unittest.TestCase):
    def config(self):
        config = copy.deepcopy(DEFAULTS)
        config["training"].update(
            features=["DcaHe"],
            model_params=small_xgb_parameters(),
            qa_shap_max_per_class=40,
            efficiency_min=0.5,
            efficiency_max=0.9,
            npoints_for_effi=9,
        )
        return config

    def enable(self, config):
        settings = config["training"]
        settings["use_optuna"] = True
        settings["optuna"].update(
            n_trials=2,
            timeout=None,
            cv_folds=2,
            n_jobs=1,
            pruner="none",
            n_startup_trials=1,
            search_space={
                "n_estimators": {"type": "int", "low": 3, "high": 5},
                "learning_rate": {
                    "type": "float",
                    "low": 0.04,
                    "high": 0.08,
                    "log": True,
                },
            },
        )

    def test_fixed_xgboost_parameters_are_carried_by_real_handler(self):
        config = self.config()
        config["training"]["model_params"].update(
            n_estimators=3, max_depth=2, max_bin=64
        )
        validate(config)
        handler, metrics, _ = train(toy_candidates(1), toy_candidates(0), config)
        self.assertIsInstance(handler, ModelHandler)
        model = handler.get_original_model()
        self.assertIsInstance(model, XGBClassifier)
        self.assertEqual(model.n_estimators, 3)
        self.assertEqual(model.max_depth, 2)
        self.assertEqual(model.max_bin, 64)
        self.assertFalse(metrics["optuna"]["enabled"])
        self.assertEqual(metrics["model_params"]["n_estimators"], 3)

    def test_xgboost_parameters_and_predict_api(self):
        config = self.config()
        config["training"]["class_weight"] = None
        validate(config)
        handler, metrics, validation = train(
            toy_candidates(1), toy_candidates(0), config
        )
        self.assertIsInstance(handler, ModelHandler)
        model = handler.get_original_model()
        self.assertEqual(model.n_estimators, 40)
        self.assertEqual(model.get_params()["random_state"], config["seed"])
        self.assertEqual(metrics["model_handler"], "hipe4ml.model_handler.ModelHandler")
        self.assertTrue(metrics["score_output_margin"])
        self.assertTrue(np.isfinite(validation.bdt_score).all())
        np.testing.assert_allclose(
            handler.predict(validation, output_margin=True), validation.bdt_score
        )

    def test_dense_curve_is_independent_of_downstream_grids_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = self.config()
            config["output"] = str(base)
            config["training"].update(
                efficiency_min=0.6, efficiency_max=0.9, npoints_for_effi=4
            )
            validate(config)
            write_batch(base / "signal", 0, toy_candidates(1))
            write_batch(base / "background", 0, toy_candidates(0))
            train_models(
                base / "signal", base / "background", base / "training", config
            )
            original = read_json(base / "training/metrics.json")
            self.assertEqual(
                original["score_efficiency_array"]["efficiency"],
                [0.6, 0.7, 0.8, 0.9],
            )
            self.assertNotIn("differential_efficiency", original)
            curve = np.loadtxt(base / "training/score_efficiency_array.txt")
            np.testing.assert_allclose(
                curve[:, 0], original["score_efficiency_array"]["threshold"]
            )
            np.testing.assert_allclose(
                curve[:, 1], original["score_efficiency_array"]["efficiency"]
            )
            config["working_point"]["signal_efficiencies"] = [0.5, 0.8]
            config["analysis"]["centrality_bins"] = [[0, 5], [5, 30]]
            validate(config)
            result = train_models(
                base / "signal", base / "background", base / "second", config
            )
            self.assertEqual(result["models"][0]["status"], "verified_reuse")
            self.assertEqual(read_json(base / "second/metrics.json"), original)
            config["training"]["efficiency_max"] = 0.95
            result = train_models(
                base / "signal", base / "background", base / "third", config
            )
            self.assertEqual(result["models"][0]["status"], "trained")

    def test_search_excludes_outer_holdout_and_saves_trial_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            optimization_directory = Path(temporary) / "optimization"
            config = self.config()
            self.enable(config)
            validate(config)
            with patch(
                "pih3l.training_optimization.optimize_parameters",
                wraps=optimize_parameters,
            ) as optimizer, patch.object(
                ModelHandler, "fit", autospec=True, side_effect=ModelHandler.fit
            ) as handler_fit, patch.object(
                ModelHandler,
                "predict",
                autospec=True,
                side_effect=ModelHandler.predict,
            ) as handler_predict:
                handler, metrics, validation = train(
                    toy_candidates(1),
                    toy_candidates(0),
                    config,
                    optimization_directory=optimization_directory,
                )
            fitting = optimizer.call_args.args[0]
            held_out_groups = set(training_groups(validation))
            self.assertFalse(set(training_groups(fitting)) & held_out_groups)
            # The four inner CV fits and the final fit must all pass through
            # real ModelHandlers, with no outer held-out candidate in a fit.
            self.assertEqual(len(handler_fit.call_args_list), 5)
            for call in handler_fit.call_args_list:
                self.assertIsInstance(call.args[0], ModelHandler)
                # Continuous toy features identify these rows even if the
                # handler receives a projection without grouping metadata.
                self.assertFalse(set(call.args[1].DcaHe) & set(validation.DcaHe))
                self.assertIn("sample_weight", call.kwargs)
            self.assertTrue(handler_predict.call_args_list)
            self.assertTrue(
                all(
                    call.kwargs.get("output_margin") is True
                    for call in handler_predict.call_args_list
                )
            )
            summary = read_json(optimization_directory / "summary.json")
            self.assertEqual(summary["completed_trials"], 2)
            self.assertFalse(summary["split"]["final_held_out_used"])
            self.assertTrue(
                all(fold["group_overlap"] == 0 for fold in summary["split"]["folds"])
            )
            self.assertEqual(metrics["model_params"], summary["best_params"])
            self.assertAlmostEqual(
                summary["best_value"],
                max(np.mean(trial["fold_auc"]) for trial in summary["trials"]),
            )
            self.assertEqual(
                handler.get_model_params()["learning_rate"],
                summary["best_params"]["learning_rate"],
            )
            self.assertTrue((optimization_directory / "trials.csv").is_file())
            self.assertTrue((optimization_directory / "split.json").is_file())
            self.assertTrue((Path(temporary) / "qa/optuna_training.pdf").is_file())

    def test_optimization_artifacts_reuse_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = self.config()
            self.enable(config)
            config["output"] = str(base)
            write_batch(base / "signal", 0, toy_candidates(1))
            write_batch(base / "background", 0, toy_candidates(0))
            train_models(
                base / "signal", base / "background", base / "training", config
            )
            loaded = load_model_registry(base / "training")
            self.assertTrue((base / "training/model_handler.pkl").is_file())
            self.assertFalse((base / "training/model.joblib").exists())
            restored_handler = ModelHandler()
            restored_handler.load_model_handler(
                str(base / "training/model_handler.pkl")
            )
            self.assertIsInstance(restored_handler.get_original_model(), XGBClassifier)
            application = toy_candidates(1)
            original_scores = score_models(application, loaded).bdt_score
            self.assertTrue(np.isfinite(original_scores).all())
            np.testing.assert_array_equal(
                restored_handler.predict(application, output_margin=True),
                original_scores,
            )
            second = train_models(
                base / "signal", base / "background", base / "second", config
            )
            self.assertEqual(second["models"][0]["status"], "verified_reuse")
            self.assertTrue((base / "second/optimization/trials.csv").exists())
            reused_scores = score_models(
                application, load_model_registry(base / "second")
            ).bdt_score
            np.testing.assert_array_equal(reused_scores, original_scores)
            (base / "second/optimization/trials.csv").write_text("corrupt")
            with self.assertRaisesRegex(AnalysisError, "checksum"):
                load_model_registry(base / "second")

    def test_invalid_options_fail_before_training(self):
        cases = [
            {"backend": "unknown"},
            {"backend": "xgboost"},
            {"max_iter": 5},
            {"min_samples_leaf": 3},
            {"model_params": {"min_split_loss": 1.5}},
            {"model_params": {"random_state": 1}},
            {"model_params": {"early_stopping": True}},
            {"class_weight": "unexpected"},
            {"use_optuna": "false"},
            {"qa_signal_efficiencies": [0.8, 0.6]},
            {"qa_signal_efficiencies": [0, 0.8]},
            {"qa_centrality_bins": [[0, 10], [5, 20]]},
            {"optuna": {"metric": "significance"}},
            {"optuna": {"cv_folds": 1}},
            {
                "optuna": {
                    "search_space": {
                        "learning_rate": {
                            "type": "float",
                            "low": 0,
                            "high": 0.1,
                            "log": True,
                        }
                    }
                }
            },
            {
                "optuna": {
                    "search_space": {
                        "n_estimators": {"type": "float", "low": 3, "high": 5}
                    }
                }
            },
            {"optuna": {"search_space": {"ct": {"type": "int", "low": 1, "high": 3}}}},
        ]
        from pih3l.config import merge

        for options in cases:
            with self.subTest(options=options):
                config = self.config()
                config["training"] = merge(config["training"], options)
                with self.assertRaises(AnalysisError):
                    validate(config)

    def test_per_bin_search_and_parameters_only_retrain_changed_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = self.config()
            config["output"] = str(base)
            config["training"]["pt_bins"] = [
                {"id": "low", "pt_range": [2, 4]},
                {
                    "id": "high",
                    "pt_range": [4, 6],
                    "model_params": {"max_depth": 1},
                    "use_optuna": True,
                    "optuna": {
                        "n_trials": 2,
                        "cv_folds": 2,
                        "pruner": "none",
                        "search_space": {
                            "n_estimators": {"type": "int", "low": 2, "high": 4}
                        },
                    },
                },
            ]
            validate(config)
            # Each pT scope needs enough held-out rows for the requested HIPE
            # efficiency range; a 15-row split can leave 0.9 unreachable.
            write_batch(base / "signal", 0, toy_candidates(1, count=240))
            write_batch(base / "background", 0, toy_candidates(0, count=240))
            train_models(
                base / "signal", base / "background", base / "training", config
            )
            self.assertFalse((base / "training/models/low/optimization").exists())
            high = read_json(base / "training/models/high/metrics.json")
            self.assertEqual(high["model_params"]["max_depth"], 1)
            self.assertTrue(high["optuna"]["enabled"])
            self.assertEqual(set(high["optuna"]["search_space"]), {"n_estimators"})
            # Unconsumed global search settings cannot invalidate the fixed bin.
            config["training"]["optuna"]["n_trials"] = 3
            config["training"]["pt_bins"][1]["model_params"]["max_depth"] = 2
            result = train_models(
                base / "signal", base / "background", base / "second", config
            )
            self.assertEqual(
                [model["status"] for model in result["models"]],
                ["verified_reuse", "trained"],
            )

    def test_model_reuse_rejects_changes_to_candidate_reader_or_arrow(self):
        import importlib.metadata
        from pih3l.io import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = self.config()
            config["output"] = str(base)
            write_batch(base / "signal", 0, toy_candidates(1))
            write_batch(base / "background", 0, toy_candidates(0))
            train_models(
                base / "signal", base / "background", base / "training", config
            )
            for filename in ("io.py", "schema.py"):
                with self.subTest(source=filename):

                    def changed_reader(path):
                        return (
                            "changed-reader"
                            if Path(path).name == filename
                            else sha256_file(path)
                        )

                    with patch("pih3l.models.sha256_file", side_effect=changed_reader):
                        result = train_models(
                            base / "signal",
                            base / "background",
                            base / filename,
                            config,
                        )
                    self.assertEqual(result["models"][0]["status"], "trained")
            version = importlib.metadata.version

            def changed_arrow(name):
                return "changed-pyarrow" if name == "pyarrow" else version(name)

            with patch(
                "pih3l.models.importlib.metadata.version", side_effect=changed_arrow
            ):
                result = train_models(
                    base / "signal", base / "background", base / "arrow", config
                )
            self.assertEqual(result["models"][0]["status"], "trained")

    def test_hipe_optuna_full_synthetic_pdf_and_physics_root_report(self):
        from pih3l.config import load_config
        from pih3l.pipeline import Pipeline
        from synthetic_data import create_test_inputs, write_test_config
        import ROOT

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, output = base / "input", base / "output"
            create_test_inputs(source)
            path = base / "analysis.yaml"
            write_test_config(path, source, output)
            config = load_config(path)
            config["training"].update(
                model_params={
                    **small_xgb_parameters(),
                    "n_estimators": 4,
                    "max_depth": 2,
                },
                qa_shap_max_per_class=40,
            )
            self.enable(config)
            validate(config)
            Pipeline(config).execute("report")
            self.assertTrue((output / "training/qa/optuna_training.pdf").is_file())
            self.assertFalse((output / "report/optuna_training.pdf").exists())
            self.assertFalse(list((output / "report").glob("*.png")))
            with ROOT.TFile.Open(
                str(output / "training/qa/training.root"), "READ"
            ) as archive:
                for name in ("trial_auc", "best_auc"):
                    graph = archive.Get(f"training/optuna/{name}")
                    self.assertTrue(graph and graph.InheritsFrom("TGraph"))
                    self.assertEqual(graph.GetN(), 2)
                settings = archive.Get("metadata/training")
                self.assertTrue(settings and settings.InheritsFrom("TObjString"))
                metadata = json.loads(str(settings.GetString()))
                self.assertEqual(
                    metadata["model_handler"], "hipe4ml.model_handler.ModelHandler"
                )
                self.assertTrue(metadata["score_output_margin"])
                self.assertEqual(
                    metadata["model_params"],
                    read_json(output / "training/metrics.json")["model_params"],
                )
                self.assertFalse(archive.Get("canvases/optuna_training"))
            wp = read_json(output / "working_point/summary.json")
            self.assertEqual(
                wp["threshold_source"], "hyper_data_pol0_expected_significance"
            )
            self.assertEqual(wp["yield_pol0"]["status"], "ok")
            self.assertGreaterEqual(wp["yield_pol0"]["distinct_thresholds"], 3)
            trained = read_json(output / "training/metrics.json")
            self.assertFalse(trained["optuna"]["split"]["final_held_out_used"])

    def test_unavailable_optuna_is_explicit_and_disabled_does_not_import_it(self):
        config = self.config()
        import builtins

        original_import = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("Optuna unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            parameters, summary = optimize_parameters(
                None, config, model_parameters(config["training"]), ["DcaHe"]
            )
            self.assertFalse(summary["enabled"])
            config["training"]["use_optuna"] = True
            with self.assertRaisesRegex(AnalysisError, "requires Optuna"):
                optimize_parameters(None, config, parameters, ["DcaHe"])


if __name__ == "__main__":
    unittest.main()
