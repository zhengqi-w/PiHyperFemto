"""One HIPE4ML/XGBoost engine with strict configuration and tuning domains."""

import copy
import unittest

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS, validate
from pih3l.errors import AnalysisError
from pih3l.training_settings import (
    XGB_DEFAULTS,
    XGB_SEARCH_SPACE,
    create_model_handler,
    model_parameters,
    search_space,
)


class TrainingSettingsTests(unittest.TestCase):
    def configuration(self):
        return copy.deepcopy(DEFAULTS)

    def test_handler_factory_uses_seed_features_and_single_reference_parameters(self):
        from hipe4ml.model_handler import ModelHandler
        from xgboost import XGBClassifier

        training = self.configuration()["training"]
        before = copy.deepcopy(training)
        handler = create_model_handler(
            training,
            42,
            features=["DcaHe"],
            parameters={"n_estimators": 5, "max_depth": 2},
        )
        self.assertIsInstance(handler, ModelHandler)
        self.assertIsInstance(handler.get_original_model(), XGBClassifier)
        self.assertEqual(handler.get_training_columns(), ["DcaHe"])
        params = handler.get_model_params()
        self.assertEqual(params["random_state"], 42)
        self.assertEqual(params["n_estimators"], 5)
        self.assertEqual(params["max_depth"], 2)
        self.assertEqual(params["gamma"], XGB_DEFAULTS["gamma"])
        self.assertEqual(training, before)
        self.assertEqual(model_parameters(training), XGB_DEFAULTS)
        self.assertEqual(search_space(training), XGB_SEARCH_SPACE)

    def test_native_handler_probability_prediction_matches_xgboost(self):
        training = self.configuration()["training"]
        training["features"] = ["DcaHe"]
        rng = np.random.default_rng(99)
        frame = pd.DataFrame(
            {"DcaHe": np.r_[rng.normal(0, 0.4, 40), rng.normal(1, 0.4, 40)]}
        )
        labels = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
        handler = create_model_handler(
            training,
            42,
            parameters={
                "n_estimators": 5,
                "max_depth": 2,
                "n_jobs": 1,
                "gamma": 0,
                "min_child_weight": 1,
            },
        )
        handler.fit(frame, labels, sample_weight=np.ones(len(labels)))
        probabilities = handler.predict(frame, output_margin=False)
        self.assertEqual(probabilities.shape, (len(labels),))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        np.testing.assert_array_equal(
            probabilities, handler.get_original_model().predict_proba(frame)[:, 1]
        )
        self.assertGreater(
            np.max(np.abs(probabilities - handler.predict(frame, output_margin=True))),
            0.01,
        )

    def test_removed_backend_and_legacy_models_are_rejected(self):
        for name, value in (
            ("backend", "xgboost"),
            ("backend", "hist_gradient_boosting"),
            ("max_iter", 5),
            ("max_leaf_nodes", 7),
            ("min_samples_leaf", 3),
            ("learning_rate", 0.1),
            ("l2_regularization", 2),
        ):
            with self.subTest(name=name, value=value):
                config = self.configuration()
                config["training"][name] = value
                with self.assertRaisesRegex(AnalysisError, "Unknown training options"):
                    validate(config)
        for name in ("max_iter", "max_leaf_nodes", "max_bins", "early_stopping"):
            with self.subTest(parameter=name):
                config = self.configuration()
                config["training"]["model_params"][name] = 5
                with self.assertRaisesRegex(
                    AnalysisError, "Unknown XGBoost model parameter"
                ):
                    validate(config)

    def test_shap_qa_per_class_limit_requires_a_positive_integer(self):
        config = self.configuration()
        self.assertEqual(config["training"]["qa_shap_max_per_class"], 1000)
        config["training"]["qa_shap_max_per_class"] = 1
        validate(config)
        for value in (0, -1, 1.5, True, float("nan"), "1000"):
            with self.subTest(limit=value):
                invalid = self.configuration()
                invalid["training"]["qa_shap_max_per_class"] = value
                with self.assertRaisesRegex(AnalysisError, "qa_shap_max_per_class"):
                    validate(invalid)
        invalid = self.configuration()
        invalid["training"]["permutation_importance_repeats"] = 3
        with self.assertRaisesRegex(AnalysisError, "Unknown training options"):
            validate(invalid)

    def test_float_distribution_cannot_target_integer_model_parameters(self):
        for name in ["n_estimators", "max_depth", "max_bin", "n_jobs", "verbosity"]:
            with self.subTest(name=name):
                config = self.configuration()
                config["training"]["optuna"]["search_space"] = {
                    name: {"type": "float", "low": 2, "high": 3}
                }
                with self.assertRaisesRegex(AnalysisError, "integer model parameters"):
                    validate(config)

    def test_integer_and_categorical_domains_are_validated(self):
        config = self.configuration()
        config["training"]["optuna"]["search_space"] = {
            "n_estimators": {"type": "int", "low": 3, "high": 5},
            "max_depth": {"type": "categorical", "choices": [0, 2, 4]},
            "tree_method": {"type": "categorical", "choices": ["hist", "approx"]},
        }
        validate(config)
        for choices in ([2.0, 4], [False, 4], [None, 2], [2, 2], [-1, 2]):
            with self.subTest(choices=choices):
                invalid = copy.deepcopy(config)
                invalid["training"]["optuna"]["search_space"]["max_depth"][
                    "choices"
                ] = choices
                with self.assertRaises(AnalysisError):
                    validate(invalid)

    def test_categorical_only_settings_reject_numeric_distributions(self):
        for name in ("tree_method", "eval_metric", "booster", "objective"):
            with self.subTest(name=name):
                config = self.configuration()
                config["training"]["optuna"]["search_space"] = {
                    name: {"type": "int", "low": 0, "high": 1}
                }
                with self.assertRaisesRegex(AnalysisError, "categorical distribution"):
                    validate(config)

    def test_xgboost_unlimited_depth_requires_a_supported_tree_method(self):
        config = self.configuration()
        config["training"]["model_params"].update(tree_method="hist", max_depth=0)
        validate(config)
        config["training"]["model_params"]["tree_method"] = "exact"
        with self.assertRaisesRegex(AnalysisError, "positive max_depth"):
            validate(config)
        with self.assertRaisesRegex(AnalysisError, "positive max_depth"):
            create_model_handler(
                self.configuration()["training"],
                73129,
                parameters={"tree_method": "exact", "max_depth": 0},
            )

    def test_search_space_cross_constraints_fail_before_trials(self):
        config = self.configuration()
        config["training"]["optuna"]["search_space"] = {
            "tree_method": {"type": "categorical", "choices": ["hist", "exact"]},
            "max_depth": {"type": "int", "low": 0, "high": 4},
        }
        with self.assertRaisesRegex(AnalysisError, "combines exact"):
            validate(config)
        config["training"]["optuna"]["search_space"]["max_depth"]["low"] = 1
        validate(config)
        config["training"]["optuna"]["search_space"]["colsample_bynode"] = {
            "type": "float",
            "low": 0.5,
            "high": 1.0,
        }
        with self.assertRaisesRegex(AnalysisError, "unsupported colsample_bynode"):
            validate(config)

    def test_global_seed_and_canonical_gamma_cannot_be_shadowed(self):
        for parameters, message in (
            ({"random_state": 1}, "top-level seed"),
            ({"seed": 1}, "top-level seed"),
            ({"min_split_loss": 1}, "canonical gamma"),
            ({"learning_rate": float("nan")}, "finite"),
            ({"subsample": 0}, "positive"),
        ):
            with self.subTest(parameters=parameters):
                config = self.configuration()
                config["training"]["model_params"].update(parameters)
                with self.assertRaisesRegex(AnalysisError, message):
                    validate(config)


if __name__ == "__main__":
    unittest.main()
