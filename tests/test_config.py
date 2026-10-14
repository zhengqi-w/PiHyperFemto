"""Configuration safety checks prevent silently changing the analysis definition."""

import copy
import tempfile
import unittest
from pathlib import Path

from pih3l.config import DEFAULTS, load_config, merge, validate
from pih3l.errors import AnalysisError


class ConfigurationTests(unittest.TestCase):
    def test_ct_is_forbidden_as_training_feature(self):
        config = copy.deepcopy(DEFAULTS)
        config["training"]["features"] = ["DcaHe", "ct"]
        with self.assertRaisesRegex(AnalysisError, "lifetime sculpting"):
            validate(config)

    def setUp(self):
        self.config = copy.deepcopy(DEFAULTS)

    def test_default_config_is_valid(self):
        validate(self.config)

    def test_pt_bin_cuts_validate_combined_global_bounds(self):
        self.config["training"]["candidate_cuts"] = {"DcaHe": {"max": 1.0}}
        self.config["training"]["pt_bins"] = [
            {"id": "low", "pt_range": [2, 4], "candidate_cuts": {"DcaHe": {"min": 1.1}}}
        ]
        with self.assertRaises(AnalysisError):
            validate(self.config)
        self.config["training"]["pt_bins"][0]["candidate_cuts"]["DcaHe"]["min"] = 0.2
        validate(self.config)

    def test_mc_range_selection_rejects_unsafe_or_invalid_settings(self):
        for setting in (
            {"features": ["mass_h3l"]},
            {"features": []},
            {"quantiles": [-0.1, 0.99]},
            {"quantiles": [0.99, 0.1]},
            {"reference_fraction": 1.0},
            {"reference_fraction": False},
            {"margin_fraction": -1},
            {"min_reference_rows": 1},
        ):
            with self.subTest(setting=setting):
                config = copy.deepcopy(self.config)
                config["training"]["mc_range_selection"].update(setting)
                with self.assertRaises(AnalysisError):
                    validate(config)

    def test_nested_merge_does_not_mutate_inputs(self):
        override = {"analysis": {"normalization": [0.2, 0.4]}}
        result = merge(self.config, override)
        result["analysis"]["candidate_cuts"]["ct"] = [1, 35]
        self.assertEqual(self.config, DEFAULTS)
        self.assertEqual(result["analysis"]["normalization"], [0.2, 0.4])
        self.assertEqual(override, {"analysis": {"normalization": [0.2, 0.4]}})

    def test_unknown_options_raise(self):
        for section, key in [(None, "analysiss"), ("analysis", "normalisaton")]:
            with self.subTest(section=section):
                config = copy.deepcopy(self.config)
                (config if section is None else config[section])[key] = {}
                with self.assertRaises(AnalysisError):
                    validate(config)

    def test_overlapping_centrality_rejected(self):
        self.config["analysis"]["centrality_bins"] = [[0, 10], [9, 50]]
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_invalid_histogram_edges_rejected(self):
        for edges in [[0, 0.5, 0.5], [0, float("nan"), 0.6], [0.4, 0.2, 0.6]]:
            with self.subTest(edges=edges):
                self.config["analysis"]["kstar_edges"] = edges
                with self.assertRaises(AnalysisError):
                    validate(self.config)

    def test_normalization_must_follow_bin_edges(self):
        self.config["analysis"]["normalization"] = [0.301, 0.501]
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_signal_sideband_overlap_rejected(self):
        self.config["analysis"]["sidebands"] = [[2.98, 2.99], [3.007, 3.025]]
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_mass_pair_truth_and_identity_features_rejected(self):
        for feature in [
            "mass_h3l",
            "PairPiTrackPt",
            "kstar",
            "IsSignal",
            "candidate_uid",
            "ChargeChannel",
        ]:
            with self.subTest(feature=feature):
                self.config["training"]["features"] = [feature]
                with self.assertRaises(AnalysisError):
                    validate(self.config)

    def test_duplicate_features_rejected(self):
        self.config["training"]["features"] = ["ct", "ct"]
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_no_automatic_primary_fraction_correction(self):
        self.config["mc"]["apply_primary_correction"] = True
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_fixed_purity_requires_physical_value(self):
        self.config["purity"]["method"] = "fixed"
        for value in [0, -0.1, 1.1, None]:
            with self.subTest(value=value):
                self.config["purity"]["fixed"] = value
                with self.assertRaises(AnalysisError):
                    validate(self.config)

    def test_variations_cannot_change_preparation_support(self):
        self.config["systematics"]["variations"] = [
            {
                "name": "wider",
                "group": "mass",
                "overrides": {"analysis": {"mass_range": [2.9, 3.1]}},
            }
        ]
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_duplicate_variation_names_rejected(self):
        variation = {
            "name": "same",
            "group": "bdt",
            "overrides": {"analysis": {"bdt_threshold": 0.4}},
        }
        self.config["systematics"]["variations"] = [variation, copy.deepcopy(variation)]
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_nonfinite_or_nonphysical_scalar_options_rejected(self):
        cases = [
            ("analysis", "bdt_threshold", float("nan")),
            ("physics", "mass_pi", -0.1),
            ("physics", "mass_he3", float("nan")),
            ("physics", "he3_momentum_scale", 0),
            ("purity", "fixed_error", -0.1),
            ("purity", "sigma_bounds", [0.004, 0.001]),
            ("io", "chunk_rows", 0.5),
        ]
        for section, option, value in cases:
            with self.subTest(section=section, option=option, value=value):
                config = copy.deepcopy(DEFAULTS)
                config[section][option] = value
                with self.assertRaises(AnalysisError):
                    validate(config)

    def test_xgboost_parameters_must_be_nested_and_finite(self):
        config = copy.deepcopy(DEFAULTS)
        config["training"]["model_params"]["learning_rate"] = float("nan")
        with self.assertRaises(AnalysisError):
            validate(config)
        for key, value in (
            ("backend", "xgboost"),
            ("max_iter", 150),
            ("learning_rate", 0.05),
        ):
            with self.subTest(option=key):
                config = copy.deepcopy(DEFAULTS)
                config["training"][key] = value
                with self.assertRaisesRegex(AnalysisError, "Unknown training options"):
                    validate(config)

    def test_examples_use_one_hipe_model_and_preserve_dense_efficiency_grid(self):
        from pih3l.training_settings import XGB_DEFAULTS

        project = Path(__file__).resolve().parents[1]
        for name in ("testset.yaml", "production.example.yaml"):
            with self.subTest(config=name):
                config = load_config(project / "configs" / name)
                self.assertNotIn("backend", config["training"])
                self.assertEqual(config["training"]["model_params"], XGB_DEFAULTS)
                self.assertEqual(config["training"]["npoints_for_effi"], 701)
                self.assertEqual(config["training"]["qa_bins"], 100)
                self.assertEqual(config["training"]["qa_shap_max_per_class"], 1000)
                self.assertNotIn("ct", config["training"]["features"])

    def test_input_role_typo_rejected(self):
        self.config["inputs"]["femtto"] = {
            "patterns": ["Femto/*.root"],
            "required": True,
        }
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_nonfinite_cut_rejected(self):
        self.config["analysis"]["pair_cuts"] = {"PairPiTrackPt": {"min": float("nan")}}
        with self.assertRaises(AnalysisError):
            validate(self.config)

    def test_paths_are_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve()
            path = folder / "analysis.yaml"
            path.write_text("data_root: data\noutput: result\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config["data_root"], str(folder / "data"))
            self.assertEqual(config["output"], str(folder / "result"))


if __name__ == "__main__":
    unittest.main()
