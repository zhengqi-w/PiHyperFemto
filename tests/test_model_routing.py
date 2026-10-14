"""pT scope routing, independently reusable models and bounded WP accumulation."""

from pih3l.validation_scores import write_validation_scores
import copy
from pathlib import Path
from unittest.mock import patch

from hipe4ml.model_handler import ModelHandler
import numpy as np
import pandas as pd
import unittest
import tempfile

from pih3l.config import DEFAULTS, validate
from pih3l.errors import AnalysisError
from pih3l.io import read_json, write_batch, write_json
from pih3l.models import (
    load_model_bundle,
    load_model_registry,
    model_specs,
    score_models,
    train_models,
)
from pih3l.selection import select_pairs, threshold_values
from pih3l.working_point import (
    resolve_wp_variation,
    scan_working_point,
    resolve_working_point_config,
)


class FeatureScore:
    """Deterministic toy classifier, so expected histograms are exact."""

    def predict_proba(self, values):
        scores = np.asarray(values)[:, 0]
        return np.column_stack((1 - scores, scores))

    def decision_function(self, values):
        return np.asarray(values)[:, 0]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        return self


def toy_handler(classifier=None, features=None):
    """Attach a deterministic test classifier to the official HIPE container."""
    handler = ModelHandler()
    handler.model = classifier if classifier is not None else FeatureScore()
    handler.training_columns = features or ["DcaHe"]
    handler.model_params = handler.model.get_params()
    handler.model_string = "sklearn"
    handler._n_classes = 2
    return handler


def base_config():
    config = copy.deepcopy(DEFAULTS)
    config["training"].update(
        features=["DcaHe"],
        model_params={"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
    )
    return config


def toy_candidates(label, count=120):
    rng = np.random.default_rng(120 + label)
    return pd.DataFrame(
        {
            "DcaHe": rng.normal(0.8 if label else 0.2, 0.05, count),
            "mass_h3l": rng.normal(2.991 if label else 3.03, 0.001, count),
            "ct": np.ones(count),
            "pt_h3l": np.tile([2.5, 4.5], count // 2),
            "HypCentFT0C": np.ones(count),
            "XPrimVtx": np.arange(count),
            "YPrimVtx": np.zeros(count),
            "ZPrimVtx": np.zeros(count),
            "_source_file": [f"{label}.root"] * count,
            "_source_tree": ["DF/tree"] * count,
        }
    )


class ModelRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_scope_edges_threshold_routing_and_nonfinite_features(self):
        frame = pd.DataFrame(
            {"pt_h3l": [1.9, 2, 4, 6, 3], "DcaHe": [0.9, 0.4, 0.7, 0.9, np.nan]}
        )
        models = [
            {
                "id": "low",
                "pt_range": [2, 4],
                "bundle": {"model": toy_handler(), "features": ["DcaHe"]},
            },
            {
                "id": "high",
                "pt_range": [4, 6],
                "bundle": {"model": toy_handler(), "features": ["DcaHe"]},
            },
        ]
        scored = score_models(frame, {"models": models})
        assert scored.bdt_model_id.tolist() == ["", "low", "high", "", "low"]
        assert scored.bdt_in_scope.tolist() == [False, True, True, False, True]
        np.testing.assert_allclose(
            scored.bdt_score, [np.nan, 0.4, 0.7, np.nan, np.nan], equal_nan=True
        )
        analysis = copy.deepcopy(DEFAULTS["analysis"])
        analysis["close_pair_reject_bits"] = 0
        mask, flow = select_pairs(scored, analysis, {"low": 0.3, "high": 0.8})
        assert mask.tolist() == [False, True, False, False, False]
        assert flow["bdt"] == 1
        with self.assertRaisesRegex(AnalysisError, "Missing working point"):
            threshold_values(scored, {"low": 0.3})
        # An explicit scalar cut cannot resurrect candidates outside a model scope.
        scored.loc[0, "bdt_score"] = 1
        assert not select_pairs(scored, analysis, 0)[0][0]

    def test_official_handler_roundtrip_checks_feature_order_and_rejects_joblib(self):
        directory = self.tmp_path
        path = directory / "model_handler.pkl"
        toy_handler().dump_model_handler(str(path))
        metadata = {
            "features": ["DcaHe"],
            "model_id": "shared",
            "score_output_margin": True,
            "score_domain": "raw_margin",
            "training_pt_range": [2, 8],
        }
        write_json(directory / "metrics.json", metadata)
        bundle = load_model_bundle(path)
        self.assertIsInstance(bundle["model"], ModelHandler)
        self.assertEqual(bundle["pt_range"], [2, 8])
        np.testing.assert_allclose(
            bundle["model"].predict(
                pd.DataFrame({"DcaHe": [0.2, 0.8]}), output_margin=False
            ),
            [0.2, 0.8],
        )
        metadata["features"] = ["DcaPi"]
        write_json(directory / "metrics.json", metadata)
        with self.assertRaisesRegex(AnalysisError, "feature order"):
            load_model_bundle(path)
        with self.assertRaisesRegex(AnalysisError, "legacy.*retraining"):
            load_model_bundle(directory / "model.joblib")
        path.unlink()
        (directory / "model.joblib").write_bytes(b"legacy artifact")
        with self.assertRaisesRegex(AnalysisError, "Legacy.*retrain"):
            load_model_registry(directory)

    def test_configuration_rejects_overlap_forbidden_features_and_ambiguous_delta(self):
        config = base_config()
        config["training"]["pt_bins"] = [[2, 4], [3, 5]]
        with self.assertRaisesRegex(AnalysisError, "disjoint"):
            validate(config)
        config["training"]["pt_bins"] = [
            {"id": "low", "pt_range": [2, 4], "features": ["ct"]}
        ]
        with self.assertRaisesRegex(AnalysisError, "lifetime"):
            validate(config)
        config["training"]["pt_bins"] = []
        config["systematics"]["variations"] = [
            {
                "name": "loose",
                "wp_efficiency_delta": 0.05,
                "overrides": {"analysis": {"bdt_threshold": 0.3}},
            }
        ]
        with self.assertRaisesRegex(AnalysisError, "cannot also"):
            validate(config)

    def test_official_handler_source_changes_invalidate_models_stages_and_score_cache(
        self,
    ):
        from pih3l.execution import stage_environment
        from pih3l.models import _model_signature
        from pih3l.score_cache import score_key

        config = base_config()
        spec = model_specs(config)[0]
        environment = {
            "python": "fixed",
            "optional_training": {"hipe4ml": "fixed", "xgboost": "fixed"},
        }
        scopes = {}
        signatures = []
        for fingerprint in ("handler-source-v1", "handler-source-v2"):
            with patch(
                "pih3l.execution.hipe4ml_model_source", return_value=fingerprint
            ):
                signatures.append(
                    _model_signature(
                        spec, config, {"signal": "fixed", "background": "fixed"}
                    )
                )
                scopes[fingerprint] = {
                    stage: stage_environment(environment, config, stage)
                    for stage in ("training", "working_point", "scored")
                }
        self.assertNotEqual(*signatures)
        for stage in ("training", "working_point", "scored"):
            self.assertNotEqual(
                scopes["handler-source-v1"][stage], scopes["handler-source-v2"][stage]
            )
        score_specs = [
            {
                "inference_checksum": "same-handler-bytes",
                "pt_range": None,
                "inference_code": {"models": "fixed"},
                "inference_environment": scopes[value]["scored"],
            }
            for value in scopes
        ]
        self.assertNotEqual(
            *(score_key({"source": "fixed"}, item) for item in score_specs)
        )

    def test_models_reuse_other_bins_and_detect_corruption(self):
        tmp_path = self.tmp_path
        config = base_config()
        config["output"] = str(tmp_path)
        config["training"]["pt_bins"] = [
            {"id": "low", "pt_range": [2, 4]},
            {"id": "high", "pt_range": [4, 6]},
        ]
        signal, background = tmp_path / "signal", tmp_path / "background"
        write_batch(signal, 0, toy_candidates(1))
        write_batch(background, 0, toy_candidates(0))
        train_models(signal, background, tmp_path / "training", config)
        self.assertFalse(list((tmp_path / "training").rglob("*.joblib")))
        for label in ("low", "high"):
            directory = tmp_path / "training/models" / label
            self.assertTrue((directory / "model_handler.pkl").is_file())
            self.assertEqual(
                read_json(directory / "model_manifest.json")["model_format"],
                "hipe4ml_model_handler_pickle_v1",
            )
        assert all(
            item["status"] == "trained"
            for item in read_json(tmp_path / "training/registry.json")["models"]
        )
        changed = copy.deepcopy(config)
        changed["training"]["pt_bins"][0]["candidate_cuts"] = {"DcaHe": {"min": 0}}
        second = train_models(signal, background, tmp_path / "second", changed)
        assert [item["status"] for item in second["models"]] == [
            "trained",
            "verified_reuse",
        ]
        loaded = load_model_registry(tmp_path / "second")
        assert len(loaded["models"]) == 2
        with (tmp_path / "second/models/high/model_handler.pkl").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(AnalysisError, "checksum"):
            load_model_registry(tmp_path / "second")

    def test_streamed_wp_matches_exact_histograms_and_fixed_denominator(self):
        tmp_path = self.tmp_path
        config = base_config()
        config["working_point"]["signal_efficiencies"] = [0.5, 0.65, 0.8]
        config["analysis"]["bdt_threshold"] = 0.6
        config["systematics"]["variations"] = [
            {"name": "loose", "wp_efficiency_delta": 0.25, "overrides": {}}
        ]
        training = tmp_path / "training"
        training.mkdir()
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        validation = pd.DataFrame(
            {"label": [1] * 4 + [0, 0], "bdt_score": np.r_[scores, 0.0, 1.0]}
        )
        write_validation_scores(training, validation)
        toy_handler().dump_model_handler(str(training / "model_handler.pkl"))
        write_json(
            training / "metrics.json",
            {
                "threshold": 0.3,
                "features": ["DcaHe"],
                "model_id": "global",
                "score_output_margin": True,
                "score_domain": "raw_margin",
                "training_pt_range": None,
            },
        )
        frame = pd.DataFrame(
            {
                "DcaHe": [0.2, 0.4, 0.6, 0.8, np.nan],
                "mass_h3l": [2.970, 2.981, 2.991, 3.010, 2.99],
            }
        )
        captured = []

        def fit(counts, edges, config, **kwargs):
            captured.append(np.array(counts))
            return {
                "status": "ok",
                "fit_pvalue": 0.9,
                "parameter_covariance": np.diag([0.01, 0.01, 1e-8, 1e-8, 0.01]),
                "parameters": {
                    "signal_yield": 2.0,
                    "background_yield": 1.0,
                    "mean": 2.991,
                    "sigma": 0.0015,
                    "slope": 0.0,
                },
            }

        # No concatenate is needed for streaming: even independently indexed batches
        # must yield identical counts, and only the retained MC validation is loaded.
        with patch(
            "pih3l.working_point.iter_parquet",
            return_value=iter([frame.iloc[:2], frame.iloc[2:]]),
        ), patch("pih3l.working_point.fit_wp_mass", side_effect=fit):
            summary = scan_working_point(
                "unused", training / "model_handler.pkl", tmp_path / "wp", config
            )
        edges = np.linspace(
            *resolve_working_point_config(config)["analysis"]["mass_range"],
            resolve_working_point_config(config)["purity"]["mass_bins"] + 1,
        )
        for item, actual in zip(summary["scan_points"], captured):
            expected = np.histogram(
                frame.loc[frame.DcaHe >= item["threshold"], "mass_h3l"], edges
            )[0]
            np.testing.assert_array_equal(actual, expected)
        assert summary["preselected_hyper_candidates"] == 4
        assert summary["validation_signal_rows"] == 4
        assert summary["nominal_mc_signal_efficiency"] == 0.5
        threshold = resolve_wp_variation(summary, 0.25)
        from pih3l.efficiency import thresholds_from_efficiencies

        expected_cut = thresholds_from_efficiencies(
            validation.label, validation.bdt_score, [0.75]
        )[0]
        self.assertAlmostEqual(threshold, expected_cut)
        assert summary["efficiency_variations"][0]["actual_signal_efficiency"] == 0.75
        with self.assertRaisesRegex(AnalysisError, "not evaluated"):
            resolve_wp_variation(summary, 0.1)

    def test_restricted_shared_model_is_explicit_and_no_silent_fallback(self):
        config = base_config()
        config["training"]["pt_range"] = [2, 8]
        assert model_specs(config)[0]["pt_range"] == [2, 8]
        frame = pd.DataFrame({"pt_h3l": [1, 2, 8], "DcaHe": [0.5] * 3})
        registry = {
            "models": [
                {
                    "id": "global",
                    "pt_range": [2, 8],
                    "bundle": {"model": toy_handler(), "features": ["DcaHe"]},
                }
            ]
        }
        np.testing.assert_array_equal(
            score_models(frame, registry).bdt_in_scope, [False, True, False]
        )

    def test_two_pt_models_synthetic_pipeline_and_root_report(self):
        tmp_path = self.tmp_path
        """Exercise routing and report integration on toys, never scarce real data."""
        import uproot
        from pih3l.config import load_config
        from pih3l.physics import derive
        from pih3l.pipeline import Pipeline
        from pih3l.schema import adapt
        from synthetic_data import create_test_inputs, daughters, write_test_config

        source, output = tmp_path / "input", tmp_path / "output"
        create_test_inputs(source)
        for folder, role, tree_name in (
            ("Hyper", "background", "O2hypcands"),
            ("MCHyper", "signal", "O2mchypcands"),
            ("Femto", "femto", "O2pihypfemto"),
        ):
            path = source / folder / "AO2D.root"
            with uproot.open(path) as archive:
                tree = archive[f"DF_1/{tree_name}"]
                arrays = tree.arrays(library="np")
                columns = {name: arrays[name] for name in tree.keys()}
            frame = derive(
                adapt(pd.DataFrame(columns), role),
                DEFAULTS["physics"],
                pairs=role == "femto",
            )
            momenta = daughters(
                frame.mass_h3l.to_numpy(),
                pt=np.where(np.arange(len(frame)) % 2, 1.2, 2.2),
            )
            prefix = "fHyp" if role == "femto" else "f"
            columns.update({prefix + name: values for name, values in momenta.items()})
            with uproot.recreate(path) as archive:
                archive[f"DF_1/{tree_name}"] = columns
        config_path = tmp_path / "analysis.yaml"
        write_test_config(config_path, source, output)
        config = load_config(config_path)
        config["training"]["pt_bins"] = [
            {"id": "low", "pt_range": [1, 2]},
            {"id": "high", "pt_range": [2, 3]},
        ]
        # A fixed WP is useful here: this test checks routing/output, not physics WP.
        config["analysis"]["bdt_threshold"] = 0.5
        config["systematics"]["variations"] = [
            {"name": "tight_wp", "wp_efficiency_delta": -0.05, "overrides": {}}
        ]
        validate(config)
        Pipeline(config).execute("report")
        summary = read_json(output / "working_point/summary.json")
        assert set(summary["threshold"]) == {"low", "high"}
        self.assertEqual(summary["threshold"], {"low": 0.5, "high": 0.5})
        assert set(resolve_wp_variation(summary, -0.05)) == {"low", "high"}
        for model_id in ("low", "high"):
            wp = read_json(output / f"working_point/models/{model_id}/summary.json")
            self.assertEqual(wp["yield_pol0"]["status"], "ok")
            self.assertGreaterEqual(wp["yield_pol0"]["distinct_thresholds"], 3)
            self.assertGreater(wp["best_scan_point"]["significance"], 0)
        import ROOT

        with ROOT.TFile.Open(str(output / "report/analysis.root"), "READ") as archive:
            assert not archive.IsZombie()
            for model_id in ("low", "high"):
                histogram = archive.Get(f"training/models/{model_id}/score_signal")
                self.assertTrue(histogram and histogram.InheritsFrom("TH1"))
                self.assertGreater(histogram.GetEntries(), 0)
                graph = archive.Get(
                    f"working_point/models/{model_id}/significance_vs_score"
                )
                self.assertTrue(graph and graph.InheritsFrom("TGraph"))
                self.assertGreater(graph.GetN(), 0)
        for model_id in ("low", "high"):
            assert (output / f"training/models/{model_id}/qa/bdt_output.pdf").is_file()
            assert (
                output / f"working_point/models/{model_id}/qa/sig_vs_score.pdf"
            ).is_file()
            assert (
                output / f"working_point/models/{model_id}/qa/working_point.root"
            ).is_file()
            assert (output / f"report/working_point__{model_id}.pdf").is_file()
        assert not list((output / "report").glob("*.png"))
