"""Dense training score maps and independent WP selection/storage contracts."""

from pih3l.validation_scores import write_validation_scores
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS, validate
from pih3l.errors import AnalysisError
from pih3l.io import read_json, write_json
from pih3l.training_settings import efficiency_grid
from pih3l.working_point import (
    _candidate_selection,
    resolve_working_point_config,
    scan_working_point,
    storage_envelope,
)


from test_model_routing import toy_handler


class WorkingPointConfigurationTests(unittest.TestCase):
    def test_explicit_wp_event_cuts_cannot_be_silently_ignored(self):
        config = copy.deepcopy(DEFAULTS)
        config["analysis"]["event_cuts"] = {"HypTrackOccupancy": {"max": 1000}}
        frame = pd.DataFrame({"mass_h3l": [2.99]})
        # Null compatibility fallback records missing columns at scan level.
        self.assertTrue(
            _candidate_selection(frame, resolve_working_point_config(config))[0]
        )
        config["working_point"]["event_cuts"] = {"HypTrackOccupancy": {"max": 1000}}
        with self.assertRaisesRegex(AnalysisError, "explicit WP event selection"):
            _candidate_selection(frame, resolve_working_point_config(config))

    def test_dense_grid_matches_spectrum_rounding_and_per_bin_validation(self):
        config = copy.deepcopy(DEFAULTS)
        expected = np.unique(np.round(np.linspace(0.29, 0.99, 701), 3))
        np.testing.assert_array_equal(efficiency_grid(config["training"]), expected)
        config["training"]["pt_bins"] = [
            {
                "id": "low",
                "pt_range": [2, 4],
                "efficiency_min": 0.4,
                "efficiency_max": 0.9,
                "npoints_for_effi": 25,
            }
        ]
        validate(config)
        config["training"]["pt_bins"][0]["efficiency_min"] = 0.95
        with self.assertRaisesRegex(AnalysisError, "efficiency bounds"):
            validate(config)

    def test_wp_cuts_mass_and_centrality_are_resolved_independently(self):
        config = copy.deepcopy(DEFAULTS)
        config["analysis"]["candidate_cuts"] = {"DcaHe": {"min": 0.9}}
        config["working_point"].update(
            candidate_cuts={"DcaHe": {"min": 0.3}},
            centrality_bins=None,
            mass_range=[2.95, 3.04],
            signal_window=[2.982, 3.001],
            purity={"mass_bins": 9},
        )
        validate(config)
        resolved = resolve_working_point_config(config)
        self.assertEqual(
            resolved["analysis"]["candidate_cuts"], {"DcaHe": {"min": 0.3}}
        )
        self.assertEqual(resolved["analysis"]["centrality_bins"], [])
        self.assertEqual(resolved["analysis"]["mass_range"], [2.95, 3.04])
        self.assertEqual(resolved["purity"]["mass_bins"], 9)
        self.assertEqual(config["analysis"]["mass_range"], [2.96, 3.025])

    def test_storage_floor_covers_nominal_and_all_declared_looser_variations(self):
        config = copy.deepcopy(DEFAULTS)
        config["working_point"]["storage"]["reserve_signal_efficiency"] = 0.75
        config["systematics"]["variations"] = [
            {"name": "loose", "overrides": {"analysis": {"bdt_threshold": 0.1}}}
        ]
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        result = storage_envelope(
            scores, 0.6, [{"delta": 0.25, "threshold": 0.35}], config
        )
        self.assertEqual(result["threshold"], 0.1)
        self.assertEqual(result["actual_mc_signal_efficiency"], 1)
        config["systematics"]["variations"] = []
        # Nominal 100% is preserved even with a tighter requested reserve.
        self.assertEqual(storage_envelope(scores, 0.0, [], config)["threshold"], 0.0)
        config["working_point"]["storage"]["enabled"] = False
        self.assertIsNone(storage_envelope(scores, 0.6, [], config)["threshold"])

    def test_independent_wp_sidebands_must_match_its_fit_geometry(self):
        config = copy.deepcopy(DEFAULTS)
        config["working_point"]["mass_range"] = [2.98, 3.005]
        # Inherited CF bands do not fit a narrower independent WP range.
        with self.assertRaisesRegex(AnalysisError, "WP sidebands must lie inside"):
            validate(config)
        config["working_point"]["sidebands"] = [[2.98, 2.984], [2.999, 3.005]]
        validate(config)
        resolved = resolve_working_point_config(config)
        self.assertEqual(
            resolved["analysis"]["sidebands"], [[2.98, 2.984], [2.999, 3.005]]
        )
        self.assertEqual(
            config["analysis"]["sidebands"], DEFAULTS["analysis"]["sidebands"]
        )
        for bands in (
            [[2.98, 2.986], [2.999, 3.005]],
            [[2.999, 3.005], [2.98, 2.984]],
            [[2.98, 2.984]],
        ):
            with self.subTest(sidebands=bands):
                invalid = copy.deepcopy(config)
                invalid["working_point"]["sidebands"] = bands
                with self.assertRaises(AnalysisError):
                    validate(invalid)

    def test_invalid_wp_configuration_and_output_column_lists_fail_early(self):
        for settings in (
            {"signal_efficiencies": [0.8, 0.6]},
            {"yield_eff_range": [0.5, 1.1]},
            {"min_pol0_points": 2},
            {"max_fit_chi2_ndf": 0},
            {"save_score_fit_frames": "false"},
            {"objective": "fit_significance"},
            {"centrality_bins": [[0, 10], [5, 20]]},
            {"mass_range": [2.995, 3.025]},
            {"purity": {"method": "fixed"}},
            {"storage": {"enabled": "true", "reserve_signal_efficiency": 0.99}},
        ):
            with self.subTest(settings=settings):
                config = copy.deepcopy(DEFAULTS)
                config["working_point"].update(settings)
                with self.assertRaises(AnalysisError):
                    validate(config)
        for columns in (["kstar", "kstar"], ["kstar", None], [""]):
            config = copy.deepcopy(DEFAULTS)
            config["io"]["scored_columns"] = columns
            with self.assertRaises(AnalysisError):
                validate(config)

    def test_wp_only_allows_dscb_and_pol1_or_pol2_background(self):
        for background in ("constant", "linear", "pol0", "auto"):
            config = copy.deepcopy(DEFAULTS)
            config["working_point"]["mass_fit"]["background"] = background
            with self.assertRaisesRegex(AnalysisError, "pol1 or pol2"):
                validate(config)
        for background in ("pol1", "pol2"):
            config = copy.deepcopy(DEFAULTS)
            config["working_point"]["mass_fit"]["background"] = background
            validate(config)
        config["working_point"]["mass_fit"]["signal_model"] = "gaussian"
        with self.assertRaisesRegex(AnalysisError, "must be dscb"):
            validate(config)

    def test_wp_uses_trained_dense_map_and_records_resolved_fit_geometry(self):
        config = copy.deepcopy(DEFAULTS)
        config["training"]["features"] = ["DcaHe"]
        config["analysis"]["candidate_cuts"] = {"DcaHe": {"min": 0.9}}
        config["analysis"]["bdt_threshold"] = 0.6
        config["analysis"]["event_cuts"] = {"HypTrackOccupancy": {"max": 1000}}
        config["working_point"].update(
            candidate_cuts={"DcaHe": {"min": 0.3}},
            mass_range=[2.95, 3.03],
            signal_window=[2.982, 3.0],
            sidebands=[[2.95, 2.975], [3.005, 3.03]],
            purity={"mass_bins": 8},
            yield_eff_range=[0.2, 0.8],
            max_fit_chi2_ndf=None,
        )
        dense = {"efficiency": [0.25, 0.5, 0.75], "threshold": [0.65, 0.5, 0.35]}
        frame = pd.DataFrame(
            {
                "DcaHe": [0.2, 0.4, 0.6, 0.8],
                "mass_h3l": [2.97, 2.98, 2.99, 3.01],
                "HypCentFT0C": [5, 25, 75, 95],
            }
        )

        def fake_fit(counts, edges, settings, **kwargs):
            self.assertEqual(settings["analysis"]["mass_range"], [2.95, 3.03])
            self.assertEqual(
                settings["analysis"]["sidebands"], [[2.95, 2.975], [3.005, 3.03]]
            )
            self.assertEqual(len(counts), 8)
            return {
                "status": "ok",
                "fit_pvalue": 0.9,
                "deviance": 1.0,
                "parameter_covariance": np.diag([0.01, 0.01, 1e-8, 1e-8, 0.01]),
                "parameters": {
                    "signal_yield": 2.0,
                    "background_yield": 1.0,
                    "mean": 2.991,
                    "sigma": 0.0015,
                    "slope": 0.0,
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            validation = pd.DataFrame(
                {"label": [1] * 4, "bdt_score": [0.2, 0.4, 0.6, 0.8]}
            )
            write_validation_scores(directory, validation)
            toy_handler().dump_model_handler(str(directory / "model_handler.pkl"))
            write_json(
                directory / "metrics.json",
                {
                    "features": ["DcaHe"],
                    "model_id": "global",
                    "score_output_margin": True,
                    "score_domain": "raw_margin",
                    "training_pt_range": None,
                    "threshold": 0.6,
                    "score_efficiency_array": dense,
                },
            )
            with patch(
                "pih3l.working_point.iter_parquet",
                return_value=iter(
                    [frame.iloc[:2].assign(HypTrackOccupancy=0), frame.iloc[2:]]
                ),
            ), patch("pih3l.working_point.fit_wp_mass", side_effect=fake_fit), patch(
                "pih3l.working_point._validation_thresholds",
                side_effect=lambda validation, efficiencies, **kwargs: (
                    {float(efficiency): 0.1 for efficiency in efficiencies},
                    {"numerical_retry": False},
                ),
            ):
                result = scan_working_point(
                    "unused", directory / "model_handler.pkl", directory / "wp", config
                )
        self.assertEqual(result["scan_grid_source"], "trained_score_efficiency_array")
        self.assertEqual(result["preselected_hyper_candidates"], 3)
        self.assertEqual(result["efficiency_grid_points"], 3)
        self.assertEqual(result["resolved_settings"]["mass_range"], [2.95, 3.03])
        self.assertEqual(
            result["resolved_settings"]["sidebands"], [[2.95, 2.975], [3.005, 3.03]]
        )
        self.assertEqual(result["unavailable_event_cuts"], ["HypTrackOccupancy"])
        self.assertEqual(result["missing_event_cut_batches"], {"HypTrackOccupancy": 1})
        self.assertLessEqual(result["storage_threshold"], result["threshold"])

    def test_failed_scan_preserves_every_efficiency_and_mass_fit_diagnostic(self):
        config = copy.deepcopy(DEFAULTS)
        config["training"]["features"] = ["DcaHe"]
        config["working_point"]["max_fit_chi2_ndf"] = None
        dense = {"efficiency": [0.5, 0.7, 0.9], "threshold": [0.3, 0.3, 0.3]}
        frame = pd.DataFrame(
            {"DcaHe": [0.4, 0.6, 0.8], "mass_h3l": [2.988, 2.991, 2.994]}
        )
        fit = {
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
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            validation = pd.DataFrame(
                {
                    "label": [1, 1, 1, 1, 0, 0],
                    "bdt_score": [0.2, 0.4, 0.6, 0.8, 0.0, 1.0],
                }
            )
            write_validation_scores(directory, validation)
            toy_handler().dump_model_handler(str(directory / "model_handler.pkl"))
            write_json(
                directory / "metrics.json",
                {
                    "features": ["DcaHe"],
                    "model_id": "global",
                    "score_output_margin": True,
                    "score_domain": "raw_margin",
                    "training_pt_range": None,
                    "threshold": 0.6,
                    "score_efficiency_array": dense,
                },
            )
            with patch(
                "pih3l.working_point.iter_parquet", return_value=iter([frame])
            ), patch(
                "pih3l.working_point.fit_wp_mass", return_value=fit
            ) as fit_mock, patch(
                "pih3l.working_point._validation_thresholds",
                return_value=({0.99: 0.1}, {"numerical_retry": False}),
            ), patch(
                "pih3l.working_point_qa.render_working_point_qa"
            ) as render:
                with self.assertRaisesRegex(AnalysisError, "WP pol0 yield fit failed"):
                    scan_working_point(
                        "unused",
                        directory / "model_handler.pkl",
                        directory / "wp",
                        config,
                    )
                self.assertEqual(fit_mock.call_count, 1)
                render.assert_called_once()
                self.assertEqual(render.call_args.args[2], {})
            summary = read_json(directory / "wp" / "summary.json")
            scan_fits = read_json(directory / "wp" / "scan_fits.json")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["yield_pol0"]["distinct_thresholds"], 1)
            self.assertEqual(len(scan_fits["points"]), 3)
            self.assertEqual(
                [point["mc_signal_efficiency"] for point in scan_fits["points"]],
                [0.5, 0.7, 0.9],
            )
            self.assertTrue(
                all(
                    len(point["counts"])
                    == resolve_working_point_config(config)["purity"]["mass_bins"]
                    for point in scan_fits["points"]
                )
            )
            self.assertFalse((directory / "wp" / "best_fit.json").exists())
