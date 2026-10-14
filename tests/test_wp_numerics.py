"""Physics regressions for the spectrum working-point yield normalization."""

import copy
import unittest

import numpy as np

from pih3l.config import DEFAULTS
from pih3l.purity import mass_expectation
from pih3l.working_point import (
    _signal_and_background,
    _window_fit_statistics,
    expected_significance,
    fit_yield_pol0,
    resolve_working_point_config,
    _sigma_window_statistics,
    spectrum_qa_normalization,
)


class WorkingPointNumericalTests(unittest.TestCase):
    def test_failed_mass_fit_has_missing_window_statistics_without_crashing(self):
        config = resolve_working_point_config(copy.deepcopy(DEFAULTS))
        edges = np.linspace(*config["analysis"]["mass_range"], 51)
        result = _sigma_window_statistics(
            {"parameters": None, "status": "insufficient_entries"},
            np.zeros(50),
            edges,
            config,
            0.8,
        )
        for multiplier in (2, 3, 4):
            for name in (
                "raw_signal",
                "raw_signal_error",
                "background",
                "significance",
            ):
                key = (
                    f"raw_signal_{multiplier}sigma_error"
                    if name == "raw_signal_error"
                    else f"{name}_{multiplier}sigma"
                )
                self.assertTrue(np.isnan(result[key]))
            self.assertTrue(np.isnan(result[f"window_{multiplier}sigma"]).all())

    def test_dynamic_sigma_windows_preserve_background_scaling_and_shared_pol0(self):
        config = resolve_working_point_config(copy.deepcopy(DEFAULTS))
        config["purity"]["background"] = "constant"
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        rows = []
        for index, efficiency in enumerate((0.5, 0.7, 0.9)):
            parameters = dict(
                signal_yield=600 * efficiency,
                background_yield=400,
                mean=2.991,
                sigma=0.0015,
                slope=0.0,
            )
            counts = mass_expectation(list(parameters.values()), edges, "constant")
            fit = {
                "parameters": parameters,
                "parameter_covariance": np.diag([1e-4, 1e-4, 1e-10, 1e-10]),
                "status": "ok",
            }
            diagnostic = _sigma_window_statistics(
                fit, counts, edges, config, efficiency
            )
            np.testing.assert_allclose(
                [diagnostic[f"background_{n}sigma"] for n in (2, 3, 4)],
                np.array([2, 3, 4]) * 400 * 2 * 0.0015 / (edges[-1] - edges[0]),
                rtol=1e-12,
            )
            rows.append(
                {
                    "valid": True,
                    "threshold": 0.8 - index * 0.2,
                    "mc_signal_efficiency": efficiency,
                    "significance": 123.0,
                    **diagnostic,
                }
            )
        qa = spectrum_qa_normalization(rows, config["working_point"])
        self.assertEqual(qa["status"], "ok")
        from scipy.special import ndtr

        self.assertAlmostEqual(
            qa["yield_pol0"]["c0"], 600 * (ndtr(3) - ndtr(-3)), places=5
        )
        for row in rows:
            self.assertEqual(row["significance"], 123.0)
            self.assertGreaterEqual(
                row["significance_2sigma"], row["significance_3sigma"]
            )
            self.assertGreaterEqual(
                row["significance_3sigma"], row["significance_4sigma"]
            )
            for n in (2, 3, 4):
                self.assertAlmostEqual(
                    row[f"significance_{n}sigma"],
                    expected_significance(
                        row["mc_signal_efficiency"],
                        qa["yield_pol0"]["c0"],
                        row[f"background_{n}sigma"],
                    ),
                )

    def test_moving_sigma_window_covariance_recomputes_clipped_edges(self):
        config = resolve_working_point_config(copy.deepcopy(DEFAULTS))
        config["analysis"]["mass_range"] = [2.990, 2.998]
        config["purity"]["background"] = "constant"
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        parameters = dict(
            signal_yield=600.0,
            background_yield=400.0,
            mean=2.991,
            sigma=0.002,
            slope=0.0,
        )
        counts = mass_expectation(list(parameters.values()), edges, "constant")
        covariance = np.diag([1e-6, 1e-6, 1e-10, 1e-10])
        fit = {"parameters": parameters, "parameter_covariance": covariance}
        result = _sigma_window_statistics(fit, counts, edges, config, 0.8)
        np.testing.assert_allclose(result["window_3sigma"], [2.990, 2.997])
        point = np.array([0.6, 0.4, 2.991, 0.002])
        derivatives = []
        for index, step in enumerate([1e-5, 1e-5, 1e-7, 1e-7]):
            values = []
            for sign in (-1, 1):
                changed = point.copy()
                changed[index] += sign * step
                altered = {
                    "parameters": dict(
                        signal_yield=changed[0] * 1000,
                        background_yield=changed[1] * 1000,
                        mean=changed[2],
                        sigma=changed[3],
                        slope=0.0,
                    )
                }
                values.append(
                    _signal_and_background(altered, config, sigma_multiplier=3)[0]
                )
            derivatives.append((values[1] - values[0]) / (2 * step))
        gradient = np.array(derivatives)
        self.assertAlmostEqual(
            result["raw_signal_3sigma_error"]
            / np.sqrt(gradient @ covariance @ gradient),
            1.0,
            places=4,
        )

    def test_diagnostic_pol0_failure_does_not_replace_nominal_quantities(self):
        settings = copy.deepcopy(DEFAULTS["working_point"])
        row = {
            "valid": True,
            "threshold": 0.7,
            "mc_signal_efficiency": 0.8,
            "significance": 4.2,
            "raw_signal_3sigma_over_efficiency": 100.0,
            "raw_signal_3sigma_over_efficiency_error": 5.0,
            **{f"background_{n}sigma": float(n) for n in (2, 3, 4)},
            **{f"significance_{n}sigma": np.nan for n in (2, 3, 4)},
        }
        qa = spectrum_qa_normalization([row], settings)
        self.assertEqual(qa["status"], "insufficient_points")
        self.assertEqual(row["significance"], 4.2)
        self.assertTrue(row["valid"])
        self.assertFalse(row["spectrum_qa_valid"])
        self.assertIsNone(qa["diagnostic_maximum"])

    def test_signal_error_scales_normalized_yield_covariance_by_event_count(self):
        config = resolve_working_point_config(copy.deepcopy(DEFAULTS))
        config["analysis"]["signal_window"] = config["analysis"]["mass_range"][:]
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        parameters = {
            "signal_yield": 600.0,
            "background_yield": 400.0,
            "mean": 2.991,
            "sigma": 0.0015,
            "slope": 0.0,
        }
        counts = mass_expectation(list(parameters.values()), edges, "linear")
        covariance = np.diag([0.0004, 0.0009, 1e-10, 1e-10, 0.0001])
        fit = {
            "parameters": parameters,
            "parameter_covariance": covariance,
            "deviance": 0.0,
            "status": "ok",
        }
        result = _window_fit_statistics(fit, counts, edges, config)
        self.assertAlmostEqual(result["signal"], 600.0)
        self.assertAlmostEqual(result["signal_error"], 20.0, places=7)
        self.assertAlmostEqual(result["mean_error"], 1e-5)
        self.assertAlmostEqual(result["sigma_error"], 1e-5)
        self.assertAlmostEqual(result["chi2_ndf"], 0.0)

    def test_asymmetric_window_error_retains_correlated_yield_and_shape_terms(self):
        config = resolve_working_point_config(copy.deepcopy(DEFAULTS))
        config["analysis"]["signal_window"] = [2.989, 2.992]
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        parameters = {
            "signal_yield": 600.0,
            "background_yield": 400.0,
            "mean": 2.991,
            "sigma": 0.0015,
            "slope": 0.1,
        }
        counts = mass_expectation(list(parameters.values()), edges, "linear")
        scales = np.array([0.004, 0.004, 1e-5, 1e-5, 0.004])
        correlation = np.eye(5)
        correlation[0, 2] = correlation[2, 0] = 0.5
        covariance = correlation * np.outer(scales, scales)
        fit = {
            "parameters": parameters,
            "parameter_covariance": covariance,
            "deviance": 0.0,
            "status": "ok",
        }
        result = _window_fit_statistics(fit, counts, edges, config)
        total = counts.sum()
        point = np.array([0.6, 0.4, parameters["mean"], parameters["sigma"], 0.1])
        rng = np.random.default_rng(701)
        draws = rng.multivariate_normal(point, covariance, size=30000)
        yields = []
        for draw in draws:
            varied = {
                "parameters": dict(
                    zip(
                        parameters,
                        [draw[0] * total, draw[1] * total, draw[2], draw[3], draw[4]],
                    )
                )
            }
            yields.append(_signal_and_background(varied, config)[0])
        self.assertAlmostEqual(
            np.std(yields, ddof=1) / result["signal_error"], 1.0, delta=0.02
        )
        fraction = result["signal"] / parameters["signal_yield"]
        self.assertGreater(
            abs(result["signal_error"] - fraction * total * scales[0]), 0.1
        )

    def test_weighted_root_pol0_matches_known_constant_estimator(self):
        settings = copy.deepcopy(DEFAULTS["working_point"])
        values, errors = np.array([110.0, 90.0, 105.0]), np.array([2.0, 4.0, 3.0])
        rows = [
            dict(
                valid=True,
                mc_signal_efficiency=efficiency,
                threshold=threshold,
                signal_over_efficiency=value,
                signal_over_efficiency_error=error,
            )
            for efficiency, threshold, value, error in zip(
                [0.5, 0.7, 0.9], [0.8, 0.6, 0.3], values, errors
            )
        ]
        rows.append(
            dict(
                valid=False,
                mc_signal_efficiency=0.8,
                threshold=0.4,
                signal_over_efficiency=10000.0,
                signal_over_efficiency_error=1.0,
            )
        )
        fit = fit_yield_pol0(rows, settings)
        weights = 1 / errors**2
        self.assertEqual(fit["status"], "ok")
        self.assertEqual(fit["points"], 3)
        self.assertAlmostEqual(fit["c0"], np.average(values, weights=weights), places=6)
        self.assertAlmostEqual(fit["c0_error"], 1 / np.sqrt(weights.sum()), places=5)

    def test_tied_score_cuts_do_not_manufacture_independent_pol0_points(self):
        settings = copy.deepcopy(DEFAULTS["working_point"])
        rows = [
            dict(
                valid=True,
                mc_signal_efficiency=efficiency,
                threshold=0.3,
                signal_over_efficiency=100.0,
                signal_over_efficiency_error=2.0,
            )
            for efficiency in [0.5, 0.7, 0.9]
        ]
        fit = fit_yield_pol0(rows, settings)
        self.assertEqual(fit["status"], "insufficient_points")
        self.assertEqual(fit["distinct_thresholds"], 1)
        # Expanding the efficiency fit range cannot cure a single score cut.
        settings["yield_eff_range"] = [0.1, 1.0]
        self.assertEqual(
            fit_yield_pol0(rows, settings)["status"], "insufficient_points"
        )

    def test_nonpositive_pol0_cannot_be_selected(self):
        settings = copy.deepcopy(DEFAULTS["working_point"])
        rows = [
            dict(
                valid=True,
                mc_signal_efficiency=efficiency,
                threshold=threshold,
                signal_over_efficiency=-100.0,
                signal_over_efficiency_error=2.0,
            )
            for efficiency, threshold in zip([0.5, 0.7, 0.9], [0.8, 0.6, 0.3])
        ]
        self.assertEqual(fit_yield_pol0(rows, settings)["status"], "invalid_pol0_fit")

    def test_expected_significance_uses_uncorrected_c0_in_denominator(self):
        result = expected_significance(0.5, 100.0, 25.0)
        self.assertAlmostEqual(result, 50 / np.sqrt(125))
        self.assertNotAlmostEqual(result, 50 / np.sqrt(75))
        self.assertTrue(np.isnan(expected_significance(0.5, -1, 25)))
        self.assertTrue(np.isnan(expected_significance(0.5, 100, -1)))

    def test_missing_or_nonpositive_mass_covariance_stays_invalid(self):
        config = resolve_working_point_config(copy.deepcopy(DEFAULTS))
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        parameters = {
            "signal_yield": 600.0,
            "background_yield": 400.0,
            "mean": 2.991,
            "sigma": 0.0015,
            "slope": 0.0,
        }
        counts = mass_expectation(list(parameters.values()), edges, "linear")
        for covariance in ([], np.diag([0.1, 0.1, 0, 0.1, 0.1])):
            with self.subTest(covariance=covariance):
                fit = {"parameters": parameters, "parameter_covariance": covariance}
                self.assertTrue(
                    np.isnan(
                        _window_fit_statistics(fit, counts, edges, config)[
                            "signal_error"
                        ]
                    )
                )


if __name__ == "__main__":
    unittest.main()
