"""Independent statistical closure tests, including normalization covariance."""

import copy
import unittest

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pih3l.config import DEFAULTS
from pih3l.correlation import accumulate, empty_histograms, normalized_correlation
from pih3l.purity import correct_correlation, fit_purity, mass_expectation
from pih3l.systematics import combine_variations


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.arange(5, dtype=float) / 10
        self.normalization = [0.2, 0.4]

    def test_constant_se_me_ratio_closes_to_unity(self):
        se = np.array([20, 30, 40, 50.0])
        me = 5 * se
        result = normalized_correlation(se, me, se, me, self.edges, self.normalization)
        np.testing.assert_allclose(result["values"], 1)
        self.assertAlmostEqual(result["alpha"], 5)
        self.assertEqual(result["status"], "ok")

    def test_covariance_matches_independent_finite_difference_jacobian(self):
        se = np.array([20, 32, 43, 56.0])
        me = np.array([120, 170, 230, 270.0])
        se2, me2 = se * 1.25, me * 0.8
        inputs = np.r_[se, me]

        def estimator(values):
            first, second = values[:4], values[4:]
            return second[2:].sum() / first[2:].sum() * first / second

        jacobian = np.empty((4, 8))
        for index in range(8):
            step = np.eye(8)[index] * 1e-3
            jacobian[:, index] = (
                estimator(inputs + step) - estimator(inputs - step)
            ) / 2e-3
        reference = (jacobian * np.r_[se2, me2]) @ jacobian.T
        result = normalized_correlation(
            se, me, se2, me2, self.edges, self.normalization
        )
        np.testing.assert_allclose(
            result["covariance"], reference, rtol=1e-7, atol=1e-10
        )
        np.testing.assert_allclose(
            result["covariance"], result["covariance"].T, atol=1e-14
        )
        self.assertGreaterEqual(np.linalg.eigvalsh(result["covariance"]).min(), -1e-12)
        self.assertGreater(abs(result["covariance"][0, 1]), 0)

    def test_single_normalization_bin_has_zero_self_variance(self):
        result = normalized_correlation(
            [10, 20], [50, 80], [10, 20], [50, 80], [0, 0.2, 0.4], [0.2, 0.4]
        )
        self.assertAlmostEqual(result["values"][1], 1)
        self.assertAlmostEqual(result["covariance"][1, 1], 0, places=12)

    def test_empty_normalization_is_unavailable(self):
        result = normalized_correlation(
            [10, 20, 0, 0],
            [50, 80, 0, 0],
            [10, 20, 0, 0],
            [50, 80, 0, 0],
            self.edges,
            self.normalization,
        )
        self.assertEqual(result["status"], "insufficient_normalization")
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["values"]).all())

    def test_empty_bins_do_not_become_zero_error_measurements(self):
        result = normalized_correlation(
            [0, 20, 30, 40],
            [50, 0, 100, 200],
            [0, 20, 30, 40],
            [50, 0, 100, 200],
            self.edges,
            self.normalization,
        )
        np.testing.assert_array_equal(result["valid"], [False, False, True, True])
        self.assertTrue(np.isnan(result["values"][:2]).all())
        self.assertTrue(np.isnan(result["covariance"][:2]).all())

    def test_histograms_use_hyper_centrality_and_se_mass_only(self):
        config = copy.deepcopy(DEFAULTS)
        frame = pd.DataFrame(
            {
                "HypCentFT0C": [5, 5, 15, 15],
                "PairCentFT0C": [25, 25, 5, 5],
                "ChargeChannel": [0, 0, 3, 3],
                "IsMixed": [False, True, False, True],
                "mass_h3l": [2.991] * 4,
                "kstar": [0.15] * 4,
            }
        )
        histograms = empty_histograms(config)
        accumulate(histograms, frame, np.array([1, 0.5, 1, 0.2]), config)
        first = histograms["cent_0_10__ch_0"]
        second = histograms["cent_10_50__ch_3"]
        self.assertEqual(len(histograms), 8)
        self.assertEqual(first["mass_counts"].sum(), 1)
        self.assertEqual(second["selected_rows"], 2)
        self.assertAlmostEqual(first["regions"]["signal"]["me"].sum(), 0.5)
        self.assertAlmostEqual(first["regions"]["signal"]["me2"].sum(), 0.25)
        self.assertEqual(histograms["cent_10_50__ch_0"]["selected_rows"], 0)


class PurityTests(unittest.TestCase):
    def test_subtraction_and_shared_purity_covariance(self):
        raw_values = np.array([1.2, 0.9, 1.1])
        background_values = np.array([0.95, 1.03, 1.02])
        raw_cov = np.diag([0.02, 0.03, 0.04])
        background_cov = np.diag([0.04, 0.05, 0.02])
        raw = {"values": raw_values, "covariance": raw_cov, "valid": np.ones(3, bool)}
        background = {
            "values": background_values,
            "covariance": background_cov,
            "valid": np.ones(3, bool),
        }
        p, error = 0.7, 0.04
        actual = correct_correlation(raw, background, p, error)
        expected = (raw_values - (1 - p) * background_values) / p
        dp = (background_values - raw_values) / (p * p)
        covariance = (
            raw_cov / p**2
            + ((1 - p) / p) ** 2 * background_cov
            + np.outer(dp, dp) * error**2
        )
        np.testing.assert_allclose(actual["values"], expected)
        np.testing.assert_allclose(actual["covariance"], covariance)

    def test_missing_sidebands_make_correction_unavailable(self):
        raw = {
            "values": [1.1, 0.9],
            "covariance": np.eye(2),
            "valid": [True, True],
            "status": "ok",
        }
        background = {
            "values": [np.nan, np.nan],
            "covariance": np.full((2, 2), np.nan),
            "valid": [False, False],
        }
        result = correct_correlation(raw, background, 0.8, 0.03)
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["values"]).all())

    def test_exact_unit_purity_does_not_require_a_background_estimate(self):
        raw = {
            "values": [1.1, 0.9],
            "covariance": np.eye(2),
            "valid": [True, True],
            "status": "ok",
        }
        background = {
            "values": [np.nan, np.nan],
            "covariance": np.full((2, 2), np.nan),
            "valid": [False, False],
        }
        result = correct_correlation(raw, background, 1.0, 0.0)
        np.testing.assert_allclose(result["values"], raw["values"])
        np.testing.assert_allclose(result["covariance"], raw["covariance"])
        self.assertTrue(result["valid"].all())

    def test_mass_fit_refuses_truncated_sidebands(self):
        config = copy.deepcopy(DEFAULTS)
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        centers = (edges[:-1] + edges[1:]) / 2
        counts = np.where((centers >= 2.98) & (centers < 3.0), 1000, 0)
        result = fit_purity(counts, edges, config)
        self.assertEqual(result["status"], "insufficient_mass_sidebands")
        self.assertTrue(np.isnan(result["value"]))

    def test_synthetic_mass_fit_recovers_injected_purity(self):
        config = copy.deepcopy(DEFAULTS)
        edges = np.linspace(*config["analysis"]["mass_range"], 66)
        parameters = [20000, 30000, 2.99131, 0.002, 0.15]
        counts = np.random.default_rng(5381).poisson(
            mass_expectation(parameters, edges)
        )
        result = fit_purity(counts, edges, config)
        self.assertEqual(result["status"], "ok", msg=str(result))
        low, high = config["analysis"]["signal_window"]
        ns, nb, mu, sigma, slope = parameters
        signal = ns * (ndtr((high - mu) / sigma) - ndtr((low - mu) / sigma))
        signal /= ndtr((edges[-1] - mu) / sigma) - ndtr((edges[0] - mu) / sigma)
        width, center = edges[-1] - edges[0], (edges[-1] + edges[0]) / 2
        background = nb * (
            (high - low) / width
            + slope * ((high - center) ** 2 - (low - center) ** 2) / width**2
        )
        self.assertAlmostEqual(
            result["value"], signal / (signal + background), delta=0.02
        )
        self.assertGreater(result["error"], 0)


class SystematicsTests(unittest.TestCase):
    def test_envelope_uses_alternative_maxima_and_quadrature_across_sources(self):
        baseline = np.array([1.0, 1.0])
        variations = [("bdt", [1.1, 0.8]), ("bdt", [0.95, 1.1]), ("mass", [1.03, 1.04])]
        result = combine_variations(baseline, variations, "envelope")
        expected = np.outer([0.1, -0.2], [0.1, -0.2]) + np.outer(
            [0.03, 0.04], [0.03, 0.04]
        )
        np.testing.assert_allclose(result["covariance"], expected, atol=1e-15)
        np.testing.assert_allclose(result["error"], np.sqrt(np.diag(expected)))

    def test_rms_is_mean_outer_product_about_nominal(self):
        result = combine_variations(
            [1, 1], [("bdt", [1.1, 1.2]), ("bdt", [0.8, 0.9])], "rms"
        )
        expected = (
            np.outer([0.1, 0.2], [0.1, 0.2]) + np.outer([-0.2, -0.1], [-0.2, -0.1])
        ) / 2
        np.testing.assert_allclose(result["covariance"], expected)

    def test_rms_missing_joint_coverage_does_not_invent_zero_covariance(self):
        result = combine_variations(
            [1, 1], [("bdt", [2, np.nan]), ("bdt", [np.nan, 2])], "rms"
        )
        self.assertFalse(np.isfinite(result["covariance"]).all())
        self.assertNotEqual(result["status"], "ok")

    def test_missing_variation_coverage_is_not_zero_systematic(self):
        result = combine_variations([1, 1], [("bdt", [1.1, np.nan])], "envelope")
        np.testing.assert_array_equal(result["valid"], [True, False])
        self.assertTrue(np.isnan(result["error"][1]))
        empty = combine_variations([1, 1], [], "envelope")
        self.assertFalse(empty["valid"].any())
        self.assertTrue(np.isnan(empty["error"]).all())


if __name__ == "__main__":
    unittest.main()
