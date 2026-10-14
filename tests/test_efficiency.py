"""HIPE mapping contract: mixed classes, precise retry and no silent fallback."""

import unittest
from unittest.mock import patch

import numpy as np
from hipe4ml import analysis_utils

from pih3l.efficiency import thresholds_from_efficiencies
from pih3l.errors import AnalysisError


class EfficiencyMappingTests(unittest.TestCase):
    def test_mixed_class_mapping_matches_official_hipe_api(self):
        random = np.random.default_rng(912)
        labels = np.r_[np.ones(400, dtype=int), np.zeros(600, dtype=int)]
        scores = np.r_[random.beta(4, 2, 400), random.beta(1.5, 5, 598), 0.0, 1.0]
        targets = np.array([0.373, 0.613, 0.827])
        official = analysis_utils.score_from_efficiency_array(
            labels, scores, targets, keep_lower=False
        )
        with patch.object(
            analysis_utils,
            "score_from_efficiency_array",
            wraps=analysis_utils.score_from_efficiency_array,
        ) as mapper:
            result, metadata = thresholds_from_efficiencies(
                labels, scores, targets, return_metadata=True
            )
        np.testing.assert_array_equal(result, official)
        self.assertFalse(metadata["numerical_retry"])
        np.testing.assert_array_equal(mapper.call_args.args[0], labels)
        np.testing.assert_array_equal(mapper.call_args.args[1], scores)
        self.assertEqual(mapper.call_args.kwargs, {"keep_lower": False})
        self.assertTrue((np.diff(result) <= 0).all())

    def test_missing_spline_root_retries_only_failed_target_in_same_batch_context(self):
        labels, scores = np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.2, 0.9])
        targets = np.array([0.3, 0.5, 0.8, 1.0])
        calls = []

        def mapping(batch_labels, batch_scores, batch_targets, **kwargs):
            calls.append(
                (batch_labels.copy(), batch_scores.copy(), batch_targets.copy(), kwargs)
            )
            if len(calls) == 1:
                raise IndexError("missing spline root")
            return np.array([0.8, 0.5, 0.2])

        class Spline:
            def __init__(self, grid, shifted_curve):
                self.target = -shifted_curve[0]

            def roots(self):
                return np.array([] if self.target == 0.5 else [0.4])

        with patch.object(
            analysis_utils, "score_from_efficiency_array", side_effect=mapping
        ), patch.object(
            analysis_utils,
            "bdt_efficiency_array",
            return_value=(np.zeros(4), np.linspace(0, 1, 4)),
        ) as curve, patch(
            "scipy.interpolate.InterpolatedUnivariateSpline", Spline
        ):
            result, metadata = thresholds_from_efficiencies(
                labels, scores, targets, return_metadata=True
            )
        np.testing.assert_array_equal(result, [0.8, 0.5, 0.2, 0.1])
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0][2], targets[:3])
        expected = targets.copy()
        expected[1] = np.nextafter(0.5, 0.0)
        np.testing.assert_array_equal(calls[1][2], expected[:3])
        np.testing.assert_array_equal(metadata["evaluated_efficiencies"], expected)
        np.testing.assert_array_equal(targets, [0.3, 0.5, 0.8, 1.0])
        self.assertTrue(metadata["numerical_retry"])
        for call in calls:
            np.testing.assert_array_equal(call[0], labels)
            np.testing.assert_array_equal(call[1], scores)
            self.assertEqual(call[3], {"keep_lower": False})
        curve.assert_called_once()
        np.testing.assert_array_equal(curve.call_args.args[0], labels)
        np.testing.assert_array_equal(curve.call_args.args[1], scores)
        self.assertEqual(
            curve.call_args.kwargs, {"n_points": 1000, "keep_lower": False}
        )

    def test_interpolation_failure_has_no_empirical_quantile_fallback(self):
        with patch.object(
            analysis_utils,
            "score_from_efficiency_array",
            side_effect=ValueError("invalid spline"),
        ), patch(
            "numpy.quantile", side_effect=AssertionError("Forbidden quantile fallback")
        ):
            with self.assertRaisesRegex(
                AnalysisError, "No quantile fallback or automatic retraining"
            ):
                thresholds_from_efficiencies([0, 1, 1], [0.1, 0.6, 0.9], [0.6])

    def test_retry_failure_is_explicit_instead_of_substituting_threshold(self):
        class NoRoot:
            def __init__(self, *args):
                pass

            def roots(self):
                return []

        with patch.object(
            analysis_utils,
            "score_from_efficiency_array",
            side_effect=IndexError("still no root"),
        ) as mapper, patch.object(
            analysis_utils,
            "bdt_efficiency_array",
            return_value=(np.zeros(4), np.linspace(0, 1, 4)),
        ), patch(
            "scipy.interpolate.InterpolatedUnivariateSpline", NoRoot
        ), patch(
            "numpy.quantile", side_effect=AssertionError("Forbidden fallback")
        ):
            with self.assertRaisesRegex(AnalysisError, "interpolation failed"):
                thresholds_from_efficiencies([0, 1, 1], [0.1, 0.6, 0.9], [0.6])
        self.assertEqual(mapper.call_count, 2)

    def test_unit_efficiency_is_inclusive_margin_minimum_without_hipe_extrapolation(
        self,
    ):
        labels, scores = np.array([0, 1, 1]), np.array([8.0, -3.0, 0.7])
        with patch.object(
            analysis_utils,
            "score_from_efficiency_array",
            side_effect=AssertionError("Unit efficiency must not be interpolated"),
        ):
            cuts, metadata = thresholds_from_efficiencies(
                labels, scores, [1.0, 1.0], return_metadata=True
            )
        np.testing.assert_array_equal(cuts, [-3.0, -3.0])
        self.assertTrue((scores[labels == 1] >= cuts[0]).all())
        self.assertFalse(metadata["numerical_retry"])


if __name__ == "__main__":
    unittest.main()
