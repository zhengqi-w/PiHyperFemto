"""Period merge oracles use sums of counts, never averages of estimators."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pih3l.analysis import empty_mixing_qa, evaluate_histograms
from pih3l.config import DEFAULTS
from pih3l.correlation import empty_histograms
from pih3l.errors import AnalysisError
from pih3l.io import read_json, write_json
from pih3l.mc import wilson_interval
from pih3l.period_merge import (
    merge_cutflows,
    merge_histograms,
    merge_mc_qa,
    merge_mixing_qa,
    merge_raw_qa,
)
from pih3l.raw_qa import collect_raw_qa


def configuration():
    config = copy.deepcopy(DEFAULTS)
    config["analysis"].update(
        centrality_bins=[[0, 10]],
        kstar_edges=[0, 0.1, 0.2, 0.3],
        normalization=[0.1, 0.3],
    )
    config["purity"].update(method="fixed", fixed=0.8, fixed_error=0, mass_bins=4)
    return config


def histograms(config, se=(20, 30, 40), me=(30, 20, 40)):
    result = empty_histograms(config)
    for item in result.values():
        for region in item["regions"].values():
            region.update(
                se=np.asarray(se, float),
                me=np.asarray(me, float),
                se2=np.asarray(se, float) * 1.25,
                me2=np.asarray(me, float) * 0.4,
            )
        item["selected_se"], item["selected_me"] = sum(se), sum(me)
        item["selected_rows"] = sum(se) + sum(me)
        item["mass_counts"][:] = [1, 2, 3, 4]
        item["mass_vs_kstar"][0] = item["mass_counts"]
    return result


class HistogramMergeTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.first = histograms(self.config)
        self.key = next(iter(self.first))

    def test_unequal_periods_normalize_after_summing_all_sufficient_statistics(self):
        second = histograms(self.config, se=(1, 5, 10), me=(100, 100, 100))
        first_result = evaluate_histograms(self.first, self.config)
        second_result = evaluate_histograms(second, self.config)
        merged = merge_histograms(
            [("a", first_result), ("b", second_result)], self.config
        )
        item = merged[self.key]
        source = item["regions"]["signal"]
        np.testing.assert_array_equal(source["se"], [21, 35, 50])
        np.testing.assert_array_equal(source["me"], [130, 120, 140])
        np.testing.assert_allclose(source["me2"], np.array([130, 120, 140]) * 0.4)
        np.testing.assert_array_equal(item["mass_counts"], [2, 4, 6, 8])
        np.testing.assert_array_equal(item["mass_vs_kstar"][0], [2, 4, 6, 8])
        self.assertNotIn("purity", item)
        self.assertNotIn("correlations", item)
        self.assertEqual(item["periods"], ["a", "b"])
        evaluated = evaluate_histograms(merged, self.config)[self.key]
        raw = evaluated["correlations"]["signal"]
        expected = (260 / 85) * np.array([21, 35, 50]) / [130, 120, 140]
        np.testing.assert_allclose(raw["values"], expected)
        averaged = (
            first_result[self.key]["correlations"]["signal"]["values"]
            + second_result[self.key]["correlations"]["signal"]["values"]
        ) / 2
        self.assertGreater(abs(raw["values"][0] - averaged[0]), 0.05)
        np.testing.assert_array_equal(self.first[self.key]["mass_counts"], [1, 2, 3, 4])

    def test_three_replicated_inputs_preserve_cf_and_scale_formal_sumw2_covariance(
        self,
    ):
        # This checks the independent-pair algebra, not physical independence of copies.
        merged = merge_histograms([(str(i), self.first) for i in range(3)], self.config)
        nominal = evaluate_histograms(self.first, self.config)[self.key][
            "correlations"
        ]["signal"]
        result = evaluate_histograms(merged, self.config)[self.key]["correlations"][
            "signal"
        ]
        np.testing.assert_allclose(result["values"], nominal["values"])
        np.testing.assert_allclose(
            result["covariance"], nominal["covariance"] / 3, atol=1e-15
        )
        np.testing.assert_allclose(
            merged[self.key]["regions"]["signal"]["me2"],
            self.first[self.key]["regions"]["signal"]["me2"] * 3,
        )

    def test_period_with_unavailable_normalization_still_contributes_counts(self):
        incomplete = histograms(self.config, se=(20, 0, 0), me=(30, 0, 0))
        evaluated = evaluate_histograms(incomplete, self.config)
        self.assertEqual(
            evaluated[self.key]["correlations"]["signal"]["status"],
            "insufficient_normalization",
        )
        merged = merge_histograms(
            [("empty_norm", evaluated), ("good", self.first)], self.config
        )
        np.testing.assert_array_equal(
            merged[self.key]["regions"]["signal"]["se"], [40, 30, 40]
        )
        self.assertTrue(
            evaluate_histograms(merged, self.config)[self.key]["correlations"][
                "signal"
            ]["valid"].all()
        )

    def test_metadata_shapes_and_variances_must_be_compatible(self):
        changes = (
            lambda item: item.update(mass_edges=np.linspace(2.9, 3.1, 5)),
            lambda item: item.update(channel=99),
            lambda item: item.update(kstar_edges=[0, 0.11, 0.2, 0.3]),
            lambda item: item["regions"]["signal"].update(me2=[1, -1, 1]),
            lambda item: item.update(mass_counts=[1, 2]),
            lambda item: item.update(selected_rows=0),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                bad = copy.deepcopy(self.first)
                change(bad[self.key])
                with self.assertRaises(AnalysisError):
                    merge_histograms([("bad", bad)], self.config)
        with self.assertRaises(AnalysisError):
            merge_histograms([("a", self.first), ("a", self.first)], self.config)
        with self.assertRaises(AnalysisError):
            merge_histograms([], self.config)
        missing = copy.deepcopy(self.first)
        del missing[self.key]
        with self.assertRaises(AnalysisError):
            merge_histograms([("missing", missing)], self.config)


class AuxiliaryMergeTests(unittest.TestCase):
    def test_raw_qa_preserves_count_definitions_and_adds_flows(self):
        config = configuration()
        first = collect_raw_qa(config, [])
        distribution = first["hyper"]["cent_0_10"]["mass"]
        distribution["counts"][1] = 5
        distribution.update(rows=10, underflow=1, overflow=2, nonfinite=2)
        first["inputs"]["background"].update(raw_rows=12, centrality_rows=10)
        merged = merge_raw_qa([("a", first), ("b", first), ("c", first)])
        self.assertEqual(merged["inputs"]["background"]["raw_rows"], 36)
        self.assertEqual(merged["hyper"]["cent_0_10"]["mass"]["rows"], 30)
        self.assertEqual(merged["hyper"]["cent_0_10"]["mass"]["counts"][1], 15)
        self.assertEqual(merged["periods"], ["a", "b", "c"])
        self.assertEqual(distribution["rows"], 10)
        for field, value in (("selection", "different cuts"), ("weight", "weighted")):
            bad = copy.deepcopy(first)
            bad[field] = value
            with self.assertRaises(AnalysisError):
                merge_raw_qa([("a", first), ("bad", bad)])
        bad = copy.deepcopy(first)
        bad["hyper"]["cent_0_10"]["mass"]["edges"][0] -= 0.01
        with self.assertRaises(AnalysisError):
            merge_raw_qa([("a", first), ("bad", bad)])

    def test_mixing_missing_flags_are_ored_and_flag_keys_normalized(self):
        first, second = empty_mixing_qa(), empty_mixing_qa()
        first["delta_centrality_se"][0] = 10
        second["delta_centrality_se"][0] = 4
        second["missing_pair_centrality"] = True
        first["close_pair_flags"] = {0: 3, 1: 7}
        second["close_pair_flags"] = {"0": 4, "2": 2}
        result = merge_mixing_qa([("a", first), ("b", second)])
        self.assertEqual(result["delta_centrality_se"][0], 14)
        self.assertTrue(result["missing_pair_centrality"])
        self.assertEqual(result["close_pair_flags"], {"0": 7, "1": 7, "2": 2})
        second["occupancy_edges"][0] = -1
        with self.assertRaises(AnalysisError):
            merge_mixing_qa([("a", first), ("b", second)])

    def test_cutflow_retains_period_specific_thresholds_and_empty_counts(self):
        a = {"threshold": {"global": -2.1}, "counts": {"input": 10, "bdt": 5}}
        b = {"threshold": {"global": 1.2}, "counts": {"input": 20, "bdt": 3}}
        c = {"threshold": 0.1, "counts": {}}
        result = merge_cutflows([("a", a), ("b", b), ("c", c)])
        self.assertEqual(result["counts"], {"input": 30, "bdt": 8})
        self.assertEqual(
            result["threshold"], {"a": {"global": -2.1}, "b": {"global": 1.2}, "c": 0.1}
        )
        a["counts"]["input"] = -1
        with self.assertRaises(AnalysisError):
            merge_cutflows([("a", a)])


NUMERATORS = (
    "primary",
    "true_pion",
    "primary_pion",
    "signal",
    "self",
    "same_collision",
    "true_primary_pair",
    "matches_hyp_collision",
    "matches_pair_collision",
)


def write_mc_period(base, name, config, *, factor=1, success=2, residual=1.0):
    output = base / name
    bins = {}
    for key in empty_histograms(config):
        for event in ("se", "me"):
            total = np.array([10, 5, 0.0]) * factor
            count = np.array([2, 1, 0.0]) * factor
            item = {
                "total": total,
                "residual_count": count,
                "residual_sum": count * residual,
                "residual_sum2": count * residual**2,
                "resolution": np.diag(count),
            }
            for numerator in NUMERATORS:
                item[numerator] = np.array([success, 1, 0.0]) * factor
                item[numerator + "_fraction"] = wilson_interval(item[numerator], total)
            # Deliberately false derived values must not be averaged.
            item["resolution_bias"] = [999, 999, None]
            item["resolution_rms"] = [999, 999, None]
            bins[key + "__" + event] = item
    result = {
        "status": "available",
        "bins": bins,
        "counters": {
            "input_rows": 120 * factor,
            "selected_rows": 120 * factor,
            "selected_signal_window_rows": 120 * factor,
            "self_correlations": 0,
        },
        "kstar_edges": config["analysis"]["kstar_edges"],
        "truth_closure_status": "available",
        "fraction_mass_window": config["analysis"]["signal_window"],
        "fraction_weighting": "unweighted_reconstructed_pair_rows",
        "limitations": ["Pair independent approximation"],
    }
    write_json(output / "mc_qa/summary.json", result)
    for closure in ("reco_closure", "truth_closure"):
        raw = histograms(
            config,
            se=(20 * factor, 30 * factor, 40 * factor),
            me=(30 * factor, 20 * factor, 40 * factor),
        )
        write_json(
            output / "mc_qa" / closure / "results.json",
            evaluate_histograms(raw, config),
        )
    return output


class MonteCarloMergeTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_fractions_wilson_and_resolution_are_recomputed_from_sums(self):
        first = write_mc_period(self.base, "a", self.config)
        second = write_mc_period(
            self.base, "b", self.config, factor=3, success=8, residual=3
        )
        result = merge_mc_qa(
            [("a", first), ("b", second)], self.base / "merged", self.config
        )
        self.assertTrue(result["complete_period_coverage"])
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["counters"]["input_rows"], 480)
        item = result["bins"]["cent_0_10__ch_0__se"]
        fraction = item["primary_pion_fraction"]
        self.assertAlmostEqual(fraction["value"][0], 26 / 40)
        self.assertNotAlmostEqual(fraction["value"][0], (0.2 + 0.8) / 2)
        reference = wilson_interval([26, 4, 0], [40, 20, 0])
        np.testing.assert_allclose(fraction["low"], reference["low"], equal_nan=True)
        np.testing.assert_allclose(fraction["high"], reference["high"], equal_nan=True)
        np.testing.assert_allclose(item["resolution_bias"][:2], 2.5)
        np.testing.assert_allclose(item["resolution_rms"][:2], np.sqrt(0.75))
        self.assertTrue(np.isnan(item["resolution_bias"][2]))
        np.testing.assert_array_equal(np.diag(item["resolution"]), [8, 4, 0])
        closure = read_json(self.base / "merged/reco_closure/results.json")
        np.testing.assert_array_equal(
            closure["cent_0_10__ch_0"]["regions"]["signal"]["se"], [80, 120, 160]
        )
        self.assertEqual(result["truth_closure_status"], "available")
        persisted = read_json(self.base / "merged/summary.json")
        self.assertIsNone(
            persisted["bins"]["cent_0_10__ch_0__se"]["resolution_bias"][2]
        )

    def test_missing_mc_period_is_explicit_and_all_missing_stays_unavailable(self):
        first = write_mc_period(self.base, "a", self.config)
        missing = self.base / "missing"
        write_json(
            missing / "mc_qa/summary.json",
            {"status": "skipped_missing_mc_femto", "reason": "empty tree"},
        )
        result = merge_mc_qa(
            [("a", first), ("missing", missing)], self.base / "partial", self.config
        )
        self.assertEqual(result["status"], "available")
        self.assertFalse(result["complete_period_coverage"])
        self.assertEqual(result["available_periods"], ["a"])
        self.assertEqual(result["missing_periods"], ["missing"])
        self.assertFalse(result["truth_closure_coverage"]["complete_period_coverage"])
        result = merge_mc_qa([("missing", missing)], self.base / "empty", self.config)
        self.assertEqual(result["status"], "skipped_missing_mc_femto")
        self.assertNotIn("bins", result)
        self.assertFalse((self.base / "empty/truth_closure").exists())
        self.assertFalse((self.base / "empty/reco_closure").exists())

    def test_missing_truth_is_not_a_measured_zero_or_subset_closure(self):
        first = write_mc_period(self.base, "a", self.config)
        second = write_mc_period(self.base, "b", self.config)
        source = read_json(second / "mc_qa/summary.json")
        source["truth_closure_status"] = "missing_truth_columns"
        for item in source["bins"].values():
            item["primary_pion"] = [0, 0, 0]
            item["primary_pion_fraction"] = {"status": "missing_truth_columns"}
        write_json(second / "mc_qa/summary.json", source)
        result = merge_mc_qa(
            [("a", first), ("b", second)], self.base / "merged", self.config
        )
        fraction = result["bins"]["cent_0_10__ch_0__se"]["primary_pion_fraction"]
        self.assertNotIn("value", fraction)
        self.assertEqual(fraction["missing_periods"], ["b"])
        self.assertEqual(result["truth_closure_status"], "missing_truth_columns")
        self.assertFalse((self.base / "merged/truth_closure").exists())
        self.assertEqual(result["reco_closure_status"], "available")

    def test_mc_grid_mismatch_and_missing_declared_artifact_are_errors(self):
        first = write_mc_period(self.base, "a", self.config)
        source = read_json(first / "mc_qa/summary.json")
        source["kstar_edges"][1] = 0.11
        write_json(first / "mc_qa/summary.json", source)
        with self.assertRaises(AnalysisError):
            merge_mc_qa([("a", first)], self.base / "bad", self.config)
        source["kstar_edges"][1] = 0.1
        write_json(first / "mc_qa/summary.json", source)
        (first / "mc_qa/reco_closure/results.json").unlink()
        with self.assertRaises(AnalysisError):
            merge_mc_qa([("a", first)], self.base / "bad", self.config)


if __name__ == "__main__":
    unittest.main()
