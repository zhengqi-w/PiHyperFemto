"""Check segmented reductions and one-pass ensembles against independent scans."""

from pih3l.scored_storage import write_storage_metadata
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pih3l.analysis import run_correlations
from pih3l.config import DEFAULTS, merge
from pih3l.correlation import accumulate, empty_histograms
from pih3l.io import iter_parquet, read_json, write_batch, write_json
from pih3l.systematics import run_systematics


class EnsembleProcessingTests(unittest.TestCase):
    def test_scoped_bundle_without_registry_preserves_application_range(self):
        from pih3l.models import load_model_registry, score_models

        with tempfile.TemporaryDirectory() as temporary:
            from test_model_routing import toy_handler

            toy_handler().dump_model_handler(str(Path(temporary) / "model_handler.pkl"))
            write_json(
                Path(temporary) / "metrics.json",
                {
                    "features": ["DcaHe"],
                    "model_id": "shared",
                    "score_output_margin": True,
                    "score_domain": "raw_margin",
                    "training_pt_range": [2, 8],
                },
            )
            registry = load_model_registry(temporary)
            frame = pd.DataFrame({"pt_h3l": [1, 2, 8], "DcaHe": [0.5] * 3})
            scored = score_models(frame, registry)
            np.testing.assert_array_equal(scored.bdt_in_scope, [False, True, False])
            self.assertEqual(scored.bdt_model_id.tolist(), ["", "shared", ""])

    def config(self):
        config = copy.deepcopy(DEFAULTS)
        config["purity"].update(method="fixed", fixed=0.8)
        config["analysis"]["close_pair_reject_bits"] = 0
        config["systematics"]["variations"] = [
            {
                "name": "norm",
                "group": "norm",
                "overrides": {"analysis": {"normalization": [0.26, 0.46]}},
            },
            {
                "name": "fit",
                "group": "fit",
                "overrides": {"purity": {"background": "constant"}},
            },
            {
                "name": "loose",
                "group": "bdt",
                "overrides": {"analysis": {"bdt_threshold": 0.3}},
            },
            {
                "name": "tight",
                "group": "bdt",
                "overrides": {"analysis": {"bdt_threshold": 0.7}},
            },
            {
                "name": "duplicate",
                "group": "bdt",
                "overrides": {
                    "analysis": {"bdt_threshold": 0.7, "normalization": [0.26, 0.46]}
                },
            },
        ]
        return config

    def frame(self):
        rng = np.random.default_rng(451)
        n = 14000
        frame = pd.DataFrame(
            {
                "HypCentFT0C": rng.choice([0, 5, 10, 30, 50], n),
                "ChargeChannel": rng.integers(0, 4, n),
                "IsMixed": rng.integers(0, 2, n).astype(bool),
                "mass_h3l": rng.uniform(2.95, 3.035, n),
                "kstar": rng.uniform(0, 0.6, n),
                "bdt_score": rng.uniform(0, 1, n),
                "MixingDepth": rng.integers(1, 6, n),
            }
        )
        # Include exact histogram edges and half-open selection bounds.
        frame.loc[:5, "kstar"] = [0, 0.5, 0.6, 0.1, np.nan, -0.01]
        frame.loc[:5, "mass_h3l"] = [2.96, 3.025, 2.985, 2.997, np.nan, 3.007]
        return frame

    def test_segmented_reductions_match_numpy_histogram_reference(self):
        config, frame = self.config(), self.frame()
        weights = np.where(frame.IsMixed, 1 / frame.MixingDepth, 1)
        actual, expected = empty_histograms(config), empty_histograms(config)
        accumulate(actual, frame, weights, config)
        a = config["analysis"]
        for key, item in expected.items():
            low, high = item["centrality"]
            mask = (
                (frame.HypCentFT0C >= low)
                & (frame.HypCentFT0C < high)
                & (frame.ChargeChannel == item["channel"])
                & (frame.kstar >= a["kstar_edges"][0])
                & (frame.kstar < a["kstar_edges"][-1])
            ).to_numpy()
            sample, w = frame.loc[mask], weights[mask]
            mass, ks, mixed = (
                sample.mass_h3l.to_numpy(),
                sample.kstar.to_numpy(),
                sample.IsMixed.to_numpy(),
            )
            self.assertEqual(actual[key]["selected_rows"], len(sample))
            self.assertEqual(actual[key]["selected_se"], int((~mixed).sum()))
            np.testing.assert_array_equal(
                actual[key]["mass_counts"],
                np.histogram(mass[~mixed], item["mass_edges"])[0],
            )
            np.testing.assert_array_equal(
                actual[key]["mass_vs_kstar"],
                np.histogram2d(
                    ks[~mixed], mass[~mixed], [a["kstar_edges"], item["mass_edges"]]
                )[0],
            )
            for region, windows in (
                ("signal", [a["signal_window"]]),
                ("left", [a["sidebands"][0]]),
                ("right", [a["sidebands"][1]]),
                ("sideband", a["sidebands"]),
            ):
                region_mask = np.zeros(len(sample), bool)
                for lo, hi in windows:
                    region_mask |= (mass >= lo) & (mass < hi)
                for label, event in (("se", ~mixed), ("me", mixed)):
                    keep = region_mask & event
                    for power, field in ((1, label), (2, label + "2")):
                        reference = np.histogram(
                            ks[keep], a["kstar_edges"], weights=w[keep] ** power
                        )[0]
                        np.testing.assert_allclose(
                            actual[key]["regions"][region][field], reference, atol=1e-11
                        )

    def test_ensemble_reads_once_and_matches_independent_scans(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            frame = self.frame()
            write_batch(base / "scored", 0, frame.iloc[:7000])
            write_batch(base / "scored", 1, frame.iloc[7000:])
            write_storage_metadata(base / "scored", config, {"storage_threshold": None})
            run_correlations(base / "scored", base / "nominal", config, 0.5)
            with patch("pih3l.systematics.iter_parquet", wraps=iter_parquet) as reader:
                run_systematics(
                    base / "scored", base / "nominal", base / "ensemble", config, 0.5
                )
                self.assertEqual(reader.call_count, 1)
            summary = read_json(base / "ensemble/summary.json")
            self.assertEqual(summary["execution"]["histogram_groups"], 2)
            self.assertEqual(summary["execution"]["nominal_histogram_reuse"], 2)
            for variation in config["systematics"]["variations"]:
                settings = merge(config, variation["overrides"])
                reference = run_correlations(
                    base / "scored",
                    base / "reference" / variation["name"],
                    settings,
                    0.5,
                )
                result = read_json(
                    base / "ensemble" / variation["name"] / "results.json"
                )
                for key, item in reference.items():
                    np.testing.assert_allclose(
                        result[key]["mass_counts"], item["mass_counts"]
                    )
                    for region in item["regions"]:
                        for field in item["regions"][region]:
                            np.testing.assert_allclose(
                                result[key]["regions"][region][field],
                                item["regions"][region][field],
                            )
                    # JSON null represents unavailable values; compare numerically as NaN.
                    np.testing.assert_allclose(
                        np.asarray(result[key]["corrected"]["values"], float),
                        item["corrected"]["values"],
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        np.asarray(result[key]["corrected"]["covariance"], float),
                        item["corrected"]["covariance"],
                        equal_nan=True,
                    )
            config["systematics"]["variations"] = config["systematics"]["variations"][
                :2
            ]
            with patch(
                "pih3l.systematics.iter_parquet",
                side_effect=AssertionError("unexpected data scan"),
            ):
                run_systematics(
                    base / "scored", base / "nominal", base / "reuse", config, 0.5
                )


if __name__ == "__main__":
    unittest.main()
