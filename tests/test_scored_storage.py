"""Lossless nominal/systematic selection inside the compact storage envelope."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pih3l.analysis import run_correlations
from pih3l.config import DEFAULTS, merge
from pih3l.errors import AnalysisError
from pih3l.io import iter_parquet, parquet_columns, parquet_paths, write_batch
from pih3l.scored_storage import (
    ensure_storage_coverage,
    write_scored_batch,
    write_storage_metadata,
)


class ScoredStorageTests(unittest.TestCase):
    def test_period_catalog_cannot_be_misread_as_one_model_dataset_or_missing_mc(self):
        from pih3l.io import write_json
        from pih3l.mc import run_mc_qa

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            catalog = base / "scored"
            write_json(
                catalog / "dataset.json",
                {"kind": "period_reference_catalog", "score_domain": "raw_margin"},
            )
            with self.assertRaisesRegex(AnalysisError, "multi-period scored catalog"):
                run_correlations(catalog, base / "nominal", self.config(), 0.5)
            with self.assertRaisesRegex(AnalysisError, "multi-period scored catalog"):
                run_mc_qa(catalog, base / "mc_qa", self.config(), 0.5)
            self.assertFalse((base / "mc_qa/summary.json").exists())

    def test_unbounded_storage_retains_negative_margins_and_rejects_old_domains(self):
        from pih3l.io import read_json, write_json

        config = self.config()
        config["working_point"]["storage"]["enabled"] = False
        frame = self.rows(6)
        frame.bdt_score = [-9.0, -2.0, 0.0, 1.0, 8.0, np.nan]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths, counts = write_scored_batch(directory, 0, frame, config, {}, "femto")
            write_storage_metadata(directory, config, {}, counts=counts)
            self.assertEqual(counts["retained_rows"], 5)
            self.assertEqual(
                sorted(pd.read_parquet(paths[0]).bdt_score), [-9, -2, 0, 1, 8]
            )
            metadata = read_json(directory / "dataset.json")
            self.assertIsNone(metadata["minimum_threshold"])
            ensure_storage_coverage(directory, -100, config)
            with self.assertRaises(AnalysisError):
                ensure_storage_coverage(directory, np.nan)
            metadata["score_domain"] = "probability"
            write_json(directory / "dataset.json", metadata)
            with self.assertRaisesRegex(AnalysisError, "score domain"):
                ensure_storage_coverage(directory, -100, config)

    def config(self):
        config = copy.deepcopy(DEFAULTS)
        config["purity"].update(method="fixed", fixed=0.8, fixed_error=0.02)
        config["analysis"]["kstar_edges"] = np.linspace(0, 0.3, 7).tolist()
        config["analysis"]["normalization"] = [0.2, 0.3]
        config["analysis"]["bdt_threshold"] = 0.5
        config["systematics"]["variations"] = [
            {"name": "loose", "overrides": {"analysis": {"bdt_threshold": 0.3}}},
            {
                "name": "pid",
                "overrides": {
                    "analysis": {
                        "pair_cuts": {"PairPiTrackTpcNSigmaPi": {"abs_max": 2}}
                    }
                },
            },
        ]
        return config

    def rows(self, count=4000):
        rng = np.random.default_rng(27)
        return pd.DataFrame(
            {
                "mass_h3l": rng.choice([2.968, 2.991, 3.016], count),
                "kstar": rng.choice(np.arange(0.025, 0.3, 0.05), count),
                "pt_h3l": rng.choice([2.5, 4.5], count),
                "HypCentFT0C": rng.choice([5, 25], count),
                "ChargeChannel": rng.integers(0, 4, count),
                "IsMixed": rng.integers(0, 2, count).astype(bool),
                "IsClosePairRejected": np.zeros(count, dtype=np.uint8),
                "bdt_score": rng.choice([0.1, 0.2, 0.3, 0.5, 0.9], count),
                "PairPiTrackTpcNSigmaPi": rng.choice([1.0, 2.5], count),
                "DcaHe": np.ones(count),
                "PtHe3": np.ones(count),
                "_source_file": ["input.root"] * count,
            }
        )

    def test_pruning_preserves_every_declared_nominal_and_systematic_histogram(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, frame = self.config(), self.rows()
            write_batch(base / "full", 0, frame)
            write_storage_metadata(base / "full", config, {"storage_threshold": None})
            wp = {"threshold": 0.5, "storage_threshold": 0.2}
            paths, counts = write_scored_batch(
                base / "compact", 0, frame, config, wp, "femto"
            )
            write_storage_metadata(base / "compact", config, wp, counts=counts)
            self.assertGreater(counts["dropped_below_storage"], 0)
            self.assertLess(counts["retained_rows"], counts["rows"])
            columns = parquet_columns(base / "compact")
            self.assertIn("PairPiTrackTpcNSigmaPi", columns)
            self.assertNotIn("DcaHe", columns)
            self.assertNotIn("PtHe3", columns)
            self.assertNotIn("_source_file", columns)
            self.assertEqual(len(paths), 1)
            variants = [{"analysis": {"bdt_threshold": 0.5}}] + [
                item["overrides"] for item in config["systematics"]["variations"]
            ]
            for index, override in enumerate(variants):
                settings = merge(config, override)
                full = run_correlations(
                    base / "full", base / f"full_result{index}", settings, 0.5
                )
                compact = run_correlations(
                    base / "compact", base / f"compact_result{index}", settings, 0.5
                )
                for key in full:
                    for region in ("signal", "left", "right", "sideband"):
                        for event in ("se", "me", "se2", "me2"):
                            np.testing.assert_array_equal(
                                full[key]["regions"][region][event],
                                compact[key]["regions"][region][event],
                            )
                    np.testing.assert_allclose(
                        full[key]["correlations"]["signal"]["covariance"],
                        compact[key]["correlations"]["signal"]["covariance"],
                        equal_nan=True,
                    )

    def test_inclusive_ties_pt_subdirectories_and_recursive_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.config()
            config["training"]["pt_bins"] = [
                {"id": "low", "pt_range": [2, 4]},
                {"id": "high", "pt_range": [4, 6]},
            ]
            frame = self.rows(12)
            frame["bdt_model_id"] = np.where(frame.pt_h3l < 4, "low", "high")
            frame["bdt_score"] = np.where(frame.pt_h3l < 4, 0.3, 0.5)
            frame["bdt_in_scope"] = True
            floor = {"low": 0.3, "high": 0.5}
            paths, counts = write_scored_batch(
                directory, 7, frame, config, {"storage_threshold": floor}, "femto"
            )
            write_storage_metadata(
                directory, config, {"storage_threshold": floor}, counts=counts
            )
            self.assertEqual(counts["retained_rows"], len(frame))
            self.assertEqual({path.parent.name for path in paths}, {"low", "high"})
            self.assertFalse(list(directory.glob("*.parquet")))
            self.assertEqual(len(parquet_paths(directory)), 2)
            loaded = pd.concat(iter_parquet(directory, batch_size=3))
            self.assertEqual(len(loaded), len(frame))
            for path in paths:
                self.assertTrue(
                    (pd.read_parquet(path).bdt_model_id == path.parent.name).all()
                )
            ensure_storage_coverage(directory, floor, config)
            with self.assertRaisesRegex(AnalysisError, "outside the saved envelope"):
                ensure_storage_coverage(directory, {"low": 0.29, "high": 0.5})

    def test_explicit_saved_variables_and_domain_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.config()
            config["io"]["scored_columns"].append("DcaHe")
            wp = {"storage_threshold": 0.2}
            write_scored_batch(directory, 0, self.rows(), config, wp, "femto")
            write_storage_metadata(directory, config, wp)
            self.assertIn("DcaHe", parquet_columns(directory))
            for override in (
                {"mass_range": [2.95, 3.025]},
                {"centrality_bins": [[0, 60]]},
                {"kstar_edges": np.linspace(0, 0.4, 9).tolist()},
            ):
                with self.subTest(override=override):
                    changed = merge(config, {"analysis": override})
                    with self.assertRaisesRegex(AnalysisError, "saved scored domain"):
                        ensure_storage_coverage(directory, 0.5, changed)


if __name__ == "__main__":
    unittest.main()
