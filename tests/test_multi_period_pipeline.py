"""Native multi-period closure, isolated caches, and period-scoped ROOT QA.

Repeating the same synthetic input is a software arithmetic test. The resulting
Poisson covariance scaling is not evidence for three independent data samples.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from synthetic_data import create_test_inputs, write_test_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIOD_IDS = ("period_a", "period_b", "period_c")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_config(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False))


def manifest_snapshot(output):
    return {
        str(path.relative_to(output)): {
            "mtime_ns": path.stat().st_mtime_ns,
            "signature": read_json(path)["signature"],
        }
        for path in Path(output).rglob("manifest.json")
        if "signature" in read_json(path)
    }


class PeriodContextNamespaceTests(unittest.TestCase):
    def contexts(self, specifications):
        from pih3l.report_models import model_contexts

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = []
            for period, models in specifications:
                relative = "periods/" + period
                directory = output / relative / "training"
                directory.mkdir(parents=True)
                metrics = (
                    {
                        "mode": "pt_bins",
                        "models": [
                            {"id": model, "relative_path": "models/" + model}
                            for model in models
                        ],
                    }
                    if models is not None
                    else {"features": ["DcaHe"]}
                )
                (directory / "metrics.json").write_text(json.dumps(metrics))
                records.append({"id": period, "relative_path": relative})
            return list(model_contexts(output, {"mode": "periods", "periods": records}))

    def test_delimiter_in_period_cannot_collide_with_period_plus_model(self):
        contexts = self.contexts([("a__b", None), ("a", ["b"])])
        self.assertEqual(contexts[0]["suffix"], "__p4_a__b__global")
        self.assertEqual(contexts[1]["suffix"], "__p1_a__m1_b")
        self.assertNotEqual(contexts[0]["suffix"], contexts[1]["suffix"])

    def test_global_and_length_marker_words_remain_unique_namespaces(self):
        contexts = self.contexts(
            [
                ("global", None),
                ("global__global", None),
                ("p1_a", ["global", "m1_b", "b__global"]),
                ("a", ["global", "b", "p1_a"]),
                ("a__m6_global", None),
            ]
        )
        names = ["working_point" + item["suffix"] for item in contexts]
        self.assertEqual(len(set(names)), len(names))
        for item in contexts:
            period = item["period_id"]
            self.assertTrue(item["suffix"].startswith(f"__p{len(period)}_{period}"))


class MultiPeriodPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.data_root = cls.base / "input"
        cls.input_pair_rows = create_test_inputs(cls.data_root)
        cls.output = cls.base / "output"
        cls.config_path = cls.base / "periods.yaml"
        write_test_config(cls.config_path, cls.data_root, cls.output)
        cls.settings = yaml.safe_load(cls.config_path.read_text())
        cls.settings.update(
            periods=[{"id": period} for period in PERIOD_IDS],
            synthetic_repetition=True,
        )
        write_config(cls.config_path, cls.settings)
        cls.environment = os.environ.copy()
        cls.environment.update(
            PYTHONPATH=str(PROJECT_ROOT / "src"),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
        )
        cls._run_cli("run", cls.config_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _run_cli(cls, command, path):
        result = subprocess.run(
            [sys.executable, "-m", "pih3l", command, "-c", str(path)],
            cwd=PROJECT_ROOT,
            env=cls.environment,
            capture_output=True,
            text=True,
            timeout=240,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    @contextmanager
    def isolated_mutation(self):
        """Restore the fixture at its original paths, including cached mtimes."""
        config_contents = self.config_path.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            backup = Path(temporary) / "output"
            shutil.copytree(self.output, backup)
            try:
                yield
            finally:
                shutil.rmtree(self.output)
                shutil.copytree(backup, self.output)
                self.config_path.write_bytes(config_contents)

    def child(self, period):
        return self.output / "periods" / period

    def test_every_period_has_its_own_model_working_point_and_scores(self):
        model_inodes = []
        score_reference = None
        threshold_reference = None
        for period in PERIOD_IDS:
            with self.subTest(period=period):
                child = self.child(period)
                model = child / "training/model_handler.pkl"
                self.assertTrue(model.is_file())
                model_inodes.append(model.stat().st_ino)
                metrics = read_json(child / "training/metrics.json")
                self.assertEqual(metrics["score_domain"], "raw_margin")
                self.assertEqual(metrics["features"], ["DcaHe"])
                wp = read_json(child / "working_point/summary.json")
                self.assertEqual(wp["status"], "ok")
                self.assertTrue(np.isfinite(wp["threshold"]))
                metadata = read_json(child / "scored/femto/dataset.json")
                self.assertEqual(metadata["score_domain"], "raw_margin")
                self.assertTrue(list((child / "scored/femto").rglob("*.parquet")))
                counts = read_json(child / "scored/summary.json")
                self.assertEqual(counts["femto"]["rows"], self.input_pair_rows)
                self.assertEqual(
                    read_json(child / "nominal/cutflow.json")["threshold"],
                    wp["threshold"],
                )
                with np.load(
                    child / "training/validation_scores.npz", allow_pickle=False
                ) as saved:
                    current = saved["bdt_score"].copy()
                if score_reference is None:
                    score_reference = current
                    threshold_reference = wp["threshold"]
                else:
                    np.testing.assert_array_equal(current, score_reference)
                    self.assertEqual(wp["threshold"], threshold_reference)
        self.assertEqual(len(set(model_inodes)), 3)

    def test_parent_indexes_do_not_invent_a_global_model_or_score_dataset(self):
        for path in ("training/metrics.json", "working_point/summary.json"):
            index = read_json(self.output / path)
            self.assertEqual(index["mode"], "periods")
            self.assertEqual({row["id"] for row in index["periods"]}, set(PERIOD_IDS))
            self.assertEqual(
                {row["relative_path"] for row in index["periods"]},
                {"periods/" + name for name in PERIOD_IDS},
            )
        training = read_json(self.output / "training/metrics.json")
        self.assertNotIn("auc_validation", training)
        self.assertTrue(training["period_metadata"]["synthetic_repetition"])
        self.assertIn(
            "not independent", training["period_metadata"]["statistical_interpretation"]
        )
        self.assertFalse((self.output / "training/model_handler.pkl").exists())
        self.assertFalse(list((self.output / "scored").rglob("*.parquet")))
        wp = read_json(self.output / "working_point/summary.json")
        self.assertEqual(set(wp["thresholds_by_period"]), set(PERIOD_IDS))
        for period in PERIOD_IDS:
            self.assertEqual(
                wp["thresholds_by_period"][period],
                read_json(self.child(period) / "working_point/summary.json")[
                    "threshold"
                ],
            )

    def test_pair_histograms_and_mass_statistics_sum_before_normalization(self):
        combined = read_json(self.output / "nominal/results.json")
        children = [
            read_json(self.child(period) / "nominal/results.json")
            for period in PERIOD_IDS
        ]
        self.assertEqual(len(combined), 8)
        for key, item in combined.items():
            with self.subTest(bin=key):
                for region in ("signal", "left", "right", "sideband"):
                    for name in ("se", "me", "se2", "me2"):
                        expected = sum(
                            np.asarray(child[key]["regions"][region][name])
                            for child in children
                        )
                        np.testing.assert_array_equal(
                            item["regions"][region][name], expected
                        )
                        np.testing.assert_array_equal(
                            expected,
                            3 * np.asarray(children[0][key]["regions"][region][name]),
                        )
                for name in ("mass_counts", "mass_vs_kstar"):
                    np.testing.assert_array_equal(
                        item[name],
                        sum(np.asarray(child[key][name]) for child in children),
                    )
                for name in ("selected_rows", "selected_se", "selected_me"):
                    self.assertEqual(
                        item[name], sum(child[key][name] for child in children)
                    )
                np.testing.assert_array_equal(
                    item["mass_edges"], children[0][key]["mass_edges"]
                )
                self.assertEqual(item["channel"], children[0][key]["channel"])
                self.assertEqual(item["centrality"], children[0][key]["centrality"])
                hist = item["regions"]["signal"]
                se, me = np.asarray(hist["se"]), np.asarray(hist["me"])
                edges = np.asarray(self.settings["analysis"]["kstar_edges"])
                low, high = self.settings["analysis"]["normalization"]
                norm = (edges[:-1] >= low - 1e-10) & (edges[1:] <= high + 1e-10)
                expected_cf = se / me * me[norm].sum() / se[norm].sum()
                np.testing.assert_allclose(
                    item["correlations"]["signal"]["values"], expected_cf, atol=1e-12
                )
                np.testing.assert_allclose(expected_cf, 1, atol=1e-12)
                np.testing.assert_allclose(
                    item["correlations"]["signal"]["covariance"],
                    np.asarray(children[0][key]["correlations"]["signal"]["covariance"])
                    / 3,
                    atol=1e-12,
                )

    def test_raw_qa_and_mixing_counts_sum_without_scaling_edges(self):
        merged = read_json(self.output / "raw_qa/results.json")
        first = read_json(self.child(PERIOD_IDS[0]) / "raw_qa/results.json")
        for role in ("femto", "background"):
            for field in ("raw_rows", "centrality_rows"):
                self.assertEqual(
                    merged["inputs"][role][field], 3 * first["inputs"][role][field]
                )
        for role, distributions in (
            ("femto", ("kstar_se", "kstar_me")),
            ("hyper", ("mass",)),
        ):
            for key in first[role]:
                for name in distributions:
                    actual, expected = merged[role][key][name], first[role][key][name]
                    np.testing.assert_array_equal(actual["edges"], expected["edges"])
                    np.testing.assert_array_equal(
                        actual["counts"], 3 * np.asarray(expected["counts"])
                    )
                    for field in ("rows", "nonfinite", "underflow", "overflow"):
                        self.assertEqual(actual[field], 3 * expected[field])
        merged = read_json(self.output / "nominal/mixing_qa.json")
        first = read_json(self.child(PERIOD_IDS[0]) / "nominal/mixing_qa.json")
        for field in (
            "occupancy_se",
            "occupancy_me",
            "delta_centrality_se",
            "delta_centrality_me",
        ):
            np.testing.assert_array_equal(merged[field], 3 * np.asarray(first[field]))
        for field in ("occupancy_edges", "delta_centrality_edges"):
            np.testing.assert_array_equal(merged[field], first[field])
        self.assertEqual(
            merged["missing_pair_centrality"], first["missing_pair_centrality"]
        )

    def test_systematic_variations_merge_each_period_before_the_cf(self):
        summary = read_json(self.output / "systematics/summary.json")
        self.assertTrue(all(v["status"] == "completed" for v in summary["variations"]))
        self.assertEqual(
            {v["name"] for v in summary["variations"]},
            {v["name"] for v in self.settings["systematics"]["variations"]},
        )
        for variation in summary["variations"]:
            relative = "systematics/" + variation["name"] + "/results.json"
            merged = read_json(self.output / relative)
            children = [
                read_json(self.child(period) / relative) for period in PERIOD_IDS
            ]
            for key in merged:
                for region in ("signal", "sideband"):
                    for field in ("se", "me", "se2", "me2"):
                        np.testing.assert_array_equal(
                            merged[key]["regions"][region][field],
                            sum(
                                np.asarray(child[key]["regions"][region][field])
                                for child in children
                            ),
                        )

    def test_missing_mc_remains_explicit_without_substituting_training_mc(self):
        for directory in [self.output, *(self.child(period) for period in PERIOD_IDS)]:
            summary = read_json(directory / "mc_qa/summary.json")
            self.assertIn("skipped", summary["status"])
            self.assertFalse(summary.get("bins"))

    def test_overlapping_femto_requires_explicit_synthetic_repetition(self):
        from pih3l.config import load_config
        from pih3l.errors import AnalysisError
        from pih3l.multi_period import MultiPeriodPipeline

        config = load_config(self.config_path)
        config["synthetic_repetition"] = False
        with self.assertRaisesRegex(AnalysisError, "overlap"):
            MultiPeriodPipeline(config)

    def test_one_failed_period_variation_cannot_become_a_partial_merged_result(self):
        from pih3l.config import load_config
        from pih3l.multi_period import MultiPeriodPipeline

        with self.isolated_mutation(), tempfile.TemporaryDirectory() as temporary:
            child_summary = self.child("period_b") / "systematics/summary.json"
            summary = read_json(child_summary)
            affected = summary["variations"][0]["name"]
            summary["variations"][0].update(
                status="failed", reason="injected missing column"
            )
            child_summary.write_text(json.dumps(summary))
            # Deliberately leave an older per-variation result present. Its mere
            # existence must not override the failed current outcome.
            self.assertTrue(
                (
                    self.child("period_b") / "systematics" / affected / "results.json"
                ).is_file()
            )
            pipeline = MultiPeriodPipeline(load_config(self.config_path))
            destination = Path(temporary)
            pipeline._systematics(destination)
            merged = read_json(destination / "summary.json")
            outcome = next(
                item for item in merged["variations"] if item["name"] == affected
            )
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(outcome["failed_periods"], ["period_b"])
            self.assertFalse((destination / affected / "results.json").exists())
            qa = read_json(destination / "qa/results.json")
            self.assertEqual(qa["variations"][affected]["status"], "failed")
            for key in merged["bins"]:
                self.assertFalse(any(merged["bins"][key]["raw"]["valid"]))
                self.assertFalse(
                    any(qa["variations"][affected]["bins"][key]["raw"]["valid"])
                )

    def test_report_links_and_root_objects_preserve_period_and_model_context(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        summary = read_json(self.output / "report/summary.json")
        links = [row["path"] for row in summary["stage_qa"]]
        self.assertTrue(
            all((self.output / "report" / link).is_file() for link in links)
        )
        for period in PERIOD_IDS:
            for stage in ("training", "working_point"):
                self.assertTrue(
                    any(f"periods/{period}/{stage}/qa/" in link for link in links)
                )
        with ROOT.TFile.Open(
            str(self.output / "report/analysis.root"), "READ"
        ) as archive:
            self.assertFalse(archive.IsZombie())
            for period in PERIOD_IDS:
                training = archive.Get(f"periods/{period}/training/score_signal")
                wp = archive.Get(
                    f"periods/{period}/working_point/significance_vs_score"
                )
                canvas = archive.Get(
                    f"canvases/working_point__p{len(period)}_{period}__global"
                )
                self.assertTrue(training and wp and canvas, period)
                metrics = read_json(self.child(period) / "training/metrics.json")
                self.assertEqual(
                    training.Integral(), metrics["split"]["1"]["validation_rows"]
                )
                self.assertGreater(canvas.GetListOfPrimitives().GetSize(), 0)
            self.assertEqual(
                archive.GetDirectory("nominal").GetListOfKeys().GetSize(), 8
            )

    def test_verified_resume_preserves_all_manifest_mtimes(self):
        before = manifest_snapshot(self.output)
        result = self._run_cli("run", self.config_path)
        self.assertEqual(manifest_snapshot(self.output), before)
        self.assertIn("Reuse verified", result.stderr)

    def test_changing_one_period_seed_reuses_the_other_periods(self):
        with self.isolated_mutation():
            before = manifest_snapshot(self.output)
            settings = copy.deepcopy(self.settings)
            settings["periods"][1]["seed"] = 87654
            write_config(self.config_path, settings)
            self._run_cli("run", self.config_path)
            after = manifest_snapshot(self.output)
            for path, old in before.items():
                if path.startswith(("periods/period_a/", "periods/period_c/")):
                    self.assertEqual(after[path], old, path)
            for path in (
                "periods/period_b/training/manifest.json",
                "training/manifest.json",
                "nominal/manifest.json",
                "report/manifest.json",
            ):
                self.assertNotEqual(
                    after[path]["signature"], before[path]["signature"], path
                )

    def test_tampered_child_histograms_invalidate_parent_and_are_repaired(self):
        from pih3l.config import load_config
        from pih3l.multi_period import MultiPeriodPipeline

        with self.isolated_mutation():
            target = self.child("period_b") / "nominal/results.json"
            original = read_json(target)
            damaged = copy.deepcopy(original)
            key = next(iter(damaged))
            damaged[key]["regions"]["signal"]["se"][0] += 1000
            target.write_text(json.dumps(damaged))
            pipeline = MultiPeriodPipeline(load_config(self.config_path))
            self.assertFalse(pipeline.is_current("nominal"))
            self.assertFalse(pipeline.is_current("report"))
            self._run_cli("run", self.config_path)
            self.assertEqual(read_json(target), original)
            parent = read_json(self.output / "nominal/results.json")
            np.testing.assert_array_equal(
                parent[key]["regions"]["signal"]["se"],
                3 * np.asarray(original[key]["regions"]["signal"]["se"]),
            )

    def test_train_and_apply_commands_only_produce_period_scoped_rows(self):
        output = self.base / "staged-output"
        path = self.base / "staged-periods.yaml"
        settings = copy.deepcopy(self.settings)
        settings["output"] = str(output)
        write_config(path, settings)
        self._run_cli("train", path)
        for period in PERIOD_IDS:
            child = output / "periods" / period
            self.assertTrue((child / "training/model_handler.pkl").is_file())
            self.assertFalse((child / "scored").exists())
        self.assertFalse((output / "nominal").exists())
        self.assertFalse((output / "report").exists())
        self._run_cli("apply", path)
        for period in PERIOD_IDS:
            self.assertTrue(
                list((output / "periods" / period / "scored/femto").rglob("*.parquet"))
            )
        self.assertFalse(list((output / "scored").rglob("*.parquet")))
        self.assertFalse((output / "nominal").exists())


if __name__ == "__main__":
    unittest.main()
