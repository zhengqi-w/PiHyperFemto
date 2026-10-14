"""Large-data cache isolation, strict projection and failure recovery contracts."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pih3l.config import load_config
from pih3l.errors import AnalysisError
from pih3l.io import (
    iter_parquet,
    parquet_columns,
    read_json,
    sha256_file,
    write_batch,
    write_json,
)
from pih3l.pipeline import Pipeline
from pih3l.score_cache import cached_model_scores, score_key

from synthetic_data import create_test_inputs, write_test_config


class ExecutionIOTests(unittest.TestCase):
    def test_score_vectors_invalidate_on_dependency_environment_change(self):
        source = {"sha256": "source", "start": 0, "rows": 2}
        spec = {
            "pt_range": [2, 4],
            "inference_checksum": "model",
            "inference_code": {"models": "code"},
            "inference_environment": {"numpy": "1"},
        }
        changed = copy.deepcopy(spec)
        changed["inference_environment"]["numpy"] = "2"
        self.assertNotEqual(score_key(source, spec), score_key(source, changed))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        create_test_inputs(self.base / "input")
        path = self.base / "config.yaml"
        write_test_config(path, self.base / "input", self.base / "output")
        self.config = load_config(path)
        self.pipeline = Pipeline(self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def changed(self, mutate):
        config = copy.deepcopy(self.config)
        mutate(config)
        # Source inventory and environment are fixed for these dependency tests.
        result = copy.copy(self.pipeline)
        result.config = config
        return result

    def test_wp_and_existing_cut_variations_do_not_invalidate_large_tables(self):
        changed = self.changed(
            lambda c: c["working_point"].update(signal_efficiencies=[0.6, 0.8])
        )
        for stage in ("prepared", "training", "scored"):
            self.assertEqual(
                self.pipeline.stage_signature(stage), changed.stage_signature(stage)
            )
        self.assertNotEqual(
            self.pipeline.stage_signature("working_point"),
            changed.stage_signature("working_point"),
        )
        changed = self.changed(
            lambda c: c["systematics"]["variations"][1]["overrides"]["analysis"].update(
                bdt_threshold=0.15
            )
        )
        for stage in ("prepared", "training", "scored"):
            self.assertEqual(
                self.pipeline.stage_signature(stage), changed.stage_signature(stage)
            )
        self.assertNotEqual(
            self.pipeline.stage_signature("systematics"),
            changed.stage_signature("systematics"),
        )

    def test_pt_scope_and_feature_selection_reuse_preparation(self):
        changed = self.changed(
            lambda c: c["training"].update(pt_range=[2, 8], features=["DcaPi"])
        )
        self.assertEqual(
            self.pipeline.stage_signature("prepared"),
            changed.stage_signature("prepared"),
        )
        self.assertNotEqual(
            self.pipeline.stage_signature("training"),
            changed.stage_signature("training"),
        )

    def test_nominal_threshold_updates_wp_without_rescoring(self):
        changed = self.changed(lambda c: c["analysis"].update(bdt_threshold=0.3))
        self.assertEqual(
            self.pipeline.stage_signature("scored"), changed.stage_signature("scored")
        )
        self.assertNotEqual(
            self.pipeline.stage_signature("working_point"),
            changed.stage_signature("working_point"),
        )

    def test_pair_range_change_does_not_change_candidate_cache_scope(self):
        changed = self.changed(lambda c: c["analysis"].update(mass_range=[2.94, 3.06]))
        for role in ("background", "signal"):
            self.assertEqual(
                self.pipeline.stage_signature("prepared", role=role),
                changed.stage_signature("prepared", role=role),
            )
        self.assertNotEqual(
            self.pipeline.stage_signature("prepared", role="femto"),
            changed.stage_signature("prepared", role="femto"),
        )

    def test_new_variation_column_invalidates_required_projection(self):
        changed = self.changed(
            lambda c: c["systematics"]["variations"][1]["overrides"]["analysis"].update(
                pair_cuts={"PairPiTrackDcaXY": {"abs_max": 0.1}}
            )
        )
        self.assertNotEqual(
            self.pipeline.stage_signature("prepared"),
            changed.stage_signature("prepared"),
        )
        self.assertNotEqual(
            self.pipeline.stage_signature("scored"), changed.stage_signature("scored")
        )

    def test_preparation_resumes_without_decoding_completed_root_ranges(self):
        from pih3l.pipeline import read_root_shard

        calls = []

        def interrupted(config, role, shard, **options):
            calls.append((role, shard["start"]))
            if len(calls) == 3:
                raise RuntimeError("simulated process interruption")
            return read_root_shard(config, role, shard, **options)

        with patch("pih3l.pipeline.read_root_shard", side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                self.pipeline._prepared(self.base / "first")
        resumed_calls = []

        def resumed(config, role, shard, **options):
            resumed_calls.append((role, shard["start"]))
            return read_root_shard(config, role, shard, **options)

        with patch("pih3l.pipeline.read_root_shard", side_effect=resumed):
            self.pipeline._prepared(self.base / "resumed")
        self.assertNotIn(("femto", 0), resumed_calls)
        self.assertNotIn(("femto", 250), resumed_calls)
        self.assertIn(("femto", 500), resumed_calls)
        self.assertEqual(
            read_json(self.base / "resumed/summary.json")["femto"]["raw_rows"], 5760
        )
        first = self.base / "first/femto/part-000000.parquet"
        second = self.base / "resumed/femto/part-000000.parquet"
        self.assertEqual(sha256_file(first), sha256_file(second))
        self.assertEqual(first.stat().st_ino, second.stat().st_ino)

    def test_corrupted_checkpoint_is_recomputed(self):
        checkpoint = self.pipeline._checkpoint("prepared", "femto")
        artifact = write_batch(self.base / "data", 0, pd.DataFrame({"x": [1]}))
        self.pipeline._save_shard(checkpoint, 0, artifact, {"start": 0}, {"rows": 1})
        (checkpoint / artifact.name).write_bytes(b"corrupted")
        self.assertIsNone(
            self.pipeline._restore_shard(checkpoint, 0, self.base / "restore")
        )

    def test_projection_requires_explicit_columns_and_accepts_optional_missing(self):
        directory = self.base / "projection"
        write_batch(directory, 0, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertEqual(parquet_columns(directory), {"a", "b"})
        frames = list(
            iter_parquet(
                directory, columns=["a"], optional_columns=["missing"], batch_size=1
            )
        )
        self.assertEqual(sum(len(frame) for frame in frames), 2)
        self.assertEqual(list(frames[0]), ["a"])
        with self.assertRaises(AnalysisError):
            list(iter_parquet(directory, columns=["missing"]))
        write_batch(directory, 1, pd.DataFrame({"a": [1]}))
        with self.assertRaises(AnalysisError):
            parquet_columns(directory)

    def test_per_model_scores_reuse_unchanged_model_after_local_retrain(self):
        frame = pd.DataFrame({"pt_h3l": [1.0, 3.0, 9.0]})
        registry = {
            "models": [
                {
                    "id": "low",
                    "pt_range": [0, 2],
                    "inference_checksum": "low-v1",
                    "inference_code": "same-code",
                },
                {
                    "id": "high",
                    "pt_range": [2, 8],
                    "inference_checksum": "high-v1",
                    "inference_code": "same-code",
                },
            ]
        }
        predictions = []

        def predict(sample, subset):
            model_id = subset["models"][0]["id"]
            predictions.append(model_id)
            return sample.assign(bdt_score=0.2 if model_id == "low" else 0.8)

        options = {
            "checkpoint_root": self.base / "scores",
            "source": {"sha256": "input1"},
            "active_paths": set(),
        }
        with patch("pih3l.models.score_models", side_effect=predict):
            first = cached_model_scores(frame, registry, **options)
            registry["models"][1]["inference_checksum"] = "high-v2"
            second = cached_model_scores(frame, registry, **options)
        self.assertEqual(predictions, ["low", "high", "high"])
        np.testing.assert_allclose(first.bdt_score, second.bdt_score, equal_nan=True)
        self.assertFalse(second.bdt_in_scope.iloc[-1])
        self.assertTrue(np.isnan(second.bdt_score.iloc[-1]))

    def test_committed_artifact_integrity_remains_strict(self):
        directory = self.pipeline.output / "nominal"
        directory.mkdir(parents=True)
        write_json(directory / "results.json", {"value": 1})
        write_json(
            directory / "manifest.json",
            {
                "signature": self.pipeline.stage_signature("nominal"),
                "artifacts": {"results.json": sha256_file(directory / "results.json")},
            },
        )
        self.assertTrue(self.pipeline.is_current("nominal"))
        write_json(directory / "results.json", {"value": 2})
        self.assertFalse(self.pipeline.is_current("nominal"))


if __name__ == "__main__":
    unittest.main()
