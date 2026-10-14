"""Direct ROOT orchestration, minimal persistence and nested shard recovery."""

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import uproot
import yaml

from pih3l.config import load_config
from pih3l.io import read_json, write_batch, write_json
from pih3l.pipeline import Pipeline

from synthetic_data import create_test_inputs, write_test_config


class StreamingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        create_test_inputs(self.base / "input")
        path = self.base / "config.yaml"
        write_test_config(path, self.base / "input", self.base / "output")
        self.config = load_config(path)
        self.config["io"]["persist_prepared"] = False
        self.pipeline = Pipeline(self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def test_preparation_is_metadata_only_and_does_not_decode_root(self):
        destination = self.base / "metadata"
        with patch(
            "pih3l.dataset.RootDataset.read_shard",
            side_effect=AssertionError("ROOT decoded"),
        ):
            self.pipeline._prepared(destination)
        self.assertFalse(list(destination.rglob("*.parquet")))
        self.assertFalse((self.pipeline.output / ".checkpoints/prepared").exists())
        summary = read_json(destination / "summary.json")
        self.assertEqual(summary["femto"]["raw_rows"], 5760)
        self.assertIsNone(summary["femto"]["retained_rows"])
        self.assertFalse(summary["signal"]["persisted"])
        self.assertEqual(summary["signal"]["source_identity"]["kind"], "streamed_root")

    def test_training_and_wp_receive_replayable_root_sources(self):
        from pih3l.dataset import RootDataset

        with patch("pih3l.models.train_models") as train:
            self.pipeline._training(self.base / "training")
        self.assertIsInstance(train.call_args.args[0], RootDataset)
        self.assertIsInstance(train.call_args.args[1], RootDataset)
        with patch("pih3l.working_point.scan_model_working_points") as scan:
            self.pipeline._working_point(self.base / "wp")
        self.assertIsInstance(scan.call_args.args[0], RootDataset)
        # Compatibility mode still receives filesystem caches.
        self.pipeline.config = copy.deepcopy(self.config)
        self.pipeline.config["io"]["persist_prepared"] = True
        self.assertIsInstance(self.pipeline._candidate_source("signal"), Path)

    def test_optional_its_output_decodes_packed_root_column_before_compact_write(self):
        from pih3l.dataset import RootDataset
        from pih3l.scored_storage import write_scored_batch

        path = self.base / "input/Femto/AO2D.root"
        with uproot.open(path) as root:
            data = root["DF_1/O2pihypfemto"].arrays(library="np")
        count = len(next(iter(data.values())))
        data["fHypItsClusterSizesHe"] = np.full(count, 0x7654321, np.uint32)
        with uproot.recreate(path) as root:
            root["DF_1/O2pihypfemto"] = data
        self.config["io"]["scored_optional_columns"].append("its_mean_size_he")
        frame = next(RootDataset(self.config, "femto").iter_batches())
        np.testing.assert_array_equal(frame.its_mean_size_he, 4.0)
        frame["bdt_score"] = 0.9
        paths, _ = write_scored_batch(
            self.base / "compact", 0, frame, self.config, {}, "femto"
        )
        saved = pd.read_parquet(paths[0])
        np.testing.assert_array_equal(saved.its_mean_size_he, 4.0)
        self.assertNotIn("ItsClusterSizesHe", saved.columns)

    def test_source_enumeration_is_lazy_and_inventory_not_repeated_per_shard(self):
        from pih3l.dataset import RootDataset

        iterator = self.pipeline._score_inputs("femto")
        with patch.object(
            RootDataset, "read_shard", side_effect=AssertionError("decoded too early")
        ):
            index, source, reader = next(iterator)
            self.assertEqual(index, 0)
            self.assertIsInstance(source["dataset"], str)
            self.assertEqual(len(source["dataset"]), 64)
            self.assertNotIn("inventory", source)
            with self.assertRaisesRegex(AssertionError, "decoded too early"):
                reader()
        iterator.close()

    def test_multiple_pt_partition_artifacts_restore_all_or_none(self):
        role = self.base / "first/femto"
        paths = [
            write_batch(role / label, 0, pd.DataFrame({"bdt_score": [0.9]}))
            for label in ("pt2_4", "pt4_8")
        ]
        checkpoint = self.base / "checkpoint"
        checkpoint.mkdir()
        source = {"dataset": "identity", "shard": {"start": 0}}
        counts = {"rows": 2, "retained_rows": 2}
        self.pipeline._save_shard(
            checkpoint, 0, paths, source, counts, artifact_root=role
        )
        restored = self.base / "restored/femto"
        record = self.pipeline._restore_shard(checkpoint, 0, restored, source=source)
        self.assertEqual(record["counts"], counts)
        self.assertEqual(
            set(record["artifacts"]),
            {"pt2_4/part-000000.parquet", "pt4_8/part-000000.parquet"},
        )
        for path in paths:
            relative = path.relative_to(role)
            self.assertEqual(path.stat().st_ino, (restored / relative).stat().st_ino)
        (checkpoint / "pt4_8/part-000000.parquet").write_bytes(b"corrupt")
        rejected = self.base / "rejected"
        self.assertIsNone(
            self.pipeline._restore_shard(checkpoint, 0, rejected, source=source)
        )
        self.assertFalse(list(rejected.rglob("*.parquet")))

    def test_candidate_stream_does_not_inherit_pair_centrality_selection(self):
        from pih3l.dataset import RootDataset
        from pih3l.io import iter_parquet

        config = copy.deepcopy(self.config)
        config["analysis"]["centrality_bins"] = [[0, 10]]
        dataset = RootDataset(config, "background", self.pipeline.inputs["background"])
        frames = list(dataset.iter_batches(columns=["HypCentFT0C", "mass_h3l"]))
        self.assertEqual(sum(len(frame) for frame in frames), 6400)
        self.assertEqual(
            sum(int((frame.HypCentFT0C == 25).sum()) for frame in frames), 3200
        )
        self.assertNotIn("centrality_bins", dataset.fingerprint()["domain"])
        # The optional persisted path must use the same candidate domain.
        config["io"].update(persist_prepared=True, checkpoint=False)
        persisted = Pipeline(config)
        destination = self.base / "persisted"
        persisted._prepared(destination)
        for role in ("signal", "background"):
            direct = pd.concat(
                RootDataset(config, role, persisted.inputs[role]).iter_batches(
                    columns=["HypCentFT0C", "mass_h3l"]
                ),
                ignore_index=True,
            )
            cached = pd.concat(iter_parquet(destination / role), ignore_index=True)
            self.assertEqual(len(cached), len(direct))
            pd.testing.assert_series_equal(
                cached.HypCentFT0C.value_counts().sort_index(),
                direct.HypCentFT0C.value_counts().sort_index(),
            )
            self.assertTrue((cached.HypCentFT0C >= 10).any())
        pairs = pd.concat(iter_parquet(destination / "femto"), ignore_index=True)
        self.assertTrue((pairs.HypCentFT0C < 10).all())

    def test_scored_cache_binds_wp_storage_floor_instead_of_nominal_qa(self):
        self.config["working_point"]["storage"]["enabled"] = True
        directory = self.pipeline.output / "working_point"
        directory.mkdir(parents=True)
        manifest = {
            "signature": "wp-v1",
            "config": self.config,
            "artifacts": {"summary.json": "checksum-v1"},
        }
        write_json(directory / "manifest.json", manifest)
        write_json(
            directory / "summary.json", {"threshold": 0.7, "storage_threshold": 0.1}
        )
        original = self.pipeline.stage_signature("scored")
        manifest.update(signature="wp-v2", artifacts={"summary.json": "checksum-v2"})
        write_json(directory / "manifest.json", manifest)
        write_json(
            directory / "summary.json", {"threshold": 0.6, "storage_threshold": 0.1}
        )
        self.assertEqual(self.pipeline.stage_signature("scored"), original)
        write_json(
            directory / "summary.json", {"threshold": 0.6, "storage_threshold": 0.05}
        )
        self.assertNotEqual(self.pipeline.stage_signature("scored"), original)

    def test_direct_stream_full_workflow_persists_only_compact_scored_pairs(self):
        config_path = self.base / "stream.yaml"
        output = self.base / "stream_output"
        config = copy.deepcopy(self.config)
        config["output"] = str(output)
        config["io"].update(
            persist_prepared=False,
            cache_model_scores=False,
            scored_partition_by_pt=True,
        )
        config["training"].update(
            pt_bins=[{"id": "pt1_2", "pt_range": [1, 2]}],
            efficiency_min=0.5,
            efficiency_max=0.9,
            npoints_for_effi=9,
        )
        config["working_point"].update(
            signal_efficiencies=None,
            storage={"enabled": True, "reserve_signal_efficiency": 1.0},
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        project = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(project / "src"),
            OMP_NUM_THREADS="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(self.base / "matplotlib"),
        )
        command = [sys.executable, "-m", "pih3l", "run", "-c", str(config_path)]
        result = subprocess.run(
            command,
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(list((output / "prepared").rglob("*.parquet")))
        self.assertFalse((output / ".checkpoints/prepared").exists())
        self.assertFalse((output / ".checkpoints/model_scores").exists())
        parquets = list(output.rglob("*.parquet"))
        self.assertTrue(parquets)
        for path in parquets:
            relative = path.relative_to(output)
            self.assertTrue(
                relative.parts[0] == "scored"
                or relative.parts[:2] == (".checkpoints", "scored"),
                f"Unexpected intermediate table: {relative}",
            )
        self.assertTrue(
            (output / "training/models/pt1_2/validation_scores.npz").exists()
        )
        self.assertTrue((output / "training/models/pt1_2/model_handler.pkl").is_file())
        self.assertFalse(list((output / "training").rglob("*.joblib")))
        paths = list((output / "scored/femto/pt1_2").glob("*.parquet"))
        self.assertTrue(paths)
        self.assertFalse(list((output / "scored/femto").glob("*.parquet")))
        frame = pd.read_parquet(paths[0])
        self.assertTrue(
            {"mass_h3l", "kstar", "pt_h3l", "bdt_score", "bdt_model_id"} <= set(frame)
        )
        self.assertNotIn("PtHe3", frame)
        self.assertNotIn("DcaHe", frame)
        self.assertNotIn("_source_file", frame)
        floor = read_json(output / "working_point/summary.json")["storage_threshold"][
            "pt1_2"
        ]
        self.assertTrue(np.isfinite(frame.bdt_score).all())
        if floor is not None:
            self.assertTrue((frame.bdt_score >= floor).all())
        self.assertEqual(len(read_json(output / "nominal/results.json")), 8)
        self.assertTrue((output / "report/analysis.root").exists())
        original = (output / "scored/manifest.json").stat().st_mtime_ns
        repeated = subprocess.run(
            command,
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
        self.assertEqual((output / "scored/manifest.json").stat().st_mtime_ns, original)


if __name__ == "__main__":
    unittest.main()
