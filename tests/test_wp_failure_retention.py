"""A failed WP must retain diagnostics and never replace a completed stage."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pih3l.config import load_config
from pih3l.errors import AnalysisError
from pih3l.io import read_json, sha256_file, write_json
from pih3l.pipeline import Pipeline

from synthetic_data import create_test_inputs, write_test_config


class WorkingPointFailureRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        create_test_inputs(self.base / "input")
        path = self.base / "config.yaml"
        write_test_config(path, self.base / "input", self.base / "output")
        self.config = load_config(path)

    def tearDown(self):
        self.temporary.cleanup()

    def pipeline(self):
        pipeline = Pipeline(self.config, force=True)
        # This regression isolates the stage transaction, not the fit numerics.
        pipeline.finished.update({"prepared", "training"})
        return pipeline

    @staticmethod
    def fail_after_scan(destination):
        write_json(
            destination / "summary.json",
            {"status": "failed", "yield_pol0": {"status": "failed"}},
        )
        write_json(
            destination / "models/pt2_4/scan_fits.json",
            {"points": [{"threshold": 0.5}]},
        )
        qa = destination / "models/pt2_4/qa"
        qa.mkdir(parents=True)
        (qa / "working_point.root").write_bytes(b"retained ROOT diagnostics")
        (qa / "significance_vs_efficiency.pdf").write_bytes(b"%PDF-retained QA")
        raise AnalysisError(
            f"No valid pol0; inspect {destination / 'models/pt2_4/scan_fits.json'}"
        )

    def assert_retained(self, pipeline, error):
        failure = read_json(pipeline.output / "failures/working_point.json")
        diagnostics = Path(failure["diagnostics_path"])
        self.assertTrue(diagnostics.is_dir())
        self.assertEqual(diagnostics.name, "working_point-" + failure["signature"])
        self.assertEqual(diagnostics.parent, pipeline.output / "failures")
        fits = diagnostics / "models/pt2_4/scan_fits.json"
        self.assertTrue(fits.is_file())
        self.assertIn(str(fits), str(error))
        self.assertIn(str(fits), failure["error"])
        self.assertEqual(
            (diagnostics / "models/pt2_4/qa/working_point.root").read_bytes(),
            b"retained ROOT diagnostics",
        )
        self.assertEqual(
            (
                diagnostics / "models/pt2_4/qa/significance_vs_efficiency.pdf"
            ).read_bytes(),
            b"%PDF-retained QA",
        )
        self.assertFalse(list(pipeline.output.glob(".working_point-*.tmp")))
        self.assertNotIn("working_point", pipeline.finished)
        return failure

    def test_failed_scan_diagnostics_are_retained_at_path_named_in_error(self):
        pipeline = self.pipeline()
        with patch.object(pipeline, "_working_point", side_effect=self.fail_after_scan):
            with self.assertRaises(AnalysisError) as caught:
                pipeline.execute("working_point")
        failure = self.assert_retained(pipeline, caught.exception)
        self.assertFalse(failure["previous_completed_stage_preserved"])
        self.assertFalse((pipeline.output / "working_point").exists())

    def test_previous_success_is_byte_identical_after_failed_rescan(self):
        success = self.pipeline()
        with patch.object(
            success,
            "_working_point",
            side_effect=lambda destination: write_json(
                destination / "summary.json", {"status": "ok", "threshold": 0.4}
            ),
        ):
            success.execute("working_point")
        directory = success.output / "working_point"
        original = {
            str(path.relative_to(directory)): sha256_file(path)
            for path in directory.rglob("*")
            if path.is_file()
        }
        old_manifest = (directory / "manifest.json").stat().st_mtime_ns
        failure_pipeline = self.pipeline()
        with patch.object(
            failure_pipeline, "_working_point", side_effect=self.fail_after_scan
        ):
            with self.assertRaises(AnalysisError) as caught:
                failure_pipeline.execute("working_point")
        failure = self.assert_retained(failure_pipeline, caught.exception)
        self.assertTrue(failure["previous_completed_stage_preserved"])
        self.assertEqual(
            {
                str(path.relative_to(directory)): sha256_file(path)
                for path in directory.rglob("*")
                if path.is_file()
            },
            original,
        )
        self.assertEqual((directory / "manifest.json").stat().st_mtime_ns, old_manifest)
        self.assertTrue(failure_pipeline.is_current("working_point"))

    def test_failure_before_scan_cleans_temporary_directory_without_fake_diagnostics(
        self,
    ):
        pipeline = self.pipeline()
        with patch.object(
            pipeline, "_working_point", side_effect=AnalysisError("No candidates")
        ):
            with self.assertRaisesRegex(AnalysisError, "No candidates"):
                pipeline.execute("working_point")
        failure = read_json(pipeline.output / "failures/working_point.json")
        self.assertIsNone(failure["diagnostics_path"])
        self.assertFalse(list(pipeline.output.glob(".working_point-*.tmp")))


if __name__ == "__main__":
    unittest.main()
