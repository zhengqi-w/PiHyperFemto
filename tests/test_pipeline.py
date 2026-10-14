"""Full CLI closure on real synthetic ROOT inputs, plus cache/integrity behavior."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pih3l.errors import AnalysisError
from pih3l.pipeline import run_lock

from synthetic_data import create_test_inputs, write_test_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_output_lock_prevents_competing_writers(self):
        with tempfile.TemporaryDirectory() as temporary:
            with run_lock(temporary):
                with self.assertRaises(AnalysisError):
                    with run_lock(temporary):
                        self.fail("A concurrent output writer acquired the lock")
            with run_lock(temporary):
                pass

    def test_cli_end_to_end_eight_groups_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            row_count = create_test_inputs(base / "input")
            config = base / "analysis.yaml"
            output = base / "output"
            write_test_config(config, base / "input", output)
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONPATH": str(PROJECT_ROOT / "src"),
                    "OMP_NUM_THREADS": "2",
                }
            )
            command = [sys.executable, "-m", "pih3l", "run", "-c", str(config)]
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            import ROOT

            ROOT.gROOT.SetBatch(True)
            root_path = output / "report/analysis.root"
            self.assertTrue(root_path.is_file())
            with ROOT.TFile.Open(str(root_path), "READ") as root_archive:
                self.assertFalse(root_archive.IsZombie())
                self.assertEqual(
                    root_archive.GetDirectory("nominal").GetListOfKeys().GetSize(), 8
                )
                canvas = root_archive.Get("canvases/correlations_cent0")
                self.assertTrue(canvas and canvas.InheritsFrom("TCanvas"))
                self.assertGreater(canvas.GetListOfPrimitives().GetSize(), 0)
            prepared = json.loads((output / "prepared/summary.json").read_text())
            self.assertEqual(prepared["femto"]["raw_rows"], row_count)
            self.assertEqual(prepared["mc_femto"]["status"], "missing_optional")
            working_point = json.loads(
                (output / "working_point/summary.json").read_text()
            )
            self.assertEqual(working_point["status"], "ok")
            self.assertEqual(
                working_point["threshold_source"],
                "hyper_data_pol0_expected_significance",
            )
            self.assertGreater(working_point["best_scan_point"]["significance"], 0)
            cutflow = json.loads((output / "nominal/cutflow.json").read_text())
            self.assertEqual(cutflow["threshold"], working_point["threshold"])
            nominal = json.loads((output / "nominal/results.json").read_text())
            self.assertEqual(len(nominal), 8)
            for name, item in nominal.items():
                with self.subTest(bin=name):
                    self.assertEqual(item["correlations"]["signal"]["status"], "ok")
                    self.assertEqual(item["corrected"]["status"], "ok")
                    np.testing.assert_allclose(
                        item["correlations"]["signal"]["values"], 1, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        item["corrected"]["values"], 1, atol=1e-12
                    )
            final_table = pd.read_csv(output / "systematics/final_observables.csv")
            self.assertEqual(len(final_table), 8 * 6 * 2)
            np.testing.assert_allclose(
                final_table.total_error**2,
                final_table.stat_error**2 + final_table.syst_error**2,
                atol=1e-12,
            )
            covariance_export = np.load(output / "systematics/covariances.npz")
            self.assertEqual(len(covariance_export.files), 8 * 2 * 2)
            for name, item in nominal.items():
                for observable in ["raw", "corrected"]:
                    source = (
                        item["correlations"]["signal"]
                        if observable == "raw"
                        else item["corrected"]
                    )
                    key = name + "__" + observable
                    np.testing.assert_allclose(
                        covariance_export[key + "__total"],
                        np.asarray(source["covariance"])
                        + covariance_export[key + "__systematic"],
                        atol=1e-12,
                    )
            acceptance = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/validate_testset.py"),
                    "--config",
                    str(config),
                    "--report",
                    str(base / "acceptance.json"),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                acceptance.returncode, 0, msg=acceptance.stdout + acceptance.stderr
            )
            self.assertEqual(
                json.loads((base / "acceptance.json").read_text())["status"], "passed"
            )
            manifest = output / "prepared/manifest.json"
            original_mtime = manifest.stat().st_mtime_ns
            repeated = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                repeated.returncode, 0, msg=repeated.stdout + repeated.stderr
            )
            self.assertEqual(manifest.stat().st_mtime_ns, original_mtime)
            self.assertIn("Reuse verified stage", repeated.stderr)
            # A tampered small artifact must invalidate its stage and be reconstructed.
            summary = output / "nominal/cutflow.json"
            summary.write_text("{}\n")
            rebuilt = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(rebuilt.returncode, 0, msg=rebuilt.stdout + rebuilt.stderr)
            self.assertIn("counts", json.loads(summary.read_text()))
            self.assertEqual(manifest.stat().st_mtime_ns, original_mtime)


if __name__ == "__main__":
    unittest.main()
