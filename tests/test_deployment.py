"""Installed-wheel closure outside the checkout, with native ROOT dependencies.

The isolated application environment inherits the current analysis environment's
dependencies (including PyROOT). This tests a fresh application installation,
not provisioning an entirely new OS/Conda environment or a 50 GB production run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import uproot
import yaml

from synthetic_data import create_test_inputs, write_test_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "raw_qa",
    "prepared",
    "training",
    "working_point",
    "scored",
    "nominal",
    "systematics",
    "mc_qa",
    "report",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InstalledPackageDeploymentTests(unittest.TestCase):
    """Exercise the distributed package, never the checkout's PYTHONPATH."""

    @classmethod
    def command(cls, arguments, *, timeout=180):
        result = subprocess.run(
            list(map(str, arguments)),
            cwd=cls.foreign_cwd,
            env=cls.environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise AssertionError(
                f"Command failed ({result.returncode}): {arguments}\n"
                + result.stdout
                + result.stderr
            )
        return result

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="pih3l-installed-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name).resolve()
        cls.foreign_cwd = cls.base / "farm_job"
        cls.foreign_cwd.mkdir()
        cls.environment = os.environ.copy()
        cls.environment.pop("PYTHONPATH", None)
        cls.environment.pop("PYTHONHOME", None)
        cls.environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "1",
                "MPLCONFIGDIR": str(cls.base / "mpl"),
            }
        )
        # Build from a snapshot so setuptools cannot mutate shared egg-info or
        # race other tests/reviewers in the actual checkout.
        cls.snapshot = cls.base / "release_source"
        cls.snapshot.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
            if (PROJECT_ROOT / name).is_file():
                shutil.copy2(PROJECT_ROOT / name, cls.snapshot / name)
        shutil.copytree(
            PROJECT_ROOT / "src/pih3l",
            cls.snapshot / "src/pih3l",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in ("configs", "docs", "scripts", "tests", ".github"):
            shutil.copytree(
                PROJECT_ROOT / name,
                cls.snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        cls.command(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                cls.base / "wheels",
                cls.snapshot,
            ]
        )
        wheels = list((cls.base / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"Expected one built application wheel: {wheels}")
        cls.wheel = wheels[0]
        cls.venv = cls.base / "application_env"
        cls.command([sys.executable, "-m", "venv", "--system-site-packages", cls.venv])
        cls.python = cls.venv / "bin/python"
        cls.cli = cls.venv / "bin/pih3l"
        cls.command(
            [
                cls.python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                cls.wheel,
            ]
        )
        cls.data = cls.base / "four_independent_inputs"
        cls.row_count = create_test_inputs(cls.data)
        # Nonempty pair MC covers truth QA and same-collision semantics in the
        # installed application, not only direct calls to the source module.
        with uproot.open(cls.data / "Femto/AO2D.root") as archive:
            tree = archive["DF_1/O2pihypfemto"]
            mc = {name: tree[name].array(library="np") for name in tree.keys()}
        mc.update(
            {
                "fHypIsSignal": np.ones(cls.row_count, bool),
                "fHadIsPhysicalPrimary": np.ones(cls.row_count, bool),
                "fHadIsTruePion": np.ones(cls.row_count, bool),
                "fSameMCCollision": ~mc["fIsMixed"],
                "fIsTruthSelfCorrelation": np.zeros(cls.row_count, bool),
                "fKstarMC": np.tile(np.repeat(np.arange(0.025, 0.3, 0.05), 120), 8),
            }
        )
        with uproot.recreate(cls.data / "MCFemto/AO2D.root") as archive:
            archive["DF_1/O2pihypfemtomc"] = mc
        cls.config_path = cls.foreign_cwd / "analysis.yaml"
        cls.output = cls.foreign_cwd / "result"
        write_test_config(cls.config_path, cls.data, cls.output)
        cls.config = yaml.safe_load(cls.config_path.read_text())
        cls.config.pop("data_root")
        cls.config["periods"] = []
        cls.config["inputs"] = {
            role: {
                "root": str(cls.data / directory),
                "patterns": ["AO2D*.root"],
                "required": role != "mc_femto",
            }
            for role, directory in (
                ("femto", "Femto"),
                ("background", "Hyper"),
                ("signal", "MCHyper"),
                ("mc_femto", "MCFemto"),
            )
        }
        cls.config["io"].update(
            {"persist_prepared": False, "cache_model_scores": False}
        )
        cls.config_path.write_text(yaml.safe_dump(cls.config, sort_keys=False))
        cls.initial_run = cls.command([cls.cli, "run", "-c", cls.config_path])

    def read(self, relative):
        return json.loads((self.output / relative).read_text())

    def test_fresh_wheel_import_and_console_entrypoint_outside_checkout(self):
        probe = self.command(
            [
                self.python,
                "-I",
                "-c",
                "import json,pih3l,sys; from importlib.metadata import distribution; "
                "print(json.dumps({'module':pih3l.__file__,'prefix':sys.prefix,"
                "'entrypoints':[(x.name,x.value) for x in distribution('pih3l-femto').entry_points]}))",
            ]
        )
        identity = json.loads(probe.stdout)
        installed = Path(identity["module"]).resolve()
        self.assertTrue(installed.is_relative_to(self.venv))
        self.assertFalse(installed.is_relative_to(PROJECT_ROOT))
        self.assertEqual(Path(identity["prefix"]).resolve(), self.venv)
        self.assertIn(["pih3l", "pih3l.cli:main"], identity["entrypoints"])
        self.assertNotIn("PYTHONPATH", self.environment)
        self.assertTrue(self.cli.is_file())

    def test_installed_doctor_checks_environment_without_a_config(self):
        result = self.command([self.cli, "doctor", "--json"])
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["environment"])
        self.assertTrue(report["checks"])
        self.assertFalse(any(row["status"] == "failed" for row in report["checks"]))

    def test_installed_doctor_preflights_all_four_roles_without_running_stages(self):
        protected = [self.output / stage / "manifest.json" for stage in STAGES]
        protected += list((self.output / "training").glob("*.pkl"))
        before = {path: (sha256(path), path.stat().st_mtime_ns) for path in protected}
        result = self.command([self.cli, "doctor", "-c", self.config_path, "--json"])
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            set(report["inputs"]), {"femto", "background", "signal", "mc_femto"}
        )
        self.assertEqual(Path(report["config"]["output"]), self.output)
        self.assertEqual(report["config"]["mode"], "single")
        self.assertEqual(
            before,
            {path: (sha256(path), path.stat().st_mtime_ns) for path in protected},
        )

    def test_installed_doctor_optional_mc_is_warning_and_missing_cut_is_failure(self):
        config = yaml.safe_load(self.config_path.read_text())
        output = self.base / "doctor_only_output"
        config["output"] = str(output)
        config["inputs"]["mc_femto"] = {
            "root": None,
            "patterns": ["AO2D*.root"],
            "required": False,
        }
        path = self.foreign_cwd / "doctor_optional.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        result = self.command([self.cli, "doctor", "-c", path, "--json"])
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(any(row["status"] == "warning" for row in report["checks"]))
        config["systematics"]["variations"].append(
            {
                "name": "missing_schema",
                "group": "missing_schema",
                "overrides": {
                    "analysis": {"pair_cuts": {"MissingTrackColumn": {"min": 1}}}
                },
            }
        )
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        failed = subprocess.run(
            [str(self.cli), "doctor", "-c", str(path), "--json"],
            cwd=self.foreign_cwd,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        report = json.loads(failed.stdout)
        self.assertEqual(report["status"], "failed")
        self.assertIn("MissingTrackColumn", failed.stdout)
        self.assertFalse(list(output.rglob("manifest.json")))
        self.assertFalse(list(output.rglob("*.pkl")))

    def test_wheel_contains_all_runtime_modules_and_no_analysis_data(self):
        with zipfile.ZipFile(self.wheel) as archive:
            names = set(archive.namelist())
        expected = {
            "pih3l/" + str(path.relative_to(PROJECT_ROOT / "src/pih3l"))
            for path in (PROJECT_ROOT / "src/pih3l").rglob("*.py")
        }
        self.assertTrue(expected <= names, expected - names)
        forbidden = {".root", ".parquet", ".pkl", ".npz", ".log"}
        self.assertFalse([name for name in names if Path(name).suffix in forbidden])
        self.assertFalse(
            [name for name in names if set(Path(name).parts) & {"outputs", "reports"}]
        )

    def test_installed_single_period_recovers_raw_and_corrected_unity(self):
        self.assertFalse((self.output / "periods").exists())
        self.assertFalse(list((self.output / "prepared").rglob("*.parquet")))
        self.assertFalse(list((self.output / "training").glob("*score*cache*.npz")))
        nominal = self.read("nominal/results.json")
        self.assertEqual(len(nominal), 8)
        for key, item in nominal.items():
            with self.subTest(bin=key):
                self.assertGreater(sum(item["regions"]["sideband"]["se"]), 0)
                for source in (item["correlations"]["signal"], item["corrected"]):
                    self.assertEqual(source["status"], "ok")
                    self.assertTrue(all(source["valid"]))
                    np.testing.assert_allclose(source["values"], 1, atol=1e-12)
                    covariance = np.asarray(source["covariance"])
                    self.assertTrue(np.isfinite(covariance).all())
                    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)

    def test_installed_working_point_systematics_and_truth_qa_are_complete(self):
        working_point = self.read("working_point/summary.json")
        self.assertEqual(working_point["status"], "ok")
        self.assertEqual(
            self.read("nominal/cutflow.json")["threshold"], working_point["threshold"]
        )
        variations = self.read("systematics/summary.json")["variations"]
        self.assertEqual(
            {row["name"] for row in variations}, {"normalization_alt", "bdt_alt"}
        )
        self.assertTrue(all(row["status"] == "completed" for row in variations))
        mc = self.read("mc_qa/summary.json")
        self.assertEqual(mc["status"], "available")
        self.assertEqual(mc["truth_closure_status"], "available")
        self.assertEqual(len(mc["bins"]), 16)
        for item in mc["bins"].values():
            fraction = item["primary_pion_fraction"]
            values = np.asarray(fraction["value"], float)
            self.assertTrue(np.asarray(fraction["valid"]).all())
            np.testing.assert_allclose(values, 1, atol=1e-12)
        truth = self.read("mc_qa/truth_closure/results.json")
        self.assertTrue(
            all(sum(item["regions"]["signal"]["me"]) > 0 for item in truth.values())
        )

    def test_installed_reports_are_pdf_and_root_with_corrected_graphs(self):
        for directory in (
            "raw_qa",
            "training",
            "working_point",
            "systematics",
            "report",
        ):
            files = list((self.output / directory).rglob("*.pdf"))
            self.assertTrue(files, directory)
            self.assertTrue(
                all(path.read_bytes().startswith(b"%PDF-") for path in files)
            )
        self.assertFalse(
            [
                path
                for path in self.output.rglob("*")
                if path.suffix.lower() in {".png", ".jpg", ".svg"}
            ]
        )
        probe = self.command(
            [
                self.python,
                "-I",
                "-c",
                "import ROOT,json; ROOT.gROOT.SetBatch(True); "
                f"a=ROOT.TFile.Open({str(self.output / 'report/analysis.root')!r}); "
                "h=a.Get('nominal/cent_0_10__ch_0/cf_corrected_stat'); "
                "g=a.Get('nominal/cent_0_10__ch_0/g_cf_corrected_stat'); "
                "c=a.Get('canvases/correlations_cent0'); "
                "print(json.dumps({'zombie':a.IsZombie(),'values':[h.GetBinContent(i+1) for i in range(h.GetNbinsX())],"
                "'points':g.GetN(),'primitives':c.GetListOfPrimitives().GetSize()})); a.Close()",
            ]
        )
        result = json.loads(probe.stdout)
        self.assertFalse(result["zombie"])
        self.assertEqual(result["points"], 6)
        self.assertGreater(result["primitives"], 0)
        np.testing.assert_allclose(result["values"], 1, atol=1e-12)

    def test_installed_cached_resume_preserves_all_manifests(self):
        paths = [self.output / stage / "manifest.json" for stage in STAGES]
        before = {path: (sha256(path), path.stat().st_mtime_ns) for path in paths}
        result = self.command([self.cli, "run", "-c", self.config_path])
        self.assertEqual(result.stderr.count("Reuse verified stage"), len(STAGES))
        self.assertEqual(
            before, {path: (sha256(path), path.stat().st_mtime_ns) for path in paths}
        )

    def test_installed_optional_mc_can_be_disabled_by_config_only(self):
        optional_output = self.base / "optional_mc_output"
        shutil.copytree(self.output, optional_output)
        config = yaml.safe_load(self.config_path.read_text())
        config["output"] = str(optional_output)
        config["inputs"]["mc_femto"] = {
            "root": None,
            "patterns": ["AO2D*.root"],
            "required": False,
        }
        path = self.foreign_cwd / "without_pair_mc.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        self.command([self.cli, "mc-qa", "-c", path])
        result = json.loads((optional_output / "mc_qa/summary.json").read_text())
        self.assertEqual(result["status"], "skipped_missing_mc_femto")
        self.assertFalse(result.get("bins"))
        self.assertFalse((optional_output / "mc_qa/truth_closure").exists())

    def test_installed_tamper_detection_rebuilds_downstream_only(self):
        damaged_output = self.base / "tampered_output"
        shutil.copytree(self.output, damaged_output)
        config = yaml.safe_load(self.config_path.read_text())
        config["output"] = str(damaged_output)
        path = self.foreign_cwd / "repair.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        # Establish this output location before testing tamper recovery; cache
        # relocation is not the contract under test here.
        self.command([self.cli, "run", "-c", path])
        training_manifest = damaged_output / "training/manifest.json"
        before = (sha256(training_manifest), training_manifest.stat().st_mtime_ns)
        expected = (damaged_output / "nominal/results.json").read_bytes()
        (damaged_output / "nominal/cutflow.json").write_text("{}\n")
        self.command([self.cli, "run", "-c", path])
        self.assertEqual(
            before, (sha256(training_manifest), training_manifest.stat().st_mtime_ns)
        )
        self.assertIn(
            "counts", json.loads((damaged_output / "nominal/cutflow.json").read_text())
        )
        self.assertEqual(
            expected, (damaged_output / "nominal/results.json").read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
