"""Native ROOT persistence and redraw checks in separate Python processes.

The toy sample has an analytic unity correlation in every populated bin. These
checks reopen the serialized file with PyROOT; in-memory objects cannot satisfy
this archive contract by accident.
"""

from pih3l.scored_storage import write_storage_metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import ROOT

from synthetic_data import create_test_inputs, write_test_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT.gROOT.SetBatch(True)


def histogram_values(histogram):
    return np.array(
        [histogram.GetBinContent(index + 1) for index in range(histogram.GetNbinsX())]
    )


def histogram_errors(histogram):
    return np.array(
        [histogram.GetBinError(index + 1) for index in range(histogram.GetNbinsX())]
    )


def matrix_values(histogram):
    return np.array(
        [
            [
                histogram.GetBinContent(i + 1, j + 1)
                for j in range(histogram.GetNbinsY())
            ]
            for i in range(histogram.GetNbinsX())
        ]
    )


class RootOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.output = cls.base / "output"
        create_test_inputs(cls.base / "input")
        cls.config_path = cls.base / "analysis.yaml"
        write_test_config(cls.config_path, cls.base / "input", cls.output)
        cls.environment = os.environ.copy()
        cls.environment.update(
            {"PYTHONPATH": str(PROJECT_ROOT / "src"), "OMP_NUM_THREADS": "2"}
        )
        result = subprocess.run(
            [sys.executable, "-m", "pih3l", "run", "-c", str(cls.config_path)],
            cwd=PROJECT_ROOT,
            env=cls.environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.archive = cls.output / "report/analysis.root"
        cls.nominal = json.loads((cls.output / "nominal/results.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def open_archive(self):
        self.assertTrue(self.archive.is_file())
        archive = ROOT.TFile.Open(str(self.archive), "READ")
        self.assertTrue(archive)
        self.assertFalse(archive.IsZombie())
        return archive

    def require_object(self, archive, path, class_name):
        item = archive.Get(path)
        self.assertTrue(item, msg=f"Missing ROOT object: {path}")
        self.assertTrue(
            item.InheritsFrom(class_name), msg=f"{path}: {item.ClassName()}"
        )
        return item

    def test_nominal_histograms_preserve_values_errors_edges_and_validity(self):
        with self.open_archive() as archive:
            for name, item in self.nominal.items():
                with self.subTest(bin=name):
                    base = "nominal/" + name + "/"
                    for event in ["se", "me"]:
                        histogram = self.require_object(
                            archive, base + event + "_signal", "TH1D"
                        )
                        source = item["regions"]["signal"]
                        np.testing.assert_allclose(
                            histogram_values(histogram), source[event]
                        )
                        np.testing.assert_allclose(
                            histogram_errors(histogram) ** 2, source[event + "2"]
                        )
                    for observable in ["raw", "corrected"]:
                        source = (
                            item["correlations"]["signal"]
                            if observable == "raw"
                            else item["corrected"]
                        )
                        histogram = self.require_object(
                            archive, base + "cf_" + observable + "_stat", "TH1D"
                        )
                        valid = self.require_object(
                            archive, base + "valid_" + observable, "TH1"
                        )
                        np.testing.assert_allclose(
                            histogram_values(histogram), source["values"]
                        )
                        np.testing.assert_allclose(
                            histogram_errors(histogram) ** 2,
                            np.diag(source["covariance"]),
                            atol=1e-12,
                        )
                        np.testing.assert_array_equal(
                            histogram_values(valid), source["valid"]
                        )
                        edges = [
                            histogram.GetXaxis().GetBinLowEdge(i + 1)
                            for i in range(histogram.GetNbinsX() + 1)
                        ]
                        np.testing.assert_allclose(
                            edges, np.linspace(0, 0.3, 7), atol=1e-12
                        )

    def test_statistical_covariances_round_trip(self):
        with self.open_archive() as archive:
            for name, item in self.nominal.items():
                for observable in ["raw", "corrected"]:
                    source = (
                        item["correlations"]["signal"]
                        if observable == "raw"
                        else item["corrected"]
                    )
                    histogram = self.require_object(
                        archive, f"nominal/{name}/cov_{observable}_stat", "TH2D"
                    )
                    actual = matrix_values(histogram)
                    np.testing.assert_allclose(actual, source["covariance"], atol=1e-12)
                    np.testing.assert_allclose(actual, actual.T, atol=1e-12)

    def test_final_covariances_and_errors_match_numerical_exports(self):
        with self.open_archive() as archive, np.load(
            self.output / "systematics/covariances.npz"
        ) as exported:
            for name in self.nominal:
                for observable in ["raw", "corrected"]:
                    for uncertainty, numerical_name in [
                        ("syst", "systematic"),
                        ("total", "total"),
                    ]:
                        prefix = f"final/{name}/"
                        label = observable + "_" + uncertainty
                        covariance = self.require_object(
                            archive, prefix + "cov_" + label, "TH2D"
                        )
                        values = exported[f"{name}__{observable}__{numerical_name}"]
                        np.testing.assert_allclose(
                            matrix_values(covariance), values, atol=1e-12
                        )
                        histogram = self.require_object(
                            archive, prefix + "cf_" + label, "TH1D"
                        )
                        np.testing.assert_allclose(
                            histogram_errors(histogram) ** 2,
                            np.diag(values),
                            atol=1e-12,
                        )
                        graph = self.require_object(
                            archive, prefix + "g_cf_" + label, "TGraphErrors"
                        )
                        valid = self.require_object(
                            archive, prefix + "valid_" + label, "TH1D"
                        )
                        self.assertEqual(
                            graph.GetN(), int(histogram_values(valid).sum())
                        )

    def test_metadata_is_json_and_canvases_contain_primitives(self):
        with self.open_archive() as archive:
            for name in ["schema", "config", "training", "systematics", "provenance"]:
                metadata = self.require_object(
                    archive, "metadata/" + name, "TObjString"
                )
                self.assertIsInstance(json.loads(str(metadata.GetString())), dict)
            for key in archive.GetDirectory("canvases").GetListOfKeys():
                canvas = key.ReadObj()
                self.assertTrue(canvas.InheritsFrom("TCanvas"))
                self.assertGreater(canvas.GetListOfPrimitives().GetSize(), 0)

    def test_invalid_bins_persist_nan_and_graphs_exclude_them(self):
        from pih3l.config import load_config
        from pih3l.root_output import render_root_report

        copy = self.base / "invalid-source"
        shutil.copytree(self.output, copy)
        data = json.loads((copy / "nominal/results.json").read_text())
        key = next(iter(data))
        raw = data[key]["correlations"]["signal"]
        raw["valid"][1] = False
        raw["values"][1] = None
        for index in range(6):
            raw["covariance"][1][index] = None
            raw["covariance"][index][1] = None
        raw["status"] = "partial_empty_bins"
        corrected = data[key]["corrected"]
        corrected.update(
            values=[None] * 6,
            valid=[False] * 6,
            covariance=[[None] * 6 for _ in range(6)],
            status="insufficient_sideband_or_signal",
        )
        (copy / "nominal/results.json").write_text(json.dumps(data))
        destination = self.base / "invalid-report"
        destination.mkdir()
        images = render_root_report(copy, destination, load_config(self.config_path))
        self.assertTrue(images)
        self.assertTrue(
            all(Path(path).is_absolute() and Path(path).is_file() for _, path in images)
        )
        with ROOT.TFile.Open(str(destination / "analysis.root"), "READ") as archive:
            prefix = f"nominal/{key}/"
            raw_hist = self.require_object(archive, prefix + "cf_raw_stat", "TH1D")
            raw_graph = self.require_object(
                archive, prefix + "g_cf_raw_stat", "TGraphErrors"
            )
            corrected_graph = self.require_object(
                archive, prefix + "g_cf_corrected_stat", "TGraphErrors"
            )
            valid = self.require_object(archive, prefix + "valid_raw", "TH1D")
            self.assertTrue(np.isnan(raw_hist.GetBinContent(2)))
            self.assertTrue(np.isnan(raw_hist.GetBinError(2)))
            self.assertEqual(valid.GetBinContent(2), 0)
            self.assertEqual(raw_graph.GetN(), 5)
            self.assertEqual(corrected_graph.GetN(), 0)
            covariance = self.require_object(archive, prefix + "cov_raw_stat", "TH2D")
            validity = self.require_object(
                archive, prefix + "valid_cov_raw_stat", "TH2D"
            )
            self.assertTrue(np.isnan(covariance.GetBinContent(2, 3)))
            self.assertEqual(validity.GetBinContent(2, 3), 0)

    def test_weighted_histogram_sumw2_and_malformed_variance(self):
        from pih3l.errors import AnalysisError
        from pih3l.root_io import RootArchive

        path = self.base / "weighted.root"
        with RootArchive(path) as archive:
            archive.hist1("weighted", [0, 0.02, 0.1], [1.5, 2.5], variance=[1.25, 3.25])
            with self.assertRaises(AnalysisError):
                archive.hist1(
                    "wrong_shape", [0, 0.02, 0.1], [1.5, 2.5], variance=[1.25]
                )
            with self.assertRaises(AnalysisError):
                archive.hist1(
                    "negative_variance",
                    [0, 0.02, 0.1],
                    [1.5, 2.5],
                    variance=[1.25, -1.0],
                )
        with ROOT.TFile.Open(str(path), "READ") as archive:
            histogram = self.require_object(archive, "weighted", "TH1D")
            np.testing.assert_allclose(histogram_values(histogram), [1.5, 2.5])
            np.testing.assert_allclose(histogram_errors(histogram) ** 2, [1.25, 3.25])
            self.assertEqual(histogram.GetEntries(), 0)
            np.testing.assert_allclose(
                [histogram.GetXaxis().GetBinLowEdge(i) for i in [1, 2, 3]],
                [0, 0.02, 0.1],
            )

    def test_nonempty_mc_fractions_resolution_and_canvas_round_trip(self):
        from pih3l.config import load_config
        from pih3l.io import write_batch
        from pih3l.mc import run_mc_qa
        from pih3l.root_output import render_root_report
        from test_mc_qa import MonteCarloQATests

        source = self.base / "mc-source"
        shutil.copytree(self.output, source)
        config = load_config(self.config_path)
        frame = MonteCarloQATests.frame()
        frame.loc[frame.kstar > 0.3, "kstar"] = 0.25
        frame.loc[frame.KstarMC > 0.3, "KstarMC"] = 0.245
        write_batch(source / "scored/mc_femto", 0, frame)
        write_storage_metadata(
            source / "scored/mc_femto", config, {"storage_threshold": None}
        )
        run_mc_qa(source / "scored/mc_femto", source / "mc_qa", config, 0.5)
        numerical = json.loads((source / "mc_qa/summary.json").read_text())
        destination = self.base / "mc-report"
        render_root_report(source, destination, config)
        with ROOT.TFile.Open(str(destination / "analysis.root"), "READ") as archive:
            key = "cent_0_10__ch_0__se"
            prefix = "mc_qa/" + key + "/"
            item = numerical["bins"][key]
            fraction = self.require_object(
                archive, prefix + "primary_pion_fraction", "TH1D"
            )
            np.testing.assert_allclose(
                histogram_values(fraction),
                np.asarray(item["primary_pion_fraction"]["value"], float),
                equal_nan=True,
            )
            graph = self.require_object(
                archive, prefix + "g_primary_pion_fraction", "TGraphAsymmErrors"
            )
            self.assertEqual(graph.GetN(), sum(item["primary_pion_fraction"]["valid"]))
            self.assertGreater(graph.GetErrorYlow(0), 0)
            self.assertGreater(graph.GetErrorYhigh(0), 0)
            response = self.require_object(archive, prefix + "resolution", "TH2D")
            np.testing.assert_allclose(matrix_values(response), item["resolution"])
            canvas = self.require_object(
                archive, "canvases/mc_primary_cent0", "TCanvas"
            )
            self.assertGreater(canvas.GetListOfPrimitives().GetSize(), 0)
            image = self.base / "mc-redrawn.pdf"
            canvas.Draw()
            canvas.SaveAs(str(image))
            self.assertGreater(image.stat().st_size, 1000)
            self.assertTrue(image.read_bytes().startswith(b"%PDF-"))

    def test_auxiliary_analysis_directories_are_persisted(self):
        with self.open_archive() as archive:
            for name in [
                "raw_qa",
                "nominal",
                "final",
                "training",
                "working_point",
                "mixing_qa",
                "mc_qa",
                "canvases",
            ]:
                directory = self.require_object(archive, name, "TDirectory")
                self.assertGreater(directory.GetListOfKeys().GetSize(), 0)
            self.require_object(
                archive, "working_point/significance_vs_score", "TGraph"
            )
            self.require_object(archive, "working_point/best_mass_data", "TH1D")
            self.require_object(archive, "canvases/working_point", "TCanvas")
            raw = self.require_object(
                archive, "raw_qa/femto/cent_0_10__ch_0/kstar_se", "TH1D"
            )
            self.assertEqual(raw.GetEntries(), 180)
            self.require_object(archive, "raw_qa/hyper/cent_0_10/mass", "TH1D")
            self.require_object(archive, "canvases/raw_kstar_cent0", "TCanvas")
            self.assertFalse(archive.Get("canvases/raw_mass_cent0"))
            self.assertFalse(archive.Get("raw_qa/femto/cent_0_10__ch_0/mass_se"))
        self.assertFalse(list((self.output / "report").glob("*.png")))
        self.assertFalse(list((self.output / "report").glob("raw_mass*.pdf")))
        self.assertTrue(list((self.output / "report").glob("*.pdf")))
        html = (self.output / "report/index.html").read_text()
        self.assertNotIn("data:image/png", html)
        self.assertIn('type="application/pdf"', html)

    def test_serialized_canvas_is_redrawable_in_a_fresh_root_process(self):
        target = self.base / "redrawn.pdf"
        code = """
import sys
from pathlib import Path
import ROOT
ROOT.gROOT.SetBatch(True)
with ROOT.TFile.Open(sys.argv[1], "READ") as archive:
    canvases = archive.GetDirectory("canvases")
    keys = list(canvases.GetListOfKeys())
    assert keys, "No persisted canvases"
    canvas = keys[0].ReadObj()
    assert canvas.InheritsFrom("TCanvas"), canvas.ClassName()
    assert canvas.GetListOfPrimitives().GetSize() > 0, "Empty serialized canvas"
    canvas.Draw()
    canvas.SaveAs(sys.argv[2])
path = Path(sys.argv[2])
assert path.read_bytes().startswith(b"%PDF-")
assert path.stat().st_size > 1000
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.archive), str(target)],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertGreater(target.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
