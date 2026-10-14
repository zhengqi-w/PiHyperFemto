"""Spectrum-style window envelopes, exact nominal markers and fit components."""

import copy
import json
import re
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import ROOT

from pih3l.io import read_json
from pih3l.wp_mass_fit import TAIL_DEFAULTS, mass_components, mass_densities
from pih3l.working_point_qa import render_working_point_qa
from pih3l.root_io import RootArchive
from pih3l.root_plot import canvas, pad, save_canvas, text


def pdf_marker_bounds(path, marker):
    """Read actual native ROOT PDF transforms without a PDF runtime dependency.

    ROOT emits text in a local `cm ... BT ... Tj` block after page transforms.
    The page's MediaBox applies before its optional viewing rotation. Testing
    transformed text bounds catches clipped panels that still exist in ROOT.
    """
    document = path.read_bytes()
    box = re.search(rb"/MediaBox\s*\[([^]]+)\]", document)
    if box is None:
        raise AssertionError("Native ROOT PDF has no MediaBox")
    page = np.array([float(value) for value in box[1].split()])
    contents = []
    for encoded in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", document, re.S):
        try:
            contents.append(zlib.decompress(encoded))
        except zlib.error:
            contents.append(encoded)
    literal = b"(" + marker.encode("ascii") + b")"
    content = next((item for item in contents if literal in item), None)
    if content is None:
        raise AssertionError(f"Native ROOT PDF lost {marker}")
    number = rb"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    matrix_pattern = re.compile(rb"\s+".join([number] * 6) + rb"\s+cm")

    def matrix(match):
        a, b, c, d, e, f = map(float, match.groups())
        return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])

    # Page transforms precede the first graphics-state save. Every subsequent
    # ROOT text block uses absolute canvas coordinates and restores its state.
    initial = re.split(rb"\bq\b", content, maxsplit=1)[0]
    transform = np.eye(3)
    for match in matrix_pattern.finditer(initial):
        transform = transform @ matrix(match)
    position = content.index(literal)
    local = list(matrix_pattern.finditer(content[:position]))[-1]
    transform = transform @ matrix(local)
    font = float(
        re.findall(rb"/F\d+\s+" + number + rb"\s+Tf", content[local.end() : position])[
            -1
        ]
    )
    # A conservative font box includes descenders and ascenders. Markers are
    # short ASCII words deliberately well inside their panel horizontally.
    corners = np.array(
        [
            [0.0, -0.2 * font, 1.0],
            [0.6 * len(marker) * font, -0.2 * font, 1.0],
            [0.0, font, 1.0],
            [0.6 * len(marker) * font, font, 1.0],
        ]
    ).T
    physical = (transform @ corners)[:2]
    return page, physical.min(axis=1), physical.max(axis=1)


class SpectrumWorkingPointQATests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_a4_pdf_keeps_lower_and_upper_panels_inside_page(self):
        for width, height in ((950, 850), (850, 950), (1600, 600)):
            with self.subTest(canvas=(width, height)):
                name = f"paper_{width}_{height}"
                with RootArchive(self.directory / (name + ".root")) as archive:
                    item, panel = canvas(
                        archive,
                        name,
                        "Two panel export regression",
                        1,
                        2,
                        width=width,
                        height=height,
                    )
                    pad(panel, 1)
                    text(archive, 0.2, 0.95, "TopPanelSentinel", 0.025)
                    pad(panel, 2)
                    text(archive, 0.2, 0.04, "BottomPanelSentinel", 0.025)
                    # The current ROOT pad is the bottom subpad, matching a
                    # real plot export; save_canvas must select the canvas.
                    path = save_canvas(archive, item, self.directory)
                for marker in ("TopPanelSentinel", "BottomPanelSentinel"):
                    page, low, high = pdf_marker_bounds(path, marker)
                    self.assertTrue((low >= page[:2]).all(), (marker, page, low))
                    self.assertTrue((high <= page[2:]).all(), (marker, page, high))

    @staticmethod
    def fixture():
        rows = []
        for index, (threshold, efficiency, significance) in enumerate(
            ((0.8, 0.4, 2.0), (0.2, 0.9, 9.0), (0.5, 0.6, 4.0))
        ):
            rows.append(
                {
                    "threshold": threshold,
                    "mc_signal_efficiency": efficiency,
                    "actual_mc_signal_efficiency": efficiency + 0.01,
                    "significance": 50.0 + index,
                    "significance_2sigma": significance + 0.7,
                    "significance_3sigma": significance,
                    "significance_4sigma": significance - 0.3,
                    "raw_signal": 100.0 * efficiency,
                    "raw_signal_error": 3.0,
                    "signal_over_efficiency": 100.0,
                    "signal_over_efficiency_error": 3.0 / efficiency,
                    "raw_signal_3sigma": 120.0 * efficiency,
                    "raw_signal_3sigma_error": 4.0,
                    "raw_signal_3sigma_over_efficiency": 120.0,
                    "raw_signal_3sigma_over_efficiency_error": 4.0 / efficiency,
                    "mean": 2.991 + index * 0.0001,
                    "mean_error": 0.0001,
                    "sigma": 0.0015 + index * 0.00005,
                    "sigma_error": 0.0001,
                    "chi2_ndf": 1.0,
                    "fit_status": "ok",
                    "valid": True,
                    "spectrum_qa_valid": index != 0,
                }
            )
        edges = np.linspace(2.98, 3.01, 13)
        parameters = {
            "signal_yield": 40.0,
            "background_yield": 20.0,
            "mean": 2.991,
            "sigma": 0.0015,
            "c0": 0.1,
            "c1": 0.0,
            **TAIL_DEFAULTS,
        }
        expected = sum(mass_components(parameters, edges))
        counts = np.round(expected + np.sin(np.arange(len(expected))) * 2).clip(0)
        points = [
            {
                "index": index,
                "threshold": row["threshold"],
                "counts": counts.tolist(),
                "expected": expected.tolist(),
                "fit": {"status": "ok", "parameters": parameters},
                "scan_point": row,
            }
            for index, row in enumerate(rows)
        ]
        pol0 = {
            "status": "ok",
            "c0": 120.0,
            "c0_error": 4.0,
            "chi2_ndf": 0.8,
            "fit_range": [0.4, 0.9],
        }
        summary = {
            "status": "ok",
            "threshold": 0.5,
            "best_scan_point": rows[2],
            "nominal_mc_signal_efficiency": 0.61,
            "scan_points": rows,
            "objective": "fixed_window_nominal_objective",
            "yield_pol0": {"status": "ok", "c0": 100.0},
            "resolved_settings": {
                "signal_window": [2.989, 2.993],
                "purity": {"background": "linear"},
            },
            "spectrum_qa": {
                "status": "ok",
                "yield_pol0": pol0,
                "window_convention": "fitted mean +/- n sigma clipped to mass_range",
                "diagnostic_maximum": rows[1],
                "nominal_point": {
                    **rows[2],
                    "efficiency_basis": "requested scan efficiency",
                },
            },
        }
        best = {**points[2], "mass_edges": edges.tolist()}
        scans = {"mass_edges": edges.tolist(), "points": points}
        return summary, best, scans

    def render(self, summary=None):
        base, best, scans = self.fixture()
        if summary is not None:
            base = summary
        render_working_point_qa(self.directory, base, best, scans)
        return base

    def archive(self):
        return ROOT.TFile.Open(str(self.directory / "qa/working_point.root"), "READ")

    def test_band_uses_common_mask_sorted_final_values_and_nominal_not_qa_maximum(self):
        summary = self.render()
        with self.archive() as archive:
            for name, expected_x in (
                ("score", [0.2, 0.5, 0.5, 0.2]),
                ("efficiency", [0.6, 0.9, 0.9, 0.6]),
            ):
                prefix = "scan/significance_vs_" + name + "/"
                band = archive.Get(prefix + "window_envelope")
                self.assertEqual(band.GetN(), 4)
                self.assertFalse(band.InheritsFrom("TGraphErrors"))
                self.assertNotEqual(band.GetFillColor(), 0)
                np.testing.assert_allclose(
                    [band.GetPointX(index) for index in range(4)], expected_x
                )
                expected_y = (
                    [9.7, 4.7, 3.7, 8.7] if name == "score" else [4.7, 9.7, 8.7, 3.7]
                )
                np.testing.assert_allclose(
                    [band.GetPointY(index) for index in range(4)], expected_y
                )
                plotted = (
                    archive.Get("canvases/significance_vs_" + name)
                    .GetPrimitive("significance_vs_" + name + "_panels")
                    .GetPad(1)
                    .GetPrimitive(band.GetName())
                )
                self.assertEqual(plotted.GetN(), band.GetN())
                np.testing.assert_allclose(
                    [plotted.GetPointY(index) for index in range(4)], expected_y
                )
                point = archive.Get(prefix + "nominal_wp")
                self.assertEqual(point.GetMarkerStyle(), 29)
                self.assertEqual(point.GetN(), 1)
                self.assertAlmostEqual(
                    point.GetPointX(0), 0.5 if name == "score" else 0.6
                )
                self.assertEqual(point.GetPointY(0), 4.0)
                definition = json.loads(
                    str(archive.Get(prefix + "window_definition").GetString())
                )
                self.assertIn("not statistical", definition["interpretation"])
                self.assertEqual(
                    archive.Get(prefix + "significance").GetPointY(0), 50.0
                )
            self.assertEqual(
                summary["spectrum_qa"]["diagnostic_maximum"]["threshold"], 0.2
            )

    def test_raw_three_sigma_pol0_components_and_single_panel_roundtrip(self):
        self.render()
        with self.archive() as archive:
            self.assertEqual(
                archive.Get("scan/spectrum_yield_pol0_function").GetParameter(0), 120.0
            )
            graph = archive.Get(
                "scan/raw_yield_over_efficiency/raw_signal_3sigma_over_efficiency"
            )
            self.assertEqual(graph.GetPointY(0), 120.0)
            self.assertAlmostEqual(graph.GetErrorY(0), 4.0 / 0.4)
            self.assertEqual(
                archive.Get(
                    "scan/raw_yield_over_efficiency/signal_over_efficiency"
                ).GetPointY(0),
                100.0,
            )
            marker = archive.Get("scan/raw_yield_over_efficiency/nominal_wp")
            self.assertAlmostEqual(marker.GetPointX(0), 0.6)
            self.assertEqual(marker.GetPointY(0), 120.0)
            mean = archive.Get("stability/mean_requested_efficiency")
            np.testing.assert_allclose(
                [mean.GetPointX(index) for index in range(3)], [0.4, 0.9, 0.6]
            )
            np.testing.assert_allclose(
                [mean.GetErrorX(index) for index in range(3)], [0.1, 0.15, 0.1]
            )
            for prefix in (
                "best_mass_fit",
                "mass_scan/point_0000",
                "mass_scan/point_0001",
                "mass_scan/point_0002",
            ):
                source = archive.Get(prefix + "/data")
                fitted = archive.Get(prefix + "/expected")
                signal = archive.Get(prefix + "/signal")
                background = archive.Get(prefix + "/background")
                self.assertFalse(archive.Get(prefix + "/pull"))
                for index in range(1, source.GetNbinsX() + 1):
                    self.assertAlmostEqual(
                        signal.GetBinContent(index) + background.GetBinContent(index),
                        fitted.GetBinContent(index),
                    )

                metadata = json.loads(
                    str(archive.Get(prefix + "/component_definition").GetString())
                )
                self.assertIn("DSCB", metadata["reference_scope"])
                total_curve = archive.Get(prefix + "/curves/total")
                self.assertTrue(total_curve.InheritsFrom("TGraph"))
                self.assertEqual(total_curve.GetN(), 1025)
                curve_x = np.array(
                    [total_curve.GetPointX(i) for i in range(total_curve.GetN())]
                )
                p = self.fixture()[1]["fit"]["parameters"]
                edges = self.fixture()[1]["mass_edges"]
                signal_y, background_y = mass_densities(
                    p, [edges[0], edges[-1]], curve_x
                )
                np.testing.assert_allclose(
                    [total_curve.GetPointY(i) for i in range(total_curve.GetN())],
                    (signal_y + background_y) * (edges[1] - edges[0]),
                    rtol=1e-12,
                )
            best_canvas = archive.Get("canvases/best_mass_fit")
            self.assertFalse(best_canvas.GetPrimitive("best_mass_fit_panels").GetPad(2))
            primitives = list(
                best_canvas.GetPrimitive("best_mass_fit_panels")
                .GetPad(1)
                .GetListOfPrimitives()
            )
            self.assertEqual(sum(item.InheritsFrom("TGraph") for item in primitives), 3)
            for component in ("expected", "signal", "background"):
                self.assertFalse(
                    any(
                        item.InheritsFrom("TH1")
                        and item.GetName() == f"best_mass_fit__{component}"
                        for item in primitives
                    )
                )
        index = read_json(self.directory / "qa/index.json")
        expected = {
            "sig_vs_score.pdf",
            "sig_vs_eff.pdf",
            "s3overeff_vs_eff.pdf",
            "mean_sigma_vs_efficiency.pdf",
            "best_fit.pdf",
        }
        self.assertTrue(expected <= set(index["files"]))
        for name in index["files"]:
            self.assertTrue((self.directory / "qa" / name).is_file())
            if name.endswith(".pdf"):
                self.assertTrue(
                    (self.directory / "qa" / name).read_bytes().startswith(b"%PDF-")
                )
        for old in (
            "significance_vs_score.pdf",
            "significance_vs_efficiency.pdf",
            "raw_yield_over_efficiency.pdf",
            "best_mass_fit.pdf",
        ):
            self.assertFalse((self.directory / "qa" / old).exists())
        self.assertFalse(list(self.directory.rglob("*.png")))
        self.assertFalse(list((self.directory / "qa").glob("mass_fit_*.pdf")))

    def test_exact_override_outside_scan_is_visible_and_invalid_override_has_no_star(
        self,
    ):
        summary, _, _ = self.fixture()
        point = copy.deepcopy(summary["spectrum_qa"]["nominal_point"])
        point.update(
            threshold=0.999,
            mc_signal_efficiency=0.15,
            actual_mc_signal_efficiency=0.15,
            significance_3sigma=15.0,
            efficiency_basis="actual held-out override",
        )
        summary.update(threshold=0.999)
        summary["spectrum_qa"]["nominal_point"] = point
        self.render(summary)
        with self.archive() as archive:
            marker = archive.Get("scan/significance_vs_score/nominal_wp")
            self.assertAlmostEqual(marker.GetPointX(0), 0.999)
            axes = (
                archive.Get("canvases/significance_vs_score")
                .GetPrimitive("significance_vs_score_panels")
                .GetPad(1)
                .GetPrimitive("hframe")
            )
            self.assertGreater(axes.GetXaxis().GetXmax(), 0.999)
            self.assertGreater(axes.GetMaximum(), 15.0)
        point.update(valid=False, spectrum_qa_valid=False)
        self.render(summary)
        with self.archive() as archive:
            self.assertEqual(
                archive.Get("scan/significance_vs_score/nominal_wp").GetN(), 0
            )
            self.assertEqual(
                archive.Get("scan/raw_yield_over_efficiency/nominal_wp").GetN(), 0
            )

    def test_missing_exact_cut_or_failed_three_sigma_pol0_never_fabricates_marker_or_ns(
        self,
    ):
        summary, _, _ = self.fixture()
        summary["threshold"] = 0.5 + 1e-13
        summary["spectrum_qa"].pop("nominal_point")
        self.render(summary)
        with self.archive() as archive:
            self.assertEqual(
                archive.Get("scan/significance_vs_score/nominal_wp").GetN(), 0
            )
        summary, _, _ = self.fixture()
        summary["spectrum_qa"].update(
            status="failed", yield_pol0={"status": "too_few_distinct_thresholds"}
        )
        for row in summary["scan_points"]:
            row.update(
                spectrum_qa_valid=False,
                significance_2sigma=None,
                significance_3sigma=None,
                significance_4sigma=None,
            )
        summary["spectrum_qa"]["nominal_point"] = summary["scan_points"][2]
        self.render(summary)
        with self.archive() as archive:
            self.assertEqual(
                archive.Get("scan/significance_vs_score/significance_3sigma").GetN(), 0
            )
            self.assertEqual(
                archive.Get("scan/significance_vs_score/window_envelope").GetN(), 0
            )
            self.assertEqual(
                archive.Get(
                    "scan/raw_yield_over_efficiency/raw_signal_3sigma_over_efficiency"
                ).GetN(),
                3,
            )
            self.assertFalse(archive.Get("scan/spectrum_yield_pol0_function"))
        self.assertEqual(read_json(self.directory / "qa/index.json")["status"], "ok")
        self.assertEqual(
            read_json(self.directory / "qa/index.json")["spectrum_qa_status"], "failed"
        )

    def test_failed_wp_labels_diagnostics_and_retains_rejected_mass_fits(self):
        summary, _, scans = self.fixture()
        summary.update(status="failed", threshold=None, save_score_fit_frames=True)
        summary["scan_points"][0]["valid"] = False
        summary["spectrum_qa"].update(status="failed", yield_pol0={"status": "failed"})
        for row in summary["scan_points"]:
            row["spectrum_qa_valid"] = False
        render_working_point_qa(self.directory, summary, {}, scans)
        with self.archive() as archive:
            raw = archive.Get(
                "scan/raw_yield_over_efficiency/raw_signal_3sigma_over_efficiency"
            )
            self.assertEqual(raw.GetN(), 2)
            self.assertEqual(
                archive.Get("scan/raw_yield_over_efficiency/nominal_wp").GetN(), 0
            )
            self.assertTrue(archive.Get("mass_scan/point_0000/data"))
            panel = (
                archive.Get("canvases/significance_vs_score")
                .GetPrimitive("significance_vs_score_panels")
                .GetPad(1)
            )
            self.assertTrue(
                any(
                    item.InheritsFrom("TLatex") and "WP unavailable" in item.GetTitle()
                    for item in panel.GetListOfPrimitives()
                )
            )
        self.assertEqual(len(list((self.directory / "qa").glob("mass_fit_*.pdf"))), 3)
        self.assertFalse((self.directory / "qa/best_fit.pdf").exists())


if __name__ == "__main__":
    unittest.main()
