"""Independent nominal-ratio mathematics and native ROOT persistence checks.

Ratio plots intentionally carry no statistical errors: the covariance between
nested nominal/varied pair samples is not measured by this QA layer.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pih3l.config import DEFAULTS, merge
from pih3l.io import write_json
from pih3l.root_io import RootArchive, load_root
from pih3l.systematics_qa import (
    build_systematic_qa,
    export_systematic_qa,
    ratio_to_nominal,
    relative_uncertainty,
    render_systematic_qa,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source(values, valid=None):
    values = np.asarray(values, float)
    return {
        "values": values,
        "valid": (
            np.ones(len(values), bool) if valid is None else np.asarray(valid, bool)
        ),
        "covariance": np.eye(len(values)) * 0.04,
        "status": "ok",
    }


def qa_fixture():
    """Three independent sources, two alternatives each; no production combiner."""
    edges = [0, 0.1, 0.2, 0.3, 0.4]
    groups = ("pion_pt_min", "pion_pt_max", "pion_tpc_rows")
    config = merge(
        DEFAULTS,
        {"analysis": {"centrality_bins": [[0, 10]], "kstar_edges": edges}},
    )
    nominal, combined = {}, {}
    for channel in range(4):
        key = f"cent_0_10__ch_{channel}"
        raw = source([2, 1, 0, 3], [True, True, True, False])
        corrected = source([1.5, 1, 0, 2], [True, False, True, False])
        nominal[key] = {
            "centrality": [0, 10],
            "channel": channel,
            "channel_label": f"charge {channel}",
            "correlations": {"signal": raw},
            "corrected": corrected,
        }
        combined[key] = {}
        for name, baseline in (("raw", raw), ("corrected", corrected)):
            errors = [np.abs(baseline["values"]) * step for step in (0.1, 0.2, 0.3)]
            combined[key][name] = {
                "error": np.sqrt(np.sum(np.square(errors), axis=0)),
                "valid": baseline["valid"].copy(),
                "groups": {
                    group: {"error": error, "coverage": np.full(4, 2)}
                    for group, error in zip(groups, errors)
                },
            }
    variations = []
    for group, magnitude in zip(groups, (0.1, 0.2, 0.3)):
        for index, shift in enumerate((magnitude / 2, -magnitude)):
            varied = copy.deepcopy(nominal)
            for item in varied.values():
                item["correlations"]["signal"]["values"] *= 1 + shift
                item["corrected"]["values"] *= 1 + shift
            variations.append(
                (
                    {"name": f"{group}_{index}", "group": group, "status": "completed"},
                    varied,
                )
            )
    config["systematics"]["variations"] = [
        {"name": meta["name"], "group": meta["group"], "overrides": {}}
        for meta, _ in variations
    ]
    return nominal, variations, combined, config


def all_qa_leaves(qa):
    for name, variation in qa["variations"].items():
        for key, observables in variation["bins"].items():
            for observable, result in observables.items():
                yield f"systematics_qa/variations/{name}/{key}/{observable}", result
    for key, observables in qa["bins"].items():
        for observable, result in observables.items():
            yield f"systematics_qa/total/{key}/{observable}", result["total"]
            for group, component in result["groups"].items():
                yield f"systematics_qa/sources/{group}/{key}/{observable}", component


class SystematicRatioMathTests(unittest.TestCase):
    def test_ratio_uses_nominal_correlation_not_its_standard_error(self):
        nominal = source([2, 4, -2])
        varied = source([3, 2, 1])
        result = ratio_to_nominal(nominal, varied)
        np.testing.assert_allclose(result["values"], [1.5, 0.5, -0.5])
        np.testing.assert_array_equal(result["valid"], [True, True, True])
        self.assertNotIn("error", result)
        self.assertNotIn("covariance", result)

    def test_identical_sources_have_unit_ratio_without_invented_error(self):
        baseline = source([0.5, 1, 2])
        result = ratio_to_nominal(baseline, copy.deepcopy(baseline))
        np.testing.assert_array_equal(result["values"], [1, 1, 1])
        self.assertNotIn("error", result)
        self.assertNotIn("covariance", result)

    def test_zero_nonfinite_and_explicitly_invalid_denominators_are_missing(self):
        nominal = source([2, 0, np.nan, 1, 1, 1], [True, True, True, False, True, True])
        varied = source([3, 2, 1, 1, np.inf, 2], [True, True, True, True, True, False])
        result = ratio_to_nominal(nominal, varied)
        np.testing.assert_array_equal(
            result["valid"], [True, False, False, False, False, False]
        )
        self.assertEqual(result["values"][0], 1.5)
        self.assertTrue(np.isnan(np.asarray(result["values"], float)[1:]).all())

    def test_zero_numerator_is_a_valid_zero_ratio(self):
        result = ratio_to_nominal(source([2]), source([0]))
        np.testing.assert_array_equal(result["valid"], [True])
        np.testing.assert_array_equal(result["values"], [0])

    def test_nonfinite_or_negative_input_variance_masks_diagnostics(self):
        nominal = source([1, 1, 1, 1])
        varied = source([2, 2, 2, 2])
        nominal["covariance"][1, 1] = -0.1
        varied["covariance"][2, 2] = -0.1
        nominal["covariance"][3, 3] = np.nan
        result = ratio_to_nominal(nominal, varied)
        np.testing.assert_array_equal(result["valid"], [True, False, False, False])
        self.assertTrue(np.isnan(np.asarray(result["values"], float)[1:]).all())
        relative = relative_uncertainty(nominal, [0.2] * 4)
        np.testing.assert_array_equal(relative["valid"], [True, False, True, False])

    def test_overflow_in_a_finite_input_ratio_or_relative_error_is_invalid(self):
        nominal = source([1e-300, 2])
        result = ratio_to_nominal(nominal, source([1e300, 3]))
        relative = relative_uncertainty(nominal, [1e300, 0.2])
        for diagnostic in (result, relative):
            np.testing.assert_array_equal(diagnostic["valid"], [False, True])
            self.assertTrue(np.isnan(diagnostic["values"][0]))

    def test_relative_uncertainty_divides_by_absolute_nominal(self):
        result = relative_uncertainty(source([2, -4, 1]), [0.2, 0.8, 0])
        np.testing.assert_allclose(result["values"], [0.1, 0.2, 0])
        np.testing.assert_array_equal(result["valid"], [True, True, True])

    def test_relative_uncertainty_preserves_invalid_and_unavailable_errors(self):
        baseline = source([2, 0, np.nan, 1, 1, 1, 1])
        errors = [0.2, 0.1, 0.1, np.nan, -0.1, 0.1, 0.1]
        baseline["valid"][-1] = False
        result = relative_uncertainty(baseline, errors, [True] * 5 + [False, True])
        np.testing.assert_array_equal(
            result["valid"], [True, False, False, False, False, False, False]
        )
        self.assertTrue(np.isnan(np.asarray(result["values"], float)[1:]).all())

    def test_build_keeps_source_alternatives_and_quadrature_total_distinct(self):
        nominal, variations, combined, config = qa_fixture()
        qa = build_systematic_qa(nominal, variations, combined, config)
        self.assertEqual(list(qa["kstar_edges"]), config["analysis"]["kstar_edges"])
        self.assertEqual(
            set(qa["variations"]), {meta["name"] for meta, _ in variations}
        )
        for key, result in qa["bins"].items():
            raw = result["raw"]
            self.assertEqual(
                set(raw["groups"]), {"pion_pt_min", "pion_pt_max", "pion_tpc_rows"}
            )
            for group, step in zip(raw["groups"], (0.1, 0.2, 0.3)):
                np.testing.assert_allclose(
                    np.asarray(raw["groups"][group]["values"])[:2], step
                )
            np.testing.assert_allclose(
                np.asarray(raw["total"]["values"])[:2], np.sqrt(0.14)
            )
            np.testing.assert_array_equal(
                raw["total"]["valid"], [True, True, False, False]
            )
            np.testing.assert_array_equal(
                result["corrected"]["total"]["valid"], [True, False, False, False]
            )
            for meta, _ in variations:
                leaf = qa["variations"][meta["name"]]
                self.assertEqual(leaf["group"], meta["group"])
                self.assertEqual(leaf["status"], "completed")
                expected = 1 + {
                    "pion_pt_min": 0.1,
                    "pion_pt_max": 0.2,
                    "pion_tpc_rows": 0.3,
                }[meta["group"]] * (0.5 if meta["name"].endswith("_0") else -1)
                np.testing.assert_allclose(
                    np.asarray(leaf["bins"][key]["raw"]["values"])[:2], expected
                )

    def test_failed_variation_remains_explicit_without_fake_unit_ratios(self):
        nominal, variations, combined, config = qa_fixture()
        variations.append(
            (
                {
                    "name": "missing_column",
                    "group": "missing",
                    "status": "failed",
                    "reason": "required branch unavailable",
                },
                None,
            )
        )
        config["systematics"]["variations"].append(
            {"name": "missing_column", "group": "missing", "overrides": {}}
        )
        qa = build_systematic_qa(nominal, variations, combined, config)
        missing = qa["variations"]["missing_column"]
        self.assertEqual(missing["status"], "failed")
        self.assertIn("required branch", missing["reason"])
        for observables in missing["bins"].values():
            for result in observables.values():
                self.assertFalse(np.asarray(result["valid"]).any())
                self.assertTrue(np.isnan(np.asarray(result["values"], float)).all())

    def test_partial_bin_coverage_preserves_available_spread_but_marks_budget(self):
        nominal, variations, combined, config = qa_fixture()
        for item in combined.values():
            item["raw"]["groups"]["pion_pt_min"]["coverage"][1] = 1
        qa = build_systematic_qa(nominal, variations, combined, config)
        for item in qa["bins"].values():
            source_qa = item["raw"]["groups"]["pion_pt_min"]
            self.assertEqual(source_qa["coverage_status"], "incomplete_bin_coverage")
            np.testing.assert_array_equal(
                source_qa["valid"], [True, True, False, False]
            )
            np.testing.assert_array_equal(
                source_qa["complete_coverage"], [True, False, False, False]
            )
            total = item["raw"]["total"]
            self.assertEqual(total["coverage_status"], "incomplete_variation_coverage")
            np.testing.assert_array_equal(
                total["complete_coverage"], [True, False, False, False]
            )
            np.testing.assert_allclose(np.asarray(total["values"])[:2], np.sqrt(0.14))

    def test_failed_alternative_marks_source_and_total_incomplete(self):
        nominal, variations, combined, config = qa_fixture()
        metadata, _ = variations[0]
        variations[0] = (
            {**metadata, "status": "failed", "reason": "missing field"},
            None,
        )
        for item in combined.values():
            for observable in ("raw", "corrected"):
                item[observable]["groups"]["pion_pt_min"]["coverage"][:] = 1
        qa = build_systematic_qa(nominal, variations, combined, config)
        self.assertEqual(
            qa["sources"]["pion_pt_min"]["status"], "incomplete_variations"
        )
        self.assertEqual(qa["sources"]["pion_pt_min"]["failed"], [metadata["name"]])
        for item in qa["bins"].values():
            self.assertFalse(
                np.any(item["raw"]["groups"]["pion_pt_min"]["complete_coverage"])
            )
            self.assertFalse(np.any(item["raw"]["total"]["complete_coverage"]))
            self.assertEqual(
                item["raw"]["total"]["coverage_status"], "incomplete_variation_coverage"
            )

    def test_rms_dropped_common_ensemble_member_marks_populated_bins_partial(self):
        nominal, variations, combined, config = qa_fixture()
        config["systematics"]["method"] = "rms"
        for item in combined.values():
            for observable in ("raw", "corrected"):
                for group in item[observable]["groups"].values():
                    group["ensemble_size"] = 2
            affected = item["raw"]["groups"]["pion_pt_min"]
            affected["coverage"][1] = 1
            affected["ensemble_size"] = 1
        qa = build_systematic_qa(nominal, variations, combined, config)
        for item in qa["bins"].values():
            affected = item["raw"]["groups"]["pion_pt_min"]
            self.assertEqual(affected["coverage"][0], 2)
            self.assertTrue(affected["valid"][0])
            self.assertEqual(affected["ensemble_size"], 1)
            self.assertFalse(np.any(affected["complete_coverage"]))
            self.assertEqual(affected["coverage_status"], "incomplete_bin_coverage")
            self.assertFalse(np.any(item["raw"]["total"]["complete_coverage"]))
            self.assertEqual(
                item["raw"]["total"]["coverage_status"], "incomplete_variation_coverage"
            )


class SystematicQaRootTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.nominal, variations, combined, self.config = qa_fixture()
        self.qa = build_systematic_qa(self.nominal, variations, combined, self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fraction_histogram_runs_use_percent_and_preserve_internal_gaps(self):
        # Separate isolated valid bins cannot become a continuous line or zero.
        diagnostic = self.qa["bins"]["cent_0_10__ch_0"]["raw"]["total"]
        diagnostic["values"] = np.asarray([0.1, np.nan, 0.3, 0.4])
        diagnostic["valid"] = np.asarray([True, False, True, True])
        target = self.directory / "gaps.root"
        with RootArchive(target) as archive:
            export_systematic_qa(archive, self.qa)
        ROOT = load_root()
        with ROOT.TFile.Open(str(target), "READ") as archive:
            prefix = "systematics_qa/total/cent_0_10__ch_0/raw"
            metadata = json.loads(
                str(archive.Get(prefix + "/percent_steps/metadata").GetString())
            )
            self.assertEqual(metadata["unit"], "percent")
            self.assertEqual(metadata["scale"], 100)
            self.assertEqual(
                metadata["runs"],
                [
                    {"name": "run_0", "start_bin": 0, "stop_bin": 1},
                    {"name": "run_1", "start_bin": 2, "stop_bin": 4},
                ],
            )
            for run, expected_edges, expected_values in (
                (0, [0, 0.1], [10]),
                (1, [0.2, 0.3, 0.4], [30, 40]),
            ):
                histogram = archive.Get(prefix + f"/percent_steps/run_{run}")
                self.assertEqual(histogram.GetNbinsX(), len(expected_values))
                np.testing.assert_allclose(
                    [
                        histogram.GetBinContent(i + 1)
                        for i in range(len(expected_values))
                    ],
                    expected_values,
                )
                np.testing.assert_allclose(
                    [
                        histogram.GetXaxis().GetBinLowEdge(i + 1)
                        for i in range(len(expected_edges))
                    ],
                    expected_edges,
                )
                self.assertTrue(
                    all(
                        histogram.GetBinError(i + 1) == 0
                        for i in range(histogram.GetNbinsX())
                    )
                )
            self.assertAlmostEqual(
                archive.Get(prefix + "/values").GetBinContent(1), 0.1
            )

    def test_breakdown_canvas_has_black_total_colored_sources_and_percent_axis(self):
        index = render_systematic_qa(self.directory, self.nominal, self.qa, self.config)
        kinds = [plot["kind"] for plot in index["plots"]]
        self.assertEqual(kinds.count("total"), 1)
        self.assertEqual(kinds.count("total_corrected"), 1)
        ROOT = load_root()
        with ROOT.TFile.Open(
            str(self.directory / "qa/systematics.root"), "READ"
        ) as archive:
            for observable, suffix in (("raw", ""), ("corrected", "__corrected")):
                item = archive.Get(f"canvases/systematic__total{suffix}__cent0")
                panel = (
                    item.GetListOfPrimitives()
                    .FindObject(item.GetName() + "_panels")
                    .GetPad(1)
                )
                histograms = [
                    o for o in panel.GetListOfPrimitives() if o.InheritsFrom("TH1")
                ]
                self.assertTrue(
                    any(
                        o.GetYaxis().GetTitle() == "Systematic uncertainty (%)"
                        for o in histograms
                    )
                )
                steps = [o for o in histograms if "percent_steps" in o.GetName()]
                self.assertEqual(len(steps), 4)  # Three sources and quadrature total.
                total = next(
                    o
                    for o in steps
                    if o.GetName().startswith("systematics_qa__total__")
                )
                self.assertEqual(total.GetLineColor(), ROOT.kBlack)
                self.assertEqual(total.GetLineWidth(), 3)
                self.assertAlmostEqual(total.GetBinContent(1), 100 * np.sqrt(0.14))
                self.assertTrue(
                    all(o.GetLineColor() != ROOT.kBlack for o in steps if o != total)
                )
                options = {}
                link = panel.GetListOfPrimitives().FirstLink()
                while link:
                    options[link.GetObject().GetName()] = str(link.GetOption()).upper()
                    link = link.Next()
                self.assertTrue(all("HIST" in options[o.GetName()] for o in steps))
                self.assertTrue(all("][" in options[o.GetName()] for o in steps))
            # A later overlay must not recolor the earlier source canvas/book page.
            item = archive.Get("canvases/systematic__source__pion_pt_min__cent0")
            panel = (
                item.GetListOfPrimitives()
                .FindObject(item.GetName() + "_panels")
                .GetPad(1)
            )
            raw = next(
                o
                for o in panel.GetListOfPrimitives()
                if "__raw__percent_steps" in o.GetName()
            )
            self.assertEqual(raw.GetLineColor(), ROOT.kBlue + 1)

    def test_source_named_total_cannot_overwrite_combined_total_pdf_or_canvas(self):
        nominal, variations, combined, config = qa_fixture()
        for metadata, _ in variations:
            if metadata["group"] == "pion_pt_min":
                metadata["group"] = "total"
        for settings in config["systematics"]["variations"]:
            if settings["group"] == "pion_pt_min":
                settings["group"] = "total"
        for item in combined.values():
            for observable in ("raw", "corrected"):
                groups = item[observable]["groups"]
                groups["total"] = groups.pop("pion_pt_min")
        qa = build_systematic_qa(nominal, variations, combined, config)
        index = render_systematic_qa(self.directory, nominal, qa, config)
        source_plot = next(
            p for p in index["plots"] if p["kind"] == "source" and p["name"] == "total"
        )
        total_plot = next(p for p in index["plots"] if p["kind"] == "total")
        self.assertNotEqual(source_plot["path"], total_plot["path"])
        self.assertNotEqual(source_plot["canvas"], total_plot["canvas"])
        for plot in (source_plot, total_plot):
            self.assertTrue(
                (self.directory / "qa" / plot["path"]).read_bytes().startswith(b"%PDF-")
            )
        ROOT = load_root()
        with ROOT.TFile.Open(
            str(self.directory / "qa/systematics.root"), "READ"
        ) as archive:
            for plot in (source_plot, total_plot):
                self.assertTrue(archive.Get("canvases/" + plot["canvas"]))
            prefix = "systematics_qa/"
            suffix = "/cent_0_10__ch_0/raw/values"
            self.assertAlmostEqual(
                archive.Get(prefix + "sources/total" + suffix).GetBinContent(1), 0.1
            )
            self.assertAlmostEqual(
                archive.Get(prefix + "total" + suffix).GetBinContent(1), np.sqrt(0.14)
            )

    def test_root_export_preserves_bins_masks_and_plain_graphs(self):
        target = self.directory / "archive.root"
        with RootArchive(target) as archive:
            export_systematic_qa(archive, self.qa)
        ROOT = load_root()
        with ROOT.TFile.Open(str(target), "READ") as archive:
            self.assertFalse(archive.IsZombie())
            edges = np.asarray(self.qa["kstar_edges"], float)
            centers = (edges[:-1] + edges[1:]) / 2
            for path, source_values in all_qa_leaves(self.qa):
                with self.subTest(path=path):
                    histogram = archive.Get(path + "/values")
                    mask = archive.Get(path + "/valid")
                    graph = archive.Get(path + "/graph")
                    self.assertTrue(histogram and mask and graph)
                    self.assertEqual(graph.ClassName(), "TGraph")
                    values = np.asarray(source_values["values"], float)
                    valid = np.asarray(source_values["valid"], bool)
                    np.testing.assert_allclose(
                        [histogram.GetBinContent(i + 1) for i in range(len(values))],
                        values,
                        equal_nan=True,
                    )
                    np.testing.assert_array_equal(
                        [mask.GetBinContent(i + 1) for i in range(len(values))], valid
                    )
                    np.testing.assert_allclose(
                        [
                            histogram.GetXaxis().GetBinLowEdge(i + 1)
                            for i in range(len(edges))
                        ],
                        edges,
                    )
                    np.testing.assert_array_equal(
                        [histogram.GetBinError(i + 1) for i in range(len(values))],
                        np.zeros(len(values)),
                    )
                    self.assertEqual(graph.GetN(), int(valid.sum()))
                    np.testing.assert_allclose(
                        [graph.GetPointX(i) for i in range(graph.GetN())],
                        centers[valid],
                    )
                    np.testing.assert_allclose(
                        [graph.GetPointY(i) for i in range(graph.GetN())], values[valid]
                    )

    def test_rendered_archive_reopens_in_fresh_process_with_drawn_ratio_points(self):
        render_systematic_qa(self.directory, self.nominal, self.qa, self.config)
        qa_dir = self.directory / "qa"
        for name in ("results.json", "index.json", "systematics.root"):
            self.assertTrue((qa_dir / name).is_file(), name)
        pdfs = list(qa_dir.glob("*.pdf"))
        self.assertTrue(pdfs)
        for pdf in pdfs:
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
        self.assertFalse(
            [
                p
                for p in qa_dir.iterdir()
                if p.suffix.lower() in {".png", ".svg", ".jpg"}
            ]
        )
        saved = json.loads((qa_dir / "results.json").read_text())
        self.assertEqual(set(saved["variations"]), set(self.qa["variations"]))
        expected = {path: result for path, result in all_qa_leaves(self.qa)}
        write_json(self.directory / "expected.json", expected)
        program = r"""
import json,sys
from pathlib import Path
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(True)
root_path,expected_path,redraw_path=sys.argv[1:]
expected=json.loads(Path(expected_path).read_text())
def descendants(obj):
    yield obj
    if obj.InheritsFrom('TPad'):
        for child in obj.GetListOfPrimitives():
            yield from descendants(child)
with ROOT.TFile.Open(root_path,'READ') as archive:
    assert not archive.IsZombie()
    for path, source in expected.items():
        hist=archive.Get(path+'/values')
        graph=archive.Get(path+'/graph')
        assert hist and graph and graph.ClassName()=='TGraph',path
        values=np.asarray(source['values'],float)
        valid=np.asarray(source['valid'],bool)
        np.testing.assert_allclose([hist.GetBinContent(i+1) for i in range(len(values))],values,equal_nan=True)
        np.testing.assert_allclose([graph.GetPointY(i) for i in range(graph.GetN())],values[valid])
    directory=archive.GetDirectory('canvases')
    assert directory and directory.GetListOfKeys().GetSize()>0
    canvases=[directory.Get(key.GetName()) for key in directory.GetListOfKeys()]
    points=[]
    unity_lines=[]
    for canvas in canvases:
        assert canvas.InheritsFrom('TCanvas')
        assert canvas.GetListOfPrimitives().GetSize()>0
        for obj in descendants(canvas):
            if obj.InheritsFrom('TGraph') and obj.GetN()>0:
                points.extend(obj.GetPointY(i) for i in range(obj.GetN()))
            if obj.InheritsFrom('TLine') and obj.GetY1()==1 and obj.GetY2()==1:
                unity_lines.append(obj)
    assert any(abs(point-1.05)<1e-12 for point in points),points
    assert any(abs(point-0.7)<1e-12 for point in points),points
    assert unity_lines,'Missing nominal ratio=1 reference'
    canvases[0].SaveAs(redraw_path)
assert Path(redraw_path).read_bytes().startswith(b'%PDF-')
"""
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(qa_dir / "systematics.root"),
                str(self.directory / "expected.json"),
                str(self.directory / "redrawn.pdf"),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
