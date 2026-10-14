"""Check unselected counts and persisted input QA without requiring a model."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
import yaml

from pih3l.config import DEFAULTS, load_config
from pih3l.errors import AnalysisError
from pih3l.raw_qa import collect_raw_qa, empty_distribution, fill_distribution
from synthetic_data import create_test_inputs, write_test_config


class RawQATests(unittest.TestCase):
    def test_every_row_accounted_for_with_half_open_edges(self):
        hist = empty_distribution([0, 1, 2])
        fill_distribution(hist, [-1, 0, 1, 1.9, 2, np.nan, np.inf])
        np.testing.assert_array_equal(hist["counts"], [1, 2])
        self.assertEqual(
            (hist["underflow"], hist["overflow"], hist["nonfinite"]), (1, 1, 2)
        )
        self.assertEqual(
            hist["rows"],
            sum(hist["counts"])
            + hist["underflow"]
            + hist["overflow"]
            + hist["nonfinite"],
        )
        from pih3l.raw_qa_output import export_raw_qa
        from pih3l.root_io import RootArchive

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flow.root"
            with RootArchive(path) as archive:
                export_raw_qa(
                    archive, {"femto": {}, "hyper": {"example": {"mass": hist}}}
                )
            with ROOT.TFile.Open(str(path)) as archive:
                saved = archive.Get("raw_qa/hyper/example/mass")
                self.assertEqual(saved.GetEntries(), 5)
                self.assertEqual(saved.GetBinContent(0), 1)
                self.assertEqual(saved.GetBinContent(3), 1)
                self.assertEqual(saved.GetBinError(0), 1)
                self.assertEqual(saved.GetBinError(3), 1)

    def test_centrality_channels_chunking_and_no_selection(self):
        config = copy.deepcopy(DEFAULTS)
        config["raw_qa"].update(
            mass_range=[2.94, 3.06], mass_bins=12, kstar_edges=[0, 0.5, 1.2]
        )
        config["analysis"].update(
            centrality_bins=[[0, 10], [10, 50]], bdt_threshold=0.999
        )
        frame = pd.DataFrame(
            {
                "HypCentFT0C": [0, 9.9, 10, 49.9, 50],
                "ChargeChannel": [0, 0, 1, 3, 0],
                "IsMixed": [False, True, False, True, False],
                "mass_h3l": [2.95, np.nan, 3.04, 3.2, 2.99],
                "kstar": [1.1, 0.2, np.nan, 1.2, 0.1],
                "bdt_score": [0] * 5,
                "cos_pa": [-1] * 5,
            }
        )
        hyper = frame[["HypCentFT0C", "mass_h3l"]]
        frame = frame.drop(columns="mass_h3l")
        result = collect_raw_qa(config, [("femto", frame), ("background", hyper)])
        split = collect_raw_qa(
            config,
            [
                (role, part)
                for role, data in (("femto", frame), ("background", hyper))
                for part in (data.iloc[:2], data.iloc[2:])
            ],
        )
        from pih3l.io import json_safe

        self.assertEqual(json_safe(result), json_safe(split))
        first = result["femto"]["cent_0_10__ch_0"]
        self.assertEqual(sum(first["kstar_se"]["counts"]), 1)
        self.assertEqual(sum(first["kstar_me"]["counts"]), 1)
        self.assertNotIn("mass_se", first)
        self.assertNotIn("mass_me", first)
        self.assertEqual(result["hyper"]["cent_0_10"]["mass"]["nonfinite"], 1)
        second = result["femto"]["cent_10_50__ch_1"]
        self.assertEqual(second["kstar_se"]["nonfinite"], 1)
        self.assertEqual(result["femto"]["cent_10_50__ch_3"]["kstar_me"]["overflow"], 1)
        self.assertEqual(
            result["inputs"]["femto"], {"raw_rows": 5, "centrality_rows": 4}
        )

    def test_raw_qa_cli_without_signal_or_training_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pair_count = create_test_inputs(base / "input")
            path = base / "config.yaml"
            write_test_config(path, base / "input", base / "output")
            overrides = yaml.safe_load(path.read_text())
            overrides["inputs"] = {
                "signal": {"patterns": ["missing/*.root"], "required": True}
            }
            overrides["training"]["features"] = ["TofMass"]
            path.write_text(yaml.safe_dump(overrides))
            result = subprocess.run(
                [sys.executable, "-m", "pih3l", "raw-qa", "-c", str(path)],
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                    "OMP_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = base / "output/raw_qa"
            numerical = json.loads((output / "results.json").read_text())
            self.assertEqual(numerical["inputs"]["femto"]["raw_rows"], pair_count)
            self.assertFalse((base / "output/training").exists())
            self.assertFalse(list(output.glob("*.png")))
            self.assertEqual(len(list(output.glob("*.pdf"))), 3)
            self.assertFalse(list(output.glob("raw_mass*.pdf")))
            ROOT.gROOT.SetBatch(True)
            with ROOT.TFile.Open(str(output / "raw_qa.root")) as archive:
                hist = archive.Get("raw_qa/femto/cent_0_10__ch_0/kstar_se")
                self.assertTrue(hist)
                self.assertEqual(hist.GetEntries(), 180)
                self.assertEqual(hist.Integral(), 180)
                self.assertAlmostEqual(
                    sum(
                        hist.GetBinError(i) ** 2 for i in range(1, hist.GetNbinsX() + 1)
                    ),
                    180,
                )
                self.assertTrue(archive.Get("canvases/raw_kstar_cent0"))
                self.assertFalse(archive.Get("canvases/raw_mass_cent0"))
                self.assertFalse(archive.Get("raw_qa/femto/cent_0_10__ch_0/mass_se"))
                self.assertFalse(archive.Get("raw_qa/femto/cent_0_10__ch_0/mass_me"))
                hyper = archive.Get("raw_qa/hyper/cent_10_50/mass")
                self.assertEqual(hyper.GetEntries(), 3200)
            for pdf in output.glob("*.pdf"):
                self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))

    def test_config_rejects_invalid_qa_binning(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for overrides in (
                {"kstar_edges": [0, 1, 1]},
                {"mass_bins": 1},
                {"mass_range": [3, 2]},
            ):
                path.write_text(yaml.safe_dump({"raw_qa": overrides}))
                with self.subTest(overrides=overrides), self.assertRaises(
                    AnalysisError
                ):
                    load_config(path)
