"""Standalone stage QA round trips preserve split and mass-scan semantics."""

import json
import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import ROOT

from pih3l.errors import AnalysisError
from pih3l.config import DEFAULTS
from pih3l.training_settings import create_model_handler
from pih3l.training_qa import (
    TRAINING_QA_ARTIFACTS,
    build_training_qa,
    render_training_qa,
)
from pih3l.working_point_qa import render_working_point_qa


class StageQATests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.train = pd.DataFrame(
            {
                "label": [0, 0, 1, 1],
                "feature": [0.0, 1.0, 3.0, 4.0],
                "constant": [1.0] * 4,
                "mass_h3l": [2.96, 3.04, 2.991, 2.992],
            }
        )
        self.test = pd.DataFrame(
            {
                "label": [0, 0, 1, 1],
                "feature": [0.2, 1.2, 3.2, 4.2],
                "constant": [1.0] * 4,
                "mass_h3l": [2.965, 3.035, 2.990, 2.993],
            }
        )
        self.train_scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.test_scores = np.array([0.15, 0.3, 0.7, 0.85])

    def tearDown(self):
        self.temporary.cleanup()

    def metrics(self):
        qa = build_training_qa(
            self.train,
            self.test,
            self.train_scores,
            self.test_scores,
            ["feature", "constant"],
            8,
        )
        return {
            "features": ["feature", "constant"],
            "qa": qa,
            "ks_signal": 0.8,
            "ks_background": 0.7,
            "score_efficiency_array": {
                "threshold": [0.9, 0.7, 0.0],
                "efficiency": [0.3, 0.8, 1.0],
                "actual_efficiency": [0.5, 1.0, 1.0],
            },
            "feature_importance": {"feature": 1.0, "constant": 0.0},
        }

    def assert_pdf(self, path):
        self.assertTrue(path.read_bytes().startswith(b"%PDF-"), path)
        self.assertGreater(path.stat().st_size, 1000)

    def test_compact_summary_preserves_class_split_and_undefined_correlations(self):
        qa = self.metrics()["qa"]
        json.dumps(qa, allow_nan=False)
        self.assertEqual(qa["rows"], {"train": 4, "test": 4})
        self.assertEqual(sum(qa["feature_distributions"]["mass_h3l"]["signal"]), 4)
        self.assertEqual(sum(qa["feature_distributions"]["feature"]["signal"]), 4)
        self.assertEqual(sum(qa["feature_distributions"]["feature"]["background"]), 4)
        for split in ("train", "test"):
            for category in ("signal", "background"):
                self.assertEqual(
                    sum(qa["score_distributions"][f"{split}_{category}"]["counts"]), 2
                )
            self.assertEqual(qa["roc"][split]["auc"], 1.0)
        self.assertEqual(sum(qa["score_distributions"]["test_all"]["counts"]), 4)
        self.assertIsNone(qa["correlations"]["signal"]["values"][1][1])
        self.assertFalse(qa["correlations"]["signal"]["valid"][1][1])
        with self.assertRaises(AnalysisError):
            build_training_qa(
                self.train, self.test, [0.2], self.test_scores, ["feature"], 8
            )

    def test_training_root_pdf_standalone_outputs_and_optuna_helper(self):
        metrics = self.metrics()
        metrics["optuna"] = {
            "enabled": True,
            "best_value": 0.92,
            "best_trial": 2,
            "trials": [
                {"number": 0, "state": "COMPLETE", "value": 0.88},
                {"number": 1, "state": "PRUNED", "value": None},
                {"number": 2, "state": "COMPLETE", "value": 0.92},
            ],
        }
        config = deepcopy(DEFAULTS)
        config["training"].update(
            features=metrics["features"], qa_bins=8, qa_shap_max_per_class=2
        )
        config["training"]["model_params"].update(
            n_estimators=8, max_depth=2, min_child_weight=0, gamma=0, n_jobs=1
        )
        handler = create_model_handler(config["training"], config["seed"])
        handler.fit(self.train, self.train.label.to_numpy())
        data = [
            self.train,
            self.train.label.to_numpy(),
            self.test,
            self.test.label.to_numpy(),
        ]
        from hipe4ml import plot_utils

        with patch.object(
            plot_utils,
            "plot_output_train_test",
            wraps=plot_utils.plot_output_train_test,
        ) as output_plot:
            paths = render_training_qa(self.directory, handler, data, metrics, config)
        self.assertIs(output_plot.call_args.args[0], handler)
        self.assertTrue(output_plot.call_args.kwargs["output_margin"])
        self.assertEqual(
            output_plot.call_args.kwargs["labels"], ["Background", "Signal"]
        )
        for name in TRAINING_QA_ARTIFACTS:
            self.assertTrue((self.directory / name).is_file(), name)
        self.assertFalse(list(self.directory.rglob("*.png")))
        self.assertFalse(list(self.directory.rglob("*.parquet")))
        for path in paths:
            if path.suffix == ".pdf":
                self.assert_pdf(path)
        with ROOT.TFile.Open(
            str(self.directory / "qa/training.root"), "READ"
        ) as archive:
            for split in ("train", "test"):
                for category in ("signal", "background"):
                    histogram = archive.Get(f"scores/{split}_{category}")
                    self.assertEqual(histogram.Integral(), 2)
                    self.assertEqual(histogram.GetEntries(), 2)
                self.assertGreater(archive.Get("roc/" + split).GetN(), 1)
            self.assertTrue(
                np.isnan(archive.Get("correlations/signal").GetBinContent(2, 2))
            )
            self.assertEqual(
                archive.Get("correlations/signal_valid").GetBinContent(2, 2), 0
            )
            self.assertEqual(archive.Get("training/optuna/trial_auc").GetN(), 2)
            self.assertFalse(archive.GetDirectory("canvases"))
            self.assertEqual(archive.Get("importance/gain").GetBinContent(1), 1)

    def test_standalone_ml_qa_initializes_root_before_shap_runtime(self):
        # A fresh interpreter catches ROOT/Numba library-order failures that a
        # suite importing ROOT up front cannot observe.
        script = r"""
import copy, sys
from pathlib import Path
import numpy as np
import pandas as pd
from pih3l.config import DEFAULTS
from pih3l.training_settings import create_model_handler
from pih3l.training_qa import build_training_qa, render_training_qa
config = copy.deepcopy(DEFAULTS)
config['training'].update(features=['feature'], qa_bins=8, qa_shap_max_per_class=2)
config['training']['model_params'].update(n_estimators=8, min_child_weight=0, gamma=0, n_jobs=1)
data = pd.DataFrame({'feature':[0.,1.,3.,4.], 'mass_h3l':[2.96,3.04,2.991,2.992], 'label':[0,0,1,1]})
labels = data.label.to_numpy()
handler = create_model_handler(config['training'], config['seed'])
handler.fit(data, labels)
scores = handler.predict(data, output_margin=True)
metrics = {'features':['feature'], 'feature_importance':{'feature':1.},
 'score_efficiency_array':{'threshold':[0.8,0.2], 'efficiency':[0.5,0.9], 'actual_efficiency':[0.5,1.]},
 'qa':build_training_qa(data, data, scores, scores, ['feature'], 8)}
render_training_qa(sys.argv[1], handler, [data, labels, data, labels], metrics, config)
assert len(list((Path(sys.argv[1])/'qa').glob('*.pdf'))) == 9
assert (Path(sys.argv[1])/'qa/training.root').is_file()
"""
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(self.directory / "matplotlib")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.directory / "standalone")],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @staticmethod
    def wp_fixture():
        rows = []
        for index, efficiency in enumerate((0.4, 0.7, 0.9)):
            row = {
                "threshold": 0.8 - 0.3 * index,
                "mc_signal_efficiency": efficiency,
                "actual_mc_signal_efficiency": efficiency + 0.01,
                "significance": 4.0 + index,
                "raw_significance": 3.0 + index,
                "raw_signal": 100 * efficiency,
                "raw_signal_error": 5.0,
                "signal_over_efficiency": 100.0,
                "signal_over_efficiency_error": 5.0 / efficiency,
                "mean": 2.991 + index * 0.0001,
                "mean_error": 0.0002,
                "sigma": 0.0014,
                "sigma_error": 0.0001,
                "chi2_ndf": 1.0 + index * 2,
                "fit_status": "ok",
                "valid": index < 2,
                "spectrum_qa_valid": index < 2,
                "raw_signal_3sigma_over_efficiency": 100.0,
                "raw_signal_3sigma_over_efficiency_error": 5.0 / efficiency,
            }
            for multiplier in (2, 3, 4):
                row[f"raw_signal_{multiplier}sigma"] = 100 * efficiency
                row[f"raw_signal_{multiplier}sigma_error"] = 5.0
                row[f"background_{multiplier}sigma"] = 10.0 * multiplier
                row[f"significance_{multiplier}sigma"] = (
                    efficiency * 100 / np.sqrt(100 + 10 * multiplier)
                )
            rows.append(row)
        edges = [2.98, 2.99, 3.0, 3.01]
        points = [
            {
                "index": index,
                "threshold": row["threshold"],
                "counts": [10, 30, 10],
                "expected": [11, 28, 11],
                "fit": {"status": "ok"},
                "scan_point": row,
            }
            for index, row in enumerate(rows)
        ]
        summary = {
            "scan_points": rows,
            "threshold": rows[1]["threshold"],
            "objective": "pol0_expected_significance",
            "significance_formula": "eff*c0/sqrt(c0+B)",
            "resolved_settings": {"signal_window": [2.989, 2.993]},
            "yield_pol0": {
                "status": "ok",
                "c0": 100.0,
                "c0_error": 4.0,
                "fit_range": [0.4, 0.9],
                "chi2_ndf": 0.0,
            },
        }
        summary["spectrum_qa"] = {
            "status": "ok",
            "yield_pol0": deepcopy(summary["yield_pol0"]),
            "nominal_point": {
                **rows[1],
                "efficiency_basis": "requested_scan_efficiency",
            },
        }
        best = {**points[1], "mass_edges": edges}
        scans = {"mass_edges": edges, "points": points}
        return summary, best, scans

    def test_wp_root_retains_rejected_fit_and_correct_pol0_denominator(self):
        summary, best, scans = self.wp_fixture()
        paths = render_working_point_qa(self.directory, summary, best, scans)
        for path in paths:
            if path.suffix == ".pdf":
                self.assert_pdf(path)
        self.assertFalse(list(self.directory.rglob("*.png")))
        self.assertFalse(list((self.directory / "qa").glob("mass_fit_*.pdf")))
        with ROOT.TFile.Open(
            str(self.directory / "qa/working_point.root"), "READ"
        ) as archive:
            self.assertEqual(archive.Get("mass_scan/point_0002/data").Integral(), 50)
            self.assertEqual(
                archive.Get("scan/significance_vs_score/significance").GetN(), 2
            )
            self.assertEqual(archive.Get("stability/chi2_ndf").GetN(), 3)
            corrected = archive.Get(
                "scan/raw_yield_over_efficiency/signal_over_efficiency"
            )
            self.assertAlmostEqual(corrected.GetPointX(0), 0.4)
            self.assertAlmostEqual(corrected.GetPointY(0), 100.0)
            self.assertAlmostEqual(
                archive.Get("scan/spectrum_yield_pol0_function").GetParameter(0), 100.0
            )
            metadata = json.loads(
                str(archive.Get("metadata/significance_definition").GetString())
            )
            self.assertEqual(metadata["signal_window"], [2.989, 2.993])
            self.assertTrue(archive.Get("best_mass_fit/expected"))

    def test_failed_wp_still_emits_all_fit_qa_and_optional_scan_frames(self):
        summary, _, scans = self.wp_fixture()
        summary.update(status="failed", threshold=None, save_score_fit_frames=True)
        summary["yield_pol0"] = {"status": "failed"}
        summary["spectrum_qa"].update(
            status="failed", yield_pol0={"status": "failed"}, nominal_point=None
        )
        for row in summary["scan_points"]:
            row["valid"] = False
            row["spectrum_qa_valid"] = False
        render_working_point_qa(self.directory, summary, {}, scans)
        self.assertFalse((self.directory / "qa/best_fit.pdf").exists())
        self.assertEqual(len(list((self.directory / "qa").glob("mass_fit_*.pdf"))), 3)
        with ROOT.TFile.Open(
            str(self.directory / "qa/working_point.root"), "READ"
        ) as archive:
            self.assertEqual(
                archive.Get("scan/significance_vs_score/significance").GetN(), 0
            )
            self.assertTrue(archive.Get("mass_scan/point_0002/fit"))


if __name__ == "__main__":
    unittest.main()
