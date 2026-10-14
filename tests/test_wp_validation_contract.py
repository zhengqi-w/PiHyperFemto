"""Independent WP acceptance oracles, without calling the production fitter."""

import importlib.util
import copy
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import ROOT

SPEC = importlib.util.spec_from_file_location(
    "testset_acceptance",
    Path(__file__).resolve().parents[1] / "scripts/validate_testset.py",
)
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)
ROOT.gROOT.SetBatch(True)


class RawMarginAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hipe4ml.model_handler import ModelHandler
        from xgboost import XGBClassifier

        cls.features = ["DcaV0Daug", "DcaHe", "DcaPi", "NSigmaHe", "cos_pa"]
        rng = np.random.default_rng(9865)
        cls.frame = pd.DataFrame({name: rng.normal(size=160) for name in cls.features})
        cls.labels = (cls.frame.DcaHe > 0).astype(int).to_numpy()
        model = XGBClassifier(
            n_estimators=20,
            max_depth=2,
            learning_rate=0.3,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            random_state=19,
        )
        cls.handler = ModelHandler(model, training_columns=cls.features)
        cls.handler.fit(cls.frame, cls.labels)

    def test_handler_matches_native_margin_and_probability_closure(self):
        probe = ACCEPTANCE.native_margin_probe(self.handler, self.frame)
        self.assertTrue(probe["handler_equals_native_margin"])
        self.assertTrue(probe["expit_margin_equals_probability"])
        self.assertLess(probe["native_margin_min"], 0)
        self.assertGreater(probe["native_margin_max"], 1)

    def test_probability_wrapper_cannot_pass_as_raw_margin(self):
        estimator = self.handler.get_original_model()

        class WrongProbabilityWrapper:
            def get_original_model(self):
                return estimator

            def predict(self, frame, output_margin):
                return estimator.predict_proba(frame)[:, 1]

        probe = ACCEPTANCE.native_margin_probe(WrongProbabilityWrapper(), self.frame)
        self.assertFalse(probe["handler_equals_native_margin"])
        self.assertTrue(probe["expit_margin_equals_probability"])

    def test_raw_margin_npz_requires_explicit_domain_without_probability_bounds(self):
        from pih3l.validation_scores import (
            write_validation_scores,
            read_validation_scores,
        )
        from pih3l.errors import AnalysisError

        expected = pd.DataFrame({"label": [0, 1, 1], "bdt_score": [-8.0, -0.3, 4.2]})
        with tempfile.TemporaryDirectory() as temporary:
            write_validation_scores(temporary, expected)
            pd.testing.assert_frame_equal(
                read_validation_scores(temporary), expected.astype({"label": "int8"})
            )
            for domain in (None, "probability"):
                payload = {"label": expected.label, "bdt_score": expected.bdt_score}
                if domain is not None:
                    payload["score_domain"] = np.asarray(domain)
                np.savez(Path(temporary) / "validation_scores.npz", **payload)
                with self.assertRaises(AnalysisError):
                    read_validation_scores(temporary)

    def test_margin_histogram_accepts_full_range_and_rejects_probability_clipping(self):
        scores = np.array([-8.0, -1.0, 0.3, 1.5, 4.2])
        edges = [-8, -2, 0, 2, 4.2]
        counts = np.histogram(scores, edges)[0]
        self.assertTrue(ACCEPTANCE.score_histogram_contract(scores, edges, counts))
        probability_edges = [0, 0.5, 1]
        self.assertFalse(
            ACCEPTANCE.score_histogram_contract(
                scores, probability_edges, np.histogram(scores, probability_edges)[0]
            )
        )
        self.assertFalse(
            ACCEPTANCE.score_histogram_contract(
                scores, edges, np.histogram(np.clip(scores, 0, 1), edges)[0]
            )
        )

    def test_null_storage_floor_preserves_negative_scores_and_zero_is_not_unbounded(
        self,
    ):
        scores = [-12.0, -2.0, 3.0]
        self.assertTrue(ACCEPTANCE.stored_margin_contract(scores, None))
        self.assertTrue(ACCEPTANCE.stored_margin_contract(scores, -13.0))
        self.assertFalse(ACCEPTANCE.stored_margin_contract(scores, 0.0))
        self.assertFalse(ACCEPTANCE.stored_margin_contract([-12.0, np.nan], None))
        self.assertTrue(
            ACCEPTANCE.stored_margin_contract(
                scores, {"low": None, "high": -5.0}, ["low", "high", "high"]
            )
        )
        self.assertFalse(
            ACCEPTANCE.stored_margin_contract(
                scores, {"low": None}, ["low", "unknown", "low"]
            )
        )

    def test_efficiency_one_endpoint_does_not_claim_unrestricted_application_acceptance(
        self,
    ):
        from pih3l.efficiency import thresholds_from_efficiencies

        cut = thresholds_from_efficiencies([0, 1, 1], [-9.0, -2.0, 3.0], [1.0])[0]
        self.assertEqual(cut, -9.0)
        self.assertTrue(ACCEPTANCE.stored_margin_contract([-12.0], None))
        self.assertFalse(ACCEPTANCE.stored_margin_contract([-12.0], cut))

    def test_actual_five_features_drive_qa_despite_extra_mass_and_topology_columns(
        self,
    ):
        from pih3l.training_qa import build_training_qa

        frame = self.frame.assign(
            label=self.labels,
            mass_h3l=np.where(self.labels, 2.991, 3.04),
            decay_radius=8.0,
            NTPCCrossedRowsHe=140,
            NTPCCrossedRowsPi=120,
        )
        training, validation = frame.iloc[:100], frame.iloc[100:]
        train_scores = self.handler.predict(training, output_margin=True)
        validation_scores = self.handler.predict(validation, output_margin=True)
        qa = build_training_qa(
            training, validation, train_scores, validation_scores, self.features, 20
        )
        self.assertEqual(
            set(qa["feature_distributions"]), set(self.features + ["mass_h3l"])
        )
        self.assertEqual(qa["correlations"]["features"], self.features)
        self.assertEqual(
            np.asarray(qa["correlations"]["signal"]["values"]).shape, (5, 5)
        )
        source = qa["score_distributions"]["test_all"]
        self.assertTrue(
            ACCEPTANCE.score_histogram_contract(
                validation_scores, source["edges"], source["counts"]
            )
        )
        for split, sample in [("train", training), ("test", validation)]:
            for label, category in [(0, "background"), (1, "signal")]:
                self.assertEqual(
                    sum(qa["score_distributions"][split + "_" + category]["counts"]),
                    int((sample.label == label).sum()),
                )

    def test_strict_mass_sampler_rejects_both_boundary_values_for_both_roles(self):
        from pih3l.config import DEFAULTS, merge
        from pih3l.training import sample_candidates

        low, high = 2.95, 3.02
        masses = [
            2.94,
            low,
            np.nextafter(low, np.inf),
            2.99,
            np.nextafter(high, -np.inf),
            high,
            3.03,
            np.nan,
        ]
        frame = pd.DataFrame(
            {
                "mass_h3l": masses,
                "DcaHe": np.ones(len(masses)),
                "_source_file": "toy",
                "_source_tree": "tree",
                "XPrimVtx": np.arange(len(masses), dtype=float),
                "YPrimVtx": 0.0,
                "ZPrimVtx": 0.0,
            }
        )
        config = merge(
            DEFAULTS,
            {
                "training": {
                    "features": ["DcaHe"],
                    "signal_mass_window": [low, high],
                    "candidate_cuts": {},
                    "pt_range": None,
                    "max_per_class": 100,
                    "min_per_class": 1,
                }
            },
        )
        reference = ACCEPTANCE.strict_role_mass_masks(masses, [low, high])
        self.assertFalse((reference["signal"] & reference["background"]).any())
        with tempfile.TemporaryDirectory() as temporary:
            frame.to_parquet(Path(temporary) / "sample.parquet", index=False)
            for role in ["signal", "background"]:
                sample, flow = sample_candidates(
                    temporary, role, config, return_cutflow=True
                )
                expected = np.asarray(masses)[reference[role]]
                np.testing.assert_array_equal(
                    np.sort(sample.mass_h3l), np.sort(expected)
                )
                self.assertEqual(flow["after_role_mass_selection"], len(expected))
                self.assertEqual(flow["sampled_rows"], len(expected))


class WorkingPointAcceptanceOracleTests(unittest.TestCase):
    def test_scope_exclusions_are_distinct_from_invalid_in_scope_scores(self):
        valid = {
            "rows": 100,
            "retained_rows": 30,
            "dropped_below_storage": 50,
            "invalid_scores": 20,
            "out_of_scope_rows": 20,
        }
        self.assertTrue(ACCEPTANCE.score_scope_accounting(valid))
        # A failed prediction inside the scope removes a row from both finite categories.
        self.assertFalse(
            ACCEPTANCE.score_scope_accounting(
                {**valid, "retained_rows": 29, "invalid_scores": 21}
            )
        )
        # Even if a finite out-of-scope score cancels that NaN count, accounting catches it.
        self.assertFalse(
            ACCEPTANCE.score_scope_accounting({**valid, "retained_rows": 29})
        )
        self.assertFalse(
            ACCEPTANCE.score_scope_accounting({**valid, "invalid_scores": 0})
        )

    def test_independent_quadrature_matches_roofit_dscb_and_chebyshev_integrals(self):
        for order in [1, 2]:
            for window in [[2.986, 2.997], [2.95, 2.990], [2.96, 3.04]]:
                with self.subTest(order=order, window=window):
                    prefix = "acceptance_" + uuid4().hex
                    mass = ROOT.RooRealVar(prefix + "_mass", "mass", 2.96, 3.04)
                    parameters = [
                        ROOT.RooRealVar(prefix + "_" + name, name, value)
                        for name, value in [
                            ("mean", 2.99131),
                            ("sigma", 0.0017),
                            ("alpha_left", 1.5),
                            ("n_left", 2.0),
                            ("alpha_right", 1.8),
                            ("n_right", 3.0),
                        ]
                    ]
                    signal = ROOT.RooCrystalBall(
                        prefix + "_signal", "signal", mass, *parameters
                    )
                    coefficients = [0.2] if order == 1 else [0.2, -0.1]
                    coefficient_variables = [
                        ROOT.RooRealVar(prefix + f"_c{i}", "c", value)
                        for i, value in enumerate(coefficients)
                    ]
                    background = ROOT.RooChebychev(
                        prefix + "_background",
                        "background",
                        mass,
                        ROOT.RooArgList(*coefficient_variables),
                    )
                    low, high = max(2.96, window[0]), min(3.04, window[1])
                    mass.setRange("window", low, high)
                    observables = ROOT.RooArgSet(mass)
                    signal_integral = signal.createIntegral(
                        observables,
                        ROOT.RooFit.NormSet(observables),
                        ROOT.RooFit.Range("window"),
                    )
                    background_integral = background.createIntegral(
                        observables,
                        ROOT.RooFit.NormSet(observables),
                        ROOT.RooFit.Range("window"),
                    )
                    reference = ACCEPTANCE.reference_mass_integrals(
                        [2.96, 3.04],
                        window,
                        signal_yield=100,
                        background_yield=250,
                        mean=2.99131,
                        sigma=0.0017,
                        alpha_left=1.5,
                        n_left=2,
                        alpha_right=1.8,
                        n_right=3,
                        background_coefficients=coefficients,
                    )
                    np.testing.assert_allclose(
                        reference,
                        [
                            100 * signal_integral.getVal(),
                            250 * background_integral.getVal(),
                        ],
                        rtol=1e-8,
                        atol=1e-9,
                    )

    def test_pol0_oracle_uses_s3_over_efficiency_and_ignores_fixed_window_signal(self):
        rows = [
            {
                "valid": True,
                "mc_signal_efficiency": efficiency,
                "raw_signal_3sigma_over_efficiency": value,
                "raw_signal_3sigma_over_efficiency_error": error,
                "signal_over_efficiency": 10000.0,
                "signal_over_efficiency_error": 0.001,
            }
            for efficiency, value, error in [
                (0.5, 100, 2),
                (0.6, 120, 4),
                (0.7, 90, 2),
                (0.99, 1000, 0.01),
            ]
        ]
        result = ACCEPTANCE.reference_pol0(rows, [0.5, 0.95])
        expected = np.average([100, 120, 90], weights=1 / np.square([2, 4, 2]))
        self.assertAlmostEqual(result["c0"], expected)
        self.assertEqual(result["points"], 3)
        self.assertAlmostEqual(
            result["c0_error"], 1 / np.sqrt(np.sum(1 / np.square([2, 4, 2])))
        )

    def test_canvas_contract_rejects_a_second_residual_panel(self):
        canvas = ROOT.TCanvas("acceptance_bad_mass", "", 800, 600)
        canvas.Divide(1, 2)
        references = []
        for index, name in [(1, "mass_data"), (2, "pull")]:
            canvas.cd(index)
            histogram = ROOT.TH1D(name, name, 5, 2.96, 3.04)
            histogram.SetDirectory(0)
            histogram.Draw()
            references.append(histogram)
        inspected = ACCEPTANCE.mass_canvas_contract(canvas)
        self.assertEqual(inspected["data_panels"], 2)
        self.assertTrue(inspected["has_residual_or_pull"])
        canvas.Close()

    def test_canvas_payload_detects_modified_graph_after_clone(self):
        name = "acceptance_clone_" + uuid4().hex
        canvas = ROOT.TCanvas(name, "original", 800, 600)
        graph = ROOT.TGraph(2)
        graph.SetName(name + "_graph")
        graph.SetPoint(0, 0.2, 1.0)
        graph.SetPoint(1, 0.5, 2.0)
        graph.Draw("ALP")
        cloned = canvas.Clone(name + "_clone")
        original = ACCEPTANCE.canvas_payload(canvas)
        self.assertEqual(original, ACCEPTANCE.canvas_payload(cloned))
        cloned.GetListOfPrimitives().FindObject(graph.GetName()).SetPoint(1, 0.5, 9.0)
        self.assertNotEqual(original, ACCEPTANCE.canvas_payload(cloned))
        cloned.Close()
        canvas.Close()

    def test_bounded_roofit_reported_error_differs_from_local_covariance(self):
        """Bounded Minuit parameter errors are not sqrt(covariance diagonal)."""
        prefix = "acceptance_bounded_" + uuid4().hex
        mass = ROOT.RooRealVar(prefix + "_mass", "mass", -5.0, 5.0)
        mean = ROOT.RooRealVar(prefix + "_mean", "mean", 0.0)
        sigma = ROOT.RooRealVar(prefix + "_sigma", "sigma", 1.0)
        signal = ROOT.RooGaussian(prefix + "_signal", "signal", mass, mean, sigma)
        signal_yield = ROOT.RooRealVar(prefix + "_yield", "yield", 98.0, 0.0, 100.0)
        model = ROOT.RooExtendPdf(prefix + "_model", "model", signal, signal_yield)
        observables = ROOT.RooArgSet(mass)
        data = ROOT.RooDataSet(prefix + "_data", "data", observables)
        for value in np.random.default_rng(102).normal(0, 1, 98):
            mass.setVal(float(value))
            data.add(observables)
        fitted = model.fitTo(
            data,
            ROOT.RooFit.Save(True),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Extended(True),
        )
        self.assertEqual(fitted.status(), 0)
        self.assertEqual(fitted.covQual(), 3)
        variance = float(fitted.covarianceMatrix()[0][0])
        # The independent extended-Poisson yield Hessian is N=98 counts.
        self.assertAlmostEqual(variance, 98.0, delta=0.1)
        self.assertFalse(np.isclose(signal_yield.getError() ** 2, variance, rtol=0.01))
        self.assertEqual(
            signal_yield.getError(), fitted.floatParsFinal().at(0).getError()
        )


class WorkingPointArtifactContractTests(unittest.TestCase):
    """Corrupt independent numerical fixtures to ensure acceptance fails usefully."""

    @classmethod
    def setUpClass(cls):
        mass_range = [2.96, 3.04]
        edges = np.linspace(*mass_range, 51)
        cls.config = {
            "working_point": {"yield_eff_range": [0.3, 0.95]},
            "analysis": {"bdt_threshold": None},
        }
        rows, points = [], []
        for efficiency, threshold, signal_yield, background_yield in [
            (0.4, 0.9, 60.0, 150.0),
            (0.6, 0.7, 90.0, 300.0),
            (0.8, 0.5, 120.0, 1000.0),
        ]:
            parameters = {
                "signal_yield": signal_yield,
                "background_yield": background_yield,
                "mean": 2.99131,
                "sigma": 0.0017,
                "alpha_left": 1.5,
                "n_left": 2.0,
                "alpha_right": 1.8,
                "n_right": 3.0,
                "c0": 0.2,
                "c1": 0.0,
            }
            integral_parameters = {
                key: value
                for key, value in parameters.items()
                if key not in {"c0", "c1"}
            }
            integral_parameters["background_coefficients"] = [0.2]

            def integral(window):
                return ACCEPTANCE.reference_mass_integrals(
                    mass_range, window, **integral_parameters
                )

            components = np.array([integral(edges[i : i + 2]) for i in range(50)])
            names = ["signal_yield", "background_yield", "mean", "sigma", "c0"]
            diagonal = [4.0, 9.0, 1e-8, 1e-8, 0.01]
            fit = {
                "signal_model": "dscb",
                "background_model": "pol1",
                "parameters": parameters,
                "parameter_names": names,
                "parameter_covariance": np.diag(diagonal).tolist(),
                "parameter_covariance_coordinates": "physical yields in counts",
                "n_free_parameters": len(names),
                "parameter_errors": dict(zip(names, np.sqrt(diagonal))),
                "minimizer_status": 0,
                "covariance_quality": 3,
                "signal_expected": components[:, 0],
                "background_expected": components[:, 1],
            }
            row = {
                "valid": True,
                "threshold": threshold,
                "mc_signal_efficiency": efficiency,
            }
            for multiplier in (2, 3, 4):
                window = [
                    parameters["mean"] - multiplier * parameters["sigma"],
                    parameters["mean"] + multiplier * parameters["sigma"],
                ]
                signal, background = integral(window)
                row.update(
                    {
                        f"window_{multiplier}sigma": window,
                        f"raw_signal_{multiplier}sigma": signal,
                        f"background_{multiplier}sigma": background,
                    }
                )
            row.update(
                {
                    "signal": row["raw_signal_3sigma"],
                    "background": row["background_3sigma"],
                    "signal_over_efficiency": row["raw_signal_3sigma"] / efficiency,
                    "raw_signal_3sigma_over_efficiency": row["raw_signal_3sigma"]
                    / efficiency,
                    "raw_signal_3sigma_over_efficiency_error": 2 / efficiency,
                }
            )
            rows.append(row)
            points.append({"fit": fit, "expected": components.sum(axis=1)})
        pol0 = ACCEPTANCE.reference_pol0(
            rows, cls.config["working_point"]["yield_eff_range"]
        )
        for row in rows:
            for multiplier in (2, 3, 4):
                row[f"significance_{multiplier}sigma"] = (
                    row["mc_signal_efficiency"]
                    * pol0["c0"]
                    / np.sqrt(pol0["c0"] + row[f"background_{multiplier}sigma"])
                )
            row["significance"] = row["significance_3sigma"]
        best = max(rows, key=lambda row: row["significance_3sigma"])
        cls.summary = {
            "resolved_settings": {
                "mass_range": mass_range,
                "mass_fit": {"mass_bins": 50, "background": "pol1"},
            },
            "scan_points": rows,
            "yield_pol0": pol0,
            "best_scan_point": best,
            "threshold": best["threshold"],
            "spectrum_qa": {"diagnostic_maximum": best},
        }
        cls.scan_fits = {"mass_edges": edges, "points": points}

    def inspect(self, summary=None, scan_fits=None):
        checks = {}
        ACCEPTANCE.validate_working_point_contract(
            self.config,
            summary or self.summary,
            scan_fits or self.scan_fits,
            lambda name, passed, detail="": checks.update({name: bool(passed)}),
        )
        return checks

    def test_independent_physical_fixture_passes_acceptance(self):
        checks = self.inspect()
        self.assertTrue(
            all(checks.values()),
            [name for name, passed in checks.items() if not passed],
        )

    def test_nonmaximum_selected_point_is_rejected(self):
        summary = copy.deepcopy(self.summary)
        summary["best_scan_point"] = min(
            summary["scan_points"], key=lambda row: row["significance_3sigma"]
        )
        checks = self.inspect(summary)
        self.assertFalse(checks["wp_best_is_displayed_3sigma_curve_argmax"])

    def test_fixed_window_pol0_cannot_replace_s3_normalization(self):
        summary = copy.deepcopy(self.summary)
        summary["yield_pol0"]["c0"] *= 1.2
        checks = self.inspect(summary)
        self.assertFalse(checks["wp_pol0_independently_uses_s3_over_efficiency"])
        self.assertFalse(checks["wp_all_windows_share_s3_pol0_and_correct_objective"])

    def test_normalized_yield_covariance_mislabeled_as_physical_is_rejected(self):
        scan_fits = copy.deepcopy(self.scan_fits)
        scan_fits["points"][0]["fit"]["parameter_covariance"][0][0] /= 10000
        checks = self.inspect(scan_fits=scan_fits)
        self.assertFalse(checks["wp_valid_fit_covariance"])

    def test_wrong_dscb_tail_integral_is_rejected(self):
        summary = copy.deepcopy(self.summary)
        summary["scan_points"][0]["raw_signal_4sigma"] *= 1.01
        self.assertFalse(self.inspect(summary)["wp_independent_sigma_integrals"])

    def test_component_totals_not_equal_to_physical_yields_are_rejected(self):
        scan_fits = copy.deepcopy(self.scan_fits)
        scan_fits["points"][0]["fit"]["signal_expected"] *= 1.1
        self.assertFalse(
            self.inspect(scan_fits=scan_fits)["wp_component_normalization"]
        )


if __name__ == "__main__":
    unittest.main()
