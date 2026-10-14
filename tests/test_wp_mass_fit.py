"""DSCB/Chebychev integrals and native RooFit WP fit contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from pih3l.config import DEFAULTS
from pih3l.errors import AnalysisError
from pih3l.root_io import load_root
from pih3l.wp_mass_fit import (
    TAIL_DEFAULTS,
    fit_wp_mass,
    fit_wp_mc_tail,
    mass_components,
    mass_densities,
    window_integrals,
)


class WorkingPointMassFitTests(unittest.TestCase):
    def settings(self, background="pol1", prefit=False):
        config = copy.deepcopy(DEFAULTS)
        config["working_point"]["mass_fit"].update(
            background=background, sideband_prefit=prefit, min_mass_entries=20
        )
        config["analysis"].update(
            mass_range=[2.96, 3.04], sidebands=[[2.96, 2.98], [3.005, 3.04]]
        )
        config["purity"]["min_mass_entries"] = 20
        return config

    def parameters(self, background="pol1"):
        return dict(
            signal_yield=4000.0,
            background_yield=6000.0,
            mean=2.991,
            sigma=0.0017,
            c0=0.1,
            c1=0.08 if background == "pol2" else 0.0,
            **TAIL_DEFAULTS
        )

    def test_component_bin_and_window_integrals_match_roofit_exact_shapes(self):
        ROOT = load_root()
        low, high = 2.96, 3.04
        for n in (0.9, 1.0, 2.0, 15.0):
            with self.subTest(n=n):
                p = dict(
                    self.parameters("pol2"),
                    alpha_left=1.2,
                    alpha_right=1.8,
                    n_left=n,
                    n_right=n,
                )
                tag = str(n).replace(".", "_")
                mass = ROOT.RooRealVar("integration_mass_" + tag, "", low, high)
                variables = {
                    name: ROOT.RooRealVar("integration_" + name + tag, "", p[name])
                    for name in (
                        "mean",
                        "sigma",
                        "alpha_left",
                        "n_left",
                        "alpha_right",
                        "n_right",
                        "c0",
                        "c1",
                    )
                }
                for name in ("alpha_left", "n_left", "alpha_right", "n_right"):
                    variables[name].setRange(0.1, 20.0)
                variables["sigma"].setRange(0.0001, 0.01)
                signal = ROOT.RooCrystalBall(
                    "integration_signal_" + tag,
                    "",
                    mass,
                    variables["mean"],
                    variables["sigma"],
                    variables["alpha_left"],
                    variables["n_left"],
                    variables["alpha_right"],
                    variables["n_right"],
                )
                background = ROOT.RooChebychev(
                    "integration_background_" + tag,
                    "",
                    mass,
                    ROOT.RooArgList(variables["c0"], variables["c1"]),
                )
                window = [2.989, 2.996]
                mass.setRange("window", *window)
                expected_s = (
                    signal.createIntegral(
                        ROOT.RooArgSet(mass),
                        ROOT.RooFit.NormSet(ROOT.RooArgSet(mass)),
                        ROOT.RooFit.Range("window"),
                    ).getVal()
                    * p["signal_yield"]
                )
                expected_b = (
                    background.createIntegral(
                        ROOT.RooArgSet(mass),
                        ROOT.RooFit.NormSet(ROOT.RooArgSet(mass)),
                        ROOT.RooFit.Range("window"),
                    ).getVal()
                    * p["background_yield"]
                )
                actual_s, actual_b = window_integrals(p, [low, high], window)
                self.assertAlmostEqual(actual_s / expected_s, 1.0, places=10)
                self.assertAlmostEqual(actual_b / expected_b, 1.0, places=10)
                positions = np.array([low, 2.985, p["mean"], 2.996, high])
                signal_density, background_density = mass_densities(
                    p, [low, high], positions
                )
                norm_set = ROOT.RooArgSet(mass)
                for x, s, b in zip(positions, signal_density, background_density):
                    mass.setVal(float(x))
                    self.assertAlmostEqual(
                        s / (signal.getVal(norm_set) * p["signal_yield"]),
                        1.0,
                        places=10,
                    )
                    self.assertAlmostEqual(
                        b / (background.getVal(norm_set) * p["background_yield"]),
                        1.0,
                        places=10,
                    )
                signal_bins, background_bins = mass_components(
                    p, np.linspace(low, high, 51)
                )
                self.assertAlmostEqual(signal_bins.sum(), p["signal_yield"], places=7)
                self.assertAlmostEqual(
                    background_bins.sum(), p["background_yield"], places=7
                )
                self.assertTrue((signal_bins >= 0).all())
                self.assertTrue((background_bins > 0).all())

    def test_global_chebychev_positivity_includes_interior_not_only_bin_centers(self):
        edges = np.linspace(2.96, 3.04, 51)
        with self.assertRaisesRegex(AnalysisError, "not positive throughout"):
            mass_components(dict(self.parameters("pol2"), c0=0, c1=1.1), edges)
        # Valid positive pol2 shapes need not satisfy the overrestrictive
        # triangle bound |c0|+|c1|<1; test the actual quadratic minimum instead.
        signal, background = mass_components(
            dict(self.parameters("pol2"), c0=0.7, c1=0.8), edges
        )
        self.assertTrue((background > 0).all())
        self.assertAlmostEqual(background.sum(), 6000, places=8)
        self.assertEqual(
            window_integrals(self.parameters(), [2.96, 3.04], [3.05, 3.06]), (0.0, 0.0)
        )

    def test_native_extended_fits_float_pol1_pol2_and_fix_tails(self):
        edges = np.linspace(2.96, 3.04, 51)
        for background in ("pol1", "pol2"):
            with self.subTest(background=background):
                p = self.parameters(background)
                signal, bg = mass_components(p, edges)
                counts = np.random.default_rng(234).poisson(signal + bg)
                fit = fit_wp_mass(
                    counts, edges, self.settings(background), tail_shape=TAIL_DEFAULTS
                )
                self.assertEqual(fit["status"], "ok")
                self.assertEqual(fit["covariance_quality"], 3)
                self.assertEqual(
                    fit["n_free_parameters"], 5 if background == "pol1" else 6
                )
                self.assertEqual(
                    fit["parameter_names"],
                    ["signal_yield", "background_yield", "mean", "sigma", "c0"]
                    + (["c1"] if background == "pol2" else []),
                )
                self.assertEqual(
                    np.asarray(fit["parameter_covariance"]).shape,
                    (fit["n_free_parameters"],) * 2,
                )
                self.assertGreater(fit["parameter_covariance"][0, 0], 1.0)
                np.testing.assert_allclose(
                    [fit["parameter_errors"][name] for name in fit["parameter_names"]],
                    np.sqrt(np.diag(fit["parameter_covariance"])),
                    rtol=1e-14,
                )
                self.assertEqual(
                    set(fit["roo_reported_parameter_errors"]),
                    set(fit["parameter_names"]),
                )
                for name, value in TAIL_DEFAULTS.items():
                    self.assertEqual(fit["parameters"][name], value)
                self.assertAlmostEqual(
                    fit["parameters"]["signal_yield"] / p["signal_yield"],
                    1.0,
                    delta=0.08,
                )
                self.assertAlmostEqual(
                    fit["parameters"]["sigma"] / p["sigma"], 1.0, delta=0.08
                )
                np.testing.assert_allclose(
                    fit["expected"], fit["signal_expected"] + fit["background_expected"]
                )
                self.assertIn(
                    "physical yields", fit["parameter_covariance_coordinates"]
                )
                self.assertIn("fixed DSCB tails", fit["uncertainty"])

    def test_prefit_freezes_coefficients_and_counts_only_actual_free_parameters(self):
        edges = np.linspace(2.96, 3.04, 51)
        for background in ("pol1", "pol2"):
            with self.subTest(background=background):
                signal, bg = mass_components(self.parameters(background), edges)
                fit = fit_wp_mass(
                    np.random.default_rng(234).poisson(signal + bg),
                    edges,
                    self.settings(background, True),
                    tail_shape=TAIL_DEFAULTS,
                )
                self.assertEqual(fit["status"], "ok")
                self.assertEqual(fit["sideband_prefit"]["status"], "ok")
                self.assertEqual(
                    fit["parameter_names"],
                    ["signal_yield", "background_yield", "mean", "sigma"],
                )
                self.assertEqual(fit["n_free_parameters"], 4)
                self.assertEqual(fit["ndof"], len(edges) - 1 - 4)
                for name in ("c0", "c1"):
                    self.assertEqual(
                        fit["parameters"][name],
                        fit["sideband_prefit"]["coefficients"][name],
                    )
                self.assertIn(
                    "Chebychev shape covariance is also excluded", fit["uncertainty"]
                )

    def test_mc_tail_calibration_and_explicit_default_fallback_states(self):
        edges = np.linspace(2.96, 3.04, 81)
        p = dict(self.parameters(), signal_yield=100000.0, background_yield=0.0)
        signal, _ = mass_components(p, edges)
        fit = fit_wp_mc_tail(signal, edges, self.settings())
        self.assertEqual(fit["status"], "ok")
        self.assertEqual(fit["n_free_parameters"], 7)
        self.assertTrue(fit["fixed_in_data_fit"])
        for name, value in TAIL_DEFAULTS.items():
            self.assertAlmostEqual(fit["tail_shape"][name], value, delta=0.02)
        low = fit_wp_mc_tail(np.zeros(len(edges) - 1), edges, self.settings())
        self.assertEqual(low["status"], "default_tail_fallback")
        self.assertEqual(low["tail_shape"], TAIL_DEFAULTS)
        self.assertEqual(low["reason"], "insufficient MC entries")
        config = self.settings()
        config["working_point"]["mass_fit"]["calibrate_mc_tails"] = False
        self.assertEqual(
            fit_wp_mc_tail(signal, edges, config)["status"], "default_tail_configured"
        )
        with patch("pih3l.wp_mass_fit._fit_to", return_value=None):
            failed = fit_wp_mc_tail(signal, edges, self.settings())
        self.assertEqual(failed["status"], "default_tail_fallback")
        self.assertEqual(failed["reason"], "fit_failed")
        self.assertEqual(failed["calibration_fit"]["status"], "fit_failed")

    def test_warm_start_does_not_mutate_cached_fit_or_override_frozen_tails(self):
        edges = np.linspace(2.96, 3.04, 51)
        signal, bg = mass_components(self.parameters(), edges)
        counts = np.random.default_rng(234).poisson(signal + bg)
        initial = {"parameters": dict(self.parameters(), alpha_left=4.5)}
        original = copy.deepcopy(initial)
        fit = fit_wp_mass(
            counts, edges, self.settings(), tail_shape=TAIL_DEFAULTS, initial=initial
        )
        self.assertEqual(initial, original)
        self.assertEqual(fit["parameters"]["alpha_left"], 1.5)
        self.assertEqual(fit["status"], "ok")

    def test_invalid_histogram_and_unsupported_background_fail_before_roofit(self):
        for counts, edges in (
            ([1, -1], [2.96, 3, 3.04]),
            ([1, 1], [3, 3, 3.04]),
            ([1, np.nan], [2.96, 3, 3.04]),
        ):
            with self.subTest(counts=counts, edges=edges), self.assertRaises(
                AnalysisError
            ):
                fit_wp_mass(counts, edges, self.settings())
        config = self.settings()
        config["working_point"]["mass_fit"]["background"] = "expo"
        with self.assertRaisesRegex(AnalysisError, "pol1/pol2"):
            fit_wp_mass(np.ones(50), np.linspace(2.96, 3.04, 51), config)


if __name__ == "__main__":
    unittest.main()
