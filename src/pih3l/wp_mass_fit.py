"""Dedicated spectrum-style RooFit DSCB/Chebychev working-point mass fits.

Classifier training and correlation-purity fits deliberately do not use this
module. Returned state is numerical: no RooFit objects or canvas references are
retained. Yield covariance coordinates are physical event counts, never fractions.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from uuid import uuid4

import numpy as np
from scipy.special import erf
from scipy.stats import chi2 as chi2_distribution

from .errors import AnalysisError
from .root_io import load_root

TAIL_DEFAULTS = {"alpha_left": 1.5, "n_left": 2.0, "alpha_right": 1.5, "n_right": 2.0}
TAIL_NAMES = tuple(TAIL_DEFAULTS)
PARAMETER_NAMES = (
    "signal_yield",
    "background_yield",
    "mean",
    "sigma",
    *TAIL_NAMES,
    "c0",
    "c1",
)


def _validated_histogram(counts, edges):
    counts, edges = np.asarray(counts, float), np.asarray(edges, float)
    if (
        counts.ndim != 1
        or edges.ndim != 1
        or len(edges) != len(counts) + 1
        or not np.isfinite(counts).all()
        or not np.isfinite(edges).all()
        or np.any(counts < 0)
        or np.any(np.diff(edges) <= 0)
    ):
        raise AnalysisError(
            "WP mass fit requires finite nonnegative counts and increasing bin edges"
        )
    return counts, edges


def _tail_shape(settings, supplied=None):
    defaults = {**TAIL_DEFAULTS, **settings.get("tail_defaults", {})}
    shape = {**defaults, **(supplied or {})}
    result = {name: float(shape[name]) for name in TAIL_NAMES}
    if any(not np.isfinite(value) or value <= 0 for value in result.values()):
        raise AnalysisError("DSCB tail alpha/n parameters must be finite and positive")
    return result


def _dscb_area(low, high, mean, sigma, alpha_left, n_left, alpha_right, n_right):
    """Exact finite-domain DSCB integral, including the n=1 logarithmic limit."""
    if sigma <= 0 or not np.isfinite(sigma):
        raise AnalysisError("DSCB sigma must be finite and positive")
    lo, hi = (low - mean) / sigma, (high - mean) / sigma
    if hi <= lo:
        return 0.0
    result = 0.0
    if lo < -alpha_left:
        end = min(hi, -alpha_left)
        a = (n_left / alpha_left) ** n_left * np.exp(-(alpha_left**2) / 2)
        b = n_left / alpha_left - alpha_left
        if abs(n_left - 1) < 1e-8:
            result += a * np.log((b - lo) / (b - end))
        else:
            result += (
                a
                * ((b - end) ** (1 - n_left) - (b - lo) ** (1 - n_left))
                / (n_left - 1)
            )
    core_low, core_high = max(lo, -alpha_left), min(hi, alpha_right)
    if core_high > core_low:
        result += np.sqrt(np.pi / 2) * (
            erf(core_high / np.sqrt(2)) - erf(core_low / np.sqrt(2))
        )
    if hi > alpha_right:
        start = max(lo, alpha_right)
        a = (n_right / alpha_right) ** n_right * np.exp(-(alpha_right**2) / 2)
        b = n_right / alpha_right - alpha_right
        if abs(n_right - 1) < 1e-8:
            result += a * np.log((b + hi) / (b + start))
        else:
            result += (
                a
                * ((b + start) ** (1 - n_right) - (b + hi) ** (1 - n_right))
                / (n_right - 1)
            )
    return float(sigma * result)


def _background_minimum(c0, c1):
    candidates = [1 - c0 + c1, 1 + c0 + c1]
    if c1 > 0:
        vertex = -c0 / (4 * c1)
        if -1 < vertex < 1:
            candidates.append(1 + c0 * vertex + c1 * (2 * vertex**2 - 1))
    return float(min(candidates))


def _component_fractions(parameters, mass_range, window):
    p = parameters
    values = [p[name] for name in PARAMETER_NAMES]
    if (
        not np.isfinite(values).all()
        or p["signal_yield"] < 0
        or p["background_yield"] < 0
        or p["sigma"] <= 0
    ):
        raise AnalysisError(
            "WP component parameters must be finite with nonnegative yields"
        )
    tails = _tail_shape({}, {name: p[name] for name in TAIL_NAMES})
    low, high = map(float, mass_range)
    if not np.isfinite([low, high, *window]).all() or not low < high:
        raise AnalysisError("WP component mass range must increase")
    start, end = max(low, window[0]), min(high, window[1])
    if _background_minimum(p["c0"], p["c1"]) <= 0:
        raise AnalysisError(
            "Chebychev background is not positive throughout the mass domain"
        )
    if end <= start:
        return 0.0, 0.0
    norm = _dscb_area(low, high, p["mean"], p["sigma"], **tails)
    signal = _dscb_area(start, end, p["mean"], p["sigma"], **tails) / norm
    width = high - low
    x1, x2 = (2 * start - low - high) / width, (2 * end - low - high) / width

    def primitive(x):
        return (1 - p["c1"]) * x + p["c0"] * x**2 / 2 + 2 * p["c1"] * x**3 / 3

    background = (primitive(x2) - primitive(x1)) / (2 - 2 * p["c1"] / 3)
    return float(signal), float(background)


def mass_components(parameters, edges):
    """DSCB and Chebychev integrals per bin, normalized on edges' full domain."""
    edges = np.asarray(edges, float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise AnalysisError("WP component bin edges must be finite and increasing")
    fractions = [
        _component_fractions(parameters, (edges[0], edges[-1]), (low, high))
        for low, high in zip(edges[:-1], edges[1:])
    ]
    signal, background = np.asarray(fractions, float).T
    return (
        signal * parameters["signal_yield"],
        background * parameters["background_yield"],
    )


def mass_densities(parameters, mass_range, points):
    """Continuous DSCB/Chebychev components in counts/GeV on the fit domain.

    This evaluates the normalized functions themselves. Histogram expectations
    remain exact bin integrals in ``mass_components``.
    """
    points = np.asarray(points, float)
    if points.ndim != 1 or not np.isfinite(points).all():
        raise AnalysisError("WP curve points must be a finite one-dimensional array")
    _component_fractions(parameters, mass_range, mass_range)
    low, high = map(float, mass_range)
    if np.any((points < low) | (points > high)):
        raise AnalysisError(
            "WP curve points must lie within the normalized mass domain"
        )
    p = parameters
    tails = _tail_shape({}, {name: p[name] for name in TAIL_NAMES})
    t = (points - p["mean"]) / p["sigma"]
    shape = np.exp(-0.5 * t**2)
    for side, mask, distance in (
        ("left", t < -p["alpha_left"], -t),
        ("right", t > p["alpha_right"], t),
    ):
        alpha, n = p["alpha_" + side], p["n_" + side]
        shape[mask] = (
            np.exp(-0.5 * alpha**2)
            * ((n / alpha) / (n / alpha - alpha + distance[mask])) ** n
        )
    norm = _dscb_area(low, high, p["mean"], p["sigma"], **tails)
    signal = p["signal_yield"] * shape / norm
    x = (2 * points - low - high) / (high - low)
    background = (
        p["background_yield"]
        * (1 + p["c0"] * x + p["c1"] * (2 * x**2 - 1))
        / ((high - low) * (1 - p["c1"] / 3))
    )
    return signal, background


def window_integrals(parameters, mass_range, window):
    """Physical S/B yields in an arbitrary window, clipped to the fit domain."""
    signal, background = _component_fractions(parameters, mass_range, window)
    return (
        signal * parameters["signal_yield"],
        background * parameters["background_yield"],
    )


@contextmanager
def _quiet_roofit(ROOT):
    service = ROOT.RooMsgService.instance()
    service.saveState()
    service.setGlobalKillBelow(ROOT.RooFit.ERROR)
    try:
        yield
    finally:
        service.restoreState()


def _settings(config):
    settings = config["working_point"]["mass_fit"]
    if settings.get("signal_model", "dscb") != "dscb" or settings["background"] not in {
        "pol1",
        "pol2",
    }:
        raise AnalysisError(
            "WP mass fitting supports DSCB with pol1/pol2 Chebychev only"
        )
    return settings


def _roofit_objects(counts, edges, settings, *, tails=None, initial=None, mc=False):
    ROOT = load_root()
    prefix = "wp_" + uuid4().hex + "_"
    mass = ROOT.RooRealVar(prefix + "mass", "m(^{3}He+#pi)", edges[0], edges[-1])
    mass.setBinning(ROOT.RooBinning(len(counts), array("d", edges)))
    hist = ROOT.TH1D(prefix + "hist", "", len(counts), array("d", edges))
    hist.SetDirectory(0)
    for index, count in enumerate(counts, 1):
        hist.SetBinContent(index, float(count))
        hist.SetBinError(index, float(np.sqrt(count)))
    data = ROOT.RooDataHist(prefix + "data", "", ROOT.RooArgList(mass), hist)
    total = float(counts.sum())
    initial = (initial or {}).get("parameters", initial or {})
    variables = {}

    def variable(name, default, bounds=None):
        value = float(initial.get(name, default))
        if bounds is None:
            if name in TAIL_NAMES:
                result = ROOT.RooRealVar(
                    prefix + name,
                    name,
                    value,
                    min(value / 2, 1e-6),
                    max(value * 2, 1e4),
                )
            else:
                result = ROOT.RooRealVar(prefix + name, name, value)
            result.setConstant(True)
        else:
            low, high = map(float, bounds)
            value = np.clip(
                value, low + (high - low) * 1e-6, high - (high - low) * 1e-6
            )
            result = ROOT.RooRealVar(prefix + name, name, float(value), low, high)
        variables[name] = result
        return result

    mean = variable("mean", 2.991, settings.get("mean_bounds", [2.985, 3.005]))
    sigma = variable("sigma", 0.0017, settings.get("sigma_bounds", [0.0014, 0.002]))
    shape = _tail_shape(settings, tails)
    for name in TAIL_NAMES:
        # Warm-start data fit tails cannot override the calibrated, frozen shape.
        saved = initial.pop(name, None) if isinstance(initial, dict) else None
        variable(
            name,
            shape[name],
            (
                settings.get("alpha_bounds", [0.7, 5])
                if mc and name.startswith("alpha")
                else (settings.get("n_bounds", [0.9, 15]) if mc else None)
            ),
        )
        if saved is not None:
            initial[name] = saved
    signal = ROOT.RooCrystalBall(
        prefix + "signal",
        "DSCB",
        mass,
        mean,
        sigma,
        variables["alpha_left"],
        variables["n_left"],
        variables["alpha_right"],
        variables["n_right"],
    )
    ns = variable("signal_yield", total if mc else 0.5 * total, [0, 5 * total])
    references = [mass, hist, data, signal, *variables.values()]
    if mc:
        pdf = ROOT.RooExtendPdf(prefix + "extended_signal", "", signal, ns)
        return ROOT, mass, data, signal, None, pdf, variables, references
    c0 = variable("c0", 0, [-10, 10])
    c1 = variable("c1", 0, [-10, 10] if settings["background"] == "pol2" else None)
    if settings["background"] == "pol1":
        c1.setVal(0)
    coefficients = ROOT.RooArgList(c0)
    if settings["background"] == "pol2":
        coefficients.add(c1)
    background = ROOT.RooChebychev(
        prefix + "background", settings["background"], mass, coefficients
    )
    nb = variable("background_yield", 0.5 * total, [0, 5 * total + 10])
    pdf = ROOT.RooAddPdf(
        prefix + "total",
        "",
        ROOT.RooArgList(signal, background),
        ROOT.RooArgList(ns, nb),
    )
    references.extend([c0, c1, nb, background, coefficients])
    return ROOT, mass, data, signal, background, pdf, variables, references


def _fit_to(ROOT, pdf, data, *, extended=True, ranges=None):
    options = [
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.PrintEvalErrors(-1),
        ROOT.RooFit.Warnings(False),
        ROOT.RooFit.Extended(extended),
        ROOT.RooFit.IntegrateBins(1e-6),
        ROOT.RooFit.Offset(True),
    ]
    if ranges is not None:
        options.append(ROOT.RooFit.Range(ranges))
    with _quiet_roofit(ROOT):
        return pdf.fitTo(data, *options)


def _empty_result(counts, edges, settings):
    return {
        "status": "insufficient_entries",
        "entries": float(counts.sum()),
        "signal_model": "dscb",
        "background_model": settings["background"],
        "parameters": None,
        "parameter_names": [],
        "parameter_covariance": [],
        "parameter_covariance_coordinates": "physical yields in counts; mean/sigma in GeV; dimensionless tail/Chebychev parameters",
        "n_free_parameters": 0,
        "expected": np.full(len(counts), np.nan),
        "signal_expected": np.full(len(counts), np.nan),
        "background_expected": np.full(len(counts), np.nan),
        "mass_edges": edges,
        "mass_range": [float(edges[0]), float(edges[-1])],
        "mass_bins": len(counts),
        "fit_method": "RooFit extended binned likelihood with integrated bin probabilities",
    }


def _extract_fit(result, variables, counts, edges, settings, *, mc=False):
    if not result:
        return {**_empty_result(counts, edges, settings), "status": "fit_failed"}
    parameters = {
        name: float(variable.getVal()) for name, variable in variables.items()
    }
    if mc:
        parameters.update(background_yield=0.0, c0=0.0, c1=0.0)
    floating = result.floatParsFinal()
    positions = {
        str(floating.at(index).GetName()): index for index in range(floating.getSize())
    }
    names = [
        name
        for name in PARAMETER_NAMES
        if name in variables and not variables[name].isConstant()
    ]
    matrix = result.covarianceMatrix()
    covariance = np.asarray(
        [
            [
                float(
                    matrix[positions[variables[left].GetName()]][
                        positions[variables[right].GetName()]
                    ]
                )
                for right in names
            ]
            for left in names
        ]
    )
    output = _empty_result(counts, edges, settings)
    output.update(
        parameters=parameters,
        parameter_names=names,
        parameter_covariance=covariance,
        n_free_parameters=len(names),
        minimizer_status=int(result.status()),
        covariance_quality=int(result.covQual()),
        edm=float(result.edm()),
        min_nll=float(result.minNll()),
        parameter_errors={
            name: (
                float(np.sqrt(covariance[index, index]))
                if covariance[index, index] >= 0
                else np.nan
            )
            for index, name in enumerate(names)
        },
        parameter_error_definition="sqrt(diagonal of physical-coordinate RooFit covariance); used for local covariance propagation",
        roo_reported_parameter_errors={
            name: float(variables[name].getError()) for name in names
        },
        roo_reported_parameter_error_definition="RooRealVar.getError; Minuit bounded-parameter transformation can differ from local covariance errors",
        fixed_parameters={
            name: value for name, value in parameters.items() if name not in names
        },
        background_positivity_minimum=_background_minimum(
            parameters["c0"], parameters["c1"]
        ),
        at_parameter_bounds=[
            name
            for name in names
            if min(
                abs(variables[name].getVal() - variables[name].getMin()),
                abs(variables[name].getMax() - variables[name].getVal()),
            )
            < 1e-5 * (variables[name].getMax() - variables[name].getMin())
        ],
    )
    if mc:
        output["background_model"] = None
    try:
        signal, background = mass_components(parameters, edges)
    except AnalysisError:
        output["status"] = "nonpositive_components"
        return output
    expected = signal + background
    active = (counts > 0) | (expected > 0)
    ndof = int(active.sum()) - len(names)
    positive = expected > 0
    deviance = (
        2
        * np.sum(
            expected
            - counts
            + np.where(
                counts > 0,
                counts
                * np.log(
                    np.divide(
                        counts,
                        expected,
                        out=np.ones_like(counts),
                        where=positive & (counts > 0),
                    )
                ),
                0,
            )
        )
        if positive.all()
        else np.nan
    )
    chi2 = float(
        np.sum((counts[active] - expected[active]) ** 2 / np.maximum(1, counts[active]))
    )
    output.update(
        signal_expected=signal,
        background_expected=background,
        expected=expected,
        chi2=chi2,
        ndof=ndof,
        chi2_ndf=chi2 / ndof if ndof > 0 else np.nan,
        deviance=float(deviance),
        fit_pvalue=float(chi2_distribution.sf(deviance, ndof)) if ndof > 0 else np.nan,
        chi2_definition="sum((observed-expected)^2/max(1,observed)); actual floated-parameter ndof",
    )
    valid_covariance = (
        covariance.shape == (len(names), len(names))
        and np.isfinite(covariance).all()
        and np.all(np.diag(covariance) > 0)
    )
    if valid_covariance:
        scale = np.sqrt(np.diag(covariance))
        valid_covariance = bool(
            np.linalg.eigvalsh(covariance / np.outer(scale, scale)).min() > 0
        )
    output["status"] = (
        "fit_failed"
        if result.status() != 0
        else (
            "invalid_fit_covariance"
            if result.covQual() < 3 or not valid_covariance
            else "invalid_fit_ndof" if ndof <= 0 else "ok"
        )
    )
    return output


def fit_wp_mass(counts, edges, config, tail_shape=None, initial=None):
    """Fit data with fixed MC/default DSCB tails and explicit pol1/pol2 background."""
    counts, edges = _validated_histogram(counts, edges)
    settings = _settings(config)
    output = _empty_result(counts, edges, settings)
    if (
        counts.sum()
        < max(1, settings.get("min_mass_entries", config["purity"]["min_mass_entries"]))
        or len(counts) <= 4
    ):
        return output
    ROOT, mass, data, signal, background, pdf, variables, references = _roofit_objects(
        counts, edges, settings, tails=tail_shape, initial=copy_parameters(initial)
    )
    prefit = {
        "enabled": bool(settings.get("sideband_prefit", True)),
        "status": "disabled",
    }
    if prefit["enabled"]:
        bands = config["analysis"]["sidebands"]
        mass.setRange("side_low", *map(float, bands[0]))
        mass.setRange("side_high", *map(float, bands[1]))
        result = _fit_to(
            ROOT, background, data, extended=False, ranges="side_low,side_high"
        )
        c0, c1 = variables["c0"].getVal(), variables["c1"].getVal()
        prefit.update(
            status=(
                "ok"
                if result and result.status() == 0 and result.covQual() == 3
                else "fit_failed"
            ),
            minimizer_status=int(result.status()) if result else None,
            covariance_quality=int(result.covQual()) if result else None,
            coefficients={"c0": float(c0), "c1": float(c1)},
            coefficient_errors={
                "c0": float(variables["c0"].getError()),
                "c1": (
                    float(variables["c1"].getError())
                    if settings["background"] == "pol2"
                    else 0.0
                ),
            },
            sidebands=bands,
            uncertainty_policy="coefficients frozen in full data fit; prefit uncertainty is not propagated into its conditional covariance",
        )
        if _background_minimum(c0, c1) <= 0:
            prefit["status"] = "nonpositive_background"
        if prefit["status"] != "ok":
            return {
                **output,
                "status": "sideband_prefit_failed",
                "sideband_prefit": prefit,
            }
        variables["c0"].setConstant(True)
        variables["c1"].setConstant(True)
    result = _fit_to(ROOT, pdf, data)
    output = _extract_fit(result, variables, counts, edges, settings)
    output.update(
        sideband_prefit=prefit,
        tail_shape=_tail_shape(settings, tail_shape),
        tail_source=(
            "supplied frozen calibration/default shape"
            if tail_shape is not None
            else "configured default tails; no MC calibration supplied"
        ),
        uncertainty="conditional on fixed DSCB tails; when sideband_prefit enabled, fixed Chebychev shape covariance is also excluded",
    )
    return output


def copy_parameters(initial):
    """Warm-start input is copied, so the caller's cached fit is never mutated."""
    return dict((initial or {}).get("parameters", initial or {}))


def fit_wp_mc_tail(counts, edges, config):
    """Calibrate DSCB tails on MC or return explicitly labeled configured defaults."""
    counts, edges = _validated_histogram(counts, edges)
    settings = _settings(config)
    defaults = _tail_shape(settings)
    fallback = {
        **_empty_result(counts, edges, settings),
        "status": "default_tail_fallback",
        "tail_shape": defaults,
        "tail_source": "configured defaults",
        "fixed_in_data_fit": True,
    }
    if not settings.get("calibrate_mc_tails", True):
        return {
            **fallback,
            "status": "default_tail_configured",
            "reason": "MC tail calibration explicitly disabled",
        }
    minimum = max(
        20, settings.get("min_mass_entries", config["purity"]["min_mass_entries"]) // 2
    )
    if counts.sum() < minimum or len(counts) <= 7:
        return {
            **fallback,
            "reason": "insufficient MC entries",
            "minimum_entries": minimum,
        }
    ROOT, mass, data, signal, background, pdf, variables, references = _roofit_objects(
        counts, edges, settings, mc=True
    )
    result = _extract_fit(
        _fit_to(ROOT, pdf, data), variables, counts, edges, settings, mc=True
    )
    result.update(
        tail_shape=(
            {name: result["parameters"][name] for name in TAIL_NAMES}
            if result["parameters"]
            else defaults
        ),
        fixed_in_data_fit=True,
        tail_source="MC extended binned DSCB fit",
    )
    if result["status"] != "ok":
        return {**fallback, "reason": result["status"], "calibration_fit": result}
    return result
