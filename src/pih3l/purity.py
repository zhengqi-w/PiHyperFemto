"""Pair-weighted mass purity and sideband subtraction with explicit failure states."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2


def mass_expectation(parameters, edges, background="linear"):
    edges = np.asarray(edges, dtype=float)
    ns, nb, mean, sigma, slope = parameters
    gauss = np.diff(ndtr((edges - mean) / sigma))
    gauss /= ndtr((edges[-1] - mean) / sigma) - ndtr((edges[0] - mean) / sigma)
    width = edges[-1] - edges[0]
    center = (edges[-1] + edges[0]) / 2
    bg = np.diff(edges) / width
    if background == "linear":
        bg = (
            bg
            + slope
            * ((edges[1:] - center) ** 2 - (edges[:-1] - center) ** 2)
            / width**2
        )
    return ns * gauss + nb * bg


def _numerical_hessian(function, point, steps):
    size = len(point)
    result = np.empty((size, size))
    f0 = function(point)
    for i in range(size):
        ei = np.eye(size)[i] * steps[i]
        result[i, i] = (function(point + ei) - 2 * f0 + function(point - ei)) / steps[
            i
        ] ** 2
        for j in range(i):
            ej = np.eye(size)[j] * steps[j]
            result[i, j] = result[j, i] = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4 * steps[i] * steps[j])
    return result


def fit_purity(counts, edges, config):
    counts, edges = np.asarray(counts, float), np.asarray(edges, float)
    p = config["purity"]
    result = {
        "value": np.nan,
        "error": np.nan,
        "status": "not_evaluated",
        "basis": "selected_SE_pair_weighted_mass",
        "entries": int(counts.sum()),
        "uncertainty": "pair_independent_mass_likelihood_approximation",
    }
    if p["method"] == "none":
        result["status"] = "disabled"
        return result
    if p["method"] == "fixed":
        result.update(value=p["fixed"], error=p["fixed_error"], status="external_fixed")
        return result
    if counts.sum() < p["min_mass_entries"]:
        result["status"] = "insufficient_mass_entries"
        return result
    centers = (edges[1:] + edges[:-1]) / 2
    for low, high in config["analysis"]["sidebands"]:
        if counts[(centers >= low) & (centers < high)].sum() < 5:
            result["status"] = "insufficient_mass_sidebands"
            return result
    # Scale yield parameters by total entries for stable numerical conditioning.
    total = counts.sum()
    linear = p["background"] == "linear"

    def unpack(x):
        return np.array([x[0] * total, x[1] * total, x[2], x[3], x[4] if linear else 0])

    def nll(x):
        expected = mass_expectation(unpack(x), edges, p["background"])
        if np.any(expected <= 0):
            return 1e100
        return float(np.sum(expected - counts * np.log(expected)))

    initial = [0.4, 0.6, config["physics"]["mass_h3l"], 0.002]
    bounds = [(1e-6, 2), (1e-6, 2), tuple(p["mean_bounds"]), tuple(p["sigma_bounds"])]
    if linear:
        initial.append(0)
        bounds.append((-0.95, 0.95))
    fit = minimize(
        nll,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-6},
    )
    if not fit.success or not np.isfinite(fit.fun):
        result.update(status="fit_failed", message=str(fit.message))
        return result
    parameters = unpack(fit.x)
    expected = mass_expectation(parameters, edges, p["background"])
    nonzero = counts > 0
    deviance = 2 * (
        np.sum(expected - counts)
        + np.sum(counts[nonzero] * np.log(counts[nonzero] / expected[nonzero]))
    )
    result.update(
        parameters=dict(
            zip(
                ["signal_yield", "background_yield", "mean", "sigma", "slope"],
                parameters,
            )
        ),
        expected=expected,
        deviance=deviance,
        ndof=len(counts) - len(fit.x),
        fit_pvalue=chi2.sf(deviance, len(counts) - len(fit.x)),
    )
    if any(
        abs(x - low) < (high - low) * 1e-4 or abs(x - high) < (high - low) * 1e-4
        for x, (low, high) in zip(fit.x, bounds)
    ):
        result["status"] = "fit_at_boundary"
        return result

    def purity_value(x):
        ns, nb, mean, sigma, slope = unpack(x)
        low, high = config["analysis"]["signal_window"]
        signal = ns * (ndtr((high - mean) / sigma) - ndtr((low - mean) / sigma))
        signal /= ndtr((edges[-1] - mean) / sigma) - ndtr((edges[0] - mean) / sigma)
        width, center = edges[-1] - edges[0], (edges[-1] + edges[0]) / 2
        background = nb * (
            (high - low) / width
            + slope * ((high - center) ** 2 - (low - center) ** 2) / width**2
        )
        return signal / (signal + background)

    steps = np.array([1e-4, 1e-4, 1e-6, 1e-6] + ([1e-4] if linear else []))
    hessian = _numerical_hessian(nll, fit.x, steps)
    try:
        if np.linalg.eigvalsh(hessian).min() <= 0:
            raise np.linalg.LinAlgError("Nonpositive likelihood curvature")
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        result["status"] = "singular_fit_covariance"
        return result
    gradient = np.array(
        [
            (
                purity_value(fit.x + np.eye(len(steps))[i] * step)
                - purity_value(fit.x - np.eye(len(steps))[i] * step)
            )
            / (2 * step)
            for i, step in enumerate(steps)
        ]
    )
    result.update(
        value=purity_value(fit.x),
        error=float(np.sqrt(max(gradient @ covariance @ gradient, 0))),
        status="ok" if result["fit_pvalue"] > 1e-4 else "poor_fit_quality",
        parameter_covariance=covariance,
        parameter_covariance_basis={
            "coordinates": [
                "signal_yield/entries",
                "background_yield/entries",
                "mean",
                "sigma",
            ]
            + (["slope"] if linear else []),
            "yield_scale": float(total),
            "description": "Covariance uses normalized yield coordinates; multiply the first two Jacobian components by entries for physical yield errors.",
        },
    )
    return result


def correct_correlation(raw, background, purity, purity_error=0.0):
    """Ctrue=(Craw-(1-P)Cbkg)/P; shared-P covariance retained.

    Cross covariance of the mass fit and pair histograms is unknown, and omitted.
    Result metadata explicitly marks that approximation.
    """
    values = np.asarray(raw["values"], float)
    size = len(values)
    result = {
        "values": np.full(size, np.nan),
        "covariance": np.full((size, size), np.nan),
        "valid": np.zeros(size, bool),
        "status": "invalid_purity",
        "uncertainty": "independent_raw_sideband_purity_approximation",
    }
    if purity is None or not np.isfinite(purity) or not 0 < purity <= 1:
        return result
    if purity_error is None or not np.isfinite(purity_error) or purity_error < 0:
        return result
    if purity == 1 and purity_error == 0:
        return {
            "values": values.copy(),
            "covariance": np.asarray(raw["covariance"]).copy(),
            "valid": np.asarray(raw["valid"]).copy(),
            "status": raw["status"],
            "uncertainty": raw.get("uncertainty", "external_exact_unit_purity"),
        }
    bkg = np.asarray(background["values"], float)
    valid = np.asarray(raw["valid"]) & np.asarray(background["valid"])
    corrected = (values - (1 - purity) * bkg) / purity
    covariance = (
        np.asarray(raw["covariance"])
        + (1 - purity) ** 2 * np.asarray(background["covariance"])
    ) / purity**2
    derivative = (bkg - values) / purity**2
    covariance += np.outer(derivative, derivative) * purity_error**2
    covariance[~valid, :] = np.nan
    covariance[:, ~valid] = np.nan
    corrected[~valid] = np.nan
    result.update(
        values=corrected,
        covariance=covariance,
        valid=valid,
        status=(
            "ok"
            if valid.all()
            else (
                "partial_empty_bins"
                if valid.any()
                else "insufficient_sideband_or_signal"
            )
        ),
    )
    return result
