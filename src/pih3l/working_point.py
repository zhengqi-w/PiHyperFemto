"""BDT working point from invariant-mass fits to Hyper reconstruction data."""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .errors import AnalysisError
from .config import merge
from .io import iter_parquet, read_json, write_json
from .purity import mass_expectation
from .wp_mass_fit import fit_wp_mass
from .selection import cut_mask, in_window, in_windows
from .schema import require_columns
from .training import score_frame


def resolve_working_point_config(config):
    """Resolve independent Hyper selection/fit settings without changing Femto."""
    resolved = copy.deepcopy(config)
    wp = config["working_point"]
    for key in (
        "candidate_cuts",
        "event_cuts",
        "mass_range",
        "signal_window",
        "sidebands",
    ):
        if wp[key] is not None:
            resolved["analysis"][key] = copy.deepcopy(wp[key])
    # None/empty means no centrality cut; training and WP are not split by CF bins.
    resolved["analysis"]["centrality_bins"] = copy.deepcopy(wp["centrality_bins"] or [])
    resolved["purity"] = merge(config["purity"], wp["purity"] or {})
    resolved["purity"]["method"] = "fit"
    mass_fit = copy.deepcopy(wp["mass_fit"])
    for name in (
        "mass_bins",
        "min_mass_entries",
        "mean_bounds",
        "sigma_bounds",
        "background",
    ):
        if name in (wp["purity"] or {}):
            mass_fit[name] = wp["purity"][name]
        resolved["purity"][name] = mass_fit[name]
    resolved["working_point"]["mass_fit"] = mass_fit
    return resolved


def resolved_working_point_settings(config):
    return {
        **{
            key: config["analysis"][key]
            for key in (
                "candidate_cuts",
                "event_cuts",
                "centrality_bins",
                "mass_range",
                "signal_window",
                "sidebands",
            )
        },
        "purity": config["purity"],
        "mass_fit": config["working_point"]["mass_fit"],
        "period_text": config["working_point"]["period_text"],
        "additional_text": config["working_point"]["additional_text"],
        "signal_integration": "fitted DSCB mean +/- 3 core sigma, clipped to mass_range",
    }


def _signal_and_background(
    fit: dict, config: dict, *, sigma_multiplier=None
) -> tuple[float, float]:
    """Integrate fitted components in the fixed or moving diagnostic window."""
    parameters = fit.get("parameters")
    if not parameters:
        return np.nan, np.nan
    if "alpha_left" in parameters:
        from .wp_mass_fit import window_integrals

        window = _fitted_sigma_window(parameters, config, sigma_multiplier or 3)
        try:
            return window_integrals(
                parameters, config["analysis"]["mass_range"], window
            )
        except AnalysisError:
            return np.nan, np.nan
    ns = float(parameters["signal_yield"])
    nb = float(parameters["background_yield"])
    mean = float(parameters["mean"])
    sigma = float(parameters["sigma"])
    slope = float(parameters["slope"])
    mass_low, mass_high = config["analysis"]["mass_range"]
    signal_low, signal_high = (
        config["analysis"]["signal_window"]
        if sigma_multiplier is None
        else _fitted_sigma_window(parameters, config, sigma_multiplier)
    )
    if sigma <= 0 or signal_low >= signal_high:
        return np.nan, np.nan
    denominator = ndtr((mass_high - mean) / sigma) - ndtr((mass_low - mean) / sigma)
    signal = (
        ns
        * (ndtr((signal_high - mean) / sigma) - ndtr((signal_low - mean) / sigma))
        / denominator
    )
    width = mass_high - mass_low
    center = (mass_high + mass_low) / 2
    background = nb * (
        (signal_high - signal_low) / width
        + slope * ((signal_high - center) ** 2 - (signal_low - center) ** 2) / width**2
    )
    return float(signal), float(background)


def _fitted_sigma_window(parameters, config, multiplier):
    """Clip a fitted mean ± nσ window to the normalized mass-fit domain."""
    low, high = config["analysis"]["mass_range"]
    mean, sigma = parameters["mean"], parameters["sigma"]
    return [max(low, mean - multiplier * sigma), min(high, mean + multiplier * sigma)]


def _window_fit_statistics(fit, counts, edges, config, *, sigma_multiplier=None):
    """Propagate the fitted covariance, including explicit failed-fit results."""
    parameters = fit.get("parameters") or {}
    signal, background = _signal_and_background(
        fit, config, sigma_multiplier=sigma_multiplier
    )
    statistics = {
        "signal": signal,
        "raw_signal": signal,
        "signal_error": np.nan,
        "raw_signal_error": np.nan,
        "background": background,
        "mean": parameters.get("mean", np.nan),
        "sigma": parameters.get("sigma", np.nan),
        "mean_error": np.nan,
        "sigma_error": np.nan,
        "chi2_ndf": np.nan,
        "deviance_ndf": np.nan,
        "raw_significance": np.nan,
    }
    if not parameters:
        return statistics
    if "alpha_left" in parameters:
        return _dscb_window_statistics(
            fit, counts, edges, config, sigma_multiplier or 3
        )
    expected = mass_expectation(
        [
            parameters[name]
            for name in ("signal_yield", "background_yield", "mean", "sigma", "slope")
        ],
        edges,
        (
            "linear"
            if config["purity"]["background"] in {"linear", "pol1", "pol2"}
            else "constant"
        ),
    )
    linear = config["purity"]["background"] in {"linear", "pol1", "pol2"}
    size = 5 if linear else 4
    ndof = len(counts) - size
    if ndof > 0:
        # Match the spectrum's observed-count fit-quality criterion; retain the mass
        # fitter's likelihood deviance separately rather than confusing them.
        statistics["chi2_ndf"] = float(
            np.sum((counts - expected) ** 2 / np.maximum(1, counts)) / ndof
        )
        statistics["deviance_ndf"] = float(fit.get("deviance", np.nan) / ndof)
    if (
        np.isfinite(signal)
        and np.isfinite(background)
        and signal > 0
        and background >= 0
    ):
        statistics["raw_significance"] = float(signal / np.sqrt(signal + background))
    covariance = np.asarray(fit.get("parameter_covariance", []), dtype=float)
    total = float(np.sum(counts))
    if (
        covariance.shape != (size, size)
        or not np.isfinite(covariance).all()
        or total <= 0
    ):
        return statistics
    scale = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).tiny))
    if (
        np.linalg.eigvalsh(
            (covariance + covariance.T) / (2 * np.outer(scale, scale))
        ).min()
        <= 0
    ):
        return statistics
    point = np.asarray(
        [
            parameters["signal_yield"] / total,
            parameters["background_yield"] / total,
            parameters["mean"],
            parameters["sigma"],
        ]
        + ([parameters["slope"]] if linear else [])
    )
    steps = np.asarray([1e-4, 1e-4, 1e-6, 1e-6] + ([1e-4] if linear else []))

    def raw_yield(coordinates):
        varied = {"parameters": dict(parameters)}
        varied["parameters"].update(
            signal_yield=coordinates[0] * total,
            background_yield=coordinates[1] * total,
            mean=coordinates[2],
            sigma=coordinates[3],
            slope=coordinates[4] if linear else 0.0,
        )
        return _signal_and_background(
            varied, config, sigma_multiplier=sigma_multiplier
        )[0]

    gradient = np.asarray(
        [
            (
                raw_yield(point + np.eye(size)[index] * step)
                - raw_yield(point - np.eye(size)[index] * step)
            )
            / (2 * step)
            for index, step in enumerate(steps)
        ]
    )
    variance = float(gradient @ covariance @ gradient)
    if np.isfinite(variance) and variance > 0:
        statistics["signal_error"] = statistics["raw_signal_error"] = float(
            np.sqrt(variance)
        )
        statistics["mean_error"] = float(np.sqrt(covariance[2, 2]))
        statistics["sigma_error"] = float(np.sqrt(covariance[3, 3]))
    return statistics


def _dscb_window_statistics(fit, counts, edges, config, multiplier):
    """Propagate physical RooFit covariance with fixed MC/prefit shape parameters."""
    signal, background = _signal_and_background(
        fit, config, sigma_multiplier=multiplier
    )
    parameters = fit["parameters"]
    expected = np.asarray(fit["expected"], float)
    names = fit.get("parameter_names", [])
    nfree = fit.get("n_free_parameters", len(names))
    ndof = len(counts) - nfree
    statistics = {
        "signal": signal,
        "raw_signal": signal,
        "signal_error": np.nan,
        "raw_signal_error": np.nan,
        "background": background,
        "mean": parameters["mean"],
        "sigma": parameters["sigma"],
        "mean_error": np.nan,
        "sigma_error": np.nan,
        "chi2_ndf": np.nan,
        "deviance_ndf": np.nan,
        "raw_significance": np.nan,
    }
    if ndof > 0 and expected.shape == np.shape(counts) and np.isfinite(expected).all():
        statistics["chi2_ndf"] = float(
            np.sum((counts - expected) ** 2 / np.maximum(1, counts)) / ndof
        )
        statistics["deviance_ndf"] = float(fit.get("deviance", np.nan) / ndof)
    if np.isfinite(signal + background) and signal > 0 and background >= 0:
        statistics["raw_significance"] = float(signal / np.sqrt(signal + background))
    covariance = np.asarray(fit.get("parameter_covariance", []), float)
    if (
        covariance.shape != (len(names), len(names))
        or not len(names)
        or not np.isfinite(covariance).all()
    ):
        return statistics
    scale = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).tiny))
    if (
        np.linalg.eigvalsh(
            (covariance + covariance.T) / (2 * np.outer(scale, scale))
        ).min()
        <= 0
    ):
        return statistics
    gradient = []
    for name in names:
        value = parameters[name]
        step = max(abs(value) * 1e-5, 1e-7 if name in {"mean", "sigma"} else 1e-5)
        if name in {"mean", "sigma"}:
            step = 1e-7
        samples = []
        for sign in (-1, 1):
            varied = {"parameters": {**parameters, name: value + sign * step}}
            samples.append(
                _signal_and_background(varied, config, sigma_multiplier=multiplier)[0]
            )
        gradient.append((samples[1] - samples[0]) / (2 * step))
    gradient = np.asarray(gradient)
    variance = float(gradient @ covariance @ gradient)
    if np.isfinite(variance) and variance > 0:
        statistics["signal_error"] = statistics["raw_signal_error"] = float(
            np.sqrt(variance)
        )
    for name in ("mean", "sigma"):
        if name in names:
            statistics[name + "_error"] = float(
                np.sqrt(covariance[names.index(name), names.index(name)])
            )
    return statistics


def _component_expectations(fit, edges, config):
    if "expected" in fit:
        return np.asarray(fit["expected"], float)
    parameters = fit.get("parameters")
    if not parameters:
        return np.full(len(edges) - 1, np.nan)
    return mass_expectation(
        [
            parameters[name]
            for name in ("signal_yield", "background_yield", "mean", "sigma", "slope")
        ],
        edges,
        (
            "linear"
            if config["purity"]["background"] in {"linear", "pol1", "pol2"}
            else "constant"
        ),
    )


def _sigma_window_statistics(fit, counts, edges, config, efficiency):
    """Nominal 3sigma and 2/4sigma integration-window diagnostics."""
    result = {}
    parameters = fit.get("parameters")
    for multiplier in (2, 3, 4):
        values = _window_fit_statistics(
            fit, counts, edges, config, sigma_multiplier=multiplier
        )
        suffix = f"{multiplier}sigma"
        result.update(
            {
                f"raw_signal_{suffix}": values["signal"],
                f"raw_signal_{suffix}_error": values["signal_error"],
                f"background_{suffix}": values["background"],
                f"window_{suffix}": (
                    _fitted_sigma_window(parameters, config, multiplier)
                    if parameters
                    else [np.nan, np.nan]
                ),
                f"significance_{suffix}": np.nan,
            }
        )
    result["raw_signal_3sigma_over_efficiency"] = (
        result["raw_signal_3sigma"] / efficiency if efficiency > 0 else np.nan
    )
    result["raw_signal_3sigma_over_efficiency_error"] = (
        result["raw_signal_3sigma_error"] / efficiency if efficiency > 0 else np.nan
    )
    return result


def spectrum_qa_normalization(rows, settings, *, pol0=None):
    """Use one pol0(S3/ε) for all three spectrum background-window curves."""
    fit_rows = [
        {
            **row,
            "signal_over_efficiency": row["raw_signal_3sigma_over_efficiency"],
            "signal_over_efficiency_error": row[
                "raw_signal_3sigma_over_efficiency_error"
            ],
        }
        for row in rows
    ]
    pol0 = fit_yield_pol0(fit_rows, settings) if pol0 is None else pol0
    for row in rows:
        row["spectrum_qa_valid"] = bool(row["valid"] and pol0["status"] == "ok")
        if row["spectrum_qa_valid"]:
            for multiplier in (2, 3, 4):
                row[f"significance_{multiplier}sigma"] = expected_significance(
                    row["mc_signal_efficiency"],
                    pol0["c0"],
                    row[f"background_{multiplier}sigma"],
                )
            row["spectrum_qa_valid"] = bool(
                np.isfinite([row[f"significance_{n}sigma"] for n in (2, 3, 4)]).all()
            )
    valid = [row for row in rows if row["spectrum_qa_valid"]]
    return {
        "status": pol0["status"],
        "yield_pol0": pol0,
        "significance_formula": "requested_efficiency * c0_3sigma / sqrt(c0_3sigma + fitted_background_nsigma)",
        "signal_model": "DSCB (RooCrystalBall); fixed MC-calibrated or explicitly configured tail parameters",
        "window_convention": "fitted mean ± n sigma, clipped to mass_range; recomputed under covariance perturbations",
        "band_interpretation": "2σ–4σ integration-window envelope; not a statistical or systematic confidence band",
        "raw_signal_error": "full physical RooFit covariance of floated yield/shape parameters; conditional on fixed tails/prefit background; reference uses signal-yield error only",
        "nominal_selection": "maximum 3sigma expected significance; same S3/eff pol0 and B3 as the displayed curve",
        "diagnostic_maximum": (
            max(valid, key=lambda row: row["significance_3sigma"]) if valid else None
        ),
    }


def fit_yield_pol0(rows, settings):
    """ROOT weighted constant fit to valid raw signal / requested MC efficiency.

    Nested WP samples are correlated; the spectrum convention weights their
    marginal mass-fit errors and does not supply a cross-WP covariance matrix.
    """
    low, high = settings["yield_eff_range"]
    selected = [
        row
        for row in rows
        if row["valid"]
        and low <= row["mc_signal_efficiency"] <= high
        and np.isfinite(row["signal_over_efficiency"])
        and np.isfinite(row["signal_over_efficiency_error"])
        and row["signal_over_efficiency_error"] > 0
    ]
    distinct = len({row["threshold"] for row in selected})
    result = {
        "status": "insufficient_points",
        "fit_range": [low, high],
        "points": len(selected),
        "distinct_thresholds": distinct,
        "minimum_points": settings["min_pol0_points"],
        "c0": np.nan,
        "c0_error": np.nan,
        "chi2": np.nan,
        "ndof": 0,
        "chi2_ndf": np.nan,
        "method": "ROOT TGraphErrors weighted TF1 pol0 fit",
        "reference_difference": "preserves every valid efficiency point; avoids TH1 bin collisions overwriting scan points",
        "efficiency_coordinate": "requested MC signal efficiency",
        "uncertainty": "marginal mass-fit errors; covariance between nested WP samples unavailable",
    }
    if distinct < settings["min_pol0_points"]:
        return result
    from .root_io import load_root

    ROOT = load_root()
    graph = ROOT.TGraphErrors(len(selected))
    for index, row in enumerate(selected):
        graph.SetPoint(
            index, row["mc_signal_efficiency"], row["signal_over_efficiency"]
        )
        graph.SetPointError(index, 0.0, row["signal_over_efficiency_error"])
    function = ROOT.TF1(f"wp_yield_pol0_{uuid4().hex}", "pol0", low, high)
    # No analytical fallback: a failed ROOT fit is a failed WP determination.
    fit_result = graph.Fit(function, "QRSN")
    status = int(fit_result)
    c0, error = float(function.GetParameter(0)), float(function.GetParError(0))
    chi2, ndof = float(function.GetChisquare()), int(function.GetNDF())
    result.update(
        root_fit_status=status,
        c0=c0,
        c0_error=error,
        chi2=chi2,
        ndof=ndof,
        chi2_ndf=chi2 / ndof if ndof > 0 else np.nan,
    )
    result["status"] = (
        "ok"
        if status == 0
        and fit_result.IsValid()
        and np.isfinite(c0)
        and c0 > 0
        and np.isfinite(error)
        and error > 0
        and ndof > 0
        else "invalid_pol0_fit"
    )
    return result


def expected_significance(efficiency, c0, background):
    """Spectrum ProcessWP metric: efficiency*c0/sqrt(c0+B), not S/sqrt(S+B)."""
    if not (np.isfinite(c0) and c0 > 0 and np.isfinite(background) and background >= 0):
        return np.nan
    return float(efficiency * c0 / np.sqrt(c0 + background))


def _validation_thresholds(validation, efficiencies, *, return_metadata=False):
    from .efficiency import thresholds_from_efficiencies

    efficiencies = sorted(set(float(value) for value in efficiencies))
    values, details = thresholds_from_efficiencies(
        validation.label.to_numpy(),
        validation.bdt_score.to_numpy(),
        efficiencies,
        return_metadata=True,
    )
    mapped = dict(zip(efficiencies, map(float, values)))
    return (mapped, details) if return_metadata else mapped


def _candidate_selection(frame: pd.DataFrame, config: dict) -> np.ndarray:
    """Use only cuts defined on the independent Hyper candidate table."""
    mask = cut_mask(frame, config["analysis"]["candidate_cuts"])
    if config["working_point"]["event_cuts"] is not None:
        require_columns(
            frame, config["analysis"]["event_cuts"], "explicit WP event selection"
        )
    shared_event_cuts = {
        name: specification
        for name, specification in config["analysis"]["event_cuts"].items()
        if name in frame
    }
    mask &= cut_mask(frame, shared_event_cuts)
    mask &= np.asarray(in_window(frame.mass_h3l, config["analysis"]["mass_range"]))
    if config["analysis"]["centrality_bins"]:
        require_columns(frame, ["HypCentFT0C"], "WP candidate centrality selection")
        mask &= in_windows(frame.HypCentFT0C, config["analysis"]["centrality_bins"])
    return np.asarray(mask)


def _mc_signal_shape(source, bundle, config):
    """Bounded MC histogram under the same WP/model preselection."""
    from .training_support import mc_range_mask
    from .wp_mass_fit import fit_wp_mc_tail

    settings = config["working_point"]["mass_fit"]
    edges = np.linspace(*config["analysis"]["mass_range"], settings["mc_mass_bins"] + 1)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    raw_rows = selected_rows = 0
    if source is not None and settings["calibrate_mc_tails"]:
        features = bundle["features"]
        columns = (
            set(features) | set(config["analysis"]["candidate_cuts"]) | {"mass_h3l"}
        )
        pt_range = bundle.get("pt_range")
        if pt_range is not None:
            columns.add("pt_h3l")
        if config["analysis"]["centrality_bins"]:
            columns.add("HypCentFT0C")
        events = set(config["analysis"]["event_cuts"])
        if config["working_point"]["event_cuts"] is not None:
            columns |= events
        for frame in iter_parquet(
            source,
            columns=sorted(columns),
            optional_columns=sorted(events - columns),
            batch_size=config["io"]["chunk_rows"],
        ):
            raw_rows += len(frame)
            mask = (
                _candidate_selection(frame, config)
                & np.isfinite(frame[features]).all(axis=1).to_numpy()
            )
            if pt_range is not None:
                mask &= np.asarray(in_window(frame.pt_h3l, pt_range))
            mask &= mc_range_mask(
                frame, bundle.get("mc_range_selection", {}).get("ranges", {})
            )
            selected_rows += int(mask.sum())
            counts += np.histogram(frame.loc[mask, "mass_h3l"].to_numpy(float), edges)[
                0
            ]
    fit = fit_wp_mc_tail(counts, edges, config)
    return {
        "mass_edges": edges,
        "counts": counts,
        "fit": fit,
        "raw_rows": raw_rows,
        "preselected_rows": selected_rows,
        "in_mass_range_rows": int(counts.sum()),
        "input_status": "provided" if source is not None else "not_provided",
        "selection": "reconstructed truth MC, WP cuts, model pT/finite feature/range domain; no BDT score cut",
        "fit_domain": "binned MC histogram for bounded streaming; spectrum MC calibration is unbinned",
        "tail_shape": fit["tail_shape"],
    }


def scan_working_point(prepared_data, model_path, directory, config, *, mc_signal=None):
    """Choose the spectrum pol0 expected-significance WP from Hyper mass fits."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = resolve_working_point_config(config)
    from .models import load_model_bundle

    bundle = load_model_bundle(model_path)
    metrics = bundle["training_metrics"]
    mc_shape = _mc_signal_shape(mc_signal, bundle, config)
    write_json(directory / "mc_signal_fit.json", mc_shape)
    from .validation_scores import read_validation_scores

    validation = read_validation_scores(Path(model_path).parent)
    signal_scores = validation.loc[validation.label == 1, "bdt_score"].to_numpy(float)
    if not len(signal_scores):
        raise AnalysisError(
            "No held-out MC signal scores are available for the WP scan"
        )
    if not np.isfinite(signal_scores).all():
        raise AnalysisError(
            "Held-out MC signal scores must be finite for a fixed efficiency denominator"
        )

    efficiencies = config["working_point"]["signal_efficiencies"]
    dense = metrics.get("score_efficiency_array")
    if efficiencies is None and dense is not None:
        thresholds = [
            (float(efficiency), float(threshold))
            for efficiency, threshold in zip(dense["efficiency"], dense["threshold"])
        ]
        scan_grid_source = "trained_score_efficiency_array"
        scan_mapping_details = dense.get("mapping", {})
    else:
        if efficiencies is None:
            from .training_settings import efficiency_grid

            efficiencies = efficiency_grid(config["training"])
            scan_grid_source = "legacy_model_reconstructed_dense_grid"
        else:
            scan_grid_source = "working_point.signal_efficiencies_override"
        mapped, scan_mapping_details = _validation_thresholds(
            validation, efficiencies, return_metadata=True
        )
        thresholds = [
            (float(efficiency), mapped[float(efficiency)])
            for efficiency in efficiencies
        ]
    # Keep the complete requested efficiency grid, including score ties. Only
    # histograms/mass fits are shared by tied cuts; their MC efficiencies differ.

    edges = np.linspace(
        *config["analysis"]["mass_range"], config["purity"]["mass_bins"] + 1
    )
    deltas = sorted(
        {
            float(item["wp_efficiency_delta"])
            for item in config["systematics"]["variations"]
            if "wp_efficiency_delta" in item
        }
    )
    bases = [threshold for _, threshold in thresholds]
    if config["analysis"]["bdt_threshold"] is not None:
        bases.append(config["analysis"]["bdt_threshold"])
    all_thresholds = set(bases)
    additional_efficiencies = {
        config["working_point"]["storage"]["reserve_signal_efficiency"]
    }
    for threshold in bases:
        nominal_efficiency = float((signal_scores >= threshold).mean())
        for delta in deltas:
            target = nominal_efficiency + delta
            if 0 < target <= 1:
                additional_efficiencies.add(target)
    mapped_additional, additional_mapping_details = _validation_thresholds(
        validation, additional_efficiencies, return_metadata=True
    )
    all_thresholds.update(mapped_additional.values())
    # These integer histograms, not candidate arrays, survive between batches.
    counts_by_threshold = {
        threshold: np.zeros(len(edges) - 1, dtype=np.int64)
        for threshold in all_thresholds
    }
    raw_rows = selected_rows = 0
    requested_event_cuts = set(config["analysis"]["event_cuts"])
    unavailable_event_cuts = set()
    missing_event_cut_batches = {}
    pt_range = bundle.get("pt_range", config["training"]["pt_range"])
    from .training_support import mc_range_mask

    mc_ranges = bundle.get("mc_range_selection", {}).get("ranges", {})
    outside_mc_range_rows = 0
    columns = (
        set(bundle["features"])
        | set(config["analysis"]["candidate_cuts"])
        | {"mass_h3l"}
    )
    if pt_range is not None:
        columns.add("pt_h3l")
    if config["analysis"]["centrality_bins"]:
        columns.add("HypCentFT0C")
    if config["working_point"]["event_cuts"] is not None:
        columns.update(config["analysis"]["event_cuts"])
    for frame in iter_parquet(
        prepared_data,
        columns=sorted(columns),
        optional_columns=sorted(requested_event_cuts - columns),
        batch_size=config["io"]["chunk_rows"],
    ):
        raw_rows += len(frame)
        if len(frame):
            missing = requested_event_cuts - set(frame.columns)
            unavailable_event_cuts |= missing
            for name in missing:
                missing_event_cut_batches[name] = (
                    missing_event_cut_batches.get(name, 0) + 1
                )
        scored = score_frame(frame, bundle["model"], bundle["features"])
        mask = _candidate_selection(scored, config) & np.isfinite(scored.bdt_score)
        if pt_range is not None:
            mask &= np.asarray(in_window(scored.pt_h3l, pt_range))
        support = mc_range_mask(scored, mc_ranges)
        outside_mc_range_rows += int((mask & ~support).sum())
        mask &= support
        selected_rows += int(mask.sum())
        masses = scored.loc[mask, "mass_h3l"].to_numpy(float)
        scores = scored.loc[mask, "bdt_score"].to_numpy(float)
        for threshold, counts in counts_by_threshold.items():
            counts += np.histogram(masses[scores >= threshold], edges)[0]
    if not selected_rows:
        raise AnalysisError(
            "No Hyper reconstruction candidates survive the WP preselection"
        )
    fit_config = copy.deepcopy(config)
    fit_config["purity"]["method"] = "fit"
    rows, fits = [], []
    fit_cache = {}
    previous_parameters = None
    for efficiency, threshold in thresholds:
        counts = counts_by_threshold[threshold]
        if threshold not in fit_cache:
            fit_cache[threshold] = fit_wp_mass(
                counts,
                edges,
                fit_config,
                tail_shape=mc_shape["tail_shape"],
                initial=previous_parameters,
            )
        fit = fit_cache[threshold]
        if fit.get("status") == "ok":
            previous_parameters = fit.get("parameters")
        statistics = _window_fit_statistics(
            fit, counts, edges, fit_config, sigma_multiplier=3
        )
        cutoff = config["working_point"]["max_fit_chi2_ndf"]
        reasons = []
        if fit.get("status") != "ok":
            reasons.append(fit.get("status", "unknown"))
        if not np.isfinite(statistics["raw_significance"]):
            reasons.append("invalid_signal_or_background")
        if (
            not np.isfinite(statistics["signal_error"])
            or statistics["signal_error"] <= 0
        ):
            reasons.append("invalid_signal_covariance")
        if not np.isfinite(statistics["chi2_ndf"]):
            reasons.append("invalid_fit_ndof")
        elif cutoff is not None and statistics["chi2_ndf"] > cutoff:
            reasons.append("chi2_ndf_above_limit")
        row = {
            "threshold": threshold,
            "mc_signal_efficiency": efficiency,
            "actual_mc_signal_efficiency": float((signal_scores >= threshold).mean()),
            "data_candidates": int(counts.sum()),
            "fit_status": fit.get("status", "unknown"),
            "fit_pvalue": fit.get("fit_pvalue", np.nan),
            "background_model": fit.get(
                "background_model", config["working_point"]["mass_fit"]["background"]
            ),
            "roo_fit_status": fit.get("minimizer_status"),
            "covariance_quality": fit.get("covariance_quality"),
            "edm": fit.get("edm"),
            "n_free_parameters": fit.get("n_free_parameters"),
            **statistics,
            **_sigma_window_statistics(fit, counts, edges, fit_config, efficiency),
            "signal_over_efficiency": statistics["signal"] / efficiency,
            "signal_over_efficiency_error": statistics["signal_error"] / efficiency,
            "significance": np.nan,
            "expected_significance": np.nan,
            "valid": not reasons,
            "rejection_reasons": reasons,
        }
        rows.append(row)
        fits.append((fit, counts))
    pol0 = fit_yield_pol0(rows, config["working_point"])
    if pol0["status"] == "ok":
        for row in rows:
            if row["valid"]:
                row["significance"] = row["expected_significance"] = (
                    expected_significance(
                        row["mc_signal_efficiency"], pol0["c0"], row["background"]
                    )
                )
                row["expected_signal_base"] = pol0["c0"]
                row["expected_selected_signal"] = (
                    row["mc_signal_efficiency"] * pol0["c0"]
                )
    spectrum_qa = spectrum_qa_normalization(rows, config["working_point"], pol0=pol0)
    from .efficiency import mapping_metadata

    summary = {
        "status": "ok" if pol0["status"] == "ok" else "failed",
        "objective": "pol0_expected_significance",
        "significance_formula": "requested_efficiency * pol0_S3_over_efficiency / sqrt(pol0_S3_over_efficiency + fitted_background_3sigma)",
        "raw_significance_formula": "fitted_raw_signal_3sigma / sqrt(fitted_raw_signal_3sigma + fitted_background_3sigma)",
        "nominal_window": "DSCB fitted mean +/- 3 core sigma, clipped to mass_range",
        "mc_signal_fit": mc_shape,
        "yield_pol0": pol0,
        "yield_normalization": pol0,
        "spectrum_qa": spectrum_qa,
        "resolved_settings": resolved_working_point_settings(config),
        "save_score_fit_frames": config["working_point"]["save_score_fit_frames"],
        "scan_grid_source": scan_grid_source,
        "score_efficiency_mapping": {
            **mapping_metadata(),
            "scan_grid": scan_mapping_details,
            "variation_and_reserve_grid": additional_mapping_details,
        },
        "efficiency_grid_points": len(thresholds),
        "distinct_score_thresholds": len(fit_cache),
        "selection_domain": "independent Hyper reconstruction candidates",
        "application_domain": "Femto and MCFemto pair rows",
        "raw_hyper_candidates": raw_rows,
        "preselected_hyper_candidates": selected_rows,
        "mc_range_selection": bundle.get("mc_range_selection", {"enabled": False}),
        "outside_mc_range_candidates_after_other_cuts": outside_mc_range_rows,
        "unavailable_event_cuts": sorted(unavailable_event_cuts),
        "missing_event_cut_batches": missing_event_cut_batches,
        "scan_points": rows,
        "training_reference_threshold": metrics["threshold"],
        "model_id": bundle.get("model_id", "global"),
        "pt_range": bundle.get("pt_range", config["training"]["pt_range"]),
        "histogram_accumulation": "streamed fixed-size integer counts; no full Hyper mass/score arrays",
        "efficiency_denominator": "fixed held-out MC signal validation rows after training selection and model pT scope",
        "validation_signal_rows": len(signal_scores),
        "fit_quality": {
            "metric": "observed-count sum((data-expected)^2/max(1,data))/ndof",
            "ndof": "mass bins minus the actual number of floated RooFit parameters",
            "max_chi2_ndf": config["working_point"]["max_fit_chi2_ndf"],
            "mass_fit_status_required": "valid RooFit status/covariance/positive components and finite yield error",
        },
    }
    scan_fits = {
        "mass_edges": edges,
        "resolved_settings": resolved_working_point_settings(config),
        "points": [],
    }
    for index, (row, (fit, counts)) in enumerate(zip(rows, fits)):
        expected = _component_expectations(fit, edges, config)
        scan_fits["points"].append(
            {
                "index": index,
                "threshold": row["threshold"],
                "mc_signal_efficiency": row["mc_signal_efficiency"],
                "counts": counts,
                "expected": expected,
                "fit": fit,
                "scan_point": row,
            }
        )
    pd.DataFrame(rows).to_csv(directory / "scan.csv", index=False)
    write_json(directory / "scan_fits.json", scan_fits)
    if pol0["status"] != "ok":
        write_json(directory / "summary.json", summary)
        from .working_point_qa import render_working_point_qa

        render_working_point_qa(directory, summary, {}, scan_fits)
        raise AnalysisError(
            f"WP pol0 yield fit failed: {pol0['status']}; valid distinct cuts in yield_eff_range={pol0['distinct_thresholds']}, "
            f"required={pol0['minimum_points']}. Full scan diagnostics saved to {directory}"
        )
    valid_indices = [
        index
        for index, row in enumerate(rows)
        if row["valid"] and np.isfinite(row["significance"])
    ]
    best_index = max(valid_indices, key=lambda index: rows[index]["significance"])
    best = rows[best_index]
    configured = config["analysis"]["bdt_threshold"]
    resolved = float(best["threshold"] if configured is None else configured)
    source = (
        "hyper_data_pol0_expected_significance"
        if configured is None
        else "analysis.bdt_threshold_override"
    )
    best_fit = {
        **scan_fits["points"][best_index],
        "mass_edges": edges,
        "resolved_settings": resolved_working_point_settings(config),
        "yield_pol0": pol0,
    }
    write_json(directory / "best_fit.json", best_fit)
    summary.update(
        best_scan_point=best,
        threshold=resolved,
        threshold_source=source,
        nominal_mc_signal_efficiency=float((signal_scores >= resolved).mean()),
    )
    if resolved == best["threshold"]:
        spectrum_qa["nominal_point"] = {
            **best,
            "efficiency_basis": "requested scan efficiency",
        }
    else:
        # Evaluate an override at its exact already-accumulated cut. Never use
        # a neighbouring grid cut or the diagnostic 3σ optimum as the marker.
        nominal_efficiency = summary["nominal_mc_signal_efficiency"]
        counts = counts_by_threshold[resolved]
        if resolved not in fit_cache:
            fit_cache[resolved] = fit_wp_mass(
                counts,
                edges,
                fit_config,
                tail_shape=mc_shape["tail_shape"],
                initial=previous_parameters,
            )
        fit = fit_cache[resolved]
        statistics = _window_fit_statistics(
            fit, counts, edges, fit_config, sigma_multiplier=3
        )
        point = {
            "threshold": resolved,
            "mc_signal_efficiency": nominal_efficiency,
            "actual_mc_signal_efficiency": nominal_efficiency,
            "efficiency_basis": "actual held-out acceptance for explicit override",
            "fit_status": fit.get("status"),
            "valid": fit.get("status") == "ok"
            and np.isfinite(statistics["raw_significance"])
            and np.isfinite(statistics["signal_error"])
            and statistics["signal_error"] > 0
            and np.isfinite(statistics["chi2_ndf"])
            and (cutoff is None or statistics["chi2_ndf"] <= cutoff),
            **statistics,
            **_sigma_window_statistics(
                fit, counts, edges, fit_config, nominal_efficiency
            ),
        }
        point["spectrum_qa_valid"] = bool(
            point["valid"] and spectrum_qa["status"] == "ok"
        )
        if point["spectrum_qa_valid"]:
            for multiplier in (2, 3, 4):
                point[f"significance_{multiplier}sigma"] = expected_significance(
                    nominal_efficiency,
                    spectrum_qa["yield_pol0"]["c0"],
                    point[f"background_{multiplier}sigma"],
                )
            point["spectrum_qa_valid"] = bool(
                np.isfinite([point[f"significance_{n}sigma"] for n in (2, 3, 4)]).all()
            )
        spectrum_qa["nominal_point"] = point
    summary["efficiency_variations"] = []
    for delta in deltas:
        target = summary["nominal_mc_signal_efficiency"] + delta
        if not 0 < target <= 1:
            raise AnalysisError(
                f"WP efficiency variation {delta:+g} for {summary['model_id']} gives target {target:g} outside (0,1]"
            )
        threshold = mapped_additional[target]
        counts = counts_by_threshold[threshold]
        if threshold not in fit_cache:
            fit_cache[threshold] = fit_wp_mass(
                counts,
                edges,
                fit_config,
                tail_shape=mc_shape["tail_shape"],
                initial=previous_parameters,
            )
        fit = fit_cache[threshold]
        statistics = _window_fit_statistics(
            fit, counts, edges, fit_config, sigma_multiplier=3
        )
        variation = {
            "delta": delta,
            "threshold": threshold,
            "target_signal_efficiency": target,
            "actual_signal_efficiency": float((signal_scores >= threshold).mean()),
            "validation_signal_rows": len(signal_scores),
            "data_candidates": int(counts.sum()),
            "fit_status": fit.get("status", "unknown"),
            **statistics,
            "signal_over_efficiency": statistics["signal"] / target,
            "signal_over_efficiency_error": statistics["signal_error"] / target,
            "expected_significance": expected_significance(
                target, pol0["c0"], statistics["background"]
            ),
        }
        summary["efficiency_variations"].append(variation)
        write_json(
            directory / "variations" / f"delta_{delta:+.12g}.json",
            {**variation, "mass_edges": edges, "counts": counts, "fit": fit},
        )
    if summary["efficiency_variations"]:
        pd.DataFrame(summary["efficiency_variations"]).to_csv(
            directory / "efficiency_variations.csv", index=False
        )
    envelope = storage_envelope(
        signal_scores,
        resolved,
        summary["efficiency_variations"],
        config,
        threshold_mapper=lambda efficiency: mapped_additional[efficiency],
    )
    summary["storage_threshold"] = envelope["threshold"]
    summary["storage_envelope"] = envelope
    write_json(directory / "summary.json", summary)
    from .working_point_qa import render_working_point_qa

    render_working_point_qa(directory, summary, best_fit, scan_fits)
    return summary


def _efficiency_threshold(scores, efficiency):
    """Fixed validation denominator, with inclusive score acceptance and ties."""
    from .efficiency import efficiency_threshold

    return efficiency_threshold(scores, efficiency)


def storage_envelope(
    signal_scores, nominal_threshold, variations, config, *, threshold_mapper=None
):
    """Smallest inclusive score cut covering reserve and all declared selections."""
    storage = config["working_point"]["storage"]
    mapper = threshold_mapper or (
        lambda efficiency: _efficiency_threshold(signal_scores, efficiency)
    )
    reserve = (
        None
        if storage["reserve_signal_efficiency"] == 1
        else mapper(storage["reserve_signal_efficiency"])
    )
    declared = [
        {"source": "reserve_signal_efficiency", "threshold": reserve},
        {"source": "nominal", "threshold": float(nominal_threshold)},
    ]
    declared += [
        {
            "source": f"wp_efficiency_delta_{variation['delta']:+g}",
            "threshold": variation["threshold"],
        }
        for variation in variations
    ]
    for variation in config["systematics"]["variations"]:
        threshold = variation["overrides"].get("analysis", {}).get("bdt_threshold")
        if threshold is not None:
            declared.append(
                {"source": variation["name"], "threshold": float(threshold)}
            )
    threshold = (
        float(min(item["threshold"] for item in declared))
        if storage["enabled"] and reserve is not None
        else None
    )
    return {
        "enabled": storage["enabled"],
        "reserve_signal_efficiency": storage["reserve_signal_efficiency"],
        "reserve_threshold": reserve,
        "threshold": threshold,
        "actual_mc_signal_efficiency": (
            float((signal_scores >= threshold).mean()) if threshold is not None else 1.0
        ),
        "included_thresholds": declared,
        "selection": "finite in-scope margin >= storage_threshold; null means unrestricted finite scores",
        "decision": (
            "minimum of reserve, nominal and all declared WP/absolute variations; automatically expands beyond requested reserve when required"
            if threshold is not None
            else "pruning disabled or reserve=1; keep all finite in-scope margins"
        ),
    }


def scan_model_working_points(
    prepared_data, training_directory, directory, config, *, mc_signal=None
):
    """Each independently trained pT model has its own data-fit working point."""
    from .models import model_specs, scoped_config

    directory = Path(directory)
    registry_path = Path(training_directory) / "registry.json"
    if not registry_path.exists() or not config["training"]["pt_bins"]:
        return scan_working_point(
            prepared_data,
            Path(training_directory) / "model_handler.pkl",
            directory,
            config,
            mc_signal=mc_signal,
        )
    registry = {entry["id"]: entry for entry in read_json(registry_path)["models"]}
    results, entries = {}, []
    for spec in model_specs(config):
        if spec["id"] not in registry:
            raise AnalysisError(f"Missing trained model for WP: {spec['id']}")
        model = registry[spec["id"]]
        relative = f"models/{spec['id']}"
        summary = scan_working_point(
            prepared_data,
            Path(training_directory) / model["relative_path"] / "model_handler.pkl",
            directory / relative,
            scoped_config(config, spec),
            mc_signal=mc_signal,
        )
        results[spec["id"]] = summary
        entries.append(
            {"id": spec["id"], "pt_range": spec["pt_range"], "relative_path": relative}
        )
    summary = {
        "status": "ok",
        "mode": "pt_bins",
        "models": entries,
        "threshold": {
            model_id: value["threshold"] for model_id, value in results.items()
        },
        "storage_threshold": {
            model_id: value["storage_threshold"] for model_id, value in results.items()
        },
        "storage_envelope": {
            model_id: value["storage_envelope"] for model_id, value in results.items()
        },
        "objective": "pol0_expected_significance",
        "significance_formula": "requested_efficiency * c0 / sqrt(c0 + fitted_background_in_signal_window)",
        "efficiency_variations": [],
    }
    for delta in sorted(
        {
            float(item["wp_efficiency_delta"])
            for item in config["systematics"]["variations"]
            if "wp_efficiency_delta" in item
        }
    ):
        variations = {
            model_id: next(
                value
                for value in result["efficiency_variations"]
                if value["delta"] == delta
            )
            for model_id, result in results.items()
        }
        summary["efficiency_variations"].append(
            {
                "delta": delta,
                "threshold": {
                    model_id: value["threshold"]
                    for model_id, value in variations.items()
                },
                "models": variations,
            }
        )
    write_json(directory / "summary.json", summary)
    return summary


def resolve_wp_variation(summary, delta):
    """Resolve only explicitly prepared efficiency shifts; never use a global score."""
    for variation in summary.get("efficiency_variations", []):
        if np.isclose(variation["delta"], delta, rtol=0, atol=1e-12):
            return variation["threshold"]
    raise AnalysisError(
        f"WP efficiency variation {delta:+g} was not evaluated; rerun working_point with the configured variation"
    )
