#!/usr/bin/env python3
"""Validate an existing TestSet run, optionally execute it first.

Use the same Python environment as the analysis. No packages are installed.
Exit code 0 certifies software/artifact checks, never physics readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pih3l.config import load_config  # noqa: E402
from pih3l.io import (
    iter_parquet,
    parquet_columns,
    parquet_paths,
    write_json,
)  # noqa: E402
from pih3l.pipeline import Pipeline, STAGES, code_fingerprint  # noqa: E402
from pih3l.scored_storage import ensure_storage_coverage  # noqa: E402
from pih3l.validation_scores import read_validation_scores  # noqa: E402


def strict_json(path):
    def invalid(token):
        raise ValueError(f"Nonstandard JSON constant {token} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def score_scope_accounting(counts):
    """Scope exclusions have NaN scores; every in-scope row must be scored."""
    names = (
        "rows",
        "retained_rows",
        "invalid_scores",
        "dropped_below_storage",
        "out_of_scope_rows",
    )
    return (
        all(counts[name] >= 0 for name in names)
        and counts["invalid_scores"] == counts["out_of_scope_rows"]
        and counts["rows"]
        == counts["retained_rows"]
        + counts["dropped_below_storage"]
        + counts["out_of_scope_rows"]
    )


def strict_role_mass_masks(masses, window):
    """Independent strict mass-domain oracle; endpoints belong to neither role."""
    masses = np.asarray(masses, float)
    low, high = window
    finite = np.isfinite(masses)
    return {
        "signal": finite & (masses > low) & (masses < high),
        "background": finite & ((masses < low) | (masses > high)),
    }


def score_histogram_contract(scores, edges, counts):
    """Reject margin clipping, omitted tails, nonfinite scores and wrong bins."""
    scores, edges, counts = map(
        lambda values: np.asarray(values, float), (scores, edges, counts)
    )
    return (
        scores.ndim == 1
        and edges.ndim == 1
        and counts.ndim == 1
        and len(edges) == len(counts) + 1
        and len(counts) > 0
        and np.isfinite(scores).all()
        and np.isfinite(edges).all()
        and np.isfinite(counts).all()
        and (counts >= 0).all()
        and (np.diff(edges) > 0).all()
        and counts.sum() == len(scores)
        and np.array_equal(np.histogram(scores, edges)[0], counts)
    )


def stored_margin_contract(scores, minimum, model_ids=None):
    """Null is an unbounded storage floor; zero is a real margin threshold."""
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all():
        return False
    if isinstance(minimum, dict):
        if model_ids is None or len(model_ids) != len(scores):
            return False
        if any(model_id not in minimum for model_id in model_ids):
            return False
        floors = np.asarray(
            [
                -np.inf if minimum[model_id] is None else minimum[model_id]
                for model_id in model_ids
            ],
            float,
        )
    else:
        floors = -np.inf if minimum is None else float(minimum)
    return bool(np.all(scores >= floors))


def native_margin_probe(handler, frame):
    """Compare the wrapper with native XGBoost margins and probability closure."""
    from scipy.special import expit

    estimator = handler.get_original_model()
    wrapped = np.asarray(handler.predict(frame, output_margin=True), float)
    native = np.asarray(estimator.predict(frame, output_margin=True), float)
    probabilities = np.asarray(estimator.predict_proba(frame)[:, 1], float)
    return {
        "rows": len(frame),
        "finite": bool(np.isfinite(wrapped).all() and np.isfinite(native).all()),
        "handler_equals_native_margin": bool(
            np.allclose(wrapped, native, rtol=1e-7, atol=1e-7)
        ),
        "expit_margin_equals_probability": bool(
            np.allclose(expit(native), probabilities, rtol=2e-6, atol=2e-7)
        ),
        "native_margin_min": float(native.min()),
        "native_margin_max": float(native.max()),
        "handler_native_max_absolute_difference": float(
            np.max(np.abs(wrapped - native))
        ),
    }


def validate_training_contract(output, config, training, handler, check):
    """Audit raw-margin persistence, selected roles and actual ML input QA."""
    validation = read_validation_scores(output / "training")
    labels = validation.label.to_numpy()
    scores = validation.bdt_score.to_numpy(float)
    features = list(config["training"]["features"])
    qa = training["qa"]
    selection = training["training_selection"]
    index = strict_json(output / "training/qa/index.json")
    check(
        "training_raw_margin_metadata",
        training.get("score_output_margin") is True
        and training.get("score_domain") == "raw_margin"
        and index["score_domain"]["output_margin"] is True,
    )
    with np.load(
        output / "training/validation_scores.npz", allow_pickle=False
    ) as saved:
        check(
            "validation_npz_raw_margin_domain_and_precision",
            str(saved["score_domain"].item()) == "raw_margin"
            and saved["bdt_score"].dtype == np.dtype("float64")
            and np.isfinite(saved["bdt_score"]).all(),
        )
    check(
        "actual_model_features_match_all_training_qa",
        training["features"] == features
        and handler.get_training_columns() == features
        and set(qa["feature_distributions"]) == set(features + ["mass_h3l"])
        and qa["correlations"]["features"] == features
        and list(training["feature_importance"]) == features
        and index["features"] == features
        and all(
            plot["api_parameters"]["columns"]
            == (
                features + ["mass_h3l"]
                if plot["id"] == "feature_distributions"
                else features
            )
            for plot in index["plots"]
            if plot["id"] in {"feature_distributions", "correlations"}
        ),
    )
    check(
        "mass_is_distribution_qa_only_and_excluded_from_model_and_correlations",
        not ({"mass_h3l", "ct"} & set(features))
        and "ct" not in qa["feature_distributions"]
        and not ({"mass_h3l", "ct"} & set(qa["correlations"]["features"]))
        and index.get("qa_only_columns") == ["mass_h3l"],
    )
    window = config["training"]["signal_mass_window"]
    check(
        "training_strict_disjoint_mass_selection_metadata",
        selection["signal"]["mass_selection"]["window"] == window
        and selection["signal"]["mass_selection"]["comparison"] == "low < mass < high"
        and selection["background"]["excluded_mass_window"] == window
        and selection["background"]["comparison"]
        == "mass < low or mass > high; both endpoints excluded",
    )
    flows = training["input_cutflow"]
    stages = (
        "input_rows",
        "after_candidate_cuts",
        "after_pt_range",
        "after_role_mass_selection",
        "after_finite_features",
        "after_mc_range",
        "sampled_rows",
    )
    for label, role in ((1, "signal"), (0, "background")):
        flow = flows[role]
        values = [flow[name] for name in stages]
        check(
            "training_" + role + "_mass_cutflow_and_split_accounting",
            all(value is not None and value >= 0 for value in values)
            and all(left >= right for left, right in zip(values, values[1:]))
            and values[-1]
            == training["split"][str(label)]["train_rows"]
            + training["split"][str(label)]["validation_rows"],
        )
    support = selection.get("mc_range_selection", {})
    if support.get("enabled"):
        expected_support = (
            config["training"]["mc_range_selection"]["features"] or features
        )
        check(
            "mc_feature_support_uses_only_declared_model_inputs",
            support["features"] == expected_support
            and set(support["ranges"]) <= set(expected_support)
            and set(expected_support) <= set(features),
        )
    for label, category in ((0, "background"), (1, "signal")):
        source = qa["score_distributions"]["test_" + category]
        check(
            "held_out_" + category + "_histogram_exact_raw_margin_counts",
            score_histogram_contract(
                scores[labels == label], source["edges"], source["counts"]
            ),
        )
    source = qa["score_distributions"]["test_all"]
    check(
        "held_out_all_histogram_exact_raw_margin_counts",
        score_histogram_contract(scores, source["edges"], source["counts"]),
    )
    curve = training["score_efficiency_array"]
    signal_scores = scores[labels == 1]
    actual = [(signal_scores >= threshold).mean() for threshold in curve["threshold"]]
    check(
        "efficiency_mapping_uses_inclusive_raw_margin_scores",
        np.allclose(actual, curve["actual_efficiency"], rtol=0, atol=1e-12)
        and str(curve["mapping"]["score_domain"]).startswith("raw_margin"),
    )
    unit_cuts = [
        threshold
        for threshold, efficiency in zip(curve["threshold"], curve["efficiency"])
        if efficiency == 1
    ]
    if unit_cuts:
        check(
            "efficiency_one_is_finite_held_out_minimum",
            all(cut == scores.min() for cut in unit_cuts),
        )
    rng = np.random.default_rng(147293)
    probe = pd.DataFrame(
        {
            name: rng.uniform(
                qa["feature_distributions"][name]["edges"][0],
                qa["feature_distributions"][name]["edges"][-1],
                32,
            )
            for name in features
        }
    )
    probe_result = native_margin_probe(handler, probe)
    check(
        "loaded_handler_matches_native_xgboost_margin_and_probability_closure",
        all(
            probe_result[name]
            for name in (
                "finite",
                "handler_equals_native_margin",
                "expit_margin_equals_probability",
            )
        ),
    )
    return {
        "score_domain": "raw_margin",
        "features": features,
        "feature_count": len(features),
        "held_out_rows": len(scores),
        "held_out_margin_min": float(scores.min()),
        "held_out_margin_max": float(scores.max()),
        "held_out_scores_outside_probability_interval": int(
            ((scores < 0) | (scores > 1)).sum()
        ),
        "signal_mass_window": window,
        "mass_selection": {role: selection[role] for role in ("signal", "background")},
        "input_cutflow": flows,
        "native_model_probe": probe_result,
    }


def reference_mass_integrals(
    mass_range,
    window,
    *,
    signal_yield,
    background_yield,
    mean,
    sigma,
    alpha_left,
    n_left,
    alpha_right,
    n_right,
    background_coefficients,
):
    """Independent quadrature of finite-range DSCB and RooChebychev PDFs.

    This deliberately does not call the production fitter, mass expectation or
    integral helpers. Chebyshev coefficients multiply T1, T2, ... after mapping
    the configured mass domain to [-1, 1], as in RooChebychev.
    """
    from scipy.integrate import quad

    low, high = map(float, mass_range)
    window_low, window_high = max(low, float(window[0])), min(high, float(window[1]))
    shape = np.array([sigma, alpha_left, n_left, alpha_right, n_right], float)
    if not np.isfinite(shape).all() or np.any(shape <= 0) or low >= high:
        raise ValueError("DSCB shape and mass range must be physical")
    if window_low >= window_high:
        return 0.0, 0.0

    def signal_density(mass):
        t = (mass - mean) / sigma
        if t < -alpha_left:
            return (
                np.exp(-(alpha_left**2) / 2)
                * (n_left / alpha_left / (n_left / alpha_left - alpha_left - t))
                ** n_left
            )
        if t > alpha_right:
            return (
                np.exp(-(alpha_right**2) / 2)
                * (n_right / alpha_right / (n_right / alpha_right - alpha_right + t))
                ** n_right
            )
        return np.exp(-t * t / 2)

    def background_density(mass):
        coordinate = 2 * (mass - low) / (high - low) - 1
        return np.polynomial.chebyshev.chebval(
            coordinate, [1.0, *background_coefficients]
        )

    def integrate(function, start, stop):
        transitions = [mean - alpha_left * sigma, mean + alpha_right * sigma]
        return quad(
            function,
            start,
            stop,
            points=[x for x in transitions if start < x < stop],
            epsabs=1e-12,
            epsrel=1e-10,
        )[0]

    signal_norm = integrate(signal_density, low, high)
    background_norm = integrate(background_density, low, high)
    if signal_norm <= 0 or background_norm <= 0:
        raise ValueError("PDF normalization must be positive")
    return (
        float(
            signal_yield
            * integrate(signal_density, window_low, window_high)
            / signal_norm
        ),
        float(
            background_yield
            * integrate(background_density, window_low, window_high)
            / background_norm
        ),
    )


def reference_pol0(rows, efficiency_range):
    """Independent weighted constant fitted exclusively to S(3sigma)/epsilon."""
    low, high = efficiency_range
    points = [
        row
        for row in rows
        if row.get("valid")
        and low <= row["mc_signal_efficiency"] <= high
        and np.isfinite(row.get("raw_signal_3sigma_over_efficiency", np.nan))
        and np.isfinite(row.get("raw_signal_3sigma_over_efficiency_error", np.nan))
        and row.get("raw_signal_3sigma_over_efficiency_error", 0) > 0
    ]
    values = np.array(
        [row["raw_signal_3sigma_over_efficiency"] for row in points], float
    )
    errors = np.array(
        [row["raw_signal_3sigma_over_efficiency_error"] for row in points], float
    )
    if not len(points):
        raise ValueError("No valid S3/efficiency normalization points")
    weights = 1 / errors**2
    mean = float(np.sum(weights * values) / np.sum(weights))
    return {
        "c0": mean,
        "c0_error": float(1 / np.sqrt(weights.sum())),
        "chi2": float(np.sum(((values - mean) / errors) ** 2)),
        "points": len(points),
    }


def root_descendants(pad):
    """Yield persisted drawing primitives recursively, including nested ROOT pads."""
    for primitive in pad.GetListOfPrimitives():
        yield primitive
        if primitive.InheritsFrom("TPad"):
            yield from root_descendants(primitive)


def root_primitive_links(pad):
    """Yield persisted objects with their pad-local ROOT drawing options."""
    link = pad.GetListOfPrimitives().FirstLink()
    while link:
        primitive = link.GetObject()
        yield pad, primitive, str(link.GetOption())
        if primitive.InheritsFrom("TPad"):
            yield from root_primitive_links(primitive)
        link = link.Next()


def canvas_payload(canvas):
    """Stable numerical/text payload for comparing a stage canvas with its clone."""
    from pih3l.io import json_safe

    objects = []
    for _, item, option in root_primitive_links(canvas):
        if item.InheritsFrom("TPad"):
            continue
        data = {
            "class": item.ClassName(),
            "name": item.GetName(),
            "title": item.GetTitle(),
            "draw_option": option,
        }
        if item.InheritsFrom("TH1") or item.InheritsFrom("TGraph"):
            data["line_style"] = [
                item.GetLineColor(),
                item.GetLineStyle(),
                item.GetLineWidth(),
            ]
            data["marker_style"] = [
                item.GetMarkerColor(),
                item.GetMarkerStyle(),
                item.GetMarkerSize(),
            ]
        if item.InheritsFrom("TH2"):
            data["axis_titles"] = [
                item.GetXaxis().GetTitle(),
                item.GetYaxis().GetTitle(),
                item.GetZaxis().GetTitle(),
            ]
            data["contents"] = [
                [item.GetBinContent(i + 1, j + 1) for j in range(item.GetNbinsY())]
                for i in range(item.GetNbinsX())
            ]
        elif item.InheritsFrom("TH1"):
            data["axis_titles"] = [
                item.GetXaxis().GetTitle(),
                item.GetYaxis().GetTitle(),
            ]
            data["contents"] = [
                item.GetBinContent(i + 1) for i in range(item.GetNbinsX())
            ]
            data["errors"] = [item.GetBinError(i + 1) for i in range(item.GetNbinsX())]
            data["edges"] = [
                item.GetXaxis().GetBinLowEdge(i + 1)
                for i in range(item.GetNbinsX() + 1)
            ]
        elif item.InheritsFrom("TGraph"):
            data["points"] = [
                [item.GetPointX(i), item.GetPointY(i)] for i in range(item.GetN())
            ]
            if item.InheritsFrom("TGraphAsymmErrors"):
                data["errors"] = [
                    [
                        item.GetErrorXlow(i),
                        item.GetErrorXhigh(i),
                        item.GetErrorYlow(i),
                        item.GetErrorYhigh(i),
                    ]
                    for i in range(item.GetN())
                ]
            elif item.InheritsFrom("TGraphErrors"):
                data["errors"] = [
                    [item.GetErrorX(i), item.GetErrorY(i)] for i in range(item.GetN())
                ]
        elif item.InheritsFrom("TF1"):
            data["parameters"] = [item.GetParameter(i) for i in range(item.GetNpar())]
            data["parameter_errors"] = [
                item.GetParError(i) for i in range(item.GetNpar())
            ]
        elif item.InheritsFrom("TLegend"):
            data["labels"] = [entry.GetLabel() for entry in item.GetListOfPrimitives()]
        objects.append(data)
    return json_safe(objects)


def mass_canvas_contract(canvas):
    """Inspect saved data-bearing pads and annotation objects without repainting."""
    primitives = list(root_descendants(canvas))
    pads = [canvas, *[item for item in primitives if item.InheritsFrom("TPad")]]
    data_pads = [
        pad
        for pad in pads
        if any(
            obj.InheritsFrom("TH1") or obj.InheritsFrom("TGraph")
            for obj in pad.GetListOfPrimitives()
        )
    ]
    text = []
    for item in primitives:
        text.extend([str(item.GetName()), str(item.GetTitle())])
        if item.InheritsFrom("TH1"):
            text.extend([item.GetXaxis().GetTitle(), item.GetYaxis().GetTitle()])
        if item.InheritsFrom("TLegend"):
            text.extend(entry.GetLabel() for entry in item.GetListOfPrimitives())
    joined = " ".join(text).lower()
    return {
        "data_panels": len(data_pads),
        "has_legend": any(item.InheritsFrom("TLegend") for item in primitives),
        "has_residual_or_pull": "pull" in joined
        or "data - fit" in joined
        or "data-fit" in joined,
        "has_significance": "significance" in joined
        or "z_{" in joined
        or "z =" in joined,
        "has_chi2": "chi" in joined and ("ndf" in joined or "n.d.f" in joined),
        "function_curve_count": sum(
            item.InheritsFrom("TGraph") and "__curves__" in item.GetName()
            for item in primitives
        ),
        "has_drawn_model_histograms": any(
            item.InheritsFrom("TH1")
            and str(item.GetName()).endswith(("__expected", "__signal", "__background"))
            for item in primitives
        ),
        "text": joined,
    }


def validate_working_point_contract(config, summary, scan_fits, check):
    """Audit DSCB fit products against quadrature and an independent WP optimum."""
    settings = summary["resolved_settings"]
    mass_range = np.asarray(settings["mass_range"], float)
    mass_fit = settings["mass_fit"]
    edges = np.asarray(scan_fits["mass_edges"], float)
    rows = summary["scan_points"]
    points = scan_fits["points"]
    mc_shape = summary.get("mc_signal_fit")
    if mc_shape is not None:
        check(
            "wp_mc_tail_calibration_histogram_accounting",
            len(mc_shape["counts"]) == mass_fit["mc_mass_bins"]
            and len(mc_shape["mass_edges"]) == mass_fit["mc_mass_bins"] + 1
            and np.allclose(
                mc_shape["mass_edges"],
                np.linspace(*mass_range, mass_fit["mc_mass_bins"] + 1),
                rtol=0,
                atol=1e-12,
            )
            and sum(mc_shape["counts"]) == mc_shape["in_mass_range_rows"]
            and mc_shape["preselected_rows"] <= mc_shape["raw_rows"],
        )
        check(
            "wp_calibrated_or_explicit_default_tails_frozen_in_data_fits",
            mc_shape["fit"]["status"]
            in {"ok", "default_tail_configured", "default_tail_fallback"}
            and mc_shape["fit"]["fixed_in_data_fit"]
            and all(
                all(
                    np.isclose(point["fit"]["parameters"][name], value, rtol=1e-12)
                    and name not in point["fit"]["parameter_names"]
                    for name, value in mc_shape["tail_shape"].items()
                )
                for point in points
                if point["fit"].get("parameters")
            ),
        )
    check(
        "wp_mass_histogram_respects_resolved_configuration",
        len(edges) == mass_fit["mass_bins"] + 1
        and np.allclose(
            edges,
            np.linspace(*mass_range, mass_fit["mass_bins"] + 1),
            rtol=0,
            atol=1e-12,
        ),
    )
    check(
        "wp_spectrum_mass_histogram_2p96_3p04_50_bins",
        np.allclose(mass_range, [2.96, 3.04], rtol=0, atol=1e-12)
        and mass_fit["mass_bins"] == 50,
        "TestSet spectrum WP contract: 2.96–3.04 GeV/c², 50 uniform bins.",
    )
    checks = {
        key: []
        for key in (
            "dscb_chebyshev_model",
            "physical_parameters",
            "covariance_shape_and_units",
            "valid_fit_covariance",
            "component_normalization",
            "independent_component_bins",
            "independent_sigma_integrals",
            "nominal_aliases_are_3sigma",
        )
    }
    fitted = 0
    integrated = 0
    valid_fitted = 0
    representative_bins_checked = 0
    checked_thresholds = set()

    def record(key, index, passed):
        if not passed:
            checks[key].append(index)

    for index, (row, point) in enumerate(zip(rows, points)):
        fit = point["fit"]
        record(
            "dscb_chebyshev_model",
            index,
            fit.get("signal_model") == "dscb"
            and fit.get("background_model") == mass_fit["background"]
            and fit.get("background_model") in {"pol1", "pol2"},
        )
        parameters = fit.get("parameters")
        if not parameters:
            continue
        fitted += 1
        shape_names = ("sigma", "alpha_left", "n_left", "alpha_right", "n_right")
        physical = (
            all(np.isfinite(value) for value in parameters.values())
            and all(parameters[name] > 0 for name in shape_names)
            and parameters["signal_yield"] >= 0
            and parameters["background_yield"] >= 0
            and mass_range[0] <= parameters["mean"] <= mass_range[1]
            and (fit["background_model"] != "pol1" or parameters["c1"] == 0)
        )
        record("physical_parameters", index, physical or not row["valid"])
        names = fit.get("parameter_names", [])
        covariance = np.asarray(fit.get("parameter_covariance", []), float)
        covariance_shape = (
            covariance.shape == (len(names), len(names))
            and len(set(names)) == len(names)
            and all(name in parameters for name in names)
            and len(names) == fit.get("n_free_parameters")
            and "physical yields in counts"
            in fit.get("parameter_covariance_coordinates", "")
        )
        record("covariance_shape_and_units", index, covariance_shape)
        if row["valid"]:
            valid_fitted += 1
            valid_covariance = (
                covariance_shape and bool(names) and np.isfinite(covariance).all()
            )
            if valid_covariance:
                diagonal = covariance.diagonal()
                valid_covariance = (
                    (diagonal > 0).all()
                    and np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
                    and np.allclose(
                        diagonal,
                        [fit["parameter_errors"][name] ** 2 for name in names],
                        rtol=5e-5,
                        atol=1e-15,
                    )
                )
                if valid_covariance:
                    correlation = covariance / np.sqrt(np.outer(diagonal, diagonal))
                    valid_covariance = np.linalg.eigvalsh(correlation).min() >= -1e-9
            record(
                "valid_fit_covariance",
                index,
                valid_covariance
                and fit.get("minimizer_status") == 0
                and fit.get("covariance_quality", -1) >= 3,
            )
        signal_bins = np.asarray(fit.get("signal_expected", []), float)
        background_bins = np.asarray(fit.get("background_expected", []), float)
        expected = np.asarray(point["expected"], float)
        finite_components = (
            len(signal_bins) == len(edges) - 1
            and len(background_bins) == len(edges) - 1
            and np.isfinite(signal_bins).all()
            and np.isfinite(background_bins).all()
        )
        normalized_components = (
            finite_components
            and (signal_bins >= -1e-10).all()
            and (background_bins >= -1e-10).all()
            and np.allclose(
                [signal_bins.sum(), background_bins.sum()],
                [parameters["signal_yield"], parameters["background_yield"]],
                rtol=2e-8,
                atol=1e-7,
            )
            and np.allclose(
                expected, signal_bins + background_bins, rtol=1e-10, atol=1e-10
            )
        )
        record(
            "component_normalization",
            index,
            normalized_components or (not row["valid"] and not finite_components),
        )
        if not physical or not finite_components:
            continue
        integrated += 1
        integral_arguments = {
            name: parameters[name]
            for name in ("signal_yield", "background_yield", "mean", *shape_names)
        }
        integral_arguments["background_coefficients"] = [
            parameters["c0"],
            parameters["c1"],
        ]
        if row["threshold"] not in checked_thresholds:
            checked_thresholds.add(row["threshold"])
            bins = {
                0,
                len(edges) - 2,
                int(
                    np.clip(
                        np.searchsorted(edges, parameters["mean"]) - 1,
                        0,
                        len(edges) - 2,
                    )
                ),
            }
            for bin_index in bins:
                reference = reference_mass_integrals(
                    mass_range, edges[bin_index : bin_index + 2], **integral_arguments
                )
                record(
                    "independent_component_bins",
                    index,
                    np.allclose(
                        reference,
                        [signal_bins[bin_index], background_bins[bin_index]],
                        rtol=2e-7,
                        atol=1e-7,
                    ),
                )
                representative_bins_checked += 1
        integrals = {}
        for multiplier in (2, 3, 4):
            window = np.clip(
                [
                    parameters["mean"] - multiplier * parameters["sigma"],
                    parameters["mean"] + multiplier * parameters["sigma"],
                ],
                *mass_range,
            )
            reference = reference_mass_integrals(
                mass_range, window, **integral_arguments
            )
            integrals[multiplier] = reference
            record(
                "independent_sigma_integrals",
                index,
                np.allclose(
                    row[f"window_{multiplier}sigma"], window, rtol=0, atol=1e-12
                )
                and np.allclose(
                    [
                        row[f"raw_signal_{multiplier}sigma"],
                        row[f"background_{multiplier}sigma"],
                    ],
                    reference,
                    rtol=2e-7,
                    atol=1e-7,
                ),
            )
        efficiency = row["mc_signal_efficiency"]
        record(
            "nominal_aliases_are_3sigma",
            index,
            np.allclose(
                [row["signal"], row["background"]], integrals[3], rtol=2e-7, atol=1e-7
            )
            and np.isclose(
                row["signal_over_efficiency"],
                integrals[3][0] / efficiency,
                rtol=2e-7,
                atol=1e-7,
            )
            and np.isclose(
                row["raw_signal_3sigma_over_efficiency"],
                integrals[3][0] / efficiency,
                rtol=2e-7,
                atol=1e-7,
            ),
        )
    check("wp_fit_coverage_nonempty", fitted > 0 and valid_fitted > 0)
    for name, failures in checks.items():
        check(
            "wp_" + name,
            not failures,
            f"Failed scan indices: {sorted(set(failures))[:30]}",
        )
    reference = reference_pol0(rows, config["working_point"]["yield_eff_range"])
    pol0 = summary["yield_pol0"]
    check(
        "wp_pol0_independently_uses_s3_over_efficiency",
        all(
            np.isclose(pol0[name], reference[name], rtol=5e-5, atol=1e-7)
            for name in ("c0", "c0_error", "chi2")
        )
        and pol0["points"] == reference["points"],
    )
    valid_rows = [row for row in rows if row["valid"]]
    objective_failures = []
    for index, row in enumerate(valid_rows):
        efficiency = row["mc_signal_efficiency"]
        for multiplier in (2, 3, 4):
            expected = (
                efficiency
                * pol0["c0"]
                / np.sqrt(pol0["c0"] + row[f"background_{multiplier}sigma"])
            )
            if not np.isclose(
                row[f"significance_{multiplier}sigma"], expected, rtol=1e-12, atol=1e-12
            ):
                objective_failures.append(index)
        if not np.isclose(
            row["significance"], row["significance_3sigma"], rtol=1e-12, atol=1e-12
        ):
            objective_failures.append(index)
    check("wp_all_windows_share_s3_pol0_and_correct_objective", not objective_failures)
    independent_best = max(
        valid_rows,
        key=lambda row: row["mc_signal_efficiency"]
        * pol0["c0"]
        / np.sqrt(pol0["c0"] + row["background_3sigma"]),
    )
    best = summary["best_scan_point"]
    check(
        "wp_best_is_displayed_3sigma_curve_argmax",
        best["threshold"] == independent_best["threshold"]
        and best["mc_signal_efficiency"] == independent_best["mc_signal_efficiency"]
        and np.isclose(
            best["significance"], independent_best["significance_3sigma"], rtol=1e-12
        ),
    )
    diagnostic = summary["spectrum_qa"]["diagnostic_maximum"]
    check(
        "wp_displayed_maximum_matches_selected_optimum",
        diagnostic["threshold"] == best["threshold"]
        and diagnostic["mc_signal_efficiency"] == best["mc_signal_efficiency"],
    )
    override = config["analysis"]["bdt_threshold"]
    check(
        "wp_nominal_threshold_is_optimum_or_explicit_override",
        summary["threshold"] == (best["threshold"] if override is None else override),
    )
    return {
        "mass_range": mass_range.tolist(),
        "mass_bins": len(edges) - 1,
        "signal_model": "dscb",
        "background_model": mass_fit["background"],
        "scan_points": len(rows),
        "parameterized_fits": fitted,
        "valid_fits": valid_fitted,
        "representative_component_bins_checked": representative_bins_checked,
        "sigma_integral_windows_checked": 3 * integrated,
        "independent_pol0": reference,
        "independent_best_efficiency": independent_best["mc_signal_efficiency"],
        "independent_best_threshold": independent_best["threshold"],
        "independent_best_significance_3sigma": independent_best["significance_3sigma"],
        "nominal_threshold": summary["threshold"],
    }


def validate_root_archive(output, nominal, check):
    """Use native ROOT to reopen persisted numerical objects and redraw a canvas."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    path = output / "report/analysis.root"
    archive = ROOT.TFile.Open(str(path), "READ")
    check("root_file_readable", bool(archive) and not archive.IsZombie())
    if not archive or archive.IsZombie():
        return {"status": "unreadable"}

    def get(name, class_name):
        item = archive.Get(name)
        exists = bool(item) and item.InheritsFrom(class_name)
        check("root_object_" + name, exists, class_name)
        return item if exists else None

    def values(histogram):
        return np.array(
            [histogram.GetBinContent(i + 1) for i in range(histogram.GetNbinsX())]
        )

    def matrix(histogram):
        return np.array(
            [
                [
                    histogram.GetBinContent(i + 1, j + 1)
                    for j in range(histogram.GetNbinsY())
                ]
                for i in range(histogram.GetNbinsX())
            ]
        )

    try:
        for name in ("schema", "config", "training", "systematics", "provenance"):
            obj = get("metadata/" + name, "TObjString")
            if obj:
                metadata = json.loads(str(obj.GetString()))
                check("root_metadata_" + name, isinstance(metadata, dict))
        with np.load(output / "systematics/covariances.npz") as final:
            for name, item in nominal.items():
                for observable in ("raw", "corrected"):
                    source = (
                        item["correlations"]["signal"]
                        if observable == "raw"
                        else item["corrected"]
                    )
                    prefix = "nominal/" + name + "/"
                    hist = get(prefix + "cf_" + observable + "_stat", "TH1")
                    valid = get(prefix + "valid_" + observable, "TH1")
                    graph = get(prefix + "g_cf_" + observable + "_stat", "TGraphErrors")
                    covariance = get(prefix + "cov_" + observable + "_stat", "TH2")
                    if hist and valid and graph and covariance:
                        expected = np.asarray(source["values"], float)
                        errors = np.array(
                            [hist.GetBinError(i + 1) for i in range(hist.GetNbinsX())]
                        )
                        check(
                            prefix + observable + "_values",
                            np.allclose(values(hist), expected, equal_nan=True),
                        )
                        check(
                            prefix + observable + "_stat_errors",
                            np.allclose(
                                errors**2,
                                np.diag(np.asarray(source["covariance"], float)),
                                atol=1e-12,
                                equal_nan=True,
                            ),
                        )
                        check(
                            prefix + observable + "_valid",
                            np.array_equal(values(valid), source["valid"]),
                        )
                        check(
                            prefix + observable + "_graph",
                            graph.GetN() == sum(source["valid"]),
                        )
                        check(
                            prefix + observable + "_covariance",
                            np.allclose(
                                matrix(covariance),
                                np.asarray(source["covariance"], float),
                                equal_nan=True,
                            ),
                        )
                    for label, exported in (("syst", "systematic"), ("total", "total")):
                        cov = get(f"final/{name}/cov_{observable}_{label}", "TH2")
                        if cov:
                            check(
                                f"root_final_{name}_{observable}_{label}",
                                np.allclose(
                                    matrix(cov),
                                    final[f"{name}__{observable}__{exported}"],
                                    equal_nan=True,
                                ),
                            )
        canvas_graphs = {}

        def descendants(pad):
            for primitive in pad.GetListOfPrimitives():
                yield primitive
                if primitive.InheritsFrom("TPad"):
                    yield from descendants(primitive)

        centralities = sorted({tuple(item["centrality"]) for item in nominal.values()})
        for cent_index, bounds in enumerate(centralities):
            canvas_name = f"correlations_cent{cent_index}"
            canvas = get("canvases/" + canvas_name, "TCanvas")
            if not canvas:
                continue
            graphs = {
                obj.GetName(): obj
                for obj in descendants(canvas)
                if obj.InheritsFrom("TGraphErrors")
                and obj.GetName().endswith("__g_cf_raw_stat")
            }
            expected = {
                name: item
                for name, item in nominal.items()
                if tuple(item["centrality"]) == bounds
                and any(item["correlations"]["signal"]["valid"])
            }
            check(
                "root_canvas_" + canvas_name + "_raw_graph_count",
                len(graphs) == len(expected),
            )
            canvas_graphs[canvas_name] = {}
            for name, item in expected.items():
                graph = graphs.get(f"nominal__{name}__g_cf_raw_stat")
                check("root_canvas_graph_" + name, graph is not None)
                if graph is not None:
                    source = item["correlations"]["signal"]
                    valid = np.asarray(source["valid"], bool)
                    values = np.array([graph.GetPointY(i) for i in range(graph.GetN())])
                    errors = np.array([graph.GetErrorY(i) for i in range(graph.GetN())])
                    check(
                        "root_canvas_graph_" + name + "_points",
                        graph.GetN() == int(valid.sum())
                        and np.allclose(
                            values, np.asarray(source["values"], float)[valid]
                        ),
                    )
                    check(
                        "root_canvas_graph_" + name + "_errors",
                        np.allclose(
                            errors**2,
                            np.diag(np.asarray(source["covariance"], float))[valid],
                            atol=1e-12,
                        ),
                    )
                    canvas_graphs[canvas_name][name] = graph.GetN()
        canvases = get("canvases", "TDirectory")
        keys = list(canvases.GetListOfKeys()) if canvases else []
        check("root_persisted_canvases", len(keys) > 0)
        check(
            "root_working_point_graph",
            bool(get("working_point/significance_vs_score", "TGraph")),
        )
        check(
            "root_working_point_mass_fit",
            bool(get("working_point/best_mass_data", "TH1D"))
            and bool(get("working_point/best_mass_fit", "TH1D")),
        )
        check(
            "root_working_point_canvas",
            bool(get("canvases/working_point", "TCanvas")),
        )
        if keys:
            canvas = keys[0].ReadObj()
            check(
                "root_canvas_primitives",
                canvas.InheritsFrom("TCanvas")
                and canvas.GetListOfPrimitives().GetSize() > 0,
            )
            with tempfile.TemporaryDirectory() as temporary:
                pdf = Path(temporary) / "redrawn.pdf"
                canvas.Draw()
                canvas.SaveAs(str(pdf))
                check(
                    "root_canvas_roundtrip_pdf",
                    pdf.is_file()
                    and pdf.stat().st_size > 1000
                    and pdf.read_bytes().startswith(b"%PDF-"),
                )
        object_classes = {}

        def count_objects(directory):
            for key in directory.GetListOfKeys():
                class_name = key.GetClassName()
                if class_name == "TDirectoryFile":
                    count_objects(directory.GetDirectory(key.GetName()))
                else:
                    object_classes[class_name] = object_classes.get(class_name, 0) + 1

        count_objects(archive)
        return {
            "path": str(path),
            "root_version": ROOT.gROOT.GetVersion(),
            "canvases": len(keys),
            "correlation_canvas_graphs": canvas_graphs,
            "object_count": sum(object_classes.values()),
            "objects_by_class": object_classes,
            "reader": "native PyROOT TFile",
        }
    finally:
        archive.Close()


def validate_stage_qa(output, training, working_point, scan_fits, report, check):
    """Verify stage-owned QA without reconstructing plots in the report."""
    from pih3l.root_io import load_root
    from pih3l.training_qa import TRAINING_QA_ARTIFACTS

    ROOT = load_root()
    mass_canvas_inspection = None
    source_canvas_payloads = {}
    canvas_sources = {
        "working_point": "significance_vs_score",
        "working_point_efficiency": "significance_vs_efficiency",
        "working_point_yield": "raw_yield_over_efficiency",
        "working_point_mass_fit": "best_mass_fit",
    }
    for name in TRAINING_QA_ARTIFACTS:
        artifact = output / "training" / name
        check("training_qa_" + Path(name).stem, artifact.is_file())
        if artifact.suffix == ".pdf" and artifact.is_file():
            check(
                "training_qa_pdf_" + artifact.stem,
                artifact.read_bytes().startswith(b"%PDF-"),
            )
    qa_files = {
        "training": (
            "training.root",
            (
                "correlations/signal",
                "correlations/background",
                "roc/train",
                "roc/test",
                "importance/gain",
                "metadata/qa_index",
            ),
        ),
        "working_point": (
            "working_point.root",
            (
                "scan/yield_pol0",
                "scan/valid",
                "mass_scan/point_0000/data",
                "canvases/significance_vs_score",
                "canvases/raw_yield_over_efficiency",
                "canvases/best_mass_fit",
                "canvases/mean_sigma_vs_efficiency",
                "scan/significance_vs_score/significance_3sigma",
                "scan/significance_vs_score/window_envelope",
                "scan/spectrum_yield_pol0",
                "metadata/qa_index",
            ),
        ),
    }
    for stage, (name, objects) in qa_files.items():
        path = output / stage / "qa" / name
        archive = ROOT.TFile.Open(str(path), "READ")
        readable = bool(archive) and not archive.IsZombie()
        check(stage + "_qa_root_readable", readable)
        if readable:
            try:
                for name in objects:
                    check(stage + "_qa_root_" + name, bool(archive.Get(name)))
                if stage == "training":
                    features = training["features"]
                    feature_directory = archive.Get("features")
                    feature_names = (
                        {key.GetName() for key in feature_directory.GetListOfKeys()}
                        if feature_directory
                        else set()
                    )
                    correlation_names = archive.Get("correlations/features")
                    check(
                        "training_root_distributions_include_mass_qa_correlations_only_model_features",
                        feature_names == set(features + ["mass_h3l"])
                        and bool(correlation_names)
                        and json.loads(str(correlation_names.GetString())) == features,
                    )
                    check(
                        "training_root_correlation_dimensions",
                        all(
                            archive.Get("correlations/" + category).GetNbinsX()
                            == len(features)
                            and archive.Get("correlations/" + category).GetNbinsY()
                            == len(features)
                            for category in ("signal", "background")
                        ),
                    )
                    for name, source in training["qa"]["score_distributions"].items():
                        histogram = archive.Get("scores/" + name)
                        check(
                            "training_root_margin_histogram_" + name,
                            bool(histogram)
                            and histogram.GetNbinsX() == len(source["counts"])
                            and np.array_equal(
                                [
                                    histogram.GetBinContent(i + 1)
                                    for i in range(histogram.GetNbinsX())
                                ],
                                source["counts"],
                            )
                            and np.allclose(
                                [
                                    histogram.GetXaxis().GetBinLowEdge(i + 1)
                                    for i in range(histogram.GetNbinsX() + 1)
                                ],
                                source["edges"],
                                rtol=0,
                                atol=1e-12,
                            )
                            and histogram.GetBinContent(0) == 0
                            and histogram.GetBinContent(histogram.GetNbinsX() + 1) == 0,
                        )
                if stage == "working_point":
                    canvas = archive.Get("canvases/significance_vs_score")
                    source_canvas_payloads = {
                        name: canvas_payload(archive.Get("canvases/" + source))
                        for name, source in canvas_sources.items()
                    }
                    mass_canvas_inspection = mass_canvas_contract(
                        archive.Get("canvases/best_mass_fit")
                    )
                    check(
                        "wp_mass_canvas_single_panel",
                        mass_canvas_inspection["data_panels"] == 1,
                    )
                    check(
                        "wp_mass_canvas_no_residual_or_pull",
                        not mass_canvas_inspection["has_residual_or_pull"],
                    )
                    check(
                        "wp_mass_fit_functions_are_continuous_curves",
                        mass_canvas_inspection["function_curve_count"] == 3
                        and not mass_canvas_inspection["has_drawn_model_histograms"],
                    )
                    check(
                        "wp_mass_canvas_legend_significance_chi2",
                        all(
                            mass_canvas_inspection[name]
                            for name in ("has_legend", "has_significance", "has_chi2")
                        ),
                    )

                    def mass_keys(directory):
                        for key in directory.GetListOfKeys():
                            yield str(key.GetName()).lower()
                            if key.GetClassName() in {"TDirectoryFile", "TDirectory"}:
                                yield from mass_keys(directory.Get(key.GetName()))

                    numerical_names = [
                        name
                        for folder in ("mass_scan", "best_mass_fit")
                        for name in mass_keys(archive.Get(folder))
                    ]
                    check(
                        "wp_mass_archive_has_no_pull_or_residual_objects",
                        not any(
                            "pull" in name or "residual" in name or "data-fit" in name
                            for name in numerical_names
                        ),
                    )
                    mass_object_failures = []
                    mass_fit_metadata_failures = []
                    for index, point in enumerate(scan_fits["points"]):
                        prefix = f"mass_scan/point_{index:04d}/"
                        for name, values in (
                            ("data", point["counts"]),
                            ("expected", point["expected"]),
                            ("signal", point["fit"]["signal_expected"]),
                            ("background", point["fit"]["background_expected"]),
                        ):
                            obj = archive.Get(prefix + name)
                            valid = bool(obj) and obj.GetNbinsX() == len(values)
                            if valid:
                                valid = np.allclose(
                                    [
                                        obj.GetBinContent(i + 1)
                                        for i in range(obj.GetNbinsX())
                                    ],
                                    np.asarray(values, float),
                                    rtol=1e-10,
                                    atol=1e-10,
                                    equal_nan=True,
                                ) and np.allclose(
                                    [
                                        obj.GetXaxis().GetBinLowEdge(i + 1)
                                        for i in range(obj.GetNbinsX() + 1)
                                    ],
                                    scan_fits["mass_edges"],
                                    rtol=0,
                                    atol=1e-12,
                                )
                                if name == "data":
                                    valid = valid and np.allclose(
                                        [
                                            obj.GetBinError(i + 1) ** 2
                                            for i in range(obj.GetNbinsX())
                                        ],
                                        values,
                                        rtol=1e-12,
                                        atol=1e-12,
                                    )
                            if not valid:
                                mass_object_failures.append(prefix + name)
                        fit_object = archive.Get(prefix + "fit")
                        if (
                            not fit_object
                            or json.loads(str(fit_object.GetString())) != point["fit"]
                        ):
                            mass_fit_metadata_failures.append(index)
                    check(
                        "wp_all_root_mass_histograms_match_numerical_fits",
                        not mass_object_failures,
                        str(mass_object_failures[:20]),
                    )
                    check(
                        "wp_all_root_fit_parameters_covariance_match_json",
                        not mass_fit_metadata_failures,
                        str(mass_fit_metadata_failures[:20]),
                    )
                    graph = archive.Get(
                        "scan/significance_vs_score/significance_3sigma"
                    )
                    reference_rows = sorted(
                        (
                            row
                            for row in working_point["scan_points"]
                            if row["spectrum_qa_valid"]
                        ),
                        key=lambda row: row["threshold"],
                    )
                    check(
                        "wp_persisted_curve_matches_all_valid_3sigma_scan_points",
                        graph.GetN() == len(reference_rows)
                        and np.allclose(
                            [
                                [graph.GetPointX(i), graph.GetPointY(i)]
                                for i in range(graph.GetN())
                            ],
                            [
                                [row["threshold"], row["significance_3sigma"]]
                                for row in reference_rows
                            ],
                            rtol=1e-12,
                            atol=1e-12,
                        ),
                    )
                    calibration = archive.Get("mc_signal_fit/calibration")
                    mc_data = archive.Get("mc_signal_fit/data")
                    expected_mc = working_point["mc_signal_fit"]
                    check(
                        "wp_mc_tail_calibration_json_and_root_match",
                        bool(calibration)
                        and json.loads(str(calibration.GetString())) == expected_mc
                        and strict_json(output / "working_point/mc_signal_fit.json")
                        == expected_mc,
                    )
                    check(
                        "wp_mc_tail_calibration_root_counts_sumw2",
                        bool(mc_data)
                        and mc_data.GetNbinsX() == len(expected_mc["counts"])
                        and np.allclose(
                            [
                                [
                                    mc_data.GetBinContent(i + 1),
                                    mc_data.GetBinError(i + 1) ** 2,
                                ]
                                for i in range(mc_data.GetNbinsX())
                            ],
                            [[value, value] for value in expected_mc["counts"]],
                            rtol=1e-12,
                            atol=1e-12,
                        ),
                    )
                    with tempfile.TemporaryDirectory() as temporary:
                        pdf = Path(temporary) / "reopened.pdf"
                        canvas.Draw()
                        canvas.SaveAs(str(pdf))
                        check(
                            stage + "_qa_canvas_roundtrip",
                            pdf.is_file() and pdf.stat().st_size > 1000,
                        )
            finally:
                archive.Close()
    expected_wp_pdfs = (
        "sig_vs_score",
        "sig_vs_eff",
        "s3overeff_vs_eff",
        "raw_signal_yield",
        "mean_sigma_vs_efficiency",
        "fit_stability",
        "best_fit",
    )
    check(
        "working_point_qa_index",
        (output / "working_point/qa/index.json").is_file(),
    )
    check(
        "working_point_qa_pdf_coverage",
        all(
            (output / "working_point/qa" / (name + ".pdf")).is_file()
            and (output / "working_point/qa" / (name + ".pdf"))
            .read_bytes()
            .startswith(b"%PDF-")
            for name in expected_wp_pdfs
        ),
    )
    linked_paths = {
        (output / "report" / item["path"]).resolve() for item in report["stage_qa"]
    }
    actual_paths = {
        path.resolve()
        for stage in ("training", "working_point")
        for path in (output / stage / "qa").glob("*")
        if path.suffix in {".pdf", ".root"}
    }
    check(
        "report_links_identical_stage_owned_qa_artifacts", linked_paths == actual_paths
    )
    combined = ROOT.TFile.Open(str(output / "report/analysis.root"), "READ")
    try:
        for category in ("signal", "background"):
            histogram = combined.Get("training/score_" + category)
            source = training["qa"]["score_distributions"]["test_" + category]
            check(
                "report_training_raw_margin_histogram_" + category,
                bool(histogram)
                and histogram.GetNbinsX() == len(source["counts"])
                and np.array_equal(
                    [
                        histogram.GetBinContent(i + 1)
                        for i in range(histogram.GetNbinsX())
                    ],
                    source["counts"],
                )
                and np.allclose(
                    [
                        histogram.GetXaxis().GetBinLowEdge(i + 1)
                        for i in range(histogram.GetNbinsX() + 1)
                    ],
                    source["edges"],
                    rtol=0,
                    atol=1e-12,
                ),
            )
        for name in canvas_sources:
            reused = combined.Get("canvases/" + name)
            check(
                "report_" + name + "_is_exact_stage_canvas_clone",
                bool(reused) and canvas_payload(reused) == source_canvas_payloads[name],
            )
            pdf = output / "report" / (name + ".pdf")
            check(
                "report_" + name + "_synchronized_pdf",
                pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-"),
            )
    finally:
        combined.Close()
    check(
        "report_wp_metadata_is_current_stage_summary",
        report["working_point"] == working_point,
    )
    check(
        "report_qa_links_resolve",
        bool(report["stage_qa"])
        and all(
            (output / "report" / item["path"]).is_file() for item in report["stage_qa"]
        ),
    )
    check(
        "report_reuses_stage_qa",
        not list((output / "report").glob("bdt_validation*.pdf"))
        and not list((output / "report").glob("optuna_training*.pdf")),
    )
    check(
        "stage_qa_no_png",
        not list((output / "training").rglob("*.png"))
        and not list((output / "working_point").rglob("*.png")),
    )
    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".eps", ".gif"}
    check(
        "all_qa_figures_are_pdf_only",
        not [
            path
            for stage in (
                "training",
                "working_point",
                "raw_qa",
                "systematics",
                "report",
            )
            for path in (output / stage).rglob("*")
            if path.suffix.lower() in image_extensions
        ],
    )
    check(
        "training_qa_score_counts",
        all(
            sum(training["qa"]["score_distributions"][f"{split}_{name}"]["counts"])
            == training["split"][str(label)][
                "train_rows" if split == "train" else "validation_rows"
            ]
            for split in ("train", "test")
            for label, name in ((0, "background"), (1, "signal"))
        ),
    )
    return {
        "mass_canvas": mass_canvas_inspection,
        "identical_stage_qa_links": len(linked_paths),
        "stage_canvas_payload_objects": {
            name: len(payload) for name, payload in source_canvas_payloads.items()
        },
        "working_point_pdf_count": len(
            list((output / "working_point/qa").glob("*.pdf"))
        ),
    }


def validate_systematics_qa(config, output, nominal, systematic, check):
    """Recompute every ratio/relative error independently and reopen both ROOTs."""
    import ROOT

    directory = output / "systematics/qa"
    qa = strict_json(directory / "results.json")
    index = strict_json(directory / "index.json")
    report = strict_json(output / "report/summary.json")
    configured = {item["name"]: item for item in config["systematics"]["variations"]}
    outcomes = {item["name"]: item for item in systematic["variations"]}
    groups = {item.get("group", name) for name, item in configured.items()}
    edges = np.asarray(config["analysis"]["kstar_edges"], float)
    centers = (edges[:-1] + edges[1:]) / 2
    failures = {}

    def audit(category, condition, detail):
        failures.setdefault(category, [])
        if not condition:
            failures[category].append(detail)

    def cf(item, observable):
        return (
            item["correlations"]["signal"] if observable == "raw" else item["corrected"]
        )

    def source_arrays(source):
        values = np.asarray(source["values"], float)
        diagonal = np.asarray(source["covariance"], float).diagonal()
        valid = np.asarray(source["valid"], bool) & np.isfinite(values)
        valid &= np.isfinite(diagonal) & (diagonal >= 0)
        return values, valid

    def oracle(source, numerator, numerator_valid, *, absolute=False):
        denominator, valid = source_arrays(source)
        if absolute:
            denominator = np.abs(denominator)
        numerator = np.asarray(numerator, float)
        valid &= (
            np.asarray(numerator_valid, bool)
            & np.isfinite(numerator)
            & (denominator != 0)
        )
        if absolute:
            valid &= numerator >= 0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = numerator / denominator
        valid &= np.isfinite(values)
        return np.where(valid, values, np.nan), valid

    leaves = {}
    percent_runs = {}

    def compare_leaf(path, diagnostic, values, valid):
        audit(
            "independent_values",
            np.allclose(
                np.asarray(diagnostic["values"], float),
                values,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ),
            path,
        )
        audit(
            "independent_valid_masks", np.array_equal(diagnostic["valid"], valid), path
        )
        status = (
            "ok"
            if valid.all()
            else ("partial_coverage" if valid.any() else "unavailable")
        )
        audit("diagnostic_status", diagnostic["status"] == status, path)
        leaves[path] = (values, valid, diagnostic)
        if "/variations/" not in path:
            transitions = np.diff(np.r_[False, valid, False].astype(int))
            percent_runs[path] = [
                {"name": f"run_{i}", "start_bin": int(start), "stop_bin": int(stop)}
                for i, (start, stop) in enumerate(
                    zip(
                        np.flatnonzero(transitions == 1),
                        np.flatnonzero(transitions == -1),
                    )
                )
            ]

    check(
        "systematic_qa_variation_and_source_coverage",
        set(qa["variations"]) == set(configured) == set(outcomes)
        and set(qa["sources"]) == groups
        and set(qa["bins"]) == set(nominal),
    )
    check(
        "systematic_qa_nominal_and_uncertainty_definitions",
        "nominal C(k*)" in qa["std_definition"]
        and qa["variation_ratio"] == "C_variation / C_nominal"
        and qa["relative_uncertainty"] == "sigma_systematic / abs(C_nominal)"
        and "not propagated" in qa["ratio_uncertainty"]
        and "cross-covariance" in qa["ratio_uncertainty"],
    )
    check("systematic_qa_kstar_edges", np.array_equal(qa["kstar_edges"], edges))
    if "cospa_tight" not in configured:
        check(
            "systematic_qa_no_removed_cospa_tight",
            "cospa_tight" not in qa["variations"]
            and not list(directory.glob("*cospa_tight*")),
        )
    for group in groups:
        expected = {
            name
            for name, item in configured.items()
            if item.get("group", name) == group
        }
        completed = {
            name for name in expected if outcomes[name]["status"] == "completed"
        }
        actual = qa["sources"][group]
        audit(
            "source_execution_coverage",
            set(actual["expected"]) == expected
            and set(actual["completed"]) == completed
            and set(actual["failed"]) == expected - completed
            and actual["status"]
            == ("complete" if completed == expected else "incomplete_variations"),
            group,
        )
    varied_results = {}
    for name, settings in configured.items():
        entry = qa["variations"][name]
        result = (
            strict_json(output / "systematics" / name / "results.json")
            if outcomes[name]["status"] == "completed"
            else None
        )
        varied_results[name] = result
        audit(
            "resolved_selection_and_status",
            entry["selection"] == settings
            and entry["group"] == settings.get("group", name)
            and entry["status"] == outcomes[name]["status"]
            and set(entry["bins"]) == set(nominal),
            name,
        )
        for key, baseline in nominal.items():
            for observable in ("raw", "corrected"):
                values, valid = (
                    source_arrays(cf(result[key], observable))
                    if result is not None
                    else (np.full(len(centers), np.nan), np.zeros(len(centers), bool))
                )
                values, valid = oracle(cf(baseline, observable), values, valid)
                compare_leaf(
                    f"systematics_qa/variations/{name}/{key}/{observable}",
                    entry["bins"][key][observable],
                    values,
                    valid,
                )
    for key, baseline in nominal.items():
        for observable in ("raw", "corrected"):
            source = cf(baseline, observable)
            summary = systematic["bins"][key][observable]
            diagnostics = qa["bins"][key][observable]
            audit(
                "source_bin_coverage",
                set(diagnostics["groups"]) == groups == set(summary["groups"]),
                key + "/" + observable,
            )
            complete_masks = []
            for group, data in summary["groups"].items():
                diagnostic = diagnostics["groups"][group]
                values, valid = oracle(
                    source, data["error"], np.ones(len(centers), bool), absolute=True
                )
                compare_leaf(
                    f"systematics_qa/sources/{group}/{key}/{observable}",
                    diagnostic,
                    values,
                    valid,
                )
                coverage = np.asarray(data["coverage"], int)
                expected = len(qa["sources"][group]["expected"])
                complete = (coverage == expected) & valid
                if (
                    config["systematics"]["method"] == "rms"
                    and data.get("ensemble_size") != expected
                ):
                    complete[:] = False
                complete_masks.append(complete)
                status = qa["sources"][group]["status"]
                if status == "complete" and np.any(~complete):
                    status = "incomplete_bin_coverage"
                audit(
                    "source_complete_coverage",
                    np.array_equal(diagnostic["coverage"], coverage)
                    and diagnostic.get("ensemble_size") == data.get("ensemble_size")
                    and np.array_equal(diagnostic["complete_coverage"], complete)
                    and diagnostic["coverage_status"] == status,
                    group + "/" + key + "/" + observable,
                )
            values, valid = oracle(
                source, summary["error"], summary["valid"], absolute=True
            )
            compare_leaf(
                f"systematics_qa/total/{key}/{observable}",
                diagnostics["total"],
                values,
                valid,
            )
            complete = (
                np.all(complete_masks, axis=0)
                if complete_masks
                else np.zeros(len(centers), bool)
            )
            missing = any(item["failed"] for item in qa["sources"].values()) or np.any(
                valid & ~complete
            )
            status = (
                "unavailable"
                if not valid.any()
                else ("incomplete_variation_coverage" if missing else "complete")
            )
            audit(
                "total_complete_coverage",
                np.array_equal(diagnostics["total"]["complete_coverage"], complete)
                and diagnostics["total"]["coverage_status"] == status,
                key + "/" + observable,
            )
    expected_plots = {
        (kind, name, tuple(cent))
        for kind, names in (
            ("variation", configured),
            ("source", groups),
            ("total", ["total"]),
            ("total_corrected", ["total"]),
        )
        for name in names
        for cent in config["analysis"]["centrality_bins"]
    }
    actual_plots = {
        (plot["kind"], plot["name"], tuple(plot["centrality"]))
        for plot in index["plots"]
    }
    expected_pdf_count = len(expected_plots)
    check(
        "systematic_qa_configured_plot_coverage",
        actual_plots == expected_plots and len(index["plots"]) == expected_pdf_count,
    )
    check(
        "systematic_qa_unique_pdf_and_canvas_names",
        len({p["path"] for p in index["plots"]}) == expected_pdf_count
        and len({p["canvas"] for p in index["plots"]}) == expected_pdf_count,
    )
    expected_pdf_paths = {
        (directory / plot["path"]).resolve() for plot in index["plots"]
    }
    all_pdf_paths = set(expected_pdf_paths)
    overview_pdf = index.get("overview_pdf")
    if overview_pdf is not None:
        overview_path = (directory / overview_pdf).resolve()
        check(
            "systematic_qa_overview_is_separate_from_single_plots",
            overview_path not in expected_pdf_paths,
        )
        all_pdf_paths.add(overview_path)
    check(
        "systematic_qa_pdf_files_exact_coverage",
        {path.resolve() for path in directory.glob("*.pdf")} == all_pdf_paths
        and all(
            path.is_file() and path.read_bytes().startswith(b"%PDF-")
            for path in all_pdf_paths
        ),
    )
    linked = report["systematic_qa"]
    linked_paths = [
        (output / "report" / plot["path"]).resolve() for plot in linked["plots"]
    ]
    check(
        "report_links_all_systematic_qa_pdfs",
        set(linked_paths) == expected_pdf_paths
        and len(linked_paths) == expected_pdf_count
        and all(path.is_file() for path in linked_paths),
    )
    check(
        "report_links_current_systematic_qa_metadata_and_root",
        linked["definitions"] == index["definitions"]
        and (output / "report" / linked["root_file"]).resolve()
        == (directory / "systematics.root").resolve(),
    )
    html = (output / "report/index.html").read_text()
    if overview_pdf is not None:
        linked_overview = linked.get("overview_pdf")
        check(
            "report_links_systematic_qa_overview",
            bool(linked_overview)
            and (output / "report" / linked_overview).resolve() == overview_path
            and 'href="' + linked_overview + '"' in html,
        )
    check(
        "report_html_has_every_systematic_qa_link",
        all('href="' + plot["path"] + '"' in html for plot in linked["plots"]),
    )
    ROOT.gROOT.SetBatch(True)
    for label, path in (
        ("stage", directory / "systematics.root"),
        ("report", output / "report/analysis.root"),
    ):
        archive = ROOT.TFile.Open(str(path), "READ")
        check(
            "systematic_qa_" + label + "_root_readable",
            bool(archive) and not archive.IsZombie(),
        )
        if not archive or archive.IsZombie():
            continue
        try:
            for prefix, (values, valid, diagnostic) in leaves.items():
                hist, mask, graph = (
                    archive.Get(prefix + "/" + name)
                    for name in ("values", "valid", "graph")
                )
                exists = (
                    bool(hist and mask and graph)
                    and hist.InheritsFrom("TH1")
                    and mask.InheritsFrom("TH1")
                    and graph.ClassName() == "TGraph"
                )
                audit(label + "_root_leaf_objects", exists, prefix)
                if not exists:
                    continue
                audit(
                    label + "_root_leaf_values",
                    hist.GetNbinsX() == len(centers)
                    and np.allclose(
                        [hist.GetBinContent(i + 1) for i in range(len(centers))],
                        values,
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    ),
                    prefix,
                )
                audit(
                    label + "_root_leaf_masks_edges_no_errors",
                    np.array_equal(
                        [mask.GetBinContent(i + 1) for i in range(len(centers))], valid
                    )
                    and np.allclose(
                        [
                            hist.GetXaxis().GetBinLowEdge(i + 1)
                            for i in range(len(edges))
                        ],
                        edges,
                        rtol=0,
                        atol=1e-12,
                    )
                    and all(hist.GetBinError(i + 1) == 0 for i in range(len(centers))),
                    prefix,
                )
                audit(
                    label + "_root_leaf_graph_points",
                    graph.GetN() == int(valid.sum())
                    and np.allclose(
                        [graph.GetPointX(i) for i in range(graph.GetN())],
                        centers[valid],
                        rtol=0,
                        atol=1e-12,
                    )
                    and np.allclose(
                        [graph.GetPointY(i) for i in range(graph.GetN())],
                        values[valid],
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                    prefix,
                )
                status = archive.Get(prefix + "/status")
                audit(
                    label + "_root_leaf_metadata",
                    bool(status)
                    and json.loads(str(status.GetString()))
                    == {
                        key: value
                        for key, value in diagnostic.items()
                        if key != "values"
                    },
                    prefix,
                )
                if prefix in percent_runs:
                    run_prefix = prefix + "/percent_steps"
                    metadata = archive.Get(run_prefix + "/metadata")
                    expected_runs = percent_runs[prefix]
                    metadata_ok = bool(metadata)
                    if metadata_ok:
                        actual_metadata = json.loads(str(metadata.GetString()))
                        metadata_ok = (
                            actual_metadata.get("unit") == "percent"
                            and actual_metadata.get("scale") == 100
                            and actual_metadata.get("runs") == expected_runs
                        )
                    audit(label + "_percent_step_metadata", metadata_ok, prefix)
                    run_directory = archive.GetDirectory(run_prefix)
                    audit(
                        label + "_percent_step_exact_run_coverage",
                        bool(run_directory)
                        and {key.GetName() for key in run_directory.GetListOfKeys()}
                        == {"metadata", *(run["name"] for run in expected_runs)},
                        prefix,
                    )
                    for run in expected_runs:
                        histogram = archive.Get(run_prefix + "/" + run["name"])
                        start, stop = run["start_bin"], run["stop_bin"]
                        exists = bool(histogram) and histogram.ClassName() == "TH1D"
                        audit(label + "_percent_step_histogram", exists, prefix)
                        if not exists:
                            continue
                        audit(
                            label + "_percent_step_values_edges_no_errors",
                            histogram.GetNbinsX() == stop - start
                            and np.allclose(
                                [
                                    histogram.GetBinContent(i + 1)
                                    for i in range(stop - start)
                                ],
                                100 * values[start:stop],
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            and np.allclose(
                                [
                                    histogram.GetXaxis().GetBinLowEdge(i + 1)
                                    for i in range(stop - start + 1)
                                ],
                                edges[start : stop + 1],
                                rtol=0,
                                atol=1e-12,
                            )
                            and all(
                                histogram.GetBinError(i + 1) == 0
                                for i in range(stop - start)
                            ),
                            prefix + "/" + run["name"],
                        )
            if label == "stage":
                canvases = archive.GetDirectory("canvases")
                check(
                    "systematic_qa_canvas_exact_coverage",
                    bool(canvases)
                    and {key.GetName() for key in canvases.GetListOfKeys()}
                    == {plot["canvas"] for plot in index["plots"]},
                )
                for plot in index["plots"]:
                    canvas = archive.Get("canvases/" + plot["canvas"])
                    exists = bool(canvas) and canvas.InheritsFrom("TCanvas")
                    audit(
                        "root_canvas_persistence",
                        exists and canvas.GetListOfPrimitives().GetSize() > 0,
                        plot["canvas"],
                    )
                    if not exists:
                        continue
                    primitives = list(root_primitive_links(canvas))
                    drawn = {obj.GetName(): obj for _, obj, _ in primitives}
                    step_objects = {
                        obj.GetName(): (parent, obj, option)
                        for parent, obj, option in primitives
                        if obj.InheritsFrom("TH1")
                        and "__percent_steps__" in obj.GetName()
                    }
                    expected_drawn_steps = set()
                    relative_plot = plot["kind"] != "variation"
                    if relative_plot:
                        audit(
                            "root_canvas_relative_percent_axis",
                            sum(
                                obj.InheritsFrom("TH1")
                                and "__percent_steps__" not in obj.GetName()
                                and obj.GetYaxis().GetTitle()
                                == "Systematic uncertainty (%)"
                                for _, obj, _ in primitives
                            )
                            == 4,
                            plot["canvas"],
                        )
                        audit(
                            "root_canvas_relative_has_no_point_graphs",
                            not any(
                                obj.InheritsFrom("TGraph") for _, obj, _ in primitives
                            ),
                            plot["canvas"],
                        )
                    for channel in range(4):
                        lo, hi = plot["centrality"]
                        key = f"cent_{lo:g}_{hi:g}__ch_{channel}"
                        if plot["kind"] == "variation":
                            diagnostics = [
                                ("variations/" + plot["name"], observable)
                                for observable in ("raw", "corrected")
                            ]
                        elif plot["kind"] == "source":
                            diagnostics = [
                                ("sources/" + plot["name"], observable)
                                for observable in ("raw", "corrected")
                            ]
                        else:
                            observable = (
                                "corrected"
                                if plot["kind"] == "total_corrected"
                                else "raw"
                            )
                            diagnostics = [
                                ("sources/" + group, observable) for group in groups
                            ] + [("total", observable)]
                            audit(
                                "root_canvas_total_observable_metadata",
                                plot.get("observable") == observable,
                                plot["canvas"],
                            )
                        for scope, observable in diagnostics:
                            prefix = f"systematics_qa/{scope}/{key}/{observable}"
                            values, valid, _ = leaves[prefix]
                            if not relative_plot and valid.any():
                                graph = drawn.get(
                                    (prefix + "/graph").replace("/", "__")
                                )
                                audit(
                                    "root_canvas_drawn_ratio_or_relative_points",
                                    graph is not None
                                    and graph.GetN() == int(valid.sum())
                                    and np.allclose(
                                        [
                                            graph.GetPointY(i)
                                            for i in range(graph.GetN())
                                        ],
                                        values[valid],
                                        rtol=1e-12,
                                        atol=1e-12,
                                    ),
                                    prefix,
                                )
                            elif relative_plot:
                                for run in percent_runs[prefix]:
                                    name = (
                                        prefix + "/percent_steps/" + run["name"]
                                    ).replace("/", "__")
                                    expected_drawn_steps.add(name)
                                    exists = name in step_objects
                                    audit(
                                        "root_canvas_percent_run_persisted",
                                        exists,
                                        name,
                                    )
                                    if not exists:
                                        continue
                                    _, histogram, option = step_objects[name]
                                    start, stop = run["start_bin"], run["stop_bin"]
                                    total = scope == "total"
                                    audit(
                                        "root_canvas_percent_step_values_style",
                                        "HIST" in option.upper()
                                        and "][" in option
                                        and histogram.GetLineWidth()
                                        == (3 if total else 2)
                                        and (
                                            histogram.GetLineColor() == ROOT.kBlack
                                            if total
                                            else histogram.GetLineColor() != ROOT.kBlack
                                        )
                                        and histogram.GetNbinsX() == stop - start
                                        and np.allclose(
                                            [
                                                histogram.GetBinContent(i + 1)
                                                for i in range(stop - start)
                                            ],
                                            100 * values[start:stop],
                                            rtol=1e-12,
                                            atol=1e-12,
                                        )
                                        and np.allclose(
                                            [
                                                histogram.GetXaxis().GetBinLowEdge(
                                                    i + 1
                                                )
                                                for i in range(stop - start + 1)
                                            ],
                                            edges[start : stop + 1],
                                            rtol=0,
                                            atol=1e-12,
                                        )
                                        and all(
                                            histogram.GetBinError(i + 1) == 0
                                            for i in range(stop - start)
                                        ),
                                        name,
                                    )
                    if relative_plot:
                        audit(
                            "root_canvas_percent_step_exact_coverage",
                            set(step_objects) == expected_drawn_steps,
                            plot["canvas"],
                        )
                        if plot["kind"] in {"total", "total_corrected"}:
                            legends = [
                                obj
                                for _, obj, _ in primitives
                                if obj.InheritsFrom("TLegend")
                            ]
                            legend_valid = len(legends) == 4
                            for legend in legends:
                                entries = list(legend.GetListOfPrimitives())
                                labels = [str(entry.GetLabel()) for entry in entries]
                                samples = [entry.GetObject() for entry in entries]
                                legend_valid &= (
                                    len(entries) == len(groups) + 1
                                    and len(set(labels)) == len(labels)
                                    and all(labels)
                                    and all(bool(sample) for sample in samples)
                                )
                                if not all(bool(sample) for sample in samples):
                                    continue
                                total_entries = [
                                    (entry, sample)
                                    for entry, sample in zip(entries, samples)
                                    if sample.GetLineColor() == ROOT.kBlack
                                    and sample.GetLineWidth() == 3
                                ]
                                legend_valid &= (
                                    len(total_entries) == 1
                                    and "total"
                                    in str(total_entries[0][0].GetLabel()).casefold()
                                    and all(
                                        (
                                            sample.GetLineColor() == ROOT.kBlack
                                            and sample.GetLineWidth() == 3
                                        )
                                        or (
                                            sample.GetLineColor() != ROOT.kBlack
                                            and sample.GetLineWidth() == 2
                                        )
                                        for sample in samples
                                    )
                                    and all(
                                        (
                                            sample.InheritsFrom("TH1")
                                            and sample.GetName() in expected_drawn_steps
                                        )
                                        or (
                                            sample.InheritsFrom("TLine")
                                            and "n/a"
                                            in str(entry.GetLabel()).casefold()
                                        )
                                        for entry, sample in zip(entries, samples)
                                    )
                                )
                            audit(
                                "root_canvas_total_all_sources_legend",
                                legend_valid,
                                plot["canvas"],
                            )
                if index["plots"]:
                    canvas = archive.Get("canvases/" + index["plots"][0]["canvas"])
                    with tempfile.TemporaryDirectory() as temporary:
                        redrawn = Path(temporary) / "systematics.pdf"
                        canvas.SaveAs(str(redrawn))
                        check(
                            "systematic_qa_native_canvas_redraw",
                            redrawn.is_file()
                            and redrawn.read_bytes().startswith(b"%PDF-"),
                        )
        finally:
            archive.Close()
    for category, failed in failures.items():
        check("systematic_qa_" + category, not failed, "; ".join(failed[:10]))
    return {
        "variations": len(configured),
        "sources": len(groups),
        "pdfs": expected_pdf_count,
        "pdf_files": len(all_pdf_paths),
        "overview_pdf": overview_pdf,
        "canvases": len(index["plots"]),
        "numerical_leaves_checked_per_root": len(leaves),
        "percent_step_leaves_checked_per_root": len(percent_runs),
        "percent_step_runs_checked_per_root": sum(map(len, percent_runs.values())),
        "relative_display": "100 * stored fraction; ROOT HIST steps split at invalid bins",
        "root_archives_checked": 2,
        "definition": qa["std_definition"],
        "ratio_uncertainty": qa["ratio_uncertainty"],
    }


def validate_run(config, *, verify_artifacts=True):
    output = Path(config["output"])
    checks = []

    def check(name, condition, detail=""):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    if verify_artifacts:
        pipeline = Pipeline(config)
        for stage in STAGES:
            check(
                "manifest_" + stage,
                pipeline.is_current(stage),
                "Configuration, source, environment and artifact SHA256 must match.",
            )
    stage_manifests = {
        stage: strict_json(output / stage / "manifest.json") for stage in STAGES
    }
    provenance = {
        "source_fingerprint": code_fingerprint(),
        "environment": stage_manifests["report"]["environment"],
        "pipeline_signature": stage_manifests["report"]["signature"],
        "stage_seconds": {
            stage: manifest["duration_seconds"]
            for stage, manifest in stage_manifests.items()
        },
        "total_stage_seconds": sum(
            manifest["duration_seconds"] for manifest in stage_manifests.values()
        ),
        "stage_artifact_counts": {
            stage: len(manifest["artifacts"])
            for stage, manifest in stage_manifests.items()
        },
    }
    # Stage signatures are intentionally distinct under schema 2. The lineage
    # records identify consumed artifact checksums, including role-scoped
    # preparation dependencies. Historical schema 1 used a shared signature.
    lineage = object.__new__(Pipeline)
    lineage.output = output
    for stage, manifest in stage_manifests.items():
        if manifest.get("schema_version", 1) >= 2:
            check(
                "dependency_lineage_" + stage,
                manifest.get("dependencies") == lineage.dependency_identities(stage),
                "Recorded dependency signatures and artifact checksums match current upstream manifests.",
            )
    legacy = {
        stage: manifest
        for stage, manifest in stage_manifests.items()
        if manifest.get("schema_version", 1) < 2
        and stage != "raw_qa"
        and not manifest.get("refresh_scope")
    }
    if legacy:
        check(
            "legacy_stage_signatures",
            len({manifest["signature"] for manifest in legacy.values()}) == 1,
        )
    prepared = strict_json(output / "prepared/summary.json")
    training = strict_json(output / "training/metrics.json")
    working_point = strict_json(output / "working_point/summary.json")
    scored = strict_json(output / "scored/summary.json")
    nominal = strict_json(output / "nominal/results.json")
    systematic = strict_json(output / "systematics/summary.json")
    mc = strict_json(output / "mc_qa/summary.json")
    report = strict_json(output / "report/summary.json")
    from pih3l.models import load_model_bundle

    handler = load_model_bundle(output / "training/model_handler.pkl")["model"]
    estimator = handler.get_original_model()
    check(
        "training_is_current_hipe4ml_xgboost_artifact",
        training["backend"]
        == "hipe4ml.model_handler.ModelHandler[xgboost.XGBClassifier]"
        and type(handler).__module__ == "hipe4ml.model_handler"
        and type(estimator).__name__ == "XGBClassifier"
        and type(estimator).__module__.startswith("xgboost")
        and not (output / "training/model.joblib").exists(),
    )
    training_contract = validate_training_contract(
        output, config, training, handler, check
    )
    expected = {
        f"cent_{lo:g}_{hi:g}__ch_{channel}"
        for lo, hi in config["analysis"]["centrality_bins"]
        for channel in range(4)
    }
    check(
        "centrality_channel_coverage",
        set(nominal) == expected,
        f"Expected {len(expected)} centrality/charge groups.",
    )
    check(
        "separate_training_candidates",
        prepared["background"]["raw_rows"] > 0
        and prepared["signal"]["raw_rows"] > 0
        and all(
            item["train_rows"] > 0 and item["validation_rows"] > 0
            for item in training["split"].values()
        ),
    )
    if config["io"]["persist_prepared"]:
        check(
            "scores_cover_prepared_pairs",
            scored["femto"]["rows"] == prepared["femto"]["retained_rows"],
        )
    else:
        check(
            "scores_cover_raw_pair_input",
            scored["femto"]["raw_rows"] == prepared["femto"]["raw_rows"],
        )
        check("no_prepared_parquet", not parquet_paths(output / "prepared"))
        check("no_training_parquet", not parquet_paths(output / "training"))
        validation = read_validation_scores(output / "training")
        check(
            "held_out_score_archive_matches_split",
            all(
                int((validation.label == label).sum())
                == training["split"][str(label)]["validation_rows"]
                for label in (0, 1)
            ),
        )
        curve = np.loadtxt(output / "training/score_efficiency_array.txt")
        check(
            "score_efficiency_array_matches_training",
            curve.shape == (len(training["score_efficiency_array"]["efficiency"]), 2)
            and np.allclose(
                curve[:, 0], training["score_efficiency_array"]["threshold"]
            )
            and np.allclose(
                curve[:, 1], training["score_efficiency_array"]["efficiency"]
            ),
        )
    scored_directory = output / "scored/femto"
    storage = strict_json(scored_directory / "dataset.json")
    check(
        "scored_storage_raw_margin_domain", storage.get("score_domain") == "raw_margin"
    )
    reserve = config["working_point"]["storage"]
    if not reserve["enabled"] or reserve["reserve_signal_efficiency"] == 1:
        check(
            "unbounded_storage_retains_all_finite_in_scope_margins",
            storage["minimum_threshold"] is None
            and working_point["storage_threshold"] is None
            and scored["femto"]["dropped_below_storage"] == 0,
        )
    schema = parquet_columns(scored_directory)
    check("scored_storage_schema_matches_files", set(storage["columns"]) == schema)
    columns = ["bdt_score"] + (["bdt_model_id"] if "bdt_model_id" in schema else [])
    stored_rows = 0
    retained_scores_valid = True
    for frame in iter_parquet(
        scored_directory, columns=columns, batch_size=config["io"]["chunk_rows"]
    ):
        stored_rows += len(frame)
        retained_scores_valid &= stored_margin_contract(
            frame.bdt_score,
            storage["minimum_threshold"],
            frame.bdt_model_id if "bdt_model_id" in frame else None,
        )
    check("scored_parquet_row_count", stored_rows == scored["femto"]["retained_rows"])
    check("scored_scores_inside_saved_envelope", retained_scores_valid)
    ensure_storage_coverage(scored_directory, working_point["threshold"], config)
    check("nominal_inside_saved_envelope", True)
    check(
        "no_png_output",
        not list((output / "raw_qa").rglob("*.png"))
        and not list((output / "report").rglob("*.png")),
    )
    check(
        "no_invalid_in_scope_pair_scores_and_complete_row_accounting",
        score_scope_accounting(scored["femto"]),
        "Scope exclusions have NaN scores. In-scope rows must all be finite and partition into retained / below-storage counts.",
    )
    check(
        "training_group_holdout",
        all(item["group_overlap"] == 0 for item in training["split"].values()),
    )
    check(
        "finite_bdt_metrics",
        np.isfinite([training["auc_validation"], training["threshold"]]).all(),
    )
    check("ct_excluded_from_training", "ct" not in training["features"])
    check(
        "working_point_from_hyper_fit",
        working_point["status"] == "ok"
        and working_point["threshold_source"] == "hyper_data_pol0_expected_significance"
        and np.isfinite(working_point["best_scan_point"]["significance"]),
    )
    mapping = training["score_efficiency_array"]["mapping"]
    check(
        "score_efficiency_provider_hipe4ml",
        mapping["provider"] == "hipe4ml.analysis_utils.score_from_efficiency_array"
        and mapping["fallback"] == "none"
        and len(mapping["implementation_sha256"]) == 64,
    )
    pol0 = working_point["yield_pol0"]
    check(
        "working_point_positive_root_pol0",
        pol0["status"] == "ok"
        and pol0["root_fit_status"] == 0
        and pol0["c0"] > 0
        and pol0["c0_error"] > 0
        and pol0["distinct_thresholds"] >= config["working_point"]["min_pol0_points"],
    )
    valid_scan = [row for row in working_point["scan_points"] if row["valid"]]
    check(
        "working_point_uses_spectrum_pol0_objective",
        working_point["objective"] == "pol0_expected_significance"
        and all(
            np.isclose(
                row["significance"],
                row["mc_signal_efficiency"]
                * pol0["c0"]
                / np.sqrt(pol0["c0"] + row["background"]),
            )
            for row in valid_scan
        ),
    )
    scan_fits = strict_json(output / "working_point/scan_fits.json")
    check(
        "working_point_retains_all_scan_fits",
        len(scan_fits["points"]) == len(working_point["scan_points"])
        and all(
            np.isclose(point["threshold"], row["threshold"])
            and len(point["counts"]) == len(scan_fits["mass_edges"]) - 1
            for point, row in zip(scan_fits["points"], working_point["scan_points"])
        ),
    )
    wp_contract = validate_working_point_contract(
        config, working_point, scan_fits, check
    )
    wp_contract["qa"] = validate_stage_qa(
        output, training, working_point, scan_fits, report, check
    )
    cutflow = strict_json(output / "nominal/cutflow.json")
    check(
        "working_point_applied_to_femto",
        np.isclose(cutflow["threshold"], working_point["threshold"]),
    )
    check(
        "all_variations_executed",
        all(v["status"] == "completed" for v in systematic["variations"]),
    )
    configured_variations = {v["name"] for v in config["systematics"]["variations"]}
    executed_variations = [v["name"] for v in systematic["variations"]]
    check(
        "configured_variation_coverage",
        set(executed_variations) == configured_variations
        and len(executed_variations) == len(configured_variations),
    )
    systematic_qa_contract = validate_systematics_qa(
        config, output, nominal, systematic, check
    )
    check("report_has_all_groups", {row["bin"] for row in report["bins"]} == expected)
    edges = np.asarray(config["analysis"]["kstar_edges"], float)
    normalization = config["analysis"]["normalization"]
    norm = (edges[:-1] >= normalization[0] - 1e-10) & (
        edges[1:] <= normalization[1] + 1e-10
    )
    group_summaries = []
    for name in sorted(nominal):
        item = nominal[name]
        raw, corrected = item["correlations"]["signal"], item["corrected"]
        hist = item["regions"]["signal"]
        se, me = np.asarray(hist["se"], float), np.asarray(hist["me"], float)
        values = np.asarray(raw["values"], float)
        valid = np.asarray(raw["valid"], bool)
        covariance = np.asarray(raw["covariance"], float)
        check(name + "_raw_coverage", valid.any(), raw["status"])
        check(
            name + "_selected_accounting",
            item["selected_rows"] == item["selected_se"] + item["selected_me"],
        )
        check(name + "_signal_se_me", se.sum() > 0 and me.sum() > 0)
        check(name + "_invalid_bins_are_null", np.isnan(values[~valid]).all())
        check(name + "_covariance_shape", covariance.shape == (len(se), len(se)))
        active = covariance[np.ix_(valid, valid)]
        check(
            name + "_covariance_symmetric",
            np.allclose(active, active.T, atol=1e-12, equal_nan=True),
        )
        check(
            name + "_covariance_positive_semidefinite",
            not active.size
            or (
                np.isfinite(active).all() and np.linalg.eigvalsh(active).min() >= -1e-8
            ),
        )
        if se[norm].sum() > 0 and me[norm].sum() > 0:
            check(
                name + "_normalization",
                np.isclose(raw["alpha"], me[norm].sum() / se[norm].sum(), rtol=1e-12),
            )
        sideband = item["regions"]["sideband"]
        empty_sideband = sum(sideband["se"]) == 0 or sum(sideband["me"]) == 0
        if empty_sideband and item["purity"]["value"] != 1:
            check(
                name + "_no_fabricated_correction",
                not any(corrected["valid"])
                and all(v is None for v in corrected["values"]),
                corrected["status"],
            )
        group_summaries.append(
            {
                "bin": name,
                "signal_se": int(se.sum()),
                "signal_me": int(me.sum()),
                "raw_valid_bins": int(valid.sum()),
                "corrected_valid_bins": int(sum(corrected["valid"])),
                "purity_status": item["purity"]["status"],
                "corrected_status": corrected["status"],
            }
        )
    final_table = pd.read_csv(output / "systematics/final_observables.csv")
    check(
        "final_table_coverage", len(final_table) == len(expected) * (len(edges) - 1) * 2
    )
    check(
        "total_errors_quadrature",
        np.allclose(
            final_table.total_error**2,
            final_table.stat_error**2 + final_table.syst_error**2,
            atol=1e-12,
            equal_nan=True,
        ),
    )
    with np.load(output / "systematics/covariances.npz") as exported:
        check("final_covariance_coverage", len(exported.files) == len(expected) * 2 * 2)
        for name, item in nominal.items():
            for observable in ("raw", "corrected"):
                source = (
                    item["correlations"]["signal"]
                    if observable == "raw"
                    else item["corrected"]
                )
                statistical = np.asarray(source["covariance"], float)
                key = name + "__" + observable
                check(
                    key + "_total_covariance",
                    np.allclose(
                        exported[key + "__total"],
                        statistical + exported[key + "__systematic"],
                        atol=1e-12,
                        equal_nan=True,
                    ),
                )
    root_archive = validate_root_archive(output, nominal, check)
    if prepared["mc_femto"]["status"] == "missing_optional":
        check("missing_mc_is_explicit", mc["status"] == "skipped_missing_mc_femto")
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "meaning": "Software and artifact validation; not a final physics measurement.",
        "output": str(output),
        "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks),
        "checks": checks,
        "prepared": prepared,
        "scored": scored,
        "scored_storage": storage,
        "training": {
            key: training[key]
            for key in [
                "auc_validation",
                "backend",
                "threshold",
                "validation_signal_efficiency",
                "validation_background_efficiency",
                "split",
            ]
        },
        "groups": group_summaries,
        "provenance": provenance,
        "systematic_variations": systematic["variations"],
        "systematic_qa_contract": systematic_qa_contract,
        "mc_status": mc["status"],
        "root_archive": root_archive,
        "training_contract": training_contract,
        "working_point_contract": wp_contract,
        "limitations": [
            "Pair rows are statistically correlated through reused candidates, tracks and events.",
            "Pair-Poisson uncertainties and pair-weighted mass fits are approximations.",
            "A missing sideband or purity fit produces unavailable corrected values.",
            "An empty MCFemto input cannot validate primary fractions or MC closure.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/testset.yaml"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--run", action="store_true", help="Execute/resume the complete pipeline first."
    )
    parser.add_argument(
        "--skip-artifact-hashes",
        action="store_true",
        help="Skip potentially expensive manifest artifact checksum verification.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/validation/testset_validation.json",
    )
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root, output=args.output)
    if args.run:
        command = [
            sys.executable,
            "-m",
            "pih3l",
            "run",
            "-c",
            str(args.config),
            "--output",
            config["output"],
            "--data-root",
            config["data_root"],
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    result = validate_run(config, verify_artifacts=not args.skip_artifact_hashes)
    write_json(args.report, result)
    print(
        f"{result['status']}: {result['checks_passed']}/{result['checks_total']} software checks; {args.report}"
    )
    print(result["meaning"])
    for check in result["checks"]:
        if not check["passed"]:
            print(f"FAILED {check['name']}: {check['detail']}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
