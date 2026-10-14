"""Standalone ROOT diagnostics for the independent Hyper WP mass-fit scan."""

from pathlib import Path

import numpy as np

from .errors import AnalysisError
from .io import write_json
from .purity import mass_expectation
from .root_io import RootArchive
from .root_plot import (
    canvas,
    configure_style,
    frame,
    legend,
    line,
    pad,
    save_canvas,
    text,
)


def _numbers(rows, key):
    return np.array([row.get(key, np.nan) for row in rows], dtype=float)


def _bounds(values, *, zero=False):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    if zero and low >= 0:
        low = 0.0
    span = high - low
    if span <= 0:
        span = max(max(abs(low), abs(high)) * 0.05, 1e-6)
    return (low if zero and low == 0 else low - 0.15 * span, high + 0.25 * span)


def _save_plot(archive, item, destination, filename=None):
    """Keep stable ROOT canvas names and one canonical PDF per plot."""
    path = save_canvas(archive, item, destination)
    if filename and path.name != filename:
        target = destination / filename
        path.replace(target)
        return target
    return path


def _nominal_point(summary, rows):
    """Never substitute a nearby scan cut or the maximum of a QA curve."""
    if summary.get("status", "ok") != "ok":
        return None
    threshold = summary.get("threshold")
    if threshold is None or not np.isfinite(threshold):
        return None
    explicit = summary.get("spectrum_qa", {}).get("nominal_point")
    if explicit and explicit.get("threshold") == threshold:
        return explicit
    matches = [row for row in rows if row.get("threshold") == threshold]
    if not matches:
        return None
    best = summary.get("best_scan_point", {})
    if best.get("threshold") == threshold:
        return best
    # Tied score thresholds can represent several requested efficiencies. An
    # actual nominal-efficiency match is unambiguous; otherwise leave no star.
    if len(matches) == 1:
        return matches[0]
    efficiency = summary.get("nominal_mc_signal_efficiency")
    if efficiency is not None:
        for row in matches:
            if np.isclose(
                row.get("actual_mc_signal_efficiency", np.nan),
                efficiency,
                rtol=0,
                atol=1e-12,
            ):
                return row
    return None


def _star(
    archive,
    path,
    row,
    xkey,
    ykey,
    target,
    ymin,
    ymax,
    *,
    threshold=None,
    validity="valid",
):
    accepted = row is not None and row.get(validity, False)
    marker = archive.graph(
        path,
        [row.get(xkey, np.nan)] if accepted else [],
        [row.get(ykey, np.nan)] if accepted else [],
    )
    marker.SetMarkerStyle(29)
    marker.SetMarkerSize(2.0)
    marker.SetMarkerColor(archive.ROOT.kRed + 1)
    archive.write(path, marker)
    if marker.GetN():
        marker.Draw("P SAME")
        x = float(row[xkey])
        line(archive, x, ymin, x, ymax, color=archive.ROOT.kRed + 1)
    elif threshold is not None and xkey == "threshold" and np.isfinite(threshold):
        line(archive, threshold, ymin, threshold, ymax, color=archive.ROOT.kRed + 1)
        text(
            archive,
            0.18,
            0.62,
            (
                "Nominal cut outside evaluated scan; no substituted point"
                if row is None
                else "Nominal WP available; 3#sigma QA coordinate unavailable"
            ),
            0.025,
        )
    return marker


def _significance_plot(archive, destination, summary, rows, xkey, suffix, nominal):
    name = "significance_vs_" + suffix
    item, panel = canvas(
        archive,
        name,
        "Expected significance | Spectrum window QA",
        1,
        1,
        width=900,
        height=650,
    )
    target = pad(panel, 1)
    spectrum = any("significance_3sigma" in row for row in rows)
    center_key = "significance_3sigma"
    x, central = _numbers(rows, xkey), _numbers(rows, center_key)
    upper, lower = _numbers(rows, "significance_2sigma"), _numbers(
        rows, "significance_4sigma"
    )
    valid = np.array([row.get("spectrum_qa_valid", False) for row in rows], bool)
    common = valid & np.isfinite(x + central + upper + lower)
    nominal_x = nominal.get(xkey, np.nan) if nominal else np.nan
    nominal_y = (
        nominal.get(center_key, np.nan)
        if nominal and nominal.get("spectrum_qa_valid", False)
        else np.nan
    )
    real_cut = (
        summary.get("threshold")
        if suffix == "score" and summary.get("status", "ok") == "ok"
        else np.nan
    )
    xmin, xmax = _bounds(np.r_[x, nominal_x, real_cut])
    ymin, ymax = _bounds(
        np.r_[central[valid], upper[common], lower[common], nominal_y], zero=True
    )
    # Reserve the upper third for four annotation lines, independently of the
    # curve's absolute significance. The spectrum-like plateau otherwise runs
    # through the fit-quality label on dense, high-significance scans.
    finite_y = np.r_[central[valid], upper[common], lower[common], nominal_y]
    finite_y = finite_y[np.isfinite(finite_y)]
    if len(finite_y) and finite_y.max() > 0:
        ymax = max(ymax, 1.55 * finite_y.max())
    xtitle = (
        "BDT score threshold"
        if suffix == "score"
        else "Requested held-out MC signal efficiency"
    )
    frame(archive, target, xmin, ymin, xmax, ymax, xtitle, "Expected significance")
    # Construct and persist the final window envelope from one common mask.
    # It is a change of mass-integration window, never a statistical error.
    order = np.argsort(x[common], kind="stable")
    coordinates = x[common][order]
    high = np.maximum(upper[common], lower[common])[order]
    low = np.minimum(upper[common], lower[common])[order]
    band = archive.graph(
        "scan/" + name + "/window_envelope",
        np.r_[coordinates, coordinates[::-1]],
        np.r_[high, low[::-1]],
    )
    band.SetFillColorAlpha(archive.ROOT.kAzure + 1, 0.25)
    band.SetLineColor(archive.ROOT.kAzure + 2)
    archive.write("scan/" + name + "/window_envelope", band)
    if len(coordinates) > 1:
        band.Draw("F SAME")
    curves = {}
    for field, values, mask in (
        (center_key, central, valid),
        ("significance_2sigma", upper, common),
        ("significance_4sigma", lower, common),
    ):
        selected = mask & np.isfinite(x + values)
        indices = np.flatnonzero(selected)[np.argsort(x[selected], kind="stable")]
        graph = archive.graph("scan/" + name + "/" + field, x[indices], values[indices])
        curves[field] = graph
    curve = curves[center_key]
    archive.graph(
        "scan/" + name + "/significance",
        x,
        _numbers(rows, "significance"),
        valid=[row.get("valid", False) for row in rows],
    )
    curve.SetLineColor(archive.ROOT.kAzure + 2)
    curve.SetLineWidth(3)
    archive.write("scan/" + name + "/" + center_key, curve)
    if curve.GetN():
        curve.Draw("L SAME")
    marker = _star(
        archive,
        "scan/" + name + "/nominal_wp",
        nominal,
        xkey,
        center_key,
        target,
        ymin,
        ymax,
        threshold=(
            summary.get("threshold") if summary.get("status", "ok") == "ok" else None
        ),
        validity="spectrum_qa_valid",
    )
    key = legend(archive, 0.53, 0.17, 0.91, 0.35)
    key.SetTextSize(0.027)
    key.AddEntry(curve, "3#sigma expected significance", "l")
    if band.GetN():
        key.AddEntry(band, "2#sigma-4#sigma windows (no stat. error)", "f")
    if marker.GetN():
        key.AddEntry(marker, "Nominal WP", "p")
    key.Draw()
    if nominal and nominal.get("spectrum_qa_valid"):
        info = [
            f"Nominal WP score = {nominal['threshold']:.4g}",
            f"#epsilon_{{BDT}} = {nominal['mc_signal_efficiency']:.4f}",
            f"Expected significance = {nominal['significance_3sigma']:.3g}",
            f"Mass-fit #chi^{{2}}/NDF = {nominal.get('chi2_ndf', np.nan):.3g}",
        ]
        for i, label in enumerate(info):
            text(archive, 0.17, 0.84 - i * 0.048, label, 0.029)
    if summary.get("status", "ok") != "ok":
        text(archive, 0.18, 0.53, "WP unavailable; scan diagnostics only", 0.035)
    elif not curve.GetN():
        text(
            archive,
            0.18,
            0.53,
            "3#sigma QA pol0 unavailable; no expected yield substituted",
            0.030,
        )
    archive.json(
        "scan/" + name + "/window_definition",
        {
            "central": (
                "3sigma" if spectrum else "unavailable (dynamic window fields missing)"
            ),
            "envelope": ["2sigma", "4sigma"],
            "interpretation": "window variation, not statistical uncertainty",
            "points": int(common.sum()),
            "nominal_marker": "actual nominal threshold; 3sigma maximum unless an explicit override is configured",
        },
    )
    return _save_plot(
        archive,
        item,
        destination,
        "sig_vs_score.pdf" if suffix == "score" else "sig_vs_eff.pdf",
    )


def _scan_plot(
    archive,
    destination,
    name,
    title,
    rows,
    xkey,
    fields,
    xtitle,
    ytitle,
    *,
    errors=None,
    valid=True,
    threshold=None,
    pol0=None,
    filename=None,
    nominal=None,
    status="ok",
    pol0_path="scan/yield_pol0_function",
):
    item, panel = canvas(archive, name, title, 1, 1, width=900, height=600)
    target = pad(panel, 1)
    x = _numbers(rows, xkey)
    extents = []
    for field, _ in fields:
        values = _numbers(rows, field)
        error = (
            _numbers(rows, errors[field])
            if errors and field in errors
            else np.zeros(len(rows))
        )
        extents.extend([values, values - error, values + error])
    all_y = np.concatenate(extents)
    nominal_x = nominal.get(xkey, np.nan) if nominal else np.nan
    nominal_y = (
        nominal.get(fields[0][0], np.nan)
        if nominal and nominal.get("valid", False)
        else np.nan
    )
    xmin, xmax = _bounds(np.r_[x, nominal_x])
    ymin, ymax = _bounds(np.r_[all_y, nominal_y], zero=True)
    frame(archive, target, xmin, ymin, xmax, ymax, xtitle, ytitle)
    key = legend(archive, 0.20, 0.73, 0.89, 0.89)
    mask = np.array([row.get("valid", True) for row in rows], bool) if valid else None
    colors = [
        archive.ROOT.kAzure + 2,
        archive.ROOT.kOrange + 7,
        archive.ROOT.kGreen + 2,
    ]
    for index, (field, label) in enumerate(fields):
        graph = archive.graph(
            "scan/" + name + "/" + field,
            x,
            _numbers(rows, field),
            yerror=(
                _numbers(rows, errors[field]) if errors and field in errors else None
            ),
            valid=mask,
            title=f";{xtitle};{ytitle}",
        )
        graph.SetLineColor(colors[index % len(colors)])
        graph.SetMarkerColor(colors[index % len(colors)])
        graph.SetMarkerStyle(20 + index)
        graph.SetLineWidth(2)
        if graph.GetN():
            graph.Draw("LP SAME")
        key.AddEntry(graph, label, "lp")
    if threshold is not None and np.isfinite(threshold):
        marker = line(
            archive, threshold, ymin, threshold, ymax, color=archive.ROOT.kRed + 1
        )
        key.AddEntry(marker, "Selected WP", "l")
    if pol0 and pol0.get("status") == "ok" and np.isfinite(pol0.get("c0", np.nan)):
        fit_range = pol0.get("fit_range") or [xmin, xmax]
        fit = archive.keep(
            archive.ROOT.TF1(name + "_pol0", "pol0", *map(float, fit_range))
        )
        fit.SetParameter(0, float(pol0["c0"]))
        fit.SetParError(0, float(pol0.get("c0_error", 0)))
        fit.SetLineColor(archive.ROOT.kRed + 1)
        fit.SetLineWidth(2)
        fit.Draw("SAME")
        archive.write(pol0_path, fit)
        key.AddEntry(
            fit, f"pol0 = {pol0['c0']:.3g} #pm {pol0.get('c0_error', 0):.2g}", "l"
        )
        text(
            archive,
            0.20,
            0.60,
            f"pol0 #chi^{{2}}/ndf = {pol0.get('chi2_ndf', float('nan')):.3g}",
            0.032,
        )
    elif pol0:
        text(
            archive,
            0.20,
            0.60,
            "pol0 unavailable: " + str(pol0.get("status", "unknown")),
            0.032,
        )
    marker = _star(
        archive,
        "scan/" + name + "/nominal_wp",
        nominal,
        xkey,
        fields[0][0],
        target,
        ymin,
        ymax,
    )
    if marker.GetN():
        key.AddEntry(marker, "Nominal WP (analysis selection)", "p")
    key.Draw()
    if not any(
        archive.objects["scan/" + name + "/" + field].GetN() for field, _ in fields
    ):
        text(archive, 0.20, 0.50, "No valid fit estimates", 0.04)
    if status != "ok":
        text(archive, 0.20, 0.48, "WP unavailable; scan diagnostics only", 0.035)
    return _save_plot(archive, item, destination, filename)


def _mass_objects(archive, prefix, source, settings):
    edges = source["mass_edges"]
    counts = np.asarray(source["counts"], float)
    expected = np.asarray(source["expected"], float)
    archive.json(prefix + "/fit", source.get("fit", {}))
    archive.json(prefix + "/scan_point", source.get("scan_point", {}))
    data = archive.hist1(
        prefix + "/data",
        edges,
        counts,
        variance=counts,
        entries=counts.sum(),
        title=";m(^{3}He+#pi) [GeV/c^{2}];Hyper candidates",
    )
    fitted = archive.hist1(
        prefix + "/expected",
        edges,
        expected,
        title=";m(^{3}He+#pi) [GeV/c^{2}];Signal + background expectation",
    )
    signal, background = np.full_like(expected, np.nan), np.full_like(expected, np.nan)
    parameters = source.get("fit", {}).get("parameters", {})
    if parameters is None:
        parameters = {}
    names = ("signal_yield", "background_yield", "mean", "sigma", "slope")
    if "alpha_left" in parameters:
        from .wp_mass_fit import mass_components

        try:
            signal, background = mass_components(parameters, edges)
        except AnalysisError:
            pass  # Failed/nonphysical fit stays unavailable in the saved diagnostics.
    elif all(name in parameters for name in names):
        values = np.array([parameters[name] for name in names], float)
        if np.isfinite(values).all() and values[3] > 0:
            mode = settings.get("purity", {}).get("background", "linear")
            signal = mass_expectation([values[0], 0.0, *values[2:]], edges, mode)
            background = mass_expectation([0.0, *values[1:]], edges, mode)
    if "signal_expected" in source:
        signal = np.asarray(source["signal_expected"], float)
    if "background_expected" in source:
        background = np.asarray(source["background_expected"], float)
    objects = {"data": data, "expected": fitted}
    for name, values in (
        ("signal", signal),
        ("background", background),
    ):
        objects[name] = archive.hist1(prefix + "/" + name, edges, values)
    # Draw sampled continuous functions in the data histogram's count units.
    # Keep the exact bin-integrated histograms above as numerical QA interfaces.
    curve_x = np.linspace(edges[0], edges[-1], 1025)
    signal_curve, background_curve = np.full((2, len(curve_x)), np.nan)
    if "alpha_left" in parameters:
        from .wp_mass_fit import mass_densities

        try:
            signal_curve, background_curve = mass_densities(
                parameters, (edges[0], edges[-1]), curve_x
            )
            bin_width = float(np.diff(edges)[0])
            signal_curve *= bin_width
            background_curve *= bin_width
        except AnalysisError:
            pass
    elif all(name in parameters for name in names):
        # Legacy Gaussian fixtures/archives retain their own continuous model.
        from scipy.special import ndtr

        ns, nb, mean, sigma, slope = (parameters[name] for name in names)
        if np.isfinite([ns, nb, mean, sigma, slope]).all() and sigma > 0:
            width, bin_width = edges[-1] - edges[0], float(np.diff(edges)[0])
            norm = ndtr((edges[-1] - mean) / sigma) - ndtr((edges[0] - mean) / sigma)
            signal_curve = (
                ns
                * np.exp(-0.5 * ((curve_x - mean) / sigma) ** 2)
                / (np.sqrt(2 * np.pi) * sigma * norm)
                * bin_width
            )
            if settings.get("purity", {}).get("background", "linear") != "linear":
                slope = 0.0
            background_curve = (
                nb
                / width
                * (1 + 2 * slope * (curve_x - (edges[-1] + edges[0]) / 2) / width)
                * bin_width
            )
    for name, values in (
        ("total", signal_curve + background_curve),
        ("signal", signal_curve),
        ("background", background_curve),
    ):
        objects[name + "_curve"] = archive.graph(
            prefix + "/curves/" + name,
            curve_x,
            values,
            title=";m(^{3}He+#pi) [GeV/c^{2}];Counts per data mass bin",
        )
    archive.json(
        prefix + "/component_definition",
        {
            "model": "DSCB RooCrystalBall + RooChebychev pol1/pol2, normalized over mass_range",
            "reference_scope": "Spectrum data histogram geometry and DSCB/3sigma WP convention",
            "layout": "single mass-fit panel; no Data-Fit or pull panel",
            "rendering": "continuous normalized PDF sampled as TGraph curves; data uses histogram errors",
            "curve_units": "counts/GeV times the data histogram bin width in GeV; no interpolation of coarse bin expectations",
            "curve_points": 1025,
        },
    )
    return objects


def _mass_plot(
    archive,
    destination,
    name,
    title,
    source,
    histograms,
    settings,
    *,
    filename=None,
    status="ok",
    nominal_threshold=None,
):
    item, panel = canvas(archive, name, title, 1, 1, width=1100, height=750)
    target = pad(panel, 1)
    edges = np.asarray(source["mass_edges"], float)
    counts = np.asarray(source["counts"], float)
    expectation = np.asarray(source["expected"], float)
    fitted = histograms["total_curve"]
    curve_y = np.array([fitted.GetPointY(i) for i in range(fitted.GetN())])
    finite = np.r_[counts + np.sqrt(counts), expectation, curve_y]
    finite = finite[np.isfinite(finite)]
    ymax = max(float(finite.max()) if len(finite) else 1.0, 1.0) * 1.65
    bin_width = float(np.diff(edges)[0] * 1000)
    frame(
        archive,
        target,
        edges[0],
        0,
        edges[-1],
        ymax,
        "m(^{3}He+#pi) [GeV/c^{2}]",
        f"Counts / ({bin_width:g} MeV/c^{{2}})",
    )
    data = histograms["data"]
    data.SetMarkerStyle(20)
    data.SetMarkerSize(0.7)
    data.SetLineColor(archive.ROOT.kBlack)
    data.Draw("E SAME")
    fitted.SetLineColor(archive.ROOT.kBlue + 1)
    fitted.SetLineWidth(2)
    if fitted.GetN():
        fitted.Draw("L SAME")
    key = legend(archive, 0.61, 0.66, 0.93, 0.87)
    key.AddEntry(data, "Data", "pe")
    if fitted.GetN():
        key.AddEntry(fitted, "DSCB + background fit", "l")
    for label, color, style in (
        ("signal", archive.ROOT.kGreen + 1, 3),
        ("background", archive.ROOT.kRed + 1, 2),
    ):
        component = histograms[label + "_curve"]
        component.SetLineColor(color)
        component.SetLineStyle(style)
        component.SetLineWidth(2)
        values = np.array([component.GetPointY(i) for i in range(component.GetN())])
        if np.isfinite(values).any() and np.nansum(values) > 0:
            component.Draw("L SAME")
            key.AddEntry(
                component, "DSCB signal" if label == "signal" else "Background", "l"
            )
    key.Draw()
    row = source.get("scan_point", {})
    fit = source.get("fit", {})
    pars = fit.get("parameters") or {}
    window = row.get("window_3sigma")
    if window is None and "mean" in pars and "sigma" in pars:
        window = [
            max(edges[0], pars["mean"] - 3 * pars["sigma"]),
            min(edges[-1], pars["mean"] + 3 * pars["sigma"]),
        ]
    if window is not None:
        for edge in window:
            line(archive, edge, 0, edge, ymax * 0.57, color=archive.ROOT.kGray + 2)
    lines = [settings.get("period_text", "ALICE Run 3 PbPb")]
    if settings.get("additional_text"):
        lines.append(settings["additional_text"])
    if row:
        lines += [
            f"BDT score #geq {row.get('threshold', float('nan')):.4g}, #epsilon_{{BDT}} = {row.get('mc_signal_efficiency', float('nan')):.4f}",
            f"Expected significance = {row.get('significance_3sigma', row.get('significance', float('nan'))):.3g}",
            f"S(3#sigma) = {row.get('raw_signal_3sigma', row.get('raw_signal', float('nan'))):.1f}, B(3#sigma) = {row.get('background_3sigma', row.get('background', float('nan'))):.1f}",
            f"#chi^{{2}}/NDF = {row.get('chi2_ndf', float('nan')):.3g}",
        ]
    if pars:
        lines.append(
            f"#mu = {pars['mean']:.5f}, #sigma = {pars['sigma']:.5f} GeV/c^{{2}}"
        )
    lines += [
        f"Background: {fit.get('background_model', settings.get('mass_fit', {}).get('background', 'pol1'))}",
        f"Fit status: {fit.get('status', 'unknown')}; cov. quality = {fit.get('covariance_quality', 'n/a')}",
    ]
    if status != "ok":
        lines.append("WP unavailable; scan diagnostic")
    elif nominal_threshold is not None and row.get("threshold") != nominal_threshold:
        lines.append(
            f"Nominal override #geq {nominal_threshold:.4g}; shown: scan optimum"
        )
    for i, label in enumerate(lines):
        text(archive, 0.17, 0.86 - i * 0.047, label, 0.026)
    target.RedrawAxis()
    return _save_plot(archive, item, destination, filename)


def _scan_step_errors(values):
    """Half nearest nonzero scan spacing; this is not an efficiency uncertainty."""
    values = np.asarray(values, float)
    unique = np.unique(values[np.isfinite(values)])
    result = np.zeros(len(values))
    for index, value in enumerate(values):
        distances = np.abs(unique - value)
        distances = distances[distances > 0]
        if len(distances):
            result[index] = distances.min() / 2
    return result


def _mean_sigma_plot(archive, destination, summary, rows, nominal):
    item, panel = canvas(
        archive,
        "mean_sigma_vs_efficiency",
        "Mass mean and resolution across WP scan",
        1,
        2,
        width=900,
        height=850,
    )
    requested = _numbers(rows, "mc_signal_efficiency")
    valid = np.array(
        [row.get("valid", False) and row.get("fit_status") == "ok" for row in rows],
        bool,
    )
    xerrors = _scan_step_errors(requested[valid])
    for index, (field, color) in enumerate(
        (("mean", archive.ROOT.kAzure + 2), ("sigma", archive.ROOT.kRed + 1)), 1
    ):
        target = pad(panel, index)
        target.SetLeftMargin(0.17)
        y, error = _numbers(rows, field), _numbers(rows, field + "_error")
        nominal_x = nominal.get("mc_signal_efficiency", np.nan) if nominal else np.nan
        xmin, xmax = _bounds(np.r_[requested[valid], nominal_x])
        ymin, ymax = _bounds(np.r_[y[valid] - error[valid], y[valid] + error[valid]])
        frame(
            archive,
            target,
            xmin,
            ymin,
            xmax,
            ymax,
            "Requested MC efficiency" if index == 2 else "",
            (
                "Fitted mean [GeV/c^{2}]"
                if field == "mean"
                else "Fitted #sigma [GeV/c^{2}]"
            ),
        )
        graph = archive.graph(
            "stability/" + field + "_requested_efficiency",
            requested[valid],
            y[valid],
            xerror=xerrors,
            yerror=error[valid],
        )
        graph.SetMarkerStyle(25)
        graph.SetMarkerColor(color)
        graph.SetLineColor(color)
        archive.write("stability/" + field + "_requested_efficiency", graph)
        if graph.GetN():
            graph.Draw("P SAME")
        if (
            nominal is not None
            and nominal.get("valid", False)
            and np.isfinite(nominal_x)
        ):
            line(archive, nominal_x, ymin, nominal_x, ymax, color=archive.ROOT.kRed + 1)
            text(archive, 0.21, 0.85, f"Nominal WP #epsilon = {nominal_x:.4g}", 0.032)
        elif summary.get("status", "ok") != "ok":
            text(
                archive, 0.21, 0.85, "WP unavailable; fit parameter diagnostics", 0.032
            )
        text(
            archive,
            0.21,
            0.77,
            "X errors: half nearest scan step | Y errors: fit covariance",
            0.027,
        )
    archive.json(
        "stability/error_definition",
        {
            "xerror": "half nearest nonzero requested-efficiency scan spacing",
            "yerror": "fitted parameter standard error",
            "not_mc_efficiency_statistical_error": True,
        },
    )
    return _save_plot(archive, item, destination)


def render_working_point_qa(directory, summary, best_fit, scan_fits):
    """Persist every scan fit; optionally export one PDF per score fit frame."""
    destination = Path(directory) / "qa"
    destination.mkdir(parents=True, exist_ok=True)
    rows = summary["scan_points"]
    points = scan_fits["points"]
    if len(points) != len(rows):
        raise AnalysisError("WP QA scan fit count does not match scan points")
    settings = summary.get("resolved_settings", best_fit.get("resolved_settings", {}))
    nominal = _nominal_point(summary, rows)
    spectrum = summary.get("spectrum_qa", {})
    paths = []
    with RootArchive(destination / "working_point.root") as archive:
        configure_style(archive.ROOT)
        archive.json("metadata/summary", summary)
        archive.json("metadata/resolved_settings", settings)
        if "mc_signal_fit" in summary:
            mc = summary["mc_signal_fit"]
            expected = mc["fit"].get("expected", np.full(len(mc["counts"]), np.nan))
            _mass_objects(
                archive, "mc_signal_fit", {**mc, "expected": expected}, settings
            )
            archive.json("mc_signal_fit/calibration", mc)
        archive.json(
            "metadata/significance_definition",
            {
                "objective": summary.get("objective"),
                "formula": summary.get("significance_formula"),
                "signal_window": settings.get("signal_window"),
                "nominal_window_convention": "DSCB mean +/- 3 core sigma; common S3/eff pol0 and B3 objective",
                "spectrum_qa": spectrum,
            },
        )
        archive.json("scan/yield_pol0", summary.get("yield_pol0", {}))
        archive.json(
            "scan/spectrum_yield_pol0",
            spectrum.get("yield_pol0", {"status": "unavailable"}),
        )
        archive.hist1(
            "scan/valid",
            np.arange(len(rows) + 1),
            [row.get("valid", False) for row in rows],
        )
        for xkey, suffix, xtitle in (
            ("threshold", "score", "BDT score threshold"),
            (
                "mc_signal_efficiency",
                "efficiency",
                "Requested held-out MC signal efficiency",
            ),
        ):
            paths.append(
                _significance_plot(
                    archive, destination, summary, rows, xkey, suffix, nominal
                )
            )
        paths.append(
            _scan_plot(
                archive,
                destination,
                "raw_yield_over_efficiency",
                "Fitted 3#sigma raw signal / MC efficiency",
                rows,
                "mc_signal_efficiency",
                [
                    (
                        "raw_signal_3sigma_over_efficiency",
                        "Raw S(3#sigma) / MC efficiency",
                    )
                ],
                "Requested held-out MC signal efficiency",
                "Fitted signal / efficiency",
                errors={
                    "raw_signal_3sigma_over_efficiency": "raw_signal_3sigma_over_efficiency_error"
                },
                pol0=spectrum.get("yield_pol0", {"status": "unavailable"}),
                valid=True,
                nominal=nominal,
                filename="s3overeff_vs_eff.pdf",
                status=summary.get("status", "ok"),
                pol0_path="scan/spectrum_yield_pol0_function",
            )
        )
        # Stable machine-readable alias for the same nominal 3sigma definition.
        archive.graph(
            "scan/raw_yield_over_efficiency/signal_over_efficiency",
            _numbers(rows, "mc_signal_efficiency"),
            _numbers(rows, "signal_over_efficiency"),
            yerror=_numbers(rows, "signal_over_efficiency_error"),
            valid=[row.get("valid", False) for row in rows],
        )
        paths.append(
            _scan_plot(
                archive,
                destination,
                "raw_signal_yield",
                "Fitted Hyper signal yield",
                rows,
                "mc_signal_efficiency",
                [("raw_signal", "Fitted raw signal")],
                "Requested held-out MC signal efficiency",
                "Fitted signal candidates",
                errors={"raw_signal": "raw_signal_error"},
                status=summary.get("status", "ok"),
            )
        )
        paths.append(_mean_sigma_plot(archive, destination, summary, rows, nominal))
        item, panel = canvas(
            archive,
            "fit_stability",
            "Mass-fit stability across WP scan",
            3,
            1,
            width=1550,
            height=550,
        )
        for index, (field, error, ytitle) in enumerate(
            (
                ("mean", "mean_error", "Fitted mass mean [GeV/c^{2}]"),
                ("sigma", "sigma_error", "Fitted #sigma [GeV/c^{2}]"),
                ("chi2_ndf", None, "#chi^{2}/ndf"),
            ),
            1,
        ):
            target = pad(panel, index)
            target.SetLeftMargin(0.22)
            x, y = _numbers(rows, "actual_mc_signal_efficiency"), _numbers(rows, field)
            xmin, xmax = _bounds(x)
            uncertainty = _numbers(rows, error) if error else np.zeros(len(rows))
            ymin, ymax = _bounds(np.concatenate([y, y - uncertainty, y + uncertainty]))
            axes = frame(
                archive,
                target,
                xmin,
                ymin,
                xmax,
                ymax,
                "Actual MC efficiency",
                ytitle,
            )
            axes.GetYaxis().SetTitleSize(0.037)
            axes.GetYaxis().SetLabelSize(0.035)
            axes.GetYaxis().SetTitleOffset(2.5)
            graph = archive.graph(
                "stability/" + field,
                x,
                y,
                yerror=_numbers(rows, error) if error else None,
                valid=[row.get("fit_status") == "ok" for row in rows],
            )
            graph.SetMarkerStyle(20)
            graph.SetLineColor(archive.ROOT.kAzure + 2)
            if graph.GetN():
                graph.Draw("LP SAME")
            if summary.get("status", "ok") != "ok":
                text(archive, 0.26, 0.82, "WP unavailable", 0.030)
        paths.append(save_canvas(archive, item, destination))
        if best_fit:
            best_objects = _mass_objects(archive, "best_mass_fit", best_fit, settings)
            paths.append(
                _mass_plot(
                    archive,
                    destination,
                    "best_mass_fit",
                    "Hypertriton DSCB mass fit | best WP",
                    best_fit,
                    best_objects,
                    settings,
                    filename="best_fit.pdf",
                    status=summary.get("status", "ok"),
                    nominal_threshold=summary.get("threshold"),
                )
            )
        save_frames = summary.get(
            "save_score_fit_frames", settings.get("save_score_fit_frames", False)
        )
        for index, (point, row) in enumerate(zip(points, rows)):
            if not np.isclose(point["threshold"], row["threshold"], rtol=0, atol=1e-12):
                raise AnalysisError(
                    f"WP QA scan fit threshold mismatch at point {index}"
                )
            source = {**point, "mass_edges": scan_fits["mass_edges"], "scan_point": row}
            objects = _mass_objects(
                archive, f"mass_scan/point_{index:04d}", source, settings
            )
            if save_frames:
                paths.append(
                    _mass_plot(
                        archive,
                        destination,
                        f"mass_fit_{index:04d}",
                        f"Hyper mass fit | score threshold {row['threshold']:.5g}",
                        source,
                        objects,
                        settings,
                        status=summary.get("status", "ok"),
                    )
                )
        index = {
            "schema_version": 2,
            "status": summary.get("status", "unknown"),
            "spectrum_qa_status": spectrum.get("status", "unavailable"),
            "window_definition": spectrum.get(
                "window_convention",
                "dynamic mean +/- n sigma clipped to mass_range; unavailable without numerical fields",
            ),
            "band_interpretation": "2sigma-4sigma window variation, not statistical error",
            "nominal_signal_window": "fitted DSCB mean +/- 3 core sigma; per-point window_3sigma",
            "configured_fixed_window_legacy": settings.get("signal_window"),
            "nominal_objective": summary.get("objective"),
            "best_fit_scope": "3sigma expected-significance scan optimum; explicit nominal overrides may differ",
            "nominal_marker": "actual nominal threshold; never a nearby cut or diagnostic maximum",
            "nominal_efficiency_basis": (
                nominal.get("efficiency_basis") if nominal else None
            ),
            "files": [
                "working_point.root",
                *[path.name for path in paths],
                "index.json",
            ],
            "saved_scan_fits": len(points),
            "individual_scan_pdfs": bool(save_frames),
        }
        archive.json("metadata/qa_index", index)
        write_json(destination / "index.json", index)
    return [destination / "working_point.root", *paths, destination / "index.json"]
