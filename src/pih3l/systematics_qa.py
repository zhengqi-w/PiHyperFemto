"""Nominal-relative systematic diagnostics, without additional pair scans.

Ratio points deliberately have no statistical error bars: the nominal and
variation samples share pairs and their cross-covariance is not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import write_json
from .errors import AnalysisError
from .root_io import RootArchive
from .root_plot import (
    CHANNEL_TITLES,
    canvas,
    configure_style,
    frame,
    legend,
    line,
    pad,
    save_canvas,
    text,
)

OBSERVABLES = ("raw", "corrected")
SOURCE_TITLES = {
    "bdt": "BDT",
    "mass_window": "Mass window",
    "mass_model": "Mass model",
    "sideband": "Sidebands",
    "normalization": "Normalization",
    "pion_pid": "Pion PID",
    "pion_dca": "Pion DCA",
    "occupancy": "Occupancy",
    "close_pair": "Close pair",
    "pion_pt_min": "Pion p_{T} min",
    "pion_pt_max": "Pion p_{T} max",
    "pion_tpc_rows": "Pion TPC rows",
}


def correlation_source(item, observable):
    return item["correlations"]["signal"] if observable == "raw" else item["corrected"]


def _valid_source(source):
    values = np.asarray(source["values"], float)
    variance = np.diag(np.asarray(source["covariance"], float))
    return (
        np.asarray(source["valid"], bool)
        & np.isfinite(values)
        & np.isfinite(variance)
        & (variance >= 0)
    )


def _diagnostic(values, valid):
    return {
        "values": np.where(valid, values, np.nan),
        "valid": valid,
        "status": (
            "ok"
            if valid.all()
            else ("partial_coverage" if valid.any() else "unavailable")
        ),
    }


def _cf_diagnostic(source):
    return {
        "values": source["values"],
        "errors": np.sqrt(
            np.maximum(np.diag(np.asarray(source["covariance"], float)), 0)
        ),
        "valid": _valid_source(source),
    }


def ratio_to_nominal(nominal_source, varied_source):
    """C_variation/C_nominal; unknown shared-sample errors are not fabricated."""
    denominator = np.asarray(nominal_source["values"], float)
    numerator = np.asarray(varied_source["values"], float)
    valid = (
        _valid_source(nominal_source)
        & _valid_source(varied_source)
        & (denominator != 0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = numerator / denominator
    return _diagnostic(values, valid & np.isfinite(values))


def relative_uncertainty(source, errors, valid=None):
    """Absolute uncertainty divided by |C_nominal|, retaining invalid bins."""
    denominator = np.abs(np.asarray(source["values"], float))
    errors = np.asarray(errors, float)
    mask = (
        _valid_source(source) & (denominator != 0) & np.isfinite(errors) & (errors >= 0)
    )
    if valid is not None:
        mask &= np.asarray(valid, bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = errors / denominator
    return _diagnostic(values, mask & np.isfinite(values))


def build_systematic_qa(nominal, variations, combined, config):
    """Build small diagnostic arrays from already evaluated correlation bins."""
    qa = {
        "std_definition": "standard = nominal C(k*), not a statistical standard deviation",
        "variation_ratio": "C_variation / C_nominal",
        "relative_uncertainty": "sigma_systematic / abs(C_nominal)",
        "fraction_plot": "100 * sigma_systematic / abs(C_nominal), in percent; ROOT histogram steps with gaps at invalid bins",
        "ratio_uncertainty": "not propagated; nominal-variation cross-covariance unavailable",
        "coverage_policy": "available variation spread; partial source/bin coverage is explicitly marked, not a complete systematic budget",
        "invalid_policy": "zero/nonfinite nominal, invalid observable or missing coverage stays null with a validity mask",
        "kstar_edges": config["analysis"]["kstar_edges"],
        "variations": {},
        "sources": {},
        "bins": {},
    }
    configured = {v["name"]: v for v in config["systematics"]["variations"]}
    for metadata, results in variations:
        name = metadata["name"]
        settings = configured[name]
        group = metadata["group"]
        entry = {
            **metadata,
            "selection": settings,
            "bins": {},
            "cf": {},
            "pair_counts": {},
        }
        qa["variations"][name] = entry
        coverage = qa["sources"].setdefault(
            group, {"expected": [], "completed": [], "failed": []}
        )
        coverage["expected"].append(name)
        coverage["completed" if results is not None else "failed"].append(name)
        for key, baseline in nominal.items():
            item = results[key] if results is not None else None
            entry["cf"][key] = {
                "nominal": _cf_diagnostic(correlation_source(baseline, "raw")),
                "variation": (
                    _cf_diagnostic(correlation_source(item, "raw"))
                    if item is not None
                    else None
                ),
            }
            entry["pair_counts"][key] = {
                label: {
                    event: float(np.sum(candidate["regions"]["signal"][event]))
                    for event in ("se", "me")
                }
                for label, candidate in (("nominal", baseline), ("variation", item))
                if candidate is not None and "regions" in candidate
            }
            entry["bins"][key] = {
                observable: ratio_to_nominal(
                    correlation_source(baseline, observable),
                    (
                        correlation_source(item, observable)
                        if item is not None
                        else {
                            "values": np.full(len(qa["kstar_edges"]) - 1, np.nan),
                            "valid": np.zeros(len(qa["kstar_edges"]) - 1, bool),
                            "covariance": np.full(
                                (len(qa["kstar_edges"]) - 1,) * 2, np.nan
                            ),
                        }
                    ),
                )
                for observable in OBSERVABLES
            }
    for group, coverage in qa["sources"].items():
        coverage["status"] = (
            "complete" if not coverage["failed"] else "incomplete_variations"
        )
    for key, baseline in nominal.items():
        qa["bins"][key] = {}
        for observable in OBSERVABLES:
            source = correlation_source(baseline, observable)
            systematic = combined[key][observable]
            groups = {}
            for group, data in systematic["groups"].items():
                groups[group] = relative_uncertainty(source, data["error"])
                groups[group]["coverage"] = data["coverage"]
                groups[group]["ensemble_size"] = data.get("ensemble_size")
                groups[group]["coverage_status"] = qa["sources"][group]["status"]
                expected = len(qa["sources"][group]["expected"])
                coverage = np.asarray(data["coverage"], int)
                groups[group]["complete_coverage"] = (
                    coverage == expected
                ) & np.asarray(groups[group]["valid"], bool)
                if (
                    config["systematics"]["method"] == "rms"
                    and data.get("ensemble_size") != expected
                ):
                    groups[group]["complete_coverage"][:] = False
                if groups[group]["coverage_status"] == "complete" and np.any(
                    ~groups[group]["complete_coverage"]
                ):
                    groups[group]["coverage_status"] = "incomplete_bin_coverage"
            total = relative_uncertainty(
                source, systematic["error"], systematic["valid"]
            )
            complete = np.zeros(len(total["valid"]), bool)
            if groups:
                complete = np.all(
                    [entry["complete_coverage"] for entry in groups.values()], axis=0
                )
            total["complete_coverage"] = complete
            total["coverage_status"] = (
                "unavailable"
                if not np.any(total["valid"])
                else (
                    "incomplete_variation_coverage"
                    if (
                        any(entry["failed"] for entry in qa["sources"].values())
                        or np.any(np.asarray(total["valid"]) & ~complete)
                    )
                    else "complete"
                )
            )
            qa["bins"][key][observable] = {
                "total": total,
                "groups": groups,
            }
    return qa


def _valid_runs(diagnostic):
    """Half-open contiguous ranges; an invalid bin must not become a zero step."""
    valid = np.asarray(diagnostic["valid"], bool) & np.isfinite(
        np.asarray(diagnostic["values"], float)
    )
    changes = np.diff(np.r_[False, valid, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _export_diagnostic(archive, path, edges, diagnostic, *, percentage=False):
    values = np.asarray(diagnostic["values"], float)
    edges = np.asarray(edges, float)
    valid = np.asarray(diagnostic["valid"], bool)
    archive.hist1(path + "/values", edges, values)
    archive.hist1(path + "/valid", edges, valid.astype(float))
    archive.graph(path + "/graph", (edges[1:] + edges[:-1]) / 2, values, valid=valid)
    archive.json(
        path + "/status", {k: v for k, v in diagnostic.items() if k != "values"}
    )
    if percentage:
        runs = []
        for index, (start, stop) in enumerate(_valid_runs(diagnostic)):
            name = f"run_{index}"
            archive.hist1(
                path + "/percent_steps/" + name,
                edges[start : stop + 1],
                100 * values[start:stop],
                title=";k* [GeV/c];Systematic uncertainty (%)",
            )
            runs.append({"name": name, "start_bin": int(start), "stop_bin": int(stop)})
        archive.json(
            path + "/percent_steps/metadata",
            {"unit": "percent", "scale": 100, "runs": runs},
        )


def export_systematic_qa(archive, qa):
    """Persist numerical ratios and explicit masks alongside editable graphs."""
    edges = qa["kstar_edges"]
    archive.json(
        "systematics_qa/metadata",
        {k: v for k, v in qa.items() if k not in {"bins", "variations"}},
    )
    for name, variation in qa["variations"].items():
        archive.json(
            f"systematics_qa/variations/{name}/metadata",
            {k: v for k, v in variation.items() if k != "bins"},
        )
        for key, item in variation["bins"].items():
            for observable, diagnostic in item.items():
                _export_diagnostic(
                    archive,
                    f"systematics_qa/variations/{name}/{key}/{observable}",
                    edges,
                    diagnostic,
                )
            centers = (np.asarray(edges[1:]) + np.asarray(edges[:-1])) / 2
            for label, source in variation["cf"][key].items():
                if source is not None:
                    archive.graph(
                        f"systematics_qa/variations/{name}/{key}/cf_{label}",
                        centers,
                        source["values"],
                        yerror=source["errors"],
                        valid=source["valid"],
                    )
    for key, item in qa["bins"].items():
        for observable, diagnostics in item.items():
            _export_diagnostic(
                archive,
                f"systematics_qa/total/{key}/{observable}",
                edges,
                diagnostics["total"],
                percentage=True,
            )
            for group, diagnostic in diagnostics["groups"].items():
                _export_diagnostic(
                    archive,
                    f"systematics_qa/sources/{group}/{key}/{observable}",
                    edges,
                    diagnostic,
                    percentage=True,
                )


def _limits(graphs, *, relative=False):
    values = [0.0] if relative else [1.0]
    for graph in graphs:
        for i in range(graph.GetN()):
            value = graph.GetPointY(i)
            error = graph.GetErrorY(i) if graph.InheritsFrom("TGraphErrors") else 0
            values.extend((value - error, value + error))
    low, high = min(values), max(values)
    span = max(high - low, 0.05 if relative else 0.2)
    return (0 if relative else low - 0.12 * span), high + 0.35 * span


def _draw_graph(graph, color, marker=20, *, errors=False):
    graph.SetMarkerColor(color)
    graph.SetLineColor(color)
    graph.SetMarkerStyle(marker)
    graph.SetMarkerSize(0.8)
    if graph.GetN():
        graph.Draw("PZ SAME" if errors else "P SAME")


def _diagnostic_panel(
    archive, target, edges, graphs, title, *, relative=False, partial=False
):
    low, high = _limits([g for _, g, _, _ in graphs], relative=relative)
    frame(
        archive,
        target,
        edges[0],
        low,
        edges[-1],
        high,
        "k* [GeV/c]",
        "#sigma_{syst} / |C_{nominal}|" if relative else "C_{variation} / C_{nominal}",
    )
    line(
        archive,
        edges[0],
        0 if relative else 1,
        edges[-1],
        0 if relative else 1,
        color=archive.ROOT.kGray + 2,
    )
    text(archive, 0.15, 0.94, title, 0.044)
    labels = legend(archive, 0.16, 0.76, 0.93, 0.88)
    labels.SetNColumns(2)
    for label, graph, color, marker in graphs:
        _draw_graph(graph, color, marker)
        labels.AddEntry(graph, label, "p")
    labels.Draw()
    warnings = []
    if partial:
        warnings.append("Partial variation coverage")
    if not any(graph.GetN() for _, graph, _, _ in graphs):
        warnings.append("No valid bins")
    elif graphs[-1][1].GetN() == 0:
        warnings.append("Corrected unavailable")
    if warnings:
        text(archive, 0.15, 0.90, "; ".join(warnings), 0.025)


def _selection_title(settings):
    descriptions = []
    aliases = {
        "PairPiTrackPt": "pion pT [GeV/c]",
        "PairPiTrackTpcNClsCrossedRows": "pion TPC crossed rows",
    }
    for section, changes in settings["overrides"].items():
        for key, value in changes.items():
            if key.endswith("_cuts"):
                for column, limits in value.items():
                    name = aliases.get(column, column)
                    if not isinstance(limits, dict):
                        limits = {"min": limits[0], "max": limits[1]}
                    operations = {
                        "min": ">=",
                        "max": "<",
                        "abs_max": "abs <",
                        "equals": "=",
                    }
                    descriptions.extend(
                        f"{name} {operations[cut]} {bound:g}"
                        for cut, bound in limits.items()
                        if bound is not None
                    )
            else:
                descriptions.append(f"{key} {value}")
    if "wp_efficiency_delta" in settings:
        descriptions.append(f"WP efficiency delta {settings['wp_efficiency_delta']:+g}")
    return "; ".join(descriptions)


def _source_styles(root, groups):
    colors = (
        root.kRed + 1,
        root.kAzure + 2,
        root.kGreen + 2,
        root.kMagenta + 1,
        root.kOrange + 7,
        root.kCyan + 2,
        root.kViolet + 1,
        root.kTeal + 3,
        root.kPink + 7,
        root.kBlue + 2,
        root.kSpring - 6,
        root.kOrange + 2,
    )
    return {
        group: (colors[index % len(colors)], 1 + index // len(colors))
        for index, group in enumerate(groups)
    }


def _fraction_panel(archive, target, edges, curves, title):
    """Spectrum-style percentage steps; keep each validity gap open."""
    maximum = 0.0
    for _, _, diagnostic, _, _, _ in curves:
        values = np.asarray(diagnostic["values"], float)
        valid = np.asarray(diagnostic["valid"], bool) & np.isfinite(values)
        if valid.any():
            maximum = max(maximum, float(np.max(100 * values[valid])))
    # Reserve a separate area above the axes for a many-source legend.
    many_sources = len(curves) > 4
    target.SetTopMargin(0.34 if many_sources else 0.24)
    frame(
        archive,
        target,
        edges[0],
        0,
        edges[-1],
        max(1.0, maximum * 1.15),
        "k* [GeV/c]",
        "Systematic uncertainty (%)",
    )
    text(archive, 0.15, 0.955, title, 0.039)
    labels = legend(archive, 0.15, 0.72 if many_sources else 0.83, 0.95, 0.915)
    labels.SetNColumns(3 if many_sources else 2)
    labels.SetTextSize(0.026 if many_sources else 0.030)
    partial, unavailable, legend_entries = False, [], []
    for label, path, diagnostic, color, style, width in curves:
        runs = _valid_runs(diagnostic)
        first = None
        for index, _ in enumerate(runs):
            step_path = path + f"/percent_steps/run_{index}"
            # Each canvas owns its style: later source overlays must not mutate
            # an earlier canvas or change its appearance in the QA PDF book.
            histogram = archive.keep(archive.objects[step_path].Clone())
            histogram.SetDirectory(0)
            histogram.SetStats(False)
            histogram.SetFillStyle(0)
            histogram.SetLineColor(color)
            histogram.SetLineStyle(style)
            histogram.SetLineWidth(width)
            # ROOT's ][ option suppresses vertical lines to zero at run edges.
            histogram.Draw("HIST ][ SAME")
            first = first or histogram
        valid = np.asarray(diagnostic["valid"], bool)
        complete = np.asarray(diagnostic["complete_coverage"], bool)
        partial |= bool(np.any(valid & ~complete))
        partial |= diagnostic["coverage_status"].startswith("incomplete")
        if first is None:
            unavailable.append(label)
            # A legend sample is not drawn as a fabricated zero histogram.
            first = archive.keep(archive.ROOT.TLine())
            first.SetLineColor(color)
            first.SetLineStyle(style)
            first.SetLineWidth(width)
        legend_entries.append((first, label + (" (n/a)" if not runs else "")))
    for sample, label in sorted(
        legend_entries, key=lambda entry: not entry[1].startswith("Total sys")
    ):
        labels.AddEntry(sample, label, "l")
    labels.Draw()
    notices = []
    if len(unavailable) == len(curves):
        notices.append("No valid bins; uncertainties unavailable")
    elif unavailable:
        notices.append("Unavailable sources marked n/a")
    if partial:
        notices.append("Partial variation coverage")
    if notices:
        text(archive, 0.15, 0.68 if many_sources else 0.79, "; ".join(notices), 0.024)


def _render_fraction_qa(archive, directory, qa, config, plots):
    edges = qa["kstar_edges"]
    styles = _source_styles(archive.ROOT, qa["sources"])
    # Per-source diagnostics retain the raw/corrected comparison.
    scopes = [("source", group, None) for group in qa["sources"]] + [
        ("total", "total", "raw"),
        ("total_corrected", "total", "corrected"),
    ]
    for kind, group, observable in scopes:
        for cent_index, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
            source_plot = kind == "source"
            if source_plot:
                name_canvas = f"systematic__source__{group}__cent{cent_index}"
                title = f"Systematic source: {SOURCE_TITLES.get(group, group)} | {low:g}-{high:g}%"
                coverage = qa["sources"][group]
                subtitle = (
                    f"100 #times #sigma_{{source}} / |C_{{nominal}}|; "
                    f"{len(coverage['completed'])}/{len(coverage['expected'])} variations completed; "
                    + config["systematics"]["method"]
                )
            else:
                suffix = "" if observable == "raw" else "__corrected"
                name_canvas = f"systematic__total{suffix}__cent{cent_index}"
                title = f"Systematic fractions | {observable.title()} C(k*) | {low:g}-{high:g}%"
                subtitle = "100 #times #sigma_{syst} / |C_{nominal}|; black: total; colors: independent sources"
            item, panels = canvas(
                archive,
                name_canvas,
                title,
                2,
                2,
                width=1600 if not source_plot else 1200,
                height=1150 if not source_plot else 900,
                subtitle=subtitle,
            )
            for channel in range(4):
                key = f"cent_{low:g}_{high:g}__ch_{channel}"
                if source_plot:
                    curves = [
                        (
                            label,
                            f"systematics_qa/sources/{group}/{key}/{obs}",
                            qa["bins"][key][obs]["groups"][group],
                            color,
                            style,
                            2,
                        )
                        for obs, label, color, style in (
                            ("raw", "Raw CF", archive.ROOT.kBlue + 1, 1),
                            ("corrected", "Corrected CF", archive.ROOT.kRed + 1, 2),
                        )
                    ]
                else:
                    curves = [
                        (
                            SOURCE_TITLES.get(source, source),
                            f"systematics_qa/sources/{source}/{key}/{observable}",
                            qa["bins"][key][observable]["groups"][source],
                            *styles[source],
                            2,
                        )
                        for source in qa["sources"]
                    ]
                    # Draw total last so coincident source lines cannot hide it.
                    curves.append(
                        (
                            "Total sys",
                            f"systematics_qa/total/{key}/{observable}",
                            qa["bins"][key][observable]["total"],
                            archive.ROOT.kBlack,
                            1,
                            3,
                        )
                    )
                _fraction_panel(
                    archive,
                    pad(panels, channel + 1),
                    edges,
                    curves,
                    CHANNEL_TITLES[channel],
                )
            path = save_canvas(archive, item, directory)
            plots.append(
                {
                    "kind": kind,
                    "name": group,
                    "observable": observable,
                    "centrality": [low, high],
                    "canvas": name_canvas,
                    "path": path.name,
                    "title": title,
                }
            )


def render_systematic_qa(directory, nominal, qa, config):
    """Draw one QA per variation/source/total and centrality, using native ROOT."""
    directory = Path(directory) / "qa"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "results.json", qa)
    edges = qa["kstar_edges"]
    plots = []
    with RootArchive(directory / "systematics.root") as archive:
        configure_style(archive.ROOT)
        export_systematic_qa(archive, qa)
        for key, baseline in nominal.items():
            for observable in OBSERVABLES:
                archive.correlation(
                    f"systematics_qa/nominal/{key}",
                    observable,
                    edges,
                    correlation_source(baseline, observable),
                )
        for name, variation in qa["variations"].items():
            for cent_index, (low, high) in enumerate(
                config["analysis"]["centrality_bins"]
            ):
                name_canvas = f"variation__{name}__cent{cent_index}"
                item, panels = canvas(
                    archive,
                    name_canvas,
                    f"Variation: {name} | {low:g}-{high:g}%",
                    4,
                    2,
                    width=1600,
                    height=900,
                    subtitle="Top: raw CF with statistical errors; bottom: variation/nominal (no shared-sample error propagation)",
                )
                for channel in range(4):
                    key = f"cent_{low:g}_{high:g}__ch_{channel}"
                    cf_graphs = [
                        (
                            label.title(),
                            archive.objects[
                                f"systematics_qa/variations/{name}/{key}/cf_{label}"
                            ],
                            color,
                            marker,
                        )
                        for label, color, marker in (
                            ("nominal", archive.ROOT.kBlack, 20),
                            ("variation", archive.ROOT.kBlue + 1, 24),
                        )
                        if f"systematics_qa/variations/{name}/{key}/cf_{label}"
                        in archive.objects
                    ]
                    target = pad(panels, channel + 1)
                    ymin, ymax = _limits([graph for _, graph, _, _ in cf_graphs])
                    frame(
                        archive,
                        target,
                        edges[0],
                        ymin,
                        edges[-1],
                        ymax,
                        "k* [GeV/c]",
                        "Raw C(k*)",
                    )
                    text(archive, 0.15, 0.94, CHANNEL_TITLES[channel], 0.044)
                    labels = legend(archive, 0.16, 0.75, 0.94, 0.88)
                    labels.SetNColumns(2)
                    for label, graph, color, marker in cf_graphs:
                        _draw_graph(graph, color, marker, errors=True)
                        labels.AddEntry(graph, label, "pe")
                    labels.Draw()
                    line(
                        archive, edges[0], 1, edges[-1], 1, color=archive.ROOT.kGray + 2
                    )
                    counts = variation["pair_counts"][key]
                    for row, label in enumerate(("nominal", "variation")):
                        if label in counts:
                            text(
                                archive,
                                0.17,
                                0.23 - 0.045 * row,
                                f"{label}: SE={counts[label]['se']:.0f}, ME={counts[label]['me']:.0f}",
                                0.026,
                            )
                    graphs = [
                        (
                            label,
                            archive.objects[
                                f"systematics_qa/variations/{name}/{key}/{observable}/graph"
                            ],
                            color,
                            marker,
                        )
                        for observable, label, color, marker in (
                            ("raw", "Raw CF ratio", archive.ROOT.kBlue + 1, 20),
                            (
                                "corrected",
                                "Corrected CF ratio",
                                archive.ROOT.kRed + 1,
                                24,
                            ),
                        )
                    ]
                    target = pad(panels, channel + 5)
                    _diagnostic_panel(
                        archive, target, edges, graphs, CHANNEL_TITLES[channel]
                    )
                    if variation["status"] != "completed":
                        text(
                            archive,
                            0.17,
                            0.30,
                            "Variation failed; see metadata reason",
                            0.028,
                        )
                # Full resolved overrides stay in ROOT/JSON; keep the PDF heading concise.
                item.cd()
                text(
                    archive,
                    0.045,
                    0.89,
                    _selection_title(variation["selection"]),
                    0.015,
                )
                path = save_canvas(archive, item, directory)
                plots.append(
                    {
                        "kind": "variation",
                        "name": name,
                        "centrality": [low, high],
                        "canvas": name_canvas,
                        "path": path.name,
                        "title": f"Variation {name}: {low:g}-{high:g}%",
                    }
                )
        _render_fraction_qa(archive, directory, qa, config, plots)
        overview = directory / "systematics_qa.pdf"
        archive.canvases[0].Print(str(overview) + "[")
        for item in archive.canvases:
            width, height = item.GetWw(), item.GetWh()
            scale = min(26.0 / width, 18.0 / height)
            archive.ROOT.gStyle.SetPaperSize(width * scale, height * scale)
            item.Print(str(overview))
        archive.canvases[-1].Print(str(overview) + "]")
        if not overview.is_file() or not overview.stat().st_size:
            raise AnalysisError("ROOT did not produce the systematic QA overview PDF")
    index = {
        "backend": "ROOT",
        "format": "PDF",
        "root_file": "systematics.root",
        "numerical_file": "results.json",
        "overview_pdf": "systematics_qa.pdf",
        "plots": plots,
        "definitions": {
            k: qa[k]
            for k in (
                "std_definition",
                "variation_ratio",
                "relative_uncertainty",
                "fraction_plot",
                "ratio_uncertainty",
                "invalid_policy",
            )
        },
    }
    write_json(directory / "index.json", index)
    return index
