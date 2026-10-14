"""Native ROOT report: editable canvases and lossless binned analysis objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import merge
from .errors import AnalysisError
from .io import read_json
from .root_io import RootArchive
from .raw_qa_output import export_raw_qa, plot_raw_qa
from .systematics_qa import export_systematic_qa
from .report_models import model_contexts

from .root_plot import (
    CHANNEL_TITLES,
    configure_style,
    text as _text,
    canvas as _canvas,
    pad as _pad,
    frame as _frame,
    line as _line,
    box as _box,
    legend as _legend,
    save_canvas as _save_canvas,
)


def _export_results(archive, prefix, results, config):
    edges = config["analysis"]["kstar_edges"]
    for key, item in results.items():
        directory = f"{prefix}/{key}"
        archive.json(
            f"{directory}/status",
            {
                "centrality": item["centrality"],
                "channel": item["channel"],
                "selected_rows": item["selected_rows"],
                "purity": item["purity"],
                "raw_status": item["correlations"]["signal"]["status"],
                "corrected_status": item["corrected"]["status"],
                "normalization_alpha": item["correlations"]["signal"]["alpha"],
            },
        )
        for region, hist in item["regions"].items():
            for event in ("se", "me"):
                # Sumw2 is required for weighted ME: sqrt(sumw) is not its error.
                entries = (
                    sum(hist[event])
                    if event == "se" or config["analysis"]["mixing_weight"] == "unit"
                    else 0
                )
                archive.hist1(
                    f"{directory}/{event}_{region}",
                    edges,
                    hist[event],
                    variance=hist[event + "2"],
                    entries=entries,
                    title=";k* [GeV/c];Weighted pair counts",
                )
            label = "raw" if region == "signal" else region
            archive.correlation(
                directory,
                label + "_stat",
                edges,
                item["correlations"][region],
                mask_name=label,
            )
        archive.correlation(
            directory, "corrected_stat", edges, item["corrected"], mask_name="corrected"
        )
        archive.hist1(
            f"{directory}/mass_se",
            item["mass_edges"],
            item["mass_counts"],
            variance=item["mass_counts"],
            entries=sum(item["mass_counts"]),
            title=";m(^{3}He+#pi) [GeV/c^{2}];SE pair rows",
        )
        archive.hist2(
            f"{directory}/mass_vs_kstar",
            edges,
            item["mass_edges"],
            item["mass_vs_kstar"],
            variance=item["mass_vs_kstar"],
            title=";k* [GeV/c];m(^{3}He+#pi) [GeV/c^{2}];SE pair rows",
        )
        if "expected" in item["purity"]:
            archive.hist1(
                f"{directory}/mass_fit", item["mass_edges"], item["purity"]["expected"]
            )


def _export_final(archive, nominal, systematic, config):
    edges = config["analysis"]["kstar_edges"]
    for key, item in nominal.items():
        directory = f"final/{key}"
        for label in ("raw", "corrected"):
            source = (
                item["correlations"]["signal"] if label == "raw" else item["corrected"]
            )
            syst = systematic["bins"][key][label]
            valid = np.asarray(source["valid"], bool) & np.asarray(syst["valid"], bool)
            for kind in ("syst", "total"):
                covariance = np.asarray(syst["covariance"], float)
                if kind == "total":
                    covariance = covariance + np.asarray(source["covariance"], float)
                archive.correlation(
                    directory,
                    f"{label}_{kind}",
                    edges,
                    {
                        "values": source["values"],
                        "covariance": covariance,
                        "valid": valid,
                    },
                    xerrors=kind == "syst",
                )
            for group, group_data in syst["groups"].items():
                subdir = f"{directory}/groups/{group}"
                archive.hist1(f"{subdir}/error_{label}", edges, group_data["error"])
                archive.hist1(
                    f"{subdir}/coverage_{label}", edges, group_data["coverage"]
                )
                archive.hist2(
                    f"{subdir}/cov_{label}_syst", edges, edges, group_data["covariance"]
                )


def _export_training(archive, output, training, prefix="training", relative="."):
    archive.json(
        f"{prefix}/model_settings",
        {
            "backend": training["backend"],
            "score_domain": training.get("score_domain"),
            "score_output_margin": training.get("score_output_margin"),
            "model_params": training.get("model_params"),
            "random_state": training.get("random_state"),
            "class_weight": training.get("class_weight"),
        },
    )
    archive.graph(
        f"{prefix}/roc",
        training["roc"]["fpr"],
        training["roc"]["tpr"],
        title=";Background efficiency;Signal efficiency",
    )
    from .validation_scores import read_validation_scores

    validation = read_validation_scores(output / "training" / relative)
    edges = np.asarray(
        training["qa"]["score_distributions"]["test_all"]["edges"], float
    )
    for label, name in ((0, "background"), (1, "signal")):
        counts = np.histogram(
            validation.loc[validation.label == label, "bdt_score"], edges
        )[0]
        archive.hist1(
            f"{prefix}/score_{name}",
            edges,
            counts,
            variance=counts,
            entries=sum(counts),
            title=";BDT score;Validation candidates",
        )
        scale = 1 / (max(1, counts.sum()) * np.diff(edges))
        archive.hist1(
            f"{prefix}/score_{name}_density",
            edges,
            counts * scale,
            variance=counts * scale**2,
            title=";BDT score;Density",
        )


def _export_working_point(archive, working_point, best_fit, prefix="working_point"):
    rows = working_point["scan_points"]
    valid = np.asarray([row["valid"] for row in rows], bool)
    thresholds = [row["threshold"] for row in rows]
    archive.graph(
        f"{prefix}/significance_vs_score",
        thresholds,
        [row["significance"] for row in rows],
        valid=valid,
        title=";BDT score;Expected significance (pol0 yield)",
    )
    archive.graph(
        f"{prefix}/efficiency_vs_score",
        thresholds,
        [row["mc_signal_efficiency"] for row in rows],
        title=";BDT score;Held-out MC signal efficiency",
    )
    archive.json(
        f"{prefix}/yield_normalization", working_point.get("yield_normalization", {})
    )
    archive.json(
        f"{prefix}/mass_fit_model",
        working_point.get("resolved_settings", {}).get("mass_fit", {}),
    )
    archive.json(
        f"{prefix}/significance_definition",
        {
            "formula": working_point.get("significance_formula"),
            "window": working_point.get("nominal_window"),
        },
    )
    archive.hist1(
        f"{prefix}/best_mass_data",
        best_fit["mass_edges"],
        best_fit["counts"],
        variance=best_fit["counts"],
        entries=sum(best_fit["counts"]),
        title=";m(^{3}He+#pi) [GeV/c^{2}];Hyper candidates",
    )
    archive.hist1(
        f"{prefix}/best_mass_fit",
        best_fit["mass_edges"],
        best_fit["expected"],
        title=";m(^{3}He+#pi) [GeV/c^{2}];Fit expectation",
    )


def _export_mixing(archive, mixing):
    archive.json("mixing_qa/status", mixing)
    for event in ("se", "me"):
        counts = np.asarray(mixing["occupancy_" + event], float)
        archive.hist1(
            f"mixing_qa/occupancy_{event}",
            mixing["occupancy_edges"],
            counts,
            variance=counts,
            entries=sum(counts),
        )
        if not mixing["missing_pair_centrality"]:
            counts = np.asarray(mixing["delta_centrality_" + event], float)
            archive.hist1(
                f"mixing_qa/delta_centrality_{event}",
                mixing["delta_centrality_edges"],
                counts,
                variance=counts,
                entries=sum(counts),
            )


def _export_mc(archive, output, mc, config):
    archive.json(
        "mc_qa/status", {key: value for key, value in mc.items() if key != "bins"}
    )
    if mc["status"] != "available":
        return
    edges = mc["kstar_edges"]
    for key, item in mc["bins"].items():
        directory = "mc_qa/" + key
        for label, value in item.items():
            if isinstance(value, dict):
                archive.json(f"{directory}/{label}_status", value)
                if "value" in value:
                    archive.hist1(f"{directory}/{label}", edges, value["value"])
                    archive.hist1(f"{directory}/valid_{label}", edges, value["valid"])
                    archive.asymmetric_graph(f"{directory}/g_{label}", edges, value)
            elif np.asarray(value).ndim == 2:
                archive.hist2(
                    f"{directory}/{label}",
                    edges,
                    edges,
                    value,
                    variance=value,
                    title=";k*_{gen} [GeV/c];k*_{reco} [GeV/c];Pairs",
                )
            else:
                archive.hist1(f"{directory}/{label}", edges, value)
    for name in ("reco_closure", "truth_closure"):
        path = output / "mc_qa" / name / "results.json"
        if path.exists():
            _export_results(archive, f"mc_qa/{name}", read_json(path), config)


def _plot_correlations(archive, directory, nominal, systematic, config):
    images = []
    edges = np.asarray(config["analysis"]["kstar_edges"], float)
    root = archive.ROOT
    for cent_index, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
        subtitle = "Independent-pair uncertainty approximation"
        if config.get("periods"):
            subtitle = f"Merged {len(config['periods'])} periods; period-specific models/WPs; SE/ME summed before normalization"
            if config["synthetic_repetition"]:
                subtitle += "; repeated-input software test"
        canvas, panel = _canvas(
            archive,
            f"correlations_cent{cent_index}",
            f"#pi-^{{3}}_{{#Lambda}}H correlations | {low:g}-{high:g}% centrality",
            2,
            2,
            subtitle=subtitle,
        )
        for channel in range(4):
            pad = _pad(panel, channel + 1)
            key = f"cent_{low:g}_{high:g}__ch_{channel}"
            item = nominal[key]
            extrema = [np.array([1.0])]
            for source in (item["correlations"]["signal"], item["corrected"]):
                value = np.asarray(source["values"], float)
                error = np.sqrt(np.diag(np.asarray(source["covariance"], float)))
                extrema.extend([value - error, value + error])
            values = np.asarray(item["correlations"]["signal"]["values"], float)
            syst = np.asarray(systematic["bins"][key]["raw"]["error"], float)
            extrema.extend([values - syst, values + syst])
            all_values = np.concatenate(extrema)
            finite = all_values[np.isfinite(all_values)]
            ymin, ymax = min(0.8, float(finite.min())), max(1.2, float(finite.max()))
            span = ymax - ymin
            ymin, ymax = ymin - 0.1 * span, ymax + 0.18 * span
            _frame(archive, pad, edges[0], ymin, edges[-1], ymax, "k* [GeV/c]", "C(k*)")
            _box(
                archive,
                *config["analysis"]["normalization"],
                ymin,
                ymax,
                root.kGray,
                0.08,
            )
            band = archive.objects[f"final/{key}/g_cf_raw_syst"]
            band.SetFillColorAlpha(root.kAzure - 9, 0.5)
            band.SetLineColor(root.kAzure - 9)
            if band.GetN():
                band.Draw("2 SAME")
            raw = archive.objects[f"nominal/{key}/g_cf_raw_stat"]
            raw.SetMarkerStyle(20)
            raw.SetMarkerSize(0.75)
            raw.SetMarkerColor(root.kAzure + 2)
            raw.SetLineColor(root.kAzure + 2)
            if raw.GetN():
                raw.Draw("PZ SAME")
            corrected = archive.objects[f"nominal/{key}/g_cf_corrected_stat"]
            corrected.SetMarkerStyle(21)
            corrected.SetMarkerColor(root.kOrange + 7)
            corrected.SetLineColor(root.kOrange + 7)
            if corrected.GetN():
                corrected.Draw("PZ SAME")
            _line(archive, edges[0], 1, edges[-1], 1, color=root.kGray + 2)
            _text(archive, 0.15, 0.93, CHANNEL_TITLES[channel], 0.045)
            legend = _legend(archive)
            legend.AddEntry(raw, "Raw C(k*)", "pe")
            if band.GetN():
                legend.AddEntry(band, "Systematic variations", "f")
            if corrected.GetN():
                legend.AddEntry(corrected, "Sideband corrected", "pe")
            legend.Draw()
            if not corrected.GetN():
                _text(archive, 0.16, 0.20, "Corrected CF unavailable", 0.028)
            pad.RedrawAxis()
        images.append(
            (
                f"Correlations: {low:g}-{high:g}%",
                _save_canvas(archive, canvas, directory),
            )
        )
        canvas, panel = _canvas(
            archive,
            f"pair_spectra_cent{cent_index}",
            f"SE / ME pair spectra | {low:g}-{high:g}% centrality",
            2,
            2,
            subtitle="Signal mass window; ME shown after high-k* normalization",
        )
        for channel in range(4):
            pad = _pad(panel, channel + 1)
            key = f"cent_{low:g}_{high:g}__ch_{channel}"
            se = archive.objects[f"nominal/{key}/se_signal"]
            me = archive.keep(
                archive.objects[f"nominal/{key}/me_signal"].Clone(
                    "me_normalized_" + key
                )
            )
            me.SetDirectory(0)
            alpha = nominal[key]["correlations"]["signal"]["alpha"]
            valid_alpha = alpha is not None and np.isfinite(alpha) and alpha > 0
            if valid_alpha:
                me.Scale(1 / alpha)
            maximum = max(
                1.0,
                me.GetMaximum(),
                *(
                    se.GetBinContent(i) + se.GetBinError(i)
                    for i in range(1, se.GetNbinsX() + 1)
                ),
            )
            _frame(
                archive,
                pad,
                edges[0],
                0,
                edges[-1],
                maximum * 1.5,
                "k* [GeV/c]",
                "Weighted pair counts",
            )
            se.SetLineColor(root.kAzure + 2)
            se.SetMarkerColor(root.kAzure + 2)
            se.SetMarkerStyle(20)
            se.SetMarkerSize(0.6)
            me.SetLineColor(root.kOrange + 7)
            me.SetLineWidth(2)
            me.Draw("HIST SAME")
            se.Draw("E1 SAME")
            _text(archive, 0.15, 0.93, CHANNEL_TITLES[channel], 0.045)
            legend = _legend(archive)
            legend.AddEntry(se, "SE", "pe")
            legend.AddEntry(
                me, "ME / #alpha" if valid_alpha else "ME (unnormalized)", "l"
            )
            legend.Draw()
        images.append(
            (
                f"Pair spectra: {low:g}-{high:g}%",
                _save_canvas(archive, canvas, directory),
            )
        )
    return images


def _reuse_stage_canvas(
    archive, source_path, source_name, destination_name, *, directory=None
):
    """Copy a persisted stage canvas into the combined archive without redrawing."""
    source = archive.ROOT.TFile.Open(str(source_path), "READ")
    if not source or source.IsZombie():
        raise AnalysisError(f"Cannot read standalone QA archive: {source_path}")
    try:
        original = source.Get("canvases/" + source_name)
        if not original or not original.InheritsFrom("TCanvas"):
            raise AnalysisError(
                f"Missing standalone QA canvas {source_name}: {source_path}"
            )
        copied = original.Clone(destination_name)
        # TCanvas::Clone can retain the source name; persistence/export keys must
        # use the report name even when ROOT ignores Clone's name argument.
        copied.SetName(destination_name)
        archive.canvases.append(copied)
        if directory is not None:
            return _save_canvas(archive, copied, directory)
        archive.write("canvases/" + destination_name, copied)
    finally:
        source.Close()


def _plot_mass(archive, directory, nominal, config):
    bins = config["analysis"]["centrality_bins"]
    root = archive.ROOT
    canvas, panel = _canvas(
        archive,
        "mass_purity",
        "SE pair-weighted invariant mass",
        4,
        len(bins),
        width=1600,
        height=max(500, 390 * len(bins)),
        subtitle="Signal window and sidebands; repeated pairs are not unique candidate yields",
    )
    for row, (low, high) in enumerate(bins):
        for channel in range(4):
            pad = _pad(panel, row * 4 + channel + 1)
            key = f"cent_{low:g}_{high:g}__ch_{channel}"
            hist = archive.objects[f"nominal/{key}/mass_se"]
            ymax = max(1, hist.GetMaximum()) * 1.32
            edges = nominal[key]["mass_edges"]
            _frame(
                archive,
                pad,
                edges[0],
                0,
                edges[-1],
                ymax,
                "m(^{3}He+#pi) [GeV/c^{2}]",
                "SE pair rows",
            )
            _box(
                archive,
                *config["analysis"]["signal_window"],
                0,
                ymax,
                root.kAzure - 9,
                0.20,
            )
            for window in config["analysis"]["sidebands"]:
                _box(archive, *window, 0, ymax, root.kGray, 0.14)
            hist.SetLineColor(root.kAzure + 2)
            hist.SetLineWidth(2)
            hist.Draw("HIST SAME")
            fit = archive.objects.get(f"nominal/{key}/mass_fit")
            if fit:
                fit.SetLineColor(root.kOrange + 7)
                fit.SetLineWidth(2)
                fit.Draw("HIST SAME")
            _text(archive, 0.15, 0.94, f"{low:g}-{high:g}% | channel {channel}", 0.044)
            status = nominal[key]["purity"]["status"].replace("_", " ")
            _text(archive, 0.16, 0.82, status, 0.029)
            pad.RedrawAxis()
    return "Mass support and purity", _save_canvas(archive, canvas, directory)


def _plot_mixing(archive, directory, mixing):
    root = archive.ROOT
    canvas, panel = _canvas(
        archive, "mixing_qa", "Event mixing diagnostics", 2, 1, height=550
    )
    pad = _pad(panel, 1)
    histograms = []
    for event in ("se", "me"):
        hist = archive.keep(
            archive.objects[f"mixing_qa/occupancy_{event}"].Clone(
                "occupancy_normalized_" + event
            )
        )
        hist.SetDirectory(0)
        if hist.Integral() > 0:
            hist.Scale(1 / hist.Integral())
        histograms.append(hist)
    ymax = max(0.01, *(h.GetMaximum() for h in histograms)) * 1.25
    _frame(
        archive,
        pad,
        mixing["occupancy_edges"][0],
        0,
        mixing["occupancy_edges"][-1],
        ymax,
        "Hypertriton-side track occupancy",
        "Fraction of selected pair rows",
    )
    legend = _legend(archive)
    for event, hist, color in zip(
        ("SE", "ME"), histograms, (root.kAzure + 2, root.kOrange + 7)
    ):
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.Draw("HIST SAME")
        legend.AddEntry(hist, event, "l")
    legend.Draw()
    pad = _pad(panel, 2)
    if mixing["missing_pair_centrality"]:
        _text(archive, 0.14, 0.56, "Pair-side centrality unavailable", 0.045)
        _text(archive, 0.14, 0.45, "in compact input schema", 0.04)
    else:
        histograms = [
            archive.objects[f"mixing_qa/delta_centrality_{event}"]
            for event in ("se", "me")
        ]
        ymax = max(1, *(h.GetMaximum() for h in histograms)) * 1.25
        edges = mixing["delta_centrality_edges"]
        _frame(
            archive,
            pad,
            edges[0],
            0,
            edges[-1],
            ymax,
            "Pair centrality - hypertriton centrality [%]",
            "Pair rows",
        )
        legend = _legend(archive)
        for label, hist, color in zip(
            ("SE", "ME"), histograms, (root.kAzure + 2, root.kOrange + 7)
        ):
            hist.SetLineColor(color)
            hist.Draw("HIST SAME")
            legend.AddEntry(hist, label, "l")
        legend.Draw()
    return "Mixing QA", _save_canvas(archive, canvas, directory)


def _plot_mc(archive, directory, mc, config):
    images = []
    if mc["status"] != "available":
        return images
    root = archive.ROOT
    edges = mc["kstar_edges"]
    subtitle = "Selected signal-window pair rows; Wilson 68% intervals"
    if mc.get("expected_periods") and not mc["complete_period_coverage"]:
        subtitle += f"; partial MC: {len(mc['available_periods'])}/{len(mc['expected_periods'])} periods"
    for cent_index, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
        canvas, panel = _canvas(
            archive,
            f"mc_primary_cent{cent_index}",
            f"MC primary-pion fraction | {low:g}-{high:g}%",
            2,
            2,
            subtitle=subtitle,
        )
        for channel in range(4):
            pad = _pad(panel, channel + 1)
            _frame(
                archive,
                pad,
                edges[0],
                0,
                edges[-1],
                1.15,
                "k* [GeV/c]",
                "Primary true pion / all pairs",
            )
            _text(archive, 0.15, 0.93, CHANNEL_TITLES[channel], 0.045)
            legend = _legend(archive)
            for event, color in (("se", root.kAzure + 2), ("me", root.kOrange + 7)):
                key = f"cent_{low:g}_{high:g}__ch_{channel}__{event}"
                graph = archive.objects.get(f"mc_qa/{key}/g_primary_pion_fraction")
                if graph and graph.GetN():
                    graph.SetMarkerStyle(20 if event == "se" else 24)
                    graph.SetMarkerColor(color)
                    graph.SetLineColor(color)
                    graph.Draw("PZ SAME")
                    legend.AddEntry(graph, event.upper(), "pe")
            legend.Draw()
        images.append(
            (
                f"MC primary fraction: {low:g}-{high:g}%",
                _save_canvas(archive, canvas, directory),
            )
        )
    return images


def render_root_report(output: Path, directory: Path, config: dict):
    """Write analysis.root and export every main native ROOT canvas as PDF."""
    output, directory = Path(output), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nominal = read_json(output / "nominal/results.json")
    training = read_json(output / "training/metrics.json")
    working_point = read_json(output / "working_point/summary.json")
    systematic = read_json(output / "systematics/summary.json")
    mixing = read_json(output / "nominal/mixing_qa.json")
    mc = read_json(output / "mc_qa/summary.json")
    raw_qa = read_json(output / "raw_qa/results.json")
    with RootArchive(directory / "analysis.root") as archive:
        root = archive.ROOT
        configure_style(root)
        archive.json(
            "metadata/schema",
            {
                "version": 1,
                "backend": "ROOT",
                "root_version": root.gROOT.GetVersion(),
                "invalid_bins": "NaN content/error/covariance; separate valid masks; graphs omit invalid points",
                "histogram_entries": "Unweighted SE/ME and QA counts use actual sums; derived or weighted summaries use Entries=0. Contents and Sumw2 are authoritative.",
                "uncertainty": "Independent-pair approximation, including normalization covariance",
            },
        )
        archive.json("metadata/config", config)
        archive.json("metadata/training", training)
        if training.get("mode") == "periods":
            archive.json("metadata/periods", training["period_metadata"])
        archive.json("metadata/working_point", working_point)
        archive.json("metadata/systematics", systematic)
        archive.json(
            "metadata/provenance",
            {
                name: read_json(output / name / "manifest.json")
                for name in (
                    "raw_qa",
                    "prepared",
                    "training",
                    "working_point",
                    "scored",
                    "nominal",
                    "systematics",
                    "mc_qa",
                )
                if (output / name / "manifest.json").exists()
            },
        )
        _export_results(archive, "nominal", nominal, config)
        _export_final(archive, nominal, systematic, config)
        systematic_qa_directory = output / "systematics/qa"
        systematic_qa = read_json(systematic_qa_directory / "results.json")
        systematic_qa_index = read_json(systematic_qa_directory / "index.json")
        export_systematic_qa(archive, systematic_qa)
        systematic_plots = []
        for plot in systematic_qa_index["plots"]:
            _reuse_stage_canvas(
                archive,
                systematic_qa_directory / "systematics.root",
                plot["canvas"],
                "systematics__" + plot["canvas"],
            )
            # All variation/source figures remain beside the standalone stage.
            if plot["kind"] in {"total", "total_corrected"}:
                systematic_plots.append(
                    (plot["title"], systematic_qa_directory / plot["path"])
                )
        for variation in systematic["variations"]:
            if variation["status"] == "completed":
                result = read_json(
                    output / "systematics" / variation["name"] / "results.json"
                )
                overrides = next(
                    v["overrides"]
                    for v in config["systematics"]["variations"]
                    if v["name"] == variation["name"]
                )
                _export_results(
                    archive,
                    "variations/" + variation["name"],
                    result,
                    merge(config, overrides),
                )
        model_plots = []
        for context in model_contexts(output, training):
            model_output = context["output"]
            relative = context["relative_path"]
            per_model = relative != "."
            metrics = read_json(model_output / "training" / relative / "metrics.json")
            wp_model = read_json(
                model_output / "working_point" / relative / "summary.json"
            )
            best_fit = read_json(
                model_output / "working_point" / relative / "best_fit.json"
            )
            training_prefix = context["root_prefix"] + (
                "training/" + relative if per_model else "training"
            )
            wp_prefix = context["root_prefix"] + (
                "working_point/" + relative if per_model else "working_point"
            )
            suffix = context["suffix"]
            label = context["label"]
            _export_training(archive, model_output, metrics, training_prefix, relative)
            _export_working_point(archive, wp_model, best_fit, wp_prefix)
            for row in wp_model.get("efficiency_variations", []):
                variation_path = (
                    model_output
                    / "working_point"
                    / relative
                    / "variations"
                    / f"delta_{row['delta']:+.12g}.json"
                )
                variation = read_json(variation_path)
                prefix = f"{wp_prefix}/efficiency_variations/delta_{row['delta']:+g}"
                archive.json(prefix + "/status", row)
                counts = variation["counts"]
                archive.hist1(
                    prefix + "/mass_data",
                    variation["mass_edges"],
                    counts,
                    variance=counts,
                    entries=sum(counts),
                    title=";m(^{3}He+#pi) [GeV/c^{2}];Hyper candidates",
                )
                if variation["fit"].get("parameters"):
                    expected = variation["fit"]["expected"]
                    archive.hist1(
                        prefix + "/mass_fit",
                        variation["mass_edges"],
                        expected,
                        title=";m(^{3}He+#pi) [GeV/c^{2}];Fit expectation",
                    )
            training_qa = model_output / "training" / relative / "qa"
            wp_qa = model_output / "working_point" / relative / "qa"
            for source_name, destination_name, caption in (
                ("significance_vs_score", "working_point", "BDT working point"),
                (
                    "significance_vs_efficiency",
                    "working_point_efficiency",
                    "Working point vs efficiency",
                ),
                (
                    "raw_yield_over_efficiency",
                    "working_point_yield",
                    "3sigma signal / model efficiency",
                ),
                (
                    "best_mass_fit",
                    "working_point_mass_fit",
                    "DSCB mass fit at the working point",
                ),
            ):
                path = _reuse_stage_canvas(
                    archive,
                    wp_qa / "working_point.root",
                    source_name,
                    destination_name + suffix,
                    directory=directory,
                )
                model_plots.append((caption + label, path))
            model_plots += [
                ("BDT training" + label, training_qa / "bdt_output.pdf"),
            ]
        _export_mixing(archive, mixing)
        _export_mc(archive, output, mc, config)
        export_raw_qa(archive, raw_qa)
        images = plot_raw_qa(archive, directory, raw_qa, config)
        images += model_plots
        images += systematic_plots
        images += _plot_correlations(archive, directory, nominal, systematic, config)
        images += [
            _plot_mass(archive, directory, nominal, config),
            _plot_mixing(archive, directory, mixing),
        ]
        images += _plot_mc(archive, directory, mc, config)
    return images
