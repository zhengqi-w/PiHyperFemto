"""Portable HTML wrapping plots drawn and persisted by native ROOT."""

from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd

from .io import read_json, write_json
from .root_output import render_root_report
from .raw_qa_output import qa_summary
from .report_models import model_contexts


def make_report(output, directory, config):
    output, directory = Path(output), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nominal = read_json(output / "nominal/results.json")
    training = read_json(output / "training/metrics.json")
    working_point = read_json(output / "working_point/summary.json")
    systematic = read_json(output / "systematics/summary.json")
    mc = read_json(output / "mc_qa/summary.json")
    prepared = read_json(output / "prepared/summary.json")
    root_plots = render_root_report(output, directory, config)
    statuses = []
    for key, item in nominal.items():
        statuses.append(
            {
                "bin": key,
                "se": int(sum(item["regions"]["signal"]["se"])),
                "me": int(sum(item["regions"]["signal"]["me"])),
                "raw_valid_bins": int(sum(item["correlations"]["signal"]["valid"])),
                "corrected_valid_bins": int(sum(item["corrected"]["valid"])),
                "correction_status": item["corrected"]["status"],
            }
        )
    summary = {
        "status": (
            "completed_with_unavailable_corrections"
            if any(row["corrected_valid_bins"] == 0 for row in statuses)
            else "completed_with_pair_uncertainty_approximation"
        ),
        "bins": statuses,
        "training_auc": training.get("auc_validation"),
        "training_models": [
            {
                "id": context["model_id"],
                "period_id": context["period_id"],
                "pt_range": read_json(
                    context["output"]
                    / "training"
                    / context["relative_path"]
                    / "metrics.json"
                ).get("training_pt_range"),
                "auc_validation": read_json(
                    context["output"]
                    / "training"
                    / context["relative_path"]
                    / "metrics.json"
                )["auc_validation"],
            }
            for context in model_contexts(output, training)
        ],
        "working_point": working_point,
        "mc_status": mc["status"],
        "prepared": prepared,
        "raw_qa": qa_summary(read_json(output / "raw_qa/results.json")),
        "systematic_variations": systematic["variations"],
    }
    summary["plot_backend"] = "ROOT"
    summary["training_plot_backend"] = "hipe4ml.plot_utils (Matplotlib)"
    summary["plot_format"] = "PDF"
    summary["root_file"] = "analysis.root"
    stage_qa_links = []
    for context in model_contexts(output, training):
        for stage in ("training", "working_point"):
            qa_directory = context["output"] / stage / context["relative_path"] / "qa"
            for artifact in sorted(qa_directory.glob("*")):
                if artifact.suffix in {".pdf", ".root"}:
                    relative = "../" + artifact.relative_to(output).as_posix()
                    stage_qa_links.append(
                        {
                            "stage": stage,
                            "model_id": context["model_id"],
                            "period_id": context["period_id"],
                            "path": relative,
                        }
                    )
    summary["stage_qa"] = stage_qa_links
    summary["periods"] = training.get("period_metadata")
    summary["mc_period_coverage"] = {
        key: mc.get(key)
        for key in (
            "expected_periods",
            "available_periods",
            "missing_periods",
            "complete_period_coverage",
        )
    }
    systematic_qa_index = read_json(output / "systematics/qa/index.json")
    systematic_qa_links = [
        {**plot, "path": "../systematics/qa/" + plot["path"]}
        for plot in systematic_qa_index["plots"]
    ]
    summary["systematic_qa"] = {
        "definitions": systematic_qa_index["definitions"],
        "plots": systematic_qa_links,
        "root_file": "../systematics/qa/systematics.root",
        "overview_pdf": "../systematics/qa/" + systematic_qa_index["overview_pdf"],
    }
    write_json(directory / "summary.json", summary)
    table = pd.DataFrame(statuses).to_html(index=False, escape=True, border=0)
    plot_html = "".join(
        f"<section><h2>{html.escape(title)}</h2>"
        f'<object data="{html.escape(os.path.relpath(path, directory))}" type="application/pdf">'
        f'<p><a href="{html.escape(os.path.relpath(path, directory))}">Open PDF: {html.escape(title)}</a></p>'
        f"</object></section>"
        for title, path in root_plots
    )
    stage_qa_html = "".join(
        f'<li><a href="{html.escape(item["path"])}">'
        f'{html.escape(item["stage"])} / {html.escape(item.get("period_id") or "single period")} / {html.escape(item["model_id"])} / '
        f'{html.escape(Path(item["path"]).name)}</a></li>'
        for item in stage_qa_links
    )
    systematic_qa_html = "".join(
        f'<li><a href="{html.escape(plot["path"])}">{html.escape(plot["title"])}</a></li>'
        for plot in systematic_qa_links
    )
    period_html = ""
    if training.get("mode") == "periods":
        metadata = training["period_metadata"]
        rows = [
            {
                "period": item["id"],
                "model": model["id"],
                "AUC": model["auc_validation"],
                "WP": working_point["thresholds_by_period"][item["id"]],
            }
            for item in metadata["periods"]
            for model in summary["training_models"]
            if model["period_id"] == item["id"]
        ]
        period_html = (
            "<section><h2>Period models and merged analysis</h2><p>"
            + html.escape(metadata["aggregation"])
            + "</p><p>"
            + html.escape(metadata["statistical_interpretation"])
            + '</p><div class="table">'
            + pd.DataFrame(rows).to_html(index=False, escape=True, border=0)
            + "</div><p>"
            + " | ".join(
                f'<a href="../{item["relative_path"]}/report/index.html">{html.escape(item["id"])} report</a>'
                for item in metadata["periods"]
            )
            + "</p></section>"
        )
    mc_coverage_html = ""
    if mc.get("expected_periods"):
        available = ", ".join(mc["available_periods"]) or "none"
        missing = ", ".join(mc["missing_periods"]) or "none"
        mc_coverage_html = (
            "<section><h2>MC period coverage</h2><p>Available periods: "
            + html.escape(available)
            + ". Missing or unavailable periods: "
            + html.escape(missing)
            + ".</p><p>MC fractions and resolution describe the available MC subset; full-period coverage: "
            + str(bool(mc["complete_period_coverage"]))
            + ".</p></section>"
        )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pi–H3L analysis report</title>
<style>body{{margin:0;background:#f1f4f7;color:#172a39;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:36px}}h1{{font-size:32px}}section,.notice{{background:white;padding:22px;margin:20px 0;border-radius:10px}}.notice{{border-left:5px solid #b87522}}object{{width:100%;height:760px;border:1px solid #d8dee4}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}.table{{overflow-x:auto}}code{{background:#edf1f4;padding:2px 5px}}</style>
<main><h1>Pion–hypertriton femtoscopy</h1><p>Reproducible candidate BDT → four charge channels → centrality-dependent correlation → variation ensemble → MC QA.</p>
<div class="notice"><b>Validation output, not a final physics measurement.</b><br>Statistical errors use an independent-pair approximation. Shared candidates/tracks/events are not independently counted events. No hypertriton ID is tracked in Femto.<br>Unavailable sidebands or mass purity produce null corrected values with reasons. MC status: <code>{html.escape(mc['status'])}</code>.</div>
<section><p>Physics plots are drawn by ROOT; training QA uses HIPE4ML plot_utils. All figures are exported only as PDF. <a href="analysis.root">Download analysis.root</a> for histograms, graphs, covariances and physics canvases.</p></section>
{period_html}{mc_coverage_html}<section><h2>Training and working-point QA</h2><p>Complete QA is saved by each stage, independently of this report.</p><ul>{stage_qa_html}</ul></section>
<section><h2>Systematic variation and source QA</h2><p>Standard (std) means nominal C(k*). Variation plots show C_variation/C_nominal; source plots show 100 × σ_source/|C_nominal| in percent. The raw and corrected breakdowns overlay colored source steps and a thick black total, following the spectrum layout; invalid bins remain gaps. Ratio statistical errors are omitted because the shared-sample cross-covariance is unavailable. <a href="../systematics/qa/systematics_qa.pdf">Complete QA PDF</a> | <a href="../systematics/qa/systematics.root">Standalone ROOT QA</a>.</p><details><summary>All variation/source PDFs</summary><ul>{systematic_qa_html}</ul></details></section>
<section><h2>Result coverage</h2><div class="table">{table}</div></section>{plot_html}
<section><h2>Reproducibility and interpretation</h2><p>All numerical results and covariance matrices are exported beside stage manifests. Training uses independent Hyper/MCHyper candidate trees with primary-vertex group holdout. The Hyper mass scan uses a DSCB signal and pol1/pol2 background. A pol0 fit of raw signal within 3 core sigma divided by model efficiency defines the expected signal S. The working point maximizes efficiency times S/sqrt(S+B), with B integrated within 3 sigma at each scan point; the exact objective is recorded in working_point/summary.json. The identical threshold is applied to Femto and MCFemto pair rows. Centrality is anchored to the hypertriton-side event.</p><p>Systematic bands show configured variation spread and include statistical fluctuations. Primary fractions and resolution require the optional MCFemto input. Fits to source radii or scattering parameters require a separate validated final-state-interaction model.</p></section></main></html>"""
    (directory / "index.html").write_text(content)
