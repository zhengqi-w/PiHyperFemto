#!/usr/bin/env python3
"""Read-only acceptance of existing period workflows and their merged products.

The complete per-period validator targets the global-model TestSet configuration.
The aggregate histogram oracle never averages already normalized correlations.
Repeated inputs certify software arithmetic, not additional physical statistics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from pih3l.config import load_config, merge, period_configs
from pih3l.io import write_json
from pih3l.multi_period import MultiPeriodPipeline
from pih3l.pipeline import STAGES, code_fingerprint
from pih3l.report_models import model_contexts
from validate_testset import (
    canvas_payload,
    strict_json,
    validate_run,
    validate_systematics_qa,
)


def array(value):
    return np.asarray(value, float)


def same(left, right):
    a, b = array(left), array(right)
    return a.shape == b.shape and np.allclose(
        a, b, rtol=1e-10, atol=1e-11, equal_nan=True
    )


def correlation_oracle(histogram, edges, normalization):
    """Direct Jacobian of alpha * SE_i / ME_i from additive pair statistics."""
    se, me, se2, me2 = (array(histogram[name]) for name in ("se", "me", "se2", "me2"))
    edges = array(edges)
    norm = (edges[:-1] >= normalization[0] - 1e-10) & (
        edges[1:] <= normalization[1] + 1e-10
    )
    n = len(se)
    sn, mn = se[norm].sum(), me[norm].sum()
    if sn <= 0 or mn <= 0:
        return np.full(n, np.nan), np.full((n, n), np.nan), np.zeros(n, bool), np.nan
    alpha = mn / sn
    valid = (se > 0) & (me > 0)
    values = np.full(n, np.nan)
    values[valid] = alpha * se[valid] / me[valid]
    jacobian = np.zeros((n, 2 * n))
    for i in np.flatnonzero(valid):
        jacobian[i, :n] = -values[i] * norm / sn
        jacobian[i, n:] = values[i] * norm / mn
        jacobian[i, i] += alpha / me[i]
        jacobian[i, n + i] -= values[i] / me[i]
    covariance = (jacobian * np.concatenate([se2, me2])) @ jacobian.T
    covariance[~valid, :] = np.nan
    covariance[:, ~valid] = np.nan
    return values, covariance, valid, alpha


def systematic_oracle(nominal, alternatives, method):
    """Independent group envelope/RMS around the merged nominal observable."""
    nominal = array(nominal)
    size = len(nominal)
    total = np.zeros((size, size))
    valid_total = np.isfinite(nominal)
    groups = {}
    for group, candidates in alternatives.items():
        delta = np.asarray([array(candidate) - nominal for candidate in candidates])
        coverage = np.isfinite(delta).sum(axis=0)
        covariance = np.full((size, size), np.nan)
        if method == "envelope":
            excursion = np.full(size, np.nan)
            for i in range(size):
                available = delta[:, i][np.isfinite(delta[:, i])]
                if len(available):
                    excursion[i] = available[np.argmax(np.abs(available))]
            covariance = np.outer(excursion, excursion)
            ensemble_size = len(delta)
        else:
            active = np.isfinite(nominal)
            accepted = [
                row for row in delta if active.any() and np.isfinite(row[active]).all()
            ]
            ensemble_size = len(accepted)
            if accepted:
                sub = np.asarray(accepted)[:, active]
                covariance[np.ix_(active, active)] = sum(
                    np.outer(row, row) for row in sub
                ) / len(sub)
        valid_total &= np.isfinite(covariance.diagonal()) & (coverage > 0)
        total += np.nan_to_num(covariance)
        groups[group] = {
            "covariance": covariance,
            "coverage": coverage,
            "ensemble_size": ensemble_size,
        }
    if not groups:
        valid_total[:] = False
    total[~valid_total, :] = np.nan
    total[:, ~valid_total] = np.nan
    return total, valid_total, groups


def validate(config):
    output = Path(config["output"])
    children = period_configs(config)
    if not children:
        raise ValueError("This validator requires configured periods")
    checks = []

    def check(name, condition, detail=""):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    period_reports = {}
    for period, child in children:
        print(
            f"Validating period {period}: complete model, WP, score, ROOT and SHA contract",
            flush=True,
        )
        report = validate_run(child)
        period_reports[period] = report
        for item in report["checks"]:
            check(
                "period/" + period + "/" + item["name"], item["passed"], item["detail"]
            )
        print(
            f"Period {period}: {report['checks_passed']}/{report['checks_total']}",
            flush=True,
        )
    print(
        "Validating aggregate dependencies and independent histogram oracles",
        flush=True,
    )
    pipeline = MultiPeriodPipeline(config)
    manifests = {}
    for stage in STAGES:
        check("aggregate_manifest_" + stage, pipeline.is_current(stage))
        manifest = strict_json(output / stage / "manifest.json")
        manifests[stage] = manifest
        check(
            "aggregate_dependency_lineage_" + stage,
            manifest["dependencies"] == pipeline.dependency_identities(stage),
        )
    periods = [name for name, _ in children]
    child_outputs = {name: Path(child["output"]) for name, child in children}
    training = strict_json(output / "training/metrics.json")
    wp = strict_json(output / "working_point/summary.json")
    stored = strict_json(output / "scored/dataset.json")
    report = strict_json(output / "report/summary.json")
    metadata = training["period_metadata"]
    check(
        "parent_period_indexes",
        all(
            index["mode"] == "periods"
            and {item["id"] for item in index["periods"]} == set(periods)
            for index in (training, wp)
        ),
    )
    check(
        "no_fake_global_model_score_or_threshold",
        "auc_validation" not in training
        and "threshold" not in wp
        and not (output / "training/model_handler.pkl").exists()
        and not list((output / "scored").rglob("*.parquet"))
        and stored["kind"] == "period_reference_catalog",
    )
    check(
        "synthetic_repetition_label",
        metadata["synthetic_repetition"] == config["synthetic_repetition"]
        and (
            not config["synthetic_repetition"]
            or "not independent" in metadata["statistical_interpretation"]
        ),
    )
    for period, child_output in child_outputs.items():
        metrics = strict_json(child_output / "training/metrics.json")
        summary = strict_json(child_output / "working_point/summary.json")
        recorded = next(item for item in training["periods"] if item["id"] == period)
        check(
            "parent_model_index_" + period,
            recorded["relative_path"] == "periods/" + period
            and recorded["metrics"] == metrics,
        )
        recorded_wp = next(item for item in wp["periods"] if item["id"] == period)
        check(
            "parent_wp_index_" + period,
            recorded_wp["summary"] == summary
            and wp["thresholds_by_period"][period] == summary["threshold"],
        )
        record = next(item for item in stored["periods"] if item["id"] == period)
        check(
            "parent_scored_catalog_" + period,
            (output / record["scored_path"]).resolve()
            == (child_output / "scored").resolve(),
        )

    nominal = strict_json(output / "nominal/results.json")
    expected_bins = {
        f"cent_{lo:g}_{hi:g}__ch_{channel}"
        for lo, hi in config["analysis"]["centrality_bins"]
        for channel in range(4)
    }
    check("merged_centrality_channel_coverage", set(nominal) == expected_bins)
    histogram_comparisons = 0

    def inspect_histograms(label, merged, sources, settings):
        nonlocal histogram_comparisons
        check(
            label + "/bin_keys",
            set(merged) == expected_bins
            and all(set(source) == expected_bins for source in sources),
        )
        for key, item in merged.items():
            prefix = label + "/" + key
            for field in ("centrality", "channel", "channel_label", "mass_edges"):
                check(
                    prefix + "/metadata_" + field,
                    all(item[field] == source[key][field] for source in sources),
                )
            check(prefix + "/period_coverage", set(item["periods"]) == set(periods))
            for field in (
                "selected_rows",
                "selected_se",
                "selected_me",
                "mass_counts",
                "mass_vs_kstar",
            ):
                check(
                    prefix + "/additive_" + field,
                    same(
                        item[field],
                        sum(array(source[key][field]) for source in sources),
                    ),
                )
            for region, histogram in item["regions"].items():
                for field in ("se", "me", "se2", "me2"):
                    histogram_comparisons += 1
                    check(
                        prefix + "/" + region + "/sum_" + field,
                        same(
                            histogram[field],
                            sum(
                                array(source[key]["regions"][region][field])
                                for source in sources
                            ),
                        ),
                    )
                values, covariance, valid, alpha = correlation_oracle(
                    histogram,
                    settings["analysis"]["kstar_edges"],
                    settings["analysis"]["normalization"],
                )
                actual = item["correlations"][region]
                check(
                    prefix + "/" + region + "/normalized_cf_covariance",
                    same(actual["values"], values)
                    and same(actual["covariance"], covariance)
                    and np.array_equal(actual["valid"], valid)
                    and same(actual["alpha"], alpha),
                )
            if (
                not any(item["regions"]["sideband"]["se"])
                and item["purity"]["value"] != 1
            ):
                check(
                    prefix + "/unavailable_correction_preserved",
                    not any(item["corrected"]["valid"])
                    and all(value is None for value in item["corrected"]["values"]),
                )

    child_nominal = [
        strict_json(path / "nominal/results.json") for path in child_outputs.values()
    ]
    inspect_histograms("nominal", nominal, child_nominal, config)
    systematic = strict_json(output / "systematics/summary.json")
    configured = {item["name"]: item for item in config["systematics"]["variations"]}
    outcomes = {item["name"]: item for item in systematic["variations"]}
    child_systematics = {
        period: strict_json(path / "systematics/summary.json")
        for period, path in child_outputs.items()
    }
    check("merged_variation_coverage", set(outcomes) == set(configured))
    variations = {}
    for name, settings in configured.items():
        statuses = {
            period: next(row for row in summary["variations"] if row["name"] == name)
            for period, summary in child_systematics.items()
        }
        failed = [
            period for period, row in statuses.items() if row["status"] != "completed"
        ]
        check(
            "variation/" + name + "/period_statuses",
            outcomes[name]["period_statuses"] == statuses,
        )
        if failed:
            check(
                "variation/" + name + "/failure_not_silently_dropped",
                outcomes[name]["status"] == "failed"
                and set(outcomes[name]["failed_periods"]) == set(failed)
                and not (output / "systematics" / name / "results.json").exists(),
            )
            continue
        check(
            "variation/" + name + "/completed", outcomes[name]["status"] == "completed"
        )
        result = strict_json(output / "systematics" / name / "results.json")
        variations[name] = result
        source_results = [
            strict_json(path / "systematics" / name / "results.json")
            for path in child_outputs.values()
        ]
        inspect_histograms(
            "variation/" + name,
            result,
            source_results,
            merge(config, settings["overrides"]),
        )
    group_names = {item.get("group", name) for name, item in configured.items()}
    with np.load(output / "systematics/covariances.npz", allow_pickle=False) as final:
        for key, item in nominal.items():
            for observable in ("raw", "corrected"):

                def extract(result):
                    return (
                        result[key]["correlations"]["signal"]
                        if observable == "raw"
                        else result[key]["corrected"]
                    )

                source = extract(nominal)
                alternatives = {group: [] for group in group_names}
                for name, result in variations.items():
                    alternatives[configured[name].get("group", name)].append(
                        extract(result)["values"]
                    )
                for group in alternatives:
                    if not alternatives[group]:
                        alternatives[group].append(
                            np.full(len(source["values"]), np.nan)
                        )
                covariance, valid, expected_groups = systematic_oracle(
                    source["values"], alternatives, config["systematics"]["method"]
                )
                actual = systematic["bins"][key][observable]
                prefix = "systematic/" + key + "/" + observable
                check(
                    prefix + "/merged_source_regrouping",
                    set(actual["groups"]) == group_names
                    and same(actual["covariance"], covariance)
                    and np.array_equal(actual["valid"], valid),
                )
                for group, expected in expected_groups.items():
                    saved = actual["groups"][group]
                    check(
                        prefix + "/group_" + group,
                        same(saved["covariance"], expected["covariance"])
                        and np.array_equal(saved["coverage"], expected["coverage"])
                        and saved["ensemble_size"] == expected["ensemble_size"],
                    )
                key_npz = key + "__" + observable
                check(
                    prefix + "/exported_covariances",
                    same(final[key_npz + "__systematic"], covariance)
                    and same(
                        final[key_npz + "__total"],
                        covariance + array(source["covariance"]),
                    ),
                )
    systematic_qa = validate_systematics_qa(config, output, nominal, systematic, check)

    raw = strict_json(output / "raw_qa/results.json")
    raw_children = [
        strict_json(path / "raw_qa/results.json") for path in child_outputs.values()
    ]
    for role in ("femto", "background"):
        for field in ("raw_rows", "centrality_rows"):
            check(
                "raw_qa/" + role + "/" + field,
                raw["inputs"][role][field]
                == sum(source["inputs"][role][field] for source in raw_children),
            )
    for role, names in (("femto", ("kstar_se", "kstar_me")), ("hyper", ("mass",))):
        for key in raw[role]:
            for name in names:
                item = raw[role][key][name]
                source = [data[role][key][name] for data in raw_children]
                check(
                    "raw_qa/" + role + "/" + key + "/" + name,
                    all(item["edges"] == x["edges"] for x in source)
                    and all(
                        same(item[field], sum(array(x[field]) for x in source))
                        for field in (
                            "counts",
                            "rows",
                            "nonfinite",
                            "underflow",
                            "overflow",
                        )
                    ),
                )
    mixing = strict_json(output / "nominal/mixing_qa.json")
    mix_children = [
        strict_json(path / "nominal/mixing_qa.json") for path in child_outputs.values()
    ]
    for field in (
        "occupancy_se",
        "occupancy_me",
        "delta_centrality_se",
        "delta_centrality_me",
    ):
        check(
            "mixing/additive_" + field,
            same(mixing[field], sum(array(source[field]) for source in mix_children)),
        )
    for field in ("occupancy_edges", "delta_centrality_edges"):
        check(
            "mixing/unchanged_" + field,
            all(mixing[field] == source[field] for source in mix_children),
        )
    check(
        "mixing/missing_centrality_propagates",
        mixing["missing_pair_centrality"]
        == any(source["missing_pair_centrality"] for source in mix_children),
    )
    mc = strict_json(output / "mc_qa/summary.json")
    mc_children = {
        period: strict_json(path / "mc_qa/summary.json")
        for period, path in child_outputs.items()
    }
    available = {
        period
        for period, summary in mc_children.items()
        if summary["status"] == "available"
    }
    check(
        "mc/period_coverage",
        set(mc["expected_periods"]) == set(periods)
        and set(mc["available_periods"]) == available
        and set(mc["missing_periods"]) == set(periods) - available
        and mc["complete_period_coverage"] == (len(available) == len(periods)),
    )
    if not available:
        check(
            "mc/no_fake_primary_fraction_or_closure",
            mc["status"] == "skipped_missing_mc_femto"
            and not mc.get("bins")
            and mc["generated_training_signal_is_not_a_substitute"]
            and mc["truth_closure_status"] == "unavailable",
        )
    check(
        "report/visible_mc_period_coverage",
        report["mc_period_coverage"]
        == {
            key: mc[key]
            for key in (
                "expected_periods",
                "available_periods",
                "missing_periods",
                "complete_period_coverage",
            )
        },
    )
    root_summary = validate_period_root(
        output, training, nominal, systematic, report, check
    )
    check(
        "all_qa_figures_pdf_only",
        not [
            path
            for directory in [
                output / name for name in ("raw_qa", "systematics", "report")
            ]
            for path in directory.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".eps", ".gif"}
        ],
    )
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "meaning": "Software, persistence and aggregation validation; repeated inputs are not independent physical statistics.",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "output": str(output),
        "periods": period_reports,
        "period_metadata": metadata,
        "histogram_arrays_checked": histogram_comparisons,
        "merged_groups": {
            key: {
                "signal_se": sum(item["regions"]["signal"]["se"]),
                "signal_me": sum(item["regions"]["signal"]["me"]),
                "raw_valid": sum(item["correlations"]["signal"]["valid"]),
                "corrected_valid": sum(item["corrected"]["valid"]),
            }
            for key, item in nominal.items()
        },
        "systematic_qa": systematic_qa,
        "root_archive": root_summary,
        "mc_status": mc["status"],
        "provenance": {
            "source_fingerprint": code_fingerprint(),
            "pipeline_signature": manifests["report"]["signature"],
            "aggregate_stage_seconds": {
                stage: record["duration_seconds"] for stage, record in manifests.items()
            },
            "environment": manifests["report"]["environment"],
        },
        "limitations": [
            "Covariance uses independent weighted pair counts and does not model reused events/tracks/candidates.",
            "Synthetic repeated inputs must not be interpreted as additional luminosity or independent events.",
            "The component MC DSCB and working-point fit-quality diagnostics remain relevant after merging.",
            "Missing mass sidebands and empty MCFemto cannot be repaired by repeating input data.",
        ],
    }


def validate_period_root(output, training, nominal, systematic, report, check):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    contexts = list(model_contexts(output, training))
    expected_links = set()
    for context in contexts:
        for stage in ("training", "working_point"):
            directory = context["output"] / stage / context["relative_path"] / "qa"
            expected_links.update(
                path.resolve()
                for path in directory.glob("*")
                if path.suffix in {".pdf", ".root"}
            )
    links = report["stage_qa"]
    check(
        "report/period_model_qa_links_exact",
        {(output / "report" / item["path"]).resolve() for item in links}
        == expected_links
        and all(path.is_file() for path in expected_links),
    )
    check(
        "report/unique_period_model_namespaces",
        len({item["suffix"] for item in contexts}) == len(contexts),
    )
    path = output / "report/analysis.root"
    with ROOT.TFile.Open(str(path), "READ") as archive:
        check("report/root_readable", bool(archive) and not archive.IsZombie())
        for context in contexts:
            period, model = context["period_id"], context["model_id"]
            relative = context["relative_path"]
            child_output = context["output"]
            suffix = f"__p{len(period)}_{period}" + (
                f"__m{len(model)}_{model}" if relative != "." else "__global"
            )
            check(
                f"report/context/{period}/{model}/encoded_suffix",
                context["suffix"] == suffix,
            )
            prefix = context["root_prefix"]
            metrics = strict_json(child_output / "training" / relative / "metrics.json")
            training_prefix = prefix + (
                "training/" + relative if relative != "." else "training"
            )
            for category in ("signal", "background"):
                histogram = archive.Get(training_prefix + "/score_" + category)
                expected = metrics["qa"]["score_distributions"]["test_" + category]
                check(
                    f"report/context/{period}/{model}/score_{category}",
                    bool(histogram)
                    and same(
                        [
                            histogram.GetBinContent(i + 1)
                            for i in range(histogram.GetNbinsX())
                        ],
                        expected["counts"],
                    ),
                )
            with ROOT.TFile.Open(
                str(
                    child_output / "working_point" / relative / "qa/working_point.root"
                ),
                "READ",
            ) as source:
                for name, destination in (
                    ("significance_vs_score", "working_point"),
                    ("significance_vs_efficiency", "working_point_efficiency"),
                    ("raw_yield_over_efficiency", "working_point_yield"),
                    ("best_mass_fit", "working_point_mass_fit"),
                ):
                    original = source.Get("canvases/" + name)
                    cloned = archive.Get("canvases/" + destination + suffix)
                    check(
                        f"report/context/{period}/{model}/{destination}_clone",
                        bool(original and cloned)
                        and canvas_payload(original) == canvas_payload(cloned),
                    )
                    pdf = output / "report" / (destination + suffix + ".pdf")
                    check(
                        f"report/context/{period}/{model}/{destination}_pdf",
                        pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-"),
                    )
        check(
            "report/no_fake_global_wp_or_training",
            not archive.Get("training/score_signal")
            and not archive.Get("working_point/significance_vs_score"),
        )
        for key, item in nominal.items():
            for observable in ("raw", "corrected"):
                source = (
                    item["correlations"]["signal"]
                    if observable == "raw"
                    else item["corrected"]
                )
                histogram = archive.Get(f"nominal/{key}/cf_{observable}_stat")
                matrix = archive.Get(f"nominal/{key}/cov_{observable}_stat")
                check(
                    f"report/root/{key}/{observable}",
                    bool(histogram and matrix)
                    and same(
                        [
                            histogram.GetBinContent(i + 1)
                            for i in range(histogram.GetNbinsX())
                        ],
                        source["values"],
                    )
                    and same(
                        [
                            [
                                matrix.GetBinContent(i + 1, j + 1)
                                for j in range(matrix.GetNbinsY())
                            ]
                            for i in range(matrix.GetNbinsX())
                        ],
                        source["covariance"],
                    ),
                )
        qa_index = strict_json(output / "systematics/qa/index.json")
        with ROOT.TFile.Open(
            str(output / "systematics/qa/systematics.root"), "READ"
        ) as source:
            for plot in qa_index["plots"]:
                original = source.Get("canvases/" + plot["canvas"])
                cloned = archive.Get("canvases/systematics__" + plot["canvas"])
                check(
                    "report/systematic_canvas_clone/" + plot["canvas"],
                    bool(original and cloned)
                    and canvas_payload(original) == canvas_payload(cloned),
                )
        canvases = archive.GetDirectory("canvases")
        return {
            "path": str(path),
            "canvases": canvases.GetListOfKeys().GetSize(),
            "model_contexts": len(contexts),
            "systematic_canvas_clones": len(qa_index["plots"]),
            "systematic_percent_step_canvases": sum(
                plot["kind"] != "variation" for plot in qa_index["plots"]
            ),
            "systematic_total_observables": sorted(
                {
                    plot["observable"]
                    for plot in qa_index["plots"]
                    if plot["kind"] in {"total", "total_corrected"}
                }
            ),
            "reader": "native PyROOT TFile",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/testset.periods.yaml"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/validation/testset_periods_validation.json",
    )
    args = parser.parse_args()
    result = validate(load_config(args.config))
    write_json(args.report, result)
    print(
        f"{result['status']}: {result['checks_passed']}/{result['checks_total']} software checks; {args.report}",
        flush=True,
    )
    for item in result["checks"]:
        if not item["passed"]:
            print("FAILED", item["name"], item["detail"], flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
