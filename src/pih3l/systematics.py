"""Named full selection/normalization/purity variations and groupwise covariance."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    CorrelationAccumulator,
    analysis_columns,
    histogram_signature,
    parquet_schema_columns,
)
from .config import merge, validate
from .correlation import empty_histograms
from .errors import AnalysisError
from .io import iter_parquet, read_json, write_json
from .systematics_qa import (
    build_systematic_qa,
    relative_uncertainty,
    render_systematic_qa,
)

LOG = logging.getLogger(__name__)


def combine_variations(nominal, variations, method="envelope"):
    """Variations in one source are alternatives; independent sources add in quadrature.

    Envelope covariance uses the signed maximal excursion as a coherent source.
    RMS uses mean outer products. Missing coverage is NaN, never zero uncertainty.
    """
    nominal = np.asarray(nominal, float)
    groups = {}
    for group, values in variations:
        groups.setdefault(group, []).append(np.asarray(values, float) - nominal)
    total = np.zeros((len(nominal), len(nominal)))
    valid_total = np.isfinite(nominal)
    details = {}
    for group, deltas in groups.items():
        delta = np.array(deltas)
        valid = np.isfinite(delta)
        coverage = valid.sum(axis=0)
        if method == "envelope":
            index = np.argmax(np.where(valid, abs(delta), -np.inf), axis=0)
            excursion = delta[index, np.arange(len(nominal))]
            covariance = np.outer(excursion, excursion)
        elif method == "rms":
            # A common ensemble is required to preserve positive semidefiniteness.
            # Pairwise deletion can invent unknown off-diagonals and non-PSD matrices.
            active = np.isfinite(nominal)
            complete = np.all(valid[:, active], axis=1) & active.any()
            covariance = np.full((len(nominal), len(nominal)), np.nan)
            if complete.any():
                matrix = delta[complete][:, active]
                covariance[np.ix_(active, active)] = matrix.T @ matrix / len(matrix)
        else:
            raise AnalysisError("Unknown systematic combination method")
        valid_total &= (coverage > 0) & np.isfinite(np.diag(covariance))
        total += np.nan_to_num(covariance)
        details[group] = {
            "coverage": coverage,
            "covariance": covariance,
            "error": np.sqrt(np.diag(covariance)),
            "ensemble_size": int(complete.sum()) if method == "rms" else len(deltas),
        }
    if not groups:
        valid_total[:] = False
    total[~valid_total, :] = np.nan
    total[:, ~valid_total] = np.nan
    return {
        "covariance": total,
        "error": np.sqrt(np.diag(total)),
        "valid": valid_total,
        "groups": details,
        "method": method,
        "status": "ok" if valid_total.all() else "incomplete_coverage",
    }


def run_systematics(scored_directory, nominal_directory, directory, config, threshold):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nominal = read_json(Path(nominal_directory) / "results.json")
    outcomes, qa_variations = [], []
    wp = (
        threshold
        if isinstance(threshold, dict)
        and "threshold" in threshold
        and isinstance(threshold.get("status"), str)
        else None
    )
    model_threshold = wp["threshold"] if wp is not None else threshold
    baseline_threshold = config["analysis"]["bdt_threshold"]
    baseline_threshold = (
        model_threshold if baseline_threshold is None else baseline_threshold
    )
    baseline_key = histogram_signature(config, baseline_threshold)
    baseline = CorrelationAccumulator(config, baseline_threshold)
    # Recover exactly the sufficient statistics, without reusing fitted results.
    for key, template in empty_histograms(config).items():
        baseline.histograms[key] = {field: nominal[key][field] for field in template}
    for filename, attribute in (("cutflow.json", "cutflow"), ("mixing_qa.json", "qa")):
        path = Path(nominal_directory) / filename
        if path.exists():
            value = read_json(path)
            setattr(
                baseline,
                attribute,
                value["counts"] if attribute == "cutflow" else value,
            )
    groups = {baseline_key: baseline}
    requests = []
    failures = {}
    available = parquet_schema_columns(scored_directory)
    for variation in config["systematics"]["variations"]:
        name = variation["name"]
        LOG.info("Systematic variation: %s", name)
        settings = merge(config, variation["overrides"])
        try:
            validate(settings)
            resolved = settings["analysis"]["bdt_threshold"]
            if "wp_efficiency_delta" in variation:
                from .working_point import resolve_wp_variation

                if wp is None:
                    raise AnalysisError(
                        "WP efficiency variations require the working-point summary"
                    )
                resolved = resolve_wp_variation(wp, variation["wp_efficiency_delta"])
            elif resolved is None:
                resolved = model_threshold
            from .scored_storage import ensure_storage_coverage

            ensure_storage_coverage(scored_directory, resolved, settings)
            signature = histogram_signature(settings, resolved)
            columns = analysis_columns(settings, available)
            missing = set(columns) - available
            if missing and signature != baseline_key:
                raise AnalysisError(
                    f"Variation requires missing scored columns: {sorted(missing)}"
                )
            if signature not in groups:
                groups[signature] = CorrelationAccumulator(settings, resolved)
            requests.append((variation, settings, signature))
        except AnalysisError as error:
            outcomes.append(
                {
                    "name": name,
                    "group": variation.get("group", name),
                    "status": "failed",
                    "reason": str(error),
                }
            )
            continue
    active = {key: value for key, value in groups.items() if key != baseline_key}
    columns = sorted(
        {
            column
            for value in active.values()
            for column in analysis_columns(value.config, available)
        }
    )
    if active:
        for frame in iter_parquet(
            scored_directory, columns=columns, batch_size=config["io"]["chunk_rows"]
        ):
            for key, accumulator in active.items():
                if key in failures:
                    continue
                try:
                    accumulator.add(frame)
                except AnalysisError as error:
                    failures[key] = str(error)
    for variation, settings, signature in requests:
        name = variation["name"]
        try:
            if signature in failures:
                raise AnalysisError(failures[signature])
            result = groups[signature].export(directory / name, settings)
        except AnalysisError as error:
            outcomes.append(
                {
                    "name": name,
                    "group": variation.get("group", name),
                    "status": "failed",
                    "reason": str(error),
                }
            )
            continue
        outcomes.append(
            {
                "name": name,
                "group": variation.get("group", name),
                "status": "completed",
                "histogram_source": (
                    "nominal" if signature == baseline_key else signature
                ),
                "threshold": groups[signature].threshold,
            }
        )
        qa_variations.append((outcomes[-1], result))
    by_name = {metadata["name"]: result for metadata, result in qa_variations}
    return finalize_systematics(
        nominal,
        [(outcome, by_name.get(outcome["name"])) for outcome in outcomes],
        directory,
        config,
        execution={
            "data_scans": int(bool(active)),
            "histogram_groups": len(active),
            "nominal_histogram_reuse": sum(
                key == baseline_key for _, _, key in requests
            ),
        },
    )


def finalize_systematics(nominal, variations, directory, config, *, execution=None):
    """Combine evaluated variations, including merged periods, with the same QA."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outcomes = [metadata for metadata, _ in variations]
    accepted = [
        (metadata["group"], result)
        for metadata, result in variations
        if result is not None
    ]
    qa_variations = variations
    combined = {}
    table = []
    covariance_arrays = {}
    required_groups = {
        v.get("group", v["name"]) for v in config["systematics"]["variations"]
    }
    for key, baseline in nominal.items():
        combined[key] = {}
        for observable in ("raw", "corrected"):

            def extract(item):
                return (
                    item["correlations"]["signal"]["values"]
                    if observable == "raw"
                    else item["corrected"]["values"]
                )

            values = [(group, extract(result[key])) for group, result in accepted]
            missing_groups = required_groups - {group for group, _ in accepted}
            values.extend(
                (group, np.full(len(extract(baseline)), np.nan))
                for group in missing_groups
            )
            combined[key][observable] = combine_variations(
                extract(baseline), values, config["systematics"]["method"]
            )
            source = (
                baseline["correlations"]["signal"]
                if observable == "raw"
                else baseline["corrected"]
            )
            statistical = np.asarray(source["covariance"], float)
            systematic = combined[key][observable]["covariance"]
            total = statistical + systematic
            relative = relative_uncertainty(
                source, np.sqrt(np.diag(systematic)), combined[key][observable]["valid"]
            )
            covariance_arrays[f"{key}__{observable}__systematic"] = systematic
            covariance_arrays[f"{key}__{observable}__total"] = total
            edges = config["analysis"]["kstar_edges"]
            for index, (low, high) in enumerate(zip(edges, edges[1:])):
                table.append(
                    {
                        "bin": key,
                        "observable": observable,
                        "kstar_low": low,
                        "kstar_high": high,
                        "value": extract(baseline)[index],
                        "stat_error": np.sqrt(statistical[index, index]),
                        "syst_error": np.sqrt(systematic[index, index]),
                        "total_error": np.sqrt(total[index, index]),
                        "relative_syst_error": relative["values"][index],
                        "valid": bool(
                            source["valid"][index]
                            and combined[key][observable]["valid"][index]
                        ),
                    }
                )
    write_json(
        directory / "summary.json",
        {
            "variations": outcomes,
            "execution": execution or {"data_scans": 0},
            "bins": combined,
            "qa": {
                "directory": "qa",
                "index": "qa/index.json",
                "root_file": "qa/systematics.root",
                "standard_definition": "nominal C(k*)",
            },
            "note": "Variation differences contain statistical fluctuations; no Barlow filtering or debiasing is applied. Different named groups are assumed independent.",
        },
    )
    pd.DataFrame(table).to_csv(directory / "final_observables.csv", index=False)
    np.savez_compressed(directory / "covariances.npz", **covariance_arrays)
    qa = build_systematic_qa(nominal, qa_variations, combined, config)
    render_systematic_qa(directory, nominal, qa, config)
    return combined
