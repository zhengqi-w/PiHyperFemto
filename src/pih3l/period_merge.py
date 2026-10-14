"""Merge period statistics before estimating correlations, fractions or errors.

Only explicitly additive quantities are combined. Model scores, fitted yields,
purities, normalized correlations and their covariance matrices are never
averaged or added. Input periods must already use compatible analysis grids.
"""

from __future__ import annotations

import copy
from numbers import Real
from pathlib import Path

import numpy as np

from .analysis import evaluate_histograms, export_results
from .correlation import empty_histograms
from .errors import AnalysisError
from .io import json_safe, read_json, write_json
from .mc import wilson_interval


def _periods(period_results):
    records = list(period_results)
    if not records:
        raise AnalysisError("Period merge requires at least one period")
    names = [name for name, _ in records]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise AnalysisError("Period identifiers must be nonempty strings")
    if len(names) != len(set(names)):
        raise AnalysisError("Period merge received duplicate period identifiers")
    return records


def _same(actual, expected, context):
    if json_safe(actual) != json_safe(expected):
        raise AnalysisError(f"Incompatible period metadata: {context}")


def _keys(actual, expected, context):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise AnalysisError(f"Incompatible period fields: {context}")


def _array(value, shape, context, *, signed=False):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"Invalid period statistic: {context}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise AnalysisError(f"Invalid shape or nonfinite period statistic: {context}")
    if not signed and np.any(array < 0):
        raise AnalysisError(f"Negative period count or variance: {context}")
    return array


def _add(target, value, context, *, signed=False):
    array = _array(value, target.shape, context, signed=signed)
    with np.errstate(over="ignore", invalid="ignore"):
        target += array
    if not np.isfinite(target).all():
        raise AnalysisError(f"Period statistic overflow: {context}")


def _count(value, context):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or value != int(value)
    ):
        raise AnalysisError(f"Invalid nonnegative integer counter: {context}")
    return int(value)


def _sum_counts(records, *, context, identical_keys=False):
    result = {}
    if identical_keys:
        for name, counts in records:
            _keys(counts, records[0][1], f"{name}/{context}")
    for name, counts in records:
        if not isinstance(counts, dict):
            raise AnalysisError(f"Invalid period counters: {name}/{context}")
        for key, count in counts.items():
            result[key] = result.get(key, 0) + _count(count, f"{name}/{context}/{key}")
    return result


def merge_histograms(period_results, config):
    """Sum compatible SE/ME and mass sufficient statistics, without any fit.

    Each input may be either accumulator histograms or evaluated result bins.
    Derived fields in evaluated results are deliberately ignored. Existing
    histogram artifacts do not store k* edges; their dimensions are checked
    against the common configuration and explicit edges, when present, agree.
    """
    records = _periods(period_results)
    merged = empty_histograms(config)
    for period, results in records:
        _keys(results, merged, f"{period}/correlation bins")
        for key, target in merged.items():
            source = results[key]
            for field in ("centrality", "channel", "channel_label", "mass_edges"):
                _same(source.get(field), target[field], f"{period}/{key}/{field}")
            if "kstar_edges" in source:
                _same(
                    source["kstar_edges"],
                    config["analysis"]["kstar_edges"],
                    f"{period}/{key}/kstar_edges",
                )
            regions = source.get("regions")
            _keys(regions, target["regions"], f"{period}/{key}/regions")
            for region, sums in target["regions"].items():
                _keys(regions[region], sums, f"{period}/{key}/{region}")
                for field, values in sums.items():
                    _add(
                        values,
                        regions[region][field],
                        f"{period}/{key}/{region}/{field}",
                    )
            for field in ("mass_counts", "mass_vs_kstar"):
                _add(target[field], source.get(field), f"{period}/{key}/{field}")
            for field in ("selected_rows", "selected_se", "selected_me"):
                target[field] += _count(source.get(field), f"{period}/{key}/{field}")
            if source["selected_rows"] != source["selected_se"] + source["selected_me"]:
                raise AnalysisError(f"Inconsistent selected row counts: {period}/{key}")
    for target in merged.values():
        target["periods"] = [name for name, _ in records]
    return merged


def merge_raw_qa(period_results):
    """Sum raw distributions and flow counters with identical definitions."""
    records = _periods(period_results)
    first = records[0][1]
    merged = copy.deepcopy(first)
    metadata = set(first) - {"inputs", "femto", "hyper", "periods"}
    for period, result in records:
        _same(
            set(result) - {"inputs", "femto", "hyper", "periods"},
            metadata,
            f"{period}/raw QA metadata fields",
        )
        for field in metadata:
            _same(result[field], first[field], f"{period}/raw QA/{field}")
        _keys(result.get("inputs"), first["inputs"], f"{period}/raw QA inputs")
    merged["inputs"] = {
        role: _sum_counts(
            [(name, result["inputs"][role]) for name, result in records],
            context=f"raw QA inputs/{role}",
            identical_keys=True,
        )
        for role in first["inputs"]
    }
    counters = ("rows", "nonfinite", "underflow", "overflow")
    for role, distributions in (
        ("femto", ("kstar_se", "kstar_me")),
        ("hyper", ("mass",)),
    ):
        for period, result in records:
            _keys(result.get(role), first[role], f"{period}/raw QA/{role}")
        for key, template in first[role].items():
            item = merged[role][key]
            for period, result in records:
                source = result[role][key]
                _keys(source, template, f"{period}/raw QA/{role}/{key}")
                for field in set(template) - set(distributions):
                    _same(
                        source[field], template[field], f"{period}/{role}/{key}/{field}"
                    )
            for field in distributions:
                reference = template[field]
                edges = np.asarray(reference["edges"], float)
                if (
                    edges.ndim != 1
                    or len(edges) < 2
                    or not np.isfinite(edges).all()
                    or np.any(np.diff(edges) <= 0)
                ):
                    raise AnalysisError(f"Invalid raw QA edges: {role}/{key}/{field}")
                target = item[field]
                target["counts"] = np.zeros(len(edges) - 1)
                for counter in counters:
                    target[counter] = 0
                for period, result in records:
                    source = result[role][key][field]
                    _keys(source, reference, f"{period}/{role}/{key}/{field}")
                    for name in set(reference) - {"counts", *counters}:
                        _same(
                            source[name],
                            reference[name],
                            f"{period}/{role}/{key}/{field}/{name}",
                        )
                    _add(
                        target["counts"],
                        source["counts"],
                        f"{period}/{role}/{key}/{field}",
                    )
                    for counter in counters:
                        target[counter] += _count(
                            source[counter], f"{period}/{role}/{key}/{field}/{counter}"
                        )
                    accounted = sum(
                        _count(source[c], f"{period}/{field}/{c}") for c in counters[1:]
                    ) + np.sum(source["counts"])
                    if accounted != source["rows"]:
                        raise AnalysisError(
                            f"Raw QA counts do not account for rows: {period}/{role}/{key}/{field}"
                        )
    merged["periods"] = [name for name, _ in records]
    return merged


def merge_mixing_qa(period_results):
    """Combine common-bin mixing QA, preserving missing-data flags."""
    records = _periods(period_results)
    first = records[0][1]
    merged = copy.deepcopy(first)
    histogram_fields = {
        f"{prefix}_{event}": f"{prefix}_edges"
        for prefix in ("delta_centrality", "occupancy")
        for event in ("se", "me")
    }
    additive = {*histogram_fields, "close_pair_flags", "periods"}
    missing = {key for key in first if key.startswith("missing_")}
    for period, result in records:
        _same(
            set(result) - {"periods"},
            set(first) - {"periods"},
            f"{period}/mixing QA fields",
        )
        for field in set(first) - additive - missing:
            _same(result[field], first[field], f"{period}/mixing QA/{field}")
    for field, edge_field in histogram_fields.items():
        edges = np.asarray(first[edge_field], float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise AnalysisError(f"Invalid mixing QA edges: {edge_field}")
        merged[field] = np.zeros(len(edges) - 1)
        for period, result in records:
            _add(merged[field], result[field], f"{period}/mixing QA/{field}")
    for field in missing:
        if any(
            not isinstance(result[field], (bool, np.bool_)) for _, result in records
        ):
            raise AnalysisError(f"Invalid mixing QA missing-data flag: {field}")
        merged[field] = any(result[field] for _, result in records)
    flags = []
    for period, result in records:
        counts = {}
        for flag, count in result["close_pair_flags"].items():
            try:
                key = str(int(flag))
            except (TypeError, ValueError) as error:
                raise AnalysisError(
                    f"Invalid close-pair flag: {period}/{flag}"
                ) from error
            counts[key] = counts.get(key, 0) + _count(count, f"{period}/flag/{flag}")
        flags.append((period, counts))
    merged["close_pair_flags"] = _sum_counts(flags, context="close_pair_flags")
    merged["periods"] = [name for name, _ in records]
    return merged


def merge_cutflows(period_results):
    """Add sequential cut counters while retaining each period's own WP map."""
    records = _periods(period_results)
    for period, result in records:
        if not isinstance(result, dict) or not {"threshold", "counts"} <= set(result):
            raise AnalysisError(f"Missing period cutflow/threshold: {period}")
    return {
        "threshold": {
            name: copy.deepcopy(result["threshold"]) for name, result in records
        },
        "threshold_scope": "period_id -> local model threshold or per-model mapping",
        "counts": _sum_counts(
            [(name, result["counts"]) for name, result in records], context="cutflow"
        ),
        "periods": [name for name, _ in records],
    }


_MC_NUMERATORS = (
    "primary",
    "true_pion",
    "primary_pion",
    "signal",
    "self",
    "same_collision",
    "true_primary_pair",
    "matches_hyp_collision",
    "matches_pair_collision",
)
_MC_ARRAYS = (
    "total",
    *_MC_NUMERATORS,
    "resolution",
    "residual_sum",
    "residual_sum2",
    "residual_count",
)


def merge_mc_qa(period_outputs, directory, config):
    """Merge reconstructed MC sufficient statistics and recompute their QA.

    Optional periods without MCFemto remain explicit in coverage. A truth
    fraction is unavailable if its truth column was missing in any period with
    MC observations; zero-filled placeholder numerators are never interpreted
    as measured zero fractions. No candidate training MC is substituted.
    """
    outputs = _periods(period_outputs)
    records = []
    for period, output in outputs:
        path = Path(output) / "mc_qa/summary.json"
        if not path.is_file():
            raise AnalysisError(f"Missing MC stage summary: {period}: {path}")
        records.append((period, read_json(path)))
    available = [
        (name, result)
        for name, result in records
        if result.get("status") == "available"
    ]
    unavailable = [
        (name, result)
        for name, result in records
        if result.get("status") != "available"
    ]
    summary = {
        "status": "available" if available else "skipped_missing_mc_femto",
        "expected_periods": [name for name, _ in records],
        "available_periods": [name for name, _ in available],
        "missing_periods": [name for name, _ in unavailable],
        "complete_period_coverage": not unavailable,
        "period_statuses": {
            name: result.get("status", "unknown") for name, result in records
        },
        "period_reasons": {name: result.get("reason") for name, result in unavailable},
        "coverage_note": "Merged MC describes only the listed available periods; it is not automatically a full-data dilution correction.",
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not available:
        summary.update(
            reason="No available reconstructed MC pair sample in any period; fractions, resolution and closure are unavailable.",
            generated_training_signal_is_not_a_substitute=True,
            truth_closure_status="unavailable",
            reco_closure_status="unavailable",
        )
        write_json(directory / "summary.json", summary)
        return summary
    edges = np.asarray(config["analysis"]["kstar_edges"], float)
    n = len(edges) - 1
    keys = {
        f"{key}__{event}" for key in empty_histograms(config) for event in ("se", "me")
    }
    first = available[0][1]
    for period, result in available:
        _same(result.get("kstar_edges"), edges, f"{period}/MC kstar_edges")
        _same(
            result.get("fraction_mass_window"),
            config["analysis"]["signal_window"],
            f"{period}/MC mass window",
        )
        _same(
            result.get("fraction_weighting"),
            first.get("fraction_weighting"),
            f"{period}/MC fraction weighting",
        )
        _keys(result.get("bins"), keys, f"{period}/MC bins")
    summary.update(
        kstar_edges=edges,
        fraction_mass_window=copy.deepcopy(config["analysis"]["signal_window"]),
        fraction_weighting=first["fraction_weighting"],
        counters=_sum_counts(
            [(name, result["counters"]) for name, result in available],
            context="MC counters",
            identical_keys=True,
        ),
        limitations=list(
            dict.fromkeys(
                note
                for _, result in available
                for note in result.get("limitations", [])
            )
        ),
        bins={},
    )
    for key in sorted(keys):
        item = {
            field: np.zeros((n, n) if field == "resolution" else n)
            for field in _MC_ARRAYS
        }
        for period, result in available:
            source = result["bins"][key]
            for field in _MC_ARRAYS:
                _add(
                    item[field],
                    source.get(field),
                    f"{period}/MC/{key}/{field}",
                    signed=field == "residual_sum",
                )
            for numerator in _MC_NUMERATORS:
                if np.any(np.asarray(source[numerator]) > np.asarray(source["total"])):
                    raise AnalysisError(
                        f"MC numerator exceeds its denominator: {period}/{key}/{numerator}"
                    )
        for numerator in _MC_NUMERATORS:
            missing_truth = [
                name
                for name, result in available
                if "value" not in result["bins"][key].get(numerator + "_fraction", {})
            ]
            if missing_truth:
                item[numerator + "_fraction"] = {
                    "status": "missing_truth_columns",
                    "missing_periods": missing_truth,
                    "reason": "A required truth flag is unavailable in at least one included MC period.",
                }
            else:
                item[numerator + "_fraction"] = wilson_interval(
                    item[numerator], item["total"]
                )
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = item["residual_sum"] / item["residual_count"]
            item["resolution_bias"] = mean
            item["resolution_rms"] = np.sqrt(
                np.maximum(item["residual_sum2"] / item["residual_count"] - mean**2, 0)
            )
        summary["bins"][key] = item
    paths = dict(outputs)
    for closure in ("reco_closure", "truth_closure"):
        included = []
        missing = []
        for period, result in available:
            if (
                closure == "truth_closure"
                and result.get("truth_closure_status") != "available"
            ):
                missing.append(period)
                continue
            path = Path(paths[period]) / "mc_qa" / closure / "results.json"
            if not path.is_file():
                raise AnalysisError(
                    f"Missing declared MC closure artifact: {period}: {path}"
                )
            included.append((period, read_json(path)))
        summary[closure + "_status"] = (
            "available" if not missing else "missing_truth_columns"
        )
        summary[closure + "_coverage"] = {
            "available_periods": [period for period, _ in included],
            "missing_periods": [*summary["missing_periods"], *missing],
            "complete_period_coverage": not unavailable and not missing,
        }
        if not missing:
            export_results(
                evaluate_histograms(merge_histograms(included, config), config),
                directory / closure,
                config,
            )
    write_json(directory / "summary.json", summary)
    return summary
