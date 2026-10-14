"""Compact pair tables with an explicit, inclusive systematic score envelope."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import AnalysisError
from .execution import scored_columns
from .io import parquet_columns, read_json, write_batch, write_json
from .models import model_specs
from .schema import require_columns
from .selection import threshold_values

METADATA = "dataset.json"


def require_role_dataset(metadata):
    """A period catalog cannot be evaluated with one model's score threshold."""
    if metadata.get("kind") == "period_reference_catalog":
        raise AnalysisError(
            "This is a multi-period scored catalog. Select a period's scored/femto "
            "or scored/mc_femto dataset; use MultiPeriodPipeline to aggregate periods."
        )


def _floor(config, working_point):
    if not config["working_point"]["storage"]["enabled"]:
        return None
    if "storage_threshold" not in working_point:
        raise AnalysisError(
            "Scored storage requires a WP storage_threshold; run working-point first"
        )
    return working_point["storage_threshold"]


def write_scored_batch(directory, index, frame, config, working_point, role):
    """Prune only scores/out-of-scope rows; keep all future selection variables."""
    directory = Path(directory)
    require_columns(frame, ["bdt_score"], "compact scored storage")
    scores = frame.bdt_score.to_numpy()
    finite = np.isfinite(scores)
    scope = (
        frame.bdt_in_scope.to_numpy(bool)
        if "bdt_in_scope" in frame
        else np.ones(len(frame), dtype=bool)
    )
    floor = _floor(config, working_point)
    values = threshold_values(frame, floor, allow_unbounded=True)
    above = scores >= values
    keep = finite & scope & above
    counts = {
        "rows": len(frame),
        "retained_rows": int(keep.sum()),
        "invalid_scores": int((~finite).sum()),
        "out_of_scope_rows": int((~scope).sum()),
        "dropped_below_storage": int((finite & scope & ~above).sum()),
    }
    columns = (
        scored_columns(config, role, frame.columns)
        if config["io"].get("slim_scored", True)
        else list(frame.columns)
    )
    retained = frame.loc[keep, columns]
    paths = []
    if config["training"]["pt_bins"] and config["io"].get(
        "scored_partition_by_pt", True
    ):
        for spec in model_specs(config):
            sample = retained.loc[retained.bdt_model_id == spec["id"]]
            if len(sample):
                paths.append(write_batch(directory / spec["id"], index, sample))
    elif len(retained):
        paths.append(write_batch(directory, index, retained))
    return paths, counts


def write_storage_metadata(directory, config, working_point, *, counts=None):
    directory = Path(directory)
    columns = sorted(parquet_columns(directory))
    write_json(
        directory / METADATA,
        {
            "schema_version": 1,
            "layout": (
                "pt_model_subdirectories"
                if config["training"]["pt_bins"]
                and config["io"].get("scored_partition_by_pt", True)
                else "flat"
            ),
            "models": [
                {"id": spec["id"], "pt_range": spec["pt_range"]}
                for spec in model_specs(config)
            ],
            "minimum_threshold": _floor(config, working_point),
            "score_domain": "raw_margin",
            "unbounded_threshold_policy": "null means no score pruning; finite in-scope scores retained",
            "score_comparison": ">= (inclusive, retaining ties)",
            "storage": config["working_point"]["storage"],
            "storage_envelope": working_point.get("storage_envelope"),
            "columns": columns,
            "requested_columns": config["io"].get("scored_columns", []),
            "automatic_columns": "pair classification, score/model routing, all nominal and declared systematic cuts, and available MC truth QA",
            "counts": counts,
            "domain": {
                "centrality_bins": config["analysis"]["centrality_bins"],
                "mass_range": config["analysis"]["mass_range"],
                "kstar_range": [
                    config["analysis"]["kstar_edges"][0],
                    config["analysis"]["kstar_edges"][-1],
                ],
            },
            "limitations": [
                "Thresholds below minimum_threshold require re-scoring the original ROOT input.",
                "Storage pruning changes the cutflow input denominator, not accepted nominal/systematic rows.",
                "Counts for invalid_scores and out_of_scope_rows may overlap.",
            ],
        },
    )


def ensure_storage_coverage(directory, threshold, config=None):
    """Reject a looser cut/domain on pruned rows, including direct library use."""
    path = Path(directory) / METADATA
    if not path.exists():
        raise AnalysisError(
            "Missing scored dataset metadata: unknown score domain; re-run apply with raw margin models"
        )
    metadata = read_json(path)
    require_role_dataset(metadata)
    if metadata.get("score_domain") != "raw_margin":
        raise AnalysisError(
            "Scored dataset has an incompatible score domain; re-run apply with raw margin models"
        )
    floor = metadata["minimum_threshold"]
    floors = (
        floor
        if isinstance(floor, dict)
        else {spec["id"]: floor for spec in metadata["models"]}
    )
    for model_id, minimum in floors.items():
        requested = (
            threshold.get(model_id) if isinstance(threshold, dict) else threshold
        )
        if (
            requested is None
            or not np.isfinite(requested)
            or (minimum is not None and requested < minimum)
        ):
            raise AnalysisError(
                f"Scored storage for {model_id} starts at score {minimum}; requested {requested} is outside the saved envelope. Re-run apply from ROOT with a wider WP storage reservation."
            )
    if config is not None:
        domain = metadata["domain"]
        analysis = config["analysis"]
        for key, requested in (
            ("mass_range", analysis["mass_range"]),
            ("kstar_range", [analysis["kstar_edges"][0], analysis["kstar_edges"][-1]]),
        ):
            available = domain[key]
            if requested[0] < available[0] or requested[1] > available[1]:
                raise AnalysisError(
                    f"Requested {key} extends beyond saved scored domain; re-run apply from ROOT"
                )
        for lower, upper in analysis["centrality_bins"]:
            covered = lower
            for start, stop in domain["centrality_bins"]:
                if start <= covered < stop:
                    covered = max(covered, stop)
                if covered >= upper:
                    break
            if covered < upper:
                raise AnalysisError(
                    "Requested centrality bins extend beyond saved scored domain; re-run apply from ROOT"
                )
