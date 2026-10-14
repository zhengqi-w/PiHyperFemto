"""Auditable cuts shared by nominal, variations, and reconstructed MC."""

from __future__ import annotations

import numpy as np

from .errors import AnalysisError
from .schema import require_columns


def in_window(values, bounds):
    return (values >= bounds[0]) & (values < bounds[1])


def in_windows(values, windows):
    mask = np.zeros(len(values), dtype=bool)
    for bounds in windows:
        mask |= np.asarray(in_window(values, bounds))
    return mask


def cut_mask(frame, cuts: dict):
    mask = np.ones(len(frame), dtype=bool)
    for column, specification in cuts.items():
        require_columns(frame, [column], "selection")
        if isinstance(specification, (list, tuple)):
            specification = {"min": specification[0], "max": specification[1]}
        if not isinstance(specification, dict) or set(specification) - {
            "min",
            "max",
            "abs_max",
            "equals",
        }:
            raise AnalysisError(f"Invalid cut for {column}: use min/max/abs_max/equals")
        values = frame[column].to_numpy()
        mask &= np.isfinite(values)
        if specification.get("min") is not None:
            mask &= values >= specification["min"]
        if specification.get("max") is not None:
            mask &= values < specification["max"]
        if specification.get("abs_max") is not None:
            mask &= abs(values) < specification["abs_max"]
        if "equals" in specification:
            mask &= values == specification["equals"]
    return mask


def threshold_values(frame, threshold, *, allow_unbounded=False):
    """Route thresholds by the model assigned at inference; never invent a fallback."""
    if isinstance(threshold, dict):
        require_columns(frame, ["bdt_model_id"], "per-model BDT threshold")
        resolved = {
            key: (-np.inf if value is None and allow_unbounded else value)
            for key, value in threshold.items()
        }
        values = frame.bdt_model_id.map(resolved).to_numpy(dtype=float)
        scoped = (
            frame.bdt_in_scope.to_numpy(bool)
            if "bdt_in_scope" in frame
            else np.isfinite(frame.bdt_score)
        )
        available = np.isfinite(values) | (allow_unbounded & (values == -np.inf))
        missing = scoped & ~available
        if missing.any():
            models = sorted(set(frame.loc[missing, "bdt_model_id"]))
            raise AnalysisError(f"Missing working point for models: {models}")
        return values
    if threshold is None and allow_unbounded:
        return np.full(len(frame), -np.inf)
    return np.full(len(frame), float(threshold))


def select_pairs(frame, analysis: dict, threshold):
    mask = np.ones(len(frame), dtype=bool)
    flow = {"input": len(frame)}
    for name in ("event_cuts", "candidate_cuts", "pair_cuts"):
        mask &= cut_mask(frame, analysis[name])
        flow[name] = int(mask.sum())
    bits = analysis["close_pair_reject_bits"]
    if bits:
        require_columns(frame, ["IsClosePairRejected"], "close-pair rejection")
        mask &= (frame.IsClosePairRejected.to_numpy(dtype=np.uint8) & bits) == 0
    flow["close_pair"] = int(mask.sum())
    require_columns(frame, ["bdt_score"], "BDT selection")
    if "bdt_in_scope" in frame:
        mask &= frame.bdt_in_scope.to_numpy(bool)
    mask &= np.isfinite(frame.bdt_score) & (
        frame.bdt_score.to_numpy() >= threshold_values(frame, threshold)
    )
    flow["bdt"] = int(mask.sum())
    return np.asarray(mask), flow


def pair_weights(frame, mode: str):
    if mode not in {"unit", "inverse_depth"}:
        raise AnalysisError("mixing_weight must be unit or inverse_depth")
    weights = np.ones(len(frame), dtype=float)
    if mode == "inverse_depth":
        require_columns(frame, ["MixingDepth"], "inverse mixing-depth weighting")
        mixed = frame.IsMixed.to_numpy(dtype=bool)
        depth = frame.MixingDepth.to_numpy()
        if np.any(~np.isfinite(depth[mixed])) or np.any(depth[mixed] <= 0):
            raise AnalysisError(
                "Mixed rows must have finite positive MixingDepth for inverse_depth weighting"
            )
        weights[mixed] = 1.0 / depth[mixed]
    return weights
