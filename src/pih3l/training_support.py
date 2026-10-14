"""Frozen MC-derived feature ranges with an independent calibration partition."""

from hashlib import sha256

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .errors import AnalysisError
from .schema import require_columns


def mc_range_mask(frame, ranges, *, return_cutflow=False):
    """Apply inclusive learned bounds without changing surviving values."""
    require_columns(frame, ranges, "MC-derived model feature range")
    accepted = np.ones(len(frame), dtype=bool)
    cutflow = {}
    for name, bounds in ranges.items():
        limits = np.asarray(bounds, float)
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or limits[0] > limits[1]
        ):
            raise AnalysisError(f"Invalid learned MC range for {name}: {bounds}")
        values = frame[name].to_numpy(float)
        within = np.isfinite(values) & (values >= limits[0]) & (values <= limits[1])
        before = int(accepted.sum())
        accepted &= within
        cutflow[name] = {
            "input_rows": before,
            "retained_rows": int(accepted.sum()),
            "rejected_rows": before - int(accepted.sum()),
            "independent_rejected_rows": int((~within).sum()),
        }
    return (accepted, cutflow) if return_cutflow else accepted


def prepare_mc_reference(signal, config):
    """Reserve complete MC groups; learn once, then select the training pool.

    The reference groups never enter classifier training, held-out efficiency
    mapping, or Optuna. The bounded priority sample defines this calibration,
    not the full-period MC population.
    """
    settings = config["training"]["mc_range_selection"]
    if not settings["enabled"]:
        return signal, {"enabled": False, "ranges": {}}
    from .training import training_groups

    features = list(settings["features"] or config["training"]["features"])
    if not features or not set(features) <= set(config["training"]["features"]):
        raise AnalysisError(
            "MC range features must be a nonempty subset of BDT features"
        )
    require_columns(signal, features, "MC range calibration")
    groups = training_groups(signal)
    if len(np.unique(groups)) < 5:
        raise AnalysisError(
            "MC range calibration requires at least five independent groups"
        )
    seed = config["seed"] + 107
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=settings["reference_fraction"], random_state=seed
    )
    pool_index, reference_index = next(splitter.split(signal, groups=groups))
    reference = signal.iloc[reference_index]
    if len(reference) < settings["min_reference_rows"]:
        raise AnalysisError(
            f"MC range reference has {len(reference)} rows; need {settings['min_reference_rows']}"
        )
    reference_groups = np.unique(groups[reference_index])
    pool_groups = np.unique(groups[pool_index])
    if np.intersect1d(reference_groups, pool_groups).size:
        raise AnalysisError("MC range reference overlaps the training pool")
    ranges, quantile_values, skipped = {}, {}, {}
    for name in features:
        values = reference[name].to_numpy(float)
        if not np.isfinite(values).all():
            raise AnalysisError(f"Nonfinite MC range reference feature: {name}")
        low, high = np.quantile(values, settings["quantiles"], method="linear")
        quantile_values[name] = [float(low), float(high)]
        span = high - low
        if span <= 0:
            skipped[name] = (
                "zero quantile span: no range inferred from a constant reference"
            )
            continue
        margin = settings["margin_fraction"] * span
        ranges[name] = [float(low - margin), float(high + margin)]
    if not ranges:
        raise AnalysisError("No nonconstant MC feature ranges can be learned")
    pool = signal.iloc[pool_index]
    accepted, per_variable = mc_range_mask(pool, ranges, return_cutflow=True)
    selected = pool.loc[accepted].copy().reset_index(drop=True)
    if len(selected) < config["training"]["min_per_class"]:
        raise AnalysisError(
            "Too few MC signal candidates after reference removal and range selection"
        )
    if len(np.unique(training_groups(selected))) < 4:
        raise AnalysisError("Too few MC signal groups after learned range selection")
    metadata = {
        "enabled": True,
        "features": features,
        "ranges": ranges,
        "quantiles": settings["quantiles"],
        "quantile_method": "numpy.quantile(method='linear')",
        "reference_quantile_values": quantile_values,
        "margin_fraction": settings["margin_fraction"],
        "bounds": "inclusive at both ends; rows removed, surviving values unchanged",
        "skipped_features": skipped,
        "scope": "training, working_point, Femto/MCFemto inference",
        "efficiency_scope": "conditional on the same frozen MC feature ranges applied to all domains",
        "shape_policy": "range selection only; no distribution matching, reweighting or Winsorization",
        "reference": {
            "source": "bounded selected finite-feature reconstructed MC signal sample",
            "sample_rows": len(signal),
            "rows": len(reference),
            "groups": len(reference_groups),
            "requested_group_fraction": settings["reference_fraction"],
            "random_state": seed,
            "group_hash_sha256": sha256(
                reference_groups.astype("<u8").tobytes()
            ).hexdigest(),
            "group_definition": "source file/tree and PV surrogate; not a producer collision ID",
            "overlap_with_training_pool": 0,
            "used_for_model_or_efficiency_mapping": False,
        },
        "signal_counts": {
            "sampled_before_reference": len(signal),
            "reference_removed": len(reference),
            "after_reference": len(pool),
            "after_mc_range": len(selected),
            "per_variable_mc_range": per_variable,
        },
        "limitations": [
            "Finite reference quantiles do not guarantee a full-population coverage fraction.",
            "A rectangular feature range does not enforce a joint-density or shape match.",
            "The MC-derived preselection acceptance needs independent MC for a final physical efficiency.",
        ],
    }
    return selected, metadata
