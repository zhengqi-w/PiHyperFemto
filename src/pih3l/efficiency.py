"""HIPE4ML score-efficiency mapping on fixed held-out raw XGBoost margins."""

from __future__ import annotations

import importlib.metadata
import hashlib
from pathlib import Path

import numpy as np

from .errors import AnalysisError


def thresholds_from_efficiencies(
    labels, scores, efficiencies, *, return_metadata=False
):
    """Use the same HIPE4ML spline mapper as the spectrum reference.

    Both validation classes determine HIPE4ML's score grid. Scores are raw
    margins without sigmoid or scaling. Target efficiency one uses the finite
    held-out lower endpoint; unrestricted storage is a separate null boundary.
    No empirical quantile is substituted when HIPE4ML cannot construct a map.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    efficiencies = np.asarray(efficiencies, dtype=float)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or not len(scores)
        or not np.isin(labels, [0, 1]).all()
        or not (labels == 1).any()
        or not np.isfinite(scores).all()
    ):
        raise AnalysisError("HIPE4ML requires finite binary held-out raw margin scores")
    if (
        efficiencies.ndim != 1
        or not np.isfinite(efficiencies).all()
        or np.any((efficiencies <= 0) | (efficiencies > 1))
    ):
        raise AnalysisError("HIPE4ML efficiency targets must lie in (0,1]")
    result = np.full(len(efficiencies), scores.min(), dtype=float)
    evaluated = efficiencies.copy()
    details = {"numerical_retry": False, "evaluated_efficiencies": evaluated}
    selected = efficiencies != 1
    if not selected.any():
        return (result, details) if return_metadata else result
    try:
        from hipe4ml import analysis_utils
    except ImportError as error:
        raise AnalysisError(
            "HIPE4ML is required for the spectrum score-efficiency mapping; "
            "install the project's declared dependencies in the analysis environment"
        ) from error
    try:
        try:
            values = analysis_utils.score_from_efficiency_array(
                labels, scores, efficiencies[selected], keep_lower=False
            )
        except IndexError:
            # FITPACK may omit every root when the requested decimal exactly
            # matches an efficiency plateau. Move only one float64 ULP toward
            # zero, then call HIPE4ML again on the same fixed sample. This is
            # numerical root resolution, not a quantile or a new training split.
            from scipy.interpolate import InterpolatedUnivariateSpline

            curve, grid = analysis_utils.bdt_efficiency_array(
                labels, scores, n_points=1000, keep_lower=False
            )
            failed = np.array(
                [
                    (
                        not len(
                            InterpolatedUnivariateSpline(grid, curve - target).roots()
                        )
                        if target != 1
                        else False
                    )
                    for target in efficiencies
                ]
            )
            evaluated[failed] = np.nextafter(efficiencies[failed], 0.0)
            details["numerical_retry"] = True
            values = analysis_utils.score_from_efficiency_array(
                labels, scores, evaluated[selected], keep_lower=False
            )
        mapped = np.asarray(values, dtype=float)
    except (ValueError, IndexError, TypeError) as error:
        raise AnalysisError(
            "HIPE4ML score-efficiency interpolation failed on the fixed held-out "
            "sample; inspect score ties/range and requested efficiencies. "
            "No quantile fallback or automatic retraining was applied."
        ) from error
    if mapped.shape != efficiencies[selected].shape or not np.isfinite(mapped).all():
        raise AnalysisError("HIPE4ML returned an invalid raw margin threshold array")
    result[selected] = mapped
    order = np.argsort(efficiencies)
    if np.any(np.diff(result[order]) > 1e-8):
        raise AnalysisError("HIPE4ML score cuts are not monotonic in target efficiency")
    return (result, details) if return_metadata else result


def efficiency_threshold(signal_scores, efficiency):
    """Compatibility mapping when only signal scores are supplied by a caller.

    Production WP always supplies both validation classes. The two non-signal
    endpoints make a signal-only diagnostic's empirical margin range explicit.
    """
    scores = np.asarray(signal_scores, dtype=float)
    return float(
        thresholds_from_efficiencies(
            np.r_[np.ones(len(scores), dtype=np.int8), 0, 0],
            np.r_[scores, scores.min(), scores.max()],
            [efficiency],
        )[0]
    )


def mapping_metadata():
    from hipe4ml import analysis_utils

    return {
        "provider": "hipe4ml.analysis_utils.score_from_efficiency_array",
        "version": importlib.metadata.version("hipe4ml"),
        "implementation_sha256": hashlib.sha256(
            Path(analysis_utils.__file__).read_bytes()
        ).hexdigest(),
        "score_domain": "raw_margin; ModelHandler.predict(output_margin=True), no sigmoid/scaling",
        "input": "both classes of the fixed held-out validation sample",
        "library_comparison": "score > threshold",
        "application_comparison": "score >= threshold (inclusive ties)",
        "unit_efficiency_policy": "inclusive minimum of fixed held-out scores; storage reserve=1 is unbounded and does not prune any finite in-scope application score",
        "fallback": "none",
        "numerical_retry_policy": "On a missing spline root only, evaluate targets one float64 ULP toward zero on the identical sample; retain requested efficiencies and report actual acceptance.",
    }
