"""Persist only held-out labels and scores required by ROOT QA and WP mapping."""

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import AnalysisError


def write_validation_scores(directory, validation):
    np.savez_compressed(
        Path(directory) / "validation_scores.npz",
        label=validation.label.to_numpy(dtype=np.int8),
        bdt_score=validation.bdt_score.to_numpy(dtype=np.float64),
        score_domain=np.asarray("raw_margin"),
    )


def read_validation_scores(directory):
    path = Path(directory) / "validation_scores.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if (
                "score_domain" not in archive
                or str(archive["score_domain"].item()) != "raw_margin"
            ):
                raise AnalysisError(
                    f"Incompatible held-out score domain: {path}; retrain using raw margins"
                )
            labels, scores = archive["label"].copy(), archive["bdt_score"].copy()
        if (
            labels.ndim != 1
            or scores.shape != labels.shape
            or not np.isin(labels, [0, 1]).all()
        ):
            raise AnalysisError(f"Invalid held-out score archive: {path}")
        return pd.DataFrame({"label": labels, "bdt_score": scores})
    raise AnalysisError(
        f"Missing raw-margin validation_scores.npz in {directory}; legacy validation tables require retraining"
    )
