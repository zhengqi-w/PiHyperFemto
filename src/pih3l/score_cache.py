"""Per-model inference checkpoints: a local retrain reuses other model scores."""

from pathlib import Path
import os

import numpy as np

from .config import digest
from .errors import AnalysisError
from .io import read_json, sha256_file, write_json
from .selection import in_window


def score_key(source, spec):
    return digest(
        {
            "model_format": "hipe4ml_model_handler_pickle_v1",
            "score_domain": "raw_margin",
            "source": source,
            "model": spec["inference_checksum"],
            "scope": spec["pt_range"],
            "mc_ranges": spec.get("bundle", {})
            .get("mc_range_selection", {})
            .get("ranges", {}),
            "code": spec["inference_code"],
            "environment": spec.get("inference_environment", {}),
        }
    )


def cached_model_scores(
    frame, registry, *, checkpoint_root, source, active_paths, force=False
):
    """Cache sparse positional score vectors, never a duplicate candidate table."""
    from .models import score_models
    from .training_support import mc_range_mask

    result = frame.copy()
    result["bdt_score"] = np.nan
    result["bdt_model_id"] = ""
    result["bdt_in_scope"] = False
    assigned = np.zeros(len(frame), dtype=bool)
    for spec in registry["models"]:
        scope = (
            np.ones(len(frame), dtype=bool)
            if spec["pt_range"] is None
            else np.asarray(in_window(frame.pt_h3l, spec["pt_range"]))
        )
        scope &= mc_range_mask(
            frame,
            spec.get("bundle", {}).get("mc_range_selection", {}).get("ranges", {}),
        )
        if (scope & assigned).any():
            raise AnalysisError("Overlapping model application scopes")
        assigned |= scope
        positions = np.flatnonzero(scope)
        if not len(positions):
            continue
        key = score_key(source, spec)
        directory = Path(checkpoint_root) / key
        artifact, manifest = (
            directory / "scores.npz",
            directory / "score_manifest.json",
        )
        active_paths.add(directory)
        values = None
        if not force and manifest.exists() and artifact.exists():
            try:
                saved = read_json(manifest)
                if saved.get("sha256") == sha256_file(artifact):
                    with np.load(artifact, allow_pickle=False) as cached:
                        if (
                            np.array_equal(cached["position"], positions)
                            and cached["score"].shape == positions.shape
                        ):
                            values = cached["score"].copy()
            except (ValueError, OSError, KeyError):
                pass  # A corrupt optional inference cache is recomputed.
        if values is None:
            scored = score_models(frame.iloc[positions], {"models": [spec]})
            values = scored.bdt_score.to_numpy()
            directory.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, position=positions, score=values)
            os.replace(temporary, artifact)
            write_json(
                manifest,
                {
                    "sha256": sha256_file(artifact),
                    "model_id": spec["id"],
                    "source": source,
                },
            )
        result.iloc[positions, result.columns.get_loc("bdt_score")] = values
        result.iloc[positions, result.columns.get_loc("bdt_model_id")] = spec["id"]
        result.iloc[positions, result.columns.get_loc("bdt_in_scope")] = True
    return result
