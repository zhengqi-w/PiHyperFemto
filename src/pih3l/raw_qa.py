"""Input-level distributions, independent of preparation, training and WP."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .io import iter_root, write_json
from .physics import derive
from .schema import CANDIDATE_COLUMNS
from .selection import in_window

LOG = logging.getLogger(__name__)
INPUT_ROLES = ("femto", "background")


def empty_distribution(edges):
    return {
        "edges": np.asarray(edges, float),
        "counts": np.zeros(len(edges) - 1, dtype=np.int64),
        "rows": 0,
        "nonfinite": 0,
        "underflow": 0,
        "overflow": 0,
    }


def fill_distribution(distribution, values):
    """Record every row; histogram intervals are [low, high), including last bin."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    edges = distribution["edges"]
    distribution["rows"] += len(values)
    distribution["nonfinite"] += int((~finite).sum())
    distribution["underflow"] += int((finite & (values < edges[0])).sum())
    distribution["overflow"] += int((finite & (values >= edges[-1])).sum())
    inside = finite & (values >= edges[0]) & (values < edges[-1])
    distribution["counts"] += np.histogram(values[inside], edges)[0]


def collect_raw_qa(config, frames):
    """Aggregate derived raw frames by role, centrality and pair charge channel.

    Only configured centrality intervals select rows; analysis cuts, training
    sidebands, mass windows and BDT scores are deliberately not consulted.
    """
    qa = config["raw_qa"]
    mass_edges = np.linspace(*qa["mass_range"], qa["mass_bins"] + 1)
    centralities = config["analysis"]["centrality_bins"]
    pairs, candidates = {}, {}
    for low, high in centralities:
        cent = f"cent_{low:g}_{high:g}"
        candidates[cent] = {
            "centrality": [low, high],
            "mass": empty_distribution(mass_edges),
        }
        for channel in range(4):
            pairs[f"{cent}__ch_{channel}"] = {
                "centrality": [low, high],
                "channel": channel,
                **{
                    f"kstar_{event}": empty_distribution(qa["kstar_edges"])
                    for event in ("se", "me")
                },
            }
    inputs = {role: {"raw_rows": 0, "centrality_rows": 0} for role in INPUT_ROLES}
    for role, frame in frames:
        inputs[role]["raw_rows"] += len(frame)
        for low, high in centralities:
            sample = frame.loc[in_window(frame.HypCentFT0C, [low, high])]
            inputs[role]["centrality_rows"] += len(sample)
            cent = f"cent_{low:g}_{high:g}"
            if role == "background":
                fill_distribution(candidates[cent]["mass"], sample.mass_h3l)
                continue
            for channel in range(4):
                item = pairs[f"{cent}__ch_{channel}"]
                channel_sample = sample.loc[sample.ChargeChannel == channel]
                for event, mixed in (("se", False), ("me", True)):
                    subset = channel_sample.loc[channel_sample.IsMixed == mixed]
                    fill_distribution(item[f"kstar_{event}"], subset.kstar)
    return {
        "selection": "producer output + configured hypertriton-side centrality only",
        "weight": "unit row counts; SE and ME are not normalized to each other",
        "mass_definition": "reconstructed invariant mass of He3 + decay pion",
        "femto_basis": "pair rows; no candidate ID tracking",
        "hyper_mass_basis": "independent Hyper reconstruction candidate rows, both matter signs",
        "bdt_applied": False,
        "analysis_cuts_applied": False,
        "range_convention": "[low, high); excluded and nonfinite rows are recorded separately",
        "inputs": inputs,
        "femto": pairs,
        "hyper": candidates,
    }


def run_raw_qa(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def frames():
        for role in INPUT_ROLES:
            columns = CANDIDATE_COLUMNS + ["HypCentFT0C"]
            if role == "femto":
                columns += [
                    "ChargeChannel",
                    "IsMixed",
                    "PairPiTrackPt",
                    "PairPiTrackEta",
                    "PairPiTrackPhi",
                ]
            for index, frame in enumerate(iter_root(config, role, columns=columns)):
                if index % 10 == 0:
                    LOG.info("Raw QA: %s chunk %s", role, index)
                yield role, derive(frame, config["physics"], pairs=role == "femto")

    result = collect_raw_qa(config, frames())
    write_json(directory / "results.json", result)
    from .raw_qa_output import render_raw_qa

    render_raw_qa(directory, result, config)
    return result
