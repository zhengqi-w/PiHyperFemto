"""MCFemto plug-in stage: primary/species fractions, truth QA, resolution, closure.

Empty optional input produces a skipped artifact. Generated-only MCHyper rows
are never substituted for reconstructed pair-level MCFemto observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import run_correlations
from .correlation import CHANNELS, accumulate, empty_histograms
from .io import iter_parquet, parquet_columns, parquet_paths, read_json, write_json
from .selection import in_window, pair_weights, select_pairs


def wilson_interval(successes, trials, z=1.0):
    successes, trials = np.asarray(successes, float), np.asarray(trials, float)
    if (
        np.any(~np.isfinite(successes))
        or np.any(~np.isfinite(trials))
        or np.any(successes < 0)
        or np.any(trials < successes)
        or z <= 0
    ):
        raise ValueError(
            "Wilson interval requires finite 0 <= successes <= trials and z > 0"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = successes / trials
        center = (fraction + z * z / (2 * trials)) / (1 + z * z / trials)
        half = (
            z
            * np.sqrt(fraction * (1 - fraction) / trials + z * z / (4 * trials**2))
            / (1 + z * z / trials)
        )
    return {
        "value": fraction,
        "low": center - half,
        "high": center + half,
        "valid": trials > 0,
        "numerator": successes,
        "denominator": trials,
        "uncertainty": "pair_independent_Wilson_68pct",
    }


def run_mc_qa(scored_directory, directory, config, model_threshold):
    from .scored_storage import require_role_dataset

    metadata_path = Path(scored_directory) / "dataset.json"
    if metadata_path.exists():
        require_role_dataset(read_json(metadata_path))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not parquet_paths(scored_directory):
        metadata_path = Path(scored_directory) / "dataset.json"
        counts = (
            (read_json(metadata_path).get("counts") or {})
            if metadata_path.exists()
            else {}
        )
        input_rows = counts.get("raw_rows", counts.get("rows", 0))
        write_json(
            directory / "summary.json",
            {
                "status": (
                    "empty_after_storage" if input_rows else "skipped_missing_mc_femto"
                ),
                "reason": "No reconstructed MC pair input; primary fractions, kstar resolution and closure are unavailable.",
                "required_input": "MCFemto/AO2D*.root containing O2pihypfemtomc or HADHYPERMC",
                "generated_training_signal_is_not_a_substitute": True,
                "storage_counts": counts,
            },
        )
        return
    edges = np.asarray(config["analysis"]["kstar_edges"])
    n = len(edges) - 1
    threshold = config["analysis"]["bdt_threshold"]
    threshold = model_threshold if threshold is None else threshold
    from .scored_storage import ensure_storage_coverage

    ensure_storage_coverage(scored_directory, threshold, config)
    bins = {}
    truth_histograms = empty_histograms(config)
    counters = {
        "input_rows": 0,
        "selected_rows": 0,
        "selected_signal_window_rows": 0,
        "self_correlations": 0,
    }
    from .analysis import analysis_columns
    from .execution import MC_OPTIONAL

    schema = parquet_columns(scored_directory)
    columns = sorted(set(analysis_columns(config, schema)) | (MC_OPTIONAL & schema))
    available = None
    for frame in iter_parquet(
        scored_directory, columns=columns, batch_size=config["io"]["chunk_rows"]
    ):
        counters["input_rows"] += len(frame)
        mask, _ = select_pairs(frame, config["analysis"], threshold)
        frame = frame.loc[mask]
        counters["selected_rows"] += len(frame)
        available = (
            set(frame.columns) if available is None else available & set(frame.columns)
        )
        if "IsTruthSelfCorrelation" in frame:
            counters["self_correlations"] += int(frame.IsTruthSelfCorrelation.sum())
        qa_frame = frame.loc[
            in_window(frame.mass_h3l, config["analysis"]["signal_window"])
        ]
        counters["selected_signal_window_rows"] += len(qa_frame)
        for lo, hi in config["analysis"]["centrality_bins"]:
            for channel in CHANNELS:
                for label, mixed in (("se", False), ("me", True)):
                    key = f"cent_{lo:g}_{hi:g}__ch_{channel}__{label}"
                    if key not in bins:
                        bins[key] = {
                            "total": np.zeros(n),
                            "primary": np.zeros(n),
                            "true_pion": np.zeros(n),
                            "primary_pion": np.zeros(n),
                            "signal": np.zeros(n),
                            "self": np.zeros(n),
                            "same_collision": np.zeros(n),
                            "true_primary_pair": np.zeros(n),
                            "matches_hyp_collision": np.zeros(n),
                            "matches_pair_collision": np.zeros(n),
                            "resolution": np.zeros((n, n)),
                            "residual_sum": np.zeros(n),
                            "residual_sum2": np.zeros(n),
                            "residual_count": np.zeros(n),
                        }
                    item = bins[key]
                    sample = qa_frame.loc[
                        (qa_frame.HypCentFT0C >= lo)
                        & (qa_frame.HypCentFT0C < hi)
                        & (qa_frame.ChargeChannel == channel)
                        & (qa_frame.IsMixed == mixed)
                    ]
                    ks = sample.kstar.to_numpy()
                    item["total"] += np.histogram(ks, edges)[0]
                    for numerator, column in (
                        ("primary", "PairPiIsPhysicalPrimary"),
                        ("signal", "HypIsSignal"),
                        ("true_pion", "PairPiIsTruePion"),
                        ("self", "IsTruthSelfCorrelation"),
                        ("same_collision", "SameMCCollision"),
                        ("true_primary_pair", "IsTruePrimaryPiHyperPair"),
                        ("matches_hyp_collision", "MatchesHypRecoMCCollision"),
                        ("matches_pair_collision", "MatchesPairRecoMCCollision"),
                    ):
                        if column in sample:
                            item[numerator] += np.histogram(
                                ks[sample[column].to_numpy(dtype=bool)], edges
                            )[0]
                    if {"PairPiIsPhysicalPrimary", "PairPiIsTruePion"} <= set(sample):
                        primary_pi = (
                            sample.PairPiIsPhysicalPrimary & sample.PairPiIsTruePion
                        )
                        item["primary_pion"] += np.histogram(ks[primary_pi], edges)[0]
                    if "KstarMC" in sample:
                        good = np.isfinite(sample.KstarMC) & (sample.KstarMC >= 0)
                        # Resolution is meaningful for a truth-matched signal and pion.
                        if "HypIsSignal" in sample:
                            good &= sample.HypIsSignal
                        if "PairPiIsTruePion" in sample:
                            good &= sample.PairPiIsTruePion
                        reco, gen = ks[good], sample.KstarMC.to_numpy()[good]
                        residual = reco - gen
                        item["resolution"] += np.histogram2d(gen, reco, [edges, edges])[
                            0
                        ]
                        item["residual_count"] += np.histogram(gen, edges)[0]
                        item["residual_sum"] += np.histogram(
                            gen, edges, weights=residual
                        )[0]
                        item["residual_sum2"] += np.histogram(
                            gen, edges, weights=residual**2
                        )[0]
        truth_columns = {
            "HypIsSignal",
            "PairPiIsPhysicalPrimary",
            "PairPiIsTruePion",
            "SameMCCollision",
            "IsTruthSelfCorrelation",
        }
        if truth_columns <= set(frame):
            truth = (
                frame.HypIsSignal
                & frame.PairPiIsPhysicalPrimary
                & frame.PairPiIsTruePion
                & ~frame.IsTruthSelfCorrelation
            )
            # ME spans different collisions by design; this truth condition is SE-only.
            truth &= frame.IsMixed | frame.SameMCCollision
            selected = frame.loc[truth]
            accumulate(
                truth_histograms,
                selected,
                pair_weights(selected, config["analysis"]["mixing_weight"]),
                config,
            )
    available = available or set()
    for item in bins.values():
        for numerator, column in (
            ("primary", "PairPiIsPhysicalPrimary"),
            ("true_pion", "PairPiIsTruePion"),
            ("signal", "HypIsSignal"),
            ("self", "IsTruthSelfCorrelation"),
            ("same_collision", "SameMCCollision"),
            ("true_primary_pair", "IsTruePrimaryPiHyperPair"),
            ("matches_hyp_collision", "MatchesHypRecoMCCollision"),
            ("matches_pair_collision", "MatchesPairRecoMCCollision"),
        ):
            item[numerator + "_fraction"] = (
                wilson_interval(item[numerator], item["total"])
                if column in available
                else {"status": "missing_truth_column", "column": column}
            )
        item["primary_pion_fraction"] = (
            wilson_interval(item["primary_pion"], item["total"])
            if {"PairPiIsTruePion", "PairPiIsPhysicalPrimary"} <= available
            else {"status": "missing_truth_columns"}
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = item["residual_sum"] / item["residual_count"]
            item["resolution_bias"] = mean
            item["resolution_rms"] = np.sqrt(
                np.maximum(item["residual_sum2"] / item["residual_count"] - mean**2, 0)
            )
    run_correlations(
        scored_directory, directory / "reco_closure", config, model_threshold
    )
    truth_columns = {
        "HypIsSignal",
        "PairPiIsPhysicalPrimary",
        "PairPiIsTruePion",
        "SameMCCollision",
        "IsTruthSelfCorrelation",
    }
    truth_status = "missing_truth_columns"
    if truth_columns <= available:
        from .analysis import evaluate_histograms, export_results

        export_results(
            evaluate_histograms(truth_histograms, config),
            directory / "truth_closure",
            config,
        )
        truth_status = "available"
    write_json(
        directory / "summary.json",
        {
            "status": "available",
            "bins": bins,
            "counters": counters,
            "kstar_edges": edges,
            "truth_closure_status": truth_status,
            "fraction_mass_window": config["analysis"]["signal_window"],
            "fraction_weighting": "unweighted_reconstructed_pair_rows",
            "limitations": [
                "Fractions are reconstructed, selection-dependent and pair-weighted.",
                "Fractions use the data signal mass window; ME fractions are unweighted QA even if CF uses inverse-depth weights.",
                "The producer true-primary-pair flag can require same MC collision and is not an ME purity definition.",
                "No automatic primary-dilution correction is applied.",
                "A generator without FSI can still have non-femtoscopic correlations; closure to unity is diagnostic.",
                "MC training candidates do not define pair reconstruction efficiency.",
            ],
        },
    )
