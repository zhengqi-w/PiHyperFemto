"""Streaming correlation analysis used identically for nominal and variations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .correlation import accumulate, empty_histograms, normalized_correlation
from .config import digest
from .io import iter_parquet, parquet_columns, write_json
from .purity import correct_correlation, fit_purity
from .selection import pair_weights, select_pairs


def evaluate_histograms(histograms, config):
    a = config["analysis"]
    results = {}
    for key, item in histograms.items():
        correlations = {}
        for region, hist in item["regions"].items():
            correlations[region] = normalized_correlation(
                hist["se"],
                hist["me"],
                hist["se2"],
                hist["me2"],
                a["kstar_edges"],
                a["normalization"],
            )
        purity = fit_purity(item["mass_counts"], item["mass_edges"], config)
        accepted = purity["status"] in {"ok", "external_fixed"}
        corrected = correct_correlation(
            correlations["signal"],
            correlations["sideband"],
            purity["value"] if accepted else np.nan,
            purity["error"],
        )
        if not accepted:
            corrected["status"] = "purity_" + purity["status"]
        results[key] = {
            **item,
            "correlations": correlations,
            "purity": purity,
            "corrected": corrected,
        }
    return results


def export_results(results, directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    edges = config["analysis"]["kstar_edges"]
    rows = []
    arrays = {}
    for key, item in results.items():
        raw, corrected = item["correlations"]["signal"], item["corrected"]
        arrays[key + "__raw_covariance"] = raw["covariance"]
        arrays[key + "__corrected_covariance"] = corrected["covariance"]
        for i, (low, high) in enumerate(zip(edges, edges[1:])):
            rows.append(
                {
                    "bin": key,
                    "cent_low": item["centrality"][0],
                    "cent_high": item["centrality"][1],
                    "channel": item["channel"],
                    "channel_label": item["channel_label"],
                    "kstar_low": low,
                    "kstar_high": high,
                    "se": item["regions"]["signal"]["se"][i],
                    "me": item["regions"]["signal"]["me"][i],
                    "raw": raw["values"][i],
                    "raw_error": np.sqrt(raw["covariance"][i, i]),
                    "raw_valid": raw["valid"][i],
                    "corrected": corrected["values"][i],
                    "corrected_error": np.sqrt(corrected["covariance"][i, i]),
                    "corrected_valid": corrected["valid"][i],
                    "purity": item["purity"]["value"],
                    "purity_status": item["purity"]["status"],
                    "corrected_status": corrected["status"],
                }
            )
    pd.DataFrame(rows).to_csv(directory / "correlations.csv", index=False)
    np.savez_compressed(directory / "covariances.npz", **arrays)
    write_json(directory / "results.json", results)


def analysis_columns(config, available):
    """Project observables/cuts, retaining optional QA and model routing fields."""
    columns = {
        "HypCentFT0C",
        "ChargeChannel",
        "IsMixed",
        "mass_h3l",
        "kstar",
        "bdt_score",
    }
    analysis = config["analysis"]
    for name in ("candidate_cuts", "pair_cuts", "event_cuts"):
        columns.update(analysis[name])
    if analysis["close_pair_reject_bits"]:
        columns.add("IsClosePairRejected")
    if analysis["mixing_weight"] == "inverse_depth":
        columns.add("MixingDepth")
    columns.update(
        set(available)
        & {
            "IsClosePairRejected",
            "HypTrackOccupancy",
            "PairCentFT0C",
            "bdt_model_id",
            "bdt_in_scope",
            "pt_h3l",
        }
    )
    return sorted(columns)


def histogram_signature(config, threshold):
    """Selection/window changes need rows; normalization/fit choices need only bins."""
    analysis = {
        key: value
        for key, value in config["analysis"].items()
        if key not in ("normalization", "bdt_threshold")
    }
    return digest(
        {
            "analysis": analysis,
            "threshold": threshold,
            "mass_bins": config["purity"]["mass_bins"],
        }
    )


def empty_mixing_qa():
    return {
        "missing_pair_centrality": False,
        "delta_centrality_edges": np.linspace(-20, 20, 81),
        "delta_centrality_se": np.zeros(80),
        "delta_centrality_me": np.zeros(80),
        "close_pair_flags": {},
        "occupancy_edges": np.linspace(0, 50000, 101),
        "occupancy_se": np.zeros(100),
        "occupancy_me": np.zeros(100),
    }


class CorrelationAccumulator:
    """A bounded set of sums; keeps no pair rows between batches."""

    def __init__(self, config, threshold):
        self.config = config
        self.threshold = threshold
        self.histograms = empty_histograms(config)
        self.cutflow = {}
        self.qa = empty_mixing_qa()

    def add(self, frame):
        qa = self.qa
        if "IsClosePairRejected" in frame:
            for flag, count in frame.IsClosePairRejected.value_counts().items():
                qa["close_pair_flags"][int(flag)] = qa["close_pair_flags"].get(
                    int(flag), 0
                ) + int(count)
        mask, flow = select_pairs(frame, self.config["analysis"], self.threshold)
        for key, value in flow.items():
            self.cutflow[key] = self.cutflow.get(key, 0) + value
        selected = frame.loc[mask]
        accumulate(
            self.histograms,
            selected,
            pair_weights(selected, self.config["analysis"]["mixing_weight"]),
            self.config,
        )
        for label, mixed in (("se", False), ("me", True)):
            sample = selected.loc[selected.IsMixed == mixed]
            if "PairCentFT0C" in sample:
                qa["delta_centrality_" + label] += np.histogram(
                    sample.PairCentFT0C - sample.HypCentFT0C,
                    qa["delta_centrality_edges"],
                )[0]
            else:
                qa["missing_pair_centrality"] = True
            if "HypTrackOccupancy" in sample:
                qa["occupancy_" + label] += np.histogram(
                    sample.HypTrackOccupancy, qa["occupancy_edges"]
                )[0]

    def export(self, directory, config=None):
        settings = self.config if config is None else config
        results = evaluate_histograms(self.histograms, settings)
        export_results(results, directory, settings)
        write_json(
            Path(directory) / "cutflow.json",
            {"threshold": self.threshold, "counts": self.cutflow},
        )
        write_json(Path(directory) / "mixing_qa.json", self.qa)
        return results


def parquet_schema_columns(directory):
    return parquet_columns(directory)


def run_correlations(scored_directory, directory, config, model_threshold):
    threshold = config["analysis"]["bdt_threshold"]
    threshold = model_threshold if threshold is None else threshold
    from .scored_storage import ensure_storage_coverage

    ensure_storage_coverage(scored_directory, threshold, config)
    accumulator = CorrelationAccumulator(config, threshold)
    columns = analysis_columns(config, parquet_schema_columns(scored_directory))
    for frame in iter_parquet(
        scored_directory, columns=columns, batch_size=config["io"]["chunk_rows"]
    ):
        accumulator.add(frame)
    return accumulator.export(directory)
