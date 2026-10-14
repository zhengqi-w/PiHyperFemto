"""Configuration loading and validation. All units are GeV, cm, and percent."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import yaml

from .errors import AnalysisError
from .training_settings import XGB_DEFAULTS

DEFAULTS = {
    "schema_version": 1,
    "data_root": ".",
    "output": "outputs/analysis",
    "seed": 73129,
    "periods": [],
    "synthetic_repetition": False,
    "io": {
        "chunk_rows": 100000,
        "fingerprint": "stat",
        "checkpoint": True,
        "slim_scored": True,
        "root_projection": True,
        "persist_prepared": False,
        "cache_model_scores": False,
        "scored_partition_by_pt": True,
        "scored_columns": ["mass_h3l", "kstar", "pt_h3l"],
        "scored_optional_columns": [
            "HypTrackOccupancy",
            "PairCentFT0C",
            "MixingDepth",
            "IsClosePairRejected",
        ],
    },
    "raw_qa": {
        "mass_range": [2.94, 3.06],
        "mass_bins": 120,
        "kstar_edges": np.linspace(0, 1.2, 61).tolist(),
    },
    "inputs": {
        "femto": {"patterns": ["Femto/AO2D*.root"], "required": True},
        "background": {"patterns": ["Hyper/AO2D*.root"], "required": True},
        "signal": {"patterns": ["MCHyper/AO2D*.root"], "required": True},
        "mc_femto": {"patterns": ["MCFemto/AO2D*.root"], "required": False},
    },
    "physics": {
        "mass_he3": 2.80839160743,
        "mass_pi": 0.13957039,
        "mass_h3l": 2.99131,
        "he3_momentum_scale": 1.0,
        "pair_mass": "nominal",
    },
    "analysis": {
        "centrality_bins": [[0, 10], [10, 50]],
        "kstar_edges": np.linspace(0, 0.6, 31).tolist(),
        "normalization": [0.3, 0.5],
        "signal_window": [2.985, 2.997],
        "sidebands": [[2.96, 2.975], [3.007, 3.025]],
        "mass_range": [2.96, 3.025],
        "mixing_weight": "unit",
        "close_pair_reject_bits": 3,
        "candidate_cuts": {},
        "pair_cuts": {},
        "event_cuts": {},
        "bdt_threshold": None,
    },
    "training": {
        "model_params": copy.deepcopy(XGB_DEFAULTS),
        "class_weight": "balanced",
        "use_optuna": False,
        "optuna": {
            "metric": "auc",
            "direction": "maximize",
            "n_trials": 20,
            "timeout": 300,
            "cv_folds": 3,
            "n_jobs": 1,
            "n_startup_trials": 5,
            "pruner": "median",
            "n_warmup_steps": 1,
            "search_space": {},
        },
        "qa_shap_max_per_class": 1000,
        "efficiency_min": 0.29,
        "efficiency_max": 0.99,
        "npoints_for_effi": 701,
        "qa_bins": 100,
        "pt_range": None,
        "pt_bins": [],
        "retrain_bins": [],
        "candidate_cuts": {},
        "signal_mass_window": [2.95, 3.02],
        "mc_range_selection": {
            "enabled": False,
            "features": None,
            "quantiles": [0.001, 0.999],
            "margin_fraction": 0.05,
            "reference_fraction": 0.1,
            "min_reference_rows": 50,
        },
        "features": [
            "DcaV0Daug",
            "DcaHe",
            "DcaPi",
            "NSigmaHe",
            "cos_pa",
        ],
        "max_per_class": 100000,
        "min_per_class": 30,
        "validation_fraction": 0.25,
        "target_signal_efficiency": 0.8,
    },
    "working_point": {
        "signal_efficiencies": None,
        "candidate_cuts": None,
        "event_cuts": None,
        "centrality_bins": None,
        "mass_range": [2.96, 3.04],
        "signal_window": None,
        "sidebands": [[2.96, 2.98], [3.005, 3.04]],
        "purity": None,
        "mass_fit": {
            "signal_model": "dscb",
            "background": "pol1",
            "mass_bins": 50,
            "mc_mass_bins": 80,
            "min_mass_entries": 80,
            "mean_bounds": [2.985, 3.005],
            "sigma_bounds": [0.0014, 0.0020],
            "calibrate_mc_tails": True,
            "sideband_prefit": True,
            "alpha_bounds": [0.7, 5.0],
            "n_bounds": [0.9, 15.0],
            "tail_defaults": {
                "alpha_left": 1.5,
                "n_left": 2.0,
                "alpha_right": 1.5,
                "n_right": 2.0,
            },
        },
        "period_text": "ALICE Run 3 PbPb",
        "additional_text": "PbPb #sqrt{#it{s_{NN}}} = 5.36 TeV",
        "storage": {"enabled": True, "reserve_signal_efficiency": 0.99},
        "objective": "pol0_expected_significance",
        "yield_eff_range": [0.5, 0.95],
        "min_pol0_points": 3,
        "max_fit_chi2_ndf": 10.0,
        "save_score_fit_frames": False,
    },
    "purity": {
        "method": "fit",
        "min_mass_entries": 80,
        "mass_bins": 65,
        "background": "linear",
        "sigma_bounds": [0.0008, 0.004],
        "mean_bounds": [2.988, 2.994],
        "fixed": None,
        "fixed_error": 0.0,
    },
    "uncertainty": {"method": "pair_poisson"},
    "systematics": {"method": "envelope", "variations": []},
    "mc": {"apply_primary_correction": False},
}


def merge(base: dict, override: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise AnalysisError("Configuration overrides must be mappings")
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _mapping(value, name, *, allowed=None, required=()):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AnalysisError(f"{name} must be a mapping with string keys")
    missing = set(required) - set(value)
    if missing:
        raise AnalysisError(f"Missing {name} options: {sorted(missing)}")
    if allowed is not None:
        unknown = set(value) - set(allowed)
        if unknown:
            raise AnalysisError(f"Unknown {name} options: {sorted(unknown)}")


def _number(value, name, *, minimum=None, maximum=None, integer=False):
    expected = Integral if integer else Real
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, expected)
        or not np.isfinite(value)
    ):
        kind = "integer" if integer else "number"
        raise AnalysisError(f"{name} must be a finite {kind}")
    if minimum is not None and value < minimum:
        raise AnalysisError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise AnalysisError(f"{name} must be <= {maximum}")


def _interval(value, name, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnalysisError(f"{name} must be a finite increasing [low, high] interval")
    for edge in value:
        _number(edge, name)
    if value[0] >= value[1] or (positive and value[0] <= 0):
        qualifier = "positive " if positive else ""
        raise AnalysisError(
            f"{name} must be a {qualifier}increasing [low, high] interval"
        )


def _cuts(cuts, name):
    _mapping(cuts, name)
    for column, specification in cuts.items():
        if not column:
            raise AnalysisError(f"{name} column names must not be empty")
        if isinstance(specification, (list, tuple)):
            if len(specification) != 2:
                raise AnalysisError(f"{name}.{column} interval cuts require [min, max]")
            specification = dict(zip(("min", "max"), specification))
        _mapping(
            specification,
            f"{name}.{column}",
            allowed={"min", "max", "abs_max", "equals"},
        )
        if not specification or all(value is None for value in specification.values()):
            raise AnalysisError(f"{name}.{column} must define at least one cut")
        for field, value in specification.items():
            if value is None and field in {"min", "max"}:
                continue
            if field == "equals" and isinstance(value, bool):
                continue
            _number(value, f"{name}.{column}.{field}")
        low, high = specification.get("min"), specification.get("max")
        if low is not None and high is not None and low >= high:
            raise AnalysisError(f"{name}.{column} requires min < max")
        if "abs_max" in specification and specification["abs_max"] <= 0:
            raise AnalysisError(f"{name}.{column}.abs_max must be positive")


def _slug(value, name):
    if (
        not isinstance(value, str)
        or not value
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in value)
    ):
        raise AnalysisError(f"{name} must use lowercase letters, digits, '_' or '-'")


def period_configs(config):
    """Resolve validated child analyses with shared physics and distinct outputs.

    Missing periods means the normal single-period workflow. Input/training/WP
    options are merged, while analysis/physics/systematics definitions stay shared.
    """
    periods = config.get("periods", [])
    if not isinstance(periods, list):
        raise AnalysisError("periods must be a list of period specifications")
    ids = []
    for spec in periods:
        _mapping(
            spec,
            "period",
            allowed={"id", "inputs", "training", "working_point", "seed"},
            required={"id"},
        )
        _slug(spec["id"], "Period id")
        for section in ("inputs", "training", "working_point"):
            if section in spec:
                _mapping(
                    spec[section],
                    f"period {spec['id']}.{section}",
                    allowed=DEFAULTS[section],
                )
        ids.append(spec["id"])
    if len(set(ids)) != len(ids):
        raise AnalysisError("Period ids must be unique")
    children = []
    for spec in periods:
        child = merge(
            config, {key: value for key, value in spec.items() if key != "id"}
        )
        child["periods"] = []
        child["output"] = str(Path(config["output"]) / "periods" / spec["id"])
        try:
            validate(child)
        except AnalysisError as error:
            raise AnalysisError(f"Invalid period {spec['id']}: {error}") from error
        children.append((spec["id"], child))
    return children


def validate(config: dict) -> None:
    """Validate a resolved configuration, including every systematic override."""
    _mapping(config, "configuration", allowed=DEFAULTS, required=DEFAULTS)
    for section in DEFAULTS:
        if isinstance(DEFAULTS[section], dict):
            _mapping(
                config[section],
                section,
                allowed=DEFAULTS[section],
                required=DEFAULTS[section],
            )
    _number(config["schema_version"], "schema_version", integer=True)
    if config["schema_version"] != 1:
        raise AnalysisError("Only configuration schema_version: 1 is supported")
    _number(config["seed"], "seed", minimum=0, maximum=2**32 - 1, integer=True)
    if not isinstance(config["synthetic_repetition"], bool):
        raise AnalysisError("synthetic_repetition must be boolean")
    for field in ("data_root", "output"):
        if not isinstance(config[field], (str, Path)) or not str(config[field]).strip():
            raise AnalysisError(f"{field} must be a nonempty filesystem path")
    for role, spec in config["inputs"].items():
        _mapping(
            spec,
            f"inputs.{role}",
            allowed={"patterns", "required", "root"},
            required={"patterns", "required"},
        )
        if not isinstance(spec["required"], bool):
            raise AnalysisError(f"inputs.{role}.required must be boolean")
        if "root" in spec:
            root = spec["root"]
            if root is None and spec["required"]:
                raise AnalysisError(
                    f"inputs.{role}.root cannot be null for a required input"
                )
            if root is not None and not (
                isinstance(root, str)
                and root.strip()
                or isinstance(root, list)
                and root
                and all(isinstance(value, str) and value.strip() for value in root)
            ):
                raise AnalysisError(
                    f"inputs.{role}.root must be a nonempty directory string, a nonempty list of directory strings, or null for an optional input"
                )
        if (
            not isinstance(spec["patterns"], (list, tuple))
            or any(
                not isinstance(pattern, str) or not pattern.strip()
                for pattern in spec["patterns"]
            )
            or (spec["required"] and not spec["patterns"])
        ):
            raise AnalysisError(
                f"inputs.{role}.patterns must be a list of nonempty path/glob strings"
            )
    _number(config["io"]["chunk_rows"], "io.chunk_rows", minimum=1, integer=True)
    for name in (
        "checkpoint",
        "slim_scored",
        "root_projection",
        "persist_prepared",
        "cache_model_scores",
        "scored_partition_by_pt",
    ):
        if not isinstance(config["io"][name], bool):
            raise AnalysisError(f"io.{name} must be boolean")
    for name in ("scored_columns", "scored_optional_columns"):
        values = config["io"][name]
        if (
            not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))
        ):
            raise AnalysisError(
                f"io.{name} must be a list of unique nonempty column names"
            )
    _interval(config["raw_qa"]["mass_range"], "raw_qa.mass_range", positive=True)
    _number(config["raw_qa"]["mass_bins"], "raw_qa.mass_bins", minimum=2, integer=True)
    qa_edges = config["raw_qa"]["kstar_edges"]
    if not isinstance(qa_edges, (list, tuple)) or len(qa_edges) < 3:
        raise AnalysisError("raw_qa.kstar_edges requires at least three edges")
    for edge in qa_edges:
        _number(edge, "raw_qa.kstar_edges", minimum=0)
    if np.any(np.diff(qa_edges) <= 0):
        raise AnalysisError("raw_qa.kstar_edges must be strictly increasing")
    if config["io"]["fingerprint"] not in ("stat", "sha256"):
        raise AnalysisError("io.fingerprint must be stat or sha256")
    physics = config["physics"]
    for field in ("mass_he3", "mass_pi", "mass_h3l", "he3_momentum_scale"):
        _number(physics[field], f"physics.{field}")
        if physics[field] <= 0:
            raise AnalysisError(f"physics.{field} must be positive")
    if physics["pair_mass"] not in ("reconstructed", "nominal"):
        raise AnalysisError("pair_mass must be reconstructed or nominal")
    a = config["analysis"]
    if not isinstance(a["kstar_edges"], (list, tuple)):
        raise AnalysisError("kstar_edges must be a sequence of histogram edges")
    for edge in a["kstar_edges"]:
        _number(edge, "kstar_edges", minimum=0)
    edges = np.asarray(a["kstar_edges"], dtype=float)
    if len(edges) < 3 or np.any(np.diff(edges) <= 0):
        raise AnalysisError("kstar_edges must be finite and strictly increasing")
    bins = a["centrality_bins"]
    if not isinstance(bins, (list, tuple)) or not bins:
        raise AnalysisError("At least one centrality bin is required")
    for i, bounds in enumerate(bins):
        _interval(bounds, "centrality_bins")
        if bounds[0] < 0 or bounds[1] > 100 or (i and bounds[0] < bins[i - 1][1]):
            raise AnalysisError(
                "Centrality bins must be sorted, disjoint and within [0,100]"
            )
    for field in ("normalization", "signal_window", "mass_range"):
        _interval(a[field], field)
    if not all(
        np.any(np.isclose(edges, x, atol=1e-10, rtol=0)) for x in a["normalization"]
    ):
        raise AnalysisError(
            "Normalization bounds must coincide with kstar histogram edges"
        )
    if not isinstance(a["sidebands"], (list, tuple)) or len(a["sidebands"]) != 2:
        raise AnalysisError("Exactly two sidebands (left and right) are required")
    windows = [a["signal_window"], *a["sidebands"]]
    for window in windows:
        _interval(window, "mass window")
        if window[0] < a["mass_range"][0] or window[1] > a["mass_range"][1]:
            raise AnalysisError("Signal and sidebands must lie inside mass_range")
    ordered = sorted(windows)
    if any(x[1] > y[0] for x, y in zip(ordered, ordered[1:])):
        raise AnalysisError("Signal window and sidebands must not overlap")
    if (
        a["sidebands"][0][1] > a["signal_window"][0]
        or a["sidebands"][1][0] < a["signal_window"][1]
    ):
        raise AnalysisError(
            "analysis.sidebands must be ordered as left and right of the signal window"
        )
    if a["mixing_weight"] not in ("unit", "inverse_depth"):
        raise AnalysisError("mixing_weight must be unit or inverse_depth")
    _number(
        a["close_pair_reject_bits"],
        "analysis.close_pair_reject_bits",
        minimum=0,
        maximum=15,
        integer=True,
    )
    if a["bdt_threshold"] is not None:
        _number(a["bdt_threshold"], "analysis.bdt_threshold")
    for name in ("candidate_cuts", "pair_cuts", "event_cuts"):
        _cuts(a[name], f"analysis.{name}")
    t = config["training"]
    from .training_settings import validate_training_settings

    validate_training_settings(t)
    _cuts(t["candidate_cuts"], "training.candidate_cuts")
    ranges = t["mc_range_selection"]
    _mapping(
        ranges,
        "training.mc_range_selection",
        allowed=set(DEFAULTS["training"]["mc_range_selection"]),
    )
    if not isinstance(ranges["enabled"], bool):
        raise AnalysisError("training.mc_range_selection.enabled must be boolean")
    _interval(ranges["quantiles"], "training.mc_range_selection.quantiles")
    if ranges["quantiles"][0] < 0 or ranges["quantiles"][1] > 1:
        raise AnalysisError("MC range quantiles must lie in [0,1]")
    _number(ranges["margin_fraction"], "MC range margin_fraction", minimum=0)
    _number(
        ranges["reference_fraction"],
        "MC range reference_fraction",
        minimum=0,
        maximum=1,
    )
    if not 0 < ranges["reference_fraction"] < 1:
        raise AnalysisError("MC range reference_fraction must lie in (0,1)")
    _number(
        ranges["min_reference_rows"],
        "MC range min_reference_rows",
        integer=True,
        minimum=2,
    )
    if ranges["features"] is not None and (
        not isinstance(ranges["features"], (list, tuple))
        or not ranges["features"]
        or any(not isinstance(name, str) for name in ranges["features"])
        or len(set(ranges["features"])) != len(ranges["features"])
        or not isinstance(t["features"], (list, tuple))
        or not set(ranges["features"]) <= set(t["features"])
    ):
        raise AnalysisError(
            "MC range features must be a nonempty unique subset of BDT features"
        )
    if t["pt_range"] is not None:
        _interval(t["pt_range"], "training.pt_range")
        if t["pt_range"][0] < 0:
            raise AnalysisError("training.pt_range must be nonnegative")
    if not isinstance(t["pt_bins"], (list, tuple)):
        raise AnalysisError("training.pt_bins must be a list")
    if t["pt_bins"] and t["pt_range"] is not None:
        raise AnalysisError("Use either training.pt_range or training.pt_bins")
    model_ids, previous_high = [], None
    for index, item in enumerate(t["pt_bins"]):
        if isinstance(item, dict):
            _mapping(
                item,
                "training.pt_bins entry",
                allowed={
                    "id",
                    "pt_range",
                    "features",
                    "candidate_cuts",
                    "model_params",
                    "use_optuna",
                    "optuna",
                    "efficiency_min",
                    "efficiency_max",
                    "npoints_for_effi",
                },
                required={"id", "pt_range"},
            )
            _slug(item["id"], "Training model id")
            model_id, bounds = item["id"], item["pt_range"]
        else:
            model_id, bounds = f"pt{index}", item
        _interval(bounds, "training.pt_bins.pt_range")
        if bounds[0] < 0 or (previous_high is not None and bounds[0] < previous_high):
            raise AnalysisError(
                "Training pT intervals must be sorted, disjoint and nonnegative"
            )
        previous_high = bounds[1]
        model_ids.append(model_id)
        if isinstance(item, dict) and set(item) & {
            "features",
            "candidate_cuts",
            "model_params",
            "use_optuna",
            "optuna",
            "efficiency_min",
            "efficiency_max",
            "npoints_for_effi",
        }:
            scoped = copy.deepcopy(config)
            scoped["training"].update(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "features",
                        "candidate_cuts",
                        "use_optuna",
                        "efficiency_min",
                        "efficiency_max",
                        "npoints_for_effi",
                    }
                }
            )
            if "candidate_cuts" in item:
                _mapping(item["candidate_cuts"], "training.pt_bins.candidate_cuts")
                scoped["training"]["candidate_cuts"] = merge(
                    t["candidate_cuts"], item["candidate_cuts"]
                )
            if "model_params" in item:
                _mapping(item["model_params"], "training.pt_bins.model_params")
                scoped["training"]["model_params"] = merge(
                    t["model_params"], item["model_params"]
                )
            if "optuna" in item:
                _mapping(item["optuna"], "training.pt_bins.optuna")
                scoped["training"]["optuna"] = merge(t["optuna"], item["optuna"])
                if "search_space" in item["optuna"]:
                    scoped["training"]["optuna"]["search_space"] = copy.deepcopy(
                        item["optuna"]["search_space"]
                    )
            scoped["training"]["pt_bins"] = []
            scoped["training"]["retrain_bins"] = []
            scoped["systematics"]["variations"] = []
            validate(scoped)
    if len(model_ids) != len(set(model_ids)):
        raise AnalysisError("Training model ids must be unique")
    if (
        not isinstance(t["retrain_bins"], (list, tuple))
        or any(not isinstance(value, str) for value in t["retrain_bins"])
        or len(t["retrain_bins"]) != len(set(t["retrain_bins"]))
    ):
        raise AnalysisError("training.retrain_bins must be a list of unique model ids")
    if set(t["retrain_bins"]) - set(model_ids or ["global"]):
        raise AnalysisError("training.retrain_bins references an unknown model")
    if isinstance(t["features"], (list, tuple)) and any(
        feature in t["features"] for feature in ("ct", "fCt")
    ):
        raise AnalysisError(
            "ct/fCt is forbidden as a BDT feature to avoid lifetime sculpting"
        )
    allowed = {
        "DcaV0Daug",
        "DcaHe",
        "DcaPi",
        "NSigmaHe",
        "TofMass",
        "cos_pa",
        "pt_h3l",
        "eta_h3l",
        "PtHe3",
        "PtPi",
        "EtaHe3",
        "EtaPi",
        "TpcSignalHe",
        "TpcSignalPi",
        "its_mean_size_he",
        "its_mean_size_pi",
        "its_ncls_he",
        "its_ncls_pi",
    }
    if (
        not isinstance(t["features"], (list, tuple))
        or not t["features"]
        or any(not isinstance(feature, str) for feature in t["features"])
        or set(t["features"]) - allowed
    ):
        raise AnalysisError(
            "BDT features must be approved candidate variables; mass/pair/truth/identity features are forbidden"
        )
    if len(set(t["features"])) != len(t["features"]):
        raise AnalysisError("BDT features must be unique")
    for name in (
        "target_signal_efficiency",
        "validation_fraction",
    ):
        _number(t[name], f"training.{name}")
    if (
        not 0 < t["target_signal_efficiency"] < 1
        or not 0 < t["validation_fraction"] < 0.5
    ):
        raise AnalysisError("Invalid training efficiency or validation_fraction")
    for name, minimum in (
        ("min_per_class", 4),
        ("max_per_class", 4),
    ):
        _number(t[name], f"training.{name}", minimum=minimum, integer=True)
    if t["min_per_class"] < 4 or t["max_per_class"] < t["min_per_class"]:
        raise AnalysisError("Invalid training sample limits")
    _interval(t["signal_mass_window"], "training.signal_mass_window", positive=True)
    wp = config["working_point"]
    if wp["objective"] != "pol0_expected_significance":
        raise AnalysisError(
            "working_point.objective must be pol0_expected_significance"
        )
    _number(t["qa_bins"], "training.qa_bins", minimum=2, integer=True)
    _interval(wp["yield_eff_range"], "working_point.yield_eff_range", positive=True)
    if wp["yield_eff_range"][1] > 1:
        raise AnalysisError("working_point.yield_eff_range must lie within (0,1]")
    _number(
        wp["min_pol0_points"], "working_point.min_pol0_points", minimum=3, integer=True
    )
    if wp["max_fit_chi2_ndf"] is not None:
        _number(wp["max_fit_chi2_ndf"], "working_point.max_fit_chi2_ndf", minimum=0)
        if wp["max_fit_chi2_ndf"] <= 0:
            raise AnalysisError(
                "working_point.max_fit_chi2_ndf must be positive or null"
            )
    if not isinstance(wp["save_score_fit_frames"], bool):
        raise AnalysisError("working_point.save_score_fit_frames must be boolean")
    efficiencies = wp["signal_efficiencies"]
    if efficiencies is not None and (
        not isinstance(efficiencies, (list, tuple))
        or len(efficiencies) < 2
        or any(
            not isinstance(value, Real) or isinstance(value, bool)
            for value in efficiencies
        )
        or any(not 0 < value <= 1 for value in efficiencies)
        or any(left >= right for left, right in zip(efficiencies, efficiencies[1:]))
    ):
        raise AnalysisError(
            "working_point.signal_efficiencies must be strictly increasing values in (0,1]"
        )
    for key in ("candidate_cuts", "event_cuts"):
        if wp[key] is not None:
            _cuts(wp[key], f"working_point.{key}")
    wp_bins = wp["centrality_bins"]
    if wp_bins is not None:
        if not isinstance(wp_bins, (list, tuple)):
            raise AnalysisError("working_point.centrality_bins must be a list or null")
        for index, bounds in enumerate(wp_bins):
            _interval(bounds, "working_point.centrality_bins")
            if (
                bounds[0] < 0
                or bounds[1] > 100
                or (index and bounds[0] < wp_bins[index - 1][1])
            ):
                raise AnalysisError(
                    "WP centrality bins must be sorted, disjoint and within [0,100]"
                )
    mass_range = a["mass_range"] if wp["mass_range"] is None else wp["mass_range"]
    signal_window = (
        a["signal_window"] if wp["signal_window"] is None else wp["signal_window"]
    )
    _interval(mass_range, "working_point.mass_range", positive=True)
    _interval(signal_window, "working_point.signal_window", positive=True)
    if signal_window[0] < mass_range[0] or signal_window[1] > mass_range[1]:
        raise AnalysisError("WP signal_window must lie inside WP mass_range")
    wp_sidebands = a["sidebands"] if wp["sidebands"] is None else wp["sidebands"]
    if not isinstance(wp_sidebands, (list, tuple)) or len(wp_sidebands) != 2:
        raise AnalysisError(
            "working_point.sidebands requires exactly two intervals, left then right"
        )
    for window in wp_sidebands:
        _interval(window, "working_point.sidebands", positive=True)
        if window[0] < mass_range[0] or window[1] > mass_range[1]:
            raise AnalysisError("WP sidebands must lie inside WP mass_range")
    if wp_sidebands[0][1] > signal_window[0] or wp_sidebands[1][0] < signal_window[1]:
        raise AnalysisError(
            "WP sidebands must be ordered left and right of the WP signal_window without overlap"
        )
    if wp["purity"] is not None:
        _mapping(wp["purity"], "working_point.purity", allowed=DEFAULTS["purity"])
        if wp["purity"].get("method", "fit") != "fit":
            raise AnalysisError("working_point.purity.method must be fit")
    mass_fit = wp["mass_fit"]
    _mapping(
        mass_fit,
        "working_point.mass_fit",
        allowed=DEFAULTS["working_point"]["mass_fit"],
    )
    if mass_fit["signal_model"] != "dscb":
        raise AnalysisError("working_point.mass_fit.signal_model must be dscb")
    if mass_fit["background"] not in {"pol1", "pol2"}:
        raise AnalysisError("working_point.mass_fit.background must be pol1 or pol2")
    for key in ("calibrate_mc_tails", "sideband_prefit"):
        if not isinstance(mass_fit[key], bool):
            raise AnalysisError(f"working_point.mass_fit.{key} must be boolean")
    for key in ("mass_bins", "mc_mass_bins", "min_mass_entries"):
        _number(
            mass_fit[key],
            f"working_point.mass_fit.{key}",
            integer=True,
            minimum=8 if key != "min_mass_entries" else 0,
        )
    for key in ("mean_bounds", "sigma_bounds", "alpha_bounds", "n_bounds"):
        _interval(mass_fit[key], f"working_point.mass_fit.{key}", positive=True)
    if (
        mass_fit["mean_bounds"][0] < mass_range[0]
        or mass_fit["mean_bounds"][1] > mass_range[1]
    ):
        raise AnalysisError("WP mean_bounds must lie inside mass_range")
    _mapping(
        mass_fit["tail_defaults"],
        "WP tail_defaults",
        allowed={"alpha_left", "alpha_right", "n_left", "n_right"},
        required={"alpha_left", "alpha_right", "n_left", "n_right"},
    )
    for name, value in mass_fit["tail_defaults"].items():
        _number(value, f"WP tail_defaults.{name}", minimum=1e-6)
    for key in ("period_text", "additional_text"):
        if not isinstance(wp[key], str):
            raise AnalysisError(f"working_point.{key} must be text")
    wp_purity = merge(config["purity"], wp["purity"] or {})
    legacy_background = (wp["purity"] or {}).get("background")
    if legacy_background is not None and legacy_background not in {"pol1", "pol2"}:
        raise AnalysisError(
            "WP background must be pol1 or pol2; configure working_point.mass_fit.background"
        )
    _number(
        wp_purity["mass_bins"],
        "working_point.purity.mass_bins",
        minimum=6,
        integer=True,
    )
    _number(
        wp_purity["min_mass_entries"],
        "working_point.purity.min_mass_entries",
        minimum=1,
        integer=True,
    )
    for key in ("mean_bounds", "sigma_bounds"):
        _interval(wp_purity[key], f"working_point.purity.{key}", positive=True)
    if (
        wp_purity["mean_bounds"][0] < mass_range[0]
        or wp_purity["mean_bounds"][1] > mass_range[1]
    ):
        raise AnalysisError("WP purity.mean_bounds must lie inside WP mass_range")
    _mapping(
        wp["storage"],
        "working_point.storage",
        allowed={"enabled", "reserve_signal_efficiency"},
        required={"enabled", "reserve_signal_efficiency"},
    )
    if not isinstance(wp["storage"]["enabled"], bool):
        raise AnalysisError("working_point.storage.enabled must be boolean")
    _number(
        wp["storage"]["reserve_signal_efficiency"],
        "working_point.storage.reserve_signal_efficiency",
        minimum=0,
        maximum=1,
    )
    if wp["storage"]["reserve_signal_efficiency"] == 0:
        raise AnalysisError(
            "working_point.storage.reserve_signal_efficiency must be positive"
        )
    p = config["purity"]
    if p["method"] not in ("fit", "fixed", "none"):
        raise AnalysisError("purity.method must be fit, fixed, or none")
    if p["background"] not in ("linear", "constant"):
        raise AnalysisError("purity.background must be linear or constant")
    _number(p["min_mass_entries"], "purity.min_mass_entries", minimum=1, integer=True)
    _number(p["mass_bins"], "purity.mass_bins", minimum=6, integer=True)
    _interval(p["sigma_bounds"], "purity.sigma_bounds", positive=True)
    _interval(p["mean_bounds"], "purity.mean_bounds", positive=True)
    if (
        p["mean_bounds"][0] < a["mass_range"][0]
        or p["mean_bounds"][1] > a["mass_range"][1]
    ):
        raise AnalysisError(
            "purity.mean_bounds must be contained in analysis.mass_range"
        )
    _number(p["fixed_error"], "purity.fixed_error", minimum=0)
    if p["method"] == "fixed" or p["fixed"] is not None:
        _number(p["fixed"], "purity.fixed", minimum=0, maximum=1)
        if p["fixed"] <= 0:
            raise AnalysisError("Fixed purity requires 0 < purity.fixed <= 1")
    if config["uncertainty"]["method"] != "pair_poisson":
        raise AnalysisError(
            "Only pair_poisson is implemented; event bootstrap requires producer event IDs and a validated resampling design"
        )
    if not isinstance(config["mc"]["apply_primary_correction"], bool):
        raise AnalysisError("mc.apply_primary_correction must be boolean")
    if config["mc"]["apply_primary_correction"]:
        raise AnalysisError(
            "Primary dilution correction is not automatic: validate residual correlations and provide an analysis-specific correction model"
        )
    if config["systematics"]["method"] not in ("envelope", "rms"):
        raise AnalysisError("systematics.method must be envelope or rms")
    if not isinstance(config["systematics"]["variations"], (list, tuple)):
        raise AnalysisError("systematics.variations must be a list")
    names = []
    for variation in config["systematics"]["variations"]:
        _mapping(
            variation,
            "variation",
            allowed={"name", "group", "overrides", "wp_efficiency_delta"},
            required={"name", "overrides"},
        )
        name = variation["name"]
        if "wp_efficiency_delta" in variation:
            delta = variation["wp_efficiency_delta"]
            _number(delta, "variation.wp_efficiency_delta")
            if not -1 < delta < 1:
                raise AnalysisError("wp_efficiency_delta must lie in (-1,1)")
        _slug(name, "Variation name")
        if "group" in variation:
            _slug(variation["group"], "Variation group")
        _mapping(
            variation["overrides"],
            f"variation {name} overrides",
            allowed={"analysis", "purity"},
        )
        for section, options in variation["overrides"].items():
            _mapping(options, f"variation {name}.{section}", allowed=DEFAULTS[section])
        if "wp_efficiency_delta" in variation and "bdt_threshold" in variation[
            "overrides"
        ].get("analysis", {}):
            raise AnalysisError(
                "A WP efficiency variation cannot also override bdt_threshold"
            )
        if set(variation["overrides"].get("analysis", {})) & {
            "mass_range",
            "centrality_bins",
            "kstar_edges",
        }:
            raise AnalysisError(
                "Variations must share preparation range, centrality bins and kstar edges"
            )
        varied = merge(config, variation["overrides"])
        varied["systematics"]["variations"] = []
        try:
            validate(varied)
        except AnalysisError as error:
            raise AnalysisError(f"Invalid variation {name}: {error}") from error
        names.append(name)
    if len(set(names)) != len(names):
        raise AnalysisError("Systematic variation names must be unique")
    period_configs(config)


def _resolve_input_roots(inputs, base):
    """Only explicit role roots are resolved; legacy patterns retain their base."""
    if not isinstance(inputs, dict):
        return
    for role, spec in inputs.items():
        if not isinstance(spec, dict) or "root" not in spec or spec["root"] is None:
            continue
        roots = spec["root"]
        values = roots if isinstance(roots, list) else [roots]
        resolved = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise AnalysisError(
                    f"inputs.{role}.root must contain nonempty directory strings"
                )
            root = Path(os.path.expandvars(os.path.expanduser(value)))
            resolved.append(
                str(
                    (base / root).resolve()
                    if not root.is_absolute()
                    else root.resolve()
                )
            )
        spec["root"] = resolved if isinstance(roots, list) else resolved[0]


def load_config(path: str | Path, *, data_root=None, output=None) -> dict:
    path = Path(path).resolve()
    try:
        with path.open() as handle:
            user = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise AnalysisError(f"Cannot read configuration {path}: {error}") from error
    if user is None:
        user = {}
    config = merge(DEFAULTS, user)
    for key, override in (("data_root", data_root), ("output", output)):
        raw = config[key] if override is None else override
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise AnalysisError(f"{key} must be a nonempty filesystem path")
        value = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        config[key] = str(
            (path.parent / value).resolve() if not value.is_absolute() else value
        )
    _resolve_input_roots(config["inputs"], path.parent)
    if isinstance(config["periods"], list):
        for spec in config["periods"]:
            if isinstance(spec, dict):
                _resolve_input_roots(spec.get("inputs"), path.parent)
    validate(config)
    return config
