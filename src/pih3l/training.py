"""Candidate-tree BDT training; Femto pairs never enter the training sample."""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from .errors import AnalysisError
from .efficiency import mapping_metadata, thresholds_from_efficiencies
from .io import iter_parquet
from .schema import require_columns
from .selection import cut_mask, in_window


def training_groups(frame):
    """Group candidates sharing a source and PV; a documented surrogate collision key.

    This is only used on the independent training candidate trees. It is not an
    exact collision ID, and no equivalent identity is built for Femto pairs.
    """
    fields = ["_source_file", "_source_tree", "XPrimVtx", "YPrimVtx", "ZPrimVtx"]
    require_columns(frame, fields, "training vertex grouping")
    return pd.util.hash_pandas_object(frame[fields], index=False).to_numpy(
        dtype=np.uint64
    )


def sample_candidates(
    directory, role, config, *, return_cutflow=False, support_ranges=None
):
    """Apply shared class cuts before a bounded, uniform priority reservoir.

    By default return (sample, available finite candidates), preserving the public
    API. return_cutflow=True provides the complete counters instead of that count.
    Raw ROOT counts are only aggregated from native batch-source metadata; saved
    Parquet inputs cannot reconstruct upstream truth/kinematic rejection counts.
    """
    if role not in {"signal", "background"}:
        raise AnalysisError("Training sampling role must be signal or background")
    limit = config["training"]["max_per_class"]
    rng = np.random.default_rng(config["seed"] + (0 if role == "signal" else 1))
    reservoir = pd.DataFrame()
    count = 0
    flow = {
        "role": role,
        "source_kind": (
            "batch_source" if hasattr(directory, "iter_batches") else "prepared_parquet"
        ),
        "source_counts_available": True,
        "raw_source_rows": 0,
        "after_source_role_filter": 0,
        "after_source_finite_kinematics": 0,
        "source_nonfinite_kinematics": 0,
        "input_rows": 0,
        "after_candidate_cuts": 0,
        "after_pt_range": 0,
        "after_role_mass_selection": 0,
        "after_finite_features": 0,
        "after_mc_range": 0,
        "per_variable_mc_range": {},
        "mc_range_stage": "after_finite_features_before_reservoir",
        "sampled_rows": 0,
        "reservoir_limit": limit,
        "upstream_count_scope": "native ROOT truth/kinematic filters precede sampler; unknown for prepared tables",
        "priority_sampling": "uniform per-row random priorities; retain max_per_class smallest priorities",
    }
    upstream_keys = {
        "raw_source_rows": "raw_rows",
        "after_source_role_filter": "after_role_filter",
        "after_source_finite_kinematics": "retained_rows",
        "source_nonfinite_kinematics": "nonfinite_kinematics",
    }
    columns = (
        set(config["training"]["features"])
        | set(config["training"]["candidate_cuts"])
        | {
            "mass_h3l",
            "_source_file",
            "_source_tree",
            "XPrimVtx",
            "YPrimVtx",
            "ZPrimVtx",
        }
        | set(support_ranges or {})
    )
    if config["training"]["pt_range"] is not None:
        columns.add("pt_h3l")
    for frame in iter_parquet(
        directory,
        columns=sorted(columns),
        optional_columns=["pt_h3l", "_entry"],
        batch_size=config["io"]["chunk_rows"],
    ):
        flow["input_rows"] += len(frame)
        source_counts = frame.attrs.get("source_counts", {})
        if hasattr(directory, "iter_batches") and set(upstream_keys.values()) <= set(
            source_counts
        ):
            for target, source in upstream_keys.items():
                flow[target] += int(source_counts[source])
        else:
            flow["source_counts_available"] = False
        selection = cut_mask(frame, config["training"]["candidate_cuts"])
        flow["after_candidate_cuts"] += int(selection.sum())
        pt_range = config["training"]["pt_range"]
        if pt_range is not None:
            require_columns(frame, ["pt_h3l"], "training pT selection")
            selection &= np.asarray(in_window(frame.pt_h3l, pt_range))
        flow["after_pt_range"] += int(selection.sum())
        frame = frame.loc[selection]
        low, high = config["training"]["signal_mass_window"]
        mass = frame.mass_h3l
        mass_mask = (
            ((mass < low) | (mass > high))
            if role == "background"
            else ((mass > low) & (mass < high))
        )
        frame = frame.loc[mass_mask].copy()
        flow["after_role_mass_selection"] += len(frame)
        features = config["training"]["features"]
        require_columns(frame, features, "BDT features")
        frame = frame.loc[np.isfinite(frame[features]).all(axis=1)].copy()
        flow["after_finite_features"] += len(frame)
        if support_ranges:
            from .training_support import mc_range_mask

            mask, per_variable = mc_range_mask(
                frame, support_ranges, return_cutflow=True
            )
            frame = frame.loc[mask].copy()
            for feature, batch_counts in per_variable.items():
                stats = flow["per_variable_mc_range"].setdefault(feature, {})
                for name, value in batch_counts.items():
                    stats[name] = stats.get(name, 0) + value
        count += len(frame)
        flow["after_mc_range"] += len(frame)
        frame["_priority"] = rng.random(len(frame))
        reservoir = pd.concat([reservoir, frame], ignore_index=True)
        if len(reservoir) > limit:
            reservoir = reservoir.nsmallest(limit, "_priority")
    if len(reservoir) < config["training"]["min_per_class"]:
        raise AnalysisError(
            f"Too few {role} candidates: {len(reservoir)}; need {config['training']['min_per_class']}"
        )
    flow["sampled_rows"] = len(reservoir)
    if not flow["source_counts_available"]:
        for name in upstream_keys:
            flow[name] = None
    sample = reservoir.drop(columns="_priority").reset_index(drop=True)
    return sample, (flow if return_cutflow else count)


def training_selection(config):
    """Resolved training input definition, distinct from final analysis/WP cuts."""
    settings = config["training"]
    return copy.deepcopy(
        {
            "common_candidate_cuts": settings["candidate_cuts"],
            "common_pt_range": settings["pt_range"],
            "selection_order": [
                "upstream signal truth",
                "upstream finite kinematics",
                "common candidate cuts",
                "common pT range",
                "strict MC signal mass window / data outside the same window",
                "finite BDT features",
                "bounded priority reservoir",
                "grouped train/validation split",
            ],
            "interval_policy": "cuts: min inclusive, max/abs_max exclusive; pT [low, high); training mass uses strict inequalities and excludes both endpoints",
            "finite_features": settings["features"],
            "source_finite_kinematics": ["mass_h3l", "pt_h3l", "ct", "cos_pa"],
            "signal": {
                "upstream_truth_flags": [
                    "HypIsReco",
                    "HypIsSignal",
                    "HypIsTwoBodyDecay",
                ],
                "truth_policy": "applied by native/previously prepared MC source, not re-evaluated by sampler",
                "mass_selection": {
                    "window": settings["signal_mass_window"],
                    "comparison": "low < mass < high",
                },
            },
            "background": {
                "excluded_mass_window": settings["signal_mass_window"],
                "comparison": "mass < low or mass > high; both endpoints excluded",
            },
            "reservoir_limit_per_class": settings["max_per_class"],
            "automatic_centrality_bins": None,
        }
    )


def train(
    signal,
    background,
    config,
    *,
    optimization_directory=None,
    qa_directory=None,
    model_id="global",
    input_cutflow=None,
    selection_metadata=None,
):
    """Use disjoint vertex groups for fitting and validation within each class."""
    t = config["training"]
    features = t["features"]
    selection = training_selection(config)
    if selection_metadata is not None:
        # Learned MC support/source/percentiles are supplied by the model-level
        # range resolver, without changing the sampler's explicit cut contract.
        selection.update(copy.deepcopy(selection_metadata))
    if t.get("mc_range_selection", {}).get("enabled", False):
        support = selection.get("mc_range_selection", {})
        if not (
            support.get("enabled") is True
            and isinstance(support.get("ranges"), dict)
            and (
                support.get("calibration_group_hash")
                or support.get("reference", {}).get("group_hash_sha256")
            )
        ):
            raise AnalysisError(
                "Enabled MC range selection requires resolved reference metadata before training"
            )
        from .training_support import mc_range_mask

        for role, sample in (("signal", signal), ("background", background)):
            if not mc_range_mask(sample, support["ranges"]).all():
                raise AnalysisError(
                    f"{role} training sample contains rows outside its resolved MC ranges"
                )
    input_cutflow = (
        copy.deepcopy(input_cutflow)
        if input_cutflow is not None
        else {
            role: {
                "source_counts_available": False,
                "raw_source_rows": None,
                "after_source_role_filter": None,
                "after_source_finite_kinematics": None,
                "source_nonfinite_kinematics": None,
                "input_rows": None,
                "after_candidate_cuts": None,
                "after_pt_range": None,
                "after_role_mass_selection": None,
                "after_finite_features": None,
                "sampled_rows": len(sample),
                "reservoir_limit": t["max_per_class"],
                "upstream_count_scope": "caller supplied selected sample; pre-sampling counts unavailable",
            }
            for role, sample in (("signal", signal), ("background", background))
        }
    )
    for role, sample in (("signal", signal), ("background", background)):
        if input_cutflow[role]["sampled_rows"] != len(sample):
            raise AnalysisError(
                f"{role} input cutflow sampled_rows differs from the supplied training sample"
            )
    train_parts, valid_parts = [], []
    split_summary = {}
    for label, sample in ((1, signal), (0, background)):
        sample = sample.copy()
        sample["label"] = label
        groups = training_groups(sample)
        if len(np.unique(groups)) < 4:
            raise AnalysisError(
                "At least four independent training vertex groups per class are required"
            )
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=t["validation_fraction"], random_state=config["seed"]
        )
        train_index, valid_index = next(splitter.split(sample, groups=groups))
        if len(train_index) < 2 or len(valid_index) < 2:
            raise AnalysisError("Training split too small")
        train_parts.append(sample.iloc[train_index])
        valid_parts.append(sample.iloc[valid_index])
        split_summary[str(label)] = {
            "train_rows": len(train_index),
            "validation_rows": len(valid_index),
            "train_groups": len(np.unique(groups[train_index])),
            "validation_groups": len(np.unique(groups[valid_index])),
            "group_overlap": len(set(groups[train_index]) & set(groups[valid_index])),
        }
    training = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(valid_parts, ignore_index=True)
    from .training_settings import (
        create_model_handler,
        model_parameters,
        efficiency_grid,
    )
    from .training_optimization import optimize_parameters, sample_weights

    parameters, optimization = optimize_parameters(
        training,
        config,
        model_parameters(t),
        features,
        directory=optimization_directory,
    )
    model = create_model_handler(t, config["seed"], parameters=parameters)
    y = training.label.to_numpy()
    weight = sample_weights(y, t["class_weight"])
    data = [training, y, validation, validation.label.to_numpy()]
    # Keep threshold comparisons identical to persisted NPZ/apply float64.
    # NumPy can round a float64 scalar cut down when comparing float32 scores.
    valid_score = np.asarray(
        model.train_test_model(
            data, return_prediction=True, output_margin=True, sample_weight=weight
        ),
        dtype=np.float64,
    )
    train_score = np.asarray(
        model.predict(training, output_margin=True), dtype=np.float64
    )
    signal_score = valid_score[validation.label.to_numpy() == 1]
    threshold = float(
        thresholds_from_efficiencies(
            validation.label, valid_score, [t["target_signal_efficiency"]]
        )[0]
    )
    fpr, tpr, thresholds = roc_curve(validation.label, valid_score)
    metrics = {
        "model_id": model_id,
        "backend": "hipe4ml.model_handler.ModelHandler[xgboost.XGBClassifier]",
        "model_handler": "hipe4ml.model_handler.ModelHandler",
        "score_output_margin": True,
        "score_domain": "raw_margin",
        "feature_scaling": "none; physical values passed unchanged to ModelHandler",
        "model_params": parameters,
        "random_state": config["seed"],
        "class_weight": t["class_weight"],
        "optuna": optimization,
        "features": features,
        "training_pt_range": t["pt_range"],
        "training_candidate_cuts": t["candidate_cuts"],
        "training_selection": selection,
        "input_cutflow": input_cutflow,
        "available_signal_candidates": input_cutflow["signal"]["after_finite_features"],
        "available_background_candidates": input_cutflow["background"][
            "after_finite_features"
        ],
        "available_candidate_definition": "finite-feature source candidates before reservoir and MC reference/support; see per-class cutflow for post-support counts",
        "available_after_mc_range_signal_candidates": input_cutflow["signal"].get(
            "after_mc_range"
        ),
        "available_after_mc_range_background_candidates": input_cutflow[
            "background"
        ].get("after_mc_range"),
        "sampled_signal_candidates": len(signal),
        "sampled_background_candidates": len(background),
        "threshold": threshold,
        "threshold_source": "held_out_signal_efficiency",
        "efficiency_scope": "held-out finite-feature MC reconstructed truth candidates after training selection and model pT scope; before final analysis cuts",
        "target_signal_efficiency": t["target_signal_efficiency"],
        "validation_signal_efficiency": float((signal_score >= threshold).mean()),
        "validation_background_efficiency": float(
            (valid_score[validation.label.to_numpy() == 0] >= threshold).mean()
        ),
        "auc_train": roc_auc_score(y, train_score),
        "auc_validation": roc_auc_score(validation.label, valid_score),
        "split": split_summary,
        "split_unit": "source-tree + primary-vertex surrogate",
        "limitations": [
            "Primary-vertex grouping is not a producer collision ID.",
            "Validation also sets the working point; use independent MC for final efficiency.",
            "Hyper/MCHyper domain differences require control-sample validation.",
        ],
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }
    grid = efficiency_grid(t)
    grid_scores, mapping_details = thresholds_from_efficiencies(
        validation.label, valid_score, grid, return_metadata=True
    )
    metrics["score_efficiency_array"] = {
        "efficiency": grid,
        "threshold": grid_scores,
        "actual_efficiency": np.array(
            [(signal_score >= score).mean() for score in grid_scores]
        ),
        "grid_definition": "unique(round(linspace(efficiency_min, efficiency_max, npoints_for_effi),3))",
        "requested_points": t["npoints_for_effi"],
        "validation_signal_rows": len(signal_score),
        "mapping": {**mapping_metadata(), **mapping_details},
    }
    for label, name in ((0, "background"), (1, "signal")):
        metrics[f"ks_{name}"] = ks_2samp(
            train_score[y == label], valid_score[validation.label.to_numpy() == label]
        ).pvalue
    bkg_mask = validation.label.to_numpy() == 0
    background_scores = valid_score[bkg_mask]
    background_mass = validation.mass_h3l[bkg_mask]
    metrics["background_score_mass_spearman"] = (
        float(spearmanr(background_mass, background_scores).statistic)
        if np.ptp(background_scores) > 0 and np.ptp(background_mass) > 0
        else np.nan
    )
    metrics["feature_importance"] = dict(
        zip(features, model.get_original_model().feature_importances_)
    )
    metrics["feature_importance_basis"] = (
        "XGBoost normalized gain; PDF SHAP diagnostics use the configured bounded held-out sample"
    )
    from .training_qa import build_training_qa

    metrics["qa"] = build_training_qa(
        training, validation, train_score, valid_score, features, t["qa_bins"]
    )
    if qa_directory is not None:
        from .training_qa import render_training_qa

        render_training_qa(qa_directory, model, data, metrics, config)
    validation = validation.assign(bdt_score=valid_score)
    return model, metrics, validation


def score_frame(frame, model, features):
    require_columns(frame, features, "BDT application")
    frame = frame.copy()
    finite = np.isfinite(frame[features]).all(axis=1).to_numpy()
    scores = np.full(len(frame), np.nan)
    if finite.any():
        scores[finite] = model.predict(frame.loc[finite, features], output_margin=True)
    frame["bdt_score"] = scores
    return frame
