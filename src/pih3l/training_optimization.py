"""Optional hyperparameter search on grouped folds within the training partition."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .errors import AnalysisError
from .io import write_json
from .training_settings import create_model_handler, search_space

LOG = logging.getLogger(__name__)


def sample_weights(labels, mode):
    """Balance each fitting partition independently; evaluation remains unweighted."""
    labels = np.asarray(labels)
    if mode is None:
        return np.ones(len(labels))
    if mode != "balanced":
        raise AnalysisError("training.class_weight must be balanced or null")
    counts = np.bincount(labels.astype(int), minlength=2)
    if (counts == 0).any():
        raise AnalysisError("Both classes are required in a training partition")
    return len(labels) / (2 * counts[labels.astype(int)])


def _index_digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def optimization_folds(training, config):
    """The outer held-out data never enter this function or any Optuna objective."""
    from .training import training_groups

    labels = training.label.to_numpy()
    groups = training_groups(training)
    count = config["training"]["optuna"]["cv_folds"]
    for label in (0, 1):
        if len(np.unique(groups[labels == label])) < count:
            raise AnalysisError(
                f"Optuna cv_folds={count} requires at least {count} training groups per class; lower cv_folds or increase the training sample"
            )
    splitter = StratifiedGroupKFold(
        n_splits=count, shuffle=True, random_state=config["seed"]
    )
    folds = list(splitter.split(training, labels, groups))
    records = []
    for index, (fit, valid) in enumerate(folds):
        if len(np.unique(labels[fit])) != 2 or len(np.unique(labels[valid])) != 2:
            raise AnalysisError(
                "Optuna grouped fold is missing a class; change cv_folds or increase independent training groups"
            )
        overlap = len(set(groups[fit]) & set(groups[valid]))
        if overlap:
            raise AnalysisError("Optuna train/validation group overlap")
        records.append(
            {
                "fold": index,
                "fit_rows": len(fit),
                "validation_rows": len(valid),
                "fit_class_counts": np.bincount(labels[fit], minlength=2),
                "validation_class_counts": np.bincount(labels[valid], minlength=2),
                "fit_groups": len(np.unique(groups[fit])),
                "validation_groups": len(np.unique(groups[valid])),
                "group_overlap": overlap,
                "fit_index_sha256": _index_digest(fit),
                "validation_index_sha256": _index_digest(valid),
            }
        )
    return folds, {
        "split_unit": "source-tree + primary-vertex surrogate",
        "scope": "inner cross-validation on outer training partition only",
        "final_held_out_used": False,
        "training_rows": len(training),
        "groups": len(np.unique(groups)),
        "folds": records,
    }


def _suggest(trial, parameter, specification):
    kind = specification["type"]
    if kind == "categorical":
        return trial.suggest_categorical(parameter, specification["choices"])
    kwargs = {"log": specification.get("log", False)}
    if "step" in specification:
        kwargs["step"] = specification["step"]
    if kind == "int":
        return trial.suggest_int(
            parameter, specification["low"], specification["high"], **kwargs
        )
    return trial.suggest_float(
        parameter, specification["low"], specification["high"], **kwargs
    )


def optimize_parameters(training, config, base_parameters, features, directory=None):
    """Select mean CV AUC, then return parameters for a fresh final model fit."""
    settings = config["training"]
    if not settings["use_optuna"]:
        return dict(base_parameters), {"enabled": False}
    try:
        import optuna
    except ImportError as error:
        raise AnalysisError(
            "training.use_optuna=true requires Optuna in the selected Python environment; disable it or provide the optional training dependency"
        ) from error
    options = settings["optuna"]
    folds, split = optimization_folds(training, config)
    space = search_space(settings)
    labels = training.label.to_numpy()

    def objective(trial):
        parameters = dict(base_parameters)
        parameters.update(
            {name: _suggest(trial, name, spec) for name, spec in space.items()}
        )
        scores = []
        for index, (fit, valid) in enumerate(folds):
            model = create_model_handler(
                settings, config["seed"], features=features, parameters=parameters
            )
            model.fit(
                training.iloc[fit],
                labels[fit],
                sample_weight=sample_weights(labels[fit], settings["class_weight"]),
            )
            auc = float(
                roc_auc_score(
                    labels[valid],
                    model.predict(training.iloc[valid], output_margin=True),
                )
            )
            scores.append(auc)
            trial.set_user_attr("fold_auc", scores)
            trial.report(float(np.mean(scores)), index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(
        seed=config["seed"], n_startup_trials=options["n_startup_trials"]
    )
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=options["n_startup_trials"],
            n_warmup_steps=options["n_warmup_steps"],
        )
        if options["pruner"] == "median"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction=options["direction"], sampler=sampler, pruner=pruner
    )
    LOG.info(
        "Optuna %s: %s trials, %s grouped folds, timeout=%ss",
        "HIPE4ML/XGBoost",
        options["n_trials"],
        len(folds),
        options["timeout"],
    )
    study.optimize(
        objective,
        n_trials=options["n_trials"],
        timeout=options["timeout"] or None,
        n_jobs=options["n_jobs"],
        show_progress_bar=False,
        gc_after_trial=True,
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise AnalysisError(
            "Optuna completed no successful trials; increase timeout or review the search space"
        )
    best = dict(base_parameters)
    best.update(study.best_params)
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "fold_auc": trial.user_attrs.get("fold_auc", []),
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration else None
            ),
        }
        for trial in study.trials
    ]
    summary = {
        "enabled": True,
        "version": optuna.__version__,
        "sampler": "TPESampler",
        "sampler_seed": config["seed"],
        "metric": "mean_inner_group_cv_auc",
        "direction": options["direction"],
        "requested_trials": options["n_trials"],
        "trial_count": len(trials),
        "completed_trials": len(completed),
        "timeout": options["timeout"],
        "n_jobs": options["n_jobs"],
        "pruner": options["pruner"],
        "search_space": space,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": best,
        "optimized_params": study.best_params,
        "split": split,
        "trials": trials,
        "limitations": [
            "AUC optimization is not Hyper mass-fit significance optimization.",
            "Final held-out samples are reserved for QA/WP; final efficiency still requires independent MC.",
            "Parallel Optuna trials and wall-time limits can change the realized search despite a fixed seed.",
        ],
    }
    if directory is not None:
        directory = Path(directory)
        write_json(directory / "summary.json", summary)
        write_json(directory / "split.json", split)
        rows = [
            {
                "number": trial["number"],
                "state": trial["state"],
                "value": trial["value"],
                "duration_seconds": trial["duration_seconds"],
                "fold_auc": json.dumps(trial["fold_auc"]),
                **{f"param_{name}": value for name, value in trial["params"].items()},
            }
            for trial in trials
        ]
        pd.DataFrame(rows).to_csv(directory / "trials.csv", index=False)
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history

        qa_directory = directory.parent / "qa"
        qa_directory.mkdir(parents=True, exist_ok=True)
        axis = plot_optimization_history(study)
        try:
            axis.figure.savefig(
                qa_directory / "optuna_training.pdf", format="pdf", bbox_inches="tight"
            )
        finally:
            plt.close(axis.figure)
    return best, summary
