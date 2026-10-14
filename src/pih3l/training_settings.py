"""Single HIPE4ML/XGBoost model definition and strictly typed tuning domains.

Group splitting, class weights and raw-margin QA belong to the training
workflow. The factory centralizes the handler contract used for train/apply.
"""

from __future__ import annotations

import copy

import numpy as np

from .errors import AnalysisError

XGB_DEFAULTS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "max_depth": 6,
    "gamma": 2.0,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "min_child_weight": 3.0,
    "subsample": 0.7,
    "colsample_bytree": 0.73,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
    "n_jobs": 4,
    "eval_metric": "auc",
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "verbosity": 0,
}

# Canonical gamma only: gamma and min_split_loss are aliases in XGBoost.
XGB_SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 8},
    "learning_rate": {"type": "float", "low": 0.015, "high": 0.12, "log": True},
    "n_estimators": {"type": "int", "low": 300, "high": 1400, "step": 100},
    "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0},
    "subsample": {"type": "float", "low": 0.60, "high": 0.95},
    "colsample_bytree": {"type": "float", "low": 0.60, "high": 1.0},
    "gamma": {"type": "float", "low": 0.0, "high": 5.0},
    "reg_alpha": {"type": "float", "low": 0.0, "high": 2.0},
    "reg_lambda": {"type": "float", "low": 0.5, "high": 5.0},
}

INTEGER_PARAMETERS = {"max_depth", "n_estimators", "max_bin", "n_jobs", "verbosity"}
CATEGORICAL_PARAMETERS = {"objective", "booster", "tree_method", "eval_metric"}


def model_parameters(training):
    """Resolve the sole XGBoost engine's defaults plus explicit model_params."""
    if "backend" in training:
        raise AnalysisError(
            "training.backend has been removed: training uses HIPE4ML/XGBoost"
        )
    result = copy.deepcopy(XGB_DEFAULTS)
    result.update(training.get("model_params", {}))
    return result


def search_space(training):
    return copy.deepcopy(training["optuna"]["search_space"] or XGB_SEARCH_SPACE)


def _parameter(name, value):
    from .config import _number

    prefix = f"training.model_params.{name}"
    if name in {"random_state", "seed"}:
        raise AnalysisError(
            "Set the top-level seed, not model_params.seed/random_state"
        )
    if name == "min_split_loss":
        raise AnalysisError(
            "Use canonical gamma; min_split_loss is a duplicate XGBoost alias"
        )
    if name not in XGB_DEFAULTS:
        raise AnalysisError(f"Unknown XGBoost model parameter: {name}")
    if name in CATEGORICAL_PARAMETERS:
        choices = {
            "objective": {"binary:logistic"},
            "booster": {"gbtree"},
            "tree_method": {"hist", "approx", "exact"},
            "eval_metric": {"auc", "logloss"},
        }
        if not isinstance(value, str) or value not in choices[name]:
            raise AnalysisError(f"{prefix} must be one of {sorted(choices[name])}")
        return
    integer_min = {
        "max_depth": 0,
        "n_estimators": 1,
        "max_bin": 2,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if name in integer_min:
        _number(
            value,
            prefix,
            minimum=integer_min[name],
            maximum=3 if name == "verbosity" else None,
            integer=True,
        )
        return
    if name in {
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "colsample_bynode",
    }:
        _number(value, prefix, minimum=0, maximum=1)
        if value <= 0:
            raise AnalysisError(f"{prefix} must be positive")
        return
    _number(value, prefix, minimum=0)


def validate_model_parameters(training, parameters):
    from .config import _mapping

    _mapping(parameters, "training.model_params")
    for name, value in parameters.items():
        _parameter(name, value)
    if parameters.get("tree_method") == "exact":
        if parameters.get("colsample_bynode", 1) != 1:
            raise AnalysisError(
                "XGBoost exact tree_method does not support colsample_bynode"
            )
        if parameters.get("max_depth") == 0:
            raise AnalysisError(
                "XGBoost exact tree_method requires positive max_depth; unlimited depth is supported by hist/approx only"
            )


def _distribution(name, specification, training):
    from .config import _mapping, _number

    prefix = f"training.optuna.search_space.{name}"
    _mapping(
        specification,
        prefix,
        allowed={"type", "low", "high", "step", "log", "choices"},
        required={"type"},
    )
    kind = specification["type"]
    if name in INTEGER_PARAMETERS and kind == "float":
        raise AnalysisError(
            f"{prefix}: integer model parameters require int or categorical distributions"
        )
    if name in CATEGORICAL_PARAMETERS and kind != "categorical":
        raise AnalysisError(
            f"{prefix}: this model parameter requires a categorical distribution"
        )
    if kind == "categorical":
        if set(specification) != {"type", "choices"}:
            raise AnalysisError(
                f"{prefix}: categorical distributions require only type/choices"
            )
        values = specification["choices"]
        if not isinstance(values, (list, tuple)) or not values:
            raise AnalysisError(f"{prefix}.choices must be nonempty")
        keys = []
        for value in values:
            _parameter(name, value)
            candidate = model_parameters(training)
            candidate[name] = value
            validate_model_parameters(training, candidate)
            keys.append((type(value).__name__, repr(value)))
        if len(keys) != len(set(keys)):
            raise AnalysisError(f"{prefix}.choices must be unique")
        return
    if (
        not isinstance(kind, str)
        or kind not in {"int", "float"}
        or not {"low", "high"} <= set(specification)
        or "choices" in specification
    ):
        raise AnalysisError(f"{prefix}: use int/float low/high or categorical choices")
    for key in ("low", "high"):
        _number(specification[key], f"{prefix}.{key}", integer=kind == "int")
        _parameter(name, specification[key])
        candidate = model_parameters(training)
        candidate[name] = specification[key]
        validate_model_parameters(training, candidate)
    low, high = specification["low"], specification["high"]
    if low > high:
        raise AnalysisError(f"{prefix}: low must not exceed high")
    log = specification.get("log", False)
    if not isinstance(log, bool):
        raise AnalysisError(f"{prefix}.log must be boolean")
    if log and low <= 0:
        raise AnalysisError(f"{prefix}: log distributions require positive low")
    step = specification.get("step", 1 if kind == "int" else None)
    if step is not None:
        _number(step, f"{prefix}.step", minimum=0, integer=kind == "int")
        if step <= 0 or (log and (kind == "float" or step != 1)):
            raise AnalysisError(
                f"{prefix}: positive step is required; log supports int step=1 only"
            )
        units = (high - low) / step
        if not np.isclose(units, round(units), atol=1e-9, rtol=0):
            raise AnalysisError(f"{prefix}: step must exactly partition [low,high]")
    if kind == "int" and step is None:
        raise AnalysisError(f"{prefix}: int step cannot be null")


def efficiency_grid(training):
    """Spectrum-compatible dense grid: linspace, three decimals, unique values."""
    return np.unique(
        np.round(
            np.linspace(
                training["efficiency_min"],
                training["efficiency_max"],
                training["npoints_for_effi"],
            ),
            3,
        )
    )


def validate_training_settings(training):
    from .config import _mapping, _number

    if training["class_weight"] is not None and training["class_weight"] != "balanced":
        raise AnalysisError("training.class_weight must be balanced or null")
    if not isinstance(training["use_optuna"], bool):
        raise AnalysisError("training.use_optuna must be boolean")
    _number(
        training["qa_shap_max_per_class"],
        "training.qa_shap_max_per_class",
        minimum=1,
        integer=True,
    )
    for key in ("efficiency_min", "efficiency_max"):
        _number(training[key], f"training.{key}", minimum=0, maximum=1)
    if not 0 < training["efficiency_min"] < training["efficiency_max"] <= 1:
        raise AnalysisError(
            "Training efficiency bounds must satisfy 0 < min < max <= 1"
        )
    _number(
        training["npoints_for_effi"],
        "training.npoints_for_effi",
        minimum=2,
        integer=True,
    )
    grid = efficiency_grid(training)
    if len(grid) < 2 or grid[0] <= 0:
        raise AnalysisError(
            "Rounded training efficiency grid must contain at least two positive points"
        )
    _mapping(training["model_params"], "training.model_params")
    validate_model_parameters(training, model_parameters(training))
    options = training["optuna"]
    _mapping(
        options,
        "training.optuna",
        allowed={
            "metric",
            "direction",
            "n_trials",
            "timeout",
            "cv_folds",
            "n_jobs",
            "n_startup_trials",
            "pruner",
            "n_warmup_steps",
            "search_space",
        },
        required={
            "metric",
            "direction",
            "n_trials",
            "timeout",
            "cv_folds",
            "n_jobs",
            "n_startup_trials",
            "pruner",
            "n_warmup_steps",
            "search_space",
        },
    )
    if options["metric"] != "auc" or options["direction"] != "maximize":
        raise AnalysisError("Optuna supports metric: auc and direction: maximize")
    for key, minimum in (
        ("n_trials", 1),
        ("cv_folds", 2),
        ("n_jobs", 1),
        ("n_startup_trials", 0),
        ("n_warmup_steps", 0),
    ):
        _number(options[key], f"training.optuna.{key}", minimum=minimum, integer=True)
    if options["timeout"] is not None:
        _number(options["timeout"], "training.optuna.timeout", minimum=0)
        if options["timeout"] <= 0:
            raise AnalysisError("training.optuna.timeout must be positive or null")
    if not isinstance(options["pruner"], str) or options["pruner"] not in {
        "median",
        "none",
    }:
        raise AnalysisError("training.optuna.pruner must be median or none")
    _mapping(options["search_space"], "training.optuna.search_space")
    for name, specification in search_space(training).items():
        _distribution(name, specification, training)
    space, parameters = search_space(training), model_parameters(training)

    def values(name):
        domain = space.get(name)
        if domain is None:
            return [parameters[name]]
        return (
            domain["choices"]
            if domain["type"] == "categorical"
            else [domain["low"], domain["high"]]
        )

    if "exact" in values("tree_method"):
        if 0 in values("max_depth"):
            raise AnalysisError(
                "Optuna search space combines exact tree_method with unlimited max_depth"
            )
        if any(value != 1 for value in values("colsample_bynode")):
            raise AnalysisError(
                "Optuna search space combines exact tree_method with unsupported colsample_bynode"
            )


def create_model_handler(training, seed, features=None, parameters=None):
    """Create the HIPE4ML handler used by every training/apply path."""
    resolved = model_parameters(training)
    if parameters is not None:
        resolved.update(parameters)
    validate_model_parameters(training, resolved)
    try:
        from hipe4ml.model_handler import ModelHandler
        from xgboost import XGBClassifier
    except ImportError as error:
        raise AnalysisError(
            "HIPE4ML and XGBoost are required for training; install the project's declared dependencies"
        ) from error
    return ModelHandler(
        XGBClassifier(**resolved, random_state=seed),
        training_columns=list(features or training["features"]),
        model_params=resolved,
    )
