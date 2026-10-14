"""Explicit pT model scopes, independently verified artifacts, and inference routing."""

from __future__ import annotations

import copy
import importlib.metadata
import platform
import pickle
import shutil
from pathlib import Path

import numpy as np
import sklearn

from .config import digest, merge
from .errors import AnalysisError
from .io import read_json, sha256_file, write_json
from .schema import require_columns
from .selection import in_window
from .training_qa import TRAINING_QA_ARTIFACTS, OPTUNA_TRAINING_QA_ARTIFACTS

CORE_ARTIFACTS = {
    "model_handler.pkl",
    "metrics.json",
    "validation_scores.npz",
    "score_efficiency_array.txt",
} | set(TRAINING_QA_ARTIFACTS)
OPTIMIZATION_ARTIFACTS = {
    "optimization/summary.json",
    "optimization/trials.csv",
    "optimization/split.json",
} | set(OPTUNA_TRAINING_QA_ARTIFACTS)


def _training_options(training, override=None):
    """Resolve per-model tuning options without leaking between pT intervals."""
    override = override or {}
    parameters = {
        **training.get("model_params", {}),
        **override.get("model_params", {}),
    }
    optuna = merge(training.get("optuna", {}), override.get("optuna", {}))
    if "search_space" in override.get("optuna", {}):
        optuna["search_space"] = copy.deepcopy(override["optuna"]["search_space"])
    return {
        "class_weight": training.get("class_weight", "balanced"),
        "model_params": parameters,
        "use_optuna": override.get("use_optuna", training.get("use_optuna", False)),
        "optuna": optuna,
        **{
            name: override.get(name, training[name])
            for name in ("efficiency_min", "efficiency_max", "npoints_for_effi")
        },
    }


def model_specs(config):
    """Resolve global / restricted shared / independent interval training."""
    training = config["training"]
    bins = training["pt_bins"]
    if not bins:
        return [
            {
                "id": "global",
                "pt_range": training["pt_range"],
                "features": training["features"],
                "candidate_cuts": training["candidate_cuts"],
                **_training_options(training),
            }
        ]
    result = []
    for index, value in enumerate(bins):
        item = (
            value
            if isinstance(value, dict)
            else {"id": f"pt{index}", "pt_range": value}
        )
        result.append(
            {
                "id": item["id"],
                "pt_range": item["pt_range"],
                "features": item.get("features", training["features"]),
                "candidate_cuts": merge(
                    training["candidate_cuts"], item.get("candidate_cuts", {})
                ),
                **_training_options(training, item),
            }
        )
    return result


def feature_union(config):
    return sorted(
        {feature for item in model_specs(config) for feature in item["features"]}
    )


def scoped_config(config, spec):
    scoped = copy.deepcopy(config)
    scoped["training"].update(
        {
            name: copy.deepcopy(spec[name])
            for name in (
                "pt_range",
                "features",
                "candidate_cuts",
                "class_weight",
                "model_params",
                "use_optuna",
                "optuna",
                "efficiency_min",
                "efficiency_max",
                "npoints_for_effi",
            )
            if name in spec
        }
    )
    scoped["training"]["pt_bins"] = []
    scoped["training"]["retrain_bins"] = []
    return scoped


def _source_digest(directory):
    # Hash only candidate data: changes to the very large Femto source cannot
    # invalidate an already trained candidate model.
    if hasattr(directory, "fingerprint"):
        return directory.fingerprint()
    return {
        path.name: sha256_file(path)
        for path in sorted(Path(directory).glob("*.parquet"))
    }


def _model_signature(spec, config, sources):
    training = copy.deepcopy(scoped_config(config, spec)["training"])
    for name in ("pt_bins", "retrain_bins"):
        training.pop(name)
    signature_spec = copy.deepcopy(spec)
    if not training.get("use_optuna", False):
        training.pop("optuna", None)
        signature_spec.pop("optuna", None)
    modules = [
        Path(__file__),
        Path(__file__).with_name("training.py"),
        Path(__file__).with_name("training_support.py"),
        Path(__file__).with_name("selection.py"),
        Path(__file__).with_name("training_settings.py"),
        Path(__file__).with_name("training_optimization.py"),
        Path(__file__).with_name("io.py"),
        Path(__file__).with_name("schema.py"),
        Path(__file__).with_name("dataset.py"),
        Path(__file__).with_name("physics.py"),
        Path(__file__).with_name("execution.py"),
        Path(__file__).with_name("validation_scores.py"),
        Path(__file__).with_name("efficiency.py"),
        Path(__file__).with_name("training_qa.py"),
        Path(__file__).with_name("root_io.py"),
    ]
    optional_environment = {}
    from .root_io import load_root

    optional_environment["ROOT"] = load_root().gROOT.GetVersion()
    optional_environment["hipe4ml"] = _optional_version("hipe4ml")
    from .efficiency import mapping_metadata

    optional_environment["hipe4ml_source"] = mapping_metadata()["implementation_sha256"]
    from .execution import hipe4ml_model_source

    optional_environment["hipe4ml_model_handler_source"] = hipe4ml_model_source()
    optional_environment["xgboost"] = _optional_version("xgboost")
    optional_environment["matplotlib"] = _optional_version("matplotlib")
    optional_environment["shap"] = _optional_version("shap")
    from hipe4ml import plot_utils

    optional_environment["hipe4ml_plot_utils_source"] = sha256_file(
        Path(plot_utils.__file__)
    )
    if training.get("use_optuna", False):
        optional_environment["optuna"] = _optional_version("optuna")
    return digest(
        {
            "spec": signature_spec,
            "training": training,
            "seed": config["seed"],
            "sources": sources,
            "code": {path.name: sha256_file(path) for path in modules},
            "sklearn": sklearn.__version__,
            "environment": {
                "python": platform.python_version(),
                **optional_environment,
                **{
                    name: importlib.metadata.version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scipy",
                        "pyarrow",
                        "uproot",
                    )
                },
            },
        }
    )


def _optional_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _verified_previous(directory, signature):
    manifest = Path(directory) / "model_manifest.json"
    if not manifest.exists():
        return False
    saved = read_json(manifest)
    if saved.get("model_format") != "hipe4ml_model_handler_pickle_v1":
        return False
    if saved.get("signature") != signature:
        return False
    artifacts = saved.get("artifacts", {})
    expected = CORE_ARTIFACTS | (
        OPTIMIZATION_ARTIFACTS
        if saved.get("spec", {}).get("use_optuna", False)
        else set()
    )
    if set(artifacts) != expected:
        return False
    return all(
        (Path(directory) / name).is_file()
        and sha256_file(Path(directory) / name) == checksum
        for name, checksum in artifacts.items()
    )


def train_models(signal_directory, background_directory, destination, config):
    """Train only missing, changed or explicitly forced model scopes."""
    from .training import sample_candidates, train
    from .training_support import prepare_mc_reference

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    specs = model_specs(config)
    sources = {
        "signal": _source_digest(signal_directory),
        "background": _source_digest(background_directory),
    }
    previous = Path(config["output"]) / "training"
    entries = []
    for spec in specs:
        relative = "." if not config["training"]["pt_bins"] else f"models/{spec['id']}"
        directory = destination / relative
        signature = _model_signature(spec, config, sources)
        reuse = spec["id"] not in config["training"][
            "retrain_bins"
        ] and _verified_previous(previous / relative, signature)
        directory.mkdir(parents=True, exist_ok=True)
        if not spec["use_optuna"] and (directory / "optimization").exists():
            shutil.rmtree(directory / "optimization")
        if reuse:
            saved = read_json(previous / relative / "model_manifest.json")
            for name in sorted(set(saved["artifacts"]) | {"model_manifest.json"}):
                source, target = previous / relative / name, directory / name
                if source.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        else:
            scoped = scoped_config(config, spec)
            signal, signal_flow = sample_candidates(
                signal_directory, "signal", scoped, return_cutflow=True
            )
            sampled_before_reference = len(signal)
            signal, support = prepare_mc_reference(signal, scoped)
            signal_flow.update(
                sampled_before_mc_reference=sampled_before_reference,
                mc_reference_rows=0,
            )
            if support.get("enabled"):
                counts = support["signal_counts"]
                signal_flow.update(
                    mc_reference_rows=counts["reference_removed"],
                    after_mc_reference=counts["after_reference"],
                    after_mc_range=len(signal),
                    sampled_rows=len(signal),
                    per_variable_mc_range=counts["per_variable_mc_range"],
                    mc_range_stage="after_reservoir_and_reference_group_removal",
                    mc_reference_audit=copy.deepcopy(support),
                )
            background, background_flow = sample_candidates(
                background_directory,
                "background",
                scoped,
                return_cutflow=True,
                support_ranges=support.get("ranges", {}),
            )
            model, metrics, validation = train(
                signal,
                background,
                scoped,
                optimization_directory=directory / "optimization",
                qa_directory=directory,
                model_id=spec["id"],
                input_cutflow={"signal": signal_flow, "background": background_flow},
                selection_metadata={"mc_range_selection": support},
            )
            model.dump_model_handler(str(directory / "model_handler.pkl"))
            write_json(directory / "metrics.json", metrics)
            curve = metrics["score_efficiency_array"]
            np.savetxt(
                directory / "score_efficiency_array.txt",
                np.column_stack((curve["threshold"], curve["efficiency"])),
            )
            from .validation_scores import write_validation_scores

            write_validation_scores(directory, validation)
            artifacts = CORE_ARTIFACTS | (
                OPTIMIZATION_ARTIFACTS if spec["use_optuna"] else set()
            )
            missing = [name for name in artifacts if not (directory / name).is_file()]
            if missing:
                raise AnalysisError(
                    f"Training did not produce required artifacts for {spec['id']}: {sorted(missing)}"
                )
            write_json(
                directory / "model_manifest.json",
                {
                    "model_format": "hipe4ml_model_handler_pickle_v1",
                    "signature": signature,
                    "spec": spec,
                    "artifacts": {
                        name: sha256_file(directory / name)
                        for name in sorted(artifacts)
                    },
                },
            )
        entries.append(
            {
                **spec,
                "relative_path": relative,
                "signature": signature,
                "status": "verified_reuse" if reuse else "trained",
            }
        )
    registry = {
        "mode": "pt_bins" if config["training"]["pt_bins"] else "global",
        "scope_policy": "half-open pT intervals; uncovered candidates receive NaN score and are excluded",
        "models": entries,
    }
    write_json(destination / "registry.json", registry)
    if registry["mode"] == "pt_bins":
        write_json(
            destination / "metrics.json",
            {
                "mode": "pt_bins",
                "models": entries,
                "global_auc": None,
                "status": "independent_model_metrics",
            },
        )
    return registry


def load_model_bundle(model_path):
    """Load one official HIPE handler plus its verified, separate metadata.

    The returned dictionary is transient consumer state, never another stored
    classifier bundle. Legacy joblib artifacts require explicit retraining.
    """
    from hipe4ml.model_handler import ModelHandler

    model_path = Path(model_path)
    if model_path.name != "model_handler.pkl":
        raise AnalysisError(
            "Expected HIPE4ML model_handler.pkl; legacy model.joblib artifacts require retraining"
        )
    metrics_path = model_path.parent / "metrics.json"
    if not model_path.is_file() or not metrics_path.is_file():
        raise AnalysisError(f"Missing HIPE4ML handler or metrics: {model_path}")
    manifest_path = model_path.parent / "model_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("model_format") != "hipe4ml_model_handler_pickle_v1":
            raise AnalysisError(
                f"Unsupported model manifest format: {manifest_path}; retrain the model"
            )
        artifacts = manifest.get("artifacts", {})
        for path in (model_path, metrics_path):
            if artifacts.get(path.name) != sha256_file(path):
                raise AnalysisError(
                    f"Model artifact failed checksum verification: {path}"
                )
    metrics = read_json(metrics_path)
    if (
        metrics.get("score_output_margin") is not True
        or metrics.get("score_domain") != "raw_margin"
    ):
        raise AnalysisError(
            f"Model has an incompatible score domain: {metrics_path}; retrain with output_margin=True"
        )
    required = {"features", "training_pt_range", "model_id"}
    if not required <= set(metrics):
        raise AnalysisError(
            f"HIPE4ML model metrics missing {sorted(required - set(metrics))}: {metrics_path}"
        )
    handler = ModelHandler()
    try:
        handler.load_model_handler(str(model_path))
    except (
        AttributeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise AnalysisError(
            f"Invalid HIPE4ML ModelHandler artifact: {model_path}"
        ) from error
    features = handler.get_training_columns()
    if not features or list(features) != metrics["features"]:
        raise AnalysisError(
            f"Handler training columns differ from metrics feature order: {model_path}"
        )
    return {
        "model": handler,
        "features": list(features),
        "training_metrics": metrics,
        "pt_range": metrics["training_pt_range"],
        "model_id": metrics["model_id"],
        "mc_range_selection": metrics.get("training_selection", {}).get(
            "mc_range_selection", {}
        ),
    }


def load_model_registry(training_directory):
    """Load official handlers once before bounded streaming application."""
    training_directory = Path(training_directory)
    path = training_directory / "registry.json"
    registry = (
        read_json(path)
        if path.exists()
        else {
            "mode": "global",
            "models": [{"id": "global", "pt_range": None, "relative_path": "."}],
        }
    )
    loaded = copy.deepcopy(registry)
    for spec in loaded["models"]:
        directory = training_directory / spec["relative_path"]
        if path.exists() and not _verified_previous(directory, spec["signature"]):
            raise AnalysisError(
                f"Model artifacts failed checksum verification: {spec['id']}"
            )
        model_path = directory / "model_handler.pkl"
        if not model_path.is_file():
            if (directory / "model.joblib").exists():
                raise AnalysisError(
                    f"Legacy model.joblib is unsupported; retrain {spec['id']} with HIPE4ML ModelHandler"
                )
            raise AnalysisError(
                f"Missing model artifact for {spec['id']}: {model_path}"
            )
        spec["bundle"] = load_model_bundle(model_path)
        if path.exists() and (
            spec["id"] != spec["bundle"]["model_id"]
            or spec["pt_range"] != spec["bundle"]["pt_range"]
        ):
            raise AnalysisError(
                f"Model registry and handler metadata scope differ: {spec['id']}"
            )
        if not path.exists():
            # A standalone handler retains its scope in metrics; loss of the
            # routing index must never broaden its application range.
            spec["pt_range"] = spec["bundle"].get("pt_range")
            spec["id"] = spec["bundle"].get("model_id", spec["id"])
    return loaded


def score_models(frame, registry):
    """Predict each row with exactly one model, retaining explicit scope status."""
    from .training import score_frame
    from .training_support import mc_range_mask

    result = frame.copy()
    result["bdt_score"] = np.nan
    result["bdt_model_id"] = ""
    result["bdt_in_scope"] = False
    assigned = np.zeros(len(frame), dtype=bool)
    for spec in registry["models"]:
        scope = np.ones(len(frame), dtype=bool)
        if spec["pt_range"] is not None:
            require_columns(frame, ["pt_h3l"], "model pT routing")
            scope = np.asarray(in_window(frame.pt_h3l, spec["pt_range"]))
        bundle = spec["bundle"]
        scope &= mc_range_mask(
            frame, bundle.get("mc_range_selection", {}).get("ranges", {})
        )
        if (scope & assigned).any():
            raise AnalysisError("Overlapping model application scopes")
        assigned |= scope
        if not scope.any():
            continue
        scored = score_frame(frame.loc[scope], bundle["model"], bundle["features"])
        result.loc[scope, "bdt_score"] = scored.bdt_score.to_numpy()
        result.loc[scope, "bdt_model_id"] = spec["id"]
        result.loc[scope, "bdt_in_scope"] = True
    return result
