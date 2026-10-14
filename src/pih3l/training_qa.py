"""HIPE4ML training plots and bounded numerical QA archives."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .errors import AnalysisError
from .io import json_safe, write_json
from .root_io import RootArchive, load_root

TRAINING_QA_ARTIFACTS = tuple(
    "qa/" + name
    for name in (
        "training.root",
        "index.json",
        "feature_distributions.pdf",
        "correlations_signal.pdf",
        "correlations_background.pdf",
        "bdt_output.pdf",
        "bdt_test_score.pdf",
        "roc_train_test.pdf",
        "efficiency_vs_score.pdf",
        "feature_importance.pdf",
        "feature_importance_shap.pdf",
    )
)
OPTUNA_TRAINING_QA_ARTIFACTS = ("qa/optuna_training.pdf",)


def _hist(values, edges):
    values = np.asarray(values, dtype=float)
    counts = np.histogram(values[np.isfinite(values)], edges)[0]
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def build_training_qa(
    training_dataframe,
    validation_dataframe,
    train_scores,
    validation_scores,
    features,
    bins,
):
    """Summarize sampled candidates without retaining any candidate table.

    Feature distributions include mass as QA only; correlations use model inputs.
    Class score comparisons and ROC preserve the independent outer split.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise AnalysisError("Training QA histogram bins must be an integer >= 2")
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise AnalysisError("Training QA features must be nonempty and unique")
    frames = {"train": training_dataframe, "test": validation_dataframe}
    scores = {
        "train": np.asarray(train_scores, float),
        "test": np.asarray(validation_scores, float),
    }
    for split, data in frames.items():
        if "label" not in data or not set(features + ["mass_h3l"]) <= set(data):
            raise AnalysisError(
                f"Training QA {split} is missing label, mass QA or model features"
            )
        if len(scores[split]) != len(data) or not np.isfinite(scores[split]).all():
            raise AnalysisError(
                f"Training QA {split} scores must be finite and match rows"
            )
        if not set(data.label.unique()) <= {0, 1}:
            raise AnalysisError(f"Training QA {split} labels must be binary")
    combined = pd.concat(list(frames.values()), ignore_index=True)
    distributions = {}
    for feature in dict.fromkeys(features + ["mass_h3l"]):
        values = combined[feature].to_numpy(float)
        finite = values[np.isfinite(values)]
        low, high = (
            (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        )
        if low == high:
            width = max(abs(low) * 0.01, 0.01)
            low, high = low - width, high + width
        edges = np.linspace(low, high, bins + 1)
        distributions[feature] = {
            "edges": edges.tolist(),
            **{
                name: _hist(values[combined.label.to_numpy() == label], edges)["counts"]
                for label, name in ((1, "signal"), (0, "background"))
            },
            "finite_rows": int(len(finite)),
        }
    correlation_features = features
    correlations = {"features": correlation_features}
    for label, name in ((1, "signal"), (0, "background")):
        values = combined.loc[combined.label == label, correlation_features].replace(
            [np.inf, -np.inf], np.nan
        )
        matrix = values.corr(method="pearson", min_periods=2).to_numpy(float)
        correlations[name] = {"values": matrix, "valid": np.isfinite(matrix)}
    combined_scores = np.concatenate(list(scores.values()))
    low, high = float(combined_scores.min()), float(combined_scores.max())
    if low == high:
        width = max(abs(low) * 0.01, 0.01)
        low, high = low - width, high + width
    edges = np.linspace(low, high, bins + 1)
    score_distributions, rocs = {}, {}
    for split, data in frames.items():
        labels = data.label.to_numpy()
        for label, name in ((1, "signal"), (0, "background")):
            score_distributions[f"{split}_{name}"] = _hist(
                scores[split][labels == label], edges
            )
        if len(np.unique(labels)) == 2:
            fpr, tpr, thresholds = roc_curve(labels, scores[split])
            rocs[split] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": float(roc_auc_score(labels, scores[split])),
            }
        else:
            rocs[split] = {"fpr": [], "tpr": [], "thresholds": [], "auc": None}
    score_distributions["test_all"] = _hist(scores["test"], edges)
    return json_safe(
        {
            "schema_version": 1,
            "scope": "Selected sampled train + outer held-out candidates; histograms are not full-period yields",
            "feature_distributions": distributions,
            "correlations": correlations,
            "score_distributions": score_distributions,
            "roc": rocs,
            "rows": {split: len(data) for split, data in frames.items()},
        }
    )


def _qa_index(handler, data, metrics, config):
    """Describe each canonical QA artifact and its statistical scope."""
    settings = config["training"]
    features = list(handler.get_training_columns())
    columns = list(dict.fromkeys(features + ["mass_h3l"]))
    entries = [
        {
            "id": "feature_distributions",
            "purpose": "Compare signal/background densities of actual model input features and invariant mass; mass is QA only, never a training feature.",
            "provider": "hipe4ml.plot_utils.plot_distr",
            "files": ["feature_distributions.pdf"],
            "root_objects": [
                f"features/{name}/{category}"
                for name in columns
                for category in ("signal", "background")
            ],
            "sample_scope": "selected sampled train + outer held-out candidates",
            "api_parameters": {
                "columns": columns,
                "bins": settings["qa_bins"],
                "labels": ["Signal", "Background"],
                "density": True,
                "log": True,
            },
        },
        {
            "id": "correlations",
            "purpose": "Inspect class-specific Pearson correlations between actual model input features.",
            "provider": "hipe4ml.plot_utils.plot_corr",
            "files": ["correlations_signal.pdf", "correlations_background.pdf"],
            "root_objects": [
                "correlations/features",
                "correlations/signal",
                "correlations/background",
                "correlations/signal_valid",
                "correlations/background_valid",
            ],
            "sample_scope": "selected sampled train + outer held-out candidates",
            "api_parameters": {"columns": features, "labels": ["Signal", "Background"]},
        },
        {
            "id": "bdt_output",
            "purpose": "Compare signal/background raw-margin densities for training and held-out samples to assess overtraining.",
            "provider": "hipe4ml.plot_utils.plot_output_train_test",
            "files": ["bdt_output.pdf"],
            "root_objects": [
                f"scores/{split}_{category}{suffix}"
                for split in ("train", "test")
                for category in ("signal", "background")
                for suffix in ("", "_density")
            ],
            "sample_scope": "train and independent outer held-out samples shown separately",
            "api_parameters": {
                "bins": settings["qa_bins"],
                "output_margin": True,
                "labels": ["Background", "Signal"],
                "density": True,
                "logscale": True,
            },
        },
        {
            "id": "bdt_test_score",
            "purpose": "Inspect signal/background raw-margin separation on the held-out sample.",
            "provider": "hipe4ml.plot_utils.plot_distr",
            "files": ["bdt_test_score.pdf"],
            "root_objects": [
                "scores/test_signal",
                "scores/test_background",
                "scores/test_all",
            ],
            "sample_scope": "outer held-out candidates only",
            "api_parameters": {
                "column": "bdt_score",
                "bins": settings["qa_bins"],
                "labels": ["Background", "Signal"],
                "density": True,
            },
        },
        {
            "id": "roc_train_test",
            "purpose": "Compare training and held-out ROC curves and AUC values; signal label 1 is the positive class.",
            "provider": "hipe4ml.plot_utils.plot_roc_train_test",
            "files": ["roc_train_test.pdf"],
            "root_objects": ["roc/train", "roc/test"],
            "sample_scope": "train and independent outer held-out samples shown separately",
            "api_parameters": {"score_domain": "raw_margin", "positive_label": 1},
        },
        {
            "id": "efficiency_vs_score",
            "purpose": "Show the HIPE4ML requested efficiency-to-score mapping and actual inclusive held-out signal acceptance used by WP/storage.",
            "provider": "hipe4ml.plot_utils.plot_bdt_eff",
            "files": ["efficiency_vs_score.pdf"],
            "root_objects": ["efficiency/efficiency", "efficiency/actual_efficiency"],
            "sample_scope": "fixed outer held-out MC signal denominator",
            "api_parameters": {
                "requested_efficiency": "score_efficiency_array.efficiency",
                "score": "score_efficiency_array.threshold",
                "actual_acceptance": "score >= threshold",
            },
        },
        {
            "id": "feature_importance",
            "purpose": "Show SHAP feature effects and their aggregate importance using the official HIPE4ML wrapper; numerical ROOT importance stores the classifier's native gain, a separate definition.",
            "provider": "hipe4ml.plot_utils.plot_feature_imp",
            "files": ["feature_importance_shap.pdf", "feature_importance.pdf"],
            "root_objects": [
                "importance/features",
                "importance/basis",
                "importance/gain",
            ],
            "sample_scope": "bounded outer held-out sample; at most qa_shap_max_per_class per class",
            "api_parameters": {
                "labels": ["Background", "Signal"],
                "n_sample": settings["qa_shap_max_per_class"],
                "approximate": True,
                "seed": config["seed"],
            },
            "root_importance_basis": metrics.get("feature_importance_basis"),
        },
    ]
    optimizer = metrics.get("optuna", {})
    if optimizer.get("enabled"):
        entries.append(
            {
                "id": "optuna_training",
                "purpose": "Inspect completed grouped-CV trial AUC and the running best value; final held-out candidates are excluded from tuning.",
                "provider": "training_optimization trial history",
                "files": ["optuna_training.pdf"],
                "root_objects": [
                    "training/optuna/summary",
                    "training/optuna/trial_auc",
                    "training/optuna/best_auc",
                ],
                "sample_scope": "grouped folds within training split only",
                "api_parameters": {"metric": "auc", "direction": "maximize"},
            }
        )
    return {
        "schema_version": 1,
        "provider": "hipe4ml.plot_utils",
        "class_labels": {"0": "Background", "1": "Signal"},
        "score_domain": {
            "definition": "ModelHandler.predict(..., output_margin=True); no sigmoid or scaling",
            "output_margin": True,
            "range": "unbounded real margins; histogram edges cover all train/test scores",
            "positive_label": 1,
        },
        "features": features,
        "distribution_columns": columns,
        "qa_only_columns": ["mass_h3l"],
        "histogram_bins": settings["qa_bins"],
        "rows": {"train": len(data[0]), "test": len(data[2])},
        "selection": metrics.get("training_selection"),
        "input_cutflow": metrics.get("input_cutflow"),
        "shap_sample_limit_per_class": settings["qa_shap_max_per_class"],
        "plots": entries,
        "root_archive": {
            "file": "training.root",
            "contents": "numerical histograms/graphs/matrices; no ML canvases",
            "metadata_objects": [
                "metadata/training",
                "metadata/plot_backend",
                "metadata/qa_index",
            ],
        },
    }


def _write_numerical_qa(destination, metrics, index):
    """Retain machine-readable ROOT quantities without redrawing ML plots."""
    qa = metrics["qa"]
    with RootArchive(destination / "training.root") as archive:
        archive.json("metadata/training", metrics)
        archive.json("metadata/qa_index", index)
        archive.json(
            "metadata/plot_backend",
            {
                "provider": "hipe4ml.plot_utils",
                "format": "PDF",
                "root_contents": "numerical histograms/graphs/matrices; no ML canvas copy",
            },
        )
        for name, source in qa["feature_distributions"].items():
            for category in ("signal", "background"):
                counts = np.asarray(source[category], float)
                archive.hist1(
                    f"features/{name}/{category}",
                    source["edges"],
                    counts,
                    variance=counts,
                    entries=counts.sum(),
                )
        correlations = qa["correlations"]
        edges = np.arange(len(correlations["features"]) + 1, dtype=float)
        archive.json("correlations/features", correlations["features"])
        for category in ("signal", "background"):
            source = correlations[category]
            archive.hist2("correlations/" + category, edges, edges, source["values"])
            archive.hist2(
                "correlations/" + category + "_valid",
                edges,
                edges,
                np.asarray(source["valid"], float),
            )
        for name, source in qa["score_distributions"].items():
            counts, edges = np.asarray(source["counts"], float), np.asarray(
                source["edges"], float
            )
            archive.hist1(
                "scores/" + name, edges, counts, variance=counts, entries=counts.sum()
            )
            scale = 1 / (max(1.0, counts.sum()) * np.diff(edges))
            archive.hist1(
                "scores/" + name + "_density",
                edges,
                counts * scale,
                variance=counts * scale**2,
            )
        for split, source in qa["roc"].items():
            archive.graph("roc/" + split, source["fpr"], source["tpr"])
        curve = metrics["score_efficiency_array"]
        for key in ("efficiency", "actual_efficiency"):
            archive.graph("efficiency/" + key, curve["threshold"], curve[key])
        importance = metrics["feature_importance"]
        archive.json("importance/features", list(importance))
        archive.json("importance/basis", metrics.get("feature_importance_basis"))
        archive.hist1(
            "importance/gain", np.arange(len(importance) + 1), list(importance.values())
        )
        optimizer = metrics.get("optuna", {})
        if optimizer.get("enabled"):
            archive.json("training/optuna/summary", optimizer)
            completed = [
                trial for trial in optimizer["trials"] if trial["state"] == "COMPLETE"
            ]
            x, y = [trial["number"] for trial in completed], [
                trial["value"] for trial in completed
            ]
            archive.graph("training/optuna/trial_auc", x, y)
            archive.graph("training/optuna/best_auc", x, np.maximum.accumulate(y))


def render_training_qa(directory, handler, data, metrics, config):
    """Use the same HIPE plot wrappers as spectrum; never retain candidate tables.

    data is [train dataframe, labels, held-out dataframe, labels]. All source
    rows are already bounded by training.max_per_class. PDF figures are closed
    immediately; numerical ROOT contents are exported once from compact QA.
    """
    # Initialize Cling before SHAP/Numba loads its LLVM runtime. This is also
    # what the normal pipeline preflight does; standalone QA needs the same order.
    load_root()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from hipe4ml import plot_utils as pu

    destination = Path(directory) / "qa"
    destination.mkdir(parents=True, exist_ok=True)
    training, train_labels, validation, valid_labels = data
    train_labels, valid_labels = np.asarray(train_labels), np.asarray(valid_labels)
    features = handler.get_training_columns()
    if features != metrics["features"]:
        raise AnalysisError("ML QA feature order differs from its ModelHandler")
    if any(set(labels) != {0, 1} for labels in (train_labels, valid_labels)):
        raise AnalysisError("ML QA requires both classes in train and held-out")
    settings = config["training"]
    bins = settings["qa_bins"]
    combined = pd.concat([training, validation], ignore_index=True)
    labels = np.r_[train_labels, valid_labels]
    classes = [combined.loc[labels == label] for label in (1, 0)]
    paths = []

    def save(figure, name):
        path = destination / (name + ".pdf")
        try:
            figure.savefig(path, format="pdf", bbox_inches="tight")
        finally:
            plt.close(figure)
        if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
            raise AnalysisError(f"HIPE4ML did not produce {path}")
        paths.append(path)

    columns = list(dict.fromkeys(features + ["mass_h3l"]))
    rows = int(np.ceil(len(columns) / 3))
    figures_before = set(plt.get_fignums())
    random_state = np.random.get_state()
    try:
        axes = pu.plot_distr(
            classes,
            columns,
            bins=bins,
            labels=["Signal", "Background"],
            colors=["red", "blue"],
            log=True,
            density=True,
            alpha=0.5,
            figsize=(14, max(4, 3.5 * rows)),
            grid=False,
        )
        figure = np.asarray(axes, dtype=object).ravel()[0].get_figure()
        for axis in np.asarray(axes, dtype=object).ravel():
            axis.set_ylabel("Probability density")
        mass_axis = np.asarray(axes, dtype=object).ravel()[columns.index("mass_h3l")]
        mass_axis.set_title("H3L mass (QA only)")
        mass_axis.set_xlabel(r"$m(^{3}\mathrm{He}+\pi)$ [GeV/$c^{2}$]")
        figure.subplots_adjust(hspace=0.55, wspace=0.55)
        save(figure, "feature_distributions")
        correlations = pu.plot_corr(classes, features, labels=["Signal", "Background"])
        for figure, category in zip(correlations, ("signal", "background")):
            save(figure, "correlations_" + category)
        figure = pu.plot_output_train_test(
            handler,
            data,
            bins=bins,
            output_margin=True,
            labels=["Background", "Signal"],
            logscale=True,
            density=True,
        )
        figure.axes[0].set_ylabel("Probability density")
        save(figure, "bdt_output")
        train_scores = handler.predict(training, output_margin=True)
        valid_scores = handler.predict(validation, output_margin=True)
        score_classes = [
            pd.DataFrame({"bdt_score": valid_scores[valid_labels == label]})
            for label in (0, 1)
        ]
        axes = pu.plot_distr(
            score_classes,
            "bdt_score",
            bins=bins,
            labels=["Background", "Signal"],
            density=True,
            alpha=0.5,
        )
        np.asarray(axes, dtype=object).ravel()[0].set_ylabel("Probability density")
        save(np.asarray(axes, dtype=object).ravel()[0].get_figure(), "bdt_test_score")
        save(
            pu.plot_roc_train_test(
                valid_labels, valid_scores, train_labels, train_scores
            ),
            "roc_train_test",
        )
        curve = metrics["score_efficiency_array"]
        figure = pu.plot_bdt_eff(curve["threshold"], curve["efficiency"])
        axis = figure.axes[0]
        axis.plot(
            curve["threshold"],
            curve["actual_efficiency"],
            label="Actual held-out acceptance",
        )
        axis.legend()
        save(figure, "efficiency_vs_score")
        # HIPE plot_feature_imp samples via NumPy's global RNG; scope its seed
        # locally so QA is repeatable and cannot change model/fold randomness.
        np.random.seed(config["seed"])
        shap_plots = pu.plot_feature_imp(
            validation,
            valid_labels,
            handler,
            labels=["Background", "Signal"],
            n_sample=settings["qa_shap_max_per_class"],
            approximate=True,
        )
        save(shap_plots[0], "feature_importance_shap")
        save(shap_plots[-1], "feature_importance")
    finally:
        np.random.set_state(random_state)
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)
    index = _qa_index(handler, data, metrics, config)
    # Canonical names are emitted once. An in-place caller upgrading an old QA
    # directory must not retain the superseded train/test PDF alongside it.
    (destination / "bdt_train_test.pdf").unlink(missing_ok=True)
    write_json(destination / "index.json", index)
    _write_numerical_qa(destination, metrics, index)
    return [destination / "training.root", destination / "index.json", *paths]
