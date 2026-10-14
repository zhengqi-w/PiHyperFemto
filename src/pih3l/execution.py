"""Explicit data contracts and stage cache scopes, independent of orchestration."""

import copy

from .schema import CANDIDATE_COLUMNS, HYP_CANDIDATE

DERIVED = {
    "mass_h3l",
    "pt_h3l",
    "eta_h3l",
    "phi_h3l",
    "ct",
    "cos_pa",
    "decay_radius",
    "kstar",
    "KstarMC",
}
ITS_DERIVED = {
    "its_mean_size_he": "ItsClusterSizesHe",
    "its_ncls_he": "ItsClusterSizesHe",
    "its_mean_size_pi": "ItsClusterSizesPi",
    "its_ncls_pi": "ItsClusterSizesPi",
}
PAIR_BASE = {
    "kstar",
    "mass_h3l",
    "pt_h3l",
    "HypCentFT0C",
    "ChargeChannel",
    "IsMixed",
    "bdt_score",
}
PAIR_OPTIONAL = {
    "HypTrackOccupancy",
    "PairCentFT0C",
    "MixingDepth",
    "IsClosePairRejected",
    "_source_file",
    "_source_tree",
    "_entry",
    "bdt_model_id",
    "bdt_in_scope",
}
MC_OPTIONAL = {
    "HypIsReco",
    "HypIsSignal",
    "HypIsTwoBodyDecay",
    "PairPiIsPhysicalPrimary",
    "PairPiIsTruePion",
    "PairPiPdgCode",
    "IsTruthSelfCorrelation",
    "SameMCCollision",
    "IsTruePrimaryPiHyperPair",
    "MatchesHypRecoMCCollision",
    "MatchesPairRecoMCCollision",
    "KstarMC",
    "HypGenPt",
    "HypGenEta",
    "HypGenPhi",
    "PairPiGenPt",
    "PairPiGenEta",
    "PairPiGenPhi",
}


def training_features(config):
    features = set(config["training"]["features"])
    for spec in config["training"].get("pt_bins", []):
        if isinstance(spec, dict):
            features.update(spec.get("features", []))
    return features


def analysis_columns(config, *, pairs=True):
    columns = set()
    sections = (
        ("candidate_cuts", "event_cuts", "pair_cuts")
        if pairs
        else ("candidate_cuts", "event_cuts")
    )
    analyses = [config["analysis"]]
    analyses.extend(
        v.get("overrides", {}).get("analysis", {})
        for v in config["systematics"]["variations"]
    )
    for analysis in analyses:
        for section in sections:
            columns.update(analysis.get(section, {}))
        if pairs and analysis.get("close_pair_reject_bits", 0):
            columns.add("IsClosePairRejected")
        if pairs and analysis.get("mixing_weight") == "inverse_depth":
            columns.add("MixingDepth")
    return columns


def preparation_columns(config, role):
    pairs = role in {"femto", "mc_femto"}
    required = set(CANDIDATE_COLUMNS) | {"HypCentFT0C"}
    required |= analysis_columns(config, pairs=pairs)
    # Candidate event cuts are optional in the existing WP contract.
    # Retain the approved candidate-variable pool once, regardless of which
    # subset a particular pT model uses. Feature changes then reuse preparation.
    optional = (
        set(HYP_CANDIDATE) if config["io"].get("persist_prepared", False) else set()
    )
    if not config["io"].get("persist_prepared", False):
        required |= training_features(config)
    requested_output = set(config["io"].get("scored_columns", [])) if pairs else set()
    required |= requested_output - {
        "bdt_score",
        "bdt_model_id",
        "bdt_in_scope",
        "_source_file",
        "_source_tree",
        "_entry",
    }
    if pairs:
        optional |= set(config["io"].get("scored_optional_columns", []))
    if role == "background":
        wp = config["working_point"]
        required |= set(wp.get("candidate_cuts") or {})
        required |= set(wp.get("event_cuts") or {})
    optional |= set(config["training"].get("candidate_cuts", {}))
    for spec in config["training"].get("pt_bins", []):
        if isinstance(spec, dict):
            optional |= set(spec.get("candidate_cuts", {}))
    if not pairs:
        optional |= set(config["analysis"]["event_cuts"])
        required -= set(config["analysis"]["event_cuts"])
    if pairs:
        required |= {
            "IsMixed",
            "ChargeChannel",
            "PairPiTrackPt",
            "PairPiTrackEta",
            "PairPiTrackPhi",
        }
        if config["io"].get("persist_prepared", False):
            optional |= PAIR_OPTIONAL
    if role == "signal":
        required |= {"HypIsReco", "HypIsSignal", "HypIsTwoBodyDecay"}
    if role == "mc_femto":
        optional |= MC_OPTIONAL
    for column, source in ITS_DERIVED.items():
        if column in required:
            required.remove(column)
            required.add(source)
        if column in optional:
            optional.remove(column)
            optional.add(source)
    required -= DERIVED
    return sorted(required), sorted(optional - required)


def scored_columns(config, role, available):
    required = (
        PAIR_BASE
        | analysis_columns(config)
        | set(config["io"].get("scored_columns", []))
    )
    if config["training"]["pt_bins"]:
        required.add("bdt_model_id")
    missing = required - set(available)
    if missing:
        from .errors import AnalysisError

        raise AnalysisError(
            f"{role} scored output: missing required columns {sorted(missing)}"
        )
    optional = set(config["io"].get("scored_optional_columns", []))
    if role == "mc_femto":
        # Generated momentum components are consumed to derive KstarMC, then
        # discarded unless the user explicitly requests them for an extra QA.
        optional |= MC_OPTIONAL - {
            "HypGenPt",
            "HypGenEta",
            "HypGenPhi",
            "PairPiGenPt",
            "PairPiGenEta",
            "PairPiGenPhi",
        }
    return sorted(required | (optional & set(available)))


STAGE_MODULES = {
    "raw_qa": [
        "raw_qa",
        "raw_qa_output",
        "root_plot",
        "root_io",
        "schema",
        "physics",
        "io",
    ],
    "prepared": ["physics", "schema", "io", "execution", "dataset"],
    "training": [
        "training",
        "training_support",
        "training_qa",
        "root_io",
        "efficiency",
        "training_settings",
        "training_optimization",
        "physics",
        "schema",
        "execution",
        "validation_scores",
        "models",
        "selection",
        "io",
        "dataset",
    ],
    "working_point": [
        "working_point",
        "wp_mass_fit",
        "training_support",
        "working_point_qa",
        "root_io",
        "root_plot",
        "efficiency",
        "physics",
        "schema",
        "execution",
        "validation_scores",
        "models",
        "selection",
        "purity",
        "io",
        "dataset",
    ],
    "scored": [
        "models",
        "training_support",
        "training",
        "training_settings",
        "selection",
        "score_cache",
        "io",
        "execution",
        "dataset",
        "physics",
        "schema",
        "scored_storage",
    ],
    "nominal": ["analysis", "correlation", "selection", "purity", "io"],
    "systematics": [
        "systematics",
        "systematics_qa",
        "root_io",
        "root_plot",
        "analysis",
        "correlation",
        "selection",
        "purity",
        "io",
    ],
    "mc_qa": ["mc", "analysis", "correlation", "selection", "purity", "io"],
    "report": [
        "report",
        "report_models",
        "root_output",
        "systematics_qa",
        "raw_qa_output",
        "root_plot",
        "root_io",
        "io",
        "validation_scores",
    ],
}


def hipe4ml_model_source():
    """Bind official handler training, serialization and prediction semantics."""
    from pathlib import Path
    from hipe4ml import model_handler
    from .io import sha256_file

    return sha256_file(Path(model_handler.__file__))


def optional_training_dependencies(config, *, optimizer=False):
    """Optional libraries affect caches only when the relevant stage consumes them."""
    training = config["training"]
    dependencies = {"hipe4ml", "xgboost"}
    if optimizer:
        scopes = training.get("pt_bins") or [{}]
        if any(
            (scope if isinstance(scope, dict) else {}).get(
                "use_optuna", training.get("use_optuna", False)
            )
            for scope in scopes
        ):
            dependencies.add("optuna")
    return dependencies


def stage_environment(environment, config, stage):
    result = {
        key: value for key, value in environment.items() if key != "optional_training"
    }
    if stage in {"training", "working_point", "scored"}:
        optional = environment.get("optional_training", {})
        for name in optional_training_dependencies(
            config, optimizer=stage == "training"
        ):
            result[name] = optional.get(name, "unavailable")
        result["hipe4ml_model_handler_source"] = hipe4ml_model_source()
        if stage == "training":
            for name in ("matplotlib", "shap"):
                result[name] = optional.get(name, "unavailable")
            from pathlib import Path
            from hipe4ml import plot_utils
            from .io import sha256_file

            result["hipe4ml_plot_utils_source"] = sha256_file(Path(plot_utils.__file__))
    if stage in {"training", "working_point"}:
        from .efficiency import mapping_metadata

        result["hipe4ml_source"] = mapping_metadata()["implementation_sha256"]
    return result


def config_scope(config, stage, role=None):
    a = config["analysis"]
    if stage == "prepared":
        roles = [role] if role else list(config["inputs"])
        scope = {
            "physics": config["physics"],
            "centrality_bins": (
                a["centrality_bins"]
                if any(r in {"femto", "mc_femto"} for r in roles)
                else None
            ),
            "persist_prepared": config["io"].get("persist_prepared", False),
            "columns": {r: list(preparation_columns(config, r)) for r in roles},
            "chunk_rows": config["io"]["chunk_rows"],
            "projection": config["io"].get("root_projection", True),
        }
        if any(r in {"femto", "mc_femto"} for r in roles):
            scope.update(
                mass_range=a["mass_range"],
                kstar_range=[a["kstar_edges"][0], a["kstar_edges"][-1]],
            )
        return scope
    if stage == "raw_qa":
        return {
            "raw_qa": config["raw_qa"],
            "physics": config["physics"],
            "centrality_bins": a["centrality_bins"],
        }
    if stage == "training":
        training = copy.deepcopy(config["training"])
        training.pop("retrain_bins", None)
        bins = training.get("pt_bins", [])
        enabled = training.get("use_optuna", False) if not bins else False
        for spec in bins:
            if isinstance(spec, dict):
                active = spec.get("use_optuna", training.get("use_optuna", False))
                enabled |= active
                if not active:
                    spec.pop("optuna", None)
            else:
                enabled |= training.get("use_optuna", False)
        if not enabled:
            training.pop("optuna", None)
        return {
            "training": training,
            "seed": config["seed"],
        }
    if stage == "working_point":
        return {
            "working_point": config["working_point"],
            "wp_efficiency_deltas": sorted(
                {
                    v["wp_efficiency_delta"]
                    for v in config["systematics"]["variations"]
                    if "wp_efficiency_delta" in v
                }
            ),
            "absolute_systematic_thresholds": [
                v.get("overrides", {}).get("analysis", {}).get("bdt_threshold")
                for v in config["systematics"]["variations"]
                if v.get("overrides", {}).get("analysis", {}).get("bdt_threshold")
                is not None
            ],
            "purity": config["purity"],
            "analysis": {
                k: a[k]
                for k in (
                    "candidate_cuts",
                    "event_cuts",
                    "mass_range",
                    "signal_window",
                    "sidebands",
                    "bdt_threshold",
                )
            },
        }
    if stage == "scored":
        return {
            "slim": config["io"].get("slim_scored", True),
            "columns": sorted(analysis_columns(config)),
            "chunk_rows": config["io"]["chunk_rows"],
            "persist_prepared": config["io"].get("persist_prepared", False),
            "partition_by_pt": config["io"].get("scored_partition_by_pt", True),
            "saved_columns": config["io"].get("scored_columns", []),
            "optional_saved_columns": config["io"].get("scored_optional_columns", []),
            "cache_model_scores": config["io"].get("cache_model_scores", False),
            "storage": config["working_point"]["storage"],
            "physics": config["physics"],
            "domain": {
                "centrality_bins": a["centrality_bins"],
                "mass_range": a["mass_range"],
                "kstar_range": [a["kstar_edges"][0], a["kstar_edges"][-1]],
            },
        }
    if stage == "systematics":
        return {
            k: config[k] for k in ("analysis", "purity", "uncertainty", "systematics")
        }
    if stage in {"nominal", "mc_qa"}:
        return {k: config[k] for k in ("analysis", "purity", "uncertainty", "mc")}
    return config
