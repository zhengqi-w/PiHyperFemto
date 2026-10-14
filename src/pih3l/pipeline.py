"""Transactional stage orchestration, artifact provenance, and resumable execution."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import importlib.metadata
import logging
import os
import platform
import shutil
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import __version__
from .analysis import run_correlations
from .config import digest
from .errors import AnalysisError
from .execution import (
    STAGE_MODULES,
    config_scope,
    preparation_columns,
    scored_columns,
    training_features,
    stage_environment,
)
from .io import (
    inventory,
    read_json,
    sha256_file,
    write_batch,
    write_json,
    root_shards,
    read_root_shard,
)
from .physics import derive
from .root_io import load_root
from .schema import require_columns
from .selection import in_window, in_windows
from .systematics import run_systematics

LOG = logging.getLogger(__name__)
STAGES = [
    "raw_qa",
    "prepared",
    "training",
    "working_point",
    "scored",
    "nominal",
    "systematics",
    "mc_qa",
    "report",
]
DEPENDENCIES = {
    "raw_qa": [],
    "prepared": [],
    "training": ["prepared"],
    "working_point": ["prepared", "training"],
    "scored": ["prepared", "training", "working_point"],
    "nominal": ["scored", "working_point"],
    "systematics": ["nominal", "scored", "working_point"],
    "mc_qa": ["scored", "working_point"],
    "report": [
        "raw_qa",
        "nominal",
        "training",
        "working_point",
        "systematics",
        "mc_qa",
    ],
}


def code_fingerprint():
    return digest(
        {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )


def environment():
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pih3l": __version__,
    }
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "uproot",
        "pyarrow",
    ):
        result[name] = importlib.metadata.version(name)
    result["ROOT"] = load_root().gROOT.GetVersion()
    result["optional_training"] = {}
    for name in ("xgboost", "optuna", "hipe4ml", "matplotlib", "shap"):
        try:
            result["optional_training"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["optional_training"][name] = "unavailable"
    return result


@contextlib.contextmanager
def run_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AnalysisError(f"Another process is writing {output}") from error
        try:
            lock.write(str(os.getpid()))
            lock.flush()
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class Pipeline:
    def __init__(self, config, *, force=False, input_roles=None):
        if config.get("periods"):
            raise AnalysisError(
                "Use MultiPeriodPipeline for configurations declaring periods"
            )
        self.config = config
        self.output = Path(config["output"])
        self.force = force
        self.inputs = inventory(config, roles=input_roles)
        self.environment = environment()
        self.code_signature = code_fingerprint()
        self.signature = digest(
            {
                "config": config,
                "inputs": self.inputs,
                "code": self.code_signature,
                "environment": self.environment,
            }
        )
        self.finished = set()

    def dependency_identities(self, stage, *, role=None):
        """Expose artifact lineage, scoped to the roles actually consumed."""
        dependencies = {}
        for dependency in DEPENDENCIES[stage]:
            manifest_path = self.output / dependency / "manifest.json"
            if not manifest_path.exists():
                dependencies[dependency] = None
                continue
            data = read_json(manifest_path)
            if dependency == "working_point" and stage == "scored":
                # WP QA/significance can change while the saved loose score
                # floor remains identical. Bind the consumed storage boundary,
                # retaining WP execution and SHA verification as a dependency.
                summary = read_json(manifest_path.parent / "summary.json")
                enabled = (
                    data["config"]["working_point"]
                    .get("storage", {})
                    .get("enabled", False)
                )
                dependencies[dependency] = {
                    "storage_threshold": (
                        summary.get("storage_threshold") if enabled else None
                    )
                }
                continue
            roles = None
            if dependency == "prepared":
                roles = (
                    ["signal", "background"]
                    if stage == "training"
                    else (
                        ["background", "signal"]
                        if stage == "working_point"
                        else [role] if role else ["femto", "mc_femto"]
                    )
                )
            if roles is not None:
                dependencies[dependency] = {
                    "artifacts": {
                        name: checksum
                        for name, checksum in data["artifacts"].items()
                        if any(name.startswith(r + "/") for r in roles)
                    },
                    "preparation": {
                        r: config_scope(data["config"], "prepared", r) for r in roles
                    },
                }
                if not data["config"]["io"].get("persist_prepared", True):
                    dependencies[dependency]["stream_sources"] = {
                        r: data["input_inventory"][r] for r in roles
                    }
            else:
                dependencies[dependency] = {
                    "signature": data["signature"],
                    "artifacts": data["artifacts"],
                }
        return dependencies

    def stage_signature(self, stage, *, role=None):
        """Bind only consumed configuration, source modules and dependency artifacts."""
        source = Path(__file__).parent
        modules = STAGE_MODULES[stage] + ["pipeline"]
        code = {
            name: sha256_file(source / f"{name}.py")
            for name in modules
            if (source / f"{name}.py").exists()
        }
        roles = (
            [role]
            if role
            else (["femto", "background"] if stage == "raw_qa" else list(self.inputs))
        )
        inputs = (
            {r: self.inputs[r] for r in roles if r in self.inputs}
            if stage in {"prepared", "raw_qa"}
            else {}
        )
        return digest(
            {
                "schema_version": 2,
                "config": config_scope(self.config, stage, role),
                "inputs": inputs,
                "code": code,
                "environment": stage_environment(self.environment, self.config, stage),
                "dependencies": self.dependency_identities(stage, role=role),
            }
        )

    def is_current(self, stage):
        manifest = self.output / stage / "manifest.json"
        if not manifest.exists():
            return False
        data = read_json(manifest)
        if data.get("signature") != self.stage_signature(stage):
            return False
        for name, checksum in data.get("artifacts", {}).items():
            artifact = manifest.parent / name
            if not artifact.is_file() or sha256_file(artifact) != checksum:
                return False
        return True

    def execute(self, stage):
        if stage in self.finished:
            return
        for dependency in DEPENDENCIES[stage]:
            self.execute(dependency)
        retrain = stage == "training" and self.config["training"].get("retrain_bins")
        if not self.force and not retrain and self.is_current(stage):
            LOG.info("Reuse verified stage: %s", stage)
            self.finished.add(stage)
            return
        started = time.time()
        stage_signature = self.stage_signature(stage)
        temporary = self.output / f".{stage}-{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        LOG.info("Start stage: %s", stage)
        try:
            getattr(self, "_" + stage)(temporary)
            if code_fingerprint() != self.code_signature:
                raise AnalysisError(
                    "Analysis source changed during execution; rerun after editing completes"
                )
            artifacts = {
                str(p.relative_to(temporary)): sha256_file(p)
                for p in sorted(temporary.rglob("*"))
                if p.is_file()
            }
            write_json(
                temporary / "manifest.json",
                {
                    "stage": stage,
                    "signature": stage_signature,
                    "schema_version": 2,
                    "cache_config": config_scope(self.config, stage),
                    "dependencies": self.dependency_identities(stage),
                    "config": self.config,
                    "input_inventory": self.inputs,
                    "environment": self.environment,
                    "duration_seconds": time.time() - started,
                    "completed_unix": time.time(),
                    "artifacts": artifacts,
                },
            )
            destination = self.output / stage
            previous = self.output / f".{stage}-previous-{uuid.uuid4().hex}"
            if destination.exists():
                os.replace(destination, previous)
            try:
                os.replace(temporary, destination)
            except BaseException:
                if previous.exists():
                    os.replace(previous, destination)
                raise
            if previous.exists():
                shutil.rmtree(previous)
            self._prune_checkpoints(stage)
        except BaseException as error:
            diagnostic_path = None
            if stage == "working_point" and any(temporary.rglob("scan_fits.json")):
                diagnostic_path = (
                    self.output / "failures" / f"{stage}-{stage_signature}"
                )
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                if diagnostic_path.exists():
                    shutil.rmtree(diagnostic_path)
                os.replace(temporary, diagnostic_path)
                if isinstance(error, AnalysisError):
                    error.args = (
                        str(error).replace(str(temporary), str(diagnostic_path)),
                    )
            write_json(
                self.output / "failures" / f"{stage}.json",
                {
                    "stage": stage,
                    "signature": stage_signature,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "diagnostics_path": (
                        str(diagnostic_path) if diagnostic_path else None
                    ),
                    "failed_unix": time.time(),
                    "previous_completed_stage_preserved": (
                        self.output / stage
                    ).exists(),
                },
            )
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.finished.add(stage)
        LOG.info("Finished %s in %.1f s", stage, time.time() - started)

    def _raw_qa(self, destination):
        from .raw_qa import run_raw_qa

        run_raw_qa(self.config, destination)

    def _prepared(self, destination):
        if not self.config["io"].get("persist_prepared", True):
            self._prepare_stream_inventory(destination)
            return
        summary = {}
        a = self.config["analysis"]
        for role in self.config["inputs"]:
            counts = {
                "raw_rows": 0,
                "after_role_filter": 0,
                "retained_rows": 0,
                "nonfinite_kinematics": 0,
                "chunks": 0,
            }
            pairs = role in {"femto", "mc_femto"}
            checkpoint = self._checkpoint("prepared", role)
            required_columns, optional_columns = preparation_columns(self.config, role)
            for index, shard, tree in root_shards(self.config, role, with_tree=True):
                record = self._restore_shard(
                    checkpoint, index, destination / role, source=shard
                )
                if record is not None:
                    for key, value in record["counts"].items():
                        counts[key] += value
                    continue
                before = counts.copy()
                frame = read_root_shard(
                    self.config,
                    role,
                    shard,
                    columns=(
                        required_columns
                        if self.config["io"].get("root_projection", True)
                        else None
                    ),
                    optional_columns=optional_columns,
                    tree=tree,
                )
                counts["raw_rows"] += len(frame)
                if role == "signal":
                    truth = ["HypIsReco", "HypIsSignal", "HypIsTwoBodyDecay"]
                    require_columns(frame, truth, "MC training signal")
                    frame = frame.loc[frame[truth].astype(bool).all(axis=1)].copy()
                counts["after_role_filter"] += len(frame)
                if frame.empty:
                    self._save_shard(
                        checkpoint,
                        index,
                        None,
                        shard,
                        {k: counts[k] - before[k] for k in before},
                    )
                    continue
                frame = derive(frame, self.config["physics"], pairs=pairs)
                required = ["mass_h3l", "pt_h3l", "ct", "cos_pa"] + (
                    ["kstar"] if pairs else []
                )
                finite = np.isfinite(frame[required]).all(axis=1)
                counts["nonfinite_kinematics"] += int((~finite).sum())
                keep = finite
                # Candidate training needs its own sidebands, not the Femto producer's narrow mass range.
                if pairs:
                    keep &= in_windows(frame.HypCentFT0C, a["centrality_bins"])
                    keep &= in_window(frame.mass_h3l, a["mass_range"])
                    keep &= in_window(
                        frame.kstar, [a["kstar_edges"][0], a["kstar_edges"][-1]]
                    )
                frame = frame.loc[keep].copy()
                require_columns(
                    frame,
                    training_features(self.config),
                    f"{role} BDT feature contract",
                )
                counts["retained_rows"] += len(frame)
                if len(frame):
                    # Float64 precision is retained for derived kinematics. Provenance strings compress well.
                    artifact = write_batch(destination / role, index, frame)
                    counts["chunks"] += 1
                else:
                    artifact = None
                self._save_shard(
                    checkpoint,
                    index,
                    artifact,
                    shard,
                    {k: counts[k] - before[k] for k in before},
                )
                if index % 10 == 0:
                    LOG.info(
                        "%s: read %s rows; retained %s",
                        role,
                        counts["raw_rows"],
                        counts["retained_rows"],
                    )
            if not counts["raw_rows"]:
                counts["status"] = (
                    "empty_input" if self.inputs[role]["files"] else "missing_optional"
                )
            elif not counts["retained_rows"]:
                counts["status"] = "empty_after_preparation"
            else:
                counts["status"] = "available"
            summary[role] = counts
            if role == "signal":
                counts["truth_reconstructed_signal"] = counts["after_role_filter"]
        write_json(destination / "summary.json", summary)
        write_json(destination / "inventory.json", self.inputs)

    def _prepare_stream_inventory(self, destination):
        """Describe ROOT sources without decoding candidates or persisting raw rows."""
        from .dataset import RootDataset

        summary = {}
        for role, record in self.inputs.items():
            rows = sum(
                tree["rows"] for source in record["files"] for tree in source["trees"]
            )
            summary[role] = {
                "raw_rows": rows,
                "after_role_filter": None,
                "retained_rows": None,
                "nonfinite_kinematics": None,
                "chunks": 0,
                "persisted": False,
                "mode": "direct_root_stream",
                "status": (
                    "available"
                    if rows
                    else "empty_input" if record["files"] else "missing_optional"
                ),
                "filter_counters": "evaluated by consuming training, WP and scored stages",
                "source_identity": RootDataset(self.config, role, record).fingerprint(),
            }
        write_json(destination / "summary.json", summary)
        write_json(destination / "inventory.json", self.inputs)

    def _candidate_source(self, role, *, config=None):
        """Use replayable ROOT readers unless the explicit compatibility cache is enabled."""
        if self.config["io"].get("persist_prepared", True):
            return self.output / "prepared" / role
        from .dataset import RootDataset

        return RootDataset(config or self.config, role, self.inputs[role])

    def _training(self, destination):
        from .models import model_specs, train_models

        config = self.config
        if self.force:
            config = copy.deepcopy(config)
            config["training"]["retrain_bins"] = [
                spec["id"] for spec in model_specs(config)
            ]

        train_models(
            self._candidate_source("signal", config=config),
            self._candidate_source("background", config=config),
            destination,
            config,
        )

    def _working_point(self, destination):
        from .working_point import scan_model_working_points

        scan_model_working_points(
            self._candidate_source("background"),
            self.output / "training",
            destination,
            self.config,
            mc_signal=self._candidate_source("signal"),
        )

    def _scored(self, destination):
        from .models import load_model_registry, score_models

        registry = load_model_registry(self.output / "training")
        inference_code = {
            name: sha256_file(Path(__file__).with_name(name + ".py"))
            for name in (
                "models",
                "training",
                "selection",
                "score_cache",
                "dataset",
                "physics",
                "schema",
                "execution",
                "io",
            )
        }
        for spec in registry["models"]:
            spec["inference_checksum"] = sha256_file(
                self.output / "training" / spec["relative_path"] / "model_handler.pkl"
            )
            spec["inference_code"] = inference_code
            spec["inference_environment"] = stage_environment(
                self.environment, self.config, "scored"
            )
        self.active_score_caches = set()
        from .score_cache import cached_model_scores, score_key
        from .scored_storage import write_scored_batch, write_storage_metadata

        wp = read_json(self.output / "working_point/summary.json")
        summary = {}
        cache_predictions = self.config["io"].get("checkpoint", True) and self.config[
            "io"
        ].get("cache_model_scores", False)
        for role in ("femto", "mc_femto"):
            totals = {
                key: 0
                for key in (
                    "rows",
                    "retained_rows",
                    "invalid_scores",
                    "dropped_below_storage",
                    "out_of_scope_rows",
                )
            }
            checkpoint = self._checkpoint("scored", role)
            for index, source, read_frame in self._score_inputs(role):
                if cache_predictions:
                    for spec in registry["models"]:
                        self.active_score_caches.add(
                            self.output
                            / ".checkpoints/model_scores"
                            / role
                            / score_key(source, spec)
                        )
                record = self._restore_shard(
                    checkpoint, index, destination / role, source=source
                )
                if record is not None:
                    for key, value in record["counts"].items():
                        totals[key] = totals.get(key, 0) + value
                    continue
                frame = read_frame()
                source_counts = frame.attrs.get("source_counts", {})
                if cache_predictions:
                    frame = cached_model_scores(
                        frame,
                        registry,
                        checkpoint_root=self.output
                        / ".checkpoints/model_scores"
                        / role,
                        source=source,
                        active_paths=self.active_score_caches,
                        force=self.force,
                    )
                else:
                    frame = score_models(frame, registry)
                artifacts, count = write_scored_batch(
                    destination / role, index, frame, self.config, wp, role
                )
                for key in ("raw_rows", "after_role_filter", "nonfinite_kinematics"):
                    if key in source_counts:
                        count[key] = source_counts[key]
                self._save_shard(
                    checkpoint,
                    index,
                    artifacts,
                    source,
                    count,
                    artifact_root=destination / role,
                )
                for key, value in count.items():
                    totals[key] = totals.get(key, 0) + value
                if index % 10 == 0:
                    LOG.info(
                        "Scored %s: input=%s retained=%s",
                        role,
                        totals["rows"],
                        totals["retained_rows"],
                    )
            summary[role] = totals
            write_storage_metadata(destination / role, self.config, wp, counts=totals)
        write_json(destination / "summary.json", summary)

    def _score_inputs(self, role):
        """Yield metadata and deferred readers so resume happens before row decoding."""
        if not self.config["io"].get("persist_prepared", True):
            from .dataset import RootDataset

            dataset = RootDataset(self.config, role, self.inputs[role])
            # The full inventory is recorded once in prepared/manifest.json;
            # repeating it in every large-period shard would inflate metadata.
            identity = digest(dataset.fingerprint())
            for index, shard, tree in dataset.shards():
                source = {"dataset": identity, "shard": shard}
                yield index, source, partial(dataset.read_shard, shard, tree)
            return
        manifest = read_json(self.output / "prepared/manifest.json")
        index = 0
        for path in sorted((self.output / "prepared" / role).glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            checksum = manifest["artifacts"][
                str(path.relative_to(self.output / "prepared"))
            ]
            size = self.config["io"]["chunk_rows"]
            columns = None
            if self.config["io"].get("slim_scored", True):
                available = set(parquet.schema_arrow.names)
                generated = {"bdt_score", "bdt_model_id", "bdt_in_scope"}
                columns = (
                    (
                        set(scored_columns(self.config, role, available | generated))
                        - generated
                    )
                    | training_features(self.config)
                    | {"pt_h3l"}
                )
                columns = sorted(columns)
            for start in range(0, parquet.metadata.num_rows, size):
                source = {
                    "path": str(path),
                    "sha256": checksum,
                    "start": start,
                    "rows": min(size, parquet.metadata.num_rows - start),
                }
                yield index, source, partial(
                    self._read_parquet_slice, parquet, start, size, columns=columns
                )
                index += 1

    @staticmethod
    def _read_parquet_slice(parquet, start, count, *, columns=None):
        """Decode only row groups overlapping one requested bounded shard."""
        offset, groups = 0, []
        first = None
        for group in range(parquet.metadata.num_row_groups):
            rows = parquet.metadata.row_group(group).num_rows
            if offset < start + count and offset + rows > start:
                groups.append(group)
                first = offset if first is None else first
            offset += rows
        if columns is not None:
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise AnalysisError(
                    f"Scoring: missing required Parquet columns {sorted(missing)}"
                )
        return (
            parquet.read_row_groups(groups, columns=columns)
            .slice(start - first, count)
            .to_pandas()
        )

    def _checkpoint(self, stage, role):
        if not self.config["io"].get("checkpoint", True):
            return None
        path = (
            self.output
            / ".checkpoints"
            / stage
            / role
            / self.stage_signature(stage, role=role)
        )
        if self.force and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _prune_checkpoints(self, stage):
        """Retire superseded generations only after a successful atomic commit."""
        if stage not in {"prepared", "scored"}:
            return
        root = self.output / ".checkpoints" / stage
        for role in (
            self.config["inputs"] if stage == "prepared" else ("femto", "mc_femto")
        ):
            current = self.stage_signature(stage, role=role)
            for path in (root / role).glob("*"):
                if path.is_dir() and path.name != current:
                    shutil.rmtree(path)
        if stage == "scored" and hasattr(self, "active_score_caches"):
            for role in ("femto", "mc_femto"):
                for path in (self.output / ".checkpoints/model_scores" / role).glob(
                    "*"
                ):
                    if path.is_dir() and path not in self.active_score_caches:
                        shutil.rmtree(path)

    def _restore_shard(self, checkpoint, index, destination, *, source=None):
        if checkpoint is None or not (checkpoint / f"{index:06d}.json").exists():
            return None
        try:
            record = read_json(checkpoint / f"{index:06d}.json")
        except (ValueError, OSError):
            return None
        payload = {
            key: value for key, value in record.items() if key != "record_digest"
        }
        if record.get("record_digest") != digest(payload):
            return None
        if source is not None and record.get("source") != source:
            return None
        artifacts = record.get("artifacts", {})
        if record.get("artifact"):
            artifacts = {record["artifact"]: record["sha256"]}
        for name, checksum in artifacts.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                return None
            artifact = checkpoint / relative
            if not artifact.is_file() or sha256_file(artifact) != checksum:
                return None
        for name in artifacts:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(checkpoint / name, target)
        return record

    @staticmethod
    def _save_shard(
        checkpoint, index, artifact, provenance, counts, *, artifact_root=None
    ):
        if checkpoint is None:
            return
        record = {"source": provenance, "counts": counts, "artifact": None}
        if isinstance(artifact, (list, tuple)):
            if artifact and artifact_root is None:
                raise AnalysisError("Multi-file checkpoints require an artifact root")
            record["artifacts"] = {}
            for path in artifact:
                name = str(path.relative_to(artifact_root))
                record["artifacts"][name] = sha256_file(path)
                target = checkpoint / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                os.link(path, target)
        elif artifact is not None:
            record.update(artifact=artifact.name, sha256=sha256_file(artifact))
            target = checkpoint / artifact.name
            if target.exists():
                target.unlink()
            os.link(artifact, target)
        record["record_digest"] = digest(record)
        write_json(checkpoint / f"{index:06d}.json", record)

    def _nominal(self, destination):
        threshold = read_json(self.output / "working_point/summary.json")["threshold"]
        run_correlations(
            self.output / "scored/femto", destination, self.config, threshold
        )

    def _systematics(self, destination):
        threshold = read_json(self.output / "working_point/summary.json")
        run_systematics(
            self.output / "scored/femto",
            self.output / "nominal",
            destination,
            self.config,
            threshold,
        )

    def _mc_qa(self, destination):
        from .mc import run_mc_qa

        threshold = read_json(self.output / "working_point/summary.json")["threshold"]
        run_mc_qa(self.output / "scored/mc_femto", destination, self.config, threshold)

    def _report(self, destination):
        from .report import make_report

        make_report(self.output, destination, self.config)
