"""Independent period models and bounded, transactional histogram aggregation."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from .analysis import evaluate_histograms, export_results
from .config import digest, merge, period_configs
from .errors import AnalysisError
from .execution import config_scope, stage_environment
from .io import read_json, sha256_file, write_json
from .period_merge import (
    merge_cutflows,
    merge_histograms,
    merge_mc_qa,
    merge_mixing_qa,
    merge_raw_qa,
)
from .pipeline import DEPENDENCIES, Pipeline, code_fingerprint, run_lock
from .systematics import finalize_systematics

LOG = logging.getLogger(__name__)


class MultiPeriodPipeline:
    """Use each period's own model/WP; combine selected SE/ME before normalizing.

    Pair tables stay in their period folders. The aggregate stages consume only
    sufficient statistics, so memory does not grow with the total pair count.
    """

    def __init__(self, config, *, force=False, input_roles=None):
        self.config = config
        self.output = Path(config["output"])
        self.force = force
        self.children = [
            (period_id, Pipeline(child, force=force, input_roles=input_roles))
            for period_id, child in period_configs(config)
        ]
        if not self.children:
            raise AnalysisError("Multi-period workflow requires declared periods")
        self.environment = self.children[0][1].environment
        self.code_signature = code_fingerprint()
        self.finished = set()
        self.records = [
            {"id": period_id, "relative_path": f"periods/{period_id}"}
            for period_id, _ in self.children
        ]
        self.overlap = self._input_overlap()
        if not config["synthetic_repetition"] and any(
            item["role"] == "femto" for item in self.overlap
        ):
            raise AnalysisError(
                "Femto files overlap across periods. Use distinct period inputs, "
                "or explicitly set synthetic_repetition: true for a repeated-data software test."
            )

    def _input_overlap(self):
        files = {}
        for period_id, child in self.children:
            for role, record in child.inputs.items():
                for item in record["files"]:
                    stat = Path(item["path"]).stat()
                    key = (role, stat.st_dev, stat.st_ino)
                    files.setdefault(
                        key, {"role": role, "path": item["path"], "periods": []}
                    )["periods"].append(period_id)
        return [record for record in files.values() if len(record["periods"]) > 1]

    def period_metadata(self):
        return {
            "periods": self.records,
            "synthetic_repetition": self.config["synthetic_repetition"],
            "input_overlap": self.overlap,
            "selection": "independent period ModelHandler and working point; producer pairs are never mixed across periods",
            "aggregation": "sum SE/ME, sumw2 and mass histograms; recompute normalization, purity and covariance",
            "statistical_interpretation": (
                "Repeated TestSet inputs are software diagnostics, not independent physical statistics; formal uncertainty scaling is not a measurement."
                if self.config["synthetic_repetition"]
                else "Independent-pair approximation; shared candidates/tracks/events are not modeled."
            ),
        }

    def dependency_identities(self, stage):
        result = {"periods": {}, "aggregate": {}}
        for period_id, child in self.children:
            path = child.output / stage / "manifest.json"
            manifest = read_json(path) if path.exists() else None
            result["periods"][period_id] = (
                {"signature": manifest["signature"], "artifacts": manifest["artifacts"]}
                if manifest is not None
                else None
            )
        for dependency in DEPENDENCIES[stage]:
            path = self.output / dependency / "manifest.json"
            manifest = read_json(path) if path.exists() else None
            result["aggregate"][dependency] = (
                {"signature": manifest["signature"], "artifacts": manifest["artifacts"]}
                if manifest is not None
                else None
            )
        return result

    def stage_signature(self, stage):
        return digest(
            {
                "schema_version": 1,
                "mode": "periods",
                "stage": stage,
                "config": config_scope(self.config, stage),
                "period_ids": [record["id"] for record in self.records],
                "synthetic_repetition": self.config["synthetic_repetition"],
                "code": self.code_signature,
                "environment": stage_environment(self.environment, self.config, stage),
                "dependencies": self.dependency_identities(stage),
            }
        )

    def _is_current(self, stage, seen):
        if stage in seen:
            return True
        seen.add(stage)
        path = self.output / stage / "manifest.json"
        if not path.exists():
            return False
        manifest = read_json(path)
        if manifest.get("signature") != self.stage_signature(stage):
            return False
        for period_id, child in self.children:
            if not child.is_current(stage):
                return False
        for dependency in DEPENDENCIES[stage]:
            if not self._is_current(dependency, seen):
                return False
        return all(
            (path.parent / name).is_file()
            and sha256_file(path.parent / name) == checksum
            for name, checksum in manifest.get("artifacts", {}).items()
        )

    def is_current(self, stage):
        """Validate child and aggregate artifact lineage, including tampering."""
        return self._is_current(stage, set())

    def execute(self, stage):
        with contextlib.ExitStack() as locks:
            for _, child in self.children:
                locks.enter_context(run_lock(child.output))
            self._execute(stage)

    def _execute(self, stage):
        if stage in self.finished:
            return
        for dependency in DEPENDENCIES[stage]:
            self._execute(dependency)
        for period_id, child in self.children:
            LOG.info("Period %s: %s", period_id, stage)
            child.execute(stage)
        if not self.force and self.is_current(stage):
            LOG.info("Reuse verified aggregate stage: %s", stage)
            self.finished.add(stage)
            return
        started = time.time()
        temporary = self.output / f".{stage}-periods-{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        try:
            getattr(self, "_" + stage)(temporary)
            if code_fingerprint() != self.code_signature:
                raise AnalysisError(
                    "Analysis source changed during multi-period execution; rerun after editing completes"
                )
            write_json(
                temporary / "manifest.json",
                {
                    "schema_version": 1,
                    "mode": "periods",
                    "stage": stage,
                    "signature": self.stage_signature(stage),
                    "dependencies": self.dependency_identities(stage),
                    "config": self.config,
                    "environment": self.environment,
                    "periods": self.period_metadata(),
                    "duration_seconds": time.time() - started,
                    "completed_unix": time.time(),
                    "artifacts": {
                        str(path.relative_to(temporary)): sha256_file(path)
                        for path in sorted(temporary.rglob("*"))
                        if path.is_file()
                    },
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
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.finished.add(stage)
        LOG.info("Finished aggregate %s in %.1f s", stage, time.time() - started)

    def _read_periods(self, relative):
        return [
            (period_id, read_json(child.output / relative))
            for period_id, child in self.children
        ]

    def _raw_qa(self, directory):
        from .raw_qa_output import render_raw_qa

        result = merge_raw_qa(self._read_periods("raw_qa/results.json"))
        write_json(directory / "results.json", result)
        render_raw_qa(directory, result, self.config)

    def _prepared(self, directory):
        records = self._read_periods("prepared/summary.json")
        summary = {}
        for role in self.config["inputs"]:
            summary[role] = {
                "mode": "period_source_catalog",
                "persisted": False,
                "raw_rows": sum(record[role]["raw_rows"] for _, record in records),
                "retained_rows": None,
                "periods": {period: record[role] for period, record in records},
            }
        write_json(directory / "summary.json", summary)
        write_json(directory / "periods.json", self.period_metadata())

    def _training(self, directory):
        records = [
            {
                **entry,
                "metrics": read_json(
                    self.output / entry["relative_path"] / "training/metrics.json"
                ),
            }
            for entry in self.records
        ]
        write_json(
            directory / "metrics.json",
            {
                "mode": "periods",
                "periods": records,
                "period_metadata": self.period_metadata(),
            },
        )

    def _working_point(self, directory):
        records = [
            {
                **entry,
                "summary": read_json(
                    self.output / entry["relative_path"] / "working_point/summary.json"
                ),
            }
            for entry in self.records
        ]
        write_json(
            directory / "summary.json",
            {
                "mode": "periods",
                "status": (
                    "ok"
                    if all(record["summary"]["status"] == "ok" for record in records)
                    else "partial_period_working_points"
                ),
                "periods": records,
                "thresholds_by_period": {
                    record["id"]: record["summary"]["threshold"] for record in records
                },
                "note": "Each threshold is evaluated only against its own period model scores. No common or averaged score threshold is defined.",
            },
        )

    def _scored(self, directory):
        write_json(
            directory / "dataset.json",
            {
                "kind": "period_reference_catalog",
                "score_domain": "raw_margin",
                "periods": [
                    {**entry, "scored_path": entry["relative_path"] + "/scored"}
                    for entry in self.records
                ],
                "note": "Scored Parquet stays under each period, and each pT model if configured; merge consumes selected histogram statistics.",
            },
        )

    def _nominal(self, directory):
        histograms = merge_histograms(
            self._read_periods("nominal/results.json"), self.config
        )
        export_results(
            evaluate_histograms(histograms, self.config), directory, self.config
        )
        write_json(
            directory / "cutflow.json",
            merge_cutflows(self._read_periods("nominal/cutflow.json")),
        )
        write_json(
            directory / "mixing_qa.json",
            merge_mixing_qa(self._read_periods("nominal/mixing_qa.json")),
        )
        write_json(directory / "periods.json", self.period_metadata())

    def _systematics(self, directory):
        summaries = dict(self._read_periods("systematics/summary.json"))
        variations = []
        for variation in self.config["systematics"]["variations"]:
            name, group = variation["name"], variation.get("group", variation["name"])
            outcomes = {
                period_id: next(
                    item
                    for item in summaries[period_id]["variations"]
                    if item["name"] == name
                )
                for period_id, _ in self.children
            }
            failed = [
                period
                for period, outcome in outcomes.items()
                if outcome["status"] != "completed"
            ]
            metadata = {"name": name, "group": group, "period_statuses": outcomes}
            if failed:
                variations.append(
                    (
                        {
                            **metadata,
                            "status": "failed",
                            "failed_periods": failed,
                            "reason": "Variation is unavailable in required periods: "
                            + ", ".join(failed),
                        },
                        None,
                    )
                )
                continue
            settings = merge(self.config, variation["overrides"])
            histograms = merge_histograms(
                self._read_periods(f"systematics/{name}/results.json"), settings
            )
            result = evaluate_histograms(histograms, settings)
            export_results(result, directory / name, settings)
            write_json(
                directory / name / "cutflow.json",
                merge_cutflows(self._read_periods(f"systematics/{name}/cutflow.json")),
            )
            write_json(
                directory / name / "mixing_qa.json",
                merge_mixing_qa(
                    self._read_periods(f"systematics/{name}/mixing_qa.json")
                ),
            )
            variations.append(
                (
                    {
                        **metadata,
                        "status": "completed",
                        "thresholds_by_period": {
                            period: outcome.get("threshold")
                            for period, outcome in outcomes.items()
                        },
                    },
                    result,
                )
            )
        finalize_systematics(
            read_json(self.output / "nominal/results.json"),
            variations,
            directory,
            self.config,
            execution={
                "data_scans": 0,
                "aggregation": "period sufficient statistics",
                "period_execution": {
                    period: summary["execution"]
                    for period, summary in summaries.items()
                },
            },
        )

    def _mc_qa(self, directory):
        merge_mc_qa(
            [(period_id, child.output) for period_id, child in self.children],
            directory,
            self.config,
        )

    def _report(self, directory):
        from .report import make_report

        make_report(self.output, directory, self.config)
