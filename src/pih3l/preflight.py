"""Farm readiness checks without training, pair scans or full-file hashing."""

from __future__ import annotations

import copy
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "uproot": "uproot",
    "awkward": "awkward",
    "pyarrow": "pyarrow",
    "PyYAML": "yaml",
    "packaging": "packaging",
    "xgboost": "xgboost",
    "hipe4ml": "hipe4ml.model_handler",
    "optuna": "optuna",
    "matplotlib": "matplotlib",
    "shap": "shap",
    "lightgbm": "lightgbm",
    "ipython": "IPython",
    "joblib": "joblib",
}


def _check(report, name, status, detail):
    report["checks"].append({"name": name, "status": status, "detail": str(detail)})


def _environment(report):
    report["environment"] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": {},
    }
    _check(
        report,
        "python",
        "passed" if sys.version_info >= (3, 10) else "failed",
        sys.version,
    )
    for distribution, module in IMPORTS.items():
        try:
            importlib.import_module(module)
            version = metadata.version(distribution)
            report["environment"]["versions"][distribution] = version
            _check(report, f"dependency/{distribution}", "passed", version)
        except Exception as error:
            _check(report, f"dependency/{distribution}", "failed", error)
    try:
        from packaging.requirements import Requirement

        parents = ["hipe4ml", "shap"]
        try:
            metadata.distribution("pih3l-femto")
            parents.append("pih3l-femto")
        except metadata.PackageNotFoundError:
            _check(
                report,
                "package_installation",
                "warning",
                "Source-tree execution; install the package to check its declared version requirements.",
            )
        for parent in parents:
            for specification in metadata.requires(parent) or []:
                requirement = Requirement(specification)
                if requirement.marker and not requirement.marker.evaluate(
                    {"extra": ""}
                ):
                    continue
                try:
                    version = metadata.version(requirement.name)
                    status = (
                        "passed"
                        if requirement.specifier.contains(version)
                        else "failed"
                    )
                    _check(
                        report,
                        f"requirement/{parent}/{requirement.name}",
                        status,
                        f"{specification}; installed {version}",
                    )
                except metadata.PackageNotFoundError as error:
                    _check(
                        report,
                        f"requirement/{parent}/{requirement.name}",
                        "failed",
                        error,
                    )
    except Exception as error:
        _check(report, "dependency_requirements", "failed", error)
    # Keep native ROOT output isolated so --json remains machine readable.
    program = r"""
import json,tempfile
from pathlib import Path
import ROOT
ROOT.gROOT.SetBatch(True)
assert hasattr(ROOT,"RooCrystalBall") and hasattr(ROOT.RooFit,"IntegrateBins")
with tempfile.TemporaryDirectory(prefix="pih3l-doctor-") as temporary:
    directory=Path(temporary)
    with ROOT.TFile.Open(str(directory/"probe.root"),"RECREATE") as archive:
        histogram=ROOT.TH1D("probe","",4,0,1)
        histogram.Fill(0.25)
        histogram.Write()
        canvas=ROOT.TCanvas("probe_canvas","",400,300)
        histogram.Draw("HIST")
        canvas.SaveAs(str(directory/"probe.pdf"))
        canvas.Write()
        canvas.Close()
    with ROOT.TFile.Open(str(directory/"probe.root"),"READ") as archive:
        assert not archive.IsZombie() and archive.Get("probe_canvas")
    assert (directory/"probe.pdf").read_bytes().startswith(b"%PDF-")
print(json.dumps({"ROOT":ROOT.gROOT.GetVersion(),"capabilities":["PyROOT","RooCrystalBall","IntegrateBins","ROOT write/reopen","PDF"]}))
"""
    try:
        native = subprocess.run(
            [sys.executable, "-B", "-c", program],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if native.returncode:
            raise RuntimeError(native.stderr or native.stdout)
        details = json.loads(native.stdout.strip().splitlines()[-1])
        report["environment"].update(details)
        _check(report, "ROOT/PDF/RooFit", "passed", details)
    except Exception as error:
        _check(report, "ROOT/PDF/RooFit", "failed", error)


def _input_contract(config, report, prefix=""):
    from .execution import (
        DERIVED,
        ITS_DERIVED,
        preparation_columns,
        scored_columns,
        training_features,
    )
    from .io import inventory, read_root_shard
    from .physics import derive
    from .schema import CANDIDATE_COLUMNS, require_columns

    bounded = copy.deepcopy(config)
    bounded["io"]["fingerprint"] = "stat"
    roles = {}
    for role in config["inputs"]:
        try:
            record = inventory(bounded, roles=[role])[role]
            roles[role] = record
            trees = [(file, tree) for file in record["files"] for tree in file["trees"]]
            rows = sum(tree["rows"] for _, tree in trees)
            status = (
                "passed"
                if rows
                else ("failed" if config["inputs"][role]["required"] else "warning")
            )
            _check(
                report,
                prefix + f"input/{role}",
                status,
                f"{len(record['files'])} files, {len(trees)} trees, {rows} rows; stat-only inventory",
            )
            required, optional = preparation_columns(config, role)
            required, optional = set(required), set(optional)
            consumers = set(training_features(config))
            if role in {"background", "signal"}:
                # Streaming candidate readers request their own training/WP
                # selections. Femto systematic cuts are not training cuts.
                consumers.update(config["training"]["candidate_cuts"])
                for model in config["training"]["pt_bins"]:
                    if isinstance(model, dict):
                        consumers.update(model.get("candidate_cuts", {}))
                wp = config["working_point"]
                if role == "background" or wp["mass_fit"]["calibrate_mc_tails"]:
                    consumers.update(
                        config["analysis"]["candidate_cuts"]
                        if wp["candidate_cuts"] is None
                        else wp["candidate_cuts"]
                    )
                    if wp["event_cuts"] is not None:
                        consumers.update(wp["event_cuts"])
                    else:
                        optional.update(config["analysis"]["event_cuts"])
                if not config["io"]["persist_prepared"]:
                    required = set(CANDIDATE_COLUMNS) | {"HypCentFT0C"}
                    if role == "signal":
                        required |= {"HypIsReco", "HypIsSignal", "HypIsTwoBodyDecay"}
                    required.update(
                        ITS_DERIVED.get(name, name) for name in consumers - DERIVED
                    )
                    optional = {
                        ITS_DERIVED.get(name, name) for name in optional - DERIVED
                    } - required
                # Persisted preparation must use its actual projection, even if
                # it retains extra columns. Missing downstream fields are then
                # checked on that derived frame, never injected into projection.
            schemas = set()
            for file, tree in trees:
                label = prefix + f"schema/{role}/{file['path']}:{tree['path']}"
                if not tree["rows"]:
                    _check(
                        report, label, "warning", "Empty tree; no sample rows to check."
                    )
                    continue
                shard = {
                    "path": file["path"],
                    "tree": tree["path"],
                    "start": 0,
                    "stop": min(32, tree["rows"]),
                }
                frame = read_root_shard(
                    bounded,
                    role,
                    shard,
                    columns=(
                        sorted(required) if config["io"]["root_projection"] else None
                    ),
                    optional_columns=sorted(optional),
                )
                frame = derive(
                    frame, config["physics"], pairs=role in {"femto", "mc_femto"}
                )
                require_columns(frame, consumers, label + " consumer contract")
                if role in {"femto", "mc_femto"}:
                    available = set(frame.columns) | {
                        "bdt_score",
                        "bdt_model_id",
                        "bdt_in_scope",
                    }
                    compact = scored_columns(config, role, available)
                    schemas.add(
                        tuple(compact)
                        if config["io"]["slim_scored"]
                        else tuple(sorted(available))
                    )
                _check(
                    report,
                    label,
                    "passed",
                    f"Configured nominal/variation projection and derived observables; decoded at most {len(frame)} rows",
                )
            if len(schemas) > 1:
                raise ValueError(
                    f"{role}: inconsistent scored schemas across ROOT trees; use one producer schema, or remove heterogeneous optional columns from io.scored_optional_columns with slim_scored enabled. MC truth fields must also be consistent."
                )
            if role in {"femto", "mc_femto"} and trees:
                _check(
                    report,
                    prefix + f"scored_schema/{role}",
                    "passed",
                    "Uniform configured columns across nonempty trees",
                )
        except Exception as error:
            _check(report, prefix + f"input_contract/{role}", "failed", error)
    return roles


def _output(config, inputs, report, prefix=""):
    output = Path(config["output"]).resolve()
    try:
        for record in inputs.values():
            for file in record["files"]:
                if Path(file["path"]).resolve().is_relative_to(output):
                    raise ValueError(
                        "Output contains input ROOT files; select a separate results directory."
                    )
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".doctor-", dir=output) as temporary:
            path = Path(temporary) / "write_probe"
            path.write_text("pih3l output probe\n")
            path.replace(path.with_suffix(".renamed"))
        _check(
            report,
            prefix + "output/writable",
            "passed",
            f"Write and atomic rename succeeded: {output}",
        )
    except Exception as error:
        _check(report, prefix + "output/writable", "failed", error)


def validate_inputs(config):
    """Fail early on Farm input/schema errors before any full workflow stage."""
    from .config import period_configs
    from .errors import AnalysisError

    report = {"checks": []}
    for name, child in period_configs(config) or [("single", config)]:
        _input_contract(child, report, name + "/")
    failed = [entry for entry in report["checks"] if entry["status"] == "failed"]
    if failed:
        raise AnalysisError(
            "Input preflight failed: " + "; ".join(entry["detail"] for entry in failed)
        )
    return report


def doctor(config=None, *, config_path=None, data_root=None, output=None):
    """Return a JSON-safe readiness report; no analysis artifacts are created."""
    report = {
        "status": "passed",
        "checks": [],
        "environment": {},
        "inputs": {},
        "config": None,
    }
    _environment(report)
    if config is None and config_path is not None:
        try:
            from .config import load_config

            config = load_config(config_path, data_root=data_root, output=output)
        except Exception as error:
            _check(report, "config", "failed", error)
    if config is not None:
        from .config import period_configs

        children = period_configs(config)
        report["config"] = {
            "path": str(config_path) if config_path else None,
            "output": config["output"],
            "mode": "periods" if children else "single",
        }
        if children:
            for name, child in children:
                inputs = _input_contract(child, report, f"period/{name}/")
                report["inputs"][name] = inputs
                _output(child, inputs, report, f"period/{name}/")
            _output(config, {}, report)
        else:
            report["inputs"] = _input_contract(config, report)
            _output(config, report["inputs"], report)
    if any(check["status"] == "failed" for check in report["checks"]):
        report["status"] = "failed"
    return report
