"""Command-line entry point. Every downstream command verifies its prerequisites."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pih3l", description="Pion–hypertriton femtoscopy analysis"
    )
    parser.add_argument(
        "command",
        choices=[
            "inspect",
            "raw-qa",
            "prepare",
            "train",
            "working-point",
            "apply",
            "correlate",
            "systematics",
            "mc-qa",
            "report",
            "run",
            "doctor",
        ],
    )
    parser.add_argument("-c", "--config", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="Machine-readable doctor report"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Legacy common root; explicit inputs.<role>.root takes precedence",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild prerequisite stages, too"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.command != "doctor" and args.config is None:
        parser.error("-c/--config is required for analysis commands")
    if args.json and args.command != "doctor":
        parser.error("--json is supported by doctor only")
    # Avoid oversubscribing shared workstations.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from .errors import AnalysisError

    try:
        if args.command == "doctor":
            from .preflight import doctor

            report = doctor(
                config_path=args.config, data_root=args.data_root, output=args.output
            )
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                for check in report["checks"]:
                    print(
                        f"{check['status'].upper()}: {check['name']}: {check['detail']}"
                    )
                print("Readiness: " + report["status"])
            return 0 if report["status"] == "passed" else 1
        from .config import load_config, period_configs
        from .io import inventory, write_json
        from .pipeline import Pipeline, run_lock

        config = load_config(args.config, data_root=args.data_root, output=args.output)
        with run_lock(config["output"]):
            handler = logging.FileHandler(Path(config["output"]) / "run.log")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logging.getLogger().addHandler(handler)
            if args.command == "run":
                from .preflight import validate_inputs

                validate_inputs(config)
            if args.command == "inspect":
                path = Path(config["output"]) / "input_inventory.json"
                inputs = (
                    {
                        period_id: inventory(child)
                        for period_id, child in period_configs(config)
                    }
                    if config["periods"]
                    else inventory(config)
                )
                write_json(path, inputs)
                print(path)
                return 0
            pipeline_type = Pipeline
            if config["periods"]:
                from .multi_period import MultiPeriodPipeline

                pipeline_type = MultiPeriodPipeline
            pipeline = pipeline_type(
                config,
                force=args.force,
                input_roles=(
                    ("femto", "background") if args.command == "raw-qa" else None
                ),
            )
            stage = {
                "raw-qa": "raw_qa",
                "prepare": "prepared",
                "train": "training",
                "working-point": "working_point",
                "apply": "scored",
                "correlate": "nominal",
                "systematics": "systematics",
                "mc-qa": "mc_qa",
                "report": "report",
                "run": "report",
            }[args.command]
            pipeline.execute(stage)
            print(Path(config["output"]) / stage)
            return 0
    except (AnalysisError, OSError, ValueError) as error:
        logging.error("%s", error)
        return 2
