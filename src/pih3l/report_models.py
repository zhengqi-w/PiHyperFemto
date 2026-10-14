"""Explicit period/model contexts for QA links and unique ROOT namespaces."""

from pathlib import Path

from .io import read_json


def model_contexts(output, training):
    output = Path(output)
    periods = training["periods"] if training.get("mode") == "periods" else [None]
    for period in periods:
        period_id = period["id"] if period is not None else None
        base = output / period["relative_path"] if period is not None else output
        metrics = (
            read_json(base / "training/metrics.json")
            if period is not None
            else training
        )
        models = (
            metrics["models"]
            if metrics.get("mode") == "pt_bins"
            else [{"id": "global", "relative_path": "."}]
        )
        for model in models:
            relative = model["relative_path"]
            names = ([period_id] if period_id is not None else []) + (
                [model["id"]] if relative != "." else []
            )
            suffix = "__" + "__".join(names) if names else ""
            if period_id is not None:
                # Length-delimited namespaces remain unique even when IDs
                # contain '__' or words resembling the model/global markers.
                suffix = f"__p{len(period_id)}_{period_id}"
                suffix += (
                    f"__m{len(model['id'])}_{model['id']}"
                    if relative != "."
                    else "__global"
                )
            yield {
                "period_id": period_id,
                "model_id": model["id"],
                "output": base,
                "relative_path": relative,
                "root_prefix": f"periods/{period_id}/" if period_id is not None else "",
                "suffix": suffix,
                "label": " (" + ", ".join(names) + ")" if names else "",
            }
