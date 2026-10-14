"""Bounded ROOT reads, canonical Parquet batches, and atomic JSON artifacts."""

from __future__ import annotations

import glob
import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

from .errors import AnalysisError
from .schema import TREE_NAMES, adapt, canonical_name


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(config: dict, role: str) -> list[Path]:
    spec = config["inputs"][role]
    if "root" in spec and spec["root"] is None:
        if spec["required"]:
            raise AnalysisError(
                f"Required input {role} is explicitly disabled with root: null"
            )
        return []
    roots = spec.get("root", config["data_root"])
    roots = roots if isinstance(roots, list) else [roots]
    paths = set()
    for pattern in spec["patterns"]:
        relative = Path(os.path.expandvars(os.path.expanduser(pattern)))
        for root in roots:
            path = (
                relative
                if relative.is_absolute()
                else Path(os.path.expandvars(os.path.expanduser(root))) / relative
            )
            paths.update(
                Path(p).resolve()
                for p in glob.glob(str(path), recursive=True)
                if Path(p).is_file()
            )
    if not paths and spec["required"]:
        raise AnalysisError(
            f"No files found for required input {role}: {spec['patterns']}"
        )
    # Resolve symlinks and deduplicate hard links as the same physical ROOT file.
    # Copied files retain distinct inodes and remain independent source entries.
    physical = {}
    for path in sorted(paths):
        stat = path.stat()
        physical.setdefault((stat.st_dev, stat.st_ino), path)
    return list(physical.values())


def tree_paths(path, role):
    with uproot.open(path) as root:
        paths = [
            key.split(";")[0]
            for key, cls in root.classnames(recursive=True).items()
            if cls in {"TTree", "ROOT::RNTuple"}
            and key.split(";")[0].split("/")[-1].lower() in TREE_NAMES[role]
        ]
    if not paths:
        raise AnalysisError(
            f"No supported {role} tree in {path}; expected {sorted(TREE_NAMES[role])}"
        )
    return sorted(set(paths))


def inventory(config, *, roles=None):
    result = {}
    for role in config["inputs"] if roles is None else roles:
        files = []
        for path in discover_files(config, role):
            stat = path.stat()
            item = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "trees": [],
            }
            if config["io"]["fingerprint"] == "sha256":
                item["sha256"] = sha256_file(path)
            paths = tree_paths(path, role)
            with uproot.open(path) as root:
                for tree_path in paths:
                    tree = root[tree_path]
                    item["trees"].append(
                        {
                            "path": tree_path,
                            "rows": tree.num_entries,
                            "branches": list(tree.keys()),
                        }
                    )
            files.append(item)
        result[role] = {
            "status": "available" if files else "missing_optional",
            "files": files,
        }
    return result


def root_shards(config, role, *, with_tree=False):
    """Enumerate bounded entry ranges without decoding baskets (resume-safe)."""
    index = 0
    for path in discover_files(config, role):
        paths = tree_paths(path, role)
        with uproot.open(path) as root:
            for tree_path in paths:
                tree = root[tree_path]
                rows = tree.num_entries
                for start in range(0, rows, config["io"]["chunk_rows"]):
                    shard = {
                        "path": str(path),
                        "tree": tree_path,
                        "start": start,
                        "stop": min(start + config["io"]["chunk_rows"], rows),
                    }
                    yield (index, shard, tree) if with_tree else (index, shard)
                    index += 1


def read_root_shard(
    config, role, shard, *, columns=None, optional_columns=(), tree=None
):
    """Read canonical columns; explicitly requested unavailable columns are errors."""
    with (
        uproot.open(shard["path"]) if tree is None else contextlib.nullcontext()
    ) as root:
        tree = root[shard["tree"]] if tree is None else tree
        branches = list(tree.keys())
        if columns is not None:
            available = {canonical_name(name): name for name in branches}
            requested = set(columns)
            # ChargeChannel and true-pion flags can be synthesized by the adapter.
            if "ChargeChannel" in requested and "ChargeChannel" not in available:
                requested.remove("ChargeChannel")
                requested.update(("IsMatter", "PairPiTrackSign"))
            if (
                "PairPiIsTruePion" in requested
                and "PairPiIsTruePion" not in available
                and "PairPiPdgCode" in available
            ):
                requested.remove("PairPiIsTruePion")
                requested.add("PairPiPdgCode")
            missing = requested - set(available)
            if missing:
                raise AnalysisError(
                    f"{role} ROOT projection: missing required columns {sorted(missing)}"
                )
            requested |= set(optional_columns) & set(available)
            branches = [name for name in branches if canonical_name(name) in requested]
        data = tree.arrays(
            expressions=branches,
            entry_start=shard["start"],
            entry_stop=shard["stop"],
            library="np",
        )
        frame = adapt(pd.DataFrame(data), role)
        frame["_source_file"] = shard["path"]
        frame["_source_tree"] = shard["tree"]
        frame["_entry"] = np.arange(shard["start"], shard["stop"], dtype=np.int64)
        return frame


def iter_root(config, role, *, columns=None, optional_columns=()):
    """Each yield is at most chunk_rows rows. All ROOT handles are scoped."""
    for _, shard, tree in root_shards(config, role, with_tree=True):
        yield read_root_shard(
            config,
            role,
            shard,
            columns=columns,
            optional_columns=optional_columns,
            tree=tree,
        )


def parquet_paths(directory):
    """Read both flat global tables and model-partitioned scored subdirectories."""
    return sorted(Path(directory).rglob("*.parquet"))


def parquet_columns(directory):
    """Return the common schema names; heterogeneous schemas fail explicitly."""
    names = None
    for path in parquet_paths(directory):
        current = set(pq.ParquetFile(path).schema_arrow.names)
        if names is not None and current != names:
            raise AnalysisError(
                f"Inconsistent Parquet columns in {directory}: {path.name}"
            )
        names = current
    return names or set()


def iter_parquet(directory, *, columns=None, optional_columns=(), batch_size=200000):
    if hasattr(directory, "iter_batches"):
        yield from directory.iter_batches(
            columns=columns, optional_columns=optional_columns, batch_size=batch_size
        )
        return
    for path in parquet_paths(directory):
        parquet = pq.ParquetFile(path)
        selected = None
        if columns is not None:
            available = set(parquet.schema_arrow.names)
            missing = set(columns) - available
            if missing:
                raise AnalysisError(
                    f"{path}: missing required Parquet columns {sorted(missing)}"
                )
            selected = sorted(set(columns) | (set(optional_columns) & available))
        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
            yield batch.to_pandas()


def write_batch(directory, index, frame):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"part-{index:06d}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return path
