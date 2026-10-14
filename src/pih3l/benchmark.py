"""Isolated ROOT I/O/kinematics benchmark; no training or physics fit is run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from .config import digest, load_config, merge
from .io import (
    discover_files,
    iter_parquet,
    parquet_columns,
    read_json,
    tree_paths,
    write_json,
)
from .physics import derive
from .schema import CANDIDATE_COLUMNS, canonical_name


def _inputs(config, max_trees):
    result = []
    for path in discover_files(config, "femto"):
        for tree in tree_paths(path, "femto"):
            result.append((str(path), tree))
            if max_trees and len(result) >= max_trees:
                return result
    return result


def _numpy_worker(config, inputs, projected):
    import uproot
    from .io import read_root_shard

    columns = CANDIDATE_COLUMNS + [
        "HypCentFT0C",
        "ChargeChannel",
        "IsMixed",
        "PairPiTrackPt",
        "PairPiTrackEta",
        "PairPiTrackPhi",
    ]
    edges = np.asarray(config["analysis"]["kstar_edges"])
    hist = np.zeros((len(config["analysis"]["centrality_bins"]), 4, 2, len(edges) - 1))
    raw, retained = 0, 0
    for path, tree_name in inputs:
        with uproot.open(path) as root:
            entries = root[tree_name].num_entries
        for start in range(0, entries, config["io"]["chunk_rows"]):
            shard = {
                "path": path,
                "tree": tree_name,
                "start": start,
                "stop": min(start + config["io"]["chunk_rows"], entries),
            }
            frame = derive(
                read_root_shard(
                    config, "femto", shard, columns=columns if projected else None
                ),
                config["physics"],
                pairs=True,
            )
            raw += len(frame)
            finite = np.isfinite(
                frame[["mass_h3l", "pt_h3l", "ct", "cos_pa", "kstar"]]
            ).all(axis=1)
            keep = (
                finite
                & frame.mass_h3l.between(
                    *config["analysis"]["mass_range"], inclusive="left"
                )
                & frame.kstar.between(edges[0], edges[-1], inclusive="left")
            )
            frame = frame.loc[keep]
            for i, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
                central = frame.loc[
                    frame.HypCentFT0C.between(low, high, inclusive="left")
                ]
                retained += len(central)
                for ch in range(4):
                    for mixed in range(2):
                        hist[i, ch, mixed] += np.histogram(
                            central.loc[
                                (central.ChargeChannel == ch)
                                & (central.IsMixed == bool(mixed)),
                                "kstar",
                            ],
                            edges,
                        )[0]
    return {"raw_rows": raw, "retained_rows": retained, "histograms": hist.tolist()}


_CPP = r"""
#include <array>
#include <cmath>
std::array<double,5> pih3l_benchmark_kinematics(
 double ah,double eh,double fh,double ap,double ep,double fp,
 double x,double y,double z,double dx,double dy,double dz,
 double bp,double be,double bf,double mh,double mpi,double m0,double scale,bool nominal) {
 const double hx=ah*scale*cos(fh)+ap*cos(fp), hy=ah*scale*sin(fh)+ap*sin(fp), hz=ah*scale*sinh(eh)+ap*sinh(ep);
 const double he=sqrt(pow(ah*scale*cosh(eh),2)+mh*mh), pe=sqrt(pow(ap*cosh(ep),2)+mpi*mpi);
 const double p2=hx*hx+hy*hy+hz*hz, mass=sqrt(std::max(0.,(he+pe)*(he+pe)-p2));
 dx-=x; dy-=y; dz-=z; const double length=sqrt(dx*dx+dy*dy+dz*dz), p=sqrt(p2);
 const double ct=length*m0/p, rawcos=(dx*hx+dy*hy+dz*hz)/(length*p);
 const double cospa=std::isfinite(rawcos)?std::max(-1.,std::min(1.,rawcos)):NAN;
 const double px=bp*cos(bf), py=bp*sin(bf), pz=bp*sinh(be), en=sqrt(px*px+py*py+pz*pz+mpi*mpi);
 const double hen=nominal?sqrt(p2+m0*m0):he+pe;
 const double ma=sqrt(std::max(0.,hen*hen-p2));
 const double s=(hen+en)*(hen+en)-pow(hx+px,2)-pow(hy+py,2)-pow(hz+pz,2);
 const double lam=(s-pow(ma+mpi,2))*(s-pow(ma-mpi,2));
 const double ks=s>0 && lam>=-1e-8?sqrt(std::max(0.,lam)/(4*s)):NAN;
 return {mass,sqrt(hx*hx+hy*hy),ct,cospa,ks};
}
"""


def benchmark_ensemble(config, scored, output):
    """Compare serial vs shared reads without writing new physics results."""
    from .analysis import CorrelationAccumulator, analysis_columns, histogram_signature
    from .working_point import resolve_wp_variation

    scored = Path(scored)
    wp = read_json(scored.parent.parent / "working_point/summary.json")
    available = parquet_columns(scored)
    requests = []
    for variation in config["systematics"]["variations"]:
        settings = merge(config, variation["overrides"])
        threshold = settings["analysis"]["bdt_threshold"]
        if "wp_efficiency_delta" in variation:
            threshold = resolve_wp_variation(wp, variation["wp_efficiency_delta"])
        elif threshold is None:
            threshold = wp["threshold"]
        requests.append((variation["name"], settings, threshold))
    results = {}
    for shared in (False, True):
        started = time.perf_counter()
        accumulators = {}
        mapping = {}
        for name, settings, threshold in requests:
            key = histogram_signature(settings, threshold) if shared else name
            mapping[name] = key
            accumulators.setdefault(key, CorrelationAccumulator(settings, threshold))
        if shared and accumulators:
            columns = sorted(
                {
                    column
                    for value in accumulators.values()
                    for column in analysis_columns(value.config, available)
                }
            )
            for frame in iter_parquet(
                scored, columns=columns, batch_size=config["io"]["chunk_rows"]
            ):
                for value in accumulators.values():
                    value.add(frame)
        elif not shared:
            for value in accumulators.values():
                for frame in iter_parquet(
                    scored,
                    columns=analysis_columns(value.config, available),
                    batch_size=config["io"]["chunk_rows"],
                ):
                    value.add(frame)
        elapsed = time.perf_counter() - started
        results["shared" if shared else "serial"] = {
            "elapsed_seconds": elapsed,
            "data_scans": 1 if shared and accumulators else len(accumulators),
            "histogram_groups": len(accumulators),
            "histogram_checksums": {
                name: digest(accumulators[key].histograms)
                for name, key in mapping.items()
            },
            "cutflows": {
                name: accumulators[key].cutflow for name, key in mapping.items()
            },
        }
    equivalent = (
        results["serial"]["histogram_checksums"]
        == results["shared"]["histogram_checksums"]
        and results["serial"]["cutflows"] == results["shared"]["cutflows"]
    )
    summary = {
        "scope": "Selection and histogram sufficient statistics from existing scored shards, no new training/WP/physics fits. Shared case includes one nominal-like histogram group rather than loading nominal statistics.",
        "scored_directory": str(scored.resolve()),
        "equivalent": equivalent,
        "results": results,
        "speedup": results["serial"]["elapsed_seconds"]
        / results["shared"]["elapsed_seconds"],
    }
    write_json(output, summary)
    return 0 if equivalent else 2


def _rdf_worker(config, inputs, threads):
    import ROOT
    import uproot
    from array import array

    ROOT.gROOT.SetBatch(True)
    if threads > 1:
        ROOT.EnableImplicitMT(threads)
    with uproot.open(inputs[0][0]) as root:
        branches = {canonical_name(key): key for key in root[inputs[0][1]].keys()}
    chain = ROOT.TChain(inputs[0][1])
    for path, tree in inputs:
        with uproot.open(path) as root:
            if root[tree].classname != "TTree":
                raise ValueError("RDF benchmark currently requires TTree inputs")
            if {canonical_name(key): key for key in root[tree].keys()} != branches:
                raise ValueError(
                    "RDF benchmark requires identical source branch schemas"
                )
        chain.AddFile(path, ROOT.TTree.kMaxEntries, tree)
    ROOT.gInterpreter.Declare(_CPP)
    node = ROOT.RDataFrame(chain)
    for canonical, branch in branches.items():
        if canonical != branch:
            node = node.Alias(canonical, branch)
    if "ChargeChannel" not in branches:
        node = node.Define(
            "ChargeChannel",
            "IsMatter ? (PairPiTrackSign>0?0:1) : (PairPiTrackSign<0?2:3)",
        )
    p = config["physics"]
    arguments = CANDIDATE_COLUMNS + [
        "PairPiTrackPt",
        "PairPiTrackEta",
        "PairPiTrackPhi",
    ]
    constants = [p["mass_he3"], p["mass_pi"], p["mass_h3l"], p["he3_momentum_scale"]]
    expression = (
        "pih3l_benchmark_kinematics("
        + ",".join(
            arguments
            + [repr(x) for x in constants]
            + ["true" if p["pair_mass"] == "nominal" else "false"]
        )
        + ")"
    )
    raw = node.Count()
    node = node.Define("bench_kin", expression)
    node = node.Filter(
        "std::isfinite(bench_kin[0]) && std::isfinite(bench_kin[1]) && std::isfinite(bench_kin[2]) && std::isfinite(bench_kin[3]) && std::isfinite(bench_kin[4])"
    )
    node = node.Define("bench_kstar", "bench_kin[4]")
    edges = config["analysis"]["kstar_edges"]
    lo, hi = config["analysis"]["mass_range"]
    node = node.Filter(
        f"bench_kin[0]>={lo} && bench_kin[0]<{hi} && bench_kstar>={edges[0]} && bench_kstar<{edges[-1]}"
    )
    actions, retained = [], []
    for i, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
        central = node.Filter(f"HypCentFT0C>={low} && HypCentFT0C<{high}")
        retained.append(central.Count())
        for ch in range(4):
            for mixed in range(2):
                filtered = central.Filter(f"ChargeChannel=={ch} && IsMixed=={mixed}")
                actions.append(
                    filtered.Histo1D(
                        (
                            f"bench_{i}_{ch}_{mixed}",
                            "",
                            len(edges) - 1,
                            array("d", edges),
                        ),
                        "bench_kstar",
                    )
                )
    raw_rows = int(raw.GetValue())  # All booked actions run in one event loop.
    hist = np.array(
        [
            [action.GetValue().GetBinContent(i + 1) for i in range(len(edges) - 1)]
            for action in actions
        ]
    )
    hist = hist.reshape(len(config["analysis"]["centrality_bins"]), 4, 2, -1)
    return {
        "raw_rows": raw_rows,
        "retained_rows": sum(int(x.GetValue()) for x in retained),
        "histograms": hist.tolist(),
        "root_version": ROOT.gROOT.GetVersion(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-trees", type=int, default=3, help="0 selects all Femto trees"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", choices=["uproot_all", "uproot_projected", "rdf"])
    parser.add_argument(
        "--scored",
        type=Path,
        help="Benchmark serial/shared variation scans of existing scored/femto instead",
    )
    args = parser.parse_args(argv)
    if args.max_trees < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("max-trees must be nonnegative; repeats/threads must be positive")
    config = load_config(args.config)
    if args.scored:
        return benchmark_ensemble(config, args.scored, args.output)
    inputs = _inputs(config, args.max_trees)
    if not inputs:
        parser.error("No Femto trees selected")
    if args.worker:
        started = time.perf_counter()
        result = (
            _rdf_worker(config, inputs, args.threads)
            if args.worker == "rdf"
            else _numpy_worker(config, inputs, args.worker == "uproot_projected")
        )
        elapsed = time.perf_counter() - started
        result.update(
            backend=args.worker,
            threads=args.threads,
            elapsed_seconds=elapsed,
            raw_rows_per_second=result["raw_rows"] / elapsed,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        )
        write_json(args.output, result)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for repeat in range(args.repeats):
        # Alternate order to reduce bias; this is not a cold disk-cache benchmark.
        cases = [
            ("uproot_all", 1),
            ("uproot_projected", 1),
            ("rdf", 1),
            ("rdf", args.threads),
        ]
        if repeat % 2:
            cases.reverse()
        for backend, threads in cases:
            destination = args.output / f"{backend}_t{threads}_r{repeat}.json"
            command = [
                sys.executable,
                "-m",
                "pih3l.benchmark",
                "-c",
                str(args.config.resolve()),
                "--output",
                str(destination.resolve()),
                "--max-trees",
                str(args.max_trees),
                "--worker",
                backend,
                "--threads",
                str(threads),
            ]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            if completed.returncode:
                results.append(
                    {
                        "backend": backend,
                        "threads": threads,
                        "repeat": repeat,
                        "status": "failed",
                        "reason": completed.stderr[-4000:],
                    }
                )
                continue
            result = json.loads(destination.read_text())
            result.update(repeat=repeat, status="completed")
            results.append(result)
            print(
                f"{backend} threads={threads}: {result['elapsed_seconds']:.3f}s",
                flush=True,
            )
    valid = [r for r in results if r["status"] == "completed"]
    reference = next((r for r in valid if r["backend"] == "uproot_all"), None)
    for result in valid:
        result["matches_reference"] = (
            reference is not None
            and result["raw_rows"] == reference["raw_rows"]
            and result["retained_rows"] == reference["retained_rows"]
            and np.array_equal(result["histograms"], reference["histograms"])
        )
    summary = {
        "scope": "ROOT reading, double-precision reconstructed kinematics, preparation selection and centrality/channel SE/ME kstar counts; excludes BDT, WP, fits and Parquet writing",
        "cache": "uncontrolled OS cache, fresh process per case; ROOT import/JIT included in worker elapsed",
        "inputs": inputs,
        "results": results,
        "equivalent": bool(reference)
        and len(valid) == len(results)
        and all(r["matches_reference"] for r in valid),
    }
    write_json(args.output / "summary.json", summary)
    return 0 if summary["equivalent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
