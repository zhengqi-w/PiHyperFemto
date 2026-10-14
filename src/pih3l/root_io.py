"""ROOT primitives and ownership for reproducible, editable analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .errors import AnalysisError
from .io import json_safe


def load_root():
    """Use the analysis environment's PyROOT; never fall back to another renderer."""
    try:
        import ROOT
    except ImportError as error:
        raise AnalysisError(
            "PyROOT is required for ROOT output and plots. Use a ROOT-enabled Python "
            "environment (the existing MLenv is supported)."
        ) from error
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    return ROOT


class RootArchive:
    """Keep primitives alive through canvas serialization; detach every histogram.

    NaNs are preserved in numerical objects. Graphs omit invalid points, and
    independent validity objects are written by the caller. Histograms rebuilt
    from summaries have Entries=0 unless an actual unweighted row count is known.
    """

    def __init__(self, path):
        self.ROOT = load_root()
        self.path = Path(path)
        self.objects = {}
        self.references = []
        self.canvases = []
        self.file = self.ROOT.TFile.Open(str(self.path), "RECREATE")
        if not self.file or self.file.IsZombie():
            raise AnalysisError(f"Cannot create ROOT archive: {self.path}")
        self.file.SetCompressionLevel(4)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for canvas in self.canvases:
            canvas.Close()
        self.file.Close()
        self.references.clear()
        self.objects.clear()

    def keep(self, obj):
        self.references.append(obj)
        return obj

    def write(self, path, obj):
        parts = path.split("/")
        directory = self.file
        for part in parts[:-1]:
            directory = directory.GetDirectory(part) or directory.mkdir(part)
        if directory.WriteTObject(obj, parts[-1], "Overwrite") <= 0:
            raise AnalysisError(f"Failed to write ROOT object: {path}")
        self.objects[path] = obj
        return obj

    def json(self, path, value):
        encoded = json.dumps(json_safe(value), allow_nan=False, sort_keys=True)
        return self.write(path, self.ROOT.TObjString(encoded))

    def hist1(self, path, edges, values, *, variance=None, title="", entries=0):
        edges, values = np.asarray(edges, dtype="float64"), np.asarray(
            values, dtype="float64"
        )
        if values.shape != (len(edges) - 1,):
            raise AnalysisError(f"ROOT histogram dimensions differ: {path}")
        if variance is not None:
            variance = np.asarray(variance, dtype=float)
            if variance.shape != values.shape or np.any(variance < 0):
                raise AnalysisError(
                    f"ROOT histogram variance must match bins and be nonnegative: {path}"
                )
        hist = self.ROOT.TH1D(path.replace("/", "__"), title, len(values), edges)
        hist.SetDirectory(0)
        hist.Sumw2()
        errors = (
            np.zeros(len(values))
            if variance is None
            else np.sqrt(np.asarray(variance, float))
        )
        for index, (value, error) in enumerate(zip(values, errors), start=1):
            hist.SetBinContent(index, float(value))
            hist.SetBinError(index, float(error))
        hist.SetEntries(float(entries))
        return self.write(path, hist)

    def hist2(self, path, xedges, yedges, values, *, variance=None, title=""):
        xedges, yedges = np.asarray(xedges, dtype="float64"), np.asarray(
            yedges, dtype="float64"
        )
        values = np.asarray(values, dtype=float)
        if values.shape != (len(xedges) - 1, len(yedges) - 1):
            raise AnalysisError(f"ROOT matrix dimensions differ: {path}")
        variance = (
            np.zeros_like(values) if variance is None else np.asarray(variance, float)
        )
        if variance.shape != values.shape or np.any(variance < 0):
            raise AnalysisError(
                f"ROOT matrix variance must match bins and be nonnegative: {path}"
            )
        hist = self.ROOT.TH2D(
            path.replace("/", "__"),
            title,
            len(xedges) - 1,
            xedges,
            len(yedges) - 1,
            yedges,
        )
        hist.SetDirectory(0)
        hist.Sumw2()
        for ix in range(values.shape[0]):
            for iy in range(values.shape[1]):
                hist.SetBinContent(ix + 1, iy + 1, float(values[ix, iy]))
                hist.SetBinError(
                    hist.GetBin(ix + 1, iy + 1), float(np.sqrt(variance[ix, iy]))
                )
        hist.SetEntries(0)
        return self.write(path, hist)

    def graph(self, path, x, y, *, xerror=None, yerror=None, valid=None, title=""):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if valid is not None:
            mask &= np.asarray(valid, bool)
        if yerror is None:
            graph = self.ROOT.TGraph()
        else:
            yerror = np.asarray(yerror, dtype=float)
            xerror = (
                np.zeros(len(x)) if xerror is None else np.asarray(xerror, dtype=float)
            )
            mask &= np.isfinite(yerror) & (yerror >= 0) & np.isfinite(xerror)
            graph = self.ROOT.TGraphErrors()
        graph.SetName(path.replace("/", "__"))
        graph.SetTitle(title)
        for point, index in enumerate(np.flatnonzero(mask)):
            graph.SetPoint(point, float(x[index]), float(y[index]))
            if yerror is not None:
                graph.SetPointError(point, float(xerror[index]), float(yerror[index]))
        return self.write(path, graph)

    def asymmetric_graph(self, path, edges, fraction):
        edges = np.asarray(edges, float)
        values = np.asarray(fraction["value"], float)
        low, high = np.asarray(fraction["low"], float), np.asarray(
            fraction["high"], float
        )
        valid = np.asarray(fraction["valid"], bool) & np.isfinite(values + low + high)
        graph = self.ROOT.TGraphAsymmErrors()
        graph.SetName(path.replace("/", "__"))
        for point, index in enumerate(np.flatnonzero(valid)):
            center = (edges[index] + edges[index + 1]) / 2
            half = (edges[index + 1] - edges[index]) / 2
            graph.SetPoint(point, float(center), float(values[index]))
            graph.SetPointError(
                point,
                float(half),
                float(half),
                max(0.0, float(values[index] - low[index])),
                max(0.0, float(high[index] - values[index])),
            )
        return self.write(path, graph)

    def correlation(
        self, directory, label, edges, source, *, mask_name=None, xerrors=False
    ):
        values = np.asarray(source["values"], float)
        covariance = np.asarray(source["covariance"], float)
        valid = (
            np.asarray(source["valid"], bool)
            & np.isfinite(values)
            & np.isfinite(np.diag(covariance))
            & (np.diag(covariance) >= 0)
        )
        values = values.copy()
        values[~valid] = np.nan
        covariance = covariance.copy()
        covariance[~valid, :] = np.nan
        covariance[:, ~valid] = np.nan
        edges = np.asarray(edges, float)
        title = ";k* [GeV/c];C(k*)"
        self.hist1(
            f"{directory}/cf_{label}",
            edges,
            values,
            variance=np.diag(covariance),
            title=title,
        )
        self.hist1(
            f"{directory}/valid_{mask_name or label}", edges, valid.astype(float)
        )
        self.hist2(
            f"{directory}/cov_{label}",
            edges,
            edges,
            covariance,
            title=";k*_{i} [GeV/c];k*_{j} [GeV/c];Covariance",
        )
        self.hist2(
            f"{directory}/valid_cov_{label}",
            edges,
            edges,
            np.isfinite(covariance).astype(float),
        )
        return self.graph(
            f"{directory}/g_cf_{label}",
            (edges[1:] + edges[:-1]) / 2,
            values,
            yerror=np.sqrt(np.diag(covariance)),
            valid=valid,
            title=title,
            xerror=np.diff(edges) / 2 if xerrors else None,
        )
