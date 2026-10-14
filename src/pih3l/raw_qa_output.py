"""ROOT persistence and PDF views of input-level distributions."""

from pathlib import Path

import numpy as np

from .io import write_json
from .root_io import RootArchive
from .root_plot import (
    CHANNEL_TITLES,
    canvas,
    configure_style,
    frame,
    legend,
    pad,
    save_canvas,
    text,
)


def qa_summary(result):
    """Small report metadata; full distributions remain in results.json/ROOT."""
    return {
        key: value for key, value in result.items() if key not in ("femto", "hyper")
    }


def export_raw_qa(archive, result):
    archive.json("raw_qa/metadata", qa_summary(result))
    for role in ("femto", "hyper"):
        for key, item in result[role].items():
            names = ("kstar_se", "kstar_me") if role == "femto" else ("mass",)
            for name in names:
                distribution = item[name]
                path = f"raw_qa/{role}/{key}/{name}"
                xlabel = (
                    "k* [GeV/c]"
                    if name.startswith("kstar")
                    else "m(^{3}He+#pi) [GeV/c^{2}]"
                )
                hist = archive.hist1(
                    path,
                    distribution["edges"],
                    distribution["counts"],
                    variance=distribution["counts"],
                    title=f";{xlabel};Rows",
                )
                for index, count in (
                    (0, distribution["underflow"]),
                    (hist.GetNbinsX() + 1, distribution["overflow"]),
                ):
                    hist.SetBinContent(index, count)
                    hist.SetBinError(index, np.sqrt(count))
                hist.SetEntries(distribution["rows"] - distribution["nonfinite"])
                archive.write(path, hist)
                archive.json(
                    path + "_counts",
                    {
                        k: v
                        for k, v in distribution.items()
                        if k not in ("edges", "counts")
                    },
                )


def _distribution_panel(archive, target, distributions, title, xlabel, *, log=False):
    """Plot absolute counts without normalizing SE and ME to each other."""
    histograms = [archive.objects[path] for _, path, _ in distributions]
    maximum = max((hist.GetMaximum() for hist in histograms), default=0)
    use_log = log and maximum > 0
    target.SetLogy(use_log)
    first = histograms[0]
    frame(
        archive,
        target,
        first.GetXaxis().GetXmin(),
        0.3 if use_log else 0,
        first.GetXaxis().GetXmax(),
        max(1, maximum) * (5 if use_log else 1.35),
        xlabel,
        "Rows / bin",
    )
    key = legend(archive, 0.58, 0.68, 0.93, 0.84)
    for (label, _, color), hist in zip(distributions, histograms):
        hist.SetLineColor(color)
        hist.SetMarkerColor(color)
        hist.SetLineWidth(2)
        hist.Draw("HIST SAME")
        key.AddEntry(hist, label, "l")
    key.Draw()
    text(archive, 0.16, 0.9, title, 0.042)
    if maximum == 0:
        text(archive, 0.2, 0.48, "No rows in display range", 0.035)


def plot_raw_qa(archive, directory, result, config):
    plots = []
    for index, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
        cent = f"cent_{low:g}_{high:g}"
        name = f"raw_kstar_cent{index}"
        title = f"Pre-BDT Femto k* | {low:g}-{high:g}%"
        item, panels = canvas(
            archive,
            name,
            title,
            2,
            2,
            subtitle="Producer selection + centrality only; absolute pair counts; SE/ME unnormalized",
        )
        for channel in range(4):
            base = f"raw_qa/femto/{cent}__ch_{channel}/kstar"
            _distribution_panel(
                archive,
                pad(panels, channel + 1),
                [
                    ("SE", base + "_se", archive.ROOT.kBlue + 1),
                    ("ME", base + "_me", archive.ROOT.kRed + 1),
                ],
                CHANNEL_TITLES[channel],
                "k* [GeV/c]",
                log=True,
            )
        plots.append((title, save_canvas(archive, item, directory)))
    count = len(config["analysis"]["centrality_bins"])
    item, panels = canvas(
        archive,
        "raw_hyper_mass",
        "Pre-BDT Hyper invariant mass",
        min(2, count),
        (count + 1) // 2,
        subtitle="Independent reconstruction candidates; both matter signs; no offline cuts",
    )
    for index, (low, high) in enumerate(config["analysis"]["centrality_bins"]):
        path = f"raw_qa/hyper/cent_{low:g}_{high:g}/mass"
        _distribution_panel(
            archive,
            pad(panels, index + 1),
            [("Hyper", path, archive.ROOT.kBlue + 1)],
            f"{low:g}-{high:g}% centrality",
            "m(^{3}He+#pi) [GeV/c^{2}]",
        )
    plots.append(
        ("Pre-BDT independent Hyper mass", save_canvas(archive, item, directory))
    )
    return plots


def render_raw_qa(directory, result, config):
    directory = Path(directory)
    with RootArchive(directory / "raw_qa.root") as archive:
        configure_style(archive.ROOT)
        archive.json("metadata/config", config)
        export_raw_qa(archive, result)
        plots = plot_raw_qa(archive, directory, result, config)
    write_json(
        directory / "summary.json",
        {
            **qa_summary(result),
            "plot_backend": "ROOT",
            "plot_format": "PDF",
            "root_file": "raw_qa.root",
            "plots": [path.name for _, path in plots],
        },
    )
