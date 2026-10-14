"""Shared native ROOT layout and PDF export primitives."""

from .errors import AnalysisError

CHANNEL_TITLES = [
    "^{3}_{#Lambda}H + #pi^{+}",
    "^{3}_{#Lambda}H + #pi^{-}",
    "^{3}_{#bar{#Lambda}}#bar{H} + #pi^{-}",
    "^{3}_{#bar{#Lambda}}#bar{H} + #pi^{+}",
]


def configure_style(root):
    root.gStyle.SetOptStat(0)
    root.gStyle.SetOptTitle(0)
    root.gStyle.SetTextFont(42)
    root.gStyle.SetCanvasColor(0)
    root.gStyle.SetPadColor(0)
    root.gStyle.SetFrameLineWidth(1)


def text(archive, x, y, label, size=0.042):
    item = archive.keep(archive.ROOT.TLatex())
    item.SetNDC(True)
    item.SetTextFont(42)
    item.SetTextSize(size)
    return archive.keep(item.DrawLatex(x, y, label))


def canvas(
    archive, name, title, columns, rows, *, width=1200, height=900, subtitle=None
):
    item = archive.ROOT.TCanvas(name, title, width, height)
    item.SetCanvasSize(width, height)
    archive.canvases.append(item)
    item.cd()
    text(archive, 0.045, 0.956, title, 0.028 if rows > 1 else 0.043)
    if subtitle:
        text(archive, 0.045, 0.915, subtitle, 0.018 if rows > 1 else 0.027)
    panel = archive.keep(archive.ROOT.TPad(name + "_panels", "", 0, 0, 1, 0.89))
    panel.Draw()
    panel.Divide(columns, rows, 0.004, 0.004)
    return item, panel


def pad(panel, index):
    item = panel.cd(index)
    item.SetLeftMargin(0.14)
    item.SetRightMargin(0.045)
    item.SetBottomMargin(0.15)
    item.SetTopMargin(0.11)
    item.SetTicks(1, 1)
    return item


def frame(archive, target, xmin, ymin, xmax, ymax, xtitle, ytitle):
    item = target.DrawFrame(float(xmin), float(ymin), float(xmax), float(ymax))
    archive.ROOT.SetOwnership(item, False)
    item.SetTitle("")
    for axis, title in ((item.GetXaxis(), xtitle), (item.GetYaxis(), ytitle)):
        axis.SetTitle(title)
        axis.SetTitleSize(0.05)
        axis.SetLabelSize(0.043)
        axis.SetTitleOffset(1.22)
        axis.SetNdivisions(505)
    return item


def line(archive, x1, y1, x2, y2, *, style=2, color=1):
    item = archive.keep(archive.ROOT.TLine(float(x1), float(y1), float(x2), float(y2)))
    item.SetLineStyle(style)
    item.SetLineColor(color)
    item.Draw()
    return item


def box(archive, xmin, xmax, ymin, ymax, color, alpha=0.12):
    item = archive.keep(
        archive.ROOT.TBox(float(xmin), float(ymin), float(xmax), float(ymax))
    )
    item.SetFillColorAlpha(color, alpha)
    item.SetLineWidth(0)
    item.Draw()


def legend(archive, x1=0.55, y1=0.69, x2=0.93, y2=0.86):
    item = archive.keep(archive.ROOT.TLegend(x1, y1, x2, y2))
    item.SetBorderSize(0)
    item.SetFillStyle(0)
    item.SetTextFont(42)
    item.SetTextSize(0.032)
    return item


def save_canvas(archive, item, directory):
    # ROOT's PDF backend uses an A4 MediaBox. A nearly square landscape
    # canvas needs a height constraint too, otherwise the lower pads extend
    # beyond the rotated page even though the canvas renders correctly.
    item.cd()
    item.Modified()
    item.Update()
    archive.write("canvases/" + item.GetName(), item)
    path = directory / f"{item.GetName()}.pdf"
    width, height = item.GetWw(), item.GetWh()
    if width > 0 and height > 0:
        maximum_width, maximum_height = (
            (26.0, 18.0) if width >= height else (18.0, 26.0)
        )
        scale = min(maximum_width / width, maximum_height / height)
        archive.ROOT.gStyle.SetPaperSize(width * scale, height * scale)
    item.SaveAs(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"ROOT did not produce {path}")
    return path
