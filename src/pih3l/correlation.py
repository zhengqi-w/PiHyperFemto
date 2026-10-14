"""Weighted SE/ME ratios with normalization-induced covariance."""

from __future__ import annotations

import numpy as np

CHANNELS = {0: "H3L + pi+", 1: "H3L + pi-", 2: "anti-H3L + pi-", 3: "anti-H3L + pi+"}


def normalized_correlation(se, me, se2, me2, edges, normalization):
    """Delta-method covariance under independent weighted pair Poisson sampling.

    The normalization counts are the same histogram bins, so their covariance
    with each bin is included through the full Jacobian. Shared tracks/events
    and reused mixed-event particles remain outside this approximation.
    """
    se, me, se2, me2, edges = (
        np.asarray(x, dtype=float) for x in (se, me, se2, me2, edges)
    )
    n = len(se)
    if any(len(x) != n for x in (me, se2, me2)) or len(edges) != n + 1:
        raise ValueError("Histogram and edge dimensions differ")
    if not all(np.isfinite(x).all() and (x >= 0).all() for x in (se, me, se2, me2)):
        raise ValueError("Histogram sums and sumw2 must be finite and nonnegative")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Histogram edges must increase")
    if not all(np.any(np.isclose(edges, x)) for x in normalization):
        raise ValueError("Normalization bounds must align with histogram edges")
    norm = (
        (edges[:-1] >= normalization[0] - 1e-10)
        & (edges[1:] <= normalization[1] + 1e-10)
    ).astype(float)
    sn, mn = float(se @ norm), float(me @ norm)
    output = {
        "values": np.full(n, np.nan),
        "covariance": np.full((n, n), np.nan),
        "valid": np.zeros(n, dtype=bool),
        "alpha": np.nan,
        "se_norm": sn,
        "me_norm": mn,
        "status": "insufficient_normalization",
        "uncertainty": "weighted_pair_poisson_delta_method",
    }
    if sn <= 0 or mn <= 0:
        return output
    valid = (me > 0) & (se > 0)
    alpha = mn / sn
    values = np.divide(alpha * se, me, out=np.zeros(n), where=me > 0)
    js = -np.outer(values / sn, norm)
    jm = np.outer(values / mn, norm)
    js += np.diag(np.divide(alpha, me, out=np.zeros(n), where=me > 0))
    jm -= np.diag(np.divide(values, me, out=np.zeros(n), where=me > 0))
    covariance = (js * se2) @ js.T + (jm * me2) @ jm.T
    covariance = (covariance + covariance.T) / 2
    covariance[~valid, :] = np.nan
    covariance[:, ~valid] = np.nan
    values[~valid] = np.nan
    output.update(
        values=values,
        covariance=covariance,
        valid=valid,
        alpha=alpha,
        status="ok" if valid.all() else "partial_empty_bins",
    )
    return output


def empty_histograms(config):
    a = config["analysis"]
    n = len(a["kstar_edges"]) - 1
    mass_edges = np.linspace(*a["mass_range"], config["purity"]["mass_bins"] + 1)
    result = {}
    for lo, hi in a["centrality_bins"]:
        for channel in CHANNELS:
            name = f"cent_{lo:g}_{hi:g}__ch_{channel}"
            result[name] = {
                "centrality": [lo, hi],
                "channel": channel,
                "channel_label": CHANNELS[channel],
                "regions": {
                    region: {key: np.zeros(n) for key in ("se", "me", "se2", "me2")}
                    for region in ("signal", "left", "right", "sideband")
                },
                "mass_edges": mass_edges.copy(),
                "mass_counts": np.zeros(len(mass_edges) - 1),
                "selected_rows": 0,
                "selected_se": 0,
                "selected_me": 0,
                "mass_vs_kstar": np.zeros((n, len(mass_edges) - 1)),
            }
    return result


def accumulate(histograms, frame, weights, config):
    """Segmented reductions avoid repeatedly slicing each channel's pair rows.

    k* and mass windows remain half-open. The inclusive final mass histogram
    edge follows numpy.histogram, as in the original implementation.
    """
    a = config["analysis"]
    edges = np.asarray(a["kstar_edges"])
    mass_edges = next(iter(histograms.values()))["mass_edges"]
    nk, nm = len(edges) - 1, len(mass_edges) - 1
    cent = frame.HypCentFT0C.to_numpy()
    channel = frame.ChargeChannel.to_numpy()
    ks, mass = frame.kstar.to_numpy(), frame.mass_h3l.to_numpy()
    mixed = frame.IsMixed.to_numpy(dtype=bool)
    kb = np.searchsorted(edges, ks, side="right") - 1
    mb = np.searchsorted(mass_edges, mass, side="right") - 1
    mb[mass == mass_edges[-1]] = nm - 1
    for low, high in a["centrality_bins"]:
        keep = (
            (cent >= low)
            & (cent < high)
            & (channel >= 0)
            & (channel < 4)
            & (channel == np.floor(channel))
            & (ks >= edges[0])
            & (ks < edges[-1])
        )
        ch = channel[keep].astype(int)
        k, m, event = kb[keep], mb[keep], mixed[keep].astype(int)
        masses, w = mass[keep], np.asarray(weights)[keep]
        counts = np.bincount(ch * 2 + event, minlength=8).reshape(4, 2)
        valid_mass = (event == 0) & (m >= 0) & (m < nm) & np.isfinite(masses)
        mass_counts = np.bincount(
            ch[valid_mass] * nm + m[valid_mass], minlength=4 * nm
        ).reshape(4, nm)
        mass2d = np.bincount(
            (ch[valid_mass] * nk + k[valid_mass]) * nm + m[valid_mass],
            minlength=4 * nk * nm,
        ).reshape(4, nk, nm)
        items = [histograms[f"cent_{low:g}_{high:g}__ch_{i}"] for i in range(4)]
        for i, item in enumerate(items):
            item["selected_se"] += int(counts[i, 0])
            item["selected_me"] += int(counts[i, 1])
            item["selected_rows"] += int(counts[i].sum())
            item["mass_counts"] += mass_counts[i]
            item["mass_vs_kstar"] += mass2d[i]
        for region, windows in (
            ("signal", [a["signal_window"]]),
            ("left", [a["sidebands"][0]]),
            ("right", [a["sidebands"][1]]),
            ("sideband", a["sidebands"]),
        ):
            mask = np.zeros(len(ch), dtype=bool)
            for lo, hi in windows:
                mask |= (masses >= lo) & (masses < hi)
            index = (ch[mask] * 2 + event[mask]) * nk + k[mask]
            sums = np.bincount(index, weights=w[mask], minlength=8 * nk).reshape(
                4, 2, nk
            )
            sums2 = np.bincount(index, weights=w[mask] ** 2, minlength=8 * nk).reshape(
                4, 2, nk
            )
            for i, item in enumerate(items):
                for j, label in enumerate(("se", "me")):
                    item["regions"][region][label] += sums[i, j]
                    item["regions"][region][label + "2"] += sums2[i, j]
