"""Vectorized relativistic kinematics; input momenta GeV/c, vertices cm."""

from __future__ import annotations

import numpy as np
import pandas as pd


def four_vector(pt, eta, phi, mass):
    pt, eta, phi = (np.asarray(v, dtype=np.float64) for v in (pt, eta, phi))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + np.asarray(mass) ** 2)
    return np.column_stack((energy, px, py, pz))


def invariant_mass(p4):
    m2 = p4[:, 0] ** 2 - np.sum(p4[:, 1:] ** 2, axis=1)
    return np.sqrt(np.maximum(m2, 0))


def kstar(p4_a, p4_b):
    """Källén formula; invalid spacelike or nonfinite pairs become NaN."""
    p4_a, p4_b = np.asarray(p4_a, dtype=float), np.asarray(p4_b, dtype=float)
    if p4_a.ndim != 2 or p4_a.shape[1] != 4 or p4_a.shape != p4_b.shape:
        raise ValueError("Four-vectors must have matching (n, 4) shapes")
    # Permit roundoff at the light cone, but never reinterpret spacelike inputs
    # as massless particles through invariant_mass's square-root clipping.
    invalid = ~np.isfinite(p4_a).all(axis=1) | ~np.isfinite(p4_b).all(axis=1)
    for vector in (p4_a, p4_b):
        momentum2 = np.sum(vector[:, 1:] ** 2, axis=1)
        mass2 = vector[:, 0] ** 2 - momentum2
        tolerance = (
            64 * np.finfo(float).eps * np.maximum(vector[:, 0] ** 2 + momentum2, 1)
        )
        invalid |= (mass2 < -tolerance) | (vector[:, 0] < 0)
    mass_a, mass_b = invariant_mass(p4_a), invariant_mass(p4_b)
    pair = p4_a + p4_b
    s = pair[:, 0] ** 2 - np.sum(pair[:, 1:] ** 2, axis=1)
    lam = (s - (mass_a + mass_b) ** 2) * (s - (mass_a - mass_b) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.sqrt(np.maximum(lam, 0) / (4 * s))
    result[invalid | (s <= 0) | (lam < -1e-8)] = np.nan
    return result


def decode_its_cluster_sizes(packed):
    packed = np.asarray(packed, dtype=np.uint32)
    sizes = np.column_stack([(packed >> (4 * layer)) & 0xF for layer in range(7)])
    count = (sizes > 0).sum(axis=1)
    mean = np.divide(
        sizes.sum(axis=1), count, out=np.zeros(len(count)), where=count > 0
    )
    return mean, count


def derive(frame: pd.DataFrame, physics: dict, *, pairs: bool = False) -> pd.DataFrame:
    frame = frame.copy()
    he = four_vector(
        frame.PtHe3 * physics["he3_momentum_scale"],
        frame.EtaHe3,
        frame.PhiHe3,
        physics["mass_he3"],
    )
    pion = four_vector(frame.PtPi, frame.EtaPi, frame.PhiPi, physics["mass_pi"])
    hyp = he + pion
    momentum = np.linalg.norm(hyp[:, 1:], axis=1)
    pt = np.hypot(hyp[:, 1], hyp[:, 2])
    displacement = frame[["XDecVtx", "YDecVtx", "ZDecVtx"]].to_numpy(dtype=float)
    # The producer already stores SV-PV in X/Y/ZDecVtx. Match spectrum's
    # CorrectAndConvertRDF: subtracting the PV here would do it twice.
    length = np.linalg.norm(displacement, axis=1)
    frame["mass_h3l"] = invariant_mass(hyp)
    frame["pt_h3l"] = pt
    frame["phi_h3l"] = np.arctan2(hyp[:, 2], hyp[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["eta_h3l"] = np.arcsinh(hyp[:, 3] / pt)
        frame["ct"] = length * physics["mass_h3l"] / momentum
        frame["cos_pa"] = np.sum(displacement * hyp[:, 1:], axis=1) / (
            length * momentum
        )
    frame["decay_radius"] = np.hypot(displacement[:, 0], displacement[:, 1])
    for suffix, column in (("he", "ItsClusterSizesHe"), ("pi", "ItsClusterSizesPi")):
        if column in frame:
            frame[f"its_mean_size_{suffix}"], frame[f"its_ncls_{suffix}"] = (
                decode_its_cluster_sizes(frame[column])
            )
    if pairs:
        pair_pi = four_vector(
            frame.PairPiTrackPt,
            frame.PairPiTrackEta,
            frame.PairPiTrackPhi,
            physics["mass_pi"],
        )
        if physics["pair_mass"] == "nominal":
            hyp[:, 0] = np.sqrt(momentum**2 + physics["mass_h3l"] ** 2)
        frame["kstar"] = kstar(hyp, pair_pi)
        if "KstarMC" not in frame and {
            "HypGenPt",
            "HypGenEta",
            "HypGenPhi",
            "PairPiGenPt",
            "PairPiGenEta",
            "PairPiGenPhi",
        } <= set(frame):
            frame["KstarMC"] = kstar(
                four_vector(
                    abs(frame.HypGenPt),
                    frame.HypGenEta,
                    frame.HypGenPhi,
                    physics["mass_h3l"],
                ),
                four_vector(
                    abs(frame.PairPiGenPt),
                    frame.PairPiGenEta,
                    frame.PairPiGenPhi,
                    physics["mass_pi"],
                ),
            )
    return frame
