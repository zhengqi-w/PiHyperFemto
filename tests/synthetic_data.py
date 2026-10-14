"""Deterministic toy inputs with exact relativistic kinematics and unity SE/ME.

These fixtures validate software closure; they are not generator-level MC physics.
"""

from pathlib import Path

import numpy as np
import uproot
import yaml

MASS_HE3 = 2.80839160743
MASS_PI = 0.13957039
MASS_H3L = 2.99131


def small_xgb_parameters():
    """Fast deterministic classifier settings, independent of production tuning."""
    return {
        "n_estimators": 40,
        "max_depth": 3,
        "learning_rate": 0.12,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 1,
        "colsample_bytree": 1,
        "n_jobs": 1,
    }


def daughters(mass, pt=1.2):
    mass = np.asarray(mass, dtype=float)
    q = np.sqrt(
        (mass**2 - (MASS_HE3 + MASS_PI) ** 2) * (mass**2 - (MASS_HE3 - MASS_PI) ** 2)
    ) / (2 * mass)
    e_he, e_pi = np.sqrt(MASS_HE3**2 + q * q), np.sqrt(MASS_PI**2 + q * q)
    boost = pt / mass
    px_he, px_pi = boost * e_he, boost * e_pi
    return {
        "PtHe3": np.hypot(px_he, q),
        "EtaHe3": np.zeros_like(mass),
        "PhiHe3": np.arctan2(q, px_he),
        "PtPi": np.hypot(px_pi, q),
        "EtaPi": np.zeros_like(mass),
        "PhiPi": np.arctan2(-q, px_pi),
    }


def candidate_columns(masses, signal, rng):
    count = len(masses)
    data = daughters(masses)
    data.update(
        {
            "XPrimVtx": np.arange(count, dtype=float) / 10000,
            "YPrimVtx": np.zeros(count),
            "ZPrimVtx": np.zeros(count),
            "XDecVtx": np.full(count, 3.0),
            "YDecVtx": np.zeros(count),
            "ZDecVtx": np.zeros(count),
            "CentralityFT0C": np.where(np.arange(count) % 2, 5.0, 25.0),
            # Deliberate class overlap provides enough background at moderate
            # signal efficiencies for identifiable signal+background mass fits.
            "DcaHe": rng.normal(1.3 if signal else 0.1, 0.7, count),
            "DcaPi": np.ones(count),
            "DcaV0Daug": np.full(count, 0.1),
            "NSigmaHe": np.zeros(count),
            "NTPCCrossedRowsHe": np.full(count, 100, np.uint8),
            "NTPCCrossedRowsPi": np.full(count, 100, np.uint8),
            "IsMatter": np.arange(count) % 2 == 0,
            "RunNumber": np.full(count, 500001, np.int32),
        }
    )
    if signal:
        data.update(
            {
                "IsReco": np.ones(count, bool),
                "IsSignal": np.ones(count, bool),
                "IsTwoBodyDecay": np.ones(count, bool),
            }
        )
    return {"f" + key: value for key, value in data.items()}


def create_test_inputs(folder):
    folder = Path(folder)
    rng = np.random.default_rng(1234)
    # Enough candidate statistics in both full-range WP sidebands to identify
    # DSCB/shape fits in each pT routing fixture and across source-dependent splits.
    signal = candidate_columns(rng.normal(MASS_H3L, 0.0015, 1920), True, rng)
    background_mass = rng.uniform(2.96, 3.04, 4800)
    background = candidate_columns(background_mass, False, rng)
    # The reconstructed data peak follows the MC signal feature distribution.
    # It remains ordinary Hyper data: truth fields are not copied into this tree.
    data_signal = candidate_columns(rng.normal(MASS_H3L, 0.0015, 1600), True, rng)
    for key in background:
        background[key] = np.concatenate([background[key], data_signal[key]])
    for role, name, data in [
        ("Hyper", "O2hypcands", background),
        ("MCHyper", "O2mchypcands", signal),
    ]:
        (folder / role).mkdir(parents=True, exist_ok=True)
        with uproot.recreate(folder / role / "AO2D.root") as root:
            root["DF_1/" + name] = data

    rows = []
    for cent in [5, 25]:
        for channel in range(4):
            for k in np.arange(0.025, 0.3, 0.05):
                for mass in [2.968, MASS_H3L, 3.016]:
                    for mixed, repeats in [(False, 10), (True, 30)]:
                        rows.extend([(cent, channel, k, mass, mixed)] * repeats)
    rows = np.asarray(rows)
    cent, channel, k, masses, mixed = rows.T
    count = len(rows)
    data = {"fHyp" + key: value for key, value in daughters(masses).items()}
    data.update(
        {"f" + key: np.zeros(count) for key in ["XPrimVtx", "YPrimVtx", "ZPrimVtx"]}
    )
    data.update(
        {
            "fHypXDecVtx": np.full(count, 3.0),
            "fHypYDecVtx": np.zeros(count),
            "fHypZDecVtx": np.zeros(count),
            "fHypDcaHe": np.full(count, 1.3),
            "fHypDcaPi": np.ones(count),
            "fHypDcaV0Daug": np.full(count, 0.1),
            "fHypNSigmaHe": np.zeros(count),
            "fHypNTPCCrossedRowsHe": np.full(count, 100, np.uint8),
            "fHypNTPCCrossedRowsPi": np.full(count, 100, np.uint8),
            "fHypIsMatter": channel < 2,
            "fCentFT0C": cent,
            "fHadSign": np.where(np.isin(channel, [0, 3]), 1, -1).astype(np.int8),
            "fIsMixed": mixed.astype(bool),
            "fIsClosePairRejected": np.zeros(count, np.uint8),
            "fMixingDepth": np.full(count, 3, np.int32),
            "fTrackOccupancy": np.full(count, 1000, np.int32),
        }
    )
    s = (np.sqrt(MASS_H3L**2 + k * k) + np.sqrt(MASS_PI**2 + k * k)) ** 2
    e_rest = (s - MASS_H3L**2 - MASS_PI**2) / (2 * MASS_H3L)
    p_rest = np.sqrt(e_rest**2 - MASS_PI**2)
    beta = 1.2 / np.sqrt(MASS_H3L**2 + 1.2**2)
    px = (-p_rest + beta * e_rest) / np.sqrt(1 - beta * beta)
    data.update(
        {
            "fHadPt": np.abs(px),
            "fHadEta": np.zeros(count),
            "fHadPhi": np.where(px >= 0, 0.0, np.pi),
        }
    )
    (folder / "Femto").mkdir(parents=True, exist_ok=True)
    with uproot.recreate(folder / "Femto/AO2D.root") as root:
        root["DF_1/O2pihypfemto"] = data
    (folder / "MCFemto").mkdir(exist_ok=True)
    return count


def write_test_config(path, data_root, output):
    config = {
        "data_root": str(Path(data_root).resolve()),
        "output": str(Path(output).resolve()),
        "physics": {"pair_mass": "nominal"},
        # Keep tests of legacy preparation/checkpoint contracts explicit. The
        # streaming suite separately exercises the new production defaults.
        "io": {"chunk_rows": 250, "persist_prepared": True, "cache_model_scores": True},
        "working_point": {
            "signal_efficiencies": [0.5, 0.6, 0.7, 0.8, 0.9],
            "yield_eff_range": [0.5, 0.9],
            "storage": {"enabled": False, "reserve_signal_efficiency": 1.0},
        },
        "analysis": {
            "kstar_edges": np.arange(0, 0.31, 0.05).round(10).tolist(),
            "normalization": [0.2, 0.3],
        },
        "training": {
            "features": ["DcaHe"],
            "model_params": small_xgb_parameters(),
            "qa_shap_max_per_class": 40,
            "efficiency_min": 0.5,
            "efficiency_max": 0.9,
            "npoints_for_effi": 9,
        },
        "purity": {"method": "fixed", "fixed": 0.8, "fixed_error": 0.03},
        "systematics": {
            "variations": [
                {
                    "name": "normalization_alt",
                    "group": "normalization",
                    "overrides": {"analysis": {"normalization": [0.15, 0.3]}},
                },
                {
                    "name": "bdt_alt",
                    "group": "bdt",
                    "overrides": {"analysis": {"bdt_threshold": 0.01}},
                },
            ]
        },
    }
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
