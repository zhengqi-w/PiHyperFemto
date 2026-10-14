"""Adapters for current compact O2 tables and the documented extended tables.

Femto rows are pairs, not unique candidates. No hypertriton ID is required,
constructed, persisted, or joined by this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import SchemaError

TREE_NAMES = {
    "femto": {"o2pihypfemto", "pihypfemto", "o2hadhypertable", "hadhypertable"},
    "mc_femto": {"o2pihypfemtomc", "pihypfemtomc", "o2hadhypermc", "hadhypermc"},
    "background": {"o2hypcands", "hypcands"},
    "signal": {"o2mchypcands", "mchypcands"},
}
CANDIDATE_COLUMNS = [
    "PtHe3",
    "EtaHe3",
    "PhiHe3",
    "PtPi",
    "EtaPi",
    "PhiPi",
    "XPrimVtx",
    "YPrimVtx",
    "ZPrimVtx",
    "XDecVtx",
    "YDecVtx",
    "ZDecVtx",
]
HYP_CANDIDATE = set(
    CANDIDATE_COLUMNS
    + [
        "IsMatter",
        "DcaV0Daug",
        "DcaHe",
        "DcaPi",
        "NSigmaHe",
        "NTPCCrossedRowsHe",
        "NTPCCrossedRowsPi",
        "TpcMomHe",
        "TpcMomPi",
        "TpcSignalHe",
        "TpcSignalPi",
        "ItsClusterSizesHe",
        "ItsClusterSizesPi",
        "TofMass",
    ]
)


def canonical_name(name: str) -> str:
    name = (
        name[1:]
        if name.startswith("f") and len(name) > 1 and name[1].isupper()
        else name
    )
    aliases = {
        "CentFT0C": "HypCentFT0C",
        "PosZ": "HypPosZ",
        "TrackOccupancy": "HypTrackOccupancy",
        "Ft0cOccupancy": "HypFt0cOccupancy",
        "CentralityFT0A": "HypCentFT0A",
        "CentralityFT0C": "HypCentFT0C",
        "CentralityFT0M": "HypCentFT0M",
        "RunNumber": "HypRunNumber",
        "TrackOccupancyInTimeRange": "HypTrackOccupancy",
        "Ft0cOccupancyInTimeRange": "HypFt0cOccupancy",
        "IsTruePrimaryHadHyperPair": "IsTruePrimaryPiHyperPair",
        "HadIsPhysicalPrimary": "PairPiIsPhysicalPrimary",
        "HadIsTruePion": "PairPiIsTruePion",
        "HadProcess": "PairPiProcess",
        "HadGenPt": "PairPiGenPt",
        "HadGenEta": "PairPiGenEta",
        "HadGenPhi": "PairPiGenPhi",
        "IsReco": "HypIsReco",
        "IsSignal": "HypIsSignal",
        "IsTwoBodyDecay": "HypIsTwoBodyDecay",
        "IsSurvEvSel": "HypIsSurvEvSel",
        "IsRecoMCCollision": "HypIsRecoMCCollision",
        "GenPt": "HypGenPt",
        "GenEta": "HypGenEta",
        "GenPhi": "HypGenPhi",
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("Hyp") and name[3:] in HYP_CANDIDATE:
        return name[3:]
    if name.startswith("Had") and name[3:] in {
        "Pt",
        "Eta",
        "Phi",
        "Sign",
        "DcaXY",
        "DcaZ",
        "TpcNClsCrossedRows",
        "TpcNClsPID",
        "TpcChi2NCl",
        "ItsClusterSizes",
        "ItsChi2NCl",
        "HasTOF",
        "TpcNSigmaPi",
        "TofNSigmaPi",
    }:
        return "PairPiTrack" + name[3:]
    return name


def require_columns(frame: pd.DataFrame, columns, context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")


def adapt(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    mapped = [canonical_name(c) for c in frame.columns]
    if len(set(mapped)) != len(mapped):
        raise SchemaError(
            "Ambiguous input: multiple branches map to the same canonical column"
        )
    frame = frame.copy()
    frame.columns = mapped
    if "PairPiIsTruePion" not in frame and "PairPiPdgCode" in frame:
        frame["PairPiIsTruePion"] = frame.PairPiPdgCode.abs() == 211
    require_columns(frame, CANDIDATE_COLUMNS + ["HypCentFT0C"], role)
    if role in {"femto", "mc_femto"}:
        require_columns(
            frame,
            ["IsMixed", "PairPiTrackPt", "PairPiTrackEta", "PairPiTrackPhi"],
            role,
        )
        if "ChargeChannel" not in frame:
            require_columns(frame, ["IsMatter", "PairPiTrackSign"], role)
            if not frame["IsMatter"].isin([False, True]).all():
                raise SchemaError("IsMatter must be boolean")
            sign = frame["PairPiTrackSign"].to_numpy()
            if not np.isin(sign, [-1, 1]).all():
                raise SchemaError("Pair pion sign must be exactly -1 or +1")
            matter = frame["IsMatter"].to_numpy(dtype=bool)
            frame["ChargeChannel"] = np.where(
                matter, np.where(sign > 0, 0, 1), np.where(sign < 0, 2, 3)
            ).astype("int8")
        if not frame["ChargeChannel"].isin([0, 1, 2, 3]).all():
            raise SchemaError("ChargeChannel must be in [0,1,2,3]")
        if not frame["IsMixed"].isin([False, True]).all():
            raise SchemaError("IsMixed must be boolean")
    return frame
