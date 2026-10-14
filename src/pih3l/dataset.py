"""Bounded ROOT sources shared by training, WP and direct pair inference.

Only a consumer's requested columns survive kinematic derivation. No candidate
or pair table is materialized on disk by this source.
"""

from __future__ import annotations

import copy

import numpy as np

from .execution import DERIVED, ITS_DERIVED, preparation_columns
from .io import inventory, read_root_shard, root_shards
from .physics import derive
from .schema import CANDIDATE_COLUMNS, require_columns
from .selection import in_window, in_windows

PROVENANCE = {"_source_file", "_source_tree", "_entry"}
GENERATED = PROVENANCE | {"bdt_score", "bdt_model_id", "bdt_in_scope"}


class RootDataset:
    """Re-iterable, projected ROOT source with explicit input/filter provenance."""

    def __init__(self, config, role, inventory_record=None):
        self.config = config
        self.role = role
        self.pairs = role in {"femto", "mc_femto"}
        self.inventory = (
            inventory(config, roles=[role])[role]
            if inventory_record is None
            else inventory_record
        )

    def fingerprint(self):
        domain = {"finite": ["mass_h3l", "pt_h3l", "ct", "cos_pa"]}
        if self.pairs:
            analysis = self.config["analysis"]
            domain.update(
                centrality_bins=analysis["centrality_bins"],
                mass_range=analysis["mass_range"],
                kstar_range=[analysis["kstar_edges"][0], analysis["kstar_edges"][-1]],
            )
            domain["finite"].append("kstar")
        return {
            "kind": "streamed_root",
            "role": self.role,
            "inventory": self.inventory,
            "physics": self.config["physics"],
            "domain": domain,
            "signal_truth_filter": self.role == "signal",
            "chunk_rows": self.config["io"]["chunk_rows"],
        }

    def shards(self):
        """Enumerate before decoding, allowing verified chunks to be skipped."""
        yield from root_shards(self.config, self.role, with_tree=True)

    def read_shard(self, shard, tree=None, *, columns=None, optional_columns=()):
        required, optional = preparation_columns(self.config, self.role)
        if columns is not None:
            required = set(CANDIDATE_COLUMNS) | {"HypCentFT0C"}
            required |= set(columns) - DERIVED - GENERATED
            optional = set(optional_columns) - DERIVED - GENERATED
            if self.pairs:
                required |= {
                    "IsMixed",
                    "ChargeChannel",
                    "PairPiTrackPt",
                    "PairPiTrackEta",
                    "PairPiTrackPhi",
                }
            if self.role == "signal":
                required |= {"HypIsReco", "HypIsSignal", "HypIsTwoBodyDecay"}
            for name, packed in ITS_DERIVED.items():
                if name in columns:
                    required.add(packed)
                    required.discard(name)
                if name in optional_columns:
                    optional.add(packed)
                    optional.discard(name)
        frame = read_root_shard(
            self.config,
            self.role,
            shard,
            columns=(
                sorted(required)
                if self.config["io"].get("root_projection", True)
                else None
            ),
            optional_columns=sorted(optional),
            tree=tree,
        )
        raw_rows = len(frame)
        if self.role == "signal":
            truth = ["HypIsReco", "HypIsSignal", "HypIsTwoBodyDecay"]
            require_columns(frame, truth, "MC training signal")
            frame = frame.loc[frame[truth].astype(bool).all(axis=1)]
        role_rows = len(frame)
        frame = derive(frame, self.config["physics"], pairs=self.pairs)
        finite_columns = ["mass_h3l", "pt_h3l", "ct", "cos_pa"] + (
            ["kstar"] if self.pairs else []
        )
        finite = np.isfinite(frame[finite_columns]).all(axis=1)
        keep = finite.copy()
        if self.pairs:
            analysis = self.config["analysis"]
            keep &= in_windows(frame.HypCentFT0C, analysis["centrality_bins"])
            keep &= in_window(frame.mass_h3l, analysis["mass_range"])
            keep &= in_window(
                frame.kstar, [analysis["kstar_edges"][0], analysis["kstar_edges"][-1]]
            )
        # Training candidates are not selected/split by analysis centrality.
        frame = frame.loc[keep]
        if columns is not None:
            require_columns(frame, columns, f"{self.role} streamed source")
            retained = sorted(
                set(columns) | (set(optional_columns) & set(frame.columns))
            )
            frame = frame.loc[:, retained]
        frame = frame.copy()
        frame.attrs["source_counts"] = {
            "raw_rows": raw_rows,
            "after_role_filter": role_rows,
            "retained_rows": len(frame),
            "nonfinite_kinematics": int((~finite).sum()),
        }
        return frame

    def iter_batches(self, *, columns=None, optional_columns=(), batch_size=None):
        source = self
        if batch_size is not None and batch_size != self.config["io"]["chunk_rows"]:
            config = copy.deepcopy(self.config)
            config["io"]["chunk_rows"] = batch_size
            source = RootDataset(config, self.role, self.inventory)
        for _, shard, tree in source.shards():
            yield source.read_shard(
                shard, tree, columns=columns, optional_columns=optional_columns
            )
