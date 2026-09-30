from __future__ import annotations

import math
from dataclasses import asdict, dataclass


VALUE_FEATURES = 1

CALENDAR_FEATURES = 8


@dataclass
class BankConfig:
    """Relation mining settings."""

    fdr_q: float = 0.05
    eps: float = 1e-8

    def validate(self) -> None:
        if not (math.isfinite(self.eps) and self.eps > 0):
            raise ValueError("invalid epsilon")
        if not 0 < self.fdr_q < 1:
            raise ValueError("fdr_q must lie in (0, 1)")


@dataclass
class ModelConfig:
    """Model settings."""

    regions: int
    steps_per_day: int
    window: int = 6
    horizon: int = 6
    embed_dim: int = 64
    relation_dim: int = 16
    fuse_dim: int = 128
    region_dim: int = 8
    temporal_depth: int = 3
    graph_depth: int = 4
    mlp_ratio: int = 4
    bank_size: int = 12
    eps: float = 1e-8
    recompute: bool = False

    @property
    def horizon_hidden_dim(self) -> int:
        """Hidden width of the readout."""
        return min(self.fuse_dim, self.embed_dim)

    def validate(self) -> None:
        if min(self.temporal_depth, self.graph_depth) < 1:
            raise ValueError("each stack needs at least one layer")
        positive = (self.regions, self.steps_per_day, self.window, self.horizon,
                    self.embed_dim, self.relation_dim, self.fuse_dim, self.region_dim,
                    self.temporal_depth, self.graph_depth, self.mlp_ratio, self.bank_size)
        if min(positive) < 1:
            raise ValueError("model dimensions must be positive")
        if self.embed_dim % 2:
            raise ValueError("embed_dim must be even")
        if self.window < 2:
            raise ValueError("require window >= 2")
        if self.horizon > self.window:
            raise ValueError("require horizon <= window")
        if self.horizon >= self.steps_per_day:
            raise ValueError("require horizon < steps_per_day")
        if not (math.isfinite(self.eps) and self.eps > 0):
            raise ValueError("invalid epsilon")
        if self.fuse_dim < 2:
            raise ValueError("fuse_dim must be >= 2: a one-wide LayerNorm is constant")
        if self.graph_depth > 1 and self.relation_dim < 2:
            raise ValueError("relation_dim must be >= 2 when the graph stream is deeper "
                             "than one layer: it is normalised before it is read")

    def to_dict(self) -> dict:
        return asdict(self) | {"horizon_hidden_dim": self.horizon_hidden_dim}
