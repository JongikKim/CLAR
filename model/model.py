"""CLAR model."""
from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CALENDAR_FEATURES, VALUE_FEATURES, ModelConfig
from .mining import CandidatePool


TEMPORAL_DROPOUT = .15
GRAPH_DROP_PATH = .3


def drop_path(x: Tensor, probability: float, training: bool) -> Tensor:
    """Stochastic depth."""
    if not training or probability == 0:
        return x
    keep = 1 - probability
    mask = (keep + torch.rand((len(x),) + (1,) * (x.ndim - 1), device=x.device)).floor()
    return x * mask / keep


GRAPH_GAIN = .1


class GraphNorm(nn.Module):
    """LayerNorm with a learned scale and no shift."""

    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, self.eps)


class SeriesEmbedding(nn.Module):
    """Causal convolutions over the seasonal level and the deviation from it."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kernel = cfg.window
        self.departure = nn.Conv1d(VALUE_FEATURES, cfg.embed_dim, cfg.window)
        self.level = nn.Conv1d(VALUE_FEATURES, cfg.embed_dim, cfg.window)

    def forward(self, x: Tensor, level: Tensor) -> Tensor:
        """x: [b, t, n, features], level: [b, t, n]; returns [b, t, n, embed]."""
        b, t, n, f = x.shape
        level = level.to(x.dtype)
        out = 0
        for part, conv in ((x[..., 0] - level, self.departure), (level, self.level)):
            series = part.permute(0, 2, 1).reshape(b * n, VALUE_FEATURES, t)
            series = F.pad(series, (self.kernel - 1, 0))
            out = out + conv(series).reshape(b, n, -1, t).permute(0, 3, 1, 2)
        return out


class Embedding(nn.Module):
    """Input embedding: series, position, calendar and region."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.value = SeriesEmbedding(cfg)
        self.slot = nn.Embedding(cfg.steps_per_day, cfg.embed_dim)
        self.weekday = nn.Embedding(7, cfg.embed_dim)
        self.steps_per_day = cfg.steps_per_day
        self.region = nn.Linear(cfg.region_dim, cfg.embed_dim)
        position = torch.arange(cfg.window)[:, None]
        frequency = torch.exp(torch.arange(0, cfg.embed_dim, 2)
                              * (-math.log(10000) / cfg.embed_dim))
        pe = torch.zeros(cfg.window, cfg.embed_dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("position", pe[None, :, None], persistent=False)

    def forward(self, x: Tensor, calendar: Tensor, region: Tensor,
                level: Tensor) -> Tensor:
        slot = (calendar[..., 0] * self.steps_per_day).round().long()
        slot = slot.clamp(0, self.steps_per_day - 1)
        weekday = calendar[..., 1:].argmax(-1)
        return (self.value(x, level) + self.position + self.slot(slot)[:, :, None]
                + self.weekday(weekday)[:, :, None] + self.region(region)[None, None])


class Bank(nn.Module):
    """Relation table as sparse propagation and read matrices."""

    def __init__(self, pool: CandidatePool, window: int, horizon: int):
        super().__init__()
        for name in ("source", "lag", "mask"):
            self.register_buffer(name, getattr(pool, name).clone())
        self.window, self.length = window, window + horizon - 1
        self.reach = horizon
        edge = self.mask.nonzero(as_tuple=False)                        # [nnz, 2] = (target, slot)
        targets, sources = edge[:, 0], self.source[edge[:, 0], edge[:, 1]]
        edge_lag = self.lag[edge[:, 0], edge[:, 1]]
        rows, columns = [], []
        for delta in range(1, window):
            take = edge_lag == delta
            if not bool(take.any()):
                continue
            instants = torch.arange(delta, self.length)
            rows.append((targets[take][:, None] * self.length + instants[None]).reshape(-1))
            columns.append((sources[take][:, None] * self.length
                            + (instants - delta)[None]).reshape(-1))
        self.register_buffer("adjacency",
                             torch.stack((torch.cat(rows), torch.cat(columns)))
                             if rows else torch.zeros(2, 0, dtype=torch.long),
                             persistent=False)
        steps = torch.arange(horizon)
        read = (window - 1 + steps + 1)[None] - self.lag[edge[:, 0], edge[:, 1]][:, None]
        self.register_buffer("forecast_adjacency",
                             torch.stack(((targets[:, None] * horizon + steps[None]).reshape(-1),
                                          (sources[:, None] * self.length + read).reshape(-1))),
                             persistent=False)
        self.register_buffer("filled", self.mask.sum(1).clamp_min(1).to(torch.float32),
                             persistent=False)
        nodes = self.source.shape[0] * self.length
        steps_out = self.source.shape[0] * horizon
        for name, index, shape in (("propagation", self.adjacency, (nodes, nodes)),
                                   ("reading", self.forecast_adjacency, (steps_out, nodes))):
            self.register_buffer(name, torch.sparse_coo_tensor(
                index, torch.ones(index.shape[1]), shape).coalesce(), persistent=False)
        axis = torch.arange(self.length)
        degree = ((self.lag[..., None] <= axis) & self.mask[..., None]).sum(1)
        self.register_buffer("degree", degree.clamp_min(1).to(torch.float32), persistent=False)
        self.empty = not bool(self.mask.any())

    def propagate(self, value: Tensor) -> Tensor:
        """Mean over each target's relations of the source values one lag earlier."""
        b, n, length, d = value.shape
        flat = value.permute(1, 2, 0, 3).reshape(n * length, b * d)
        gathered = torch.sparse.mm(self.propagation, flat).reshape(n, length, b, d)
        return gathered.permute(2, 0, 1, 3) / self.degree[None, :, :, None]

    def read(self, features: Tensor) -> Tensor:
        """Mean over each target's relations of the source features at each step's read address."""
        f, b, n, length, d = features.shape
        flat = features.permute(2, 3, 0, 1, 4).reshape(n * length, f * b * d)
        gathered = torch.sparse.mm(self.reading, flat).reshape(n, self.reach, f, b, d)
        return gathered.permute(2, 3, 0, 1, 4) / self.filled[None, None, :, None, None]


class GraphPropagation(nn.Module):
    """One graph layer."""

    def __init__(self, cfg: ModelConfig, path_rate: float, first: bool):
        super().__init__()
        self.cfg, self.path_rate = cfg, path_rate
        self.first = first
        value_in = cfg.embed_dim if first else cfg.relation_dim
        self.value = nn.Linear(value_in, cfg.relation_dim, bias=False)
        if not first:
            self.source_norm = GraphNorm(cfg.relation_dim)
        self.relation_gain = nn.Parameter(torch.full((cfg.relation_dim,), GRAPH_GAIN))

    def forward(self, s: Tensor | None, t_read: Tensor, bank: Bank):
        cfg = self.cfg
        b, n, t, _ = t_read.shape
        if self.first:
            graph_read, s_prev = t_read, 0
        else:
            graph_read = self.source_norm(s).permute(0, 2, 1, 3)
            s_prev = s
        if bank.empty:
            message = t_read.new_zeros(b, n, t, cfg.relation_dim, dtype=torch.float32)
        else:
            message = bank.propagate(self.value(graph_read).float())
        message = F.elu(message)
        return s_prev + drop_path(self.relation_gain * message.permute(0, 2, 1, 3),
                                  self.path_rate, self.training)


class ForecastDecoder(nn.Module):
    """Forecast read for every step."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.forecast_value = nn.Linear(cfg.embed_dim, cfg.relation_dim, bias=False)
        self.source_graph = nn.Linear(cfg.relation_dim, cfg.relation_dim, bias=False)
        self.message = nn.Linear(cfg.relation_dim, cfg.relation_dim, False)

    def _signal(self, message: Tensor) -> Tensor:
        """Future signal: a self-gated projection of the forecast message."""
        g = self.message(message)
        return torch.sigmoid(g) * g

    def forward(self, target: Tensor, bank: Bank, level: Tensor,
                season: Tensor, level_form: Tensor, graph: Tensor) -> Tensor:
        cfg = self.cfg
        b, n, t, _ = target.shape
        if bank.empty:
            return target.new_zeros(b, n, cfg.horizon, cfg.relation_dim, dtype=torch.float32)
        H = cfg.horizon
        value = self.forecast_value(target).float() + self.source_graph(graph).float()
        source_pair = torch.stack((torch.ones_like(level), level))
        aggregated = bank.read(source_pair[..., None] * value[None])         # [2, b, n, H, d]
        target_pair = torch.stack((torch.ones_like(season), season), -1)
        forecast = aggregated[0] + torch.einsum(
            "bnhj,cij,ibnhc->bnhc", target_pair, level_form, aggregated)
        return self._signal(forecast)


class Encoder(nn.Module):
    """One temporal layer (residual MLP)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.mlp_ratio * cfg.embed_dim),
                                 nn.GELU(), nn.Dropout(TEMPORAL_DROPOUT),
                                 nn.Linear(cfg.mlp_ratio * cfg.embed_dim, cfg.embed_dim))
        self.mlp_gain = nn.Parameter(torch.ones(cfg.embed_dim))

    def forward(self, h: Tensor):
        return h + self.mlp_gain * self.mlp(h)


class ForecastModel(nn.Module):
    """CLAR forecasting model."""

    def __init__(self, cfg: ModelConfig, pool: CandidatePool):
        super().__init__()
        cfg.validate()
        shape = (cfg.regions, cfg.bank_size)
        for name, kind in (("source", (torch.int32, torch.int64)),
                           ("lag", (torch.int32, torch.int64)), ("mask", (torch.bool,))):
            field = getattr(pool, name)
            if field.shape != shape:
                raise ValueError(f"bank field {name!r} is not regions x bank_size")
            if field.dtype not in kind:
                raise ValueError(f"bank field {name!r} has dtype {field.dtype}, expected {kind[0]}")
        held = pool.mask
        if bool(((pool.lag < 1) | (pool.lag > cfg.window - 1)).any()):
            raise ValueError("a bank slot has a lag outside 1 .. window-1")
        if bool(((pool.source < 0) | (pool.source >= cfg.regions)).any()):
            raise ValueError("a bank slot names a region outside the grid")
        own = pool.source == torch.arange(cfg.regions)[:, None]
        if bool(own[held].any()):
            raise ValueError("a filled slot makes a region its own source")
        self.cfg = cfg
        self.bank = Bank(pool, cfg.window, cfg.horizon)
        self.region = nn.Parameter(torch.randn(cfg.regions, cfg.region_dim))
        self.embedding = Embedding(cfg)
        self.encoders = nn.ModuleList(Encoder(cfg) for _ in range(cfg.temporal_depth))
        self.state_norm = nn.LayerNorm(cfg.embed_dim)
        graph_rates = torch.linspace(0, GRAPH_DROP_PATH, cfg.graph_depth)
        self.graph = nn.ModuleList(
            GraphPropagation(cfg, float(rate), first=(layer == 0))
            for layer, rate in enumerate(graph_rates))
        self.decoder = ForecastDecoder(cfg)
        self.branches = (cfg.embed_dim, cfg.relation_dim)
        self.branch_weight = nn.Parameter(torch.ones(len(self.branches), cfg.horizon))
        self.head_lift = nn.Linear(sum(self.branches), cfg.fuse_dim)
        self.horizon = nn.Linear(cfg.fuse_dim, cfg.horizon * cfg.horizon_hidden_dim)
        self.output = nn.Linear(cfg.horizon_hidden_dim, 1)
        self.level_form = nn.Parameter(torch.zeros(cfg.relation_dim, 2, 2))

    def _run(self, layer, *inputs):
        """Run a layer, recomputing its activations in the backward pass if enabled."""
        if not (self.cfg.recompute and self.training):
            return layer(*inputs)
        return torch.utils.checkpoint.checkpoint(
            layer, *inputs, use_reentrant=False, preserve_rng_state=True)

    def forward(self, x: Tensor, calendar: Tensor, season: Tensor | None = None,
                level: Tensor | None = None) -> Tensor:
        cfg = self.cfg
        if x.shape[1:] != (cfg.window, cfg.regions, VALUE_FEATURES):
            raise ValueError("input shape does not match ModelConfig")
        if calendar.shape != (x.shape[0], cfg.window, CALENDAR_FEATURES):
            raise ValueError("calendar does not match the input batch and window")
        if season is None or season.shape != (x.shape[0], cfg.regions, cfg.horizon):
            raise ValueError("seasonal reference has the wrong shape")
        if level is None or level.shape != (x.shape[0], self.bank.length, cfg.regions):
            raise ValueError("seasonal level must cover the window and the forecast instants")
        window_level = level[:, :cfg.window]
        hidden = self.embedding(x, calendar, self.region, window_level)
        level = level.permute(0, 2, 1).float()     # [b, regions, positions]
        reference = season.float()
        if (len(self.encoders), len(self.graph)) != (cfg.temporal_depth, cfg.graph_depth):
            raise RuntimeError("each stack must hold the layers its depth names")
        if self.branch_weight.shape != (len(self.branches), cfg.horizon):
            raise RuntimeError("the head needs one weight per branch per step")
        for encoder in self.encoders:
            hidden = self._run(encoder, hidden)
        t_read = self.state_norm(hidden).permute(0, 2, 1, 3).contiguous()

        t_read = F.pad(t_read, (0, 0, 0, cfg.horizon - 1))
        graph = None
        for layer in self.graph:
            graph = self._run(layer, graph, t_read, self.bank)
        signal = self.decoder(t_read, self.bank, level, reference,
                              self.level_form, graph.permute(0, 2, 1, 3))
        own = t_read[:, :, cfg.window - 1]
        wide = [own[:, :, None].expand(-1, -1, cfg.horizon, -1), signal]
        wide = [part * self.branch_weight[i][None, None, :, None] for i, part in enumerate(wide)]
        fused = self.head_lift(torch.cat(wide, -1))
        r = cfg.horizon_hidden_dim
        weight = self.horizon.weight.view(cfg.horizon, r, cfg.fuse_dim)
        bias = self.horizon.bias.view(cfg.horizon, r)
        stream = torch.einsum("bnhs,hrs->bnhr", F.relu(fused), weight) + bias
        return self.output(F.relu(stream)).permute(0, 2, 1, 3)
