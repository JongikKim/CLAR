from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def _tensor(value: Any, dtype: torch.dtype, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"dataset loader did not provide {name}")
    raw = torch.as_tensor(value).detach().cpu()
    if raw.is_complex():
        raise ValueError(f"{name} must be real")
    if raw.is_floating_point() and not torch.isfinite(raw).all():
        raise ValueError(f"{name} contains non-finite values")
    if not dtype.is_floating_point:
        if raw.dtype == torch.bool:
            raise ValueError(f"{name} must be an integer vector")
        if raw.is_floating_point() and not torch.equal(raw, raw.round()):
            raise ValueError(f"{name} must contain integer values")
    return raw.to(dtype).contiguous()


@dataclass
class DemandData:
    """Series and calendar of one dataset."""

    y: Tensor                          # [N,T]
    dow: Tensor                        # [T]
    tod: Tensor                        # [T]
    steps_per_day: int
    name: str = "dataset"

    def validate(self) -> None:
        if self.y.ndim != 2 or self.y.numel() == 0:
            raise ValueError("y must be a nonempty [regions,time] tensor")
        if not torch.isfinite(self.y).all():
            raise ValueError("y contains non-finite values")
        if torch.any(self.y < 0):
            raise ValueError("y must be a nonnegative [regions,time] tensor")
        if self.dow.dtype not in (torch.int32, torch.int64) or \
                self.tod.dtype not in (torch.int32, torch.int64):
            raise ValueError("dow and tod_slot must be integer tensors")
        t = self.y.shape[1]
        if self.dow.shape != (t,) or self.tod.shape != (t,):
            raise ValueError("calendar vectors must align with y")
        if torch.any((self.dow < 0) | (self.dow > 6)):
            raise ValueError("dow must be in [0,6]")
        if self.steps_per_day < 1 or torch.any((self.tod < 0) | (self.tod >= self.steps_per_day)):
            raise ValueError("invalid steps_per_day/tod_slot")


def load_data(data_dir: str | Path) -> DemandData:
    """Load a converted dataset directory (data.npz and meta.json)."""
    directory = Path(data_dir).expanduser().resolve()
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "data.npz") as arrays:
        y = _tensor(arrays["y"], torch.float32, "y")
        dow = _tensor(arrays["dow"], torch.long, "dow")
        tod = _tensor(arrays["tod"], torch.long, "tod")
    data = DemandData(y, dow, tod, int(meta["steps_per_day"]), str(meta["name"]))
    data.validate()
    return data


@dataclass(frozen=True)
class Split:
    train: range
    val: range
    test: range
    train_end: int


def split_data(total: int, window: int, horizon: int, train_rate=.7, val_rate=.1) -> Split:
    """Chronological train/validation/test split of the forecast origins."""
    first, stop = window, total - horizon + 1
    if stop - first < 3 or not 0 < train_rate < 1 or not 0 <= val_rate < 1 - train_rate:
        raise ValueError("invalid data length or split rates")
    val = round(total * train_rate) + 1
    test = val + round(total * val_rate)
    if not first < val < test < stop:
        raise ValueError("each split must contain a sample")
    return Split(range(first, val), range(val, test), range(test, stop), val + horizon - 1)


@dataclass
class StandardScaler:
    """Z-score scaler."""

    mean: Tensor
    std: Tensor

    def to(self, device: torch.device) -> StandardScaler:
        return StandardScaler(self.mean.to(device), self.std.to(device))

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.std.to(x) + self.mean.to(x)


def fit_scaler(data: DemandData, split: Split, window: int) -> StandardScaler:
    """Fit the scaler on the training windows."""
    windows = data.y.unfold(1, window, 1)[:, :len(split.train)]
    return StandardScaler(windows.mean().view(1),
                          windows.std(unbiased=False).clamp_min(1e-6).view(1))


def _context(demand: Tensor, spd: int, pad: Tensor,
             observed: Tensor | None = None) -> Tensor:
    """Exponentially weighted mean of the same time of day on the previous 14 days."""
    total = demand.shape[0]
    weighted = torch.zeros_like(demand)
    weights = torch.zeros_like(demand) if observed is not None else demand.new_zeros(total, 1, 1)
    seen = None if observed is None else observed.to(demand.dtype)
    for k in range(1, 15):
        lag = k * spd
        if lag >= total:
            break
        weight = .85 ** (k - 1)
        if observed is None:
            weighted[lag:] += weight * demand[:-lag]
            weights[lag:] += weight
        else:
            weighted[lag:] += weight * demand[:-lag] * seen[:-lag]
            weights[lag:] += weight * seen[:-lag]
    return torch.where(weights > 0, weighted / weights.clamp_min(1e-12),
                       pad.expand_as(weighted))


class DemandDataset(Dataset):
    """Input windows, targets and seasonal levels of one split."""

    def __init__(self, data: DemandData, targets: range, window: int, horizon: int,
                 scaler: StandardScaler, train_end: int, missing: float | None = None):
        self.targets, self.window, self.horizon = targets, window, horizon
        key = (train_end, float(scaler.mean), float(scaler.std), missing)
        cached = getattr(data, "_clar_features", None)
        if cached is None or cached[0] != key:
            scaled = scaler.transform(data.y.T[..., None])
            observed = None if missing is None else (data.y.T[..., None] != missing)
            if observed is None:
                pad = scaled[:train_end].mean(0, keepdim=True)
            else:
                keep = observed[:train_end].to(scaled.dtype)
                seen = (scaled[:train_end] * keep).sum(0, keepdim=True)
                days = keep.sum(0, keepdim=True)
                pad = torch.where(days > 0, seen / days.clamp_min(1),
                                  seen.sum() / days.sum().clamp_min(1))
            context = _context(scaled, data.steps_per_day, pad, observed)
            level = context.squeeze(-1)                             # [T, N]
            obs = (torch.ones_like(scaled[..., :1], dtype=torch.bool) if observed is None
                   else observed)
            cached = (key, scaled.contiguous(), obs.contiguous(),
                      level.contiguous(),
                      torch.cat((data.tod.float()[:, None] / data.steps_per_day,
                                 F.one_hot(data.dow, 7).float()), -1))
            data._clar_features = cached
        _, self.values, self.observed, self.level, self.calendar = cached
        total = self.values.shape[0]
        if len(targets) and (targets[0] < window or targets[-1] + horizon > total):
            raise ValueError("targets must lie in [window, T - horizon]")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        target = self.targets[index]
        x = self.values[target - self.window:target]
        cal = self.calendar[target - self.window:target]
        y = self.values[target:target + self.horizon, :, :1]
        obs = self.observed[target:target + self.horizon]
        season = self.level[target:target + self.horizon].transpose(0, 1)
        level = self.level[target - self.window:target + self.horizon - 1]
        return x, cal, y, obs, season, level
