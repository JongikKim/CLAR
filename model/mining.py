from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import BankConfig


@dataclass
class Evidence:
    z: Tensor
    raw: Tensor


@dataclass
class CandidatePool:
    """Mined relation table: one row of slots per target region."""

    source: Tensor                 # [N, bank_size]
    lag: Tensor
    raw: Tensor
    evidence: Tensor
    mask: Tensor
    anchor: Tensor                 # [N, 7, steps_per_day] calendar mean


def calendar_anchor(y: Tensor, dow: Tensor, tod: Tensor, spd: int,
                    observed: Tensor | None = None) -> Tensor:
    """Mean value per (weekday, time-of-day) cell."""
    n = y.shape[1]
    cell = (dow * spd + tod).long()                       # [T] calendar cell of each instant
    filled = torch.zeros(7 * spd, dtype=torch.long, device=y.device)
    filled.index_add_(0, cell, torch.ones_like(cell))
    if int(filled.min()) == 0:
        empty = int((filled == 0).nonzero()[0])
        raise ValueError(f"training calendar cell is empty: dow={empty // spd}, slot={empty % spd}")
    if observed is None:
        table = y.new_zeros(7 * spd, n).index_add_(0, cell, y) / filled[:, None]
        return table.T.reshape(n, 7, spd)
    held = observed.to(y.dtype)
    total = y.new_zeros(7 * spd, n).index_add_(0, cell, y * held)
    count = y.new_zeros(7 * spd, n).index_add_(0, cell, held)
    seen, days = (y * held).sum(0), held.sum(0)
    level = torch.where(days > 0, seen / days.clamp_min(1),
                        seen.sum() / days.sum().clamp_min(1))
    table = torch.where(count > 0, total / count.clamp_min(1), level[None].expand_as(total))
    return table.T.reshape(n, 7, spd)


def _pearson(a: Tensor, b: Tensor, eps: float) -> Tensor:
    a, b = a - a.mean(0), b - b.mean(0)
    den = (a.square().sum(0)[:, None] * b.square().sum(0)[None]).sqrt()
    return torch.where(den > eps, a.T @ b / den.clamp_min(eps), torch.zeros_like(den)).clamp(-1, 1)


def _rho1(x: Tensor, eps: float) -> Tensor:
    if len(x) < 3:
        return x.new_zeros(x.shape[1])
    a, b = x[1:] - x[1:].mean(0), x[:-1] - x[:-1].mean(0)
    den = (a.square().sum(0) * b.square().sum(0)).sqrt()
    return torch.where(den > eps, (a * b).sum(0) / den.clamp_min(eps), 0).clamp(-1, 1)


def _held_pearson(a: Tensor, b: Tensor, ma: Tensor, mb: Tensor,
                  eps: float) -> tuple[Tensor, Tensor]:
    """Pearson correlation over the steps observed in both series."""
    xa, xb = a * ma, b * mb
    n = ma.T @ mb
    sx, sy = xa.T @ mb, ma.T @ xb
    sxx, syy = (xa * a).T @ mb, ma.T @ (xb * b)
    sxy = xa.T @ xb
    num = n * sxy - sx * sy
    den = ((n * sxx - sx.square()).clamp_min(0) * (n * syy - sy.square()).clamp_min(0)).sqrt()
    r = torch.where(den > eps, num / den.clamp_min(eps), torch.zeros_like(den)).clamp(-1, 1)
    return r, n


def _held_rho1(x: Tensor, m: Tensor, eps: float) -> Tensor:
    if len(x) < 3:
        return x.new_zeros(x.shape[1])
    a, b, ma, mb = x[1:], x[:-1], m[1:], m[:-1]
    xa, xb = a * ma, b * mb
    n = (ma * mb).sum(0)
    sx, sy = (xa * mb).sum(0), (ma * xb).sum(0)
    sxx, syy = (xa * a * mb).sum(0), (ma * xb * b).sum(0)
    sxy = (xa * xb).sum(0)
    num = n * sxy - sx * sy
    den = ((n * sxx - sx.square()).clamp_min(0) * (n * syy - sy.square()).clamp_min(0)).sqrt()
    return torch.where(den > eps, num / den.clamp_min(eps), torch.zeros_like(den)).clamp(-1, 1)


def _effective(count, a: Tensor, b: Tensor) -> Tensor:
    """Bartlett's effective sample size."""
    product = (a[:, None] * b[None]).clamp(-.99, .99)
    scaled = count * (1 - product) / (1 + product)
    if torch.is_tensor(count):
        return torch.minimum(scaled, count).clamp_min(3)
    return scaled.clamp(3, count)


def _z(raw: Tensor, degrees: Tensor) -> Tensor:
    return torch.atanh(raw.clamp(-1 + 1e-6, 1 - 1e-6)) * degrees.clamp_min(0).sqrt()


def _marginal(x: Tensor, window: int, cfg: BankConfig,
              observed: Tensor | None = None) -> Evidence:
    """Lagged correlation and Fisher z of every ordered pair at every lag."""
    shape = x.shape[1], x.shape[1], window - 1
    raw, z = x.new_zeros(shape), x.new_zeros(shape)
    held = None if observed is None else observed.to(x.dtype)
    for i in range(window - 1):
        a, b = x[i + 1:], x[:-(i + 1)]
        if held is None:
            r = _pearson(a, b, cfg.eps)
            ne = _effective(len(a), _rho1(a, cfg.eps), _rho1(b, cfg.eps))
        else:
            ma, mb = held[i + 1:], held[:-(i + 1)]
            r, count = _held_pearson(a, b, ma, mb, cfg.eps)
            ne = _effective(count, _held_rho1(a, ma, cfg.eps), _held_rho1(b, mb, cfg.eps))
        raw[..., i], z[..., i] = r, _z(r, ne - 3)
    return Evidence(z, raw)


def _bh(z: Tensor, tested: Tensor, q: float) -> Tensor:
    """Benjamini-Hochberg test applied to each target row."""
    n = z.shape[0]
    p, valid = torch.erfc(z.abs() / 2 ** .5).reshape(n, -1), tested.reshape(n, -1)
    ordered = torch.sort(torch.where(valid, p, torch.full_like(p, 2)), dim=1).values
    rank = torch.arange(1, p.shape[1] + 1, device=p.device, dtype=p.dtype)[None]
    size = valid.sum(1, keepdim=True)
    passed = (rank <= size) & (ordered <= q * rank / size.clamp_min(1))
    last = (passed * rank.long()).max(1, keepdim=True).values
    cutoff = ordered.gather(1, (last - 1).clamp_min(0))
    return (valid & (p <= cutoff) & (last > 0)).reshape_as(tested)


@torch.no_grad()
def mine_pool(y: Tensor, dow: Tensor, tod: Tensor, *, steps_per_day: int, window: int,
              bank_size: int, config: BankConfig, missing: float | None = None,
              device: torch.device | str = "cpu") -> CandidatePool:
    """Mine the relation table from the training data."""
    config.validate()
    if bank_size < 1:
        raise ValueError("bank_size must be positive")
    y, dow, tod = y.float().cpu(), dow.long().cpu(), tod.long().cpu()
    observed = None if missing is None else (y != missing)
    if window < 2:
        raise ValueError("require window >= 2")
    anchor = calendar_anchor(y, dow, tod, steps_per_day, observed)
    residual = y - anchor[:, dow, tod].T
    device = torch.device(device)
    observed_device = None if observed is None else observed.to(device)
    whole = _marginal(residual.to(device), window, config, observed_device)

    tested = torch.ones_like(whole.z, dtype=torch.bool)
    diagonal = torch.arange(y.shape[1], device=tested.device)
    tested[diagonal, diagonal] = False
    qualified = (tested & _bh(whole.z, tested, config.fdr_q)).cpu()
    z, raw = whole.z.cpu(), whole.raw.cpu()

    n, lags = y.shape[1], window - 1
    survivors = (qualified & (z != 0)).reshape(n, -1)
    strength = raw.abs().reshape(n, -1).masked_fill(~survivors, float("-inf"))
    size = min(bank_size, strength.shape[1])
    best_strength, flat_index = strength.topk(size, dim=1)
    mask = torch.zeros(n, bank_size, dtype=torch.bool)
    source = torch.zeros(n, bank_size, dtype=torch.long)
    lag = torch.ones(n, bank_size, dtype=torch.long)
    pool_raw, evidence = torch.zeros(n, bank_size), torch.zeros(n, bank_size)
    keep = best_strength > float("-inf")
    rows = torch.arange(n)[:, None].expand_as(flat_index)
    source_index, lag_index = flat_index // lags, flat_index % lags
    mask[:, :size] = keep
    source[:, :size] = torch.where(keep, source_index, torch.zeros_like(source_index))
    lag[:, :size] = torch.where(keep, lag_index + 1, torch.ones_like(lag_index))
    pool_raw[:, :size] = torch.where(keep, raw[rows, source_index, lag_index],
                                     torch.zeros_like(best_strength))
    evidence[:, :size] = torch.where(keep, z[rows, source_index, lag_index],
                                     torch.zeros_like(best_strength))
    return CandidatePool(source, lag, pool_raw, evidence, mask, anchor)
