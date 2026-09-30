"""Train and evaluate CLAR."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import BankConfig, ModelConfig
from .data import DemandDataset, fit_scaler, load_data, split_data
from .mining import mine_pool
from .model import ForecastModel


def parser() -> argparse.ArgumentParser:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description="Train CLAR",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("logs/clar"))
    p.add_argument("--window", type=int, default=6); p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--train-rate", type=float, default=.7); p.add_argument("--val-rate", type=float, default=.1)
    p.add_argument("--embed-dim", type=int, default=64,
                   help="model width")
    p.add_argument("--relation-dim", type=int, default=16,
                   help="width of the graph stream and the forecast read")
    p.add_argument("--fuse-dim", type=int, default=128)
    p.add_argument("--region-dim", type=int, default=8,
                   help="width of the region embedding")
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--temporal-depth", type=int, default=ModelConfig.temporal_depth,
                   help="number of temporal layers")
    p.add_argument("--graph-depth", type=int, default=ModelConfig.graph_depth,
                   help="number of graph layers")
    p.add_argument("--bank-size", type=int, default=12,
                   help="relations kept per target region")
    p.add_argument("--fdr-q", type=float, default=.05,
                   help="false discovery rate of the BH test")
    p.add_argument("--epochs", type=int, default=120); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=.05)
    p.add_argument("--warmup-epochs", type=int, default=5); p.add_argument("--warmup-lr", type=float, default=1e-6)
    p.add_argument("--eta-min", type=float, default=3e-5)
    p.add_argument("--grad-clip", type=float, default=5,
                   help="gradient clipping norm")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--metric-mask", type=float, default=.5)
    p.add_argument("--loss-mask", type=float, default=None,
                   help="value that marks a missing reading (e.g. 0 for traffic data); such "
                        "cells are left out of the loss, model selection and mining")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--cpu", action="store_true"); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--recompute", action="store_true",
                   help="recompute activations in the backward pass to save memory")
    return p


def _loader(dataset, batch: int, shuffle: bool, workers: int, seed: int,
            pin: bool = False) -> DataLoader:
    options = {"persistent_workers": True, "prefetch_factor": 4} if workers else {}
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=pin,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None, **options)


def _batch(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def _finite(value):
    """Replace non-finite floats with None for JSON."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _loss(output, target, observed: Tensor | None = None):
    """L1 loss over the observed cells."""
    if observed is None:
        return F.l1_loss(output, target)
    seen = observed.to(output.dtype)
    return ((output - target).abs() * seen).sum() / seen.sum().clamp_min(1)


@torch.inference_mode()
def evaluate(model, loader, scaler, device, metric_mask, amp=False):
    """Errors on the original scale, overall and per step."""
    model.eval()
    H = model.cfg.horizon
    totals = torch.zeros(7, H, dtype=torch.float64, device=device)
    count = 0
    for batch in loader:
        x, calendar, y, observed, season, level = _batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(x, calendar, season, level)
        prediction = scaler.inverse(output.float())
        target = scaler.inverse(y.float())
        error = prediction - target
        mask = (target > metric_mask) & observed
        absolute, square = error.abs(), error.square()
        axes = (0, 2, 3)
        totals += torch.stack((absolute.sum(axes, dtype=torch.float64),
                               square.sum(axes, dtype=torch.float64),
                               (absolute * mask).sum(axes, dtype=torch.float64),
                               (square * mask).sum(axes, dtype=torch.float64),
                               mask.sum(axes, dtype=torch.float64),
                               (absolute * observed).sum(axes, dtype=torch.float64),
                               observed.sum(axes, dtype=torch.float64)))
        count += error.shape[0] * error.shape[2] * error.shape[3]
    if not count:
        raise RuntimeError("evaluation loader is empty")
    step = totals.cpu()
    (absolute, square, masked_absolute, masked_square, masked_count,
     observed_absolute, observed_count) = step.sum(1).tolist()
    total_cells = count * H
    step_mask_count = step[4].clamp_min(1)
    step_empty = (step[4] == 0).tolist()
    step_masked_mae = (step[2] / step_mask_count).tolist()
    step_masked_rmse = (step[3] / step_mask_count).sqrt().tolist()
    return {"mae": absolute / total_cells, "rmse": math.sqrt(square / total_cells),
            "masked_mae": masked_absolute / masked_count if masked_count else float("nan"),
            "masked_rmse": math.sqrt(masked_square / masked_count) if masked_count else float("nan"),
            "observed_mae": observed_absolute / observed_count if observed_count else float("nan"),
            "per_horizon_mae": (step[0] / count).tolist(),
            "per_horizon_rmse": (step[1] / count).sqrt().tolist(),
            "per_horizon_masked_mae": [float("nan") if e else v for v, e in zip(step_masked_mae, step_empty)],
            "per_horizon_masked_rmse": [float("nan") if e else v for v, e in zip(step_masked_rmse, step_empty)]}


def _lr(epoch: int, args) -> float:
    if epoch < args.warmup_epochs:
        fraction = epoch / max(args.warmup_epochs - 1, 1)
        return args.warmup_lr + fraction * (args.learning_rate - args.warmup_lr)
    progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    return args.eta_min + .5 * (args.learning_rate - args.eta_min) * (1 + math.cos(math.pi * progress))


EMA_DECAY = 0.999


def train_seed(seed, args, cfg, pool, scaler, datasets, device):
    """Train one seed and evaluate the selected weights on the test split."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    scaler = scaler.to(device)
    use_missing = args.loss_mask is not None
    model = ForecastModel(cfg, pool).to(device)
    train, val, test = datasets
    pin = device.type == "cuda"
    loaders = (_loader(train, args.batch_size, True, args.workers, seed, pin),
               _loader(val, args.eval_batch_size, False, args.workers, seed, pin),
               _loader(test, args.eval_batch_size, False, args.workers, seed, pin))
    if not len(loaders[0]):
        raise ValueError("training loader is empty: the training split has no samples")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  fused=device.type == "cuda")
    average = copy.deepcopy(model)
    for parameter in average.parameters():
        parameter.requires_grad_(False)
    model_state, ema_state = model.state_dict(), average.state_dict()
    names = [k for k in ema_state if ema_state[k].dtype.is_floating_point]
    ema_tensors = [ema_state[k] for k in names]
    model_tensors = [model_state[k] for k in names]
    steps = 0
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    amp = False
    if args.amp and device.type == "cuda":
        with torch.cuda.device(device):
            amp = torch.cuda.is_bf16_supported()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group["lr"] = _lr(epoch, args)
        model.train()
        train_sum = torch.zeros((), device=device)
        train_cells = 0
        for batch in loaders[0]:
            cells = int(batch[3].sum()) if use_missing else batch[2].numel()
            if cells == 0:
                continue
            x, calendar, y, observed, season, level = _batch(batch, device)
            obs = observed if use_missing else None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(x, calendar, season, level)
            loss = _loss(scaler.inverse(output.float()), scaler.inverse(y), obs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            steps += 1
            decay = min(EMA_DECAY, (1 + steps) / (10 + steps))
            with torch.no_grad():
                torch._foreach_mul_(ema_tensors, decay)
                torch._foreach_add_(ema_tensors, model_tensors, alpha=1 - decay)
            train_sum = train_sum + loss.detach() * cells
            train_cells += cells
        if train_cells == 0:
            raise RuntimeError("no observed training target in the epoch")
        validation = evaluate(average, loaders[1], scaler, device, args.metric_mask, amp)
        score = validation["observed_mae" if args.loss_mask is not None else "mae"]
        if not math.isfinite(score):
            raise ValueError("validation MAE is non-finite")
        if score < best:
            best_state = {k: v.detach().cpu().clone() for k, v in average.state_dict().items()}
            best, best_epoch, stale = score, epoch + 1, 0
        else:
            stale += 1
        label = "val_observed_mae" if args.loss_mask is not None else "val_mae"
        train_l1 = float(train_sum) / train_cells
        print(f"seed={seed} epoch={epoch+1:03d} lr={_lr(epoch, args):.2e} "
              f"train_l1={train_l1:.5f} {label}={score:.5f} "
              f"val_masked_mae={validation['masked_mae']:.5f}", flush=True)
        if args.patience and stale >= args.patience: break
    model.load_state_dict(best_state)
    metrics = evaluate(model, loaders[2], scaler, device, args.metric_mask, amp)
    checkpoint = args.output_dir / f"seed{seed}.pt"
    torch.save({"model": best_state, "config": cfg.to_dict(),
                "split": {"train_rate": args.train_rate, "val_rate": args.val_rate,
                          "metric_mask": args.metric_mask, "loss_mask": args.loss_mask},
                "pool": asdict(pool),
                "scaler": {"mean": scaler.mean.cpu(), "std": scaler.std.cpu()}, "test": metrics}, checkpoint)
    return {"seed": seed, "best_epoch": best_epoch, "best_val": best,
            "selection": "observed_mae" if args.loss_mask is not None else "mae",
            "test": metrics, "seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint)}


def run(args) -> dict:
    """Mine the relation table and train every seed."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu")
    data = load_data(args.data_dir)
    split = split_data(data.y.shape[1], args.window, args.horizon, args.train_rate, args.val_rate)
    scaler = fit_scaler(data, split, args.window)
    cfg = ModelConfig(
        regions=data.y.shape[0], steps_per_day=data.steps_per_day, window=args.window,
        horizon=args.horizon, embed_dim=args.embed_dim,
        relation_dim=args.relation_dim, fuse_dim=args.fuse_dim, region_dim=args.region_dim,
        **{name: value for name, value in (("temporal_depth", args.temporal_depth),
                                           ("graph_depth", args.graph_depth))
           if value is not None},
        mlp_ratio=args.mlp_ratio,
        bank_size=args.bank_size, recompute=args.recompute)
    cfg.validate()
    bank_cfg = BankConfig(fdr_q=args.fdr_q)
    pool = mine_pool(data.y[:, :split.train_end].T, data.dow[:split.train_end],
                     data.tod[:split.train_end], steps_per_day=data.steps_per_day,
                     window=args.window, bank_size=args.bank_size,
                     config=bank_cfg, missing=args.loss_mask,
                     device=torch.device("cpu"))
    datasets = tuple(DemandDataset(data, target, args.window, args.horizon, scaler,
                                   split.train_end, args.loss_mask)
                     for target in (split.train, split.val, split.test))
    results = [train_seed(seed, args, cfg, pool, scaler, datasets, device)
               for seed in args.seeds]
    aggregate = {metric: {"mean": float(np.mean([x["test"][metric] for x in results])),
                          "std": float(np.std([x["test"][metric] for x in results], ddof=1))
                                 if len(results) > 1 else 0.0}
                 for metric in ("mae", "rmse", "masked_mae", "masked_rmse")}
    report = {"model": "CLAR", "dataset": data.name, "config": cfg.to_dict(),
              "bank": asdict(bank_cfg), "results": results, "aggregate": aggregate}
    (args.output_dir / "results.json").write_text(
        json.dumps(_finite(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    args = parser().parse_args()
    if min(args.window, args.horizon, args.epochs, args.batch_size,
           args.eval_batch_size) < 1 or args.workers < 0 or args.patience < 0:
        parser().error("window, horizon, epochs, batch sizes, workers and patience "
                       "must be sensible")
    schedule = (args.learning_rate, args.warmup_lr, args.eta_min, args.weight_decay,
                args.grad_clip, args.metric_mask)
    if not all(math.isfinite(value) for value in schedule):
        parser().error("training hyperparameters must be finite")
    if (args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0
            or args.warmup_lr <= 0 or args.eta_min < 0 or args.warmup_epochs < 0
            or args.warmup_lr > args.learning_rate or args.eta_min > args.learning_rate
            or not 0 < args.train_rate < 1 or not 0 < args.val_rate < 1
            or args.train_rate + args.val_rate >= 1
            or (args.loss_mask is not None and not math.isfinite(args.loss_mask))):
        parser().error("training hyperparameters are outside their domains")
    print(json.dumps(run(args)["aggregate"], indent=2))


if __name__ == "__main__":
    main()
