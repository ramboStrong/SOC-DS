from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str((_Path(__file__).resolve().parents[1] / "src")))

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm

from experiment_utils import make_generator, seed_everything, seed_worker, write_json
from sensitivity_dataset import PartitionSensitivityDataset
from sensitivity_model import SENSITIVITY_SPEC, build_sensitivity_model, count_trainable_parameters
from sensitivity_partition import get_partition_spec, load_partition_config
from sensitivity_utils import compute_transport_outputs_and_loss
from train import collect_paths, load_raft, resolve_scale_roots, unpack_batch

try:
    from RAFT.core.utils.utils import InputPadder
except Exception:
    InputPadder = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "t"}:
        return True
    if value in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args():
    p = argparse.ArgumentParser("Train SOC-DS for one distance-partition sensitivity design")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--partition_config", required=True)
    p.add_argument("--partition_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--raft_model", required=True)
    p.add_argument("--flow_mode", choices=["bidirectional", "shared_forward"], default="bidirectional")
    p.add_argument("--raft_iters", type=int, default=4)
    p.add_argument("--small", action="store_true")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--alternate_corr", action="store_true")

    p.add_argument("--train_start", type=int, default=1)
    p.add_argument("--train_end", type=int, default=30)
    p.add_argument("--val_start", type=int, default=31)
    p.add_argument("--val_end", type=int, default=35)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=None)
    p.add_argument("--copy_count", default=None)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--deterministic", type=str2bool, default=True)
    p.add_argument("--use_star_enhanced", type=str2bool, default=True)
    p.add_argument("--fallback_to_raw", type=str2bool, default=True)
    p.add_argument("--use_pretrained_frontend", type=str2bool, default=True)
    p.add_argument("--channels_per_stratum", type=int, default=10)

    p.add_argument("--lambda_layer", type=float, default=1.0)
    p.add_argument("--lambda_total", type=float, default=0.2)
    p.add_argument("--lambda_consistency", type=float, default=1.0)
    p.add_argument("--lambda_depth", type=float, default=0.1)
    p.add_argument("--checkpoint_metric", choices=["stratum_mae", "total_mae", "composite"], default="stratum_mae")
    p.add_argument("--checkpoint_total_weight", type=float, default=0.2)

    p.add_argument("--resume_checkpoint", default=None)
    p.add_argument("--auto_resume", type=str2bool, default=True)
    p.add_argument("--show_progress", type=str2bool, default=True)
    p.add_argument("--progress_mininterval", type=float, default=1.0)
    p.add_argument("--progress_ascii", type=str2bool, default=True)
    return p.parse_args()


def save_rows(path: Path, rows: List[Dict]):
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def capture_rng_state(generator: torch.Generator) -> Dict:
    payload = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: Dict, generator: torch.Generator):
    if not payload:
        return
    if payload.get("python_random") is not None:
        random.setstate(payload["python_random"])
    if payload.get("numpy_random") is not None:
        np.random.set_state(payload["numpy_random"])
    if payload.get("torch_cpu") is not None:
        torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    if payload.get("loader_generator") is not None:
        generator.set_state(payload["loader_generator"])


def score_from_metrics(metrics: Dict[str, float], metric: str, total_weight: float) -> float:
    if metric == "stratum_mae":
        return float(metrics["stratum_mae"])
    if metric == "total_mae":
        return float(metrics["abs_total"])
    if metric == "composite":
        return float(metrics["stratum_mae"] + total_weight * metrics["abs_total"])
    raise ValueError(metric)


def running_metrics(accum: Dict, num_strata: int) -> Dict[str, float]:
    n = max(1, int(accum["n"]))
    per = [accum["abs_strata"][k] / n for k in range(num_strata)]
    return {
        "loss": accum["loss"] / n,
        "abs_total": accum["abs_total"] / n,
        "stratum_mae": float(np.mean(per)),
        **{f"abs_s{k}": per[k] for k in range(num_strata)},
    }


def run_epoch(model, loader, raft, args, device, num_strata: int, optimizer=None, epoch: int = 0):
    training = optimizer is not None
    model.train(training)
    raft.eval()
    criterion_sum = nn.MSELoss(reduction="sum").to(device)
    accum = {
        "loss": 0.0,
        "loss_layer": 0.0,
        "loss_total": 0.0,
        "loss_consistency": 0.0,
        "loss_depth": 0.0,
        "abs_total": 0.0,
        "abs_strata": np.zeros(num_strata, dtype=np.float64),
        "n": 0,
    }
    phase = "Train" if training else "Val"
    progress = tqdm(
        loader,
        desc=f"{args.partition_id} | Epoch {epoch + 1:02d}/{args.epochs:02d} | {phase}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=max(0.1, args.progress_mininterval),
        ascii=args.progress_ascii,
        disable=not args.show_progress,
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in progress:
            data = unpack_batch(batch, device)
            loss, losses, pred = compute_transport_outputs_and_loss(
                model,
                data,
                raft,
                args,
                criterion_sum,
                device,
                InputPadder,
                num_strata=num_strata,
                channels_per_stratum=args.channels_per_stratum,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            gt_counts = data["curr_layer"][0].sum(dim=(-2, -1)).detach().cpu().numpy()
            pred_counts = pred["layers"].sum(dim=(-2, -1)).detach().cpu().numpy()
            gt_total = float(data["curr_total"][0].sum().detach().cpu())
            pred_total = float(pred["count"].detach().cpu())
            accum["abs_strata"] += np.abs(pred_counts - gt_counts)
            accum["abs_total"] += abs(pred_total - gt_total)
            accum["loss"] += float(loss.detach().cpu())
            for name in ["layer", "total", "consistency", "depth"]:
                accum[f"loss_{name}"] += float(losses[name].detach().cpu())
            accum["n"] += 1

            metrics = running_metrics(accum, num_strata)
            progress.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "T-MAE": f"{metrics['abs_total']:.3f}",
                    "S-MAE": f"{metrics['stratum_mae']:.3f}",
                },
                refresh=False,
            )
    result = running_metrics(accum, num_strata)
    n = max(1, int(accum["n"]))
    for name in ["layer", "total", "consistency", "depth"]:
        result[f"loss_{name}"] = accum[f"loss_{name}"] / n
    result["n"] = int(accum["n"])
    return result


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Structured transport reconstruction currently requires batch_size=1.")
    if InputPadder is None:
        raise ImportError("RAFT InputPadder is unavailable. Add the RAFT code root to PYTHONPATH.")

    config = load_partition_config(args.partition_config)
    spec = get_partition_spec(config, args.partition_id)
    if args.train_start != int(config["split"]["train"][0]) or args.train_end != int(config["split"]["train"][1]):
        print("[WARN] Training split differs from partition-config metadata.")
    if args.val_start != int(config["split"]["val"][0]) or args.val_end != int(config["split"]["val"][1]):
        print("[WARN] Validation split differs from partition-config metadata.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, args.deterministic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_roots = resolve_scale_roots(args.root_dir, args.copy_count)
    train_paths, train_breakdown = collect_paths(
        scale_roots, args.train_start, args.train_end, args.frame_start, args.frame_end
    )
    val_paths, val_breakdown = collect_paths(
        scale_roots, args.val_start, args.val_end, args.frame_start, args.frame_end
    )
    if not train_paths or not val_paths:
        raise RuntimeError("Training or validation path list is empty.")

    model = build_sensitivity_model(
        spec.num_strata,
        channels_per_stratum=args.channels_per_stratum,
        use_pretrained_frontend=args.use_pretrained_frontend,
    ).to(device)
    raft = load_raft(args, device, required=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = PartitionSensitivityDataset(
        train_paths,
        partition_id=spec.partition_id,
        num_strata=spec.num_strata,
        transform=transform,
        train=True,
        use_star_enhanced=args.use_star_enhanced,
        use_depth_guidance=True,
        fallback_to_raw=args.fallback_to_raw,
    )
    val_ds = PartitionSensitivityDataset(
        val_paths,
        partition_id=spec.partition_id,
        num_strata=spec.num_strata,
        transform=transform,
        train=False,
        use_star_enhanced=args.use_star_enhanced,
        use_depth_guidance=True,
        fallback_to_raw=args.fallback_to_raw,
    )
    generator = make_generator(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 1),
    )

    resume = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume is None and args.auto_resume and (output_dir / "checkpoint_last.pth.tar").exists():
        resume = output_dir / "checkpoint_last.pth.tar"
    start_epoch = 0
    best_score = float("inf")
    epoch_rows = read_rows(output_dir / "epoch_metrics.csv") if resume else []
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        if checkpoint.get("partition_id") != spec.partition_id:
            raise ValueError("Resume checkpoint belongs to a different partition.")
        if int(checkpoint.get("num_strata", -1)) != spec.num_strata:
            raise ValueError("Resume checkpoint has a different number of strata.")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_selection_score", float("inf")))
        restore_rng_state(checkpoint.get("rng_state", {}), generator)

    run_config = {
        **vars(args),
        "experiment_type": "distance_partition_sensitivity",
        "code_version": "partition_sensitivity_v1",
        "variant": "A0_full_socds",
        "partition": {
            "partition_id": spec.partition_id,
            "num_strata": spec.num_strata,
            "thresholds": list(spec.thresholds),
            "threshold_source": spec.threshold_source,
            "description": spec.description,
        },
        "model_spec": SENSITIVITY_SPEC,
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_breakdown": train_breakdown,
        "val_breakdown": val_breakdown,
    }
    write_json(output_dir / "run_config.json", run_config)

    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)
    if start_epoch >= args.epochs:
        print(f"Training already completed through epoch {start_epoch - 1}.")
        return

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model, train_loader, raft, args, device, spec.num_strata,
            optimizer=optimizer, epoch=epoch,
        )
        val_metrics = run_epoch(
            model, val_loader, raft, args, device, spec.num_strata,
            optimizer=None, epoch=epoch,
        )
        score = score_from_metrics(
            val_metrics, args.checkpoint_metric, args.checkpoint_total_weight
        )
        is_best = score < best_score
        if is_best:
            best_score = score

        row = {
            "epoch": epoch,
            "partition_id": spec.partition_id,
            "num_strata": spec.num_strata,
            "selection_score": score,
            "best_selection_score": best_score,
            "epoch_time_sec": time.time() - epoch_start,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        epoch_rows.append(row)
        save_rows(output_dir / "epoch_metrics.csv", epoch_rows)

        checkpoint = {
            "epoch": epoch,
            "experiment_type": "distance_partition_sensitivity",
            "code_version": "partition_sensitivity_v1",
            "variant": "A0_full_socds",
            "run_label": spec.partition_id,
            "partition_id": spec.partition_id,
            "num_strata": spec.num_strata,
            "thresholds": list(spec.thresholds),
            "channels_per_stratum": args.channels_per_stratum,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "selection_score": score,
            "best_selection_score": best_score,
            "val_metrics": val_metrics,
            "rng_state": capture_rng_state(generator),
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pth.tar")
        torch.save(checkpoint, output_dir / "checkpoint_last.pth.tar")
        if is_best:
            torch.save(checkpoint, output_dir / "model_best.pth.tar")
            torch.save(checkpoint, output_dir / "model_best_selection.pth.tar")
        print(
            f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.6f} | "
            f"val_total_mae={val_metrics['abs_total']:.6f} | "
            f"val_stratum_mae={val_metrics['stratum_mae']:.6f} | "
            f"selection={score:.6f} | best={best_score:.6f} | "
            f"time={(time.time() - epoch_start) / 60.0:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
