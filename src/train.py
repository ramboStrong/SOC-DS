from __future__ import annotations

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

import dataset
from model_variants import VARIANT_SPECS, build_model, count_trainable_parameters
from experiment_utils import (
    build_boundary_mask,
    count_from_density,
    estimate_raft_pair,
    make_generator,
    masked_mse,
    model_requires_raft,
    propagation_consistency_loss,
    reconstruct_stratified,
    seed_everything,
    seed_worker,
    selection_score,
    write_json,
)

try:
    from RAFT.core.raft import RAFT
    from RAFT.core.utils.utils import InputPadder
except Exception:
    RAFT = None
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
    parser = argparse.ArgumentParser("SOC-DS baseline and loss-ablation trainer")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--variant", choices=list(VARIANT_SPECS), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_label", default=None, help="Optional unique run label, e.g. B1_no_total_loss")
    parser.add_argument("--raft_model", default=None)
    parser.add_argument("--flow_mode", choices=["bidirectional", "shared_forward"], default="bidirectional")
    parser.add_argument("--raft_iters", type=int, default=4)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr", action="store_true")

    parser.add_argument("--train_start", type=int, default=1)
    parser.add_argument("--train_end", type=int, default=30)
    parser.add_argument("--val_start", type=int, default=31)
    parser.add_argument("--val_end", type=int, default=35)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--copy_count", default=None)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--use_star_enhanced", type=str2bool, default=True)
    parser.add_argument("--fallback_to_raw", type=str2bool, default=True)
    parser.add_argument("--use_pretrained_frontend", type=str2bool, default=True)

    parser.add_argument("--lambda_layer", type=float, default=1.0)
    parser.add_argument("--lambda_total", type=float, default=0.2)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--lambda_count", type=float, default=1.0)
    parser.add_argument("--checkpoint_metric", choices=["stratum_mae", "total_mae", "composite"], default="stratum_mae")
    parser.add_argument("--checkpoint_total_weight", type=float, default=0.2)

    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--auto_resume", type=str2bool, default=True)

    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--min_epochs", type=int, default=0)

    parser.add_argument("--show_progress", type=str2bool, default=True)
    parser.add_argument("--progress_mininterval", type=float, default=1.0)
    parser.add_argument("--progress_ascii", type=str2bool, default=True)
    parser.add_argument("--progress_smoothing", type=float, default=0.15)
    parser.add_argument("--progress_status_interval", type=float, default=30.0)
    return parser.parse_args()


def resolve_scale_roots(root_dir: str, copy_count=None) -> List[Path]:
    root = Path(root_dir)
    if copy_count is not None:
        candidate = root / str(copy_count)
        if candidate.exists():
            return [candidate]
        if root.name == str(copy_count):
            return [root]
        raise FileNotFoundError(candidate)
    numeric = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if numeric:
        return numeric
    return [root]


def collect_paths(scale_roots: List[Path], sample_start: int, sample_end: int, frame_start: int, frame_end):
    paths = []
    breakdown = []
    for scale_root in scale_roots:
        count = 0
        for sample_idx in range(sample_start, sample_end + 1):
            sample_dir = scale_root / f"sample_{sample_idx:04d}"
            if not sample_dir.exists():
                continue
            for frame_path in sorted(sample_dir.glob("frame_*.png")):
                try:
                    frame_idx = int(frame_path.stem.split("_")[-1])
                except Exception:
                    continue
                if frame_idx < frame_start:
                    continue
                if frame_end is not None and frame_idx > frame_end:
                    continue
                paths.append(str(frame_path))
                count += 1
        breakdown.append({"scale": scale_root.name, "count": count})
    return paths, breakdown


def save_rows(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log(handle, message: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def load_existing_epoch_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def capture_rng_state(generator: torch.Generator) -> Dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict, generator: torch.Generator) -> None:
    if not state:
        return
    if state.get("python_random") is not None:
        random.setstate(state["python_random"])
    if state.get("numpy_random") is not None:
        np.random.set_state(state["numpy_random"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("loader_generator") is not None:
        generator.set_state(state["loader_generator"])


def format_duration(seconds) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(float(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def numeric_epoch_times(epoch_rows: List[Dict]) -> List[float]:
    values = []
    for row in epoch_rows:
        try:
            value = float(row.get("epoch_time_sec", "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            values.append(value)
    return values


def estimate_overall_eta_seconds(
    *,
    invocation_start: float,
    start_epoch: int,
    current_epoch: int,
    total_epochs: int,
    phase: str,
    phase_completed: int,
    train_steps: int,
    val_steps: int,
    completed_epoch_times: List[float],
) -> float:
    work_per_epoch = max(1, train_steps + val_steps)
    phase_done = phase_completed if phase == "train" else train_steps + phase_completed
    phase_done = min(max(phase_done, 0), work_per_epoch)
    current_fraction = phase_done / float(work_per_epoch)

    recent = [x for x in completed_epoch_times[-5:] if np.isfinite(x) and x > 0]
    future_epochs = max(0, total_epochs - current_epoch - 1)

    if recent:
        mean_epoch_time = float(np.mean(recent))
        current_epoch_remaining = mean_epoch_time * max(0.0, 1.0 - current_fraction)
        return current_epoch_remaining + future_epochs * mean_epoch_time

    elapsed = max(0.0, time.time() - invocation_start)
    completed_work = (
        max(0, current_epoch - start_epoch) * work_per_epoch + phase_done
    )
    total_work = max(1, total_epochs - start_epoch) * work_per_epoch
    remaining_work = max(0, total_work - completed_work)
    if completed_work <= 0 or elapsed <= 0:
        return float("nan")
    seconds_per_work_unit = elapsed / float(completed_work)
    return seconds_per_work_unit * remaining_work


def build_running_metrics(accum: Dict, family: str) -> Dict[str, float]:
    n = max(1, int(accum["n"]))
    metrics = {
        "loss": accum["loss"] / n,
        "total_mae": accum["abs_total"] / n,
    }
    if family in {"transport", "direct_stratified"}:
        near = accum["abs_near"] / n
        mid = accum["abs_mid"] / n
        far = accum["abs_far"] / n
        metrics.update(
            near_mae=near,
            mid_mae=mid,
            far_mae=far,
            stratum_mae=(near + mid + far) / 3.0,
        )
    return metrics


def write_progress_status(
    output_dir: Path,
    *,
    variant: str,
    epoch: int,
    total_epochs: int,
    phase: str,
    completed: int,
    total: int,
    metrics: Dict[str, float],
    phase_elapsed_sec: float,
    overall_elapsed_sec: float,
    overall_eta_sec: float,
) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": variant,
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "phase": phase,
        "phase_completed_batches": completed,
        "phase_total_batches": total,
        "phase_progress": completed / max(1, total),
        "metrics": metrics,
        "phase_elapsed_sec": phase_elapsed_sec,
        "overall_elapsed_sec": overall_elapsed_sec,
        "overall_eta_sec": overall_eta_sec,
        "overall_eta_text": format_duration(overall_eta_sec),
    }
    write_json(output_dir / "progress_status.json", payload)


def unpack_batch(batch, device):
    (
        prev_rgb,
        curr_rgb,
        prev_layer_target,
        curr_layer_target,
        prev_total_target,
        curr_total_target,
        prev_depth_target,
        curr_depth_target,
        prev_depth_mask,
        curr_depth_mask,
        raft_image1,
        raft_image2,
        sample_name,
        frame_name,
        frame_index,
    ) = batch
    return {
        "prev_rgb": prev_rgb.to(device),
        "curr_rgb": curr_rgb.to(device),
        "prev_layer": prev_layer_target.float().to(device),
        "curr_layer": curr_layer_target.float().to(device),
        "prev_total": prev_total_target.float().to(device),
        "curr_total": curr_total_target.float().to(device),
        "prev_depth": prev_depth_target.float().to(device),
        "curr_depth": curr_depth_target.float().to(device),
        "prev_mask": prev_depth_mask.float().to(device),
        "curr_mask": curr_depth_mask.float().to(device),
        "raft_image1": raft_image1,
        "raft_image2": raft_image2,
        "sample_name": sample_name,
        "frame_name": frame_name,
        "frame_index": frame_index,
    }


def zero_flow(batch_size: int, height: int, width: int, device):
    return torch.zeros((batch_size, 2, height, width), device=device)


def compute_outputs_and_loss(model, spec, data, raft, args, criterion_sum, count_criterion, device):
    family = str(spec["family"])
    need_raft = model_requires_raft(spec)
    if need_raft:
        flow_fwd, flow_bwd = estimate_raft_pair(
            raft,
            data["raft_image1"],
            data["raft_image2"],
            InputPadder,
            args.flow_mode,
            args.raft_iters,
            device,
        )
    else:
        flow_fwd = flow_bwd = None

    output_curr = model(data["prev_rgb"], data["curr_rgb"], flow_fwd)
    output_prev = model(data["curr_rgb"], data["prev_rgb"], flow_bwd)

    zero = data["curr_total"].new_tensor(0.0)
    losses = {
        "layer": zero,
        "total": zero,
        "consistency": zero,
        "depth": zero,
        "count": zero,
    }
    pred = {"family": family, "layers": None, "total": None, "count": None}

    if family == "transport":
        propagation_fwd = output_curr.prediction
        propagation_bwd = output_prev.prediction
        boundary = build_boundary_mask(propagation_fwd.shape[-2:], propagation_fwd.device, propagation_fwd.dtype)
        curr_forward = reconstruct_stratified(propagation_fwd, boundary, inverse=False)
        curr_backward = reconstruct_stratified(propagation_bwd, boundary, inverse=True)
        prev_from_forward = reconstruct_stratified(propagation_fwd, boundary, inverse=True)
        pred_layers = (curr_forward + curr_backward) / 2.0
        pred_total = pred_layers.sum(dim=0)

        curr_gt = data["curr_layer"][0]
        prev_gt = data["prev_layer"][0]
        curr_total_gt = data["curr_total"][0]
        prev_total_gt = data["prev_total"][0]

        losses["layer"] = (
            criterion_sum(curr_forward, curr_gt)
            + criterion_sum(curr_backward, curr_gt)
            + criterion_sum(prev_from_forward, prev_gt)
        )
        losses["total"] = (
            criterion_sum(curr_forward.sum(dim=0), curr_total_gt)
            + criterion_sum(curr_backward.sum(dim=0), curr_total_gt)
            + criterion_sum(prev_from_forward.sum(dim=0), prev_total_gt)
        )
        losses["consistency"] = propagation_consistency_loss(propagation_fwd, propagation_bwd, criterion_sum)
        losses["depth"] = masked_mse(output_curr.depth, data["curr_depth"], data["curr_mask"]) + masked_mse(
            output_prev.depth, data["prev_depth"], data["prev_mask"]
        )
        pred.update(layers=pred_layers, total=pred_total, count=pred_total.sum())

    elif family == "direct_stratified":
        curr_pred = output_curr.prediction[0]
        prev_pred = output_prev.prediction[0]
        curr_total_pred = curr_pred.sum(dim=0)
        prev_total_pred = prev_pred.sum(dim=0)
        losses["layer"] = criterion_sum(curr_pred, data["curr_layer"][0]) + criterion_sum(
            prev_pred, data["prev_layer"][0]
        )
        losses["total"] = criterion_sum(curr_total_pred, data["curr_total"][0]) + criterion_sum(
            prev_total_pred, data["prev_total"][0]
        )
        losses["depth"] = masked_mse(output_curr.depth, data["curr_depth"], data["curr_mask"]) + masked_mse(
            output_prev.depth, data["prev_depth"], data["prev_mask"]
        )
        pred.update(layers=curr_pred, total=curr_total_pred, count=curr_total_pred.sum())

    elif family == "direct_total":
        curr_pred = output_curr.prediction[0, 0]
        prev_pred = output_prev.prediction[0, 0]
        losses["total"] = criterion_sum(curr_pred, data["curr_total"][0]) + criterion_sum(
            prev_pred, data["prev_total"][0]
        )
        losses["depth"] = masked_mse(output_curr.depth, data["curr_depth"], data["curr_mask"]) + masked_mse(
            output_prev.depth, data["prev_depth"], data["prev_mask"]
        )
        pred.update(total=curr_pred, count=curr_pred.sum())

    elif family == "count":
        target_curr = data["curr_total"].sum(dim=(-2, -1))
        target_prev = data["prev_total"].sum(dim=(-2, -1))
        count_scale = 500.0
        losses["count"] = count_criterion(output_curr.prediction / count_scale, target_curr / count_scale) + count_criterion(
            output_prev.prediction / count_scale, target_prev / count_scale
        )
        losses["depth"] = masked_mse(output_curr.depth, data["curr_depth"], data["curr_mask"]) + masked_mse(
            output_prev.depth, data["prev_depth"], data["prev_mask"]
        )
        pred.update(count=output_curr.prediction[0])
    else:
        raise RuntimeError(family)

    total_loss = (
        args.lambda_layer * losses["layer"]
        + args.lambda_total * losses["total"]
        + args.lambda_consistency * losses["consistency"]
        + args.lambda_depth * losses["depth"]
        + args.lambda_count * losses["count"]
    )
    return total_loss, losses, pred


def run_epoch(
    model,
    loader,
    spec,
    raft,
    args,
    device,
    optimizer=None,
    *,
    epoch: int,
    invocation_start: float,
    start_epoch: int,
    train_steps: int,
    val_steps: int,
    completed_epoch_times: List[float],
    output_dir: Path,
):
    training = optimizer is not None
    phase_key = "train" if training else "val"
    phase_label = "Train" if training else "Val"
    family = str(spec["family"])

    model.train(training)
    if raft is not None:
        raft.eval()

    criterion_sum = nn.MSELoss(reduction="sum").to(device)
    count_criterion = nn.SmoothL1Loss(reduction="mean").to(device)
    accum = {
        "loss": 0.0,
        "loss_layer": 0.0,
        "loss_total": 0.0,
        "loss_consistency": 0.0,
        "loss_depth": 0.0,
        "loss_count": 0.0,
        "abs_near": 0.0,
        "abs_mid": 0.0,
        "abs_far": 0.0,
        "abs_total": 0.0,
        "n": 0,
    }

    total_batches = len(loader)
    phase_start = time.time()
    last_postfix_update = 0.0
    last_status_write = 0.0

    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"{args.variant} | Epoch {epoch + 1:02d}/{args.epochs:02d} | {phase_label}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=max(0.1, args.progress_mininterval),
        smoothing=min(max(args.progress_smoothing, 0.0), 1.0),
        ascii=args.progress_ascii,
        disable=not args.show_progress,
        leave=True,
    )

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(progress, start=1):
            data = unpack_batch(batch, device)
            total_loss, losses, pred = compute_outputs_and_loss(
                model, spec, data, raft, args, criterion_sum, count_criterion, device
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            gt_total_count = float(data["curr_total"][0].sum().detach().cpu())
            pred_total_count = float(pred["count"].detach().cpu())
            if pred["layers"] is not None:
                gt_layer_counts = data["curr_layer"][0].sum(dim=(-2, -1)).detach().cpu().numpy()
                pred_layer_counts = pred["layers"].sum(dim=(-2, -1)).detach().cpu().numpy()
                accum["abs_near"] += abs(float(pred_layer_counts[0] - gt_layer_counts[0]))
                accum["abs_mid"] += abs(float(pred_layer_counts[1] - gt_layer_counts[1]))
                accum["abs_far"] += abs(float(pred_layer_counts[2] - gt_layer_counts[2]))

            accum["abs_total"] += abs(pred_total_count - gt_total_count)
            accum["loss"] += float(total_loss.detach().cpu())
            for key in ["layer", "total", "consistency", "depth", "count"]:
                accum[f"loss_{key}"] += float(losses[key].detach().cpu())
            accum["n"] += 1

            now = time.time()
            should_update = (
                batch_index == 1
                or batch_index == total_batches
                or now - last_postfix_update >= max(0.1, args.progress_mininterval)
            )
            if should_update:
                running = build_running_metrics(accum, family)
                overall_eta = estimate_overall_eta_seconds(
                    invocation_start=invocation_start,
                    start_epoch=start_epoch,
                    current_epoch=epoch,
                    total_epochs=args.epochs,
                    phase=phase_key,
                    phase_completed=batch_index,
                    train_steps=train_steps,
                    val_steps=val_steps,
                    completed_epoch_times=completed_epoch_times,
                )

                postfix = {
                    "loss": f"{running['loss']:.4f}",
                    "T-MAE": f"{running['total_mae']:.3f}",
                }
                if family in {"transport", "direct_stratified"}:
                    postfix.update(
                        {
                            "S-MAE": f"{running['stratum_mae']:.3f}",
                            "N/M/F": (
                                f"{running['near_mae']:.2f}/"
                                f"{running['mid_mae']:.2f}/"
                                f"{running['far_mae']:.2f}"
                            ),
                        }
                    )
                postfix["ETA-all"] = format_duration(overall_eta)
                progress.set_postfix(postfix, refresh=True)
                last_postfix_update = now

                if (
                    batch_index == total_batches
                    or now - last_status_write >= max(1.0, args.progress_status_interval)
                ):
                    write_progress_status(
                        output_dir,
                        variant=args.variant,
                        epoch=epoch,
                        total_epochs=args.epochs,
                        phase=phase_label,
                        completed=batch_index,
                        total=total_batches,
                        metrics=running,
                        phase_elapsed_sec=now - phase_start,
                        overall_elapsed_sec=now - invocation_start,
                        overall_eta_sec=overall_eta,
                    )
                    last_status_write = now

    progress.close()

    n = max(1, accum["n"])
    result = {key: value / n for key, value in accum.items() if key != "n"}
    result["n"] = accum["n"]
    if family in {"transport", "direct_stratified"}:
        result["stratum_mae"] = (
            result["abs_near"] + result["abs_mid"] + result["abs_far"]
        ) / 3.0
    else:
        result["stratum_mae"] = float("nan")
    result["phase_time_sec"] = time.time() - phase_start
    return result


def load_raft(args, device, required: bool):
    if not required:
        return None
    if RAFT is None or InputPadder is None:
        raise ImportError("RAFT package is unavailable but this variant requires optical flow.")
    if not args.raft_model or not Path(args.raft_model).exists():
        raise FileNotFoundError(f"RAFT model not found: {args.raft_model}")
    raft = torch.nn.DataParallel(RAFT(args))
    state = torch.load(args.raft_model, map_location=device)
    raft.load_state_dict(state)
    raft = raft.module.to(device)
    raft.eval()
    for parameter in raft.parameters():
        parameter.requires_grad_(False)
    return raft


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("The current transport reconstruction protocol requires batch_size=1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, args.deterministic)
    spec = VARIANT_SPECS[args.variant]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
    elif args.auto_resume:
        candidate = output_dir / "checkpoint_last.pth.tar"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    log_mode = "a" if resume_path is not None else "w"
    with (output_dir / "train_log.txt").open(log_mode, encoding="utf-8") as log_file:
        scale_roots = resolve_scale_roots(args.root_dir, args.copy_count)
        train_paths, train_breakdown = collect_paths(
            scale_roots, args.train_start, args.train_end, args.frame_start, args.frame_end
        )
        val_paths, val_breakdown = collect_paths(
            scale_roots, args.val_start, args.val_end, args.frame_start, args.frame_end
        )
        if not train_paths or not val_paths:
            raise RuntimeError("Training or validation path list is empty.")

        model = build_model(args.variant, use_pretrained_frontend=args.use_pretrained_frontend).to(device)
        raft = load_raft(args, device, model_requires_raft(spec))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        use_depth = bool(spec["use_depth_guidance"])
        train_dataset = dataset.listDataset(
            train_paths,
            shuffle=False,
            transform=rgb_transform,
            train=True,
            use_star_enhanced=args.use_star_enhanced,
            use_depth_guidance=use_depth,
            fallback_to_raw=args.fallback_to_raw,
            return_frame_index=True,
        )
        val_dataset = dataset.listDataset(
            val_paths,
            shuffle=False,
            transform=rgb_transform,
            train=False,
            use_star_enhanced=args.use_star_enhanced,
            use_depth_guidance=use_depth,
            fallback_to_raw=args.fallback_to_raw,
            return_frame_index=True,
        )

        generator = make_generator(args.seed)
        start_epoch = 0
        best_score = float("inf")
        epochs_without_improvement = 0
        epoch_rows = load_existing_epoch_rows(output_dir / "epoch_metrics.csv") if resume_path else []

        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=device)
            if checkpoint.get("variant") != args.variant:
                raise ValueError(
                    f"Checkpoint variant {checkpoint.get('variant')} does not match requested {args.variant}"
                )
            model.load_state_dict(checkpoint["state_dict"])
            if checkpoint.get("optimizer") is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint.get("best_selection_score", checkpoint.get("selection_score", float("inf"))))
            epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
            restore_rng_state(checkpoint.get("rng_state", {}), generator)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed + 1),
        )

        config = vars(args).copy()
        config.update(
            code_version="stage4_v2_progress",
            variant_spec=spec,
            device=str(device),
            trainable_parameters=count_trainable_parameters(model),
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            train_breakdown=train_breakdown,
            val_breakdown=val_breakdown,
            resumed_from=str(resume_path) if resume_path is not None else None,
            start_epoch=start_epoch,
        )
        write_json(output_dir / "run_config.json", config)

        log(log_file, f"variant={args.variant} | family={spec['family']} | seed={args.seed}")
        log(log_file, f"run_label={args.run_label or args.variant}")
        log(log_file, f"description={spec['description']}")
        log(log_file, f"device={device} | parameters={config['trainable_parameters']:,}")
        log(log_file, f"flow_mode={args.flow_mode} | requires_raft={model_requires_raft(spec)}")
        log(log_file, f"train_samples={len(train_dataset)} | val_samples={len(val_dataset)}")
        if resume_path is not None:
            log(
                log_file,
                f"RESUME from {resume_path} | start_epoch={start_epoch} | "
                f"best_score={best_score:.6f} | no_improvement={epochs_without_improvement}",
            )

        if start_epoch >= args.epochs:
            log(log_file, f"Run already completed through epoch {start_epoch - 1}; target epochs={args.epochs}.")
            return

        start = time.time()
        invocation_start = start
        completed_epoch_times = numeric_epoch_times(epoch_rows)
        train_steps = len(train_loader)
        val_steps = len(val_loader)

        log(
            log_file,
            f"Progress display enabled={args.show_progress} | "
            f"train_batches/epoch={train_steps} | val_batches/epoch={val_steps}",
        )

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                spec,
                raft,
                args,
                device,
                optimizer=optimizer,
                epoch=epoch,
                invocation_start=invocation_start,
                start_epoch=start_epoch,
                train_steps=train_steps,
                val_steps=val_steps,
                completed_epoch_times=completed_epoch_times,
                output_dir=output_dir,
            )
            val_metrics_raw = run_epoch(
                model,
                val_loader,
                spec,
                raft,
                args,
                device,
                optimizer=None,
                epoch=epoch,
                invocation_start=invocation_start,
                start_epoch=start_epoch,
                train_steps=train_steps,
                val_steps=val_steps,
                completed_epoch_times=completed_epoch_times,
                output_dir=output_dir,
            )
            val_metrics = {
                "val_total_mae": val_metrics_raw["abs_total"],
                "val_near_mae": val_metrics_raw["abs_near"],
                "val_mid_mae": val_metrics_raw["abs_mid"],
                "val_far_mae": val_metrics_raw["abs_far"],
                "val_stratum_mae": val_metrics_raw["stratum_mae"],
            }
            score = selection_score(
                val_metrics,
                str(spec["family"]),
                args.checkpoint_metric,
                args.checkpoint_total_weight,
            )
            is_best = score < best_score
            if is_best:
                best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "selection_score": score,
                "best_selection_score": best_score,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch_time_sec": time.time() - epoch_start,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics_raw.items()},
            }
            epoch_rows.append(row)
            save_rows(output_dir / "epoch_metrics.csv", epoch_rows)

            checkpoint = {
                "epoch": epoch,
                "variant": args.variant,
                "run_label": args.run_label or args.variant,
                "code_version": "stage4_v2_progress",
                "variant_spec": spec,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "selection_score": score,
                "best_selection_score": best_score,
                "epochs_without_improvement": epochs_without_improvement,
                "val_metrics": val_metrics,
                "rng_state": capture_rng_state(generator),
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pth.tar")
            torch.save(checkpoint, output_dir / "checkpoint_last.pth.tar")
            if is_best:
                torch.save(checkpoint, output_dir / "model_best_selection.pth.tar")
                torch.save(checkpoint, output_dir / "model_best.pth.tar")

            completed_epoch_times.append(float(row["epoch_time_sec"]))
            elapsed_all = time.time() - invocation_start
            completed_invocation_epochs = epoch - start_epoch + 1
            mean_epoch_time = elapsed_all / max(1, completed_invocation_epochs)
            remaining_epoch_count = max(0, args.epochs - epoch - 1)
            eta_all = mean_epoch_time * remaining_epoch_count

            write_progress_status(
                output_dir,
                variant=args.variant,
                epoch=epoch,
                total_epochs=args.epochs,
                phase="Epoch complete",
                completed=train_steps + val_steps,
                total=train_steps + val_steps,
                metrics={
                    "train_loss": train_metrics["loss"],
                    "val_total_mae": val_metrics["val_total_mae"],
                    "val_stratum_mae": val_metrics["val_stratum_mae"],
                    "selection_score": score,
                    "best_selection_score": best_score,
                },
                phase_elapsed_sec=row["epoch_time_sec"],
                overall_elapsed_sec=elapsed_all,
                overall_eta_sec=eta_all,
            )

            log(
                log_file,
                (
                    f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.6f} | "
                    f"val_total_mae={val_metrics['val_total_mae']:.6f} | "
                    f"val_stratum_mae={val_metrics['val_stratum_mae']:.6f} | "
                    f"selection={score:.6f} | best={best_score:.6f} | "
                    f"no_improvement={epochs_without_improvement} | "
                    f"epoch_time={format_duration(row['epoch_time_sec'])} | "
                    f"elapsed_all={format_duration(elapsed_all)} | "
                    f"ETA_all={format_duration(eta_all)}"
                ),
            )

            if (
                args.early_stopping_patience > 0
                and (epoch + 1) >= args.min_epochs
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                log(
                    log_file,
                    f"Early stopping at epoch {epoch}: patience={args.early_stopping_patience}, "
                    f"min_epochs={args.min_epochs}.",
                )
                break

        log(log_file, f"Training invocation finished in {(time.time() - start) / 3600.0:.3f} h")


if __name__ == "__main__":
    main()
