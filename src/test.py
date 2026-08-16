from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm

import dataset
from model_variants import VARIANT_SPECS, build_model, count_trainable_parameters
from experiment_utils import seed_everything
from train import (
    collect_paths,
    compute_outputs_and_loss,
    load_raft,
    resolve_scale_roots,
    unpack_batch,
)


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
    parser = argparse.ArgumentParser("SOC-DS test runner")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raft_model", default=None)
    parser.add_argument("--flow_mode", choices=["bidirectional", "shared_forward"], default=None)
    parser.add_argument("--raft_iters", type=int, default=None)
    parser.add_argument("--test_start", type=int, default=36)
    parser.add_argument("--test_end", type=int, default=40)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_star_enhanced", type=str2bool, default=True)
    parser.add_argument("--fallback_to_raw", type=str2bool, default=True)

    parser.add_argument("--show_progress", type=str2bool, default=True)
    parser.add_argument("--progress_mininterval", type=float, default=1.0)
    parser.add_argument("--progress_ascii", type=str2bool, default=True)
    parser.add_argument("--progress_smoothing", type=float, default=0.15)
    parser.add_argument("--progress_status_interval", type=float, default=30.0)

    parser.add_argument("--evaluation_label", default="best")
    parser.add_argument(
        "--evaluation_protocol",
        choices=["official_independent_seed_best", "checkpoint_sensitivity"],
        default="official_independent_seed_best",
    )
    return parser.parse_args()


def save_rows(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def estimate_test_eta(start_time: float, completed: int, total: int) -> float:
    elapsed = max(0.0, time.perf_counter() - start_time)
    if completed <= 0 or elapsed <= 0:
        return float("nan")
    remaining = max(0, total - completed)
    return elapsed / float(completed) * remaining


def running_test_metrics(accum: Dict, stratified: bool) -> Dict[str, float]:
    n = max(1, int(accum["n"]))
    result = {
        "total_mae": accum["abs_total"] / n,
        "relative_error_pct": 100.0 * accum["rel_total"] / n,
        "inference_ms": 1000.0 * accum["inference_time"] / n,
        "throughput_fps": n / max(accum["wall_elapsed"], 1e-8),
    }
    if accum["total_density_rmse_n"] > 0:
        result["total_density_rmse"] = (
            accum["total_density_rmse"] / accum["total_density_rmse_n"]
        )
    if stratified:
        near = accum["abs_near"] / n
        mid = accum["abs_mid"] / n
        far = accum["abs_far"] / n
        result.update(
            near_mae=near,
            mid_mae=mid,
            far_mae=far,
            stratum_mae=(near + mid + far) / 3.0,
        )
        if accum["allocation_diag_n"] > 0:
            result["allocation_diagonal_mean"] = (
                accum["allocation_diag"] / accum["allocation_diag_n"]
            )
    return result


def write_test_progress_status(
    output_dir: Path,
    *,
    variant: str,
    run_label: str,
    seed: int,
    completed: int,
    total: int,
    copy_count: int,
    sample_name: str,
    frame_index: int,
    metrics: Dict[str, float],
    elapsed_sec: float,
    eta_sec: float,
) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": variant,
        "run_label": run_label,
        "seed": seed,
        "phase": "test",
        "completed_frames": completed,
        "total_frames": total,
        "progress": completed / max(1, total),
        "current_copy_count": copy_count,
        "current_sample_name": sample_name,
        "current_frame_index": frame_index,
        "metrics": metrics,
        "elapsed_sec": elapsed_sec,
        "elapsed_text": format_duration(elapsed_sec),
        "eta_sec": eta_sec,
        "eta_text": format_duration(eta_sec),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_progress_status.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def format_scale_summary(scale: int, accum: Dict, stratified: bool) -> str:
    metrics = running_test_metrics(accum, stratified)
    parts = [
        f"scale={scale}",
        f"frames={int(accum['n'])}",
        f"T-MAE={metrics['total_mae']:.3f}",
        f"Rel={metrics['relative_error_pct']:.2f}%",
    ]
    if stratified:
        parts.extend(
            [
                f"S-MAE={metrics['stratum_mae']:.3f}",
                (
                    f"N/M/F={metrics['near_mae']:.2f}/"
                    f"{metrics['mid_mae']:.2f}/"
                    f"{metrics['far_mae']:.2f}"
                ),
            ]
        )
    parts.append(f"Inf={metrics['inference_ms']:.2f}ms")
    return " | ".join(parts)


def strip_module_prefix(state_dict):
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def allocation_components(gt_layers, pred_layers, eps=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_layers = np.clip(np.asarray(gt_layers, dtype=np.float64), 0, None)
    pred_layers = np.clip(np.asarray(pred_layers, dtype=np.float64), 0, None)
    fractions = pred_layers / (pred_layers.sum(axis=0, keepdims=True) + eps)
    numerator = np.zeros((3, 3), dtype=np.float64)
    denominator = gt_layers.sum(axis=(1, 2)).astype(np.float64)
    for i in range(3):
        for j in range(3):
            numerator[i, j] = np.sum(gt_layers[i] * fractions[j])
    matrix = numerator / (denominator[:, None] + eps)
    return matrix, numerator, denominator


def matrix_to_flat(prefix: str, matrix: np.ndarray) -> Dict[str, float]:
    names = ["near", "mid", "far"]
    out = {}
    for i, source in enumerate(names):
        for j, target in enumerate(names):
            out[f"{prefix}_{source}_to_{target}"] = float(matrix[i, j])
    return out


def aggregate_sequence_rows(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["variant", "run_label", "seed", "copy_count", "sample_name"]
    error_cols = [
        c
        for c in frame_df.columns
        if c.startswith("abs_error_")
        or c.startswith("density_rmse_")
        or c == "total_density_rmse"
        or c == "rel_error_total"
    ]
    for keys, group in frame_df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for col in error_cols:
            row[col] = float(group[col].mean())
        if "alloc_den_near" in group.columns:
            num = np.zeros((3, 3), dtype=np.float64)
            den = np.zeros(3, dtype=np.float64)
            for i, source in enumerate(["near", "mid", "far"]):
                den[i] = float(group[f"alloc_den_{source}"].sum())
                for j, target in enumerate(["near", "mid", "far"]):
                    num[i, j] = float(group[f"alloc_num_{source}_to_{target}"].sum())
            micro = num / (den[:, None] + 1e-8)
            frame_macro = np.zeros((3, 3), dtype=np.float64)
            for i, source in enumerate(["near", "mid", "far"]):
                for j, target in enumerate(["near", "mid", "far"]):
                    frame_macro[i, j] = float(group[f"alloc_{source}_to_{target}"].mean())
            row.update(matrix_to_flat("alloc_micro", micro))
            row.update(matrix_to_flat("alloc_frame_macro", frame_macro))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_scale_rows(sequence_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["variant", "run_label", "seed", "copy_count"]
    metric_cols = [c for c in sequence_df.columns if c not in group_cols + ["sample_name"]]
    for keys, group in sequence_df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cli = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    variant = checkpoint.get("variant")
    if variant not in VARIANT_SPECS:
        raise KeyError(f"Checkpoint variant missing or invalid: {variant}")
    spec = VARIANT_SPECS[variant]
    run_label = checkpoint.get("run_label", variant)
    train_args = checkpoint.get("args", {})
    seed = cli.seed if cli.seed is not None else int(train_args.get("seed", 12345))
    seed_everything(seed, deterministic=True)

    args = SimpleNamespace(
        flow_mode=cli.flow_mode or train_args.get("flow_mode", "bidirectional"),
        raft_iters=cli.raft_iters or int(train_args.get("raft_iters", 4)),
        lambda_layer=float(train_args.get("lambda_layer", 1.0)),
        lambda_total=float(train_args.get("lambda_total", 0.2)),
        lambda_consistency=float(train_args.get("lambda_consistency", 1.0)),
        lambda_depth=float(train_args.get("lambda_depth", 0.1)),
        lambda_count=float(train_args.get("lambda_count", 1.0)),
        raft_model=cli.raft_model,
        small=bool(train_args.get("small", False)),
        mixed_precision=bool(train_args.get("mixed_precision", False)),
        alternate_corr=bool(train_args.get("alternate_corr", False)),
    )

    model = build_model(variant, use_pretrained_frontend=False).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]), strict=True)
    model.eval()
    raft = load_raft(args, device, bool(spec.get("use_flow_conditioning", False)))

    scale_roots = resolve_scale_roots(cli.root_dir, None)
    paths, _ = collect_paths(scale_roots, cli.test_start, cli.test_end, cli.frame_start, cli.frame_end)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = dataset.listDataset(
        paths,
        shuffle=False,
        transform=transform,
        train=False,
        use_star_enhanced=cli.use_star_enhanced,
        use_depth_guidance=bool(spec["use_depth_guidance"]),
        fallback_to_raw=cli.fallback_to_raw,
        return_frame_index=True,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cli.workers)
    criterion_sum = nn.MSELoss(reduction="sum").to(device)
    count_criterion = nn.SmoothL1Loss(reduction="mean").to(device)

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stratified = str(spec["family"]) in {"transport", "direct_stratified"}
    total_frames = len(loader)
    frame_rows = []
    total_start = time.perf_counter()
    last_postfix_update = 0.0
    last_status_write = 0.0
    current_scale = None
    current_scale_accum = None

    global_accum = {
        "n": 0,
        "abs_total": 0.0,
        "rel_total": 0.0,
        "abs_near": 0.0,
        "abs_mid": 0.0,
        "abs_far": 0.0,
        "total_density_rmse": 0.0,
        "total_density_rmse_n": 0,
        "allocation_diag": 0.0,
        "allocation_diag_n": 0,
        "inference_time": 0.0,
        "wall_elapsed": 0.0,
    }

    print(
        (
            f"[TEST] variant={variant} | run_label={run_label} | seed={seed} | "
            f"device={device} | frames={len(ds)} | "
            f"flow_mode={args.flow_mode} | requires_raft={bool(spec.get('use_flow_conditioning', False))} | "
            f"parameters={count_trainable_parameters(model):,}"
        ),
        flush=True,
    )

    progress = tqdm(
        loader,
        total=total_frames,
        desc=f"{variant} | Test",
        unit="frame",
        dynamic_ncols=True,
        mininterval=max(0.1, cli.progress_mininterval),
        smoothing=min(max(cli.progress_smoothing, 0.0), 1.0),
        ascii=cli.progress_ascii,
        disable=not cli.show_progress,
        leave=True,
    )

    with torch.no_grad():
        for completed, batch in enumerate(progress, start=1):
            data = unpack_batch(batch, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            _, _, pred = compute_outputs_and_loss(
                model, spec, data, raft, args, criterion_sum, count_criterion, device
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_elapsed = time.perf_counter() - inference_start

            path = Path(paths[completed - 1])
            scale = int(path.parents[1].name)
            sample_name = path.parent.name
            frame_index = int(path.stem.split("_")[-1])

            if current_scale is None:
                current_scale = scale
                current_scale_accum = {
                    "n": 0,
                    "abs_total": 0.0,
                    "rel_total": 0.0,
                    "abs_near": 0.0,
                    "abs_mid": 0.0,
                    "abs_far": 0.0,
                    "total_density_rmse": 0.0,
                    "total_density_rmse_n": 0,
                    "allocation_diag": 0.0,
                    "allocation_diag_n": 0,
                    "inference_time": 0.0,
                    "wall_elapsed": 0.0,
                }
            elif scale != current_scale:
                tqdm.write("[TEST][Scale complete] " + format_scale_summary(
                    current_scale, current_scale_accum, stratified
                ))
                current_scale = scale
                current_scale_accum = {
                    "n": 0,
                    "abs_total": 0.0,
                    "rel_total": 0.0,
                    "abs_near": 0.0,
                    "abs_mid": 0.0,
                    "abs_far": 0.0,
                    "total_density_rmse": 0.0,
                    "total_density_rmse_n": 0,
                    "allocation_diag": 0.0,
                    "allocation_diag_n": 0,
                    "inference_time": 0.0,
                    "wall_elapsed": 0.0,
                }

            gt_total = data["curr_total"][0].detach().cpu().numpy()
            gt_layers = data["curr_layer"][0].detach().cpu().numpy()
            gt_total_count = float(gt_total.sum())
            pred_count = float(pred["count"].detach().cpu())
            abs_total_error = abs(pred_count - gt_total_count)
            relative_total_error = abs_total_error / max(gt_total_count, 1e-8)

            row = {
                "variant": variant,
                "run_label": run_label,
                "seed": seed,
                "evaluation_label": cli.evaluation_label,
                "evaluation_protocol": cli.evaluation_protocol,
                "checkpoint_path": str(Path(cli.checkpoint).resolve()),
                "copy_count": scale,
                "sample_name": sample_name,
                "frame_index": frame_index,
                "gt_total_count": gt_total_count,
                "pred_total_count": pred_count,
                "abs_error_total": abs_total_error,
                "rel_error_total": relative_total_error,
                "inference_time_sec": inference_elapsed,
            }

            for accum in [global_accum, current_scale_accum]:
                accum["n"] += 1
                accum["abs_total"] += abs_total_error
                accum["rel_total"] += relative_total_error
                accum["inference_time"] += inference_elapsed

            if pred["total"] is not None:
                pred_total = pred["total"].detach().cpu().numpy()
                total_density_rmse = float(np.sqrt(np.mean((pred_total - gt_total) ** 2)))
                row["total_density_rmse"] = total_density_rmse
                for accum in [global_accum, current_scale_accum]:
                    accum["total_density_rmse"] += total_density_rmse
                    accum["total_density_rmse_n"] += 1

            if pred["layers"] is not None:
                pred_layers = pred["layers"].detach().cpu().numpy()
                gt_counts = gt_layers.sum(axis=(1, 2))
                pred_counts = pred_layers.sum(axis=(1, 2))
                frame_layer_errors = {}
                for layer_name, layer_idx in [("near", 0), ("mid", 1), ("far", 2)]:
                    layer_abs_error = abs(float(pred_counts[layer_idx] - gt_counts[layer_idx]))
                    frame_layer_errors[layer_name] = layer_abs_error
                    row[f"gt_{layer_name}_count"] = float(gt_counts[layer_idx])
                    row[f"pred_{layer_name}_count"] = float(pred_counts[layer_idx])
                    row[f"abs_error_{layer_name}"] = layer_abs_error
                    row[f"density_rmse_{layer_name}"] = float(
                        np.sqrt(np.mean((pred_layers[layer_idx] - gt_layers[layer_idx]) ** 2))
                    )

                for accum in [global_accum, current_scale_accum]:
                    accum["abs_near"] += frame_layer_errors["near"]
                    accum["abs_mid"] += frame_layer_errors["mid"]
                    accum["abs_far"] += frame_layer_errors["far"]

                matrix, numerator, denominator = allocation_components(gt_layers, pred_layers)
                frame_diag = float(np.trace(matrix) / 3.0)
                for accum in [global_accum, current_scale_accum]:
                    accum["allocation_diag"] += frame_diag
                    accum["allocation_diag_n"] += 1

                row.update(matrix_to_flat("alloc", matrix))
                for i, source in enumerate(["near", "mid", "far"]):
                    row[f"alloc_den_{source}"] = float(denominator[i])
                    for j, target in enumerate(["near", "mid", "far"]):
                        row[f"alloc_num_{source}_to_{target}"] = float(numerator[i, j])

            frame_rows.append(row)

            now = time.perf_counter()
            wall_elapsed = now - total_start
            global_accum["wall_elapsed"] = wall_elapsed
            current_scale_accum["wall_elapsed"] = wall_elapsed
            eta_all = estimate_test_eta(total_start, completed, total_frames)

            should_update = (
                completed == 1
                or completed == total_frames
                or now - last_postfix_update >= max(0.1, cli.progress_mininterval)
            )
            if should_update:
                running = running_test_metrics(global_accum, stratified)
                postfix = {
                    "scale": scale,
                    "sample": sample_name.replace("sample_", ""),
                    "frame": frame_index,
                    "T-MAE": f"{running['total_mae']:.3f}",
                    "Rel": f"{running['relative_error_pct']:.2f}%",
                }
                if stratified:
                    postfix.update(
                        {
                            "S-MAE": f"{running['stratum_mae']:.3f}",
                            "N/M/F": (
                                f"{running['near_mae']:.2f}/"
                                f"{running['mid_mae']:.2f}/"
                                f"{running['far_mae']:.2f}"
                            ),
                            "Alloc": f"{100.0 * running.get('allocation_diagonal_mean', float('nan')):.1f}%",
                        }
                    )
                if "total_density_rmse" in running:
                    postfix["D-RMSE"] = f"{running['total_density_rmse']:.5f}"
                postfix.update(
                    {
                        "Inf": f"{running['inference_ms']:.1f}ms",
                        "FPS": f"{running['throughput_fps']:.2f}",
                        "ETA-all": format_duration(eta_all),
                    }
                )
                progress.set_postfix(postfix, refresh=True)
                last_postfix_update = now

                if (
                    completed == total_frames
                    or now - last_status_write >= max(1.0, cli.progress_status_interval)
                ):
                    write_test_progress_status(
                        output_dir,
                        variant=variant,
                        run_label=run_label,
                        seed=seed,
                        completed=completed,
                        total=total_frames,
                        copy_count=scale,
                        sample_name=sample_name,
                        frame_index=frame_index,
                        metrics=running,
                        elapsed_sec=wall_elapsed,
                        eta_sec=eta_all,
                    )
                    last_status_write = now

    progress.close()
    if current_scale_accum is not None:
        tqdm.write("[TEST][Scale complete] " + format_scale_summary(
            current_scale, current_scale_accum, stratified
        ))

    final_running = running_test_metrics(global_accum, stratified)
    print(
        (
            f"[TEST][Completed] variant={variant} | frames={total_frames} | "
            f"T-MAE={final_running['total_mae']:.4f} | "
            + (
                f"S-MAE={final_running['stratum_mae']:.4f} | "
                f"N/M/F={final_running['near_mae']:.3f}/"
                f"{final_running['mid_mae']:.3f}/"
                f"{final_running['far_mae']:.3f} | "
                if stratified else ""
            )
            + f"Rel={final_running['relative_error_pct']:.3f}% | "
            f"mean inference={final_running['inference_ms']:.2f} ms | "
            f"elapsed={format_duration(time.perf_counter() - total_start)}"
        ),
        flush=True,
    )
    save_rows(output_dir / "frame_results.csv", frame_rows)
    frame_df = pd.DataFrame(frame_rows)
    sequence_df = aggregate_sequence_rows(frame_df)
    scale_df = aggregate_scale_rows(sequence_df)
    sequence_df.to_csv(output_dir / "sequence_results.csv", index=False, encoding="utf-8-sig")
    scale_df.to_csv(output_dir / "copy_count_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "variant": variant,
        "run_label": run_label,
        "code_version": checkpoint.get("code_version", "unknown"),
        "test_runner_version": "stage4_v3_test_progress",
        "seed": seed,
        "evaluation_label": cli.evaluation_label,
        "evaluation_protocol": cli.evaluation_protocol,
        "checkpoint_path": str(Path(cli.checkpoint).resolve()),
        "family": spec["family"],
        "description": spec["description"],
        "num_frames": int(len(frame_df)),
        "num_sequences": int(sequence_df.shape[0]),
        "trainable_parameters": int(count_trainable_parameters(model)),
        "mean_total_count_mae": float(frame_df["abs_error_total"].mean()),
        "mean_total_relative_error": float(frame_df["rel_error_total"].mean()),
        "mean_inference_time_sec": float(frame_df["inference_time_sec"].mean()),
        "total_elapsed_sec": float(time.perf_counter() - total_start),
    }
    if "total_density_rmse" in frame_df.columns:
        summary["mean_total_density_rmse"] = float(frame_df["total_density_rmse"].mean())
    if "abs_error_near" in frame_df.columns:
        summary["mean_near_mae"] = float(frame_df["abs_error_near"].mean())
        summary["mean_mid_mae"] = float(frame_df["abs_error_mid"].mean())
        summary["mean_far_mae"] = float(frame_df["abs_error_far"].mean())
        summary["mean_stratum_mae"] = float(
            np.mean([summary["mean_near_mae"], summary["mean_mid_mae"], summary["mean_far_mae"]])
        )
        summary["mean_near_density_rmse"] = float(frame_df["density_rmse_near"].mean())
        summary["mean_mid_density_rmse"] = float(frame_df["density_rmse_mid"].mean())
        summary["mean_far_density_rmse"] = float(frame_df["density_rmse_far"].mean())

        num = np.zeros((3, 3), dtype=np.float64)
        den = np.zeros(3, dtype=np.float64)
        for i, source in enumerate(["near", "mid", "far"]):
            den[i] = float(frame_df[f"alloc_den_{source}"].sum())
            for j, target in enumerate(["near", "mid", "far"]):
                num[i, j] = float(frame_df[f"alloc_num_{source}_to_{target}"].sum())
        micro = num / (den[:, None] + 1e-8)
        sequence_macro = np.zeros((3, 3), dtype=np.float64)
        frame_macro = np.zeros((3, 3), dtype=np.float64)
        for i, source in enumerate(["near", "mid", "far"]):
            for j, target in enumerate(["near", "mid", "far"]):
                sequence_macro[i, j] = float(sequence_df[f"alloc_micro_{source}_to_{target}"].mean())
                frame_macro[i, j] = float(frame_df[f"alloc_{source}_to_{target}"].mean())
        summary["allocation_micro"] = micro.tolist()
        summary["allocation_sequence_macro"] = sequence_macro.tolist()
        summary["allocation_frame_macro"] = frame_macro.tolist()
        summary["allocation_micro_diagonal_mean"] = float(np.trace(micro) / 3.0)
        summary["allocation_sequence_macro_diagonal_mean"] = float(np.trace(sequence_macro) / 3.0)

        allocation_rows = []
        for name, matrix in [
            ("micro", micro),
            ("sequence_macro", sequence_macro),
            ("frame_macro", frame_macro),
        ]:
            for i, source in enumerate(["near", "mid", "far"]):
                row = {"aggregation": name, "reference_stratum": source}
                for j, target in enumerate(["near", "mid", "far"]):
                    row[f"pred_{target}"] = float(matrix[i, j])
                allocation_rows.append(row)
        save_rows(output_dir / "allocation_matrices.csv", allocation_rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
