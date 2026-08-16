from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str((_Path(__file__).resolve().parents[1] / "src")))

import argparse
import csv
import json
import os
import time
from collections import defaultdict
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

from experiment_utils import seed_everything
from sensitivity_dataset import PartitionSensitivityDataset
from sensitivity_model import build_sensitivity_model, count_trainable_parameters
from sensitivity_partition import get_partition_spec, load_partition_config
from sensitivity_utils import (
    FramePositionCache,
    allocation_components,
    boundary_bin_labels,
    build_boundary_bin_density,
    compute_transport_outputs_and_loss,
    flatten_matrix,
    parse_bin_edges,
)
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
    p = argparse.ArgumentParser("Test one SOC-DS distance-partition sensitivity checkpoint")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--partition_config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--raft_model", required=True)
    p.add_argument("--flow_mode", choices=["bidirectional", "shared_forward"], default=None)
    p.add_argument("--raft_iters", type=int, default=None)
    p.add_argument("--test_start", type=int, default=36)
    p.add_argument("--test_end", type=int, default=40)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_end", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_star_enhanced", type=str2bool, default=True)
    p.add_argument("--fallback_to_raw", type=str2bool, default=True)
    p.add_argument("--show_progress", type=str2bool, default=True)
    p.add_argument("--progress_mininterval", type=float, default=1.0)
    p.add_argument("--progress_ascii", type=str2bool, default=True)
    p.add_argument("--boundary_analysis", type=str2bool, default=False)
    p.add_argument("--boundary_bin_edges", default="0,5,10,20,inf")
    return p.parse_args()


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


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


def aggregate_sequence_rows(frame_df: pd.DataFrame, num_strata: int) -> pd.DataFrame:
    rows = []
    group_cols = ["partition_id", "seed", "copy_count", "sample_name"]
    for keys, group in frame_df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["frame_count"] = int(len(group))
        row["total_count_mae"] = float(group["abs_error_total"].mean())
        row["total_relative_error"] = float(group["rel_error_total"].mean())
        row["total_density_rmse"] = float(group["total_density_rmse"].mean())
        relative_strata = []
        num = np.zeros((num_strata, num_strata), dtype=np.float64)
        den = np.zeros(num_strata, dtype=np.float64)
        for i in range(num_strata):
            row[f"stratum_{i}_count_mae"] = float(group[f"abs_error_s{i}"].mean())
            row[f"stratum_{i}_density_rmse"] = float(group[f"density_rmse_s{i}"].mean())
            gt_mass = float(group[f"gt_s{i}_count"].sum())
            abs_mass = float(group[f"abs_error_s{i}"].sum())
            rel = abs_mass / max(gt_mass, 1e-8)
            row[f"stratum_{i}_relative_count_error"] = rel
            relative_strata.append(rel)
            den[i] = float(group[f"alloc_den_s{i}"].sum())
            for j in range(num_strata):
                num[i, j] = float(group[f"alloc_num_s{i}_to_s{j}"].sum())
        row["macro_stratum_relative_error"] = float(np.mean(relative_strata))
        matrix = num / (den[:, None] + 1e-8)
        row["allocation_diagonal_mean"] = float(np.trace(matrix) / num_strata)
        row.update(flatten_matrix("alloc_micro", matrix))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_scale_rows(sequence_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["partition_id", "seed", "copy_count"]
    metric_cols = [c for c in sequence_df.columns if c not in group_cols + ["sample_name"]]
    for keys, group in sequence_df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_rows_from_store(store, spec, bin_labels):
    rows = []
    for (copy_count, sample_name, bin_index), payload in sorted(store.items()):
        num = payload["num"]
        den = payload["den"]
        matrix = num / (den[:, None] + 1e-8)
        valid = den > 1e-8
        diagonal = np.diag(matrix)
        row = {
            "partition_id": spec.partition_id,
            "seed": payload["seed"],
            "copy_count": copy_count,
            "sample_name": sample_name,
            "bin_index": bin_index,
            "boundary_distance_bin": bin_labels[bin_index],
            "target_mass": float(den.sum()),
            "valid_strata": int(valid.sum()),
            "correct_stratum_mass_fraction": float(diagonal[valid].mean()) if valid.any() else np.nan,
        }
        for i in range(spec.num_strata):
            row[f"den_s{i}"] = float(den[i])
            row[f"diag_num_s{i}"] = float(num[i, i])
        row.update(flatten_matrix("alloc", matrix))
        rows.append(row)
    return rows


def main():
    cli = parse_args()
    if InputPadder is None:
        raise ImportError("RAFT InputPadder is unavailable. Add the RAFT code root to PYTHONPATH.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    if checkpoint.get("experiment_type") != "distance_partition_sensitivity":
        raise ValueError("Checkpoint is not a partition-sensitivity checkpoint.")

    config = load_partition_config(cli.partition_config)
    spec = get_partition_spec(config, checkpoint["partition_id"])
    if int(checkpoint["num_strata"]) != spec.num_strata:
        raise ValueError("Checkpoint K does not match partition configuration.")
    if [float(x) for x in checkpoint["thresholds"]] != [float(x) for x in spec.thresholds]:
        raise ValueError("Checkpoint thresholds do not match partition configuration.")

    train_args = checkpoint.get("args", {})
    seed = int(cli.seed if cli.seed is not None else train_args.get("seed", 12345))
    seed_everything(seed, deterministic=True)
    channels_per_stratum = int(checkpoint.get("channels_per_stratum", 10))
    args = SimpleNamespace(
        flow_mode=cli.flow_mode or train_args.get("flow_mode", "bidirectional"),
        raft_iters=cli.raft_iters or int(train_args.get("raft_iters", 4)),
        lambda_layer=float(train_args.get("lambda_layer", 1.0)),
        lambda_total=float(train_args.get("lambda_total", 0.2)),
        lambda_consistency=float(train_args.get("lambda_consistency", 1.0)),
        lambda_depth=float(train_args.get("lambda_depth", 0.1)),
        raft_model=cli.raft_model,
        small=bool(train_args.get("small", False)),
        mixed_precision=bool(train_args.get("mixed_precision", False)),
        alternate_corr=bool(train_args.get("alternate_corr", False)),
    )

    model = build_sensitivity_model(
        spec.num_strata,
        channels_per_stratum=channels_per_stratum,
        use_pretrained_frontend=False,
    ).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]), strict=True)
    model.eval()
    raft = load_raft(args, device, required=True)

    roots = resolve_scale_roots(cli.root_dir, None)
    paths, _ = collect_paths(roots, cli.test_start, cli.test_end, cli.frame_start, cli.frame_end)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = PartitionSensitivityDataset(
        paths,
        partition_id=spec.partition_id,
        num_strata=spec.num_strata,
        transform=transform,
        train=False,
        use_star_enhanced=cli.use_star_enhanced,
        use_depth_guidance=True,
        fallback_to_raw=cli.fallback_to_raw,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cli.workers)
    criterion_sum = nn.MSELoss(reduction="sum").to(device)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = parse_bin_edges(cli.boundary_bin_edges)
    bin_labels = boundary_bin_labels(bin_edges)
    boundary_enabled = bool(cli.boundary_analysis)
    if boundary_enabled and not spec.thresholds:
        raise ValueError("Boundary analysis requires at least one threshold.")
    position_cache = FramePositionCache()
    boundary_store = defaultdict(
        lambda: {
            "num": np.zeros((spec.num_strata, spec.num_strata), dtype=np.float64),
            "den": np.zeros(spec.num_strata, dtype=np.float64),
            "seed": seed,
        }
    )

    frame_rows: List[Dict] = []
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"{spec.partition_id} | Test | seed={seed}",
        unit="frame",
        dynamic_ncols=True,
        mininterval=max(0.1, cli.progress_mininterval),
        ascii=cli.progress_ascii,
        disable=not cli.show_progress,
    )
    total_start = time.perf_counter()
    with torch.no_grad():
        for completed, batch in enumerate(progress, start=1):
            data = unpack_batch(batch, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _, _, pred = compute_transport_outputs_and_loss(
                model,
                data,
                raft,
                args,
                criterion_sum,
                device,
                InputPadder,
                num_strata=spec.num_strata,
                channels_per_stratum=channels_per_stratum,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            path = Path(paths[completed - 1])
            copy_count = int(path.parents[1].name)
            sample_name = path.parent.name
            frame_index = int(path.stem.split("_")[-1])
            gt_layers = data["curr_layer"][0].detach().cpu().numpy()
            gt_total = data["curr_total"][0].detach().cpu().numpy()
            pred_layers = pred["layers"].detach().cpu().numpy()
            pred_total = pred["total"].detach().cpu().numpy()
            gt_counts = gt_layers.sum(axis=(1, 2))
            pred_counts = pred_layers.sum(axis=(1, 2))
            gt_total_count = float(gt_total.sum())
            pred_total_count = float(pred["count"].detach().cpu())

            row = {
                "partition_id": spec.partition_id,
                "num_strata": spec.num_strata,
                "thresholds": json.dumps(list(spec.thresholds)),
                "seed": seed,
                "checkpoint_path": str(Path(cli.checkpoint).resolve()),
                "copy_count": copy_count,
                "sample_name": sample_name,
                "frame_index": frame_index,
                "gt_total_count": gt_total_count,
                "pred_total_count": pred_total_count,
                "abs_error_total": abs(pred_total_count - gt_total_count),
                "rel_error_total": abs(pred_total_count - gt_total_count) / max(gt_total_count, 1e-8),
                "total_density_rmse": float(np.sqrt(np.mean((pred_total - gt_total) ** 2))),
                "inference_time_sec": elapsed,
            }
            for i in range(spec.num_strata):
                row[f"gt_s{i}_count"] = float(gt_counts[i])
                row[f"pred_s{i}_count"] = float(pred_counts[i])
                row[f"abs_error_s{i}"] = abs(float(pred_counts[i] - gt_counts[i]))
                row[f"density_rmse_s{i}"] = float(
                    np.sqrt(np.mean((pred_layers[i] - gt_layers[i]) ** 2))
                )

            matrix, numerator, denominator = allocation_components(gt_layers, pred_layers)
            row["allocation_diagonal_mean"] = float(np.trace(matrix) / spec.num_strata)
            row.update(flatten_matrix("alloc", matrix))
            for i in range(spec.num_strata):
                row[f"alloc_den_s{i}"] = float(denominator[i])
                for j in range(spec.num_strata):
                    row[f"alloc_num_s{i}_to_s{j}"] = float(numerator[i, j])
            frame_rows.append(row)

            if boundary_enabled:
                targets = position_cache.get(path.parent, frame_index)
                bin_gt = build_boundary_bin_density(
                    targets,
                    thresholds=spec.thresholds,
                    num_strata=spec.num_strata,
                    bin_edges=bin_edges,
                    image_size=tuple(config["image_size"]),
                    density_size=tuple(config["density_size"]),
                    sigma=float(config["gaussian_sigma"]),
                    radius=int(config["gaussian_radius"]),
                )
                fractions = np.clip(pred_layers, 0.0, None)
                fractions = fractions / (fractions.sum(axis=0, keepdims=True) + 1e-8)
                for b in range(len(bin_labels)):
                    den_b = bin_gt[b].sum(axis=(1, 2)).astype(np.float64)
                    if float(den_b.sum()) <= 0:
                        continue
                    num_b = np.zeros((spec.num_strata, spec.num_strata), dtype=np.float64)
                    for i in range(spec.num_strata):
                        for j in range(spec.num_strata):
                            num_b[i, j] = float(np.sum(bin_gt[b, i] * fractions[j]))
                    payload = boundary_store[(copy_count, sample_name, b)]
                    payload["num"] += num_b
                    payload["den"] += den_b

            if completed == 1 or completed == len(loader) or completed % 50 == 0:
                recent = pd.DataFrame(frame_rows[-min(len(frame_rows), 500):])
                progress.set_postfix(
                    {
                        "T-MAE": f"{recent['abs_error_total'].mean():.3f}",
                        "S-MAE": f"{np.mean([recent[f'abs_error_s{i}'].mean() for i in range(spec.num_strata)]):.3f}",
                        "D-RMSE": f"{recent['total_density_rmse'].mean():.5f}",
                    },
                    refresh=False,
                )

    frame_df = pd.DataFrame(frame_rows)
    sequence_df = aggregate_sequence_rows(frame_df, spec.num_strata)
    scale_df = aggregate_scale_rows(sequence_df)
    frame_df.to_csv(output_dir / "frame_results.csv", index=False, encoding="utf-8-sig")
    sequence_df.to_csv(output_dir / "sequence_results.csv", index=False, encoding="utf-8-sig")
    scale_df.to_csv(output_dir / "copy_count_results.csv", index=False, encoding="utf-8-sig")

    num = np.zeros((spec.num_strata, spec.num_strata), dtype=np.float64)
    den = np.zeros(spec.num_strata, dtype=np.float64)
    for i in range(spec.num_strata):
        den[i] = float(frame_df[f"alloc_den_s{i}"].sum())
        for j in range(spec.num_strata):
            num[i, j] = float(frame_df[f"alloc_num_s{i}_to_s{j}"].sum())
    micro = num / (den[:, None] + 1e-8)
    allocation_rows = []
    for i in range(spec.num_strata):
        allocation_rows.append(
            {
                "aggregation": "micro",
                "reference_stratum": f"s{i}",
                **{f"pred_s{j}": float(micro[i, j]) for j in range(spec.num_strata)},
            }
        )
    save_rows(output_dir / "allocation_matrices.csv", allocation_rows)

    boundary_rows = boundary_rows_from_store(boundary_store, spec, bin_labels) if boundary_enabled else []
    save_rows(output_dir / "boundary_sequence_results.csv", boundary_rows)

    summary = {
        "experiment_type": "distance_partition_sensitivity",
        "test_runner_version": "partition_sensitivity_v1",
        "partition_id": spec.partition_id,
        "num_strata": spec.num_strata,
        "thresholds": list(spec.thresholds),
        "seed": seed,
        "checkpoint_path": str(Path(cli.checkpoint).resolve()),
        "num_frames": int(len(frame_df)),
        "num_sequences": int(len(sequence_df)),
        "trainable_parameters": int(count_trainable_parameters(model)),
        "mean_total_count_mae": float(frame_df["abs_error_total"].mean()),
        "mean_total_relative_error": float(frame_df["rel_error_total"].mean()),
        "mean_total_density_rmse": float(frame_df["total_density_rmse"].mean()),
        "mean_macro_stratum_relative_error": float(sequence_df["macro_stratum_relative_error"].mean()),
        "mean_stratum_count_mae": float(
            np.mean([frame_df[f"abs_error_s{i}"].mean() for i in range(spec.num_strata)])
        ),
        "allocation_micro": micro.tolist(),
        "allocation_micro_diagonal_mean": float(np.trace(micro) / spec.num_strata),
        "mean_inference_time_sec": float(frame_df["inference_time_sec"].mean()),
        "total_elapsed_sec": float(time.perf_counter() - total_start),
        "boundary_analysis": boundary_enabled,
        "boundary_bin_edges": list(bin_edges),
    }
    for i in range(spec.num_strata):
        summary[f"mean_s{i}_count_mae"] = float(frame_df[f"abs_error_s{i}"].mean())
        summary[f"mean_s{i}_density_rmse"] = float(frame_df[f"density_rmse_s{i}"].mean())
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
