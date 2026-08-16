from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SEED_SPLIT = {
    "train": (1, 30),
    "val": (31, 35),
    "test": (36, 40),
}


@dataclass(frozen=True)
class PartitionConfig:
    partition_id: str
    num_strata: int
    thresholds: Tuple[float, ...]
    threshold_source: str
    description: str

    def validate(self) -> None:
        if self.num_strata < 2:
            raise ValueError(f"{self.partition_id}: num_strata must be >= 2")
        if len(self.thresholds) != self.num_strata - 1:
            raise ValueError(
                f"{self.partition_id}: expected {self.num_strata - 1} thresholds, "
                f"got {len(self.thresholds)}"
            )
        if not all(np.isfinite(self.thresholds)):
            raise ValueError(f"{self.partition_id}: thresholds contain non-finite values")
        if any(b <= a for a, b in zip(self.thresholds[:-1], self.thresholds[1:])):
            raise ValueError(f"{self.partition_id}: thresholds must be strictly increasing")


@dataclass
class Point:
    frame_idx: int
    pixel_x: float
    pixel_y: float
    camera_depth: float
    original_layer_id: Optional[int]


@dataclass
class SampleRows:
    all_frame_indices: List[int]
    visible_points_by_frame: Dict[int, List[Point]]
    invalid_visible_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SOC-DS labels for distance-partition sensitivity experiments."
    )
    parser.add_argument("--root_dir", required=True, help="Dataset root containing numeric scale folders.")
    parser.add_argument(
        "--metadata_dir_name",
        default="partition_sensitivity_metadata",
        help="Metadata directory created directly under root_dir.",
    )
    parser.add_argument(
        "--sample_output_dir_name",
        default="partition_sensitivity",
        help="Directory created inside each sample_xxxx folder.",
    )
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--density_width", type=int, default=80)
    parser.add_argument("--density_height", type=int, default=45)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Gaussian truncation radius. Default: int(3 * sigma), matching the original script.",
    )
    parser.add_argument("--train_start", type=int, default=1)
    parser.add_argument("--train_end", type=int, default=30)
    parser.add_argument("--val_start", type=int, default=31)
    parser.add_argument("--val_end", type=int, default=35)
    parser.add_argument("--test_start", type=int, default=36)
    parser.add_argument("--test_end", type=int, default=40)
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["P2_Q", "P3_L", "P3_R", "P3_U", "P4_Q"],
        choices=["P2_Q", "P3_L", "P3_R", "P3_U", "P4_Q"],
        help="Subset of partition designs to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing partition-specific layer_density_maps directories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute thresholds and source-data audit without writing density maps.",
    )
    parser.add_argument(
        "--verify_reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For P3_R, compare regenerated labels with the existing layer_density_maps.",
    )
    parser.add_argument("--integral_tolerance", type=float, default=5e-5)
    parser.add_argument("--reference_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def safe_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def sample_index_from_name(name: str) -> Optional[int]:
    if not name.startswith("sample_"):
        return None
    return safe_int(name.split("_")[-1], default=None)


def split_for_sample(sample_idx: int, args: argparse.Namespace) -> Optional[str]:
    if args.train_start <= sample_idx <= args.train_end:
        return "train"
    if args.val_start <= sample_idx <= args.val_end:
        return "val"
    if args.test_start <= sample_idx <= args.test_end:
        return "test"
    return None


def discover_scale_dirs(root_dir: Path) -> List[Path]:
    numeric = sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if numeric:
        return numeric
    if any(p.is_dir() and p.name.startswith("sample_") for p in root_dir.iterdir()):
        return [root_dir]
    raise RuntimeError(
        f"No numeric scale folders or sample_xxxx directories were found under {root_dir}"
    )


def iter_samples(root_dir: Path, args: argparse.Namespace) -> Iterable[Tuple[str, int, str, Path]]:
    for scale_dir in discover_scale_dirs(root_dir):
        scale_name = scale_dir.name if scale_dir != root_dir else "root"
        for sample_dir in sorted(scale_dir.glob("sample_*")):
            if not sample_dir.is_dir():
                continue
            sample_idx = sample_index_from_name(sample_dir.name)
            if sample_idx is None:
                continue
            split = split_for_sample(sample_idx, args)
            if split is None:
                continue
            yield scale_name, sample_idx, split, sample_dir


def parse_original_layer_id(row: Mapping[str, str]) -> Optional[int]:
    for key in ("computed_layer_id", "assigned_layer_id"):
        value = safe_int(row.get(key, ""), default=None)
        if value is not None:
            return value

    one_hot_keys = ("layer_near", "layer_mid", "layer_far")
    if any(key in row for key in one_hot_keys):
        flags = [safe_int(row.get(key, 0), default=0) for key in one_hot_keys]
        active = [idx for idx, flag in enumerate(flags) if flag == 1]
        if len(active) == 1:
            return active[0]
    return None


def load_sample_rows(
    csv_path: Path,
    *,
    image_width: int,
    image_height: int,
) -> SampleRows:
    visible_points_by_frame: Dict[int, List[Point]] = defaultdict(list)
    all_frames = set()
    invalid_visible_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = {"frame", "visible", "pixel_x", "pixel_y", "camera_depth"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        for row in reader:
            frame_idx = safe_int(row.get("frame"), default=None)
            if frame_idx is None or frame_idx < 0:
                continue
            all_frames.add(frame_idx)

            if safe_int(row.get("visible"), default=0) != 1:
                continue

            pixel_x = safe_float(row.get("pixel_x"))
            pixel_y = safe_float(row.get("pixel_y"))
            camera_depth = safe_float(row.get("camera_depth"))
            if not (np.isfinite(pixel_x) and np.isfinite(pixel_y) and np.isfinite(camera_depth)):
                invalid_visible_rows += 1
                continue
            if not (0.0 <= pixel_x < image_width and 0.0 <= pixel_y < image_height):
                invalid_visible_rows += 1
                continue

            visible_points_by_frame[frame_idx].append(
                Point(
                    frame_idx=frame_idx,
                    pixel_x=float(pixel_x),
                    pixel_y=float(pixel_y),
                    camera_depth=float(camera_depth),
                    original_layer_id=parse_original_layer_id(row),
                )
            )

    return SampleRows(
        all_frame_indices=sorted(all_frames),
        visible_points_by_frame=dict(visible_points_by_frame),
        invalid_visible_rows=invalid_visible_rows,
    )


def collect_training_depths(root_dir: Path, args: argparse.Namespace) -> np.ndarray:
    depths: List[float] = []
    sample_count = 0
    invalid_count = 0
    for _, _, split, sample_dir in iter_samples(root_dir, args):
        if split != "train":
            continue
        csv_path = sample_dir / "frame_positions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing frame_positions.csv: {csv_path}")
        sample_rows = load_sample_rows(
            csv_path,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        sample_count += 1
        invalid_count += sample_rows.invalid_visible_rows
        for points in sample_rows.visible_points_by_frame.values():
            depths.extend(point.camera_depth for point in points)

    if not depths:
        raise RuntimeError("No finite visible training-target camera_depth values were found.")
    depth_array = np.asarray(depths, dtype=np.float64)
    print(
        f"[Training-depth scan] samples={sample_count}, visible targets={depth_array.size}, "
        f"invalid visible rows={invalid_count}, range=[{depth_array.min():.6f}, {depth_array.max():.6f}]"
    )
    return depth_array


def quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="linear"))


def build_partition_configs(
    training_depths: np.ndarray,
    requested: Sequence[str],
) -> List[PartitionConfig]:
    q25 = quantile(training_depths, 0.25)
    q50 = quantile(training_depths, 0.50)
    q75 = quantile(training_depths, 0.75)

    all_configs = {
        "P2_Q": PartitionConfig(
            partition_id="P2_Q",
            num_strata=2,
            thresholds=(q50,),
            threshold_source="visible training-target camera-depth median",
            description="Two-stratum quantile partition.",
        ),
        "P3_L": PartitionConfig(
            partition_id="P3_L",
            num_strata=3,
            thresholds=(200.0, 300.0),
            threshold_source="fixed left-shifted thresholds",
            description="Three-stratum design with both thresholds shifted toward smaller depth.",
        ),
        "P3_R": PartitionConfig(
            partition_id="P3_R",
            num_strata=3,
            thresholds=(220.0, 325.0),
            threshold_source="reference thresholds",
            description="Reference three-stratum design used by the main SOC-DS experiment.",
        ),
        "P3_U": PartitionConfig(
            partition_id="P3_U",
            num_strata=3,
            thresholds=(240.0, 350.0),
            threshold_source="fixed right-shifted thresholds",
            description="Three-stratum design with both thresholds shifted toward larger depth.",
        ),
        "P4_Q": PartitionConfig(
            partition_id="P4_Q",
            num_strata=4,
            thresholds=(q25, q50, q75),
            threshold_source="visible training-target camera-depth quartiles",
            description="Four-stratum quantile partition.",
        ),
    }

    configs = [all_configs[name] for name in requested]
    for config in configs:
        config.validate()
    return configs


def assign_stratum(camera_depth: float, thresholds: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(thresholds, dtype=np.float64), camera_depth, side="right"))


def add_normalized_gaussian(
    density_map: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    sigma: float,
    radius: int,
) -> None:
    h, w = density_map.shape
    x0 = max(0, int(math.floor(center_x - radius)))
    x1 = min(w - 1, int(math.ceil(center_x + radius)))
    y0 = max(0, int(math.floor(center_y - radius)))
    y1 = min(h - 1, int(math.ceil(center_y + radius)))
    if x1 < x0 or y1 < y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    kernel = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
    )
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0.0:
        kernel = kernel / kernel_sum
    density_map[y0 : y1 + 1, x0 : x1 + 1] += kernel.astype(np.float32)


def generate_frame_density(
    points: Sequence[Point],
    config: PartitionConfig,
    *,
    image_width: int,
    image_height: int,
    density_width: int,
    density_height: int,
    sigma: float,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    density = np.zeros(
        (config.num_strata, density_height, density_width), dtype=np.float32
    )
    counts = np.zeros(config.num_strata, dtype=np.int64)
    original_id_mismatches = 0

    scale_x = image_width / float(density_width)
    scale_y = image_height / float(density_height)
    for point in points:
        stratum = assign_stratum(point.camera_depth, config.thresholds)
        if not 0 <= stratum < config.num_strata:
            raise RuntimeError(
                f"Invalid stratum={stratum} for depth={point.camera_depth}, config={config}"
            )
        counts[stratum] += 1
        if config.partition_id == "P3_R" and point.original_layer_id in {0, 1, 2}:
            original_id_mismatches += int(stratum != point.original_layer_id)

        gx = min(max(point.pixel_x / scale_x, 0.0), density_width - 1e-6)
        gy = min(max(point.pixel_y / scale_y, 0.0), density_height - 1e-6)
        add_normalized_gaussian(
            density[stratum], gx, gy, sigma=sigma, radius=radius
        )

    return density, counts, original_id_mismatches


def ensure_channel_first(arr: np.ndarray, num_strata: int) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D layer density map, got {arr.shape}")
    if arr.shape[0] == num_strata:
        return arr.astype(np.float32)
    if arr.shape[-1] == num_strata:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    raise ValueError(f"Cannot identify channel axis in shape={arr.shape}, K={num_strata}")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if args.density_width <= 0 or args.density_height <= 0:
        raise ValueError("Density-map dimensions must be positive")
    if args.sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(3 * args.sigma) if args.radius is None else int(args.radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    start_time = time.time()
    training_depths = collect_training_depths(root_dir, args)
    configs = build_partition_configs(training_depths, args.partitions)

    metadata_dir = root_dir / args.metadata_dir_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(root_dir),
        "label_layout": (
            f"<sample>/{args.sample_output_dir_name}/<partition_id>/layer_density_maps/frame_xxx.npy"
        ),
        "image_size": [args.image_width, args.image_height],
        "density_size": [args.density_width, args.density_height],
        "gaussian_sigma": args.sigma,
        "gaussian_radius": radius,
        "threshold_equality_rule": "depth exactly equal to a threshold is assigned to the upper stratum",
        "training_visible_depth_count": int(training_depths.size),
        "training_visible_depth_min": float(training_depths.min()),
        "training_visible_depth_max": float(training_depths.max()),
        "training_visible_depth_quantiles": {
            "q25": quantile(training_depths, 0.25),
            "q50": quantile(training_depths, 0.50),
            "q75": quantile(training_depths, 0.75),
        },
        "split": {
            "train": [args.train_start, args.train_end],
            "val": [args.val_start, args.val_end],
            "test": [args.test_start, args.test_end],
        },
        "partitions": [
            {
                **asdict(config),
                "thresholds": list(config.thresholds),
            }
            for config in configs
        ],
        "dry_run": bool(args.dry_run),
    }
    with (metadata_dir / "partition_configs.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, ensure_ascii=False)

    print("[Partition configs]")
    for config in configs:
        formatted = ", ".join(f"{x:.6f}" for x in config.thresholds)
        print(f"  {config.partition_id}: K={config.num_strata}, thresholds=[{formatted}]")

    audit: MutableMapping[Tuple[str, str, int], Dict[str, object]] = defaultdict(
        lambda: {
            "target_count": 0,
            "frame_count": 0,
            "zero_target_frame_count": 0,
            "max_stratum_integral_error": 0.0,
            "max_total_integral_error": 0.0,
            "reference_layer_id_mismatch_count": 0,
            "reference_compared_frame_count": 0,
            "reference_missing_frame_count": 0,
            "max_reference_map_abs_diff": 0.0,
            "invalid_visible_rows": 0,
        }
    )
    split_partition_target_totals: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    failures: List[Dict[str, object]] = []
    invalid_rows_by_split: MutableMapping[str, int] = defaultdict(int)

    samples = list(iter_samples(root_dir, args))
    if not samples:
        raise RuntimeError("No samples were found in the requested split ranges.")

    for sample_no, (scale, sample_idx, split, sample_dir) in enumerate(samples, start=1):
        csv_path = sample_dir / "frame_positions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing frame_positions.csv: {csv_path}")
        sample_rows = load_sample_rows(
            csv_path,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        invalid_rows_by_split[split] += int(sample_rows.invalid_visible_rows)
        print(
            f"[{sample_no}/{len(samples)}] scale={scale} sample={sample_idx:04d} "
            f"split={split} frames={len(sample_rows.all_frame_indices)}"
        )

        for config in configs:
            output_dir = (
                sample_dir
                / args.sample_output_dir_name
                / config.partition_id
                / "layer_density_maps"
            )
            if output_dir.exists() and args.overwrite and not args.dry_run:
                shutil.rmtree(output_dir)
            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx in sample_rows.all_frame_indices:
                points = sample_rows.visible_points_by_frame.get(frame_idx, [])
                density, counts, id_mismatches = generate_frame_density(
                    points,
                    config,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    density_width=args.density_width,
                    density_height=args.density_height,
                    sigma=args.sigma,
                    radius=radius,
                )
                map_sums = density.sum(axis=(1, 2), dtype=np.float64)
                stratum_errors = np.abs(map_sums - counts.astype(np.float64))
                total_error = abs(float(map_sums.sum()) - float(counts.sum()))
                split_partition_target_totals[(config.partition_id, split)] += int(counts.sum())

                reference_diff = float("nan")
                reference_exists = False
                if args.verify_reference and config.partition_id == "P3_R":
                    original_path = sample_dir / "layer_density_maps" / f"frame_{frame_idx:03d}.npy"
                    if original_path.exists():
                        original = ensure_channel_first(np.load(original_path), 3)
                        if original.shape != density.shape:
                            failures.append(
                                {
                                    "failure_type": "reference_shape_mismatch",
                                    "partition_id": config.partition_id,
                                    "split": split,
                                    "scale": scale,
                                    "sample": sample_idx,
                                    "frame": frame_idx,
                                    "observed": str(original.shape),
                                    "expected": str(density.shape),
                                }
                            )
                        else:
                            reference_exists = True
                            reference_diff = float(np.max(np.abs(original - density)))
                            if reference_diff > args.reference_tolerance:
                                failures.append(
                                    {
                                        "failure_type": "reference_map_difference",
                                        "partition_id": config.partition_id,
                                        "split": split,
                                        "scale": scale,
                                        "sample": sample_idx,
                                        "frame": frame_idx,
                                        "observed": reference_diff,
                                        "tolerance": args.reference_tolerance,
                                    }
                                )

                if float(np.max(stratum_errors, initial=0.0)) > args.integral_tolerance:
                    failures.append(
                        {
                            "failure_type": "stratum_integral_error",
                            "partition_id": config.partition_id,
                            "split": split,
                            "scale": scale,
                            "sample": sample_idx,
                            "frame": frame_idx,
                            "observed": float(np.max(stratum_errors)),
                            "tolerance": args.integral_tolerance,
                        }
                    )
                if total_error > args.integral_tolerance:
                    failures.append(
                        {
                            "failure_type": "total_integral_error",
                            "partition_id": config.partition_id,
                            "split": split,
                            "scale": scale,
                            "sample": sample_idx,
                            "frame": frame_idx,
                            "observed": total_error,
                            "tolerance": args.integral_tolerance,
                        }
                    )

                for stratum_idx in range(config.num_strata):
                    key = (config.partition_id, split, stratum_idx)
                    item = audit[key]
                    item["target_count"] = int(item["target_count"]) + int(counts[stratum_idx])
                    item["frame_count"] = int(item["frame_count"]) + 1
                    item["zero_target_frame_count"] = int(item["zero_target_frame_count"]) + int(
                        counts[stratum_idx] == 0
                    )
                    item["max_stratum_integral_error"] = max(
                        float(item["max_stratum_integral_error"]),
                        float(stratum_errors[stratum_idx]),
                    )
                    item["max_total_integral_error"] = max(
                        float(item["max_total_integral_error"]), total_error
                    )
                    if config.partition_id == "P3_R":
                        if stratum_idx == 0:
                            item["reference_layer_id_mismatch_count"] = int(
                                item["reference_layer_id_mismatch_count"]
                            ) + int(id_mismatches)
                        if args.verify_reference:
                            if reference_exists:
                                item["reference_compared_frame_count"] = int(
                                    item["reference_compared_frame_count"]
                                ) + 1
                                item["max_reference_map_abs_diff"] = max(
                                    float(item["max_reference_map_abs_diff"]), reference_diff
                                )
                            else:
                                item["reference_missing_frame_count"] = int(
                                    item["reference_missing_frame_count"]
                                ) + 1

                out_path = output_dir / f"frame_{frame_idx:03d}.npy"
                if not args.dry_run:
                    if out_path.exists() and not args.overwrite:
                        existing = ensure_channel_first(np.load(out_path), config.num_strata)
                        if existing.shape != density.shape or not np.allclose(
                            existing, density, atol=args.integral_tolerance, rtol=0.0
                        ):
                            raise FileExistsError(
                                f"Existing label differs from regenerated label: {out_path}. "
                                "Use --overwrite only after reviewing the current output."
                            )
                    else:
                        np.save(out_path, density)

    p3_reference_mismatch_by_split = {
        split: int(audit.get(("P3_R", split, 0), {}).get("reference_layer_id_mismatch_count", 0))
        for split in ("train", "val", "test")
    }

    audit_rows: List[Dict[str, object]] = []
    config_lookup = {config.partition_id: config for config in configs}
    for (partition_id, split, stratum_idx), item in sorted(audit.items()):
        config = config_lookup[partition_id]
        partition_total = split_partition_target_totals[(partition_id, split)]
        target_count = int(item["target_count"])
        audit_rows.append(
            {
                "partition_id": partition_id,
                "num_strata": config.num_strata,
                "thresholds": json.dumps(list(config.thresholds)),
                "split": split,
                "stratum_index": stratum_idx,
                "target_count": target_count,
                "target_fraction": target_count / max(1, partition_total),
                "frame_count": int(item["frame_count"]),
                "zero_target_frame_count": int(item["zero_target_frame_count"]),
                "empty_stratum_in_split": int(target_count == 0),
                "max_stratum_integral_error": float(item["max_stratum_integral_error"]),
                "max_total_integral_error": float(item["max_total_integral_error"]),
                "invalid_visible_rows": int(invalid_rows_by_split[split]),
                "reference_layer_id_mismatch_count": (
                    p3_reference_mismatch_by_split.get(split, 0)
                    if partition_id == "P3_R"
                    else ""
                ),
                "reference_compared_frame_count": (
                    int(item["reference_compared_frame_count"])
                    if partition_id == "P3_R"
                    else ""
                ),
                "reference_missing_frame_count": (
                    int(item["reference_missing_frame_count"])
                    if partition_id == "P3_R"
                    else ""
                ),
                "max_reference_map_abs_diff": (
                    float(item["max_reference_map_abs_diff"])
                    if partition_id == "P3_R"
                    else ""
                ),
            }
        )

    audit_path = metadata_dir / "partition_label_audit_summary.csv"
    write_csv(audit_path, audit_rows)
    failure_path = metadata_dir / "partition_label_audit_failures.csv"
    if failures:
        write_csv(failure_path, failures)
    elif failure_path.exists():
        failure_path.unlink()

    elapsed = time.time() - start_time
    print("\n[Completed]")
    print(f"  metadata: {metadata_dir}")
    print(f"  audit summary: {audit_path}")
    print(f"  failures: {len(failures)}")
    print(f"  elapsed: {elapsed:.1f} s")
    if failures:
        print(f"  failure details: {failure_path}")
        sys.exit(2)


if __name__ == "__main__":
    main()
