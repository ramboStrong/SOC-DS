import argparse
import os
import csv
import math
import numpy as np


IMAGE_W = 640
IMAGE_H = 360

MAP_W = 80
MAP_H = 45

DOWNSAMPLE = IMAGE_W // MAP_W

DEPTH_OUTPUT_DIR_NAME = "camera_depth_maps"
MASK_OUTPUT_DIR_NAME = "camera_depth_masks"

CHECK_CSV_NAME = "camera_depth_check.csv"

DEPTH_NORM_MIN = 140.0
DEPTH_NORM_MAX = 460.0

CLIP_DEPTH_TO_01 = True

SIGMA = 1.5
RADIUS = int(3 * SIGMA)

SAVE_DTYPE = np.float32

OVERWRITE = True

VERBOSE = True


def log(text):
    if VERBOSE:
        print(text)


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def normalize_depth(camera_depth):
    denom = DEPTH_NORM_MAX - DEPTH_NORM_MIN
    if denom <= 1e-8:
        raise ValueError("DEPTH_NORM_MAX must be greater than DEPTH_NORM_MIN")

    d = (camera_depth - DEPTH_NORM_MIN) / denom

    if CLIP_DEPTH_TO_01:
        d = min(max(d, 0.0), 1.0)

    return float(d)


def parse_frame_index_from_name(name):
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    return int(stem.split("_")[-1])


def add_depth_gaussian(depth_sum, weight_sum, center_x, center_y, depth_value, sigma=1.5, radius=4):
    h, w = depth_sum.shape

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
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma ** 2)
    ).astype(np.float32)

    depth_sum[y0:y1 + 1, x0:x1 + 1] += depth_value * kernel
    weight_sum[y0:y1 + 1, x0:x1 + 1] += kernel


def load_camera_depth_points(csv_path):
    all_frames = set()
    frame_points = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_fields = ["frame", "pixel_x", "pixel_y", "visible", "camera_depth"]
        for field in required_fields:
            if field not in reader.fieldnames:
                raise KeyError(f"{csv_path}  is missing required field: {field}")

        for row in reader:
            frame_idx = safe_int(row.get("frame", -1), default=-1)
            if frame_idx < 0:
                continue

            all_frames.add(frame_idx)

            visible = safe_int(row.get("visible", 0), default=0)
            if visible != 1:
                continue

            pixel_x = safe_float(row.get("pixel_x", 0.0))
            pixel_y = safe_float(row.get("pixel_y", 0.0))
            camera_depth = safe_float(row.get("camera_depth", 0.0))

            if pixel_x < 0 or pixel_x >= IMAGE_W:
                continue
            if pixel_y < 0 or pixel_y >= IMAGE_H:
                continue

            depth_norm = normalize_depth(camera_depth)

            frame_points.setdefault(frame_idx, []).append({
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "camera_depth": camera_depth,
                "depth_norm": depth_norm,
            })

    return all_frames, frame_points


def generate_one_frame_depth_map(points):
    depth_sum = np.zeros((MAP_H, MAP_W), dtype=np.float32)
    weight_sum = np.zeros((MAP_H, MAP_W), dtype=np.float32)

    for p in points:
        gx = p["pixel_x"] / DOWNSAMPLE
        gy = p["pixel_y"] / DOWNSAMPLE

        gx = min(max(gx, 0.0), MAP_W - 1e-6)
        gy = min(max(gy, 0.0), MAP_H - 1e-6)

        add_depth_gaussian(
            depth_sum=depth_sum,
            weight_sum=weight_sum,
            center_x=gx,
            center_y=gy,
            depth_value=p["depth_norm"],
            sigma=SIGMA,
            radius=RADIUS
        )

    eps = 1e-8
    depth_map = np.zeros_like(depth_sum, dtype=np.float32)
    valid = weight_sum > eps
    depth_map[valid] = depth_sum[valid] / (weight_sum[valid] + eps)

    depth_mask = np.clip(weight_sum, 0.0, 1.0).astype(np.float32)

    depth_map_1ch = depth_map[None, :, :].astype(SAVE_DTYPE)
    depth_mask_1ch = depth_mask[None, :, :].astype(SAVE_DTYPE)

    return depth_map_1ch, depth_mask_1ch


def process_one_sample(sample_dir):
    csv_path = os.path.join(sample_dir, "frame_positions.csv")
    if not os.path.exists(csv_path):
        log(f"[skip] frame_positions.csv not found: {sample_dir}")
        return False

    depth_out_dir = os.path.join(sample_dir, DEPTH_OUTPUT_DIR_NAME)
    mask_out_dir = os.path.join(sample_dir, MASK_OUTPUT_DIR_NAME)

    os.makedirs(depth_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    all_frames, frame_points = load_camera_depth_points(csv_path)

    if len(all_frames) == 0:
        log(f"[warning] no valid frames in CSV: {csv_path}")
        return False

    all_frames = sorted(list(all_frames))

    check_rows = []

    for frame_idx in all_frames:
        depth_out_path = os.path.join(depth_out_dir, f"frame_{frame_idx:03d}.npy")
        mask_out_path = os.path.join(mask_out_dir, f"frame_{frame_idx:03d}.npy")

        if (not OVERWRITE) and os.path.exists(depth_out_path) and os.path.exists(mask_out_path):
            continue

        points = frame_points.get(frame_idx, [])

        depth_map, depth_mask = generate_one_frame_depth_map(points)

        np.save(depth_out_path, depth_map)
        np.save(mask_out_path, depth_mask)

        if len(points) > 0:
            raw_depths = [p["camera_depth"] for p in points]
            norm_depths = [p["depth_norm"] for p in points]
            raw_min = float(np.min(raw_depths))
            raw_max = float(np.max(raw_depths))
            raw_mean = float(np.mean(raw_depths))
            norm_min = float(np.min(norm_depths))
            norm_max = float(np.max(norm_depths))
            norm_mean = float(np.mean(norm_depths))
        else:
            raw_min = raw_max = raw_mean = 0.0
            norm_min = norm_max = norm_mean = 0.0

        check_rows.append({
            "frame": frame_idx,
            "visible_target_count": len(points),
            "raw_depth_min": raw_min,
            "raw_depth_max": raw_max,
            "raw_depth_mean": raw_mean,
            "norm_depth_min": norm_min,
            "norm_depth_max": norm_max,
            "norm_depth_mean": norm_mean,
            "depth_map_min": float(depth_map.min()),
            "depth_map_max": float(depth_map.max()),
            "depth_map_mean": float(depth_map.mean()),
            "mask_sum": float(depth_mask.sum()),
            "mask_max": float(depth_mask.max()),
        })

    check_csv_path = os.path.join(sample_dir, CHECK_CSV_NAME)
    with open(check_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "frame",
            "visible_target_count",
            "raw_depth_min",
            "raw_depth_max",
            "raw_depth_mean",
            "norm_depth_min",
            "norm_depth_max",
            "norm_depth_mean",
            "depth_map_min",
            "depth_map_max",
            "depth_map_mean",
            "mask_sum",
            "mask_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(check_rows)

    log(f"[done] {sample_dir} | frames={len(all_frames)}")
    return True


def collect_sample_dirs(root_dir):
    sample_dirs = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    for level1 in sorted(os.listdir(root_dir)):
        level1_path = os.path.join(root_dir, level1)

        if not os.path.isdir(level1_path):
            continue

        if not level1.isdigit():
            continue

        for name in sorted(os.listdir(level1_path)):
            sample_dir = os.path.join(level1_path, name)

            if os.path.isdir(sample_dir) and name.startswith("sample_"):
                sample_dirs.append(sample_dir)

    return sample_dirs


def batch_process_root(root_dir):
    sample_dirs = collect_sample_dirs(root_dir)

    if len(sample_dirs) == 0:
        print("No sample_xxxx directories found.")
        return

    print("=" * 90)
    print("Generating camera_depth_maps and camera_depth_masks")
    print(f"ROOT_DIR: {root_dir}")
    print(f"sample count: {len(sample_dirs)}")
    print(f"output directories: {DEPTH_OUTPUT_DIR_NAME}, {MASK_OUTPUT_DIR_NAME}")
    print(f"label resolution: {MAP_W}x{MAP_H}")
    print(f"depth normalization range: [{DEPTH_NORM_MIN}, {DEPTH_NORM_MAX}]")
    print("=" * 90)

    success_count = 0

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print("-" * 90)
        print(f"[{idx}/{len(sample_dirs)}] processing: {sample_dir}")

        ok = process_one_sample(sample_dir)
        if ok:
            success_count += 1

    print("=" * 90)
    print(f"Completed: {success_count}/{len(sample_dirs)}  samples successfully.")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    args = parser.parse_args()
    batch_process_root(args.root_dir)
