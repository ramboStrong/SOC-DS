import argparse
import os
import csv
import math
import numpy as np


IMAGE_W = 640
IMAGE_H = 360

DENSITY_W = 80
DENSITY_H = 45

DOWNSAMPLE = IMAGE_W // DENSITY_W

SIGMA = 1.5
RADIUS = int(3 * SIGMA)

OUTPUT_DIR_NAME = "layer_density_maps"

SAVE_TOTAL_DENSITY = True
TOTAL_OUTPUT_DIR_NAME = "total_density_maps"

SAVE_CHECK_CSV = True


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default


def add_normalized_gaussian(density_map, center_x, center_y, sigma=1.5, radius=4):
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

    kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    kernel_sum = kernel.sum()

    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    density_map[y0:y1 + 1, x0:x1 + 1] += kernel.astype(np.float32)


def parse_layer_id(row):
    if "computed_layer_id" in row and row["computed_layer_id"] != "":
        return safe_int(row["computed_layer_id"], default=-1)

    near_flag = safe_int(row.get("layer_near", 0))
    mid_flag = safe_int(row.get("layer_mid", 0))
    far_flag = safe_int(row.get("layer_far", 0))

    if near_flag == 1:
        return 0
    elif mid_flag == 1:
        return 1
    elif far_flag == 1:
        return 2
    else:
        return -1


def load_points_from_csv(csv_path):
    frame_points = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_idx = safe_int(row.get("frame", -1), default=-1)
            visible = safe_int(row.get("visible", 0), default=0)

            if frame_idx < 0:
                continue

            if visible != 1:
                continue

            pixel_x = safe_float(row.get("pixel_x", 0.0))
            pixel_y = safe_float(row.get("pixel_y", 0.0))
            layer_id = parse_layer_id(row)

            if layer_id not in [0, 1, 2]:
                continue

            if pixel_x < 0 or pixel_x >= IMAGE_W or pixel_y < 0 or pixel_y >= IMAGE_H:
                continue

            if frame_idx not in frame_points:
                frame_points[frame_idx] = []

            frame_points[frame_idx].append((pixel_x, pixel_y, layer_id))

    return frame_points


def generate_density_for_one_frame(points):
    density_3ch = np.zeros((3, DENSITY_H, DENSITY_W), dtype=np.float32)

    for pixel_x, pixel_y, layer_id in points:
        gx = pixel_x / DOWNSAMPLE
        gy = pixel_y / DOWNSAMPLE

        gx = min(max(gx, 0.0), DENSITY_W - 1e-6)
        gy = min(max(gy, 0.0), DENSITY_H - 1e-6)

        add_normalized_gaussian(
            density_map=density_3ch[layer_id],
            center_x=gx,
            center_y=gy,
            sigma=SIGMA,
            radius=RADIUS
        )

    return density_3ch


def process_one_sample(sample_dir):
    csv_path = os.path.join(sample_dir, "frame_positions.csv")
    if not os.path.exists(csv_path):
        print(f"[skip] not found: {csv_path}")
        return

    output_dir = os.path.join(sample_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    total_output_dir = None
    if SAVE_TOTAL_DENSITY:
        total_output_dir = os.path.join(sample_dir, TOTAL_OUTPUT_DIR_NAME)
        os.makedirs(total_output_dir, exist_ok=True)

    frame_points = load_points_from_csv(csv_path)
    if len(frame_points) == 0:
        print(f"[warning] {csv_path}  has no visible targets.")
        return

    all_frames = sorted(frame_points.keys())

    check_rows = []
    print(f"[processing] {sample_dir} | total {len(all_frames)} frames")

    for frame_idx in all_frames:
        points = frame_points[frame_idx]
        density_3ch = generate_density_for_one_frame(points)

        out_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.npy")
        np.save(out_path, density_3ch)

        if SAVE_TOTAL_DENSITY:
            total_density = density_3ch.sum(axis=0)
            total_out_path = os.path.join(total_output_dir, f"frame_{frame_idx:03d}.npy")
            np.save(total_out_path, total_density)

        gt_counts = [0, 0, 0]
        for _, _, layer_id in points:
            gt_counts[layer_id] += 1

        map_sums = [
            float(density_3ch[0].sum()),
            float(density_3ch[1].sum()),
            float(density_3ch[2].sum())
        ]

        total_gt = sum(gt_counts)
        total_map = sum(map_sums)

        check_rows.append([
            frame_idx,
            gt_counts[0], gt_counts[1], gt_counts[2], total_gt,
            map_sums[0], map_sums[1], map_sums[2], total_map
        ])

    if SAVE_CHECK_CSV:
        check_csv_path = os.path.join(sample_dir, "density_map_check.csv")
        with open(check_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "gt_near_count", "gt_mid_count", "gt_far_count", "gt_total_count",
                "sum_near_density", "sum_mid_density", "sum_far_density", "sum_total_density"
            ])
            writer.writerows(check_rows)

    print(f"[done] {sample_dir}")


def batch_process_root(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    copy_count_dirs = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            copy_count_dirs.append(path)

    copy_count_dirs = sorted(copy_count_dirs, key=lambda x: os.path.basename(x))

    total_samples = 0
    for copy_dir in copy_count_dirs:
        for name in os.listdir(copy_dir):
            sample_dir = os.path.join(copy_dir, name)
            if os.path.isdir(sample_dir) and name.startswith("sample_"):
                total_samples += 1

    processed = 0

    for copy_dir in copy_count_dirs:
        for name in sorted(os.listdir(copy_dir)):
            sample_dir = os.path.join(copy_dir, name)
            if os.path.isdir(sample_dir) and name.startswith("sample_"):
                processed += 1
                print("=" * 80)
                print(f"[{processed}/{total_samples}] processing: {sample_dir}")
                process_one_sample(sample_dir)

    print("=" * 80)
    print("All stratified density maps were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    args = parser.parse_args()
    batch_process_root(args.root_dir)
