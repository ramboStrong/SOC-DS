import argparse
import os
import csv
import cv2
import time
import math
import random
import hashlib
import numpy as np


STAR_ENHANCEMENT_VERSION = "deterministic-v1"

FRAME_PREFIX = "frame_"
FRAME_SUFFIX = ".png"

OUTPUT_SUBDIR = "star_enhanced"

SAVE_BASE_STARFIELD = True
BASE_STARFIELD_NAME = "base_starfield.png"

SAVE_METADATA = True
METADATA_FILENAME = "star_enhancement_metadata.txt"

VERBOSE = True


STAR_COUNT_MIN = 120
STAR_COUNT_MAX = 260

BRIGHT_STAR_RATIO = 0.08

EXCLUSION_RADIUS = 5

NORMAL_STAR_RADIUS_CHOICES = [1, 1, 1, 1, 2]
BRIGHT_STAR_RADIUS_CHOICES = [2, 2, 3]

NORMAL_STAR_INTENSITY_RANGE = (90, 180)
BRIGHT_STAR_INTENSITY_RANGE = (180, 255)

COLOR_JITTER = 0.08

ENABLE_SOFT_GLOW = True
SOFT_GLOW_SIGMA_SMALL = 0.8
SOFT_GLOW_SIGMA_LARGE = 1.8

ENABLE_FRAME_TWINKLE = True
TWINKLE_STRENGTH = 0.08

ENABLE_SENSOR_NOISE = True
NOISE_STD = 3.0

ENABLE_BACKGROUND_OFFSET = True
BACKGROUND_OFFSET_RANGE = (0, 4)

USE_SAMPLE_BASED_SEED = True


def log(msg):
    if VERBOSE:
        print(msg)


def safe_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def get_deterministic_seed_from_string(s):
    md5 = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(md5[:8], 16)


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_frame_images(sample_dir):
    frame_files = []
    for name in os.listdir(sample_dir):
        if name.startswith(FRAME_PREFIX) and name.endswith(FRAME_SUFFIX):
            frame_files.append(name)
    frame_files.sort()
    return [os.path.join(sample_dir, x) for x in frame_files]


def load_all_visible_positions(csv_path):
    all_positions = []

    if not os.path.exists(csv_path):
        return all_positions

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            visible = safe_int(row.get("visible", 0), default=0)
            if visible != 1:
                continue

            x = safe_float(row.get("pixel_x", 0.0))
            y = safe_float(row.get("pixel_y", 0.0))

            all_positions.append((x, y))

    return all_positions


def build_exclusion_mask(height, width, positions, exclusion_radius):
    mask = np.zeros((height, width), dtype=np.uint8)

    for x, y in positions:
        cx = int(round(x))
        cy = int(round(y))

        if 0 <= cx < width and 0 <= cy < height:
            cv2.circle(mask, (cx, cy), exclusion_radius, 255, thickness=-1)

    return mask.astype(bool)


def compute_star_count(width, height, rng):
    base_area = 640 * 360
    area = width * height
    scale = area / base_area

    low = max(10, int(STAR_COUNT_MIN * scale))
    high = max(low + 1, int(STAR_COUNT_MAX * scale))
    return rng.randint(low, high)


def sample_star_positions(width, height, star_count, exclusion_mask, rng):
    positions = []

    max_attempts = star_count * 50
    attempts = 0

    while len(positions) < star_count and attempts < max_attempts:
        attempts += 1
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)

        if exclusion_mask[y, x]:
            continue

        positions.append((x, y))

    return positions


def generate_star_parameters(star_positions, rng):
    star_params = []

    bright_star_count = int(len(star_positions) * BRIGHT_STAR_RATIO)
    bright_indices = set(rng.sample(range(len(star_positions)), bright_star_count)) if len(star_positions) > 0 else set()

    for idx, (x, y) in enumerate(star_positions):
        is_bright = idx in bright_indices

        if is_bright:
            radius = rng.choice(BRIGHT_STAR_RADIUS_CHOICES)
            intensity = rng.randint(*BRIGHT_STAR_INTENSITY_RANGE)
            glow_sigma = SOFT_GLOW_SIGMA_LARGE
        else:
            radius = rng.choice(NORMAL_STAR_RADIUS_CHOICES)
            intensity = rng.randint(*NORMAL_STAR_INTENSITY_RANGE)
            glow_sigma = SOFT_GLOW_SIGMA_SMALL

        base = intensity / 255.0
        r_scale = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)
        g_scale = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)
        b_scale = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)

        color = np.array([
            np.clip(base * b_scale, 0.0, 1.0),
            np.clip(base * g_scale, 0.0, 1.0),
            np.clip(base * r_scale, 0.0, 1.0),
        ], dtype=np.float32)

        twinkle_phase = rng.uniform(0.0, 2.0 * math.pi)

        star_params.append({
            "x": x,
            "y": y,
            "radius": radius,
            "color": color,
            "glow_sigma": glow_sigma,
            "is_bright": is_bright,
            "twinkle_phase": twinkle_phase,
        })

    return star_params


def render_starfield(height, width, star_params, py_rng, frame_idx=None):
    canvas = np.zeros((height, width, 3), dtype=np.float32)

    if ENABLE_BACKGROUND_OFFSET:
        offset = py_rng.randint(*BACKGROUND_OFFSET_RANGE)
        canvas += float(offset)

    for p in star_params:
        x = p["x"]
        y = p["y"]
        radius = p["radius"]
        color = p["color"].copy()

        if ENABLE_FRAME_TWINKLE and frame_idx is not None:
            factor = 1.0 + TWINKLE_STRENGTH * math.sin(0.25 * frame_idx + p["twinkle_phase"])
            color = np.clip(color * factor, 0.0, 1.0)

        draw_color = tuple(float(c * 255.0) for c in color)
        cv2.circle(canvas, (x, y), radius, draw_color, thickness=-1, lineType=cv2.LINE_AA)

    if ENABLE_SOFT_GLOW:
        glow_small = cv2.GaussianBlur(canvas, ksize=(0, 0), sigmaX=SOFT_GLOW_SIGMA_SMALL, sigmaY=SOFT_GLOW_SIGMA_SMALL)
        glow_large = cv2.GaussianBlur(canvas, ksize=(0, 0), sigmaX=SOFT_GLOW_SIGMA_LARGE, sigmaY=SOFT_GLOW_SIGMA_LARGE)
        canvas = 0.75 * canvas + 0.18 * glow_small + 0.10 * glow_large

    return np.clip(canvas, 0, 255)


def enhance_one_image(image_bgr, starfield_bgr, rng):
    img = image_bgr.astype(np.float32)
    stars = starfield_bgr.astype(np.float32)

    enhanced = img + stars

    if ENABLE_SENSOR_NOISE:
        noise = rng.normal(loc=0.0, scale=NOISE_STD, size=enhanced.shape).astype(np.float32)
        enhanced += noise

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def save_metadata(sample_dir, width, height, star_count, visible_target_count, seed, sample_key):
    if not SAVE_METADATA:
        return

    out_path = os.path.join(sample_dir, OUTPUT_SUBDIR, METADATA_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"version: {STAR_ENHANCEMENT_VERSION}\n")
        f.write(f"sample_key: {sample_key}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"image_width: {width}\n")
        f.write(f"image_height: {height}\n")
        f.write(f"star_count: {star_count}\n")
        f.write(f"visible_target_count_all_frames: {visible_target_count}\n")
        f.write(f"exclusion_radius: {EXCLUSION_RADIUS}\n")
        f.write(f"bright_star_ratio: {BRIGHT_STAR_RATIO}\n")
        f.write(f"normal_star_intensity_range: {NORMAL_STAR_INTENSITY_RANGE}\n")
        f.write(f"bright_star_intensity_range: {BRIGHT_STAR_INTENSITY_RANGE}\n")
        f.write(f"enable_soft_glow: {ENABLE_SOFT_GLOW}\n")
        f.write(f"enable_frame_twinkle: {ENABLE_FRAME_TWINKLE}\n")
        f.write(f"twinkle_strength: {TWINKLE_STRENGTH}\n")
        f.write(f"enable_sensor_noise: {ENABLE_SENSOR_NOISE}\n")
        f.write(f"noise_std: {NOISE_STD}\n")
        f.write(f"enable_background_offset: {ENABLE_BACKGROUND_OFFSET}\n")
        f.write(f"background_offset_range: {BACKGROUND_OFFSET_RANGE}\n")


def process_one_sample(sample_dir):
    frame_paths = find_frame_images(sample_dir)
    if len(frame_paths) == 0:
        log(f"[skip] no frame images found: {sample_dir}")
        return False

    csv_path = os.path.join(sample_dir, "frame_positions.csv")
    visible_positions = load_all_visible_positions(csv_path)

    first_img = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        log(f"[skip] failed to read first frame: {frame_paths[0]}")
        return False

    height, width = first_img.shape[:2]

    output_dir = os.path.join(sample_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    sample_key = f"{os.path.basename(os.path.dirname(sample_dir))}/{os.path.basename(sample_dir)}"
    if USE_SAMPLE_BASED_SEED:
        seed = get_deterministic_seed_from_string(sample_key)
    else:
        seed = random.SystemRandom().randint(0, 10**9)

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    exclusion_mask = build_exclusion_mask(
        height=height,
        width=width,
        positions=visible_positions,
        exclusion_radius=EXCLUSION_RADIUS
    )

    star_count = compute_star_count(width, height, py_rng)
    star_positions = sample_star_positions(width, height, star_count, exclusion_mask, py_rng)
    star_params = generate_star_parameters(star_positions, py_rng)

    if SAVE_BASE_STARFIELD:
        base_starfield = render_starfield(height, width, star_params, py_rng, frame_idx=None)
        base_starfield_path = os.path.join(output_dir, BASE_STARFIELD_NAME)
        cv2.imwrite(base_starfield_path, np.clip(base_starfield, 0, 255).astype(np.uint8))

    for idx, img_path in enumerate(frame_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            log(f"[warning] failed to read image; skipped: {img_path}")
            continue

        frame_idx = idx
        starfield = render_starfield(height, width, star_params, py_rng, frame_idx=frame_idx)
        enhanced = enhance_one_image(img, starfield, np_rng)

        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, enhanced)

    save_metadata(
        sample_dir=sample_dir,
        width=width,
        height=height,
        star_count=len(star_positions),
        visible_target_count=len(visible_positions),
        seed=seed,
        sample_key=sample_key
    )

    log(f"[done] {sample_dir} | stars={len(star_positions)} | visible_targets(all frames)={len(visible_positions)}")
    return True


def batch_process_root(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    sample_dirs = []

    for level1 in sorted(os.listdir(root_dir)):
        level1_path = os.path.join(root_dir, level1)
        if not os.path.isdir(level1_path):
            continue

        for name in sorted(os.listdir(level1_path)):
            sample_dir = os.path.join(level1_path, name)
            if os.path.isdir(sample_dir) and name.startswith("sample_"):
                sample_dirs.append(sample_dir)

    total = len(sample_dirs)
    if total == 0:
        print("No sample_xxxx directories found.")
        return

    start_time = time.time()
    done = 0

    for i, sample_dir in enumerate(sample_dirs, start=1):
        print("=" * 90)
        print(f"[{i}/{total}] processing: {sample_dir}")

        t0 = time.time()
        ok = process_one_sample(sample_dir)
        done += 1 if ok else 0

        elapsed = time.time() - start_time
        avg = elapsed / i
        remain = avg * (total - i)

        print(f"[progress] {i}/{total} | elapsed {format_seconds(elapsed)} | remaining {format_seconds(remain)}")

    total_elapsed = time.time() - start_time
    print("=" * 90)
    print(f"Star enhancement completed. Processed {done}/{total}  samples.")
    print(f"Total time: {format_seconds(total_elapsed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    args = parser.parse_args()
    batch_process_root(args.root_dir)
