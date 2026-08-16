from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
STRATA = {0: 'Near', 1: 'Middle', 2: 'Far'}
SCALES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

def find_frame(sample_dir, frame):
    for fmt in (f'frame_{frame:03d}.png', f'frame_{frame:04d}.png'):
        p = sample_dir / fmt
        if p.exists():
            return p
    raise FileNotFoundError(sample_dir / f'frame_{frame:03d}.png')

def threshold(gray, floor=8.0, mad_multiplier=6.0):
    values = gray.astype(np.float32).ravel()
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return min(med + max(floor, mad_multiplier * 1.4826 * mad), 254.0)

def appearance(gray, frame_df, max_radius=14.0, min_pixels=2):
    h, w = gray.shape
    d = frame_df[(frame_df.visible == 1) & frame_df.computed_layer_id.isin([0, 1, 2])].copy()
    if d.empty:
        return []
    x = d.pixel_x.to_numpy(float)
    y = d.pixel_y.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    d = d.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    x, y = (x[valid], y[valid])
    if len(d) == 0:
        return []
    fy, fx = np.nonzero(gray >= threshold(gray))
    if len(fx) == 0:
        return []
    tree = cKDTree(np.column_stack([x, y]))
    dist, idx = tree.query(np.column_stack([fx, fy]), k=1, workers=-1)
    keep = dist <= max_radius
    idx = idx[keep]
    vals = gray[fy[keep], fx[keep]].astype(float)
    areas = np.bincount(idx, minlength=len(d))
    sums = np.bincount(idx, weights=vals, minlength=len(d))
    means = np.divide(sums, areas, out=np.full(len(d), np.nan), where=areas > 0)
    rows = []
    for i, r in d.iterrows():
        if areas[i] < min_pixels or not np.isfinite(means[i]):
            continue
        layer = int(r.computed_layer_id)
        rows.append({'satellite_name': r.satellite_name, 'layer_id': layer, 'stratum': STRATA[layer], 'apparent_size_px': float(areas[i]), 'mean_intensity': float(means[i])})
    return rows

def displacements(df):
    d = df[(df.visible == 1) & df.computed_layer_id.isin([0, 1, 2])].copy()
    rows = []
    for name, g in d.groupby('satellite_name', sort=False):
        g = g.sort_values('frame')
        f = g.frame.to_numpy(int)
        x = g.pixel_x.to_numpy(float)
        y = g.pixel_y.to_numpy(float)
        layer = g.computed_layer_id.to_numpy(int)
        if len(g) < 2:
            continue
        mag = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        for j in np.flatnonzero(f[1:] == f[:-1] + 1):
            rows.append({'satellite_name': name, 'frame': int(f[j + 1]), 'layer_id': int(layer[j + 1]), 'stratum': STRATA[int(layer[j + 1])], 'displacement_px': float(mag[j])})
    return rows

def stats(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return (int(len(v)), float(np.percentile(v, 50)), float(np.percentile(v, 25)), float(np.percentile(v, 75)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--test_start', type=int, default=36)
    p.add_argument('--test_end', type=int, default=40)
    args = p.parse_args()
    root, out = (Path(args.root_dir), Path(args.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    app_rows, disp_rows = ([], [])
    for scale in SCALES:
        for sample_idx in range(args.test_start, args.test_end + 1):
            sample = root / str(scale) / f'sample_{sample_idx:04d}'
            pos = pd.read_csv(sample / 'frame_positions.csv')
            for c in ['frame', 'visible', 'computed_layer_id']:
                pos[c] = pd.to_numeric(pos[c], errors='raise').astype(int)
            for c in ['pixel_x', 'pixel_y']:
                pos[c] = pd.to_numeric(pos[c], errors='raise')
            for row in displacements(pos):
                row.update(population_scale=scale, sample_index=sample_idx)
                disp_rows.append(row)
            grouped = {int(f): g for f, g in pos.groupby('frame')}
            for frame in range(100):
                gray = cv2.imread(str(find_frame(sample, frame)), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise RuntimeError(f'cannot read frame {frame} in {sample}')
                for row in appearance(gray, grouped.get(frame, pos.iloc[0:0])):
                    row.update(population_scale=scale, sample_index=sample_idx, frame=frame)
                    app_rows.append(row)
    app = pd.DataFrame(app_rows)
    disp = pd.DataFrame(disp_rows)
    app.to_csv(out / 'procedural_cue_appearance_values.csv.gz', index=False, compression='gzip')
    disp.to_csv(out / 'procedural_cue_displacement_values.csv.gz', index=False, compression='gzip')
    summary = []
    for s in ['Near', 'Middle', 'Far']:
        a = app[app.stratum == s]
        d = disp[disp.stratum == s]
        ns, sm, sq1, sq3 = stats(a.apparent_size_px)
        _, im, iq1, iq3 = stats(a.mean_intensity)
        nd, dm, dq1, dq3 = stats(d.displacement_px)
        summary.append({'Stratum': s, 'Appearance_N': ns, 'Size_Median': sm, 'Size_Q1': sq1, 'Size_Q3': sq3, 'Intensity_Median': im, 'Intensity_Q1': iq1, 'Intensity_Q3': iq3, 'Displacement_N': nd, 'Displacement_Median': dm, 'Displacement_Q1': dq1, 'Displacement_Q3': dq3})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'procedural_cue_summary.csv', index=False)
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
