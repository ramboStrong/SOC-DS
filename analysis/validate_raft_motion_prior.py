from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
try:
    from RAFT.core.raft import RAFT
    from RAFT.core.utils.utils import InputPadder
except Exception:
    RAFT = None
    InputPadder = None
STRATUM_NAMES = {0: 'Near', 1: 'Middle', 2: 'Far'}

class RAFTArgs(dict):

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean: {value}')

def parse_args():
    parser = argparse.ArgumentParser(description='Validate RAFT against simulator target displacement with sequence-aware uncertainty.')
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--raft_model', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sample_start', type=int, default=36)
    parser.add_argument('--sample_end', type=int, default=40)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=99)
    parser.add_argument('--use_star_enhanced', type=str2bool, default=True)
    parser.add_argument('--fallback_to_raw', type=str2bool, default=True)
    parser.add_argument('--raft_iters', type=int, default=4)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260723)
    parser.add_argument('--scatter_reservoir', type=int, default=100000)
    parser.add_argument('--max_pairs', type=int, default=None, help='Debug-only cap on frame pairs.')
    return parser.parse_args()

def strip_module_prefix(state_dict):
    return {key[7:] if key.startswith('module.') else key: value for key, value in state_dict.items()}

def load_raft(args, device):
    if RAFT is None or InputPadder is None:
        raise ImportError('RAFT source package is not importable. Add the cited RAFT implementation to PYTHONPATH.')
    raft_args = RAFTArgs()
    raft_args.small = args.small
    raft_args.mixed_precision = args.mixed_precision
    raft_args.alternate_corr = args.alternate_corr
    raft_args.dropout = 0
    raft = torch.nn.DataParallel(RAFT(raft_args))
    state = torch.load(args.raft_model, map_location=device)
    raft.load_state_dict(state)
    raft = raft.module.to(device)
    raft.eval()
    return raft

def load_raft_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)

def find_image(sample_dir: Path, frame: int, use_star: bool, fallback: bool) -> Optional[Path]:
    name = f'frame_{frame:03d}.png'
    candidates: List[Path] = []
    if use_star:
        candidates.append(sample_dir / 'star_enhanced' / name)
    if fallback or not use_star:
        candidates.append(sample_dir / name)
    return next((path for path in candidates if path.exists()), None)

def read_positions(csv_path: Path) -> Dict[Tuple[int, str], dict]:
    records: Dict[Tuple[int, str], dict] = {}
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row['frame']))
            name = row['satellite_name']
            records[frame, name] = {'satellite_name': name, 'pixel_x': float(row['pixel_x']), 'pixel_y': float(row['pixel_y']), 'visible': int(float(row['visible'])), 'layer': int(float(row.get('computed_layer_id', row.get('assigned_layer_id', 0))))}
    return records

def bilinear_sample_flow(flow: torch.Tensor, xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    _, _, height, width = flow.shape
    coords = torch.as_tensor(xy, dtype=flow.dtype, device=flow.device)
    x_norm = 2.0 * coords[:, 0] / max(width - 1, 1) - 1.0
    y_norm = 2.0 * coords[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(flow, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1).detach().cpu().numpy()

def angular_error_deg(pred: np.ndarray, gt: np.ndarray, eps: float=1e-08) -> float:
    pred_norm = float(np.linalg.norm(pred))
    gt_norm = float(np.linalg.norm(gt))
    if pred_norm < eps or gt_norm < eps:
        return float('nan')
    cosine = float(np.dot(pred, gt) / (pred_norm * gt_norm + eps))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

class MetricAccumulator:

    def __init__(self):
        self.n = 0
        self.sums = defaultdict(float)

    def add(self, **values):
        self.n += 1
        for key, value in values.items():
            if np.isfinite(value):
                self.sums[key] += float(value)
                self.sums[f'{key}__n'] += 1.0

    def row(self, copy_count: int, sample_name: str, stratum: str) -> dict:
        output = {'copy_count': int(copy_count), 'sample_name': sample_name, 'stratum': stratum, 'n_targets': int(self.n)}
        for key in ['gt_magnitude_px', 'raft_magnitude_px', 'epe_px', 'magnitude_abs_error_px', 'angular_error_deg']:
            denom = self.sums.get(f'{key}__n', 0.0)
            output[key] = self.sums.get(key, 0.0) / denom if denom > 0 else np.nan
        return output

def reservoir_add(reservoir: List[tuple], item: tuple, seen: int, capacity: int, rng: random.Random):
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        index = rng.randrange(seen)
        if index < capacity:
            reservoir[index] = item

def hierarchical_bootstrap_ci(sequence_df: pd.DataFrame, metric: str, stratum: str, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    group = sequence_df[sequence_df['stratum'] == stratum].dropna(subset=[metric]).copy()
    if group.empty:
        return (np.nan, np.nan, np.nan)
    point = float(group[metric].mean())
    boot = np.empty(n_boot, dtype=np.float64)
    scales = sorted(group['copy_count'].unique())
    for b in range(n_boot):
        sampled_values = []
        for scale in scales:
            scale_df = group[group['copy_count'] == scale]
            indices = rng.integers(0, len(scale_df), size=len(scale_df))
            sampled_values.extend(scale_df.iloc[indices][metric].to_numpy(float).tolist())
        boot[b] = float(np.mean(sampled_values))
    return (point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

def summarize_with_ci(sequence_df: pd.DataFrame, pooled_counts: Dict[str, int], pooled_p90: Dict[str, float], args) -> pd.DataFrame:
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    for stratum in ['Overall', 'Near', 'Middle', 'Far']:
        row = {'stratum': stratum, 'n_targets': int(pooled_counts.get(stratum, 0)), 'n_sequences': int(sequence_df[sequence_df['stratum'] == stratum][['copy_count', 'sample_name']].drop_duplicates().shape[0]), 'epe_p90_pooled_px': float(pooled_p90.get(stratum, np.nan))}
        for metric in ['gt_magnitude_px', 'raft_magnitude_px', 'epe_px', 'magnitude_abs_error_px', 'angular_error_deg']:
            point, low, high = hierarchical_bootstrap_ci(sequence_df, metric, stratum, args.bootstrap, rng)
            row[f'{metric}_mean'] = point
            row[f'{metric}_ci_low'] = low
            row[f'{metric}_ci_high'] = high
        rows.append(row)
    return pd.DataFrame(rows)

def write_latex_table(summary: pd.DataFrame, path: Path):

    def ci_text(row, prefix, decimals=2):
        return f"{row[prefix + '_mean']:.{decimals}f} [{row[prefix + '_ci_low']:.{decimals}f}, {row[prefix + '_ci_high']:.{decimals}f}]"
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Validation of the fixed RAFT apparent-motion prior against simulator-derived target displacement.}', '\\label{tab:raft_motion_prior_validation}', '\\scriptsize', '\\setlength{\\tabcolsep}{4.2pt}', '\\renewcommand{\\arraystretch}{1.12}', '\\begin{tabular}{lrrrrr}', '\\toprule', 'Group & Targets & Simulator magnitude (px) & EPE (px) & Magnitude error (px) & Angular error ($^{\\circ}$) \\\\', '\\midrule']
    for _, row in summary.iterrows():
        lines.append(f"{row['stratum']} & {int(row['n_targets']):,} & {ci_text(row, 'gt_magnitude_px')} & {ci_text(row, 'epe_px')} & {ci_text(row, 'magnitude_abs_error_px')} & {ci_text(row, 'angular_error_deg')}" + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\vspace{2pt}', '\\parbox{\\textwidth}{\\scriptsize Values are sequence-macro means with population-scale-stratified hierarchical-bootstrap 95\\% confidence intervals. RAFT is evaluated as an image-plane conditioning prior rather than as a physical orbital-motion estimator.}', '\\end{table*}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def save_figure(reservoir: Sequence[tuple], summary: pd.DataFrame, path: Path):
    sample = np.asarray(reservoir, dtype=float)
    fig, axes = plt.subplots(1, 2)
    if sample.size:
        gt_mag = sample[:, 0]
        raft_mag = sample[:, 1]
        hb = axes[0].hexbin(gt_mag, raft_mag, gridsize=55, mincnt=1, bins='log')
        lo = float(min(gt_mag.min(), raft_mag.min()))
        hi = float(max(gt_mag.max(), raft_mag.max()))
        axes[0].plot([lo, hi], [lo, hi])
        axes[0].set_xlim(lo, hi)
        axes[0].set_ylim(lo, hi)
        fig.colorbar(hb, ax=axes[0], label='log target count')
    axes[0].set_xlabel('Simulator displacement magnitude (px)')
    axes[0].set_ylabel('RAFT flow magnitude (px)')
    axes[0].set_title('(a) Target-supported magnitude agreement')
    order = ['Overall', 'Near', 'Middle', 'Far']
    plot_df = summary.set_index('stratum').loc[order].reset_index()
    x = np.arange(len(plot_df))
    y = plot_df['epe_px_mean'].to_numpy(float)
    lower = y - plot_df['epe_px_ci_low'].to_numpy(float)
    upper = plot_df['epe_px_ci_high'].to_numpy(float) - y
    axes[1].errorbar(x, y, yerr=np.vstack([lower, upper]), fmt='o')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel('Endpoint error (px)')
    axes[1].set_title('(b) Sequence-aware EPE by stratum')
    axes[1].grid(axis='y')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding='utf-8')
    device = torch.device(args.device)
    raft = load_raft(args, device)
    root = Path(args.dataset_root)
    accumulators: Dict[Tuple[int, str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
    pooled_epe: Dict[str, List[float]] = defaultdict(list)
    pooled_counts: Dict[str, int] = defaultdict(int)
    reservoir: List[tuple] = []
    reservoir_rng = random.Random(args.bootstrap_seed)
    seen_targets = 0
    pair_count = 0
    copy_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    for copy_dir in copy_dirs:
        copy_count = int(copy_dir.name)
        for sample_idx in range(args.sample_start, args.sample_end + 1):
            sample_dir = copy_dir / f'sample_{sample_idx:04d}'
            positions_path = sample_dir / 'frame_positions.csv'
            if not positions_path.exists():
                continue
            records = read_positions(positions_path)
            for frame in range(max(1, args.frame_start), args.frame_end + 1):
                prev_path = find_image(sample_dir, frame - 1, args.use_star_enhanced, args.fallback_to_raw)
                curr_path = find_image(sample_dir, frame, args.use_star_enhanced, args.fallback_to_raw)
                if prev_path is None or curr_path is None:
                    continue
                names = sorted({name for fr, name in records if fr == frame})
                pairs = []
                for name in names:
                    prev = records.get((frame - 1, name))
                    curr = records.get((frame, name))
                    if prev is None or curr is None or prev['visible'] != 1 or (curr['visible'] != 1):
                        continue
                    pairs.append((prev, curr))
                if not pairs:
                    continue
                image1 = load_raft_image(prev_path, device)
                image2 = load_raft_image(curr_path, device)
                padder = InputPadder(image1.shape)
                image1_pad, image2_pad = padder.pad(image1, image2)
                with torch.no_grad():
                    _, flow_pad = raft(image1_pad, image2_pad, iters=args.raft_iters, test_mode=True)
                flow = padder.unpad(flow_pad)
                xy = np.asarray([[prev['pixel_x'], prev['pixel_y']] for prev, _ in pairs], dtype=np.float32)
                raft_vectors = bilinear_sample_flow(flow, xy)
                for (prev, curr), raft_vec in zip(pairs, raft_vectors):
                    gt_vec = np.asarray([curr['pixel_x'] - prev['pixel_x'], curr['pixel_y'] - prev['pixel_y']], dtype=np.float64)
                    raft_vec = np.asarray(raft_vec, dtype=np.float64)
                    gt_mag = float(np.linalg.norm(gt_vec))
                    raft_mag = float(np.linalg.norm(raft_vec))
                    epe = float(np.linalg.norm(raft_vec - gt_vec))
                    mag_error = abs(raft_mag - gt_mag)
                    angle = angular_error_deg(raft_vec, gt_vec)
                    layer_name = STRATUM_NAMES.get(int(curr['layer']), f"Layer{int(curr['layer'])}")
                    values = dict(gt_magnitude_px=gt_mag, raft_magnitude_px=raft_mag, epe_px=epe, magnitude_abs_error_px=mag_error, angular_error_deg=angle)
                    for stratum in ['Overall', layer_name]:
                        accumulators[copy_count, sample_dir.name, stratum].add(**values)
                        pooled_epe[stratum].append(epe)
                        pooled_counts[stratum] += 1
                    seen_targets += 1
                    reservoir_add(reservoir, (gt_mag, raft_mag, epe, float(curr['layer'])), seen_targets, args.scatter_reservoir, reservoir_rng)
                pair_count += 1
                if pair_count % 100 == 0:
                    print(f'Processed frame pairs={pair_count}, target comparisons={seen_targets}', flush=True)
                if args.max_pairs is not None and pair_count >= args.max_pairs:
                    break
            if args.max_pairs is not None and pair_count >= args.max_pairs:
                break
        if args.max_pairs is not None and pair_count >= args.max_pairs:
            break
    sequence_rows = [acc.row(*key) for key, acc in sorted(accumulators.items())]
    sequence_df = pd.DataFrame(sequence_rows)
    sequence_df.to_csv(out_dir / 'raft_sequence_metrics.csv', index=False, encoding='utf-8-sig')
    pooled_p90 = {key: float(np.percentile(np.asarray(values, dtype=np.float32), 90)) for key, values in pooled_epe.items() if values}
    summary = summarize_with_ci(sequence_df, pooled_counts, pooled_p90, args)
    summary.to_csv(out_dir / 'raft_summary_with_ci.csv', index=False, encoding='utf-8-sig')
    write_latex_table(summary, out_dir / 'tab_raft_motion_prior_validation.tex')
    save_figure(reservoir, summary, out_dir / 'fig_raft_motion_prior_validation.pdf')
    manifest = {'frame_pairs': pair_count, 'target_comparisons': seen_targets, 'sequence_rows': int(len(sequence_df)), 'bootstrap_resamples': args.bootstrap, 'outputs': ['raft_sequence_metrics.csv', 'raft_summary_with_ci.csv', 'tab_raft_motion_prior_validation.tex', 'fig_raft_motion_prior_validation.pdf']}
    (out_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
