from __future__ import annotations
import argparse
import json
import math
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
STRATA = ('near', 'mid', 'far')
STRATUM_LABELS = {'near': 'Near', 'mid': 'Middle', 'far': 'Far'}
DEFAULT_EXPECTED_SEEDS = (12345, 23456, 34567)
REQUIRED_FRAME_COLUMNS = {'variant', 'seed', 'copy_count', 'sample_name', 'frame_index', 'gt_total_count', 'pred_total_count', 'abs_error_total', 'rel_error_total', 'inference_time_sec', 'total_density_rmse'}
for _s in STRATA:
    REQUIRED_FRAME_COLUMNS.update({f'gt_{_s}_count', f'pred_{_s}_count', f'abs_error_{_s}', f'density_rmse_{_s}', f'alloc_den_{_s}'})
    for _t in STRATA:
        REQUIRED_FRAME_COLUMNS.add(f'alloc_num_{_s}_to_{_t}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate concise Chapter 4 A0 tables and plotting CSV files.')
    parser.add_argument('--input', required=True, help='A0_full_socds directory or ZIP archive.')
    parser.add_argument('--output', required=True, help='Output directory.')
    parser.add_argument('--n_bootstrap', type=int, default=20000, help='Number of hierarchical-bootstrap resamples (default: 20000).')
    parser.add_argument('--bootstrap_seed', type=int, default=20260723, help='Random seed used only for bootstrap resampling.')
    parser.add_argument('--expected_seeds', default='12345,23456,34567', help='Comma-separated formal training seeds.')
    parser.add_argument('--strict', action='store_true', help='Fail when expected row/sequence/scale counts are not met.')
    return parser.parse_args()

def parse_seed_list(value: str) -> Tuple[int, ...]:
    seeds = tuple((int(x.strip()) for x in value.split(',') if x.strip()))
    if not seeds:
        raise ValueError('--expected_seeds must contain at least one seed.')
    return seeds

def fmt_mean_sd(mean: float, sd: float, decimals: int) -> str:
    if not np.isfinite(mean):
        return '--'
    if not np.isfinite(sd):
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}'

def fmt_ci(low: float, high: float, decimals: int) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return '--'
    return f'[{low:.{decimals}f}, {high:.{decimals}f}]'

def sample_sd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float('nan')
    return float(np.std(arr, ddof=1))

def percentile_ci(samples: np.ndarray, alpha: float=0.05) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    return (np.nanpercentile(samples, 100.0 * alpha / 2.0, axis=0), np.nanpercentile(samples, 100.0 * (1.0 - alpha / 2.0), axis=0))

def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float=1e-12) -> np.ndarray:
    return np.asarray(num, dtype=np.float64) / (np.asarray(den, dtype=np.float64) + eps)

def extract_input(input_path: Path, temp_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.is_file() and input_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(input_path, 'r') as archive:
            archive.extractall(temp_dir)
        return temp_dir
    raise FileNotFoundError(f'Input must be an existing directory or ZIP archive: {input_path}')

def discover_frame_files(root: Path) -> Dict[int, Path]:
    candidates = sorted(root.rglob('frame_results.csv'))
    found: Dict[int, Path] = {}
    for path in candidates:
        parts = list(path.parts)
        seed = None
        for part in reversed(parts):
            match = re.fullmatch('seed_(\\d+)', part)
            if match:
                seed = int(match.group(1))
                break
        if seed is None:
            try:
                probe = pd.read_csv(path, usecols=['seed'], nrows=5)
                unique = probe['seed'].dropna().astype(int).unique()
                if len(unique) == 1:
                    seed = int(unique[0])
            except Exception:
                seed = None
        if seed is None:
            continue
        if seed in found:
            raise RuntimeError(f'Multiple frame_results.csv files found for seed {seed}:\n  {found[seed]}\n  {path}')
        found[seed] = path
    return found

def validate_frame_df(df: pd.DataFrame, seed: int, source: Path) -> None:
    missing = sorted(REQUIRED_FRAME_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f'{source} is missing required columns:\n  ' + '\n  '.join(missing))
    file_seeds = sorted(df['seed'].dropna().astype(int).unique().tolist())
    if file_seeds != [seed]:
        raise ValueError(f'{source} seed column is inconsistent. Expected {seed}, found {file_seeds}.')
    if df.duplicated(['seed', 'copy_count', 'sample_name', 'frame_index']).any():
        dup = df.loc[df.duplicated(['seed', 'copy_count', 'sample_name', 'frame_index'], keep=False), ['seed', 'copy_count', 'sample_name', 'frame_index']].head()
        raise ValueError(f'Duplicate frame identifiers found in {source}:\n{dup}')

def load_frames(root: Path, expected_seeds: Sequence[int]) -> Tuple[pd.DataFrame, Dict[int, str]]:
    files = discover_frame_files(root)
    missing_seeds = [seed for seed in expected_seeds if seed not in files]
    if missing_seeds:
        raise FileNotFoundError('Missing frame_results.csv for seeds: ' + ', '.join((str(x) for x in missing_seeds)))
    frames: List[pd.DataFrame] = []
    provenance: Dict[int, str] = {}
    for seed in expected_seeds:
        path = files[seed]
        df = pd.read_csv(path)
        validate_frame_df(df, seed, path)
        df = df.copy()
        df['seed'] = df['seed'].astype(int)
        df['copy_count'] = df['copy_count'].astype(int)
        df['sample_name'] = df['sample_name'].astype(str)
        df['frame_index'] = df['frame_index'].astype(int)
        frames.append(df)
        provenance[seed] = str(path.resolve())
    combined = pd.concat(frames, ignore_index=True)
    combined['stratum_mae'] = combined[['abs_error_near', 'abs_error_mid', 'abs_error_far']].mean(axis=1)
    combined['rel_error_total_pct'] = 100.0 * combined['rel_error_total']
    combined['inference_time_ms'] = 1000.0 * combined['inference_time_sec']
    return (combined, provenance)

def sequence_aggregate(frame_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['variant', 'seed', 'copy_count', 'sample_name']
    scalar_cols = ['abs_error_total', 'rel_error_total', 'rel_error_total_pct', 'total_density_rmse', 'abs_error_near', 'abs_error_mid', 'abs_error_far', 'density_rmse_near', 'density_rmse_mid', 'density_rmse_far', 'stratum_mae', 'inference_time_ms']
    rows: List[Dict[str, float]] = []
    for keys, group in frame_df.groupby(group_cols, sort=True):
        row: Dict[str, float] = dict(zip(group_cols, keys))
        row['num_frames'] = int(len(group))
        for col in scalar_cols:
            row[col] = float(group[col].mean())
        num = np.zeros((3, 3), dtype=np.float64)
        den = np.zeros(3, dtype=np.float64)
        for i, source in enumerate(STRATA):
            den[i] = float(group[f'alloc_den_{source}'].sum())
            for j, target in enumerate(STRATA):
                num[i, j] = float(group[f'alloc_num_{source}_to_{target}'].sum())
        matrix = safe_ratio(num, den[:, None])
        for i, source in enumerate(STRATA):
            row[f'alloc_den_{source}'] = den[i]
            for j, target in enumerate(STRATA):
                row[f'alloc_num_{source}_to_{target}'] = num[i, j]
                row[f'alloc_seq_{source}_to_{target}'] = matrix[i, j]
        row['alloc_seq_diagonal_mean'] = float(np.trace(matrix) / 3.0)
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(['seed', 'copy_count', 'sample_name']).reset_index(drop=True)

def aggregate_matrix_from_sequences(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    num = np.zeros((3, 3), dtype=np.float64)
    den = np.zeros(3, dtype=np.float64)
    seq_matrices = np.zeros((len(group), 3, 3), dtype=np.float64)
    for r, (_, row) in enumerate(group.iterrows()):
        for i, source in enumerate(STRATA):
            den[i] += float(row[f'alloc_den_{source}'])
            for j, target in enumerate(STRATA):
                num[i, j] += float(row[f'alloc_num_{source}_to_{target}'])
                seq_matrices[r, i, j] = float(row[f'alloc_seq_{source}_to_{target}'])
    micro = safe_ratio(num, den[:, None])
    macro = np.mean(seq_matrices, axis=0)
    return (micro, macro)

def build_seed_group_metrics(sequence_df: pd.DataFrame, by_scale: bool) -> pd.DataFrame:
    group_cols = ['seed', 'copy_count'] if by_scale else ['seed']
    scalar_cols = ['abs_error_total', 'rel_error_total_pct', 'stratum_mae', 'total_density_rmse', 'abs_error_near', 'abs_error_mid', 'abs_error_far', 'density_rmse_near', 'density_rmse_mid', 'density_rmse_far', 'inference_time_ms']
    rows: List[Dict[str, float]] = []
    group_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in sequence_df.groupby(group_key, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, float] = dict(zip(group_cols, keys))
        row['num_sequences'] = int(len(group))
        for col in scalar_cols:
            row[col] = float(group[col].mean())
        micro, macro = aggregate_matrix_from_sequences(group)
        for i, source in enumerate(STRATA):
            for j, target in enumerate(STRATA):
                row[f'micro_{source}_to_{target}'] = float(micro[i, j])
                row[f'macro_{source}_to_{target}'] = float(macro[i, j])
        row['micro_diagonal_mean'] = float(np.trace(micro) / 3.0)
        row['macro_diagonal_mean'] = float(np.trace(macro) / 3.0)
        rows.append(row)
    sort_cols = ['seed', 'copy_count'] if by_scale else ['seed']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def check_integrity(frame_df: pd.DataFrame, sequence_df: pd.DataFrame, expected_seeds: Sequence[int], strict: bool) -> Dict:
    report: Dict[str, object] = {'seeds': sorted(frame_df['seed'].unique().astype(int).tolist()), 'population_scales': sorted(frame_df['copy_count'].unique().astype(int).tolist()), 'frames_per_seed': {str(int(k)): int(v) for k, v in frame_df.groupby('seed').size().to_dict().items()}, 'sequences_per_seed': {str(int(k)): int(v) for k, v in sequence_df.groupby('seed').size().to_dict().items()}, 'sequences_per_seed_scale': {f'{int(seed)}:{int(scale)}': int(n) for (seed, scale), n in sequence_df.groupby(['seed', 'copy_count']).size().to_dict().items()}, 'frames_per_sequence_min': int(sequence_df['num_frames'].min()), 'frames_per_sequence_max': int(sequence_df['num_frames'].max()), 'warnings': []}
    warnings: List[str] = []
    if tuple(report['seeds']) != tuple(expected_seeds):
        warnings.append(f"Observed seeds {report['seeds']} differ from expected {list(expected_seeds)}.")
    expected_scales = list(range(50, 501, 50))
    if report['population_scales'] != expected_scales:
        warnings.append(f"Observed population scales {report['population_scales']} differ from expected {expected_scales}.")
    for seed in expected_seeds:
        n_frames = report['frames_per_seed'].get(str(seed), 0)
        n_sequences = report['sequences_per_seed'].get(str(seed), 0)
        if n_frames != 4950:
            warnings.append(f'Seed {seed}: expected 4950 frames, observed {n_frames}.')
        if n_sequences != 50:
            warnings.append(f'Seed {seed}: expected 50 sequences, observed {n_sequences}.')
    if report['frames_per_sequence_min'] != 99 or report['frames_per_sequence_max'] != 99:
        warnings.append('Not every sequence contains exactly 99 evaluated two-frame samples.')
    report['warnings'] = warnings
    if strict and warnings:
        raise RuntimeError('Integrity validation failed:\n- ' + '\n- '.join(warnings))
    return report

def prepare_bootstrap_groups(sequence_df: pd.DataFrame) -> Dict[str, object]:
    seeds = np.asarray(sorted(sequence_df['seed'].unique()), dtype=int)
    scales = np.asarray(sorted(sequence_df['copy_count'].unique()), dtype=int)
    grouped: Dict[Tuple[int, int], pd.DataFrame] = {}
    sequence_counts = set()
    for seed in seeds:
        for scale in scales:
            group = sequence_df[(sequence_df['seed'] == seed) & (sequence_df['copy_count'] == scale)].sort_values('sample_name').reset_index(drop=True)
            if group.empty:
                raise ValueError(f'No sequence rows for seed={seed}, scale={scale}.')
            grouped[int(seed), int(scale)] = group
            sequence_counts.add(len(group))
    if len(sequence_counts) != 1:
        raise ValueError(f'Hierarchical bootstrap requires the same number of sequences for every seed-scale cell; observed counts={sorted(sequence_counts)}.')
    q = int(next(iter(sequence_counts)))
    s_count, c_count = (len(seeds), len(scales))
    scalar_columns = ['abs_error_total', 'rel_error_total_pct', 'stratum_mae', 'total_density_rmse', 'abs_error_near', 'abs_error_mid', 'abs_error_far', 'density_rmse_near', 'density_rmse_mid', 'density_rmse_far', 'inference_time_ms']
    metrics = {col: np.empty((s_count, c_count, q), dtype=np.float64) for col in scalar_columns}
    numerators = np.empty((s_count, c_count, q, 3, 3), dtype=np.float64)
    denominators = np.empty((s_count, c_count, q, 3), dtype=np.float64)
    matrices = np.empty((s_count, c_count, q, 3, 3), dtype=np.float64)
    for si, seed in enumerate(seeds):
        for ci, scale in enumerate(scales):
            group = grouped[int(seed), int(scale)]
            for col in scalar_columns:
                metrics[col][si, ci] = group[col].to_numpy(dtype=np.float64)
            for qi, (_, row) in enumerate(group.iterrows()):
                for i, source in enumerate(STRATA):
                    denominators[si, ci, qi, i] = float(row[f'alloc_den_{source}'])
                    for j, target in enumerate(STRATA):
                        numerators[si, ci, qi, i, j] = float(row[f'alloc_num_{source}_to_{target}'])
                        matrices[si, ci, qi, i, j] = float(row[f'alloc_seq_{source}_to_{target}'])
    return {'seeds': seeds, 'scales': scales, 'num_sequences': q, 'metrics': metrics, 'numerators': numerators, 'denominators': denominators, 'matrices': matrices}

def _scale_indices(groups: Dict[str, object], scales: Sequence[int]) -> np.ndarray:
    all_scales = np.asarray(groups['scales'], dtype=int)
    index_map = {int(scale): idx for idx, scale in enumerate(all_scales)}
    missing = [int(scale) for scale in scales if int(scale) not in index_map]
    if missing:
        raise KeyError(f'Bootstrap scales not found: {missing}')
    return np.asarray([index_map[int(scale)] for scale in scales], dtype=int)

def hierarchical_bootstrap_scalar(groups: Dict[str, object], metric: str, scales: Sequence[int], n_bootstrap: int, rng: np.random.Generator, chunk_size: int=1000) -> np.ndarray:
    values = np.asarray(groups['metrics'][metric], dtype=np.float64)
    scale_idx = _scale_indices(groups, scales)
    n_seeds = values.shape[0]
    n_sequences = int(groups['num_sequences'])
    out = np.empty(n_bootstrap, dtype=np.float64)
    cursor = 0
    while cursor < n_bootstrap:
        b = min(chunk_size, n_bootstrap - cursor)
        seed_draw = rng.integers(0, n_seeds, size=(b, n_seeds))
        seq_draw = rng.integers(0, n_sequences, size=(b, n_seeds, len(scale_idx), n_sequences))
        selected_seed_means = np.empty((b, n_seeds), dtype=np.float64)
        for draw_pos in range(n_seeds):
            actual_seed = seed_draw[:, draw_pos]
            selected_scale_means = np.empty((b, len(scale_idx)), dtype=np.float64)
            for local_ci, ci in enumerate(scale_idx):
                idx = seq_draw[:, draw_pos, local_ci, :]
                sampled = values[actual_seed[:, None], ci, idx]
                selected_scale_means[:, local_ci] = sampled.mean(axis=1)
            selected_seed_means[:, draw_pos] = selected_scale_means.mean(axis=1)
        out[cursor:cursor + b] = selected_seed_means.mean(axis=1)
        cursor += b
    return out

def hierarchical_bootstrap_allocation(groups: Dict[str, object], scales: Sequence[int], n_bootstrap: int, rng: np.random.Generator, chunk_size: int=500) -> Tuple[np.ndarray, np.ndarray]:
    numerators = np.asarray(groups['numerators'], dtype=np.float64)
    denominators = np.asarray(groups['denominators'], dtype=np.float64)
    matrices = np.asarray(groups['matrices'], dtype=np.float64)
    scale_idx = _scale_indices(groups, scales)
    n_seeds = numerators.shape[0]
    n_sequences = int(groups['num_sequences'])
    micro_samples = np.empty((n_bootstrap, 3, 3), dtype=np.float64)
    macro_samples = np.empty((n_bootstrap, 3, 3), dtype=np.float64)
    cursor = 0
    while cursor < n_bootstrap:
        b = min(chunk_size, n_bootstrap - cursor)
        seed_draw = rng.integers(0, n_seeds, size=(b, n_seeds))
        seq_draw = rng.integers(0, n_sequences, size=(b, n_seeds, len(scale_idx), n_sequences))
        seed_micro = np.empty((b, n_seeds, 3, 3), dtype=np.float64)
        seed_macro = np.empty((b, n_seeds, 3, 3), dtype=np.float64)
        for draw_pos in range(n_seeds):
            actual_seed = seed_draw[:, draw_pos]
            total_num = np.zeros((b, 3, 3), dtype=np.float64)
            total_den = np.zeros((b, 3), dtype=np.float64)
            total_matrix = np.zeros((b, 3, 3), dtype=np.float64)
            total_sequence_count = 0
            for local_ci, ci in enumerate(scale_idx):
                idx = seq_draw[:, draw_pos, local_ci, :]
                sampled_num = numerators[actual_seed[:, None], ci, idx]
                sampled_den = denominators[actual_seed[:, None], ci, idx]
                sampled_matrix = matrices[actual_seed[:, None], ci, idx]
                total_num += sampled_num.sum(axis=1)
                total_den += sampled_den.sum(axis=1)
                total_matrix += sampled_matrix.sum(axis=1)
                total_sequence_count += n_sequences
            seed_micro[:, draw_pos] = safe_ratio(total_num, total_den[:, :, None])
            seed_macro[:, draw_pos] = total_matrix / float(total_sequence_count)
        micro_samples[cursor:cursor + b] = seed_micro.mean(axis=1)
        macro_samples[cursor:cursor + b] = seed_macro.mean(axis=1)
        cursor += b
    return (micro_samples, macro_samples)

def summarize_seed_values(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (float(np.mean(arr)), sample_sd(arr))

def build_population_scale_outputs(sequence_df: pd.DataFrame, seed_scale_df: pd.DataFrame, groups: Dict[int, Dict[int, pd.DataFrame]], n_bootstrap: int, bootstrap_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scales = sorted(sequence_df['copy_count'].unique().astype(int).tolist())
    metrics = {'total_count_mae': 'abs_error_total', 'relative_count_error_pct': 'rel_error_total_pct', 'stratum_wise_mae': 'stratum_mae', 'total_density_rmse': 'total_density_rmse'}
    rows: List[Dict[str, float]] = []
    for scale_idx, scale in enumerate(scales):
        row: Dict[str, float] = {'population_scale': int(scale)}
        subset = seed_scale_df[seed_scale_df['copy_count'] == scale]
        for metric_idx, (output_name, col) in enumerate(metrics.items()):
            values = subset[col].to_numpy(dtype=np.float64)
            mean, sd = summarize_seed_values(values)
            rng = np.random.default_rng(bootstrap_seed + 10000 * scale_idx + 101 * metric_idx)
            samples = hierarchical_bootstrap_scalar(groups, metric=col, scales=[scale], n_bootstrap=n_bootstrap, rng=rng)
            low, high = percentile_ci(samples)
            row[f'{output_name}_mean'] = mean
            row[f'{output_name}_sd'] = sd
            row[f'{output_name}_ci_low'] = float(low)
            row[f'{output_name}_ci_high'] = float(high)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    points = sequence_df[['seed', 'copy_count', 'sample_name', 'abs_error_total', 'rel_error_total_pct', 'stratum_mae', 'total_density_rmse']].rename(columns={'copy_count': 'population_scale', 'abs_error_total': 'total_count_mae', 'rel_error_total_pct': 'relative_count_error_pct', 'stratum_mae': 'stratum_wise_mae'})
    return (summary_df, points)

def build_allocation_outputs(sequence_df: pd.DataFrame, seed_scale_df: pd.DataFrame, seed_overall_df: pd.DataFrame, groups: Dict[int, Dict[int, pd.DataFrame]], n_bootstrap: int, bootstrap_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scales = sorted(sequence_df['copy_count'].unique().astype(int).tolist())
    by_scale_rows: List[Dict[str, float]] = []
    for scale_idx, scale in enumerate(scales):
        rng = np.random.default_rng(bootstrap_seed + 500000 + scale_idx)
        micro_samples, macro_samples = hierarchical_bootstrap_allocation(groups, scales=[scale], n_bootstrap=n_bootstrap, rng=rng)
        micro_low, micro_high = percentile_ci(micro_samples)
        macro_low, macro_high = percentile_ci(macro_samples)
        seed_subset = seed_scale_df[seed_scale_df['copy_count'] == scale]
        for i, stratum in enumerate(STRATA):
            micro_vals = seed_subset[f'micro_{stratum}_to_{stratum}'].to_numpy(dtype=np.float64)
            macro_vals = seed_subset[f'macro_{stratum}_to_{stratum}'].to_numpy(dtype=np.float64)
            by_scale_rows.append({'population_scale': int(scale), 'stratum': STRATUM_LABELS[stratum], 'micro_diagonal_mean': 100.0 * float(np.mean(micro_vals)), 'micro_diagonal_sd': 100.0 * sample_sd(micro_vals), 'micro_diagonal_ci_low': 100.0 * float(micro_low[i, i]), 'micro_diagonal_ci_high': 100.0 * float(micro_high[i, i]), 'macro_diagonal_mean': 100.0 * float(np.mean(macro_vals)), 'macro_diagonal_sd': 100.0 * sample_sd(macro_vals), 'macro_diagonal_ci_low': 100.0 * float(macro_low[i, i]), 'macro_diagonal_ci_high': 100.0 * float(macro_high[i, i])})
    rng = np.random.default_rng(bootstrap_seed + 900000)
    micro_samples, macro_samples = hierarchical_bootstrap_allocation(groups, scales=scales, n_bootstrap=n_bootstrap, rng=rng)
    micro_low, micro_high = percentile_ci(micro_samples)
    macro_low, macro_high = percentile_ci(macro_samples)
    matrix_rows: List[Dict[str, float]] = []
    for aggregation, prefix, low, high in [('Micro', 'micro', micro_low, micro_high), ('Sequence-macro', 'macro', macro_low, macro_high)]:
        for i, source in enumerate(STRATA):
            for j, target in enumerate(STRATA):
                values = seed_overall_df[f'{prefix}_{source}_to_{target}'].to_numpy(dtype=np.float64)
                matrix_rows.append({'aggregation': aggregation, 'reference_stratum': STRATUM_LABELS[source], 'predicted_stratum': STRATUM_LABELS[target], 'mean_pct': 100.0 * float(np.mean(values)), 'sd_pct': 100.0 * sample_sd(values), 'ci_low_pct': 100.0 * float(low[i, j]), 'ci_high_pct': 100.0 * float(high[i, j])})
    stratum_rows: List[Dict[str, float]] = []
    metric_map = {'near': ('abs_error_near', 'density_rmse_near'), 'mid': ('abs_error_mid', 'density_rmse_mid'), 'far': ('abs_error_far', 'density_rmse_far')}
    for i, stratum in enumerate(STRATA):
        count_col, density_col = metric_map[stratum]
        count_vals = seed_overall_df[count_col].to_numpy(dtype=np.float64)
        density_vals = seed_overall_df[density_col].to_numpy(dtype=np.float64)
        micro_vals = seed_overall_df[f'micro_{stratum}_to_{stratum}'].to_numpy(dtype=np.float64)
        macro_vals = seed_overall_df[f'macro_{stratum}_to_{stratum}'].to_numpy(dtype=np.float64)
        stratum_rows.append({'stratum': STRATUM_LABELS[stratum], 'count_mae_mean': float(np.mean(count_vals)), 'count_mae_sd': sample_sd(count_vals), 'density_rmse_mean': float(np.mean(density_vals)), 'density_rmse_sd': sample_sd(density_vals), 'micro_diagonal_mean_pct': 100.0 * float(np.mean(micro_vals)), 'micro_diagonal_sd_pct': 100.0 * sample_sd(micro_vals), 'micro_diagonal_ci_low_pct': 100.0 * float(micro_low[i, i]), 'micro_diagonal_ci_high_pct': 100.0 * float(micro_high[i, i]), 'macro_diagonal_mean_pct': 100.0 * float(np.mean(macro_vals)), 'macro_diagonal_sd_pct': 100.0 * sample_sd(macro_vals), 'macro_diagonal_ci_low_pct': 100.0 * float(macro_low[i, i]), 'macro_diagonal_ci_high_pct': 100.0 * float(macro_high[i, i])})
    total_count_vals = seed_overall_df['abs_error_total'].to_numpy(dtype=np.float64)
    total_density_vals = seed_overall_df['total_density_rmse'].to_numpy(dtype=np.float64)
    micro_diag_vals = seed_overall_df['micro_diagonal_mean'].to_numpy(dtype=np.float64)
    macro_diag_vals = seed_overall_df['macro_diagonal_mean'].to_numpy(dtype=np.float64)
    micro_diag_samples = np.trace(micro_samples, axis1=1, axis2=2) / 3.0
    macro_diag_samples = np.trace(macro_samples, axis1=1, axis2=2) / 3.0
    micro_diag_low, micro_diag_high = percentile_ci(micro_diag_samples)
    macro_diag_low, macro_diag_high = percentile_ci(macro_diag_samples)
    stratum_rows.append({'stratum': 'Overall', 'count_mae_mean': float(np.mean(total_count_vals)), 'count_mae_sd': sample_sd(total_count_vals), 'density_rmse_mean': float(np.mean(total_density_vals)), 'density_rmse_sd': sample_sd(total_density_vals), 'micro_diagonal_mean_pct': 100.0 * float(np.mean(micro_diag_vals)), 'micro_diagonal_sd_pct': 100.0 * sample_sd(micro_diag_vals), 'micro_diagonal_ci_low_pct': 100.0 * float(micro_diag_low), 'micro_diagonal_ci_high_pct': 100.0 * float(micro_diag_high), 'macro_diagonal_mean_pct': 100.0 * float(np.mean(macro_diag_vals)), 'macro_diagonal_sd_pct': 100.0 * sample_sd(macro_diag_vals), 'macro_diagonal_ci_low_pct': 100.0 * float(macro_diag_low), 'macro_diagonal_ci_high_pct': 100.0 * float(macro_diag_high)})
    return (pd.DataFrame(by_scale_rows), pd.DataFrame(matrix_rows), pd.DataFrame(stratum_rows))

def write_population_scale_tex(df: pd.DataFrame, path: Path) -> None:
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Population-scale performance of A0 over three independent training seeds.}', '\\label{tab:a0_population_scale_performance}', '\\scriptsize', '\\setlength{\\tabcolsep}{4.5pt}', '\\renewcommand{\\arraystretch}{1.12}', '\\begin{tabular}{ccccc}', '\\toprule', 'Population scale', '& Total-count MAE $\\downarrow$', '& Relative error (\\%) $\\downarrow$', '& Stratum-wise MAE $\\downarrow$', '& Density RMSE $\\downarrow$ \\\\', '\\midrule']
    for _, row in df.iterrows():
        lines.append(f"{int(row['population_scale'])} & {fmt_mean_sd(row['total_count_mae_mean'], row['total_count_mae_sd'], 3)} & {fmt_mean_sd(row['relative_count_error_pct_mean'], row['relative_count_error_pct_sd'], 3)} & {fmt_mean_sd(row['stratum_wise_mae_mean'], row['stratum_wise_mae_sd'], 3)} & {fmt_mean_sd(row['total_density_rmse_mean'], row['total_density_rmse_sd'], 5)} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\vspace{2pt}', '\\parbox{\\textwidth}{\\scriptsize Values are the mean $\\pm$ sample standard deviation over three independently trained seeds.}', '\\end{table*}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')

def write_stratum_summary_tex(df: pd.DataFrame, path: Path, n_bootstrap: int) -> None:
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Distance-stratum estimation and allocation fidelity of A0 over three independent training seeds.}', '\\label{tab:a0_stratum_allocation_summary}', '\\scriptsize', '\\setlength{\\tabcolsep}{3.2pt}', '\\renewcommand{\\arraystretch}{1.12}', '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{lcccccc}', '\\toprule', 'Stratum', '& Count MAE $\\downarrow$', '& Density RMSE $\\downarrow$', '& Micro diagonal (\\%) $\\uparrow$', '& Micro 95\\% CI', '& Macro diagonal (\\%) $\\uparrow$', '& Macro 95\\% CI \\\\', '\\midrule']
    for _, row in df.iterrows():
        lines.append(f"{row['stratum']} & {fmt_mean_sd(row['count_mae_mean'], row['count_mae_sd'], 3)} & {fmt_mean_sd(row['density_rmse_mean'], row['density_rmse_sd'], 5)} & {fmt_mean_sd(row['micro_diagonal_mean_pct'], row['micro_diagonal_sd_pct'], 2)} & {fmt_ci(row['micro_diagonal_ci_low_pct'], row['micro_diagonal_ci_high_pct'], 2)} & {fmt_mean_sd(row['macro_diagonal_mean_pct'], row['macro_diagonal_sd_pct'], 2)} & {fmt_ci(row['macro_diagonal_ci_low_pct'], row['macro_diagonal_ci_high_pct'], 2)} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}%', '}', '\\vspace{2pt}', f'\\parbox{{\\textwidth}}{{\\scriptsize Mean $\\pm$ sample standard deviation is computed over independently trained seeds. Confidence intervals use {n_bootstrap:,} population-scale-stratified hierarchical-bootstrap resamples. The overall allocation score is the mean of the three diagonal entries.}}', '\\end{table*}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')

def write_allocation_matrix_tex(df: pd.DataFrame, path: Path) -> None:
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Micro- and sequence-macro soft allocation matrices with hierarchical-bootstrap 95\\% confidence intervals.}', '\\label{tab:a0_allocation_matrices_ci}', '\\scriptsize', '\\setlength{\\tabcolsep}{5pt}', '\\renewcommand{\\arraystretch}{1.15}', '\\begin{tabular}{llccc}', '\\toprule', 'Aggregation & Reference stratum', '& Pred. Near (\\%)', '& Pred. Middle (\\%)', '& Pred. Far (\\%) \\\\', '\\midrule']
    for aggregation in ('Micro', 'Sequence-macro'):
        subset = df[df['aggregation'] == aggregation]
        for row_idx, reference in enumerate(('Near', 'Middle', 'Far')):
            cells = []
            for predicted in ('Near', 'Middle', 'Far'):
                row = subset[(subset['reference_stratum'] == reference) & (subset['predicted_stratum'] == predicted)].iloc[0]
                cells.append(f"{row['mean_pct']:.2f} {fmt_ci(row['ci_low_pct'], row['ci_high_pct'], 2)}")
            aggregation_cell = aggregation if row_idx == 0 else ''
            lines.append(f'{aggregation_cell} & {reference} & {cells[0]} & {cells[1]} & {cells[2]} \\\\')
        if aggregation == 'Micro':
            lines.append('\\addlinespace[3pt]')
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\vspace{2pt}', '\\parbox{\\textwidth}{\\scriptsize Each cell reports the mean across independently trained seeds followed by its population-scale-stratified hierarchical-bootstrap 95\\% confidence interval. Rows are reference strata and columns are predicted strata.}', '\\end{table*}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')

def latex_command(name: str, value: str) -> str:
    return f'\\newcommand{{\\{name}}}{{{value}}}'

def write_macros(seed_overall_df: pd.DataFrame, stratum_df: pd.DataFrame, path: Path) -> None:
    total_count_mean, total_count_sd = summarize_seed_values(seed_overall_df['abs_error_total'])
    stratum_mean, stratum_sd = summarize_seed_values(seed_overall_df['stratum_mae'])
    density_mean, density_sd = summarize_seed_values(seed_overall_df['total_density_rmse'])
    micro_mean, micro_sd = summarize_seed_values(100.0 * seed_overall_df['micro_diagonal_mean'])
    macro_mean, macro_sd = summarize_seed_values(100.0 * seed_overall_df['macro_diagonal_mean'])
    rel_mean, rel_sd = summarize_seed_values(seed_overall_df['rel_error_total_pct'])
    inf_mean, inf_sd = summarize_seed_values(seed_overall_df['inference_time_ms'])
    lines = ['% Automatically generated. Do not edit manually.', latex_command('AZeroTotalCountMAE', f'{total_count_mean:.3f}'), latex_command('AZeroTotalCountMAESD', f'{total_count_sd:.3f}'), latex_command('AZeroRelativeCountError', f'{rel_mean:.3f}\\%'), latex_command('AZeroRelativeCountErrorSD', f'{rel_sd:.3f}\\%'), latex_command('AZeroStratumMAE', f'{stratum_mean:.3f}'), latex_command('AZeroStratumMAESD', f'{stratum_sd:.3f}'), latex_command('AZeroDensityRMSE', f'{density_mean:.5f}'), latex_command('AZeroDensityRMSESD', f'{density_sd:.5f}'), latex_command('AZeroMicroDiagonal', f'{micro_mean:.2f}\\%'), latex_command('AZeroMicroDiagonalSD', f'{micro_sd:.2f}\\%'), latex_command('AZeroMacroDiagonal', f'{macro_mean:.2f}\\%'), latex_command('AZeroMacroDiagonalSD', f'{macro_sd:.2f}\\%'), latex_command('AZeroInferenceTime', f'{inf_mean:.2f}'), latex_command('AZeroInferenceTimeSD', f'{inf_sd:.2f}')]
    for _, row in stratum_df[stratum_df['stratum'] != 'Overall'].iterrows():
        prefix = row['stratum'].replace('Middle', 'Mid')
        lines.extend([latex_command(f'AZero{prefix}CountMAE', f"{row['count_mae_mean']:.3f}"), latex_command(f'AZero{prefix}DensityRMSE', f"{row['density_rmse_mean']:.5f}"), latex_command(f'AZero{prefix}MicroDiagonal', f"{row['micro_diagonal_mean_pct']:.2f}\\%"), latex_command(f'AZero{prefix}MacroDiagonal', f"{row['macro_diagonal_mean_pct']:.2f}\\%")])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def write_run_summary(output_dir: Path, provenance: Mapping[int, str], integrity: Mapping, n_bootstrap: int, bootstrap_seed: int, seed_overall_df: pd.DataFrame) -> None:
    payload = {'input_frame_results': {str(k): v for k, v in provenance.items()}, 'n_bootstrap': int(n_bootstrap), 'bootstrap_seed': int(bootstrap_seed), 'integrity': integrity, 'seed_overall_metrics': seed_overall_df.to_dict(orient='records')}
    (output_dir / 'run_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    expected_seeds = parse_seed_list(args.expected_seeds)
    if args.n_bootstrap < 1000:
        raise ValueError('--n_bootstrap should be at least 1000.')
    if output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True)
    canonical_dir = output_dir / 'canonical'
    csv_dir = output_dir / 'csv'
    tex_dir = output_dir / 'tex'
    for directory in (canonical_dir, csv_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='socds_a0_') as temp:
        root = extract_input(input_path, Path(temp))
        frame_df, provenance = load_frames(root, expected_seeds)
    sequence_df = sequence_aggregate(frame_df)
    seed_scale_df = build_seed_group_metrics(sequence_df, by_scale=True)
    seed_overall_df = build_seed_group_metrics(sequence_df, by_scale=False)
    integrity = check_integrity(frame_df, sequence_df, expected_seeds=expected_seeds, strict=args.strict)
    groups = prepare_bootstrap_groups(sequence_df)
    population_scale_df, sequence_points_df = build_population_scale_outputs(sequence_df, seed_scale_df, groups, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    allocation_by_scale_df, allocation_cells_df, stratum_summary_df = build_allocation_outputs(sequence_df, seed_scale_df, seed_overall_df, groups, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    frame_df.to_csv(canonical_dir / 'official_frame_metrics.csv', index=False, encoding='utf-8-sig')
    sequence_df.to_csv(canonical_dir / 'official_sequence_metrics.csv', index=False, encoding='utf-8-sig')
    seed_scale_df.to_csv(canonical_dir / 'official_seed_scale_metrics.csv', index=False, encoding='utf-8-sig')
    seed_overall_df.to_csv(canonical_dir / 'official_seed_overall_metrics.csv', index=False, encoding='utf-8-sig')
    population_scale_df.to_csv(csv_dir / 'population_scale_performance.csv', index=False, encoding='utf-8-sig')
    sequence_points_df.to_csv(csv_dir / 'population_scale_sequence_points.csv', index=False, encoding='utf-8-sig')
    allocation_by_scale_df.to_csv(csv_dir / 'allocation_by_scale.csv', index=False, encoding='utf-8-sig')
    allocation_cells_df.to_csv(csv_dir / 'allocation_matrix_cells.csv', index=False, encoding='utf-8-sig')
    stratum_summary_df.to_csv(csv_dir / 'stratum_summary.csv', index=False, encoding='utf-8-sig')
    seed_scale_df.to_csv(csv_dir / 'seed_scale_metrics.csv', index=False, encoding='utf-8-sig')
    seed_overall_df.to_csv(csv_dir / 'seed_overall_metrics.csv', index=False, encoding='utf-8-sig')
    write_population_scale_tex(population_scale_df, tex_dir / 'population_scale_performance.tex')
    write_stratum_summary_tex(stratum_summary_df, tex_dir / 'stratum_allocation_summary.tex', args.n_bootstrap)
    write_allocation_matrix_tex(allocation_cells_df, tex_dir / 'allocation_matrices_with_ci_supp.tex')
    write_macros(seed_overall_df, stratum_summary_df, tex_dir / 'chapter4_macros.tex')
    (output_dir / 'data_integrity_report.json').write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding='utf-8')
    write_run_summary(output_dir, provenance, integrity, args.n_bootstrap, args.bootstrap_seed, seed_overall_df)
    print('=' * 72)
    print('SOC-DS Chapter 4 A0 result generation completed.')
    print(f'Input : {input_path}')
    print(f'Output: {output_dir}')
    print(f'Seeds : {list(expected_seeds)}')
    print(f'Frames: {len(frame_df):,}')
    print(f'Sequences: {len(sequence_df):,}')
    print(f'Bootstrap resamples: {args.n_bootstrap:,}')
    if integrity['warnings']:
        print('Warnings:')
        for warning in integrity['warnings']:
            print(f'  - {warning}')
    else:
        print('Integrity checks: passed')
    print('Generated LaTeX files:')
    for path in sorted(tex_dir.glob('*.tex')):
        print(f'  - {path}')
    print('Generated plotting CSV files:')
    for path in sorted(csv_dir.glob('*.csv')):
        print(f'  - {path}')
    print('=' * 72)
if __name__ == '__main__':
    main()
