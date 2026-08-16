from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
SEEDS = [12345, 23456, 34567]
DISPLAY = {'A0_full_socds': 'A0 Full SOC-DS', 'A1_single_frame_stratified': 'A1 Single frame', 'A2_no_depth': 'A2 No depth', 'A3_no_flow': 'A3 No flow', 'A4_direct_stratified_head': 'A4 Direct stratified', 'A5_single_projected_density': 'A5 Single density', 'A6_direct_count_regression': 'A6 Count regression', 'B1_no_total_loss': 'B1 No total loss', 'B2_no_consistency_loss': 'B2 No consistency loss', 'B3_no_depth_loss': 'B3 No depth loss', 'B4_no_layer_loss': 'B4 No layer loss'}
BASELINE_ORDER = ['A0_full_socds', 'A1_single_frame_stratified', 'A2_no_depth', 'A3_no_flow', 'A4_direct_stratified_head', 'A5_single_projected_density', 'A6_direct_count_regression']
ABLATION_ORDER = ['A0_full_socds', 'B1_no_total_loss', 'B2_no_consistency_loss', 'B3_no_depth_loss', 'B4_no_layer_loss']
METRICS = [('mean_total_count_mae', 'Total MAE', 'min', 1.0), ('mean_total_relative_error', 'Relative error (\\%)', 'min', 100.0), ('mean_stratum_mae', 'Stratum MAE', 'min', 1.0), ('mean_total_density_rmse', 'Density RMSE', 'min', 1.0), ('allocation_micro_diagonal_mean', 'Micro allocation (\\%)', 'max', 100.0), ('allocation_sequence_macro_diagonal_mean', 'Macro allocation (\\%)', 'max', 100.0), ('mean_inference_time_sec', 'Inference (ms)', 'min', 1000.0)]

def parse_args():
    p = argparse.ArgumentParser('Aggregate SOC-DS multi-seed experiment results')
    p.add_argument('--mode', choices=['baseline', 'ablation'], required=True)
    p.add_argument('--results_root', required=True)
    p.add_argument('--reference_root', default=None)
    p.add_argument('--checkpoint_sensitivity_root', default=None)
    p.add_argument('--reference_checkpoint_sensitivity_root', default=None)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--bootstrap', type=int, default=20000)
    p.add_argument('--bootstrap_seed', type=int, default=20260718)
    return p.parse_args()

def discover(root: Path, protocol: str) -> List[Tuple[Path, Path]]:
    pairs = []
    if not root.exists():
        return pairs
    for summary_path in root.rglob('summary.json'):
        seq_path = summary_path.parent / 'sequence_results.csv'
        if not seq_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('evaluation_protocol', 'official_independent_seed_best') != protocol:
            continue
        pairs.append((summary_path, seq_path))
    return sorted(pairs)

def load_results(pairs):
    summaries, sequences = ([], [])
    for summary_path, seq_path in tqdm(pairs, desc='Loading result directories', unit='run'):
        s = json.loads(summary_path.read_text(encoding='utf-8'))
        s['result_dir'] = str(summary_path.parent)
        summaries.append(s)
        q = pd.read_csv(seq_path)
        q['run_label'] = s.get('run_label', s['variant'])
        q['variant'] = s['variant']
        q['seed'] = int(s['seed'])
        q['evaluation_label'] = s.get('evaluation_label', 'best')
        q['evaluation_protocol'] = s.get('evaluation_protocol', 'official_independent_seed_best')
        sequences.append(q)
    return (pd.DataFrame(summaries), pd.concat(sequences, ignore_index=True) if sequences else pd.DataFrame())

def validate_independent_seeds(summary: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for label, g in summary.groupby('run_label'):
        seeds = sorted((int(x) for x in g['seed'].unique()))
        protocols = sorted(g['evaluation_protocol'].astype(str).unique())
        labels = sorted(g['evaluation_label'].astype(str).unique())
        valid = seeds == SEEDS and protocols == ['official_independent_seed_best'] and (labels == ['best'])
        rows.append({'run_label': label, 'seeds': ','.join(map(str, seeds)), 'n_seeds': len(seeds), 'evaluation_protocols': ','.join(protocols), 'evaluation_labels': ','.join(labels), 'valid_three_independent_seeds': valid})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / 'independent_seed_audit.csv', index=False, encoding='utf-8-sig')
    return audit

def scalar_seed_stats(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in summary.groupby('run_label', sort=False):
        row = {'run_label': label, 'display_name': DISPLAY.get(label, label), 'variant': g['variant'].iloc[0], 'family': g['family'].iloc[0], 'n_seeds': g['seed'].nunique(), 'seeds': ','.join((str(int(x)) for x in sorted(g['seed'].unique()))), 'trainable_parameters': float(pd.to_numeric(g['trainable_parameters']).mean())}
        for key, _, _, scale in METRICS:
            if key in g:
                values = pd.to_numeric(g[key], errors='coerce').dropna().to_numpy(float) * scale
                if values.size:
                    row[f'{key}_mean'] = float(values.mean())
                    row[f'{key}_sd'] = float(values.std(ddof=1)) if values.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def ordered(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(order)}
    out = df.copy()
    out['_rank'] = out['run_label'].map(rank).fillna(999)
    return out.sort_values('_rank').drop(columns='_rank').reset_index(drop=True)

def latex_escape(value: str) -> str:
    replacements = {'&': '\\&', '%': '\\%', '_': '\\_', '#': '\\#', '$': '\\$', '{': '\\{', '}': '\\}'}
    return ''.join((replacements.get(ch, ch) for ch in str(value)))

def fmt_mean_sd(mean, sd, digits=3):
    if pd.isna(mean):
        return '--'
    if pd.isna(sd):
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'

def best_labels(stats: pd.DataFrame) -> Dict[str, str]:
    best = {}
    for key, _, direction, _ in METRICS:
        col = f'{key}_mean'
        if col not in stats:
            continue
        valid = stats[['run_label', col]].dropna()
        if valid.empty:
            continue
        idx = valid[col].idxmin() if direction == 'min' else valid[col].idxmax()
        best[key] = stats.loc[idx, 'run_label']
    return best

def write_main_latex(stats: pd.DataFrame, path: Path, caption: str, label: str):
    best = best_labels(stats)
    cols = [('mean_total_count_mae', 'Total MAE', 3), ('mean_total_relative_error', 'Rel. error (\\%)', 3), ('mean_stratum_mae', 'Stratum MAE', 3), ('mean_total_density_rmse', 'Density RMSE', 5), ('allocation_micro_diagonal_mean', 'Micro alloc. (\\%)', 2), ('allocation_sequence_macro_diagonal_mean', 'Macro alloc. (\\%)', 2), ('mean_inference_time_sec', 'Time (ms)', 2)]
    lines = ['\\begin{table*}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\setlength{\\tabcolsep}{4.2pt}', '\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule', 'Method & ' + ' & '.join((x[1] for x in cols)) + ' \\\\', '\\midrule']
    for _, row in stats.iterrows():
        cells = [latex_escape(row['display_name'])]
        for key, _, digits in cols:
            value = fmt_mean_sd(row.get(f'{key}_mean'), row.get(f'{key}_sd'), digits)
            if best.get(key) == row['run_label'] and value != '--':
                value = '\\textbf{' + value + '}'
            cells.append(value)
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')

def add_sequence_allocation_diag(seq: pd.DataFrame) -> pd.DataFrame:
    cols = ['alloc_micro_near_to_near', 'alloc_micro_mid_to_mid', 'alloc_micro_far_to_far']
    if all((c in seq.columns for c in cols)):
        seq = seq.copy()
        seq['allocation_diag_mean'] = seq[cols].mean(axis=1)
    return seq

def hierarchical_bootstrap(merged: pd.DataFrame, col: str, n_boot: int, rng) -> Tuple[float, float]:
    seeds = sorted(merged['seed'].unique())
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        values = []
        for seed in sampled_seeds:
            seed_df = merged[merged['seed'] == seed]
            for _, scale_df in seed_df.groupby('copy_count'):
                idx = rng.integers(0, len(scale_df), len(scale_df))
                values.extend(scale_df.iloc[idx][col].to_numpy(float))
        boot[b] = np.mean(values)
    return (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))

def paired_comparisons(seq: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    seq = add_sequence_allocation_diag(seq)
    ref = seq[seq['run_label'] == 'A0_full_socds']
    metrics = [('abs_error_total', 'error'), ('rel_error_total', 'error'), ('abs_error_near', 'error'), ('abs_error_mid', 'error'), ('abs_error_far', 'error'), ('total_density_rmse', 'error'), ('allocation_diag_mean', 'allocation')]
    rows = []
    labels = [x for x in seq['run_label'].unique() if x != 'A0_full_socds']
    total_work = sum((1 for label in labels for metric, _ in metrics if metric in ref.columns and metric in seq[seq['run_label'] == label].columns))
    progress = tqdm(total=total_work, desc='Paired hierarchical bootstrap', unit='comparison')
    for label in labels:
        other = seq[seq['run_label'] == label]
        for metric, kind in metrics:
            if metric not in ref.columns or metric not in other.columns:
                continue
            keys = ['seed', 'copy_count', 'sample_name']
            merged = ref[keys + [metric]].rename(columns={metric: 'A0'}).merge(other[keys + [metric]].rename(columns={metric: 'other'}), on=keys, how='inner').dropna()
            if merged.empty:
                progress.update(1)
                continue
            merged['difference'] = merged['other'] - merged['A0'] if kind == 'error' else merged['A0'] - merged['other']
            diff = merged['difference'].to_numpy(float)
            lo, hi = hierarchical_bootstrap(merged, 'difference', n_boot, rng)
            sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
            rows.append({'comparison': f'A0_full_socds vs {label}', 'alternative': label, 'metric': metric, 'n_seed_sequences': len(merged), 'mean_difference_positive_favors_A0': diff.mean(), 'difference_sd': sd, 'effect_size_dz': diff.mean() / sd if np.isfinite(sd) and sd > 0 else np.nan, 'ci_low': lo, 'ci_high': hi})
            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)

def write_paired_latex(df: pd.DataFrame, path: Path):
    if df.empty:
        path.write_text('% No paired comparisons available.\n', encoding='utf-8')
        return
    subset = df[df['metric'].isin(['abs_error_total', 'rel_error_total', 'total_density_rmse', 'allocation_diag_mean'])]
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Paired sequence-level comparisons against the full SOC-DS model. Positive differences favor A0.}', '\\label{tab:paired_comparisons_a0}', '\\begin{tabular}{llrrrr}', '\\toprule', 'Alternative & Metric & Mean difference & 95\\% CI low & 95\\% CI high & $d_z$ \\\\', '\\midrule']
    for _, row in subset.iterrows():
        lines.append(f"{latex_escape(DISPLAY.get(row['alternative'], row['alternative']))} & {latex_escape(row['metric'])} & {row['mean_difference_positive_favors_A0']:.4f} & {row['ci_low']:.4f} & {row['ci_high']:.4f} & {row['effect_size_dz']:.3f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')

def per_scale(seq: pd.DataFrame) -> pd.DataFrame:
    seq = add_sequence_allocation_diag(seq)
    metric_cols = [c for c in ['abs_error_total', 'rel_error_total', 'abs_error_near', 'abs_error_mid', 'abs_error_far', 'total_density_rmse', 'allocation_diag_mean'] if c in seq.columns]
    seed_scale = seq.groupby(['run_label', 'seed', 'copy_count'], as_index=False)[metric_cols].mean()
    if all((c in seed_scale for c in ['abs_error_near', 'abs_error_mid', 'abs_error_far'])):
        seed_scale['stratum_mae'] = seed_scale[['abs_error_near', 'abs_error_mid', 'abs_error_far']].mean(axis=1)
    rows = []
    for keys, g in seed_scale.groupby(['run_label', 'copy_count']):
        row = {'run_label': keys[0], 'copy_count': keys[1]}
        for col in [x for x in seed_scale.columns if x not in {'run_label', 'seed', 'copy_count'}]:
            vals = pd.to_numeric(g[col], errors='coerce').dropna().to_numpy(float)
            if vals.size:
                row[f'{col}_mean'] = vals.mean()
                row[f'{col}_sd'] = vals.std(ddof=1) if vals.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def plot_bars(stats: pd.DataFrame, out: Path):
    specs = [('mean_total_count_mae', 'Total-count MAE', 'total_count_mae'), ('mean_total_relative_error', 'Mean relative count error (%)', 'relative_count_error'), ('mean_stratum_mae', 'Mean stratum-wise MAE', 'stratum_mae'), ('mean_total_density_rmse', 'Total-density RMSE', 'total_density_rmse'), ('allocation_micro_diagonal_mean', 'Micro allocation diagonal (%)', 'allocation_micro'), ('allocation_sequence_macro_diagonal_mean', 'Sequence-macro allocation diagonal (%)', 'allocation_macro'), ('mean_inference_time_sec', 'Mean inference time (ms)', 'inference_time')]
    for key, ylabel, filename in tqdm(specs, desc='Drawing model comparison figures', unit='figure'):
        mean_col, sd_col = (f'{key}_mean', f'{key}_sd')
        if mean_col not in stats or stats[mean_col].notna().sum() == 0:
            continue
        data = stats.dropna(subset=[mean_col])
        fig, ax = plt.subplots()
        x = np.arange(len(data))
        yerr = data[sd_col].fillna(0).to_numpy() if sd_col in data else None
        ax.bar(x, data[mean_col], yerr=yerr)
        ax.set_xticks(x)
        ax.set_xticklabels(data['display_name'], rotation=25, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y')
        fig.tight_layout()
        fig.savefig(out / f'{filename}.png', bbox_inches='tight')
        fig.savefig(out / f'{filename}.pdf', bbox_inches='tight')
        plt.close(fig)

def plot_per_scale(scale_df: pd.DataFrame, out: Path):
    for metric, ylabel, filename in [('abs_error_total_mean', 'Sequence-level total-count MAE', 'per_scale_total_mae'), ('stratum_mae_mean', 'Sequence-level stratum MAE', 'per_scale_stratum_mae')]:
        if metric not in scale_df:
            continue
        fig, ax = plt.subplots()
        for label, g in scale_df.groupby('run_label', sort=False):
            g = g.sort_values('copy_count')
            ax.plot(g['copy_count'], g[metric], marker='o', label=DISPLAY.get(label, label))
        ax.set_xlabel('Nominal object population')
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{filename}.png', bbox_inches='tight')
        fig.savefig(out / f'{filename}.pdf', bbox_inches='tight')
        plt.close(fig)

def plot_paired(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    for metric, filename, xlabel in [('abs_error_total', 'paired_total_mae_difference', 'Alternative minus A0 total MAE'), ('total_density_rmse', 'paired_density_rmse_difference', 'Alternative minus A0 density RMSE'), ('allocation_diag_mean', 'paired_allocation_difference', 'A0 minus alternative allocation diagonal')]:
        d = df[df['metric'] == metric].copy()
        if d.empty:
            continue
        d['name'] = d['alternative'].map(lambda x: DISPLAY.get(x, x))
        y = np.arange(len(d))
        center = d['mean_difference_positive_favors_A0'].to_numpy()
        lo = center - d['ci_low'].to_numpy()
        hi = d['ci_high'].to_numpy() - center
        fig, ax = plt.subplots()
        ax.errorbar(center, y, xerr=np.vstack([lo, hi]), fmt='o')
        ax.axvline(0)
        ax.set_yticks(y)
        ax.set_yticklabels(d['name'])
        ax.set_xlabel(xlabel + ' (positive favors A0)')
        ax.grid(axis='x')
        fig.tight_layout()
        fig.savefig(out / f'{filename}.png', bbox_inches='tight')
        fig.savefig(out / f'{filename}.pdf', bbox_inches='tight')
        plt.close(fig)

def checkpoint_sensitivity(pairs, out: Path):
    if not pairs:
        return
    summary, _ = load_results(pairs)
    rows = []
    for label, g in summary.groupby('run_label'):
        for _, r in g.iterrows():
            row = {'run_label': label, 'checkpoint_state': r['evaluation_label'], 'seed': int(r['seed'])}
            for key, _, _, scale in METRICS:
                if key in r and pd.notna(r[key]):
                    row[key] = float(r[key]) * scale
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'checkpoint_sensitivity_results_not_random_seeds.csv', index=False, encoding='utf-8-sig')
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Late-checkpoint sensitivity within seed 12345. These checkpoints are correlated states from one optimization trajectory and are not treated as independent random-seed repetitions.}', '\\label{tab:checkpoint_sensitivity}', '\\begin{tabular}{llrrrr}', '\\toprule', 'Method & Checkpoint & Total MAE & Rel. error (\\%) & Stratum MAE & Density RMSE \\\\', '\\midrule']
    for _, row in df.iterrows():
        lines.append(f"{latex_escape(DISPLAY.get(row['run_label'], row['run_label']))} & {latex_escape(row['checkpoint_state'])} & {row.get('mean_total_count_mae', np.nan):.3f} & {row.get('mean_total_relative_error', np.nan):.3f} & {row.get('mean_stratum_mae', np.nan):.3f} & {row.get('mean_total_density_rmse', np.nan):.5f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    (out / 'checkpoint_sensitivity_table.tex').write_text('\n'.join(lines), encoding='utf-8')

def write_interpretation(stats: pd.DataFrame, paired: pd.DataFrame, out: Path):
    lines = ['# Automatically generated factual result summary', '', 'This file reports what the completed experiments show; it does not force a predetermined ranking.', '']
    for key, title, direction, _ in METRICS:
        col = f'{key}_mean'
        if col not in stats or stats[col].notna().sum() == 0:
            continue
        valid = stats.dropna(subset=[col])
        idx = valid[col].idxmin() if direction == 'min' else valid[col].idxmax()
        r = valid.loc[idx]
        lines.append(f"- Best {title}: **{r['display_name']}** ({r[col]:.6g}).")
    if not paired.empty:
        supported = paired[paired['A0_supported_by_CI']]
        lines += ['', f'- Paired comparisons whose 95% CI supports A0: {len(supported)} of {len(paired)} tested metric-comparison pairs.', '- A positive paired difference favors A0 by construction.']
    (out / 'result_interpretation_factual.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    official_pairs = discover(Path(args.results_root), 'official_independent_seed_best')
    if args.reference_root:
        official_pairs += discover(Path(args.reference_root), 'official_independent_seed_best')
    if not official_pairs:
        raise FileNotFoundError('No official independent-seed best-checkpoint results were found.')
    summary, seq = load_results(official_pairs)
    if args.mode == 'baseline':
        wanted = BASELINE_ORDER
    else:
        wanted = ABLATION_ORDER
    summary = summary[summary['run_label'].isin(wanted)].copy()
    seq = seq[seq['run_label'].isin(wanted)].copy()
    audit = validate_independent_seeds(summary, out)
    stats = ordered(scalar_seed_stats(summary), wanted)
    stats.to_csv(out / 'model_mean_sd_across_independent_seeds.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(out / 'seed_level_summary.csv', index=False, encoding='utf-8-sig')
    seq.to_csv(out / 'all_sequence_results.csv', index=False, encoding='utf-8-sig')
    caption = 'Comparison of independently trained baselines using three random seeds.' if args.mode == 'baseline' else 'Retraining-based loss ablations using three independent random seeds.'
    write_main_latex(stats, out / f'{args.mode}_main_table.tex', caption, f'tab:{args.mode}_main')
    scale_df = per_scale(seq)
    scale_df.to_csv(out / 'per_scale_mean_sd.csv', index=False, encoding='utf-8-sig')
    paired = paired_comparisons(seq, args.bootstrap, rng)
    paired.to_csv(out / 'paired_sequence_comparisons_vs_A0.csv', index=False, encoding='utf-8-sig')
    write_paired_latex(paired, out / f'{args.mode}_paired_table.tex')
    plot_bars(stats, out)
    plot_per_scale(scale_df, out)
    plot_paired(paired, out)
    sensitivity_pairs = []
    if args.checkpoint_sensitivity_root:
        sensitivity_pairs += discover(Path(args.checkpoint_sensitivity_root), 'checkpoint_sensitivity')
    if args.reference_checkpoint_sensitivity_root:
        sensitivity_pairs += discover(Path(args.reference_checkpoint_sensitivity_root), 'checkpoint_sensitivity')
    if sensitivity_pairs:
        temp_summary, _ = load_results(sensitivity_pairs)
        relevant_dirs = set(temp_summary[temp_summary['run_label'].isin(wanted)]['result_dir'])
        sensitivity_pairs = [pair for pair in sensitivity_pairs if str(pair[0].parent) in relevant_dirs]
        checkpoint_sensitivity(sensitivity_pairs, out)
    write_interpretation(stats, paired, out)
    manifest = {'mode': args.mode, 'official_result_directories': len(official_pairs), 'expected_seeds': SEEDS, 'all_runs_have_three_independent_seeds': bool(audit['valid_three_independent_seeds'].all()), 'checkpoint_sensitivity_is_excluded_from_seed_statistics': True, 'bootstrap_iterations': args.bootstrap}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)
if __name__ == '__main__':
    main()
