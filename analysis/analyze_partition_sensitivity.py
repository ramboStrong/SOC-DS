from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'partition'))
from sensitivity_partition import get_partition_spec, list_partition_ids, load_partition_config

def parse_args():
    p = argparse.ArgumentParser('Build the compact distance-partition sensitivity table and boundary figure')
    p.add_argument('--experiment_root', required=True)
    p.add_argument('--partition_config', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seeds', default='12345,23456,34567')
    p.add_argument('--bootstrap_reps', type=int, default=2000)
    p.add_argument('--bootstrap_seed', type=int, default=20260723)
    return p.parse_args()

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]

def fmt_mean_sd(mean: float, sd: float, digits: int) -> str:
    return f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'

def load_runs(root: Path, partition_ids: List[str], seeds: List[int]) -> pd.DataFrame:
    rows = []
    missing = []
    for pid in partition_ids:
        for seed in seeds:
            path = root / 'test' / pid / f'seed_{seed}' / 'best' / 'summary.json'
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open('r', encoding='utf-8') as handle:
                summary = json.load(handle)
            rows.append({'partition_id': pid, 'seed': seed, 'num_strata': int(summary['num_strata']), 'thresholds': json.dumps(summary['thresholds']), 'total_count_mae': float(summary['mean_total_count_mae']), 'total_density_rmse': float(summary['mean_total_density_rmse']), 'macro_stratum_relative_error_pct': 100.0 * float(summary['mean_macro_stratum_relative_error']), 'allocation_micro_diagonal_pct': 100.0 * float(summary['allocation_micro_diagonal_mean']), 'trainable_parameters': int(summary['trainable_parameters'])})
    if missing:
        raise FileNotFoundError('Missing official test summaries:\n' + '\n'.join(missing))
    return pd.DataFrame(rows)

def aggregate_main(run_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    rows = []
    order = list_partition_ids(config)
    for pid in order:
        group = run_df[run_df['partition_id'] == pid]
        spec = get_partition_spec(config, pid)
        row = {'partition_id': pid, 'num_strata': spec.num_strata, 'thresholds': ', '.join((f'{x:.3f}' if abs(x - round(x)) > 1e-08 else f'{x:.0f}' for x in spec.thresholds)), 'seed_count': int(len(group))}
        for metric in ['total_count_mae', 'total_density_rmse', 'macro_stratum_relative_error_pct', 'allocation_micro_diagonal_pct', 'trainable_parameters']:
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_sd'] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def write_latex_table(summary_df: pd.DataFrame, path: Path):
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Sensitivity of SOC-DS to the number and placement of camera-centric distance strata. Results are mean $\\pm$ standard deviation over three independent training seeds.}', '\\label{tab:partition_sensitivity}', '\\setlength{\\tabcolsep}{6pt}', '\\begin{tabular}{lccccc}', '\\toprule', 'Design & $K$ & Thresholds & Total-count MAE & Total-density RMSE & Macro stratum error (\\%) \\\\', '\\midrule']
    for _, row in summary_df.iterrows():
        thresholds = row['thresholds'] if row['thresholds'] else '--'
        lines.append(f"{row['partition_id']} & {int(row['num_strata'])} & {thresholds} & {fmt_mean_sd(row['total_count_mae_mean'], row['total_count_mae_sd'], 3)} & {fmt_mean_sd(row['total_density_rmse_mean'], row['total_density_rmse_sd'], 5)} & {fmt_mean_sd(row['macro_stratum_relative_error_pct_mean'], row['macro_stratum_relative_error_pct_sd'], 2)} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table*}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def load_boundary_rows(root: Path, seeds: List[int], partition_ids=('P3_L', 'P3_U')) -> pd.DataFrame:
    frames = []
    missing = []
    for pid in partition_ids:
        for seed in seeds:
            path = root / 'test' / pid / f'seed_{seed}' / 'best' / 'boundary_sequence_results.csv'
            if not path.exists():
                missing.append(str(path))
                continue
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f'Boundary file is empty: {path}')
            frames.append(df)
    if missing:
        raise FileNotFoundError('Missing boundary-analysis files:\n' + '\n'.join(missing))
    return pd.concat(frames, ignore_index=True)

def hierarchical_bootstrap_ci(group: pd.DataFrame, *, reps: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    seeds = sorted(group['seed'].unique())
    seed_means = group.groupby('seed')['correct_stratum_mass_fraction'].mean()
    point = float(seed_means.mean())
    values = []
    for _ in range(reps):
        sampled_seed_ids = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_seed_means = []
        for seed in sampled_seed_ids:
            subset = group[group['seed'] == seed]['correct_stratum_mass_fraction'].dropna().to_numpy()
            sampled = rng.choice(subset, size=len(subset), replace=True)
            sampled_seed_means.append(float(np.mean(sampled)))
        values.append(float(np.mean(sampled_seed_means)))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (point, float(lo), float(hi))

def build_boundary_summary(boundary_df: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (pid, bin_index, label), group in boundary_df.groupby(['partition_id', 'bin_index', 'boundary_distance_bin'], sort=True):
        point, lo, hi = hierarchical_bootstrap_ci(group, reps=reps, rng=rng)
        rows.append({'partition_id': pid, 'bin_index': int(bin_index), 'boundary_distance_bin': label, 'correct_stratum_mass_pct': 100.0 * point, 'ci_lower_pct': 100.0 * lo, 'ci_upper_pct': 100.0 * hi, 'sequence_count': int(len(group))})
    return pd.DataFrame(rows).sort_values(['partition_id', 'bin_index'])

def plot_boundary(summary: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots()
    for pid, group in summary.groupby('partition_id', sort=False):
        group = group.sort_values('bin_index')
        x = np.arange(len(group))
        y = group['correct_stratum_mass_pct'].to_numpy()
        yerr = np.vstack([y - group['ci_lower_pct'].to_numpy(), group['ci_upper_pct'].to_numpy() - y])
        ax.errorbar(x, y, yerr=yerr, marker='o', label=pid)
    labels = summary.sort_values('bin_index').drop_duplicates('bin_index')['boundary_distance_bin'].tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Distance to the nearest partition threshold')
    ax.set_ylabel('Correct-stratum allocated mass (%)')
    ax.legend()
    ax.grid(axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_partition_boundary_fidelity.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'fig_partition_boundary_fidelity.png', bbox_inches='tight')
    plt.close(fig)

def main():
    args = parse_args()
    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_partition_config(args.partition_config)
    seeds = parse_int_list(args.seeds)
    run_df = load_runs(root, list_partition_ids(config), seeds)
    summary_df = aggregate_main(run_df, config)
    run_df.to_csv(output_dir / 'partition_sensitivity_seed_results.csv', index=False, encoding='utf-8-sig')
    summary_df.to_csv(output_dir / 'partition_sensitivity_summary.csv', index=False, encoding='utf-8-sig')
    write_latex_table(summary_df, output_dir / 'tab_partition_sensitivity.tex')
    boundary_df = load_boundary_rows(root, seeds)
    boundary_summary = build_boundary_summary(boundary_df, reps=args.bootstrap_reps, seed=args.bootstrap_seed)
    boundary_df.to_csv(output_dir / 'boundary_sequence_results_all.csv', index=False, encoding='utf-8-sig')
    boundary_summary.to_csv(output_dir / 'boundary_fidelity_summary.csv', index=False, encoding='utf-8-sig')
    plot_boundary(boundary_summary, output_dir)
    manifest = {'experiment_root': str(root.resolve()), 'partition_config': str(Path(args.partition_config).resolve()), 'seeds': seeds, 'main_table': 'tab_partition_sensitivity.tex', 'boundary_figure': 'fig_partition_boundary_fidelity.pdf', 'bootstrap_reps': args.bootstrap_reps, 'bootstrap_seed': args.bootstrap_seed}
    (output_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary_df.to_string(index=False))
    print(boundary_summary.to_string(index=False))
if __name__ == '__main__':
    main()
