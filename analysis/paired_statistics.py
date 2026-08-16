from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm
A0 = 'A0_full_socds'
ORDER = ['A0_full_socds', 'A1_single_frame_stratified', 'A2_no_depth', 'A3_no_flow', 'A4_direct_stratified_head', 'A5_single_projected_density', 'A6_direct_count_regression']
DISPLAY = {'A0_full_socds': 'A0 Full SOC-DS', 'A1_single_frame_stratified': 'A1 Single frame', 'A2_no_depth': 'A2 No depth', 'A3_no_flow': 'A3 No flow', 'A4_direct_stratified_head': 'A4 Direct stratified', 'A5_single_projected_density': 'A5 Single density', 'A6_direct_count_regression': 'A6 Count regression'}
METRICS = [{'key': 'total_count_mae', 'label': 'Total-count MAE', 'direction': 'error', 'scale': 1.0, 'digits': 3}, {'key': 'stratum_mae', 'label': 'Mean stratum-wise MAE', 'direction': 'error', 'scale': 1.0, 'digits': 3}, {'key': 'density_rmse', 'label': 'Total-density RMSE', 'direction': 'error', 'scale': 1.0, 'digits': 5}, {'key': 'allocation_diag', 'label': 'Allocation diagonal (percentage points)', 'direction': 'allocation', 'scale': 100.0, 'digits': 2}]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rebuild the manuscript baseline tables and paired forest plot from the output of 05_build_baseline_latex_and_figures.bat.')
    parser.add_argument('--input_dir', required=True, help='Directory containing model_mean_sd_across_independent_seeds.csv and all_sequence_results.csv.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260720)
    return parser.parse_args()

def latex_escape(text: str) -> str:
    replacements = {'&': '\\&', '%': '\\%', '_': '\\_', '#': '\\#', '$': '\\$', '{': '\\{', '}': '\\}'}
    return ''.join((replacements.get(ch, ch) for ch in str(text)))

def format_mean_sd(mean: float, sd: float, digits: int) -> str:
    if pd.isna(mean):
        return '--'
    if pd.isna(sd):
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'

def ordered(df: pd.DataFrame) -> pd.DataFrame:
    ranks = {name: index for index, name in enumerate(ORDER)}
    result = df.copy()
    result['_rank'] = result['run_label'].map(ranks).fillna(999)
    return result.sort_values('_rank').drop(columns='_rank').reset_index(drop=True)

def first_existing_column(df: pd.DataFrame, names: List[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f'None of these columns exists: {names}')

def prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    required = {'run_label', 'trainable_parameters'}
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f'Summary CSV is missing required columns: {sorted(missing)}')
    mappings = {'total_mean': ['mean_total_count_mae_mean', 'total_count_mae_mean'], 'total_sd': ['mean_total_count_mae_sd', 'total_count_mae_sd'], 'stratum_mean': ['mean_stratum_mae_mean', 'stratum_mae_mean'], 'stratum_sd': ['mean_stratum_mae_sd', 'stratum_mae_sd'], 'rmse_mean': ['mean_total_density_rmse_mean', 'total_density_rmse_mean'], 'rmse_sd': ['mean_total_density_rmse_sd', 'total_density_rmse_sd'], 'micro_mean': ['allocation_micro_diagonal_mean_mean', 'micro_allocation_diagonal_mean'], 'micro_sd': ['allocation_micro_diagonal_mean_sd', 'micro_allocation_diagonal_sd'], 'macro_mean': ['allocation_sequence_macro_diagonal_mean_mean', 'macro_allocation_diagonal_mean'], 'macro_sd': ['allocation_sequence_macro_diagonal_mean_sd', 'macro_allocation_diagonal_sd'], 'time_mean': ['mean_inference_time_sec_mean', 'inference_time_sec_mean'], 'time_sd': ['mean_inference_time_sec_sd', 'inference_time_sec_sd']}
    result = pd.DataFrame()
    result['run_label'] = summary['run_label']
    result['display_name'] = result['run_label'].map(DISPLAY).fillna(result['run_label'])
    result['params_m'] = pd.to_numeric(summary['trainable_parameters'], errors='coerce') / 1000000.0
    for target, candidates in mappings.items():
        try:
            source = first_existing_column(summary, candidates)
            result[target] = pd.to_numeric(summary[source], errors='coerce')
        except KeyError:
            result[target] = np.nan
    result['micro_mean'] *= 100.0
    result['micro_sd'] *= 100.0
    result['macro_mean'] *= 100.0
    result['macro_sd'] *= 100.0
    result['time_mean'] *= 1000.0
    result['time_sd'] *= 1000.0
    return ordered(result)

def write_main_table(summary: pd.DataFrame, output_path: Path) -> None:
    best_by_column = {'total_mean': summary['total_mean'].idxmin(), 'stratum_mean': summary['stratum_mean'].idxmin(), 'rmse_mean': summary['rmse_mean'].idxmin(), 'micro_mean': summary['micro_mean'].idxmax(), 'macro_mean': summary['macro_mean'].idxmax(), 'time_mean': summary['time_mean'].idxmin()}
    lines = ['\\begin{table*}[t]', '\\centering', "\\caption{Retraining-based comparison with independently trained baselines. Results are reported as mean $\\pm$ standard deviation over three random seeds. Downward and upward arrows indicate lower-is-better and higher-is-better metrics, respectively; ``--'' denotes a metric that is not applicable to the corresponding output representation.}", '\\label{tab:baseline_main}', '\\scriptsize', '\\setlength{\\tabcolsep}{2.4pt}', '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{lcccccccc}', '\\toprule', '\\multirow{2}{*}{Model configuration} & \\multicolumn{1}{c}{Count accuracy} & \\multicolumn{2}{c}{Stratified-density accuracy} & \\multicolumn{2}{c}{Allocation fidelity} & \\multicolumn{2}{c}{Complexity and efficiency} \\\\', '\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}', ' & Total-count MAE $\\downarrow$ & Mean stratum-wise MAE $\\downarrow$ & Total-density RMSE $\\downarrow$ & Micro diagonal (\\%) $\\uparrow$ & Sequence-macro diagonal (\\%) $\\uparrow$ & Parameters (M) & Time (ms/frame) $\\downarrow$ \\\\', '\\midrule']
    columns = [('total_mean', 'total_sd', 3), ('stratum_mean', 'stratum_sd', 3), ('rmse_mean', 'rmse_sd', 5), ('micro_mean', 'micro_sd', 2), ('macro_mean', 'macro_sd', 2), ('params_m', None, 2), ('time_mean', 'time_sd', 2)]
    for index, row in summary.iterrows():
        cells = [latex_escape(row['display_name'])]
        for mean_col, sd_col, digits in columns:
            mean = row[mean_col]
            sd = row[sd_col] if sd_col else np.nan
            value = format_mean_sd(mean, sd, digits)
            if mean_col in best_by_column and best_by_column[mean_col] == index:
                if value != '--':
                    value = '\\textbf{' + value + '}'
            cells.append(value)
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}', '']
    output_path.write_text('\n'.join(lines), encoding='utf-8')

def add_sequence_metrics(sequence: pd.DataFrame) -> pd.DataFrame:
    result = sequence.copy()
    if all((column in result.columns for column in ['abs_error_near', 'abs_error_mid', 'abs_error_far'])):
        result['stratum_mae'] = result[['abs_error_near', 'abs_error_mid', 'abs_error_far']].mean(axis=1)
    diagonal_candidates = [['alloc_micro_near_to_near', 'alloc_micro_mid_to_mid', 'alloc_micro_far_to_far'], ['alloc_near_to_near', 'alloc_mid_to_mid', 'alloc_far_to_far']]
    for columns in diagonal_candidates:
        if all((column in result.columns for column in columns)):
            result['allocation_diag'] = result[columns].mean(axis=1)
            break
    return result

def hierarchical_bootstrap_ci(merged: pd.DataFrame, difference_column: str, iterations: int, rng: np.random.Generator) -> Tuple[float, float]:
    seeds = sorted(merged['seed'].unique())
    bootstrap_means = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_values: List[float] = []
        for seed in sampled_seeds:
            seed_frame = merged[merged['seed'] == seed]
            for _, scale_frame in seed_frame.groupby('copy_count', sort=False):
                indices = rng.integers(low=0, high=len(scale_frame), size=len(scale_frame))
                sampled_values.extend(scale_frame.iloc[indices][difference_column].to_numpy(float))
        bootstrap_means[iteration] = float(np.mean(sampled_values))
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return (float(low), float(high))

def paired_results(sequence: pd.DataFrame, iterations: int, rng: np.random.Generator) -> pd.DataFrame:
    sequence = add_sequence_metrics(sequence)
    reference = sequence[sequence['run_label'] == A0].copy()
    alternatives = [label for label in ORDER if label != A0 and label in sequence['run_label'].unique()]
    source_columns = {'total_count_mae': 'abs_error_total', 'stratum_mae': 'stratum_mae', 'density_rmse': 'total_density_rmse', 'allocation_diag': 'allocation_diag'}
    rows = []
    total_jobs = sum((1 for alternative in alternatives for metric in METRICS if source_columns[metric['key']] in reference.columns and source_columns[metric['key']] in sequence[sequence['run_label'] == alternative].columns))
    progress = tqdm(total=total_jobs, desc='Paired hierarchical bootstrap', unit='comparison')
    for alternative in alternatives:
        alternative_frame = sequence[sequence['run_label'] == alternative].copy()
        for metric in METRICS:
            source = source_columns[metric['key']]
            if source not in reference.columns or source not in alternative_frame.columns:
                continue
            keys = ['seed', 'copy_count', 'sample_name']
            merged = reference[keys + [source]].rename(columns={source: 'A0'}).merge(alternative_frame[keys + [source]].rename(columns={source: 'alternative'}), on=keys, how='inner').dropna()
            if merged.empty:
                progress.update(1)
                continue
            if metric['direction'] == 'error':
                merged['difference'] = merged['alternative'] - merged['A0']
            else:
                merged['difference'] = merged['A0'] - merged['alternative']
            difference = merged['difference'].to_numpy(float)
            mean_difference = float(difference.mean())
            standard_deviation = float(difference.std(ddof=1)) if difference.size > 1 else np.nan
            ci_low, ci_high = hierarchical_bootstrap_ci(merged, 'difference', iterations, rng)
            scale = metric['scale']
            rows.append({'alternative': alternative, 'display_name': DISPLAY.get(alternative, alternative), 'metric': metric['key'], 'outcome': metric['label'], 'mean_difference': mean_difference * scale, 'ci_low': ci_low * scale, 'ci_high': ci_high * scale, 'effect_size_dz': mean_difference / standard_deviation if np.isfinite(standard_deviation) and standard_deviation > 0 else np.nan, 'n_seed_sequences': int(len(merged)), 'positive_favors_A0': True})
            progress.update(1)
    progress.close()
    result = pd.DataFrame(rows)
    if not result.empty:
        invalid = result[result['ci_low'] > result['ci_high']]
        if not invalid.empty:
            raise RuntimeError('At least one confidence interval has ci_low > ci_high:\n' + invalid.to_string(index=False))
    return result

def write_paired_table(paired: pd.DataFrame, output_path: Path) -> None:
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Paired sequence-level comparisons between the full SOC-DS model and each retrained baseline. Differences are defined so that positive values favor A0. Confidence intervals are obtained using a population-scale-stratified hierarchical bootstrap.}', '\\label{tab:paired_comparisons_a0}', '\\scriptsize', '\\setlength{\\tabcolsep}{4.0pt}', '\\begin{tabular}{llccc}', '\\toprule', 'Comparator & Outcome & Mean paired difference & Stratified-bootstrap 95\\% CI & $d_z$ \\\\', '\\midrule']
    rank = {name: index for index, name in enumerate(ORDER)}
    metric_rank = {metric['key']: index for index, metric in enumerate(METRICS)}
    paired = paired.copy()
    paired['_model_rank'] = paired['alternative'].map(rank)
    paired['_metric_rank'] = paired['metric'].map(metric_rank)
    paired = paired.sort_values(['_model_rank', '_metric_rank'])
    digits_by_metric = {metric['key']: metric['digits'] for metric in METRICS}
    previous_model = None
    for _, row in paired.iterrows():
        if previous_model is not None and row['alternative'] != previous_model:
            lines.append('\\addlinespace[1.5pt]')
        digits = digits_by_metric[row['metric']]
        mean_text = f"{row['mean_difference']:.{digits}f}"
        ci_text = f"[{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
        effect_text = f"{row['effect_size_dz']:.3f}"
        lines.append(f"{latex_escape(row['display_name'])} & {latex_escape(row['outcome'])} & {mean_text} & {ci_text} & {effect_text} \\\\")
        previous_model = row['alternative']
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    output_path.write_text('\n'.join(lines), encoding='utf-8')

def plot_paired_forest(paired: pd.DataFrame, output_dir: Path) -> None:
    metric_order = [metric['key'] for metric in METRICS]
    title_by_metric = {metric['key']: metric['label'] for metric in METRICS}
    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()
    for panel_index, (axis, metric_key) in enumerate(zip(axes, metric_order)):
        frame = paired[paired['metric'] == metric_key].copy()
        if frame.empty:
            axis.axis('off')
            continue
        ranks = {name: index for index, name in enumerate(ORDER)}
        frame['_rank'] = frame['alternative'].map(ranks)
        frame = frame.sort_values('_rank', ascending=False).reset_index(drop=True)
        y = np.arange(len(frame))
        center = frame['mean_difference'].to_numpy(float)
        lower = frame['ci_low'].to_numpy(float)
        upper = frame['ci_high'].to_numpy(float)
        xerr = np.vstack([center - lower, upper - center])
        axis.errorbar(center, y, xerr=xerr, fmt='o')
        axis.axvline(0.0)
        axis.set_yticks(y)
        axis.set_yticklabels(frame['display_name'])
        axis.set_xlabel('Paired difference (positive favors A0)')
        axis.set_title(title_by_metric[metric_key])
        axis.grid(axis='x')
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for spine in ['top', 'right']:
            axis.spines[spine].set_visible(False)
        axis.text(-0.12, 1.04, f'({chr(97 + panel_index)})', transform=axis.transAxes, va='bottom')
    figure_png = output_dir / 'fig_paired_baseline_forest.png'
    figure_pdf = output_dir / 'fig_paired_baseline_forest.pdf'
    fig.savefig(figure_png, bbox_inches='tight')
    fig.savefig(figure_pdf, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = input_dir / 'model_mean_sd_across_independent_seeds.csv'
    sequence_path = input_dir / 'all_sequence_results.csv'
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not sequence_path.exists():
        raise FileNotFoundError(sequence_path)
    raw_summary = pd.read_csv(summary_path)
    raw_sequence = pd.read_csv(sequence_path)
    summary = prepare_summary(raw_summary)
    summary.to_csv(output_dir / 'baseline_required_metrics_mean_sd.csv', index=False, encoding='utf-8-sig')
    write_main_table(summary, output_dir / 'baseline_main_table_revised.tex')
    rng = np.random.default_rng(args.bootstrap_seed)
    paired = paired_results(raw_sequence, iterations=args.bootstrap, rng=rng)
    paired.to_csv(output_dir / 'paired_sequence_results_revised.csv', index=False, encoding='utf-8-sig')
    write_paired_table(paired, output_dir / 'paired_comparisons_a0_revised.tex')
    plot_paired_forest(paired, output_dir)
    manifest = {'input_summary': str(summary_path.resolve()), 'input_sequence_results': str(sequence_path.resolve()), 'bootstrap_iterations': args.bootstrap, 'bootstrap_seed': args.bootstrap_seed, 'paired_difference_definition': {'error_metrics': 'alternative minus A0', 'allocation_metric': 'A0 minus alternative'}, 'positive_values_favor_A0': True, 'outputs': ['baseline_main_table_revised.tex', 'paired_comparisons_a0_revised.tex', 'fig_paired_baseline_forest.png', 'fig_paired_baseline_forest.pdf']}
    (output_dir / 'rebuild_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
