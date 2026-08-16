from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm.auto import tqdm
import dataset
from model_variants import VARIANT_SPECS, build_model
from experiment_utils import estimate_raft_pair, seed_everything
from train import InputPadder, collect_paths, load_raft, resolve_scale_roots, unpack_batch
STRATA = ['Near', 'Middle', 'Far']
COMPONENT_GROUPS = ['Center retention', 'Off-center transport', 'Boundary compensation']

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
    parser = argparse.ArgumentParser(description='Secondary multi-seed SOC-DS transport diagnostics')
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--checkpoint', action='append', default=[], help='Repeat for each A0 checkpoint.')
    parser.add_argument('--checkpoint_root', default=None, help='Optional baseline_train root for auto-discovery.')
    parser.add_argument('--raft_model', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--test_start', type=int, default=36)
    parser.add_argument('--test_end', type=int, default=40)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=99)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--raft_iters', type=int, default=4)
    parser.add_argument('--use_star_enhanced', type=str2bool, default=True)
    parser.add_argument('--fallback_to_raw', type=str2bool, default=True)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260723)
    parser.add_argument('--max_frames_per_seed', type=int, default=None, help='Debug-only cap.')
    return parser.parse_args()

def strip_module_prefix(state_dict):
    return {key[7:] if key.startswith('module.') else key: value for key, value in state_dict.items()}

def discover_checkpoints(args) -> List[Path]:
    paths = [Path(item) for item in args.checkpoint]
    if args.checkpoint_root:
        root = Path(args.checkpoint_root)
        candidate_root = root / 'A0_full_socds' if (root / 'A0_full_socds').exists() else root
        paths.extend(sorted(candidate_root.glob('seed_*/model_best.pth.tar')))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            if not path.exists():
                raise FileNotFoundError(path)
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ValueError('Provide --checkpoint one or more times, or set --checkpoint_root.')
    return unique

def parse_scale_from_path(path: str) -> int:
    sample_dir = Path(path).parent
    if sample_dir.name == 'star_enhanced':
        sample_dir = sample_dir.parent
    return int(sample_dir.parent.name)

def frame_metrics(weight: torch.Tensor, propagation: torch.Tensor, support: torch.Tensor) -> dict:
    support = support.bool()
    if not bool(support.any()):
        support = torch.ones_like(support, dtype=torch.bool)
    w = weight[:, support]
    mean_w = w.mean(dim=1)
    entropy = (-(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(dim=0) / np.log(3.0)).mean()
    selectivity = (w.max(dim=0).values - 1.0 / 3.0).mean()
    spread = (w.max(dim=0).values - w.min(dim=0).values).mean()
    p = propagation.view(3, 10, propagation.shape[-2], propagation.shape[-1])
    support_f = support.to(dtype=p.dtype)
    supported_energy = (p * support_f[None, None]).sum(dim=(-2, -1))
    global_energy = p.sum(dim=(-2, -1))
    row = {'branch_1_mean': float(mean_w[0].detach().cpu()), 'branch_2_mean': float(mean_w[1].detach().cpu()), 'branch_3_mean': float(mean_w[2].detach().cpu()), 'branch_entropy_normalized': float(entropy.detach().cpu()), 'branch_selectivity': float(selectivity.detach().cpu()), 'branch_max_minus_min': float(spread.detach().cpu())}
    for prefix, energy in [('supported', supported_energy), ('global', global_energy)]:
        denom = energy.sum(dim=1).clamp_min(1e-12)
        center = energy[:, 4] / denom
        offcenter = (energy[:, :4].sum(dim=1) + energy[:, 5:9].sum(dim=1)) / denom
        boundary = energy[:, 9] / denom
        q_ratio = energy / denom[:, None]
        for layer_idx, layer_name in enumerate(STRATA):
            row[f'{prefix}_{layer_name.lower()}_center_ratio'] = float(center[layer_idx].detach().cpu())
            row[f'{prefix}_{layer_name.lower()}_offcenter_ratio'] = float(offcenter[layer_idx].detach().cpu())
            row[f'{prefix}_{layer_name.lower()}_boundary_ratio'] = float(boundary[layer_idx].detach().cpu())
            for q in range(10):
                row[f'{prefix}_{layer_name.lower()}_q{q + 1}_ratio'] = float(q_ratio[layer_idx, q].detach().cpu())
    return row

def aggregate_sequence(frame_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ['seed', 'copy_count', 'sample_name']
    metric_cols = [column for column in frame_df.columns if column not in id_cols + ['frame_index']]
    return frame_df.groupby(id_cols, as_index=False)[metric_cols].mean()

def hierarchical_bootstrap(sequence_df: pd.DataFrame, metric: str, n_boot: int, rng: np.random.Generator):
    data = sequence_df.dropna(subset=[metric]).copy()
    seeds = sorted(data['seed'].unique())
    scales = sorted(data['copy_count'].unique())
    point = float(data[metric].mean())
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        values = []
        for seed in sampled_seeds:
            seed_df = data[data['seed'] == seed]
            for scale in scales:
                scale_df = seed_df[seed_df['copy_count'] == scale]
                if scale_df.empty:
                    continue
                indices = rng.integers(0, len(scale_df), size=len(scale_df))
                values.extend(scale_df.iloc[indices][metric].to_numpy(float).tolist())
        boot[b] = float(np.mean(values))
    return (point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

def build_summary(sequence_df: pd.DataFrame, args) -> pd.DataFrame:
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    branch_metrics = [('Branch 1 mean', 'branch_1_mean'), ('Branch 2 mean', 'branch_2_mean'), ('Branch 3 mean', 'branch_3_mean'), ('Normalized branch entropy', 'branch_entropy_normalized'), ('Branch selectivity', 'branch_selectivity'), ('Branch max-minus-min', 'branch_max_minus_min')]
    for label, metric in branch_metrics:
        mean, low, high = hierarchical_bootstrap(sequence_df, metric, args.bootstrap, rng)
        rows.append({'diagnostic_family': 'branch', 'group': 'Overall', 'metric': label, 'mean': mean, 'ci_low': low, 'ci_high': high})
    for stratum in STRATA:
        for group_label, suffix in [('Center retention', 'center_ratio'), ('Off-center transport', 'offcenter_ratio'), ('Boundary compensation', 'boundary_ratio')]:
            metric = f'supported_{stratum.lower()}_{suffix}'
            mean, low, high = hierarchical_bootstrap(sequence_df, metric, args.bootstrap, rng)
            rows.append({'diagnostic_family': 'transport', 'group': stratum, 'metric': group_label, 'mean': mean, 'ci_low': low, 'ci_high': high})
    return pd.DataFrame(rows)

def save_figure(summary: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2)
    branch = summary[(summary['diagnostic_family'] == 'branch') & summary['metric'].str.startswith('Branch ')]
    branch = branch[branch['metric'].isin(['Branch 1 mean', 'Branch 2 mean', 'Branch 3 mean'])]
    x = np.arange(3)
    y = branch['mean'].to_numpy(float)
    lower = y - branch['ci_low'].to_numpy(float)
    upper = branch['ci_high'].to_numpy(float) - y
    axes[0].errorbar(x, y, yerr=np.vstack([lower, upper]), fmt='o')
    axes[0].axhline(1.0 / 3.0, label='Uniform weight')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Branch 1', 'Branch 2', 'Branch 3'])
    axes[0].set_ylabel('Target-supported mean weight')
    axes[0].set_title('(a) RAFT-conditioned branch weighting')
    axes[0].legend()
    axes[0].grid(axis='y')
    transport = summary[summary['diagnostic_family'] == 'transport']
    x = np.arange(len(STRATA))
    offsets = np.linspace(-0.2, 0.2, len(COMPONENT_GROUPS))
    for offset, component in zip(offsets, COMPONENT_GROUPS):
        comp = transport[transport['metric'] == component].set_index('group').loc[STRATA].reset_index()
        y = 100.0 * comp['mean'].to_numpy(float)
        lower = 100.0 * (comp['mean'] - comp['ci_low']).to_numpy(float)
        upper = 100.0 * (comp['ci_high'] - comp['mean']).to_numpy(float)
        axes[1].errorbar(x + offset, y, yerr=np.vstack([lower, upper]), fmt='o', label=component)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(STRATA)
    axes[1].set_ylabel('Target-supported output-energy ratio (%)')
    axes[1].set_title('(b) Grouped transport-component composition')
    axes[1].legend()
    axes[1].grid(axis='y')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)

def write_latex_table(summary: pd.DataFrame, path: Path):

    def row_for(group, metric):
        row = summary[(summary['group'] == group) & (summary['metric'] == metric)].iloc[0]
        return f"{100.0 * row['mean']:.2f} [{100.0 * row['ci_low']:.2f}, {100.0 * row['ci_high']:.2f}]"
    branch_entropy = summary[(summary['group'] == 'Overall') & (summary['metric'] == 'Normalized branch entropy')].iloc[0]
    branch_select = summary[(summary['group'] == 'Overall') & (summary['metric'] == 'Branch selectivity')].iloc[0]
    lines = ['\\begin{table}[t]', '\\centering', '\\caption{Secondary diagnostics of target-supported branch weighting and grouped transport-output composition.}', '\\label{tab:secondary_transport_diagnostics}', '\\scriptsize', '\\setlength{\\tabcolsep}{3.4pt}', '\\begin{tabular}{lccc}', '\\toprule', 'Stratum & Center (\\%) & Off-center (\\%) & Boundary (\\%) \\\\', '\\midrule']
    for stratum in STRATA:
        lines.append(f"{stratum} & {row_for(stratum, 'Center retention')} & {row_for(stratum, 'Off-center transport')} & {row_for(stratum, 'Boundary compensation')}" + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\vspace{2pt}', f"\\parbox{{0.98\\columnwidth}}{{\\scriptsize Normalized branch entropy: {branch_entropy['mean']:.4f} [{branch_entropy['ci_low']:.4f}, {branch_entropy['ci_high']:.4f}]; mean selectivity above $1/3$: {branch_select['mean']:.4f} [{branch_select['ci_low']:.4f}, {branch_select['ci_high']:.4f}]. Values use target-supported regions and population-scale-stratified hierarchical-bootstrap 95\\% confidence intervals.}}", '\\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    args = parse_args()
    checkpoints = discover_checkpoints(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale_roots = resolve_scale_roots(args.root_dir, None)
    paths, _ = collect_paths(scale_roots, args.test_start, args.test_end, args.frame_start, args.frame_end)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ds = dataset.listDataset(paths, shuffle=False, transform=transform, train=False, use_star_enhanced=args.use_star_enhanced, use_depth_guidance=True, fallback_to_raw=args.fallback_to_raw, return_frame_index=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)
    raft_args = SimpleNamespace(raft_model=args.raft_model, small=False, mixed_precision=False, alternate_corr=False)
    raft = load_raft(raft_args, device, required=True)
    spec = VARIANT_SPECS['A0_full_socds']
    frame_rows: List[dict] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        variant = checkpoint.get('variant', 'A0_full_socds')
        if variant != 'A0_full_socds':
            raise ValueError(f'Expected A0_full_socds checkpoint, got {variant}: {checkpoint_path}')
        seed = int(checkpoint.get('args', {}).get('seed', checkpoint.get('seed', 12345)))
        seed_everything(seed, deterministic=True)
        model = build_model('A0_full_socds', use_pretrained_frontend=False).to(device)
        model.load_state_dict(strip_module_prefix(checkpoint['state_dict']), strict=True)
        model.eval()
        progress = tqdm(loader, desc=f'Transport diagnostics seed={seed}', dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress, start=1):
                data = unpack_batch(batch, device)
                flow_fwd, _ = estimate_raft_pair(raft, data['raft_image1'], data['raft_image2'], InputPadder, 'shared_forward', args.raft_iters, device)
                output = model(data['prev_rgb'], data['curr_rgb'], flow_fwd, return_intermediate=True)
                if output.intermediate is None:
                    raise RuntimeError('Model did not return intermediate tensors.')
                weight = output.intermediate['branch_weight'][0]
                propagation = output.prediction[0]
                support = data['curr_mask'][0, 0] > 0
                row = {'seed': seed, 'copy_count': parse_scale_from_path(paths[batch_idx - 1]), 'sample_name': data['sample_name'][0], 'frame_index': int(data['frame_index'].item())}
                row.update(frame_metrics(weight, propagation, support))
                frame_rows.append(row)
                if args.max_frames_per_seed is not None and batch_idx >= args.max_frames_per_seed:
                    break
    frame_df = pd.DataFrame(frame_rows)
    sequence_df = aggregate_sequence(frame_df)
    summary = build_summary(sequence_df, args)
    frame_df.to_csv(out_dir / 'transport_frame_diagnostics.csv', index=False, encoding='utf-8-sig')
    sequence_df.to_csv(out_dir / 'transport_sequence_diagnostics.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(out_dir / 'transport_diagnostic_summary_with_ci.csv', index=False, encoding='utf-8-sig')
    save_figure(summary, out_dir / 'fig_secondary_transport_diagnostics.pdf')
    write_latex_table(summary, out_dir / 'tab_secondary_transport_diagnostics.tex')
    manifest = {'checkpoints': [str(path.resolve()) for path in checkpoints], 'seeds': sorted((int(value) for value in frame_df['seed'].unique())), 'num_frame_rows': int(len(frame_df)), 'num_sequence_rows': int(len(sequence_df)), 'interpretation': 'secondary descriptive diagnostics; not a retraining-based ablation', 'outputs': ['transport_frame_diagnostics.csv', 'transport_sequence_diagnostics.csv', 'transport_diagnostic_summary_with_ci.csv', 'fig_secondary_transport_diagnostics.pdf', 'tab_secondary_transport_diagnostics.tex']}
    (out_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
