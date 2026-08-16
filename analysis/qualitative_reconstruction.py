from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
STRATA = ['Near', 'Middle', 'Far', 'Total']
CASE_BEST = 'Best case'
CASE_TYPICAL = 'Typical case'

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
    parser = argparse.ArgumentParser(description='Build main-text and appendix qualitative SOC-DS figures')
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--baseline_train_root', required=True)
    parser.add_argument('--baseline_test_root', required=True)
    parser.add_argument('--raft_model', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--use_star_enhanced', type=str2bool, default=True)
    parser.add_argument('--fallback_to_raw', type=str2bool, default=True)
    parser.add_argument('--raft_iters', type=int, default=4)
    parser.add_argument('--generate_appendix', type=str2bool, default=True)
    parser.add_argument('--density_percentile', type=float, default=100.0, help='Percentile used for the shared density-map vmax.')
    parser.add_argument('--error_percentile', type=float, default=100.0, help='Percentile used for the shared absolute-error vmax.')
    return parser.parse_args()

def strip_module_prefix(state_dict):
    return {key[7:] if key.startswith('module.') else key: value for key, value in state_dict.items()}

def discover_seed_results(test_root: Path) -> List[Tuple[int, Path, Path]]:
    root = test_root / 'A0_full_socds' if (test_root / 'A0_full_socds').exists() else test_root
    items: List[Tuple[int, Path, Path]] = []
    for seed_dir in sorted(root.glob('seed_*')):
        best = seed_dir / 'best'
        summary = best / 'summary.json'
        frame = best / 'frame_results.csv'
        if summary.exists() and frame.exists():
            items.append((int(seed_dir.name.split('_')[-1]), summary, frame))
    if not items:
        raise FileNotFoundError(f'No A0 seed test results under {test_root}')
    return items

def choose_representative_seed(items: List[Tuple[int, Path, Path]]) -> Tuple[int, Path]:
    records = []
    for seed, summary_path, frame_path in items:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        records.append({'seed': seed, 'frame_path': str(frame_path), 'total_mae': float(summary['mean_total_count_mae']), 'stratum_mae': float(summary['mean_stratum_mae']), 'density_rmse': float(summary['mean_total_density_rmse']), 'allocation': float(summary['allocation_micro_diagonal_mean'])})
    df = pd.DataFrame(records)
    metrics = ['total_mae', 'stratum_mae', 'density_rmse', 'allocation']
    center = df[metrics].mean()
    scale = df[metrics].std(ddof=1).replace(0, 1.0).fillna(1.0)
    df['distance_to_multiseed_mean'] = ((df[metrics] - center) / scale).pow(2).sum(axis=1)
    selected = df.sort_values(['distance_to_multiseed_mean', 'seed']).iloc[0]
    return (int(selected['seed']), Path(selected['frame_path']))

def robust_z(series: pd.Series) -> pd.Series:
    values = series.to_numpy(float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    fallback = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    denom = 1.4826 * mad if mad > 1e-12 else max(fallback, 1e-12)
    return (series.astype(float) - median) / denom

def prepare_frame_metrics(frame_df: pd.DataFrame) -> pd.DataFrame:
    required = ['copy_count', 'sample_name', 'frame_index', 'abs_error_total', 'abs_error_near', 'abs_error_mid', 'abs_error_far', 'total_density_rmse']
    missing = [column for column in required if column not in frame_df.columns]
    if missing:
        raise KeyError(f'frame_results.csv missing columns: {missing}')
    df = frame_df.copy()
    df['copy_count'] = df['copy_count'].astype(int)
    df['frame_index'] = df['frame_index'].astype(int)
    df['frame_stratum_mae'] = df[['abs_error_near', 'abs_error_mid', 'abs_error_far']].mean(axis=1)
    groups = []
    for _, group in df.groupby('copy_count', sort=True):
        group = group.copy()
        z_total = robust_z(group['abs_error_total'])
        z_stratum = robust_z(group['frame_stratum_mae'])
        z_density = robust_z(group['total_density_rmse'])
        group['low_error_score'] = z_total + z_stratum + z_density
        group['typical_distance'] = np.sqrt(z_total.pow(2) + z_stratum.pow(2) + z_density.pow(2))
        groups.append(group)
    return pd.concat(groups, ignore_index=True)

def case_key(row: Mapping) -> Tuple[int, str, int]:
    return (int(row['copy_count']), str(row['sample_name']), int(row['frame_index']))

def select_best_typical_pair(group_df: pd.DataFrame) -> pd.DataFrame:
    df = group_df.copy()
    best = df.sort_values(['low_error_score', 'abs_error_total', 'frame_stratum_mae', 'total_density_rmse', 'sample_name', 'frame_index'], ascending=[True, True, True, True, True, True]).iloc[0]
    best_key = case_key(best)
    typical_candidates = df[df.apply(lambda row: case_key(row) != best_key, axis=1)].copy()
    if typical_candidates.empty:
        raise ValueError('Cannot select a distinct typical case from a one-row group.')
    typical = typical_candidates.sort_values(['typical_distance', 'abs_error_total', 'frame_stratum_mae', 'total_density_rmse', 'sample_name', 'frame_index'], ascending=[True, True, True, True, True, True]).iloc[0]
    selected = pd.DataFrame([best, typical]).copy()
    selected.insert(0, 'case_type', [CASE_BEST, CASE_TYPICAL])
    return selected

def select_main_cases(frame_df: pd.DataFrame) -> pd.DataFrame:
    best_group = frame_df[frame_df['copy_count'] == 100].copy()
    if best_group.empty:
        raise ValueError('No 100-object frames are available for the main-text best case.')
    best = best_group.sort_values(['low_error_score', 'abs_error_total', 'frame_stratum_mae', 'total_density_rmse', 'sample_name', 'frame_index'], ascending=[True, True, True, True, True, True]).iloc[0]
    best_key = case_key(best)
    typical_candidates = frame_df[frame_df.apply(lambda row: case_key(row) != best_key, axis=1)].copy()
    if typical_candidates.empty:
        raise ValueError('Cannot select a distinct main-text typical case.')
    typical = typical_candidates.sort_values(['typical_distance', 'abs_error_total', 'frame_stratum_mae', 'total_density_rmse', 'sample_name', 'frame_index'], ascending=[True, True, True, True, True, True]).iloc[0]
    selected = pd.DataFrame([best, typical]).copy()
    selected.insert(0, 'case_type', [CASE_BEST, CASE_TYPICAL])
    selected.insert(1, 'figure_scope', 'main')
    return selected

def select_appendix_cases(frame_df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for copy_count, group in frame_df.groupby('copy_count', sort=True):
        selected = select_best_typical_pair(group)
        selected.insert(1, 'figure_scope', 'appendix')
        selected['appendix_population_scale'] = int(copy_count)
        groups.append(selected)
    if not groups:
        return pd.DataFrame()
    return pd.concat(groups, ignore_index=True)

def frame_path(root: Path, copy_count: int, sample_name: str, frame_index: int) -> Path:
    path = root / str(int(copy_count)) / sample_name / f'frame_{int(frame_index):03d}.png'
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def build_loader(paths: Sequence[Path], args):
    from torchvision import transforms
    import dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ds = dataset.listDataset([str(path) for path in paths], shuffle=False, transform=transform, train=False, use_star_enhanced=args.use_star_enhanced, use_depth_guidance=True, fallback_to_raw=args.fallback_to_raw, return_frame_index=True)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

def load_rgb(root: Path, row: Mapping, use_star: bool, fallback: bool) -> np.ndarray:
    sample_dir = root / str(int(row['copy_count'])) / str(row['sample_name'])
    name = f"frame_{int(row['frame_index']):03d}.png"
    candidates: List[Path] = []
    if use_star:
        candidates.append(sample_dir / 'star_enhanced' / name)
    if fallback or not use_star:
        candidates.append(sample_dir / name)
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(candidates)
    return np.asarray(Image.open(path).convert('RGB'))

def canonical_layers(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] == 3:
        return array
    if array.ndim == 3 and array.shape[-1] == 3:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f'{name} must have shape [3,H,W] or [H,W,3], got {array.shape}')

def canonical_total(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f'{name} must have shape [H,W], [1,H,W], or [H,W,1], got {array.shape}')

def run_unique_cases(selected_unique: pd.DataFrame, checkpoint_path: Path, args) -> Dict[Tuple[int, str, int], dict]:
    from model_variants import VARIANT_SPECS, build_model
    from train import compute_outputs_and_loss, load_raft, unpack_batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model('A0_full_socds', use_pretrained_frontend=False).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint['state_dict']), strict=True)
    model.eval()
    spec = VARIANT_SPECS['A0_full_socds']
    raft_args = SimpleNamespace(raft_model=args.raft_model, small=False, mixed_precision=False, alternate_corr=False, flow_mode='bidirectional', raft_iters=args.raft_iters, lambda_layer=1.0, lambda_total=0.2, lambda_consistency=1.0, lambda_depth=0.1, lambda_count=1.0)
    raft = load_raft(raft_args, device, required=True)
    paths = [frame_path(Path(args.root_dir), int(row.copy_count), str(row.sample_name), int(row.frame_index)) for row in selected_unique.itertuples()]
    loader = build_loader(paths, args)
    criterion_sum = nn.MSELoss(reduction='sum').to(device)
    count_criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    outputs: Dict[Tuple[int, str, int], dict] = {}
    with torch.no_grad():
        for (_, selected_row), batch in zip(selected_unique.iterrows(), loader):
            data = unpack_batch(batch, device)
            _, _, pred = compute_outputs_and_loss(model, spec, data, raft, raft_args, criterion_sum, count_criterion, device)
            gt_layers = canonical_layers(data['curr_layer'][0].detach().cpu().numpy(), 'gt_layers')
            pred_layers = canonical_layers(pred['layers'].detach().cpu().numpy(), 'pred_layers')
            gt_total = canonical_total(data['curr_total'][0].detach().cpu().numpy(), 'gt_total')
            pred_total = canonical_total(pred['total'].detach().cpu().numpy(), 'pred_total')
            key = case_key(selected_row)
            outputs[key] = {'copy_count': int(selected_row['copy_count']), 'sample_name': str(selected_row['sample_name']), 'frame_index': int(selected_row['frame_index']), 'rgb': load_rgb(Path(args.root_dir), selected_row, args.use_star_enhanced, args.fallback_to_raw), 'gt': np.concatenate([gt_layers, gt_total[None]], axis=0), 'pred': np.concatenate([pred_layers, pred_total[None]], axis=0)}
    return outputs

def attach_outputs(selected: pd.DataFrame, output_lookup: Mapping) -> List[dict]:
    cases: List[dict] = []
    for _, row in selected.iterrows():
        key = case_key(row)
        if key not in output_lookup:
            raise KeyError(f'Missing inferred output for case {key}')
        case = dict(output_lookup[key])
        case['case_type'] = str(row['case_type'])
        case['figure_scope'] = str(row['figure_scope'])
        cases.append(case)
    return cases

def compute_density_vmax(cases: Iterable[dict], percentile: float) -> float:
    values: List[np.ndarray] = []
    for case in cases:
        gt = np.asarray(case['gt'], dtype=np.float32)
        pred = np.asarray(case['pred'], dtype=np.float32)
        values.extend([gt.reshape(-1), pred.reshape(-1)])
    merged = np.concatenate(values) if values else np.array([1.0], dtype=np.float32)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return 1.0
    vmax = float(np.percentile(merged, percentile))
    return max(vmax, 1e-08)

def compute_error_vmax(cases: Iterable[dict], percentile: float) -> float:
    values: List[np.ndarray] = []
    for case in cases:
        gt = np.asarray(case['gt'], dtype=np.float32)
        pred = np.asarray(case['pred'], dtype=np.float32)
        err = np.abs(pred - gt)
        values.append(err.reshape(-1))
    merged = np.concatenate(values) if values else np.array([1.0], dtype=np.float32)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return 1.0
    vmax = float(np.percentile(merged, percentile))
    return max(vmax, 1e-08)

def draw_count_text(ax, count_value: float):
    ax.text(0.035, 0.045, f'N={float(count_value):.1f}', transform=ax.transAxes, ha='left', va='bottom')

def style_image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def add_dual_colorbars(fig, density_axes, error_axes, density_vmax: float, error_vmax: float):
    density_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=density_vmax))
    density_sm.set_array([])
    cb_density = fig.colorbar(density_sm, ax=density_axes, orientation='horizontal')
    cb_density.set_label('Shared density-map scale')
    error_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=error_vmax))
    error_sm.set_array([])
    cb_error = fig.colorbar(error_sm, ax=error_axes, orientation='horizontal')
    cb_error.set_label('Shared absolute-error scale')

def save_pdf_png(fig, pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(pdf_path.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)

def save_main_figure(cases: Sequence[dict], path: Path, density_vmax: float, error_vmax: float):
    if len(cases) != 2:
        raise ValueError(f'Main figure expects exactly two cases, got {len(cases)}')
    fig = plt.figure()
    grid = fig.add_gridspec(nrows=8, ncols=4, width_ratios=[1.34, 1.0, 1.0, 1.0], hspace=0.18, wspace=0.055)
    density_axes, error_axes = ([], [])
    row_anchor_axes = []
    for case_index, case in enumerate(cases):
        row0 = case_index * 4
        rgb_ax = fig.add_subplot(grid[row0:row0 + 4, 0])
        rgb_ax.imshow(case['rgb'])
        rgb_ax.set_title(f"{case['case_type']}\n{int(case['copy_count'])}-object configuration")
        style_image_axis(rgb_ax)
        for layer_index, layer_name in enumerate(STRATA):
            gt = np.asarray(case['gt'][layer_index], dtype=np.float32)
            pred = np.asarray(case['pred'][layer_index], dtype=np.float32)
            error = np.abs(pred - gt)
            axes = [fig.add_subplot(grid[row0 + layer_index, column]) for column in [1, 2, 3]]
            row_anchor_axes.append(axes[0])
            axes[0].imshow(gt, vmin=0.0, vmax=density_vmax, interpolation='nearest')
            axes[1].imshow(pred, vmin=0.0, vmax=density_vmax, interpolation='nearest')
            axes[2].imshow(error, vmin=0.0, vmax=error_vmax, interpolation='nearest')
            draw_count_text(axes[0], float(gt.sum()))
            draw_count_text(axes[1], float(pred.sum()))
            axes[0].set_ylabel(layer_name)
            for idx, ax in enumerate(axes):
                style_image_axis(ax)
                if idx < 2:
                    density_axes.append(ax)
                else:
                    error_axes.append(ax)
            if case_index == 0 and layer_index == 0:
                axes[0].set_title('Reference')
                axes[1].set_title('Prediction')
                axes[2].set_title('Absolute error')
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.09, wspace=0.055, hspace=0.18)
    fig.canvas.draw()
    upper_bottom = row_anchor_axes[3].get_position().y0
    lower_top = row_anchor_axes[4].get_position().y1
    y_sep = 0.58 * lower_top + 0.42 * upper_bottom
    fig.add_artist(mpl.lines.Line2D([0.025, 0.995], [y_sep, y_sep], transform=fig.transFigure))
    map_bottom = min((ax.get_position().y0 for ax in density_axes + error_axes))
    colorbar_height = 0.012
    colorbar_gap = 0.012
    colorbar_y = max(0.012, map_bottom - colorbar_gap - colorbar_height)
    add_dual_colorbars(fig, density_axes, error_axes, density_vmax=density_vmax, error_vmax=error_vmax)
    save_pdf_png(fig, path)

def save_appendix_combined_figure(selected_df: pd.DataFrame, output_lookup: Mapping, path: Path, density_vmax: float, error_vmax: float):
    best_df = selected_df[selected_df['case_type'] == CASE_BEST].copy().sort_values('copy_count')
    typical_df = selected_df[selected_df['case_type'] == CASE_TYPICAL].copy().sort_values('copy_count')
    if len(best_df) != 10 or len(typical_df) != 10:
        raise ValueError('Appendix combined figure expects 10 best rows and 10 typical rows.')
    ordered_df = pd.concat([best_df, typical_df], ignore_index=True)
    ordered_cases = attach_outputs(ordered_df, output_lookup)
    column_titles = ['Input', 'Ref. Near', 'Pred. Near', 'Ref. Middle', 'Pred. Middle', 'Ref. Far', 'Pred. Far', 'Ref. Total', 'Pred. Total', 'Abs. error']
    n_rows = len(ordered_cases)
    fig, axes = plt.subplots(n_rows, len(column_titles))
    density_axes, error_axes = ([], [])
    for r, (case, (_, row)) in enumerate(zip(ordered_cases, ordered_df.iterrows())):
        axes[r, 0].imshow(case['rgb'])
        section_prefix = 'Best' if r < 10 else 'Typical'
        axes[r, 0].set_ylabel(f"{int(row['copy_count'])} objects", rotation=0, va='center')
        axes[r, 0].text(0.02, 0.96, section_prefix, transform=axes[r, 0].transAxes, ha='left', va='top')
        style_image_axis(axes[r, 0])
        gt = np.asarray(case['gt'], dtype=np.float32)
        pred = np.asarray(case['pred'], dtype=np.float32)
        total_error = np.abs(pred[3] - gt[3])
        maps = [gt[0], pred[0], gt[1], pred[1], gt[2], pred[2], gt[3], pred[3], total_error]
        counts = [float(gt[0].sum()), float(pred[0].sum()), float(gt[1].sum()), float(pred[1].sum()), float(gt[2].sum()), float(pred[2].sum()), float(gt[3].sum()), float(pred[3].sum()), None]
        for c, (array, count) in enumerate(zip(maps, counts), start=1):
            ax = axes[r, c]
            if c == 9:
                ax.imshow(array, vmin=0.0, vmax=error_vmax, interpolation='nearest')
                error_axes.append(ax)
            else:
                ax.imshow(array, vmin=0.0, vmax=density_vmax, interpolation='nearest')
                density_axes.append(ax)
                if count is not None:
                    draw_count_text(ax, count)
            style_image_axis(ax)
    for c, title in enumerate(column_titles):
        axes[0, c].set_title(title)
    fig.text(0.042, 0.97, 'Best', ha='left', va='top')
    fig.text(0.042, 0.52, 'Typical', ha='left', va='top')
    fig.subplots_adjust(left=0.082, right=0.992, top=0.97, bottom=0.078, wspace=0.032, hspace=0.035)
    fig.canvas.draw()
    sep_upper_bottom = axes[9, 0].get_position().y0
    sep_lower_top = axes[10, 0].get_position().y1
    y_sep = 0.5 * (sep_upper_bottom + sep_lower_top)
    fig.add_artist(mpl.lines.Line2D([0.028, 0.992], [y_sep, y_sep], transform=fig.transFigure))
    map_bottom = min((ax.get_position().y0 for ax in density_axes + error_axes))
    colorbar_height = 0.01
    colorbar_gap = 0.01
    colorbar_y = max(0.01, map_bottom - colorbar_gap - colorbar_height)
    add_dual_colorbars(fig, density_axes, error_axes, density_vmax=density_vmax, error_vmax=error_vmax)
    save_pdf_png(fig, path)

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    appendix_dir = output_dir / 'appendix'
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.generate_appendix:
        appendix_dir.mkdir(parents=True, exist_ok=True)
    items = discover_seed_results(Path(args.baseline_test_root))
    seed, frame_csv = choose_representative_seed(items)
    frame_df = prepare_frame_metrics(pd.read_csv(frame_csv))
    main_selected = select_main_cases(frame_df)
    appendix_selected = select_appendix_cases(frame_df) if args.generate_appendix else pd.DataFrame(columns=main_selected.columns)
    all_selected = pd.concat([main_selected, appendix_selected], ignore_index=True, sort=False)
    all_selected['case_key'] = all_selected.apply(lambda row: f"{int(row['copy_count'])}|{row['sample_name']}|{int(row['frame_index'])}", axis=1)
    unique_selected = all_selected.drop_duplicates('case_key').reset_index(drop=True)
    train_root = Path(args.baseline_train_root)
    a0_root = train_root / 'A0_full_socds' if (train_root / 'A0_full_socds').exists() else train_root
    checkpoint_path = a0_root / f'seed_{seed}' / 'model_best.pth.tar'
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output_lookup = run_unique_cases(unique_selected, checkpoint_path, args)
    main_cases = attach_outputs(main_selected, output_lookup)
    appendix_cases = attach_outputs(appendix_selected, output_lookup) if args.generate_appendix else []
    all_rendered_cases = list(main_cases) + list(appendix_cases)
    density_vmax = compute_density_vmax(all_rendered_cases, args.density_percentile)
    error_vmax = compute_error_vmax(all_rendered_cases, args.error_percentile)
    main_manifest = main_selected.copy()
    main_manifest.insert(1, 'representative_seed', seed)
    main_manifest.to_csv(output_dir / 'qualitative_main_case_manifest.csv', index=False, encoding='utf-8-sig')
    if args.generate_appendix:
        appendix_manifest = appendix_selected.copy()
        appendix_manifest.insert(1, 'representative_seed', seed)
        appendix_manifest.to_csv(output_dir / 'qualitative_appendix_case_manifest.csv', index=False, encoding='utf-8-sig')
    save_main_figure(main_cases, output_dir / 'fig_qualitative_reconstruction.pdf', density_vmax=density_vmax, error_vmax=error_vmax)
    if args.generate_appendix:
        save_appendix_combined_figure(appendix_selected, output_lookup, appendix_dir / 'fig_appendix_qualitative_all_scales.pdf', density_vmax=density_vmax, error_vmax=error_vmax)
    manifest = {'representative_seed': seed, 'main_case_selection': ['minimum joint low-error score within the 100-object configuration', 'minimum robust distance to the three within-scale medians over the full test set'], 'appendix_case_selection': 'one best / low-error and one typical / median-level frame per population scale; rendered as one combined 20-row appendix figure', 'shared_density_scale': True, 'shared_error_scale': True, 'density_percentile': float(args.density_percentile), 'error_percentile': float(args.error_percentile), 'density_vmax': float(density_vmax), 'error_vmax': float(error_vmax), 'appendix_population_scales': sorted(appendix_selected['copy_count'].unique().tolist()) if args.generate_appendix and (not appendix_selected.empty) else []}
    (output_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
