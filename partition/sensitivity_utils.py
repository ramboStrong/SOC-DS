from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from experiment_utils import build_boundary_mask, estimate_raft_pair, masked_mse, propagation_consistency_loss, reconstruct_density, split_transport

def reconstruct_stratified(propagation: torch.Tensor, boundary_mask: torch.Tensor, *, num_strata: int, channels_per_stratum: int=10, inverse: bool=False) -> torch.Tensor:
    layers = split_transport(propagation, num_layers=int(num_strata), channels_per_layer=int(channels_per_stratum))
    return torch.stack([reconstruct_density(x, boundary_mask, inverse=inverse) for x in layers], dim=0)

def compute_transport_outputs_and_loss(model, data: Dict, raft, args, criterion_sum: nn.Module, device, padder_cls, *, num_strata: int, channels_per_stratum: int=10):
    flow_fwd, flow_bwd = estimate_raft_pair(raft, data['raft_image1'], data['raft_image2'], padder_cls, args.flow_mode, args.raft_iters, device)
    output_curr = model(data['prev_rgb'], data['curr_rgb'], flow_fwd)
    output_prev = model(data['curr_rgb'], data['prev_rgb'], flow_bwd)
    propagation_fwd = output_curr.prediction
    propagation_bwd = output_prev.prediction
    boundary = build_boundary_mask(propagation_fwd.shape[-2:], propagation_fwd.device, propagation_fwd.dtype)
    curr_forward = reconstruct_stratified(propagation_fwd, boundary, num_strata=num_strata, channels_per_stratum=channels_per_stratum, inverse=False)
    curr_backward = reconstruct_stratified(propagation_bwd, boundary, num_strata=num_strata, channels_per_stratum=channels_per_stratum, inverse=True)
    prev_from_forward = reconstruct_stratified(propagation_fwd, boundary, num_strata=num_strata, channels_per_stratum=channels_per_stratum, inverse=True)
    curr_gt = data['curr_layer'][0]
    prev_gt = data['prev_layer'][0]
    curr_total_gt = data['curr_total'][0]
    prev_total_gt = data['prev_total'][0]
    raw_layer_loss = criterion_sum(curr_forward, curr_gt) + criterion_sum(curr_backward, curr_gt) + criterion_sum(prev_from_forward, prev_gt)
    layer_loss = raw_layer_loss * (3.0 / float(num_strata))
    total_loss_term = criterion_sum(curr_forward.sum(dim=0), curr_total_gt) + criterion_sum(curr_backward.sum(dim=0), curr_total_gt) + criterion_sum(prev_from_forward.sum(dim=0), prev_total_gt)
    consistency_loss = propagation_consistency_loss(propagation_fwd, propagation_bwd, criterion_sum, num_layers=num_strata, channels_per_layer=channels_per_stratum)
    depth_loss = masked_mse(output_curr.depth, data['curr_depth'], data['curr_mask']) + masked_mse(output_prev.depth, data['prev_depth'], data['prev_mask'])
    pred_layers = (curr_forward + curr_backward) / 2.0
    pred_total = pred_layers.sum(dim=0)
    losses = {'layer': layer_loss, 'total': total_loss_term, 'consistency': consistency_loss, 'depth': depth_loss}
    loss = args.lambda_layer * layer_loss + args.lambda_total * total_loss_term + args.lambda_consistency * consistency_loss + args.lambda_depth * depth_loss
    pred = {'layers': pred_layers, 'total': pred_total, 'count': pred_total.sum(), 'propagation_fwd': propagation_fwd, 'propagation_bwd': propagation_bwd}
    return (loss, losses, pred)

def allocation_components(gt_layers: np.ndarray, pred_layers: np.ndarray, eps: float=1e-08) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.clip(np.asarray(gt_layers, dtype=np.float64), 0.0, None)
    pred = np.clip(np.asarray(pred_layers, dtype=np.float64), 0.0, None)
    if gt.ndim != 3 or pred.shape != gt.shape:
        raise ValueError(f'Allocation expects matching [K,H,W] arrays, got {gt.shape}, {pred.shape}')
    k = gt.shape[0]
    fractions = pred / (pred.sum(axis=0, keepdims=True) + eps)
    numerator = np.zeros((k, k), dtype=np.float64)
    denominator = gt.sum(axis=(1, 2)).astype(np.float64)
    for i in range(k):
        for j in range(k):
            numerator[i, j] = float(np.sum(gt[i] * fractions[j]))
    matrix = numerator / (denominator[:, None] + eps)
    return (matrix, numerator, denominator)

def flatten_matrix(prefix: str, matrix: np.ndarray) -> Dict[str, float]:
    k = int(matrix.shape[0])
    return {f'{prefix}_s{i}_to_s{j}': float(matrix[i, j]) for i in range(k) for j in range(k)}

def add_normalized_gaussian(density_map: np.ndarray, center_x: float, center_y: float, *, sigma: float, radius: int) -> None:
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
    kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma ** 2))
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    density_map[y0:y1 + 1, x0:x1 + 1] += kernel.astype(np.float32)

class FramePositionCache:

    def __init__(self):
        self._cache: Dict[Path, Dict[int, List[Tuple[float, float, float]]]] = {}

    def get(self, sample_dir: str | Path, frame_index: int) -> List[Tuple[float, float, float]]:
        sample_dir = Path(sample_dir)
        if sample_dir not in self._cache:
            csv_path = sample_dir / 'frame_positions.csv'
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            frames: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
            with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    try:
                        frame = int(float(row.get('frame', -1)))
                        visible = int(float(row.get('visible', 0)))
                        x = float(row.get('pixel_x', 'nan'))
                        y = float(row.get('pixel_y', 'nan'))
                        depth = float(row.get('camera_depth', 'nan'))
                    except (TypeError, ValueError):
                        continue
                    if visible == 1 and frame >= 0 and np.isfinite([x, y, depth]).all():
                        frames[frame].append((x, y, depth))
            self._cache[sample_dir] = dict(frames)
        return self._cache[sample_dir].get(int(frame_index), [])

def parse_bin_edges(text: str) -> Tuple[float, ...]:
    values = []
    for token in str(text).split(','):
        token = token.strip().lower()
        if token in {'inf', '+inf', 'infinity', '+infinity'}:
            values.append(float('inf'))
        else:
            values.append(float(token))
    if len(values) < 2 or values[0] != 0.0 or any((b <= a for a, b in zip(values[:-1], values[1:]))):
        raise ValueError(f'Invalid boundary-bin edges: {values}')
    if not np.isinf(values[-1]):
        raise ValueError('The final boundary-bin edge must be inf.')
    return tuple(values)

def boundary_bin_labels(edges: Sequence[float]) -> List[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append(f'{lo:g}+')
        else:
            labels.append(f'{lo:g}-{hi:g}')
    return labels

def build_boundary_bin_density(targets: Iterable[Tuple[float, float, float]], *, thresholds: Sequence[float], num_strata: int, bin_edges: Sequence[float], image_size: Tuple[int, int], density_size: Tuple[int, int], sigma: float, radius: int) -> np.ndarray:
    image_w, image_h = image_size
    density_w, density_h = density_size
    sx = image_w / float(density_w)
    sy = image_h / float(density_h)
    out = np.zeros((len(bin_edges) - 1, num_strata, density_h, density_w), dtype=np.float32)
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    for x, y, depth in targets:
        if not (0 <= x < image_w and 0 <= y < image_h):
            continue
        stratum = int(np.searchsorted(thresholds_arr, depth, side='right'))
        delta = float(np.min(np.abs(thresholds_arr - depth)))
        bin_idx = int(np.searchsorted(np.asarray(bin_edges[1:]), delta, side='right'))
        bin_idx = min(bin_idx, len(bin_edges) - 2)
        gx = min(max(x / sx, 0.0), density_w - 1e-06)
        gy = min(max(y / sy, 0.0), density_h - 1e-06)
        add_normalized_gaussian(out[bin_idx, stratum], gx, gy, sigma=float(sigma), radius=int(radius))
    return out
