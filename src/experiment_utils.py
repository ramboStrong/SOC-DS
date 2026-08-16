from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_boundary_mask(spatial_shape, device, dtype):
    h, w = spatial_shape
    mask = torch.zeros((h, w), device=device, dtype=dtype)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    mask[:, 0] = 1.0
    mask[:, -1] = 1.0
    return mask


def reconstruct_density(flow: torch.Tensor, boundary_mask: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[0] != 1 or flow.shape[1] != 10:
        raise ValueError(f"Expected flow [1,10,H,W], got {tuple(flow.shape)}")
    if inverse:
        return torch.sum(flow[0, :9], dim=0) + flow[0, 9] * boundary_mask
    return (
        F.pad(flow[0, 0, 1:, 1:], (0, 1, 0, 1))
        + F.pad(flow[0, 1, 1:, :], (0, 0, 0, 1))
        + F.pad(flow[0, 2, 1:, :-1], (1, 0, 0, 1))
        + F.pad(flow[0, 3, :, 1:], (0, 1, 0, 0))
        + flow[0, 4]
        + F.pad(flow[0, 5, :, :-1], (1, 0, 0, 0))
        + F.pad(flow[0, 6, :-1, 1:], (0, 1, 1, 0))
        + F.pad(flow[0, 7, :-1, :], (0, 0, 1, 0))
        + F.pad(flow[0, 8, :-1, :-1], (1, 0, 1, 0))
        + flow[0, 9] * boundary_mask
    )


def split_transport(propagation: torch.Tensor, num_layers: int = 3, channels_per_layer: int = 10):
    expected = num_layers * channels_per_layer
    if propagation.ndim != 4 or propagation.shape[1] != expected:
        raise ValueError(f"Expected propagation [B,{expected},H,W], got {tuple(propagation.shape)}")
    return [
        propagation[:, layer * channels_per_layer : (layer + 1) * channels_per_layer]
        for layer in range(num_layers)
    ]


def reconstruct_stratified(propagation: torch.Tensor, boundary_mask: torch.Tensor, inverse: bool = False):
    return torch.stack(
        [reconstruct_density(layer_flow, boundary_mask, inverse=inverse) for layer_flow in split_transport(propagation)],
        dim=0,
    )


def propagation_consistency_loss(forward_prop, backward_prop, criterion_sum, num_layers=3, channels_per_layer=10):
    total = forward_prop.new_tensor(0.0)
    for layer_idx in range(num_layers):
        offset = layer_idx * channels_per_layer
        f = forward_prop
        g = backward_prop
        layer_loss = (
            criterion_sum(f[0, offset + 0, 1:, 1:], g[0, offset + 8, :-1, :-1])
            + criterion_sum(f[0, offset + 1, 1:, :], g[0, offset + 7, :-1, :])
            + criterion_sum(f[0, offset + 2, 1:, :-1], g[0, offset + 6, :-1, 1:])
            + criterion_sum(f[0, offset + 3, :, 1:], g[0, offset + 5, :, :-1])
            + criterion_sum(f[0, offset + 4], g[0, offset + 4])
            + criterion_sum(f[0, offset + 5, :, :-1], g[0, offset + 3, :, 1:])
            + criterion_sum(f[0, offset + 6, :-1, 1:], g[0, offset + 2, 1:, :-1])
            + criterion_sum(f[0, offset + 7, :-1, :], g[0, offset + 1, 1:, :])
            + criterion_sum(f[0, offset + 8, :-1, :-1], g[0, offset + 0, 1:, 1:])
        )
        total = total + layer_loss
    return total / float(num_layers)


def masked_mse(pred: Optional[torch.Tensor], target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    if pred is None:
        return target.new_tensor(0.0)
    denom = mask.sum()
    if float(denom.detach().cpu()) <= eps:
        return target.new_tensor(0.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def prepare_raft_images(image1: torch.Tensor, image2: torch.Tensor):
    if image1.ndim == 5 and image1.shape[1] == 1:
        image1 = image1[:, 0]
    if image2.ndim == 5 and image2.shape[1] == 1:
        image2 = image2[:, 0]
    return image1, image2


def estimate_raft_pair(raft, image1, image2, padder_cls, flow_mode: str, iters: int, device):
    image1, image2 = prepare_raft_images(image1, image2)
    image1 = image1.to(device)
    image2 = image2.to(device)
    padder = padder_cls(image1.shape)
    image1_pad, image2_pad = padder.pad(image1, image2)
    with torch.no_grad():
        _, flow_fwd = raft(image1_pad, image2_pad, iters=iters, test_mode=True)
        if flow_mode == "bidirectional":
            _, flow_bwd = raft(image2_pad, image1_pad, iters=iters, test_mode=True)
        elif flow_mode == "shared_forward":
            flow_bwd = flow_fwd
        else:
            raise ValueError(f"Unsupported flow_mode: {flow_mode}")
    return flow_fwd, flow_bwd


def count_from_density(density: torch.Tensor) -> torch.Tensor:
    return density.sum(dim=(-2, -1))


def model_requires_raft(spec: Dict[str, object]) -> bool:
    return bool(spec.get("use_flow_conditioning", False))


def selection_score(metrics: Dict[str, float], family: str, metric_name: str, total_weight: float):
    if family in {"transport", "direct_stratified"}:
        if metric_name == "stratum_mae":
            return metrics["val_stratum_mae"]
        if metric_name == "total_mae":
            return metrics["val_total_mae"]
        if metric_name == "composite":
            return metrics["val_stratum_mae"] + total_weight * metrics["val_total_mae"]
        raise ValueError(metric_name)
    return metrics["val_total_mae"]
