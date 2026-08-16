from __future__ import annotations
import os
import random
from typing import Any, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from image import resolve_sample_dir
from sensitivity_image import load_partition_data

class PartitionSensitivityDataset(Dataset):

    def __init__(self, root: Sequence[str], *, partition_id: str, num_strata: int, shuffle: bool=False, transform: Optional[Any]=None, train: bool=False, use_star_enhanced: bool=True, use_depth_guidance: bool=True, fallback_to_raw: bool=True, return_frame_index: bool=True):
        self.lines = list(root)
        if shuffle:
            random.shuffle(self.lines)
        self.partition_id = str(partition_id)
        self.num_strata = int(num_strata)
        if self.num_strata < 2:
            raise ValueError('num_strata must be at least 2 for this sensitivity experiment.')
        self.transform = transform
        self.train = bool(train)
        self.use_star_enhanced = bool(use_star_enhanced)
        self.use_depth_guidance = bool(use_depth_guidance)
        self.fallback_to_raw = bool(fallback_to_raw)
        self.return_frame_index = bool(return_frame_index)

    def __len__(self):
        return len(self.lines)

    @staticmethod
    def _tensor(arr) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.asarray(arr, dtype=np.float32))).float()

    def __getitem__(self, index: int):
        img_path = self.lines[index]
        values = load_partition_data(img_path, partition_id=self.partition_id, num_strata=self.num_strata, use_star_enhanced=self.use_star_enhanced, use_depth_guidance=self.use_depth_guidance, fallback_to_raw=self.fallback_to_raw)
        prev_rgb, curr_rgb, prev_layer, curr_layer, prev_total, curr_total, prev_depth, curr_depth, prev_mask, curr_mask, raft1, raft2, frame_index = values
        if self.transform is not None:
            prev_rgb = self.transform(prev_rgb)
            curr_rgb = self.transform(curr_rgb)
        prev_layer = self._tensor(prev_layer)
        curr_layer = self._tensor(curr_layer)
        prev_total = self._tensor(prev_total)
        curr_total = self._tensor(curr_total)
        prev_depth = self._tensor(prev_depth)
        curr_depth = self._tensor(curr_depth)
        prev_mask = self._tensor(prev_mask)
        curr_mask = self._tensor(curr_mask)
        for name, tensor in [('prev_layer', prev_layer), ('curr_layer', curr_layer)]:
            if tensor.ndim != 3 or tensor.shape[0] != self.num_strata:
                raise ValueError(f'{name} must be [{self.num_strata},H,W], got {tuple(tensor.shape)}; partition={self.partition_id}, path={img_path}')
        for name, tensor in [('prev_total', prev_total), ('curr_total', curr_total)]:
            if tensor.ndim != 2:
                raise ValueError(f'{name} must be [H,W], got {tuple(tensor.shape)}')
        for name, tensor in [('prev_depth', prev_depth), ('curr_depth', curr_depth), ('prev_mask', prev_mask), ('curr_mask', curr_mask)]:
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError(f'{name} must be [1,H,W], got {tuple(tensor.shape)}')
        sample_dir = resolve_sample_dir(img_path)
        output = (prev_rgb, curr_rgb, prev_layer, curr_layer, prev_total, curr_total, prev_depth, curr_depth, prev_mask, curr_mask, raft1, raft2, os.path.basename(sample_dir), f'frame_{frame_index:03d}.png')
        if self.return_frame_index:
            output += (frame_index,)
        return output
