import os
import random
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from image import load_data, resolve_sample_dir


class listDataset(Dataset):

    def __init__(
        self,
        root: Sequence[str],
        shuffle: bool = True,
        transform: Optional[Any] = None,
        transform2: Optional[Any] = None,
        train: bool = False,
        use_star_enhanced: bool = True,
        use_depth_guidance: bool = True,
        fallback_to_raw: bool = True,
        return_frame_index: bool = True,
    ):
        self.lines = list(root)
        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.transform2 = transform2
        self.train = train
        self.use_star_enhanced = use_star_enhanced
        self.use_depth_guidance = use_depth_guidance
        self.fallback_to_raw = fallback_to_raw
        self.return_frame_index = return_frame_index

    def __len__(self):
        return self.nSamples

    @staticmethod
    def _to_float_tensor(arr: np.ndarray) -> torch.Tensor:
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        arr = np.ascontiguousarray(arr.astype(np.float32))
        return torch.from_numpy(arr).float()

    @staticmethod
    def _check_shape(tensor: torch.Tensor, expected_dim: int, name: str):
        if tensor.dim() != expected_dim:
            raise ValueError(
                f"{name} dimension mismatch: expected {expected_dim}D, got shape={tuple(tensor.shape)}"
            )

    def __getitem__(self, index: int):
        assert index < len(self), 'index range error'

        img_path = self.lines[index]

        (
            prev_rgb,
            curr_rgb,
            prev_layer_target,
            curr_layer_target,
            prev_total_target,
            curr_total_target,
            prev_depth_target,
            curr_depth_target,
            prev_depth_mask,
            curr_depth_mask,
            RAFT_image1,
            RAFT_image2,
            frame_index,
        ) = load_data(
            img_path,
            use_star_enhanced=self.use_star_enhanced,
            use_depth_guidance=self.use_depth_guidance,
            fallback_to_raw=self.fallback_to_raw,
        )

        if self.transform is not None:
            prev_rgb = self.transform(prev_rgb)
            curr_rgb = self.transform(curr_rgb)

        prev_layer_target = self._to_float_tensor(prev_layer_target)
        curr_layer_target = self._to_float_tensor(curr_layer_target)
        prev_total_target = self._to_float_tensor(prev_total_target)
        curr_total_target = self._to_float_tensor(curr_total_target)
        prev_depth_target = self._to_float_tensor(prev_depth_target)
        curr_depth_target = self._to_float_tensor(curr_depth_target)
        prev_depth_mask = self._to_float_tensor(prev_depth_mask)
        curr_depth_mask = self._to_float_tensor(curr_depth_mask)

        self._check_shape(prev_layer_target, 3, "prev_layer_target")
        self._check_shape(curr_layer_target, 3, "curr_layer_target")
        self._check_shape(prev_total_target, 2, "prev_total_target")
        self._check_shape(curr_total_target, 2, "curr_total_target")
        self._check_shape(prev_depth_target, 3, "prev_depth_target")
        self._check_shape(curr_depth_target, 3, "curr_depth_target")
        self._check_shape(prev_depth_mask, 3, "prev_depth_mask")
        self._check_shape(curr_depth_mask, 3, "curr_depth_mask")

        sample_dir = resolve_sample_dir(img_path)
        sample_name = os.path.basename(sample_dir)
        frame_name = f"frame_{frame_index:03d}.png"

        output = (
            prev_rgb,
            curr_rgb,
            prev_layer_target,
            curr_layer_target,
            prev_total_target,
            curr_total_target,
            prev_depth_target,
            curr_depth_target,
            prev_depth_mask,
            curr_depth_mask,
            RAFT_image1,
            RAFT_image2,
            sample_name,
            frame_name,
        )

        if self.return_frame_index:
            output = output + (frame_index,)

        return output
