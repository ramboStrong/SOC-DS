import os
import re
import numpy as np
from PIL import Image
import torch


RGB_SIZE = (640, 360)
DENSITY_SIZE = (80, 45)
NUM_LAYERS = 3


def parse_frame_index(filename: str) -> int:
    base = os.path.basename(filename)
    match = re.match(r"frame_(\d+)\.(png|jpg|jpeg|bmp)$", base, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse frame index from filename: {filename}")
    return int(match.group(1))


def resolve_sample_dir(frame_path: str) -> str:
    parent_dir = os.path.dirname(frame_path)
    parent_name = os.path.basename(parent_dir)
    if parent_name == "star_enhanced":
        return os.path.dirname(parent_dir)
    return parent_dir


def resolve_rgb_path(
    sample_dir: str,
    frame_index: int,
    use_star_enhanced: bool = True,
    fallback_to_raw: bool = True,
) -> str:
    frame_name = f"frame_{frame_index:03d}.png"

    if use_star_enhanced:
        enhanced_path = os.path.join(sample_dir, "star_enhanced", frame_name)
        if os.path.exists(enhanced_path):
            return enhanced_path
        if not fallback_to_raw:
            raise FileNotFoundError(f"star-enhanced image not found: {enhanced_path}")

    raw_path = os.path.join(sample_dir, frame_name)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw image not found: {raw_path}")
    return raw_path


def load_RAFT_image(image_path: str):
    img = Image.open(image_path).convert("RGB").resize(RGB_SIZE)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.unsqueeze(0)


def load_rgb_as_pil(image_path: str, target_size=RGB_SIZE):
    return Image.open(image_path).convert("RGB").resize(target_size)


def load_layer_density_map(npy_path: str, num_layers: int = NUM_LAYERS):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"stratified density map not found: {npy_path}")

    arr = np.load(npy_path).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"layer density must be 3D; current shape={arr.shape}, path={npy_path}")

    if arr.shape[0] == num_layers:
        return arr

    if arr.shape[-1] == num_layers:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)

    raise ValueError(
        f"unrecognized layer-density channel dimension; expected [3,H,W] or [H,W,3]; "
        f"current shape={arr.shape}, path={npy_path}"
    )


def load_total_density_map(npy_path: str, layer_density: np.ndarray = None):
    if os.path.exists(npy_path):
        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"total density must be [H,W] or [1,H,W]; current shape={arr.shape}, path={npy_path}")
        return arr

    if layer_density is None:
        raise FileNotFoundError(f"total density not found and layer_density was not provided: {npy_path}")

    return layer_density.sum(axis=0).astype(np.float32)


def load_one_channel_map(npy_path: str, map_name: str):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"not found: {map_name}: {npy_path}")

    arr = np.load(npy_path).astype(np.float32)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        pass
    else:
        raise ValueError(
            f"{map_name} must be [H,W] or [1,H,W]; current shape={arr.shape}, path={npy_path}"
        )

    return arr.astype(np.float32)


def make_empty_depth_and_mask():
    h = DENSITY_SIZE[1]
    w = DENSITY_SIZE[0]
    depth = np.zeros((1, h, w), dtype=np.float32)
    mask = np.zeros((1, h, w), dtype=np.float32)
    return depth, mask


def load_data(
    curr_rgb_path: str,
    use_star_enhanced: bool = True,
    use_depth_guidance: bool = True,
    fallback_to_raw: bool = True,
):
    sample_dir = resolve_sample_dir(curr_rgb_path)
    index = parse_frame_index(curr_rgb_path)

    prev_index = max(0, index - 1)

    prev_rgb_path = resolve_rgb_path(
        sample_dir=sample_dir,
        frame_index=prev_index,
        use_star_enhanced=use_star_enhanced,
        fallback_to_raw=fallback_to_raw,
    )

    curr_rgb_path = resolve_rgb_path(
        sample_dir=sample_dir,
        frame_index=index,
        use_star_enhanced=use_star_enhanced,
        fallback_to_raw=fallback_to_raw,
    )

    prev_layer_path = os.path.join(sample_dir, "layer_density_maps", f"frame_{prev_index:03d}.npy")
    curr_layer_path = os.path.join(sample_dir, "layer_density_maps", f"frame_{index:03d}.npy")

    prev_total_path = os.path.join(sample_dir, "total_density_maps", f"frame_{prev_index:03d}.npy")
    curr_total_path = os.path.join(sample_dir, "total_density_maps", f"frame_{index:03d}.npy")

    prev_depth_path = os.path.join(sample_dir, "camera_depth_maps", f"frame_{prev_index:03d}.npy")
    curr_depth_path = os.path.join(sample_dir, "camera_depth_maps", f"frame_{index:03d}.npy")

    prev_depth_mask_path = os.path.join(sample_dir, "camera_depth_masks", f"frame_{prev_index:03d}.npy")
    curr_depth_mask_path = os.path.join(sample_dir, "camera_depth_masks", f"frame_{index:03d}.npy")

    prev_rgb = load_rgb_as_pil(prev_rgb_path, target_size=RGB_SIZE)
    curr_rgb = load_rgb_as_pil(curr_rgb_path, target_size=RGB_SIZE)

    prev_layer_target = load_layer_density_map(prev_layer_path, num_layers=NUM_LAYERS)
    curr_layer_target = load_layer_density_map(curr_layer_path, num_layers=NUM_LAYERS)

    prev_total_target = load_total_density_map(prev_total_path, layer_density=prev_layer_target)
    curr_total_target = load_total_density_map(curr_total_path, layer_density=curr_layer_target)

    if use_depth_guidance:
        prev_depth_target = load_one_channel_map(prev_depth_path, "prev camera_depth_map")
        curr_depth_target = load_one_channel_map(curr_depth_path, "curr camera_depth_map")
        prev_depth_mask = load_one_channel_map(prev_depth_mask_path, "prev camera_depth_mask")
        curr_depth_mask = load_one_channel_map(curr_depth_mask_path, "curr camera_depth_mask")
    else:
        prev_depth_target, prev_depth_mask = make_empty_depth_and_mask()
        curr_depth_target, curr_depth_mask = make_empty_depth_and_mask()

    RAFT_image1 = load_RAFT_image(prev_rgb_path)
    RAFT_image2 = load_RAFT_image(curr_rgb_path)

    return (
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
        index,
    )
