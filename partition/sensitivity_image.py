from __future__ import annotations
from pathlib import Path
from image import RGB_SIZE, load_RAFT_image, load_layer_density_map, load_one_channel_map, load_rgb_as_pil, load_total_density_map, make_empty_depth_and_mask, parse_frame_index, resolve_rgb_path, resolve_sample_dir
from sensitivity_partition import partition_label_dir

def load_partition_data(curr_rgb_path: str, *, partition_id: str, num_strata: int, use_star_enhanced: bool=True, use_depth_guidance: bool=True, fallback_to_raw: bool=True):
    sample_dir = Path(resolve_sample_dir(curr_rgb_path))
    index = parse_frame_index(curr_rgb_path)
    prev_index = max(0, index - 1)
    prev_rgb_path = resolve_rgb_path(str(sample_dir), prev_index, use_star_enhanced=use_star_enhanced, fallback_to_raw=fallback_to_raw)
    curr_rgb_path = resolve_rgb_path(str(sample_dir), index, use_star_enhanced=use_star_enhanced, fallback_to_raw=fallback_to_raw)
    label_root = partition_label_dir(sample_dir, partition_id)
    prev_layer_path = label_root / f'frame_{prev_index:03d}.npy'
    curr_layer_path = label_root / f'frame_{index:03d}.npy'
    prev_total_path = sample_dir / 'total_density_maps' / f'frame_{prev_index:03d}.npy'
    curr_total_path = sample_dir / 'total_density_maps' / f'frame_{index:03d}.npy'
    prev_depth_path = sample_dir / 'camera_depth_maps' / f'frame_{prev_index:03d}.npy'
    curr_depth_path = sample_dir / 'camera_depth_maps' / f'frame_{index:03d}.npy'
    prev_mask_path = sample_dir / 'camera_depth_masks' / f'frame_{prev_index:03d}.npy'
    curr_mask_path = sample_dir / 'camera_depth_masks' / f'frame_{index:03d}.npy'
    prev_rgb = load_rgb_as_pil(prev_rgb_path, target_size=RGB_SIZE)
    curr_rgb = load_rgb_as_pil(curr_rgb_path, target_size=RGB_SIZE)
    prev_layers = load_layer_density_map(str(prev_layer_path), num_layers=num_strata)
    curr_layers = load_layer_density_map(str(curr_layer_path), num_layers=num_strata)
    prev_total = load_total_density_map(str(prev_total_path), layer_density=prev_layers)
    curr_total = load_total_density_map(str(curr_total_path), layer_density=curr_layers)
    if use_depth_guidance:
        prev_depth = load_one_channel_map(str(prev_depth_path), 'prev camera_depth_map')
        curr_depth = load_one_channel_map(str(curr_depth_path), 'curr camera_depth_map')
        prev_mask = load_one_channel_map(str(prev_mask_path), 'prev camera_depth_mask')
        curr_mask = load_one_channel_map(str(curr_mask_path), 'curr camera_depth_mask')
    else:
        prev_depth, prev_mask = make_empty_depth_and_mask()
        curr_depth, curr_mask = make_empty_depth_and_mask()
    return (prev_rgb, curr_rgb, prev_layers, curr_layers, prev_total, curr_total, prev_depth, curr_depth, prev_mask, curr_mask, load_RAFT_image(prev_rgb_path), load_RAFT_image(curr_rgb_path), index)
