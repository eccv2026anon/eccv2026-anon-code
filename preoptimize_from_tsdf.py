#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import yaml
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.networks.decoders import Decoders


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    print(f"\n{title}")
    print("-" * 80)


def load_config(config_path):
    print_section("Loading config")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if 'model' not in cfg:
        print("WARNING: missing 'model' section in config, creating defaults")
        cfg['model'] = {}

    defaults = {
        'prior_tsdf_path': 'output/ICPARKOSM_generated/Dparking_prior_tsdf.npy',
        'prior_tsdf_origin_xyz': [37.92651181, -3.50169141, -0.24],
        'prior_tsdf_voxel_size': 0.06,
        'truncation': 0.06,
        'c_dim': 32,
        'hidden_size': 64,
        'n_blocks': 2,
        'prior_init_iterations': 2000,
        'prior_init_lr_planes': 0.008,
        'prior_init_samples_per_batch': 32768,
        'use_importance_sampling': True,
        'importance_tau_s': 1.0,
        'importance_epsilon': 1e-3,
        'importance_eta': 0.5,
    }

    missing = []
    for key, val in defaults.items():
        if key not in cfg['model']:
            cfg['model'][key] = val
            missing.append(f"{key} = {val}")

    if missing:
        print("WARNING: using default values for:")
        for item in missing:
            print(f"  {item}")

    return cfg


def load_tsdf(tsdf_path):
    print_section("Loading TSDF prior")

    if not os.path.exists(tsdf_path):
        raise FileNotFoundError(f"TSDF file not found: {tsdf_path}")

    print(f"TSDF file: {tsdf_path}")
    tsdf_volume = np.load(tsdf_path)

    print("INFO: TSDF loaded")
    print(f"  shape: {tsdf_volume.shape}")
    print(f"  dtype: {tsdf_volume.dtype}")
    print(f"  value range: [{tsdf_volume.min():.4f}, {tsdf_volume.max():.4f}]")
    print(f"  file size: {os.path.getsize(tsdf_path) / (1024*1024):.2f} MB")

    return tsdf_volume


def create_interpolator(tsdf_volume, voxel_size, origin):
    print_section("Building TSDF interpolator")

    nz, ny, nx = tsdf_volume.shape

    x_coords = origin[0] + np.arange(nx) * voxel_size
    y_coords = origin[1] + np.arange(ny) * voxel_size
    z_coords = origin[2] + np.arange(nz) * voxel_size

    interpolator = RegularGridInterpolator(
        (z_coords, y_coords, x_coords),
        tsdf_volume,
        method='linear',
        bounds_error=False,
        fill_value=1.0
    )

    print("INFO: interpolator created")
    print(f"  X range: [{x_coords[0]:.2f}, {x_coords[-1]:.2f}]")
    print(f"  Y range: [{y_coords[0]:.2f}, {y_coords[-1]:.2f}]")
    print(f"  Z range: [{z_coords[0]:.2f}, {z_coords[-1]:.2f}]")

    return interpolator, (x_coords, y_coords, z_coords)


def initialize_feature_planes(cfg, device):
    print_section("Initializing feature planes")

    c_dim = cfg['model'].get('c_dim', 32)

    if 'bound' in cfg:
        bound = torch.tensor(cfg['bound'], device=device)
    elif 'mapping' in cfg and 'bound' in cfg['mapping']:
        bound = torch.tensor(cfg['mapping']['bound'], device=device)
    else:
        print("WARNING: scene bound not found, using a parking-lot default")
        bound = torch.tensor([[37.9265, 293.2444], [-3.5017, 262.6937], [-0.2400, 3.0000]], device=device)

    planes_res_coarse = cfg.get('planes_res', {}).get('coarse', 0.24)
    planes_res_fine = cfg.get('planes_res', {}).get('fine', 0.06)

    print(f"c_dim: {c_dim}")
    print(f"bound: {bound.tolist()}")
    print(f"coarse resolution: {planes_res_coarse}")
    print(f"fine resolution: {planes_res_fine}")

    if bound.shape == (3, 2):
        bound = bound.transpose(0, 1)

    bound_range = bound[1] - bound[0]

    def _safe_hw(length, res):
        return max(1, int(length / res))

    h_coarse_xy = _safe_hw(bound_range[1].item(), planes_res_coarse)
    w_coarse_xy = _safe_hw(bound_range[0].item(), planes_res_coarse)
    planes_xy_coarse = torch.nn.Parameter(
        torch.randn(1, c_dim, h_coarse_xy, w_coarse_xy, device=device) * 0.1
    )

    h_fine_xy = _safe_hw(bound_range[1].item(), planes_res_fine)
    w_fine_xy = _safe_hw(bound_range[0].item(), planes_res_fine)
    planes_xy_fine = torch.nn.Parameter(
        torch.randn(1, c_dim, h_fine_xy, w_fine_xy, device=device) * 0.1
    )

    h_coarse_xz = _safe_hw(bound_range[2].item(), planes_res_coarse)
    w_coarse_xz = _safe_hw(bound_range[0].item(), planes_res_coarse)
    planes_xz_coarse = torch.nn.Parameter(
        torch.randn(1, c_dim, h_coarse_xz, w_coarse_xz, device=device) * 0.1
    )

    h_fine_xz = _safe_hw(bound_range[2].item(), planes_res_fine)
    w_fine_xz = _safe_hw(bound_range[0].item(), planes_res_fine)
    planes_xz_fine = torch.nn.Parameter(
        torch.randn(1, c_dim, h_fine_xz, w_fine_xz, device=device) * 0.1
    )

    h_coarse_yz = _safe_hw(bound_range[2].item(), planes_res_coarse)
    w_coarse_yz = _safe_hw(bound_range[1].item(), planes_res_coarse)
    planes_yz_coarse = torch.nn.Parameter(
        torch.randn(1, c_dim, h_coarse_yz, w_coarse_yz, device=device) * 0.1
    )

    h_fine_yz = _safe_hw(bound_range[2].item(), planes_res_fine)
    w_fine_yz = _safe_hw(bound_range[1].item(), planes_res_fine)
    planes_yz_fine = torch.nn.Parameter(
        torch.randn(1, c_dim, h_fine_yz, w_fine_yz, device=device) * 0.1
    )

    planes = {
        'xy_coarse': planes_xy_coarse,
        'xy_fine': planes_xy_fine,
        'xz_coarse': planes_xz_coarse,
        'xz_fine': planes_xz_fine,
        'yz_coarse': planes_yz_coarse,
        'yz_fine': planes_yz_fine,
    }

    return planes, bound


def initialize_decoder(cfg, device):
    print_section("Initializing decoder")

    c_dim = cfg['model'].get('c_dim', 32)
    hidden_size = cfg['model'].get('hidden_size', 64)
    n_blocks = cfg['model'].get('n_blocks', 2)
    truncation = cfg['model'].get('truncation', 0.06)
    learnable_beta = cfg.get('rendering', {}).get('learnable_beta', True)

    decoder = Decoders(
        c_dim=c_dim,
        hidden_size=hidden_size,
        truncation=truncation,
        n_blocks=n_blocks,
        learnable_beta=learnable_beta,
        device=device,
    ).to(device)

    decoder.use_block_manager = False
    decoder.block_manager = None

    if 'bound' in cfg:
        decoder.bound = torch.tensor(cfg['bound'], device=device)
    elif 'mapping' in cfg and 'bound' in cfg['mapping']:
        decoder.bound = torch.tensor(cfg['mapping']['bound'], device=device)
    else:
        decoder.bound = torch.tensor([[37.9265, 293.2444], [-3.5017, 262.6937], [-0.2400, 3.0000]], device=device)

    if decoder.bound.shape == (3, 2):
        decoder.bound = decoder.bound.transpose(0, 1)

    for param in decoder.parameters():
        param.requires_grad = False

    return decoder


def compute_prior_gradient(interpolator, point, voxel_size, epsilon=1e-6):
    point_np = point.detach().cpu().numpy()
    grad = np.zeros(3)
    for i in range(3):
        p_plus = point_np.copy()
        p_plus[i] += epsilon
        val_plus = interpolator((p_plus[2], p_plus[1], p_plus[0]))

        p_minus = point_np.copy()
        p_minus[i] -= epsilon
        val_minus = interpolator((p_minus[2], p_minus[1], p_minus[0]))

        grad[i] = (val_plus - val_minus) / (2 * epsilon)

    return torch.from_numpy(grad).float()


def is_point_in_tsdf_bounds(point, coords):
    x_coords, y_coords, z_coords = coords
    x, y, z = point
    return (x_coords[0] <= x <= x_coords[-1] and
            y_coords[0] <= y <= y_coords[-1] and
            z_coords[0] <= z <= z_coords[-1])


def compute_importance_weights(points, interpolator, coords, voxel_size, tau_s=1.0, epsilon=1e-3, eta=0.5):
    weights = []
    for point in points:
        if is_point_in_tsdf_bounds(point.cpu().numpy(), coords):
            point_np = point.cpu().numpy()
            phi_prior = interpolator((point_np[2], point_np[1], point_np[0]))
            grad_prior = compute_prior_gradient(interpolator, point, voxel_size)
            grad_norm = torch.norm(grad_prior)
            weight = (np.exp(-abs(phi_prior) / tau_s) * np.power(epsilon + grad_norm.item(), eta))
        else:
            weight = 0.0
        weights.append(weight)

    return torch.tensor(weights, device=points.device, dtype=torch.float32)


def sample_points(bound, num_samples, coords, device, use_importance_sampling=True, importance_sampling_params=None):
    x_coords, y_coords, z_coords = coords
    tsdf_min = torch.tensor([x_coords[0], y_coords[0], z_coords[0]], device=device)
    tsdf_max = torch.tensor([x_coords[-1], y_coords[-1], z_coords[-1]], device=device)

    sample_min = torch.max(bound[0], tsdf_min)
    sample_max = torch.min(bound[1], tsdf_max)

    if not use_importance_sampling or importance_sampling_params is None:
        points = torch.rand(num_samples, 3, device=device)
        return sample_min + points * (sample_max - sample_min)

    interpolator = importance_sampling_params['interpolator']
    voxel_size = importance_sampling_params['voxel_size']
    tau_s = importance_sampling_params.get('tau_s', 1.0)
    epsilon = importance_sampling_params.get('epsilon', 1e-3)
    eta = importance_sampling_params.get('eta', 0.5)

    candidate_multiplier = 10
    n_candidates = num_samples * candidate_multiplier

    candidate_points = torch.rand(n_candidates, 3, device=device)
    candidate_points = sample_min + candidate_points * (sample_max - sample_min)

    weights = compute_importance_weights(candidate_points, interpolator, coords, voxel_size, tau_s, epsilon, eta)

    if weights.sum() <= 0:
        points = torch.rand(num_samples, 3, device=device)
        return sample_min + points * (sample_max - sample_min)

    weights = weights / weights.sum()

    replacement = num_samples > weights.numel()
    try:
        selected_indices = torch.multinomial(weights, num_samples, replacement=replacement)
        return candidate_points[selected_indices]
    except RuntimeError:
        points = torch.rand(num_samples, 3, device=device)
        return sample_min + points * (sample_max - sample_min)


def query_decoder_sdf(decoder, planes, points):
    all_planes = (
        [planes['xy_coarse'], planes['xy_fine']],
        [planes['xz_coarse'], planes['xz_fine']],
        [planes['yz_coarse'], planes['yz_fine']],
        [], [], []
    )
    pred = decoder(points, all_planes, need_rgb=False)
    return pred['sdf'].reshape(-1)


def huber_loss(x, delta):
    abs_x = torch.abs(x)
    return torch.where(abs_x < delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))


def train_iteration(decoder, planes, bound, interpolator, coords, optimizer, num_samples, device,
                    truncation, voxel_size, use_importance_sampling=True, tau_s=1.0, epsilon=1e-3, eta=0.5):
    if use_importance_sampling:
        importance_params = {
            'interpolator': interpolator,
            'voxel_size': voxel_size,
            'tau_s': tau_s,
            'epsilon': epsilon,
            'eta': eta
        }
        points = sample_points(bound, num_samples, coords, device, True, importance_params)
    else:
        points = sample_points(bound, num_samples, coords, device, False)

    points_np = points.detach().cpu().numpy()
    points_zyx = points_np[:, [2, 1, 0]]
    prior_sdf = interpolator(points_zyx)
    prior_sdf = torch.from_numpy(prior_sdf).float().to(device)

    if prior_sdf.abs().max() > 1.5:
        prior_sdf = prior_sdf / max(truncation, 1e-6)
    prior_sdf = torch.clamp(prior_sdf, -1.0, 1.0)

    pred_sdf = query_decoder_sdf(decoder, planes, points)

    valid_mask = torch.tensor([
        is_point_in_tsdf_bounds(p, coords) for p in points_np
    ], device=device, dtype=torch.float32)

    loss = huber_loss(pred_sdf - prior_sdf, delta=0.1)
    loss = (loss * valid_mask).sum() / (valid_mask.sum().clamp_min(1.0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def optimize(cfg, decoder, planes, bound, interpolator, coords, device, args):
    print_section("Starting pre-optimization")

    iterations = args.iterations or cfg['model'].get('prior_init_iterations', 2000)
    lr = args.lr or cfg['model'].get('prior_init_lr_planes', 0.008)
    batch_size = args.batch_size or cfg['model'].get('prior_init_samples_per_batch', 32768)
    truncation = cfg['model'].get('truncation', 0.06)

    use_importance_sampling = cfg['model'].get('use_importance_sampling', True)
    tau_s = cfg['model'].get('importance_tau_s', 1.0)
    epsilon = cfg['model'].get('importance_epsilon', 1e-3)
    eta = cfg['model'].get('importance_eta', 0.5)
    voxel_size = cfg['model'].get('prior_tsdf_voxel_size', 0.06)

    optimizer = torch.optim.Adam(list(planes.values()), lr=lr)

    start_time = time.time()
    losses = []
    for i in range(iterations):
        loss = train_iteration(
            decoder, planes, bound, interpolator, coords, optimizer, batch_size, device,
            truncation, voxel_size, use_importance_sampling, tau_s, epsilon, eta
        )
        losses.append(loss)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_loss = np.mean(losses[-100:])
            print(f"  Iter {i + 1:4d}/{iterations}, loss: {loss:.6f}, avg(100): {avg_loss:.6f}, time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    final_loss = np.mean(losses[-100:])

    print("\nINFO: Pre-optimization finished")
    print(f"  total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  final loss: {final_loss:.6f}")
    print(f"  initial loss: {losses[0]:.6f}")

    return losses


def save_results(planes, decoder, cfg, args):
    print_section("Saving results")

    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        planes_path = os.path.join(output_dir, "preoptimized_geometry_planes.pth")
    else:
        planes_path = cfg['model'].get('preoptimized_planes_path', 'output/preoptimized_geometry_planes.pth')
        output_dir = os.path.dirname(planes_path)
        os.makedirs(output_dir, exist_ok=True)

    planes_to_save = {
        'geo_planes_xy_coarse': planes['xy_coarse'].cpu().data,
        'geo_planes_xy_fine': planes['xy_fine'].cpu().data,
        'geo_planes_xz_coarse': planes['xz_coarse'].cpu().data,
        'geo_planes_xz_fine': planes['xz_fine'].cpu().data,
        'geo_planes_yz_coarse': planes['yz_coarse'].cpu().data,
        'geo_planes_yz_fine': planes['yz_fine'].cpu().data,
        'structrecon_bound_at_save_time': decoder.bound.cpu().numpy() if decoder.bound is not None else None,
        'is_block_mode': False,
        'save_timestamp': time.time(),
        'source': 'preoptimize_from_tsdf.py',
    }

    torch.save(planes_to_save, planes_path)
    print(f"INFO: saved planes -> {planes_path}")

    decoder_path = planes_path.replace('.pth', '_decoder.pth')
    decoder_checkpoint = {
        'decoder_state_dict': decoder.state_dict(),
        'decoder_config': {
            'c_dim': decoder.c_dim,
            'truncation': decoder.truncation,
            'n_blocks': decoder.n_blocks,
            'hidden_size': decoder.linears[0].out_features if len(decoder.linears) > 0 else 64,
        },
        'bound': decoder.bound.cpu().numpy() if decoder.bound is not None else None,
        'save_timestamp': time.time(),
        'source': 'preoptimize_from_tsdf.py',
    }

    torch.save(decoder_checkpoint, decoder_path)
    print(f"INFO: saved decoder -> {decoder_path}")

    info_path = planes_path.replace('.pth', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Prior-guided pre-optimization info\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"TSDF: {cfg['model']['prior_tsdf_path']}\n\n")
        f.write("Training:\n")
        f.write(f"  iterations: {args.iterations or cfg['model'].get('prior_init_iterations', 2000)}\n")
        f.write(f"  lr: {args.lr or cfg['model'].get('prior_init_lr_planes', 0.008)}\n")
        f.write(f"  batch_size: {args.batch_size or cfg['model'].get('prior_init_samples_per_batch', 32768)}\n\n")
        f.write("Decoder:\n")
        f.write(f"  n_blocks: {decoder_checkpoint['decoder_config']['n_blocks']}\n")
        f.write(f"  hidden_size: {decoder_checkpoint['decoder_config']['hidden_size']}\n")
        f.write(f"  c_dim: {decoder_checkpoint['decoder_config']['c_dim']}\n")
        f.write(f"  truncation: {decoder_checkpoint['decoder_config']['truncation']}\n\n")
        f.write("Outputs:\n")
        f.write(f"  planes: {planes_path}\n")
        f.write(f"  decoder: {decoder_path}\n")
        f.write(f"  info: {info_path}\n")

    print(f"INFO: saved info -> {info_path}")

    return planes_path, decoder_path


def main():
    parser = argparse.ArgumentParser(
        description='Pre-optimize geometry planes from a TSDF prior.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preoptimize_from_tsdf.py --config configs/structrecon.yaml
  python preoptimize_from_tsdf.py --config configs/structrecon.yaml --iterations 5000
  python preoptimize_from_tsdf.py --config configs/structrecon.yaml --output output/custom/
  python preoptimize_from_tsdf.py --config configs/structrecon.yaml --device cuda:1
"""
    )

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--iterations', type=int, default=None, help='Override iterations')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device string (default: cuda:0)')

    args = parser.parse_args()

    print_header("Prior-guided pre-optimization")
    print(f"Script: preoptimize_from_tsdf.py")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if 'cuda' in args.device:
            if not torch.cuda.is_available():
                print("WARNING: CUDA is not available, falling back to CPU")
                args.device = 'cpu'
            else:
                print(f"INFO: using device {args.device}")

        device = torch.device(args.device)

        cfg = load_config(args.config)

        tsdf_path = cfg['model']['prior_tsdf_path']
        tsdf_volume = load_tsdf(tsdf_path)

        voxel_size = cfg['model']['prior_tsdf_voxel_size']
        origin = cfg['model']['prior_tsdf_origin_xyz']
        interpolator, coords = create_interpolator(tsdf_volume, voxel_size, origin)

        planes, bound = initialize_feature_planes(cfg, device)
        decoder = initialize_decoder(cfg, device)

        optimize(cfg, decoder, planes, bound, interpolator, coords, device, args)
        planes_path, decoder_path = save_results(planes, decoder, cfg, args)

        print_header("Pre-optimization finished")
        print("Outputs:")
        print(f"  1) planes: {planes_path}")
        print(f"  2) decoder: {decoder_path}")
        print(f"  3) info: {planes_path.replace('.pth', '_info.txt')}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
