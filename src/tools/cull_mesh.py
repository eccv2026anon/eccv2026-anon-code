import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.utils.datasets import get_dataset
from src import config
import os

def cull_mesh(mesh_file, cfg, args, device, estimate_c2w_list=None):

    frame_reader = get_dataset(cfg, args, 1, device=device)

    eval_rec = cfg['meshing']['eval_rec']
    truncation = cfg['model']['truncation']
    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

    if estimate_c2w_list is not None:
        n_imgs = len(estimate_c2w_list)
    else:
        n_imgs = len(frame_reader)

    mesh = trimesh.load(mesh_file, process=False)
    pc = mesh.vertices

                                            
    use_block_mode = False
    blocks = []
    try:
        from src.utils.BlockManager import BlockManager
        bm = BlockManager(cfg, device, bound=torch.tensor(cfg['mapping']['bound']).to(device) if 'mapping' in cfg and 'bound' in cfg['mapping'] else None)
        if bm.presegmented_blocks_dir and os.path.isdir(bm.presegmented_blocks_dir):
            blk_files = [f for f in os.listdir(bm.presegmented_blocks_dir) if f.startswith('block_') and f.endswith('.pth')]
            if len(blk_files) > 0:
                use_block_mode = True
                for fname in blk_files:
                    try:
                        base = os.path.splitext(fname)[0]
                        parts = base.split('_')
                        bi, bj, bk = int(parts[1]), int(parts[2]), int(parts[3])
                        blocks.append((bi, bj, bk))
                    except Exception:
                        continue
    except Exception:
        use_block_mode = False

    whole_mask = np.ones(pc.shape[0]).astype('bool')

                          
    def _update_visibility_for_indices(idx_array):
        nonlocal whole_mask
        if idx_array.size == 0:
            return
        sub_points_np = pc[idx_array]
        for i in range(0, n_imgs, 1):
            _, _, depth, c2w = frame_reader[i]
            depth, c2w = depth.to(device), c2w.to(device)

            if not estimate_c2w_list is None:
                c2w = estimate_c2w_list[i].to(device)

            points = torch.from_numpy(sub_points_np).to(device)

                         
            try:
                det = torch.det(c2w)
                if torch.abs(det) < 1e-6:
                          
                    continue
                w2c = torch.inverse(c2w)
            except torch._C._LinAlgError:
                continue
            K = torch.from_numpy(
                np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).to(device)
            ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
            homo_points = torch.cat(
                [points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
            cam_cord_homo = w2c@homo_points
            cam_cord = cam_cord_homo[:, :3]

            cam_cord[:, 0] *= -1
            uv = K.float()@cam_cord.float()
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.squeeze(-1)

            grid = uv[None, None].clone()
            grid[..., 0] = grid[..., 0] / W
            grid[..., 1] = grid[..., 1] / H
            grid = 2 * grid - 1
            depth_samples = F.grid_sample(depth[None, None], grid, padding_mode='zeros', align_corners=True).squeeze()

            edge = 0
            if eval_rec:
                mask = (depth_samples + truncation >= -z[:, 0, 0]) & (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
            else:
                mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W -edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)

            mask = mask.detach().cpu().numpy()
            whole_mask[idx_array] &= ~mask

    if pc.shape[0] == 0:
        return

    batch_size = 200000
    all_indices = np.arange(pc.shape[0])
    for start in tqdm(range(0, pc.shape[0], batch_size), desc="Culling mesh vertices"):
        end = min(start + batch_size, pc.shape[0])
        _update_visibility_for_indices(all_indices[start:end])

    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_faces(~face_mask)
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=False)

    mesh_ext = mesh_file.split('.')[-1]
    output_file = mesh_file[:-len(mesh_ext) - 1] + '_culled.' + mesh_ext

    mesh.export(output_file)

                                                                                                                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to cull the mesh.'
    )

    parser.add_argument('config', type=str,  help='path to the config file')
    parser.add_argument('--input_mesh', type=str, help='path to the mesh to be culled')

    args = parser.parse_args()
    args.input_folder = None

    default_config = Path(__file__).resolve().parents[2] / 'configs' / 'structrecon.yaml'
    cfg = config.load_config(args.config, str(default_config))

    cull_mesh(args.input_mesh, cfg, args, 'cuda')
