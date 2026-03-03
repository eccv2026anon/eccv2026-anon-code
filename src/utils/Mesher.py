import numpy as np
import open3d as o3d
import skimage
import torch
import trimesh
from packaging import version
from src.utils.datasets import get_dataset
import os
import matplotlib.pyplot as plt
import gc
from src.networks.decoders import Decoders
                                               
class Mesher(object):
       
                                    
                                   
                                                          
                                             
                                    
    def __init__(self, cfg, args, structrecon, points_batch_size=100000, ray_batch_size=30000):
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size                       
        self.renderer = structrecon.renderer                                  
        self.scale = cfg['scale']                              
                                                      
        self.cfg = cfg

                           
        resolution = cfg['meshing']['resolution']                       
        
                            
        if resolution < 0.01:
            print(f"Warning: Resolution set too small ({resolution}), may cause memory overflow. Automatically adjusted to 0.01.")
            resolution = 0.01
            
        self.resolution = resolution
        self.level_set = cfg['meshing']['level_set']                         
        self.mesh_bound_scale = cfg['meshing']['mesh_bound_scale']                                

        self.bound = structrecon.bound                               
        self.verbose = structrecon.verbose                    

        self.marching_cubes_bound = torch.from_numpy(
            np.array(cfg['mapping']['marching_cubes_bound']) * self.scale)                                                                 

                 
                                      
                                                                    
                                                     
                                             
        self.bound_source = cfg.get('meshing', {}).get('bound_source', 'frames')

        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu')                                 
        self.n_img = len(self.frame_reader)         

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = structrecon.H, structrecon.W, structrecon.fx, structrecon.fy, structrecon.cx, structrecon.cy                          
        
                  
        if self.verbose:
            print(f"Grid generation parameters: resolution={self.resolution}, point batch size={self.points_batch_size}, ray batch size ={self.ray_batch_size}")

                 
        self.debug_visualize = cfg.get('meshing', {}).get('debug_visualize', False)
                   
        self.output = getattr(structrecon, 'output', None)
                                        
        try:
            self.block_manager = getattr(structrecon, 'block_manager', None)
        except Exception:
            self.block_manager = None

    def get_bound_from_frames(self, keyframe_dict, scale=1):
           

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
                                                             
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
                               
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict.values():              
            c2w = keyframe['est_c2w'].cpu().numpy()
                                           
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh

    def eval_points(self, p, all_planes, decoders, need_rgb: bool = True):
           

        N = p.shape[0]
        bound = self.bound
                                    
        out_all = torch.empty((N, 4), device=p.device, dtype=p.dtype)

                         
        debug_stats = {'in_bound': 0, 'out_bound': 0, 'decoder_calls': 0, 'sdf_ranges': []}

        start = 0
        while start < N:
            end = min(start + self.points_batch_size, N)
            pi = p[start:end]

                                          
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z
            
                         
            debug_stats['in_bound'] += mask.sum().item()
            debug_stats['out_bound'] += (~mask).sum().item()

                                     
            with torch.inference_mode():
                debug_stats['decoder_calls'] += 1
                                              
                if not hasattr(self, '_printed_weight_source'):
                    try:
                        src = getattr(decoders, '_weight_source', 'unknown')
                        path = getattr(decoders, '_weight_source_path', None)
                        if path is not None:
                            print(f"[Mesher.eval_points] Decoder weight source: {src} | path: {path}")
                        else:
                            print(f"[Mesher.eval_points] Decoder weight source: {src}")
                    except Exception:
                        pass
                    setattr(self, '_printed_weight_source', True)
                out = decoders(pi, all_planes=all_planes, need_rgb=need_rgb)

                                                                      
            if isinstance(out, dict):
                sdf = out.get('sdf', None)
                rgb = out.get('rgb', None)
                if sdf is None:
                    raise ValueError('Dictionary returned by Decoders.forward is missing sdf key')
                                                
                if sdf.dim() == 3:
                    sdf = sdf.mean(dim=1)
                if need_rgb:
                    if rgb is None:
                        rgb = torch.zeros(sdf.shape[0], 3, device=pi.device, dtype=pi.dtype)
                    elif rgb.dim() == 3:
                        rgb = rgb.mean(dim=1)
                else:
                    rgb = torch.zeros(sdf.shape[0], 3, device=pi.device, dtype=pi.dtype)
                ret = torch.cat([rgb, sdf], dim=-1)         
            else:
                ret = out              

                               
            if mask.any():
                sdf_in_bound = ret[mask, -1]
                if len(sdf_in_bound) > 0:
                    debug_stats['sdf_ranges'].append((sdf_in_bound.min().item(), sdf_in_bound.max().item()))

                                                    
            if getattr(self, 'debug_visualize', False):
                try:
                    import os
                    import numpy as np
                    import matplotlib.pyplot as plt
                    try:
                        import open3d as o3d
                    except Exception:
                        o3d = None

                    debug_dir_root = self.output if getattr(self, 'output', None) else '.'
                    debug_dir = os.path.join(debug_dir_root, "mesh_debug")
                    os.makedirs(debug_dir, exist_ok=True)

                    batch_tag = f"{debug_stats['decoder_calls']:04d}"

                                                            
                    if mask.any():
                        sdf_vals = sdf_in_bound.detach().float().cpu().numpy()

                                                          
                        clip_from_model = getattr(decoders, 'truncation', None)
                        if clip_from_model is not None and np.isfinite(clip_from_model):
                            clip_min, clip_max = -float(clip_from_model), float(clip_from_model)
                        else:
                            q1, q99 = np.percentile(sdf_vals, [1, 99])
                            clip_min, clip_max = float(q1), float(q99)

                        clipped_low = int((sdf_vals < clip_min).sum())
                        clipped_high = int((sdf_vals > clip_max).sum())

                                                    
                        plt.figure(figsize=(4, 3))
                        plt.hist(sdf_vals, bins=80, range=(clip_min, clip_max))
                        plt.title(f"SDF hist [{clip_min:.3f}, {clip_max:.3f}]  clip_low={clipped_low}, clip_high={clipped_high}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(debug_dir, f"sdf_hist_batch_{batch_tag}.png"))
                        plt.close()

                                        
                        plt.figure(figsize=(4, 3))
                        counts, bins, _ = plt.hist(sdf_vals, bins=80)
                        plt.yscale('log')
                        plt.title("SDF histogram (global, log-scale)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(debug_dir, f"sdf_hist_global_log_{batch_tag}.png"))
                        plt.close()

                                             
                        try:
                            trunc = float(clip_from_model) if (clip_from_model is not None) else float(max(abs(clip_min), abs(clip_max)))
                            near_mask = np.abs(sdf_vals) <= 0.5 * trunc
                            ns_vals = sdf_vals[near_mask]
                            if ns_vals.size > 0:
                                ns_bias = float(ns_vals.mean())
                                ns_std = float(ns_vals.std())
                                            
                                try:
                                    print(f"[Mesher.eval_points] batch {batch_tag} near-surface |bias={ns_bias:.4f}, std={ns_std:.4f}| count={ns_vals.size}")
                                except Exception:
                                    pass
                                try:
                                    if isinstance(debug_stats, dict):
                                        lst = debug_stats.setdefault('near_surface_stats', [])
                                        lst.append({
                                            'batch': int(debug_stats.get('decoder_calls', 0)),
                                            'bias': ns_bias,
                                            'std': ns_std,
                                            'count': int(ns_vals.size)
                                        })
                                                                            
                                    try:
                                        if not hasattr(self, '_near_surface_stats') or self._near_surface_stats is None:
                                            self._near_surface_stats = []
                                        self._near_surface_stats.append({
                                            'batch': int(debug_stats.get('decoder_calls', 0)),
                                            'bias': ns_bias,
                                            'std': ns_std,
                                            'count': int(ns_vals.size)
                                        })
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                plt.figure(figsize=(4, 3))
                                plt.hist(ns_vals, bins=80)
                                                
                                plt.axvline(0.0, color='k', linestyle='--', linewidth=1)
                                plt.axvline(-trunc, color='r', linestyle=':', linewidth=1)
                                plt.axvline(+trunc, color='r', linestyle=':', linewidth=1)
                                plt.title(f"SDF near-surface |bias={ns_bias:.4f}, std={ns_std:.4f}|")
                                plt.tight_layout()
                                plt.savefig(os.path.join(debug_dir, f"sdf_hist_nearsurf_{batch_tag}.png"))
                                plt.close()
                        except Exception as _:
                            pass

                                                 
                    if mask.any() and o3d is not None:
                        pi_cpu = pi.detach().cpu()
                        ret_cpu = ret.detach().cpu()
                        pts_in = pi_cpu[mask]
                        rgb_in = ret_cpu[mask, :3]                  
                        sdf_in = ret_cpu[mask, -1:]

                                        
                        if torch.allclose(rgb_in.abs().sum(), torch.tensor(0.0)):
                            s = sdf_in.numpy().flatten()
                            s = np.clip((s - s.min()) / (s.max() - s.min() + 1e-12), 0, 1)
                            import matplotlib.cm as cm
                            cmap = cm.get_cmap('coolwarm')
                            rgb_np = cmap(s)[:, :3].astype(np.float32)
                        else:
                            rgb_np = rgb_in.numpy().astype(np.float32)

                                                    
                        max_pts = 1000000
                        if len(pts_in) > max_pts:
                            sel = np.random.choice(len(pts_in), max_pts, replace=False)
                            pts_np = pts_in.numpy()[sel]
                            rgb_np = rgb_np[sel]
                        else:
                            pts_np = pts_in.numpy()

                                              
                        finite_mask = np.isfinite(pts_np).all(axis=1) & np.isfinite(rgb_np).all(axis=1)
                        if finite_mask.any():
                            pts_np = pts_np[finite_mask]
                            rgb_np = np.clip(rgb_np[finite_mask], 0.0, 1.0)
                        else:
                                          
                            pts_np = None

                        if pts_np is not None and len(pts_np) > 0:
                            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_np))
                            pcd.colors = o3d.utility.Vector3dVector(rgb_np)
                                                
                            o3d.io.write_point_cloud(
                                os.path.join(debug_dir, f"inbound_points_{batch_tag}.ply"),
                                pcd,
                                write_ascii=True,
                                compressed=False
                            )

                                         
                    if mask.any():
                        pi_cpu = pi.detach().cpu()
                        pts_in = pi_cpu[mask]
                        sdf_in = ret.detach().cpu()[mask, -1]
                        z_vals = pts_in[:, 2].numpy()
                        z0 = float(np.median(z_vals))
                        eps = 0.01
                        slice_mask = (z_vals > z0 - eps) & (z_vals < z0 + eps)
                        if slice_mask.any():
                            xs = pts_in[:, 0].numpy()[slice_mask]
                            ys = pts_in[:, 1].numpy()[slice_mask]
                            ss = sdf_in.numpy().flatten()[slice_mask]
                            plt.figure(figsize=(4, 4))
                            plt.scatter(xs, ys, c=np.clip(ss, -0.1, 0.1), cmap='coolwarm', s=1)
                            plt.colorbar(label='SDF')
                            plt.title(f"Slice z{z0:.3f}")
                            plt.tight_layout()
                            plt.savefig(os.path.join(debug_dir, f"sdf_slice_z_{batch_tag}.png"))
                            plt.close()
                except Exception as e:
                    print(f"Debugging visualization failed:{e}")

                                        
            ret = ret.to(device=pi.device, dtype=p.dtype, non_blocking=True)

                                             
            if ret.shape[-1] == 4:
                ret[~mask, -1] = -1
                ret[~mask, :3] = 0

            out_all[start:end] = ret
            start = end

                            
        if debug_stats['sdf_ranges']:
            all_mins = [r[0] for r in debug_stats['sdf_ranges']]
            all_maxs = [r[1] for r in debug_stats['sdf_ranges']]
            print(f"eval_points debug: within bounds{debug_stats['in_bound']}, outside the boundary{debug_stats['out_bound']}, "
                  f"Decoder call{debug_stats['decoder_calls']}Second-rate")
            print(f"SDF range within bounds: [{min(all_mins):.6f}, {max(all_maxs):.6f}]")
        else:
            print(f"eval_points debug: within bounds{debug_stats['in_bound']}, outside the boundary{debug_stats['out_bound']}, "
                  f"but no valid SDF value")

        return out_all

    def get_grid_uniform(self, resolution):
           
        bound = self.marching_cubes_bound

        padding = 0.05

                   
        nsteps_x = ((bound[0][1] - bound[0][0] + 2 * padding) / resolution).round().int().item()
        nsteps_y = ((bound[1][1] - bound[1][0] + 2 * padding) / resolution).round().int().item()
        nsteps_z = ((bound[2][1] - bound[2][0] + 2 * padding) / resolution).round().int().item()
        
                 
        total_points = nsteps_x * nsteps_y * nsteps_z
        
                                     
        max_points = 100000000                 
        
        if total_points > max_points:
                                  
            scale_factor = (total_points / max_points) ** (1/3)
            
                   
            resolution = resolution * scale_factor
            
                    
            nsteps_x = ((bound[0][1] - bound[0][0] + 2 * padding) / resolution).round().int().item()
            nsteps_y = ((bound[1][1] - bound[1][0] + 2 * padding) / resolution).round().int().item()
            nsteps_z = ((bound[2][1] - bound[2][0] + 2 * padding) / resolution).round().int().item()
            
            total_points = nsteps_x * nsteps_y * nsteps_z
            print(f"Warning: There are too many grid points and the resolution has been automatically adjusted. New resolution:{resolution:.4f}, number of grid points:{total_points:,}")
        
                  
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding, nsteps_x)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding, nsteps_y)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding, nsteps_z)

                       
        x_t, y_t, z_t = torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float()
        
                                 
        grid_points_list = []
        chunk_size = 1000000           
        
                            
        for i in range(0, nsteps_x, max(1, nsteps_x // 10)):
            end_i = min(i + max(1, nsteps_x // 10), nsteps_x)
            x_chunk = x_t[i:end_i]
            
            for j in range(0, nsteps_y, max(1, nsteps_y // 10)):
                end_j = min(j + max(1, nsteps_y // 10), nsteps_y)
                y_chunk = y_t[j:end_j]
                
                for k in range(0, nsteps_z, max(1, nsteps_z // 10)):
                    end_k = min(k + max(1, nsteps_z // 10), nsteps_z)
                    z_chunk = z_t[k:end_k]
                    
                              
                    grid_x, grid_y, grid_z = torch.meshgrid(x_chunk, y_chunk, z_chunk, indexing='xy')
                    chunk_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)
                    grid_points_list.append(chunk_points)
        
               
        grid_points_t = torch.cat(grid_points_list, dim=0)

        return {"grid_points": grid_points_t, "xyz": [x, y, z]}

    def _visualize_active_feature_planes(self, block_manager, output_dir: str, normalize: bool = True, grid_line_width: int = 2, block_index_fontsize: int = 6):
           
        os.makedirs(output_dir, exist_ok=True)

        if not block_manager.active_blocks:
            print("Visualization: There are currently no active blocks loaded, feature plane export is skipped.")
            return

                                               
        some_block = next(iter(block_manager.active_blocks.values()))
        plane_keys = [k for k in some_block.keys() if k.startswith('geo_feat_') or k.startswith('app_feat_')]
        if not plane_keys:
            print("Visualization: Feature plane key not found in active block, skipped.")
            return

                                        
        block_items = list(block_manager.active_blocks.items())
        block_indices = [idx for idx, _ in block_items]

        for feature_key in plane_keys:
            try:
                                                       
                valid = [(idx, blk[feature_key]) for idx, blk in block_items if feature_key in blk]
                if not valid:
                    continue
                                        
                c_dim, block_height, block_width = valid[0][1].shape

                                                
                parts = feature_key.split('_')
                plane_type = parts[2] if len(parts) > 2 else 'unknown'

                x_list = [idx[0] for idx, _ in valid]
                y_list = [idx[1] for idx, _ in valid]
                z_list = [idx[2] for idx, _ in valid]
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                z_min, z_max = min(z_list), max(z_list)

                if plane_type == 'xy':
                    grid_rows = y_max - y_min + 1
                    grid_cols = x_max - x_min + 1
                    pos_key = lambda idx: (idx[0], idx[1])
                    position_map = {(idx[0], idx[1]): (idx[1] - y_min, idx[0] - x_min) for idx, _ in valid}
                elif plane_type == 'xz':
                    grid_rows = z_max - z_min + 1
                    grid_cols = x_max - x_min + 1
                    pos_key = lambda idx: (idx[0], idx[2])
                    position_map = {(idx[0], idx[2]): (idx[2] - z_min, idx[0] - x_min) for idx, _ in valid}
                elif plane_type == 'yz':
                    grid_rows = z_max - z_min + 1
                    grid_cols = y_max - y_min + 1
                    pos_key = lambda idx: (idx[1], idx[2])
                    position_map = {(idx[1], idx[2]): (idx[2] - z_min, idx[1] - y_min) for idx, _ in valid}
                else:
                                             
                    import math
                    n = len(valid)
                    grid_rows = grid_cols = int(math.ceil(math.sqrt(n)))
                    pos_key = None
                    position_map = {}

                buffer_size = 1
                eff_h = block_height + 2 * buffer_size
                eff_w = block_width + 2 * buffer_size
                full_h = grid_rows * eff_h + (grid_rows - 1) * grid_line_width
                full_w = grid_cols * eff_w + (grid_cols - 1) * grid_line_width
                full_image = np.ones((full_h, full_w), dtype=np.float32) * np.nan

                                                          
                gmin, gmax = float('inf'), float('-inf')
                if normalize:
                    for _, feat in valid:
                        avg_map = feat.detach().mean(dim=0).cpu().numpy()
                        if np.isfinite(avg_map).any():
                            gmin = min(gmin, np.nanmin(avg_map))
                            gmax = max(gmax, np.nanmax(avg_map))

                             
                for i, (idx, feat) in enumerate(valid):
                    avg_map = feat.detach().mean(dim=0).cpu().numpy()
                    if pos_key is None:
                        r = i // grid_cols
                        c = i % grid_cols
                    else:
                        key = pos_key(idx)
                        if key in position_map:
                            r, c = position_map[key]
                        else:
                            r = i // grid_cols
                            c = i % grid_cols
                    rs = r * (eff_h + grid_line_width)
                    cs = c * (eff_w + grid_line_width)
                    full_image[rs + buffer_size:rs + buffer_size + block_height,
                               cs + buffer_size:cs + buffer_size + block_width] = avg_map

                plt.figure(figsize=(12, 10))
                if normalize and np.isfinite([gmin, gmax]).all() and gmax > gmin:
                    vmin, vmax = gmin, gmax
                else:
                    vmin = vmax = None
                img = plt.imshow(full_image, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax, origin='lower')
                cbar = plt.colorbar(img, label='Feature Value', fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                plt.title(f'Feature Plane: {feature_key} (Channel Average)')

                                                          
                for i, (idx, feat) in enumerate(valid):
                    if pos_key is None:
                        r = i // grid_cols
                        c = i % grid_cols
                    else:
                        key = pos_key(idx)
                        if key in position_map:
                            r, c = position_map[key]
                        else:
                            r = i // grid_cols
                            c = i % grid_cols
                    row_top = r * (eff_h + grid_line_width) + buffer_size
                    col_center = c * (eff_w + grid_line_width) + buffer_size + block_width // 2
                    plt.text(col_center, row_top, f"{idx[0]},{idx[1]},{idx[2]}", color='yellow', fontsize=block_index_fontsize, ha='center', va='top')

                plt.axis('off')
                out_path = os.path.join(output_dir, f"{feature_key}_avg.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=500)
                plt.close()
                print(f"Feature visualization image saved to:{out_path}")
            except Exception as e:
                print(f"Visualization{feature_key}An error occurred:{e}")

    def _save_decoder_and_features(self, mesh_out_file, decoders, all_planes, device):
           
        try:
            import os
            import time
            
                    
            mesh_dir = os.path.dirname(mesh_out_file)
            mesh_name = os.path.splitext(os.path.basename(mesh_out_file))[0]
            
                     
            decoder_save_path = os.path.join(mesh_dir, f"{mesh_name}_decoder_weights.pth")
            print(f"Save decoder weights to:{decoder_save_path}")
            
                     
            save_data = {
                'decoder_state_dict': decoders.state_dict(),
                'decoder_config': {
                    'c_dim': getattr(decoders, 'c_dim', 32),
                    'truncation': getattr(decoders, 'truncation', 0.08),
                    'n_blocks': getattr(decoders, 'n_blocks', 2),
                    'use_block_manager': getattr(decoders, 'use_block_manager', False),
                    'use_color_multires': getattr(decoders, 'use_color_multires', False),
                },
                'bound': decoders.bound.cpu() if hasattr(decoders, 'bound') and decoders.bound is not None else None,
                'save_timestamp': time.time(),
                'mesh_file': mesh_out_file,
            }
            
                           
            if hasattr(decoders, '_weight_source'):
                save_data['weight_source'] = getattr(decoders, '_weight_source', 'unknown')
            if hasattr(decoders, '_weight_source_path'):
                save_data['weight_source_path'] = getattr(decoders, '_weight_source_path', None)
            
            torch.save(save_data, decoder_save_path)
            
                      
            color_stats = self._analyze_color_features(all_planes, decoders, device)
            
                    
            stats_save_path = os.path.join(mesh_dir, f"{mesh_name}_feature_stats.json")
            print(f"Save feature statistics to:{stats_save_path}")
            
            import json
            with open(stats_save_path, 'w', encoding='utf-8') as f:
                json.dump(color_stats, f, indent=2, ensure_ascii=False)
            
            print(f"Decoder weights and feature statistics are saved")
            print(f"- Decoder weights:{decoder_save_path}")
            print(f"- Feature statistics:{stats_save_path}")
            
        except Exception as e:
            print(f"Failed to save decoder weights and feature statistics:{e}")

    def _analyze_color_features(self, all_planes, decoders, device):
           
        import time
        stats = {
            'has_color_planes': False,
            'color_plane_stats': {},
            'sample_color_analysis': {},
            'analysis_timestamp': time.time()
        }
        
        try:
            import numpy as np
            import time
            
                         
            if all_planes and len(all_planes) >= 6:
                c_planes_xy, c_planes_xz, c_planes_yz = all_planes[3], all_planes[4], all_planes[5]
                
                if c_planes_xy and c_planes_xz and c_planes_yz:
                    stats['has_color_planes'] = True
                    
                                 
                    plane_names = ['xy', 'xz', 'yz']
                    plane_groups = [c_planes_xy, c_planes_xz, c_planes_yz]
                    
                    for name, planes in zip(plane_names, plane_groups):
                        if planes:
                            plane_stats = []
                            for i, plane in enumerate(planes):
                                if isinstance(plane, torch.nn.Parameter):
                                    plane_tensor = plane.data
                                else:
                                    plane_tensor = plane
                                
                                            
                                plane_np = plane_tensor.detach().cpu().numpy()
                                
                                        
                                plane_stat = {
                                    'level': i,                    
                                    'shape': list(plane_np.shape),
                                    'mean': float(np.mean(plane_np)),
                                    'std': float(np.std(plane_np)),
                                    'min': float(np.min(plane_np)),
                                    'max': float(np.max(plane_np)),
                                    'zero_ratio': float(np.mean(np.abs(plane_np) < 1e-6)),
                                    'non_zero_count': int(np.sum(np.abs(plane_np) >= 1e-6)),
                                    'total_elements': int(plane_np.size)
                                }
                                
                                                
                                plane_stat['has_meaningful_values'] = (
                                    plane_stat['std'] > 1e-4 and 
                                    plane_stat['zero_ratio'] < 0.95 and
                                    plane_stat['non_zero_count'] > plane_stat['total_elements'] * 0.05
                                )
                                
                                plane_stats.append(plane_stat)
                            
                            stats['color_plane_stats'][name] = plane_stats
                    
                              
                    stats['sample_color_analysis'] = self._sample_color_decoding(decoders, device)

                                                                                     
            if not stats['has_color_planes']:
                bm = getattr(self, 'block_manager', None)
                if bm is not None and hasattr(bm, 'active_blocks') and isinstance(bm.active_blocks, dict) and len(bm.active_blocks) > 0:
                                             
                    color_keys = ['app_feat_xy_coarse', 'app_feat_xz_coarse', 'app_feat_yz_coarse',
                                  'app_feat_xy_fine', 'app_feat_xz_fine', 'app_feat_yz_fine',
                                  'app_feat_xy', 'app_feat_xz', 'app_feat_yz']
                    found_any = False
                    aggregate = {k: [] for k in color_keys}
                    try:
                        for blk_idx, planes in bm.active_blocks.items():
                            for k in color_keys:
                                if k in planes:
                                    found_any = True
                                    tensor_param = planes[k]
                                    tensor_t = tensor_param.data if isinstance(tensor_param, torch.nn.Parameter) else tensor_param
                                    aggregate[k].append(tensor_t.detach().cpu().numpy())
                    except Exception as e:
                        print(f"Color plane aggregation failed:{e}")

                    if found_any:
                        stats['has_color_planes'] = True
                                                     
                        dir_map = {
                            'xy': ['app_feat_xy_coarse', 'app_feat_xy_fine', 'app_feat_xy'],
                            'xz': ['app_feat_xz_coarse', 'app_feat_xz_fine', 'app_feat_xz'],
                            'yz': ['app_feat_yz_coarse', 'app_feat_yz_fine', 'app_feat_yz']
                        }
                        for dir_name, keys in dir_map.items():
                            plane_stats = []
                            level = 0
                            for k in keys:
                                arrs = aggregate.get(k, [])
                                for arr in arrs:
                                    try:
                                        plane_stat = {
                                            'level': level,
                                            'shape': list(arr.shape),
                                            'mean': float(np.mean(arr)),
                                            'std': float(np.std(arr)),
                                            'min': float(np.min(arr)),
                                            'max': float(np.max(arr)),
                                            'zero_ratio': float(np.mean(np.abs(arr) < 1e-6)),
                                            'non_zero_count': int(np.sum(np.abs(arr) >= 1e-6)),
                                            'total_elements': int(arr.size)
                                        }
                                        plane_stat['has_meaningful_values'] = (
                                            plane_stat['std'] > 1e-4 and 
                                            plane_stat['zero_ratio'] < 0.95 and
                                            plane_stat['non_zero_count'] > plane_stat['total_elements'] * 0.05
                                        )
                                        plane_stats.append(plane_stat)
                                    except Exception:
                                        pass
                                level += 1
                            if plane_stats:
                                stats['color_plane_stats'][dir_name] = plane_stats
                                  
                        stats['sample_color_analysis'] = self._sample_color_decoding(decoders, device)
            
                      
            stats['color_quality_summary'] = self._summarize_color_quality(stats)
            
        except Exception as e:
            stats['analysis_error'] = str(e)
            print(f"Color feature analysis failed:{e}")
        
        return stats

    def _sample_color_decoding(self, decoders, device, num_samples=1000):
           
        analysis = {
            'num_samples': num_samples,
            'successful_samples': 0,
            'color_stats': {},
            'has_valid_colors': False
        }
        
        try:
                                        
            bm = getattr(self, 'block_manager', None)
            sampled_pts = None
            if bm is not None and hasattr(bm, 'active_blocks') and isinstance(bm.active_blocks, dict) and len(bm.active_blocks) > 0 and hasattr(bm, 'get_block_bound'):
                try:
                    active_indices = list(bm.active_blocks.keys())
                                          
                    per_block = max(1, num_samples // max(1, min(len(active_indices), 20)))
                    pts_list = []
                    rng = np.random.default_rng(42)
                    for blk_idx in active_indices[:20]:
                        bb = bm.get_block_bound(blk_idx)
                                                                                         
                        if isinstance(bb, torch.Tensor):
                            bb_np = bb.detach().cpu().numpy()
                        else:
                            bb_np = np.array(bb)
                        bmin = bb_np[0]
                        bmax = bb_np[1]
                                         
                        pts = rng.uniform(bmin, bmax, size=(per_block, 3))
                        pts_list.append(pts)
                    if pts_list:
                        sampled_pts = np.concatenate(pts_list, axis=0)
                        if sampled_pts.shape[0] > num_samples:
                            sampled_pts = sampled_pts[:num_samples]
                except Exception as e:
                    print(f"_sample_color_decoding: AABB:{e}")

            if sampled_pts is None:
                                        
                if hasattr(decoders, 'bound') and decoders.bound is not None:
                    bound = decoders.bound
                    bound_min = bound[:, 0].cpu().numpy()
                    bound_max = bound[:, 1].cpu().numpy()
                else:
                            
                    bound_min = np.array([-2.0, -2.0, -1.0])
                    bound_max = np.array([2.0, 2.0, 1.0])
                rng = np.random.default_rng(42)
                sampled_pts = rng.uniform(bound_min, bound_max, (num_samples, 3))

                         
            sample_points_tensor = torch.from_numpy(sampled_pts).float().to(device)
            
                    
            batch_size = min(100, num_samples)
            all_colors = []
            
            printed_once = False
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    batch_points = sample_points_tensor[i:i+batch_size]
                    try:
                                                                                              
                                                            
                        if getattr(decoders, 'use_block_manager', False):
                            output = decoders(batch_points, all_planes=None, need_rgb=True)
                            if isinstance(output, dict):
                                           
                                color_keys_try = ['rgb', 'app_rgb', 'color', 'colors']
                                colors = None
                                for ck in color_keys_try:
                                    if ck in output and isinstance(output[ck], torch.Tensor):
                                        colors = output[ck]
                                        break
                                if colors is None:
                                    colors = torch.zeros(len(batch_points), 3, device=device)
                            else:
                                                      
                                colors = output[..., :3]
                        else:
                                                       
                            if hasattr(decoders, 'get_raw_rgb'):
                                colors = decoders.get_raw_rgb(batch_points, all_planes=None)
                                output = {'rgb_from_get_raw_rgb': colors}
                            else:
                                output = decoders(batch_points, all_planes=None, need_rgb=True)
                                if isinstance(output, dict):
                                    color_keys_try = ['rgb', 'app_rgb', 'color', 'colors']
                                    colors = None
                                    for ck in color_keys_try:
                                        if ck in output and isinstance(output[ck], torch.Tensor):
                                            colors = output[ck]
                                            break
                                    if colors is None:
                                        colors = torch.zeros(len(batch_points), 3, device=device)
                                else:
                                    colors = output[..., :3]

                                        
                        if not printed_once:
                            try:
                                if isinstance(output, dict):
                                    keys = list(output.keys())
                                    print(f"[ColorSampleDBG] decoder output keys: {keys}")
                                    for k in keys[:6]:
                                        v = output[k]
                                        if isinstance(v, torch.Tensor):
                                            vstats = (float(v.nanmean().item()) if torch.isfinite(v).any() else 0.0,
                                                      float(v.nanstd().item()) if torch.isfinite(v).any() else 0.0)
                                            print(f"[ColorSampleDBG] key={k} shape={list(v.shape)} mean/std={vstats}")
                                else:
                                    print(f"[ColorSampleDBG] decoder non-dict output shape={list(output.shape)}")
                                printed_once = True
                            except Exception:
                                printed_once = True
                        
                        all_colors.append(colors.cpu().numpy())
                        analysis['successful_samples'] += len(batch_points)
                        
                    except Exception as e:
                        print(f"color decoding batch{i//batch_size + 1}fail:{e}")
                                    
                        all_colors.append(np.zeros((len(batch_points), 3)))
            
            if all_colors:
                all_colors = np.concatenate(all_colors, axis=0)
                
                        
                analysis['color_stats'] = {
                    'mean_rgb': [float(np.mean(all_colors[:, i])) for i in range(3)],
                    'std_rgb': [float(np.std(all_colors[:, i])) for i in range(3)],
                    'min_rgb': [float(np.min(all_colors[:, i])) for i in range(3)],
                    'max_rgb': [float(np.max(all_colors[:, i])) for i in range(3)],
                    'zero_color_ratio': float(np.mean(np.sum(np.abs(all_colors), axis=1) < 1e-6)),
                    'valid_color_ratio': float(np.mean(np.sum(np.abs(all_colors), axis=1) >= 1e-6)),
                    'color_diversity': float(np.std(np.mean(all_colors, axis=1))),         
                }
                
                           
                analysis['has_valid_colors'] = (
                    analysis['color_stats']['valid_color_ratio'] > 0.1 and
                    analysis['color_stats']['color_diversity'] > 0.01 and
                    max(analysis['color_stats']['std_rgb']) > 0.01
                )
                
        except Exception as e:
            analysis['sampling_error'] = str(e)
            print(f"Color sampling test failed:{e}")
        
        return analysis

    def _summarize_color_quality(self, stats):
           
        summary = {
            'overall_quality': 'unknown',
            'has_trained_colors': False,
            'quality_score': 0.0,
            'recommendations': []
        }
        
        try:
            if not stats.get('has_color_planes', False):
                summary['overall_quality'] = 'no_color_planes'
                summary['recommendations'].append('No color feature plane detected')
                return summary
            
                      
            plane_quality_scores = []
            meaningful_planes = 0
            
            for plane_name, plane_list in stats.get('color_plane_stats', {}).items():
                for plane_stat in plane_list:
                    if plane_stat.get('has_meaningful_values', False):
                        meaningful_planes += 1
                                          
                        std_score = min(plane_stat['std'] * 10, 1.0)           
                        nonzero_score = 1.0 - plane_stat['zero_ratio']            
                        plane_score = (std_score + nonzero_score) / 2
                        plane_quality_scores.append(plane_score)
            
                      
            sample_analysis = stats.get('sample_color_analysis', {})
            sample_quality = 0.0
            
            if sample_analysis.get('has_valid_colors', False):
                color_stats = sample_analysis.get('color_stats', {})
                diversity_score = min(color_stats.get('color_diversity', 0) * 100, 1.0)
                valid_ratio_score = color_stats.get('valid_color_ratio', 0)
                std_score = min(max(color_stats.get('std_rgb', [0])) * 10, 1.0)
                sample_quality = (diversity_score + valid_ratio_score + std_score) / 3
            
                  
            if plane_quality_scores:
                plane_avg_score = sum(plane_quality_scores) / len(plane_quality_scores)
                summary['quality_score'] = (plane_avg_score + sample_quality) / 2
            else:
                summary['quality_score'] = sample_quality
            
                         
            summary['has_trained_colors'] = (
                meaningful_planes >= 3 and             
                summary['quality_score'] > 0.3 and
                sample_analysis.get('has_valid_colors', False)
            )
            
                  
            if summary['quality_score'] > 0.7:
                summary['overall_quality'] = 'excellent'
                summary['recommendations'].append('Color features are of excellent quality and contain rich color information')
            elif summary['quality_score'] > 0.5:
                summary['overall_quality'] = 'good'
                summary['recommendations'].append('Color features are of good quality and contain valid color information')
            elif summary['quality_score'] > 0.3:
                summary['overall_quality'] = 'fair'
                summary['recommendations'].append('Color feature quality is average and may require more training')
            elif summary['quality_score'] > 0.1:
                summary['overall_quality'] = 'poor'
                summary['recommendations'].append('Color feature quality is poor, it is recommended to check the color loss weights and training data')
            else:
                summary['overall_quality'] = 'very_poor'
                summary['recommendations'].append('Color features have almost no training, it is recommended to enable color loss and increase training')
            
                  
            if meaningful_planes < 3:
                summary['recommendations'].append(f'only{meaningful_planes}color plane contains meaningful values, it is recommended to check the color plane initialization')
            
            if not sample_analysis.get('has_valid_colors', False):
                summary['recommendations'].append('The sampling test shows that the color decoding output is invalid, it is recommended to check the decoder color branch')
            
        except Exception as e:
            summary['summary_error'] = str(e)
            print(f"Color quality summary failed:{e}")
        
        return summary

    def get_mesh(self, mesh_out_file, all_planes, decoders, keyframe_dict, device='cuda:0', color=True, visualize_planes_dir: str = None, use_init_decoder: bool = False, save_decoder_weights: bool = True, visualize_active_blocks: bool = True):
           

        with torch.no_grad():
            print("Start generating mesh...")
                                                   
            if use_init_decoder:
                try:
                    model_cfg = self.cfg.get('model', {})
                    decoder_cfg = model_cfg.get('decoder', {})
                    dec_init = Decoders(
                        c_dim=model_cfg.get('c_dim', 32),
                        hidden_size=decoder_cfg.get('hidden_size', 128),
                        truncation=model_cfg.get('truncation', 0.1),
                        n_blocks=decoder_cfg.get('n_blocks', 4),
                        device=device
                    ).to(device)
                                              
                    dec_init.bound = self.bound
                    if getattr(self, 'block_manager', None) is not None:
                        dec_init.block_manager = self.block_manager
                               
                    setattr(dec_init, '_weight_source', 'init')
                    setattr(dec_init, '_weight_source_path', None)
                    decoders = dec_init
                    print("[Mesher.get_mesh] Switched to initializing decoder weights for mesh evaluation")
                except Exception as e:
                    print(f"[Mesher.get_mesh] Failed to initialize the decoder, fall back to using the incoming decoder:{e}")

                                  
            try:
                self._near_surface_stats = []
            except Exception:
                pass
            
                                       
            memory_tracking = False
            memory_before = 0
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024      
                print(f"Current memory usage:{memory_before:.2f} MB")
                memory_tracking = True
            except ImportError:
                print("Note: The psutil library is not installed and memory usage will not be displayed. To monitor memory, please install: pip install psutil")
            
                    
            print("Calculate scene boundaries...")
                        
            if self.bound_source == 'frames':
                mesh_bound = self.get_bound_from_frames(keyframe_dict, self.scale)
                if self.verbose:
                    print("Boundary source: keyframe convex hull (frames)")
            else:
                              
                                                                                     
                if self.bound_source in ('structrecon', 'eslam'):
                    aabb = self.bound         
                    if self.verbose:
                        print("Boundary source: StructRecon.bound (AABB)")
                else:
                    aabb = self.marching_cubes_bound         
                    if self.verbose:
                        print("Boundary source: marching_cubes_bound (AABB)")
                mesh_bound = None            
            
                                                                                       
            use_block_mode = False
            blocks_dir = None
            try:
                from src.utils.BlockManager import BlockManager
                bm = BlockManager(self.cfg, device, bound=self.bound)
                blocks_dir = bm.presegmented_blocks_dir
                if blocks_dir and os.path.isdir(blocks_dir):
                               
                    blk_files = [f for f in os.listdir(blocks_dir) if f.startswith('block_') and f.endswith('.pth')]
                    use_block_mode = len(blk_files) > 0
                    if use_block_mode:
                        blk_files.sort()
                        print(f"Pre-split block directory detected, total{len(blk_files)}blocks, the path will be evaluated using the block-merged SDF grid.")
            except Exception as e:
                print(f"Block mode initialization fails and will fall back to the global evaluation path:{e}")
                use_block_mode = False

                             
            if visualize_active_blocks and hasattr(decoders, 'block_manager') and decoders.block_manager is not None:
                try:
                               
                    mesh_dir = os.path.dirname(mesh_out_file)
                    mesh_name = os.path.splitext(os.path.basename(mesh_out_file))[0]
                    active_blocks_dir = os.path.join(mesh_dir, f"{mesh_name}_active_blocks")
                    
                    print(f"Start visualizing the currently active blocks into the directory:{active_blocks_dir}")
                    self._visualize_active_feature_planes(
                        decoders.block_manager, 
                        active_blocks_dir,
                        normalize=True,
                        grid_line_width=2,
                        block_index_fontsize=8
                    )
                    print(f"The activation block visualization is completed and the PNG file has been saved to:{active_blocks_dir}")
                except Exception as e:
                    print(f"Activation block visualization failed:{e}")

                                          
            if use_block_mode:
                try:
                                                                             
                    aabb = self.bound         
                    x_min_g, y_min_g, z_min_g = float(aabb[0, 0]), float(aabb[1, 0]), float(aabb[2, 0])
                    x_max_g, y_max_g, z_max_g = float(aabb[0, 1]), float(aabb[1, 1]), float(aabb[2, 1])

                    resolution = float(self.resolution)
                    x_axis = np.arange(x_min_g, x_max_g, resolution)
                    y_axis = np.arange(y_min_g, y_max_g, resolution)
                    z_axis = np.arange(z_min_g, z_max_g, resolution)
                    print(f"Block Mode: Global Grid Dimension X={len(x_axis)} Y={len(y_axis)} Z={len(z_axis)}resolution ={resolution}")

                                                         
                    sdf_grid = np.full((len(x_axis), len(y_axis), len(z_axis)), np.inf, dtype=np.float32)

                         
                    eval_bs = min(self.points_batch_size, 100000)

                                                                                                         
                    def _build_planes_from_block_data(block_data, torch_device):
                        def get_plane(name):
                            if name not in block_data:
                                raise KeyError(f"Missing intra-block features:{name}")
                            return torch.nn.Parameter(block_data[name].float().to(torch_device))

                        planes_xy = [get_plane('geo_feat_xy_coarse'), get_plane('geo_feat_xy_fine')]
                        planes_xz = [get_plane('geo_feat_xz_coarse'), get_plane('geo_feat_xz_fine')]
                        planes_yz = [get_plane('geo_feat_yz_coarse'), get_plane('geo_feat_yz_fine')]
                                     
                        c_planes_xy = [torch.nn.Parameter(torch.zeros_like(p)) for p in planes_xy]
                        c_planes_xz = [torch.nn.Parameter(torch.zeros_like(p)) for p in planes_xz]
                        c_planes_yz = [torch.nn.Parameter(torch.zeros_like(p)) for p in planes_yz]
                        return (planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz)

                                                       
                    def _eval_sdf_pts(dec, pts, planes_tuple):
                        if hasattr(dec, 'get_raw_sdf') and callable(getattr(dec, 'get_raw_sdf')):
                            return dec.get_raw_sdf(pts, planes_tuple)
                                                 
                        out = dec(pts, all_planes=planes_tuple, need_rgb=False)
                        if isinstance(out, dict):
                            s = out.get('sdf', None)
                            if s is None:
                                raise RuntimeError('Decoder did not return sdf in dict output')
                            if s.dim() == 3:
                                s = s.mean(dim=1)
                            return s
                                               
                        return out[..., -1]

                                       
                    for idx, fname in enumerate(blk_files):
                        try:
                            base = os.path.splitext(fname)[0]
                            parts = base.split('_')
                            bi, bj, bk = int(parts[1]), int(parts[2]), int(parts[3])
                        except Exception:
                            print(f"Skip unresolved filenames:{fname}")
                            continue

                        bpath = os.path.join(blocks_dir, fname)
                        try:
                            block_data = torch.load(bpath, map_location=device)
                        except Exception as e:
                            print(f"Failed to load chunk{fname}: {e}")
                            continue

                        try:
                            planes = _build_planes_from_block_data(block_data, device)
                        except Exception as e:
                            print(f"Building block plane failed{fname}: {e}")
                            continue

                                               
                        bb = bm.get_block_bound((bi, bj, bk))         
                        decoders.bound = bb
                        bb_min = bb[0].detach().cpu().numpy()
                        bb_max = bb[1].detach().cpu().numpy()

                                                  
                        xi0 = int(np.rint((bb_min[0] - x_min_g) / resolution)) - 1
                        yi0 = int(np.rint((bb_min[1] - y_min_g) / resolution)) - 1
                        zi0 = int(np.rint((bb_min[2] - z_min_g) / resolution)) - 1
                        xi1 = int(np.rint((bb_max[0] - x_min_g) / resolution)) + 1
                        yi1 = int(np.rint((bb_max[1] - y_min_g) / resolution)) + 1
                        zi1 = int(np.rint((bb_max[2] - z_min_g) / resolution)) + 1

                                 
                        xi0, yi0, zi0 = max(xi0, 0), max(yi0, 0), max(zi0, 0)
                        xi1, yi1, zi1 = min(xi1, len(x_axis)), min(yi1, len(y_axis)), min(zi1, len(z_axis))
                        if xi0 >= xi1 or yi0 >= yi1 or zi0 >= zi1:
                            continue

                        xs = x_axis[xi0:xi1]
                        ys = y_axis[yi0:yi1]
                        zs = z_axis[zi0:zi1]

                               
                        local_shape = (len(xs), len(ys), len(zs))
                        local_vals = []
                        with torch.no_grad():
                            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
                            pts = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1).astype(np.float32)
                            for s in range(0, len(pts), eval_bs):
                                bpts = torch.from_numpy(pts[s:s + eval_bs]).to(device).float()
                                sdf = _eval_sdf_pts(decoders, bpts, planes)
                                local_vals.append(sdf.detach().cpu().numpy())
                            del xx, yy, zz, pts
                        local_sdf = np.concatenate(local_vals, axis=0).reshape(local_shape)

                                           
                        old = sdf_grid[xi0:xi1, yi0:yi1, zi0:zi1]
                        replace = np.abs(local_sdf) < np.abs(old)
                        old[replace] = local_sdf[replace]
                        sdf_grid[xi0:xi1, yi0:yi1, zi0:zi1] = old

                        if (idx + 1) % 10 == 0:
                            print(f"Processed{idx + 1}/{len(blk_files)}blocks")

                                           
                    print("The block mode merging is completed and marching cubes begins...")

                                    
                    try:
                        spacing = (resolution, resolution, resolution)
                        verts, faces, normals, values = skimage.measure.marching_cubes(
                            volume=sdf_grid,
                            level=float(self.level_set),
                            spacing=spacing
                        )
                    except Exception as e:
                        print(f"Block mode marching_cubes failed:{e}")
                        return

                                    
                    vertices = verts + np.array([x_axis[0], y_axis[0], z_axis[0]])

                                      
                    del sdf_grid
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                                
                    if color:
                        print("Start calculating vertex colors...")
                                    
                        if hasattr(decoders, 'block_manager') and decoders.block_manager is not None:
                            decoders.block_manager.active_blocks.clear()
                            decoders.block_manager.block_usage_time.clear()
                        gc.collect(); torch.cuda.empty_cache()
                        points_np = verts                                  
                        points_t = torch.from_numpy(points_np).float()
                        colors = torch.zeros(points_t.shape[0], 3)
                        bs_col = min(2048, points_t.shape[0])
                        for i, pnts in enumerate(torch.split(points_t, bs_col, dim=0)):
                            try:
                                z_color = self.eval_points(pnts.to(device).float(), all_planes, decoders, need_rgb=True).cpu()[..., :3]
                                colors[i * bs_col:i * bs_col + len(pnts)] = z_color
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    sub_bs = max(256, bs_col // 2)
                                    for j, sub in enumerate(torch.split(pnts, sub_bs, dim=0)):
                                        zc = self.eval_points(sub.to(device).float(), all_planes, decoders, need_rgb=True).cpu()[..., :3]
                                        colors[i * bs_col + j * sub_bs:i * bs_col + j * sub_bs + len(sub)] = zc
                                else:
                                    raise e
                            if (i + 1) % 3 == 0:
                                torch.cuda.empty_cache()
                        vertex_colors = colors.numpy()
                    else:
                        vertex_colors = None

                        
                    print("Save Grid (Block Mode)...")
                    vertices /= self.scale
                    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
                    mesh.export(mesh_out_file)
                    
                                  
                    if save_decoder_weights:
                        self._save_decoder_and_features(mesh_out_file, decoders, all_planes, device)
                    
                    if self.verbose:
                        print(f'Successfully saved mesh to{mesh_out_file}')
                        print(f'Grid Statistics:{len(vertices):,}vertex,{len(faces):,}noodle')

                                 
                    try:
                        stats = getattr(self, '_near_surface_stats', None)
                        if stats and len(stats) > 0:
                            total_count = sum(int(s.get('count', 0)) for s in stats)
                            if total_count > 0:
                                weighted_bias_sum = sum(float(s.get('bias', 0.0)) * int(s.get('count', 0)) for s in stats)
                                mean_bias = weighted_bias_sum / total_count
                                import math
                                var_sum = 0.0
                                for s in stats:
                                    n = int(s.get('count', 0))
                                    if n <= 0:
                                        continue
                                    b = float(s.get('bias', 0.0))
                                    sd = float(s.get('std', 0.0))
                                    var_sum += n * (sd * sd + (b - mean_bias) * (b - mean_bias))
                                pooled_std = math.sqrt(var_sum / total_count)
                                print(f"[Mesher.get_mesh:block] aggregated near-surface stats: count={total_count}, mean_bias={mean_bias:.6f}, pooled_std={pooled_std:.6f}")
                    except Exception:
                        pass

                    return              

                except Exception as e:
                    print(f"Block mode execution fails and falls back to the global evaluation path:{e}")

                               
            print("Generate uniform grid points...")
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            
            if memory_tracking:
                try:
                    memory_after_grid = process.memory_info().rss / 1024 / 1024
                    print(f"Memory usage after grid point generation:{memory_after_grid:.2f}MB (increase{memory_after_grid - memory_before:.2f} MB)")
                except (AttributeError, OSError) as e:
                    memory_tracking = False
                    print(f"Memory tracking failed ({type(e).__name__}: {e}), disabled")
                except Exception as e:
                    memory_tracking = False
                    print(f"Memory trace failed (unexpected exception{type(e).__name__}: {e}), disabled")
            
            print(f"Number of grid points generated:{points.shape[0]:,}")

                                
            print("Check if grid points are within bounds...")
            mask_list = []
            batch_size = min(self.points_batch_size, 1000000)           

            if mesh_bound is not None:
                                     
                for i, pnts in enumerate(torch.split(points, batch_size, dim=0)):
                    if i % 10 == 0:
                        print(f"process batch{i+1}/{(points.shape[0] + batch_size - 1) // batch_size}...")
                    mask_list.append(mesh_bound.contains(pnts.cpu().numpy()))
                mask = np.concatenate(mask_list, axis=0)
            else:
                                               
                                                 
                ax_min, ay_min, az_min = float(aabb[0, 0]), float(aabb[1, 0]), float(aabb[2, 0])
                ax_max, ay_max, az_max = float(aabb[0, 1]), float(aabb[1, 1]), float(aabb[2, 1])
                for i, pnts in enumerate(torch.split(points, batch_size, dim=0)):
                    if i % 10 == 0:
                        print(f"process batch{i+1}/{(points.shape[0] + batch_size - 1) // batch_size}...")
                    pn = pnts.cpu().numpy()
                    in_x = (pn[:, 0] >= ax_min) & (pn[:, 0] <= ax_max)
                    in_y = (pn[:, 1] >= ay_min) & (pn[:, 1] <= ay_max)
                    in_z = (pn[:, 2] >= az_min) & (pn[:, 2] <= az_max)
                    mask_list.append(in_x & in_y & in_z)
                mask = np.concatenate(mask_list, axis=0)
            
                        
            points_in_bound = mask.sum()
            print(f"Number of points within the boundary:{points_in_bound:,} ({points_in_bound / points.shape[0] * 100:.2f}%)")
                          
            print("Calculate SDF value...")
            z_list = []
            batch_size = min(self.points_batch_size, 500000)           

                          
            sdf_stats = {'min_vals': [], 'max_vals': [], 'mean_vals': [], 'valid_counts': []}
            did_visualize = False
            
            for i, pnts in enumerate(torch.split(points, batch_size, dim=0)):
                if i % 5 == 0:
                    print(f"Evaluate SDF batches{i+1}/{(points.shape[0] + batch_size - 1) // batch_size}...")
                
                                
                if i == 0:
                    print(f"First batch coordinate range: X[{pnts[:, 0].min():.4f}, {pnts[:, 0].max():.4f}], "
                          f"Y[{pnts[:, 1].min():.4f}, {pnts[:, 1].max():.4f}], "
                          f"Z[{pnts[:, 2].min():.4f}, {pnts[:, 2].max():.4f}]")
                
                batch_result = self.eval_points(pnts.to(device), all_planes, decoders, need_rgb=False).cpu().numpy()[:, -1]
                                                                                                             
                if visualize_planes_dir and not did_visualize:
                    try:
                        self._visualize_active_feature_planes(decoders.block_manager, visualize_planes_dir)
                    except Exception as e:
                        print(f"Warning: Failed to visualize feature plane:{e}")
                    did_visualize = True
                z_list.append(batch_result)
                
                               
                valid_mask = np.isfinite(batch_result)
                if valid_mask.any():
                    valid_sdf = batch_result[valid_mask]
                    sdf_stats['min_vals'].append(valid_sdf.min())
                    sdf_stats['max_vals'].append(valid_sdf.max())
                    sdf_stats['mean_vals'].append(valid_sdf.mean())
                    sdf_stats['valid_counts'].append(valid_mask.sum())
                    
                    if i < 3:            
                        print(f"batch{i+1} SDF: min={valid_sdf.min():.6f}, max={valid_sdf.max():.6f}, "
                              f"mean={valid_sdf.mean():.6f}, valid={valid_mask.sum()}/{len(batch_result)}")
            
            z = np.concatenate(z_list, axis=0)
                                      
                                             
            
                            
            if sdf_stats['min_vals']:
                print(f"SDF calculation completion statistics:")
                print(f"Global scope: min={min(sdf_stats['min_vals']):.6f}, max={max(sdf_stats['max_vals']):.6f}")
                print(f"Average range:{min(sdf_stats['mean_vals']):.6f} ~ {max(sdf_stats['mean_vals']):.6f}")
                print(f"Total number of valid points:{sum(sdf_stats['valid_counts'])}/{len(z)} ({sum(sdf_stats['valid_counts'])/len(z)*100:.1f}%)")
            else:
                print("Warning: All SDF calculation batches have no valid values!")
            
            if memory_tracking:
                try:
                    memory_after_sdf = process.memory_info().rss / 1024 / 1024
                    print(f"Memory usage after SDF calculation:{memory_after_sdf:.2f}MB (increase{memory_after_sdf - memory_after_grid:.2f} MB)")
                except (AttributeError, OSError) as e:
                    memory_tracking = False
                    print(f"Memory tracking failed ({type(e).__name__}: {e}), disabled")
                except Exception as e:
                    memory_tracking = False
                    print(f"Memory trace failed (unexpected exception{type(e).__name__}: {e}), disabled")

                              
            try:
                z_min = float(np.nanmin(z))
                z_max = float(np.nanmax(z))
            except Exception:
                z_min, z_max = float('nan'), float('nan')
            has_nan = bool(np.isnan(z).any())
            has_inf = bool(np.isinf(z).any())
            x_axis, y_axis, z_axis = grid['xyz']
            print(
                f"SDF statistics: min={z_min:.6f}, max={z_max:.6f}, level={self.level_set}; NaN={has_nan}, Inf={has_inf}\n"
                f"Grid axis: X({len(x_axis)}step) [{x_axis[0]:.4f},{x_axis[-1]:.4f}], "
                f"Y({len(y_axis)}step) [{y_axis[0]:.4f},{y_axis[-1]:.4f}], "
                f"Z({len(z_axis)}step) [{z_axis[0]:.4f},{z_axis[-1]:.4f}]"
            )

                                        
            try:
                lvl = float(self.level_set)
            except Exception:
                lvl = 0.0
            if not (np.isfinite(z_min) and np.isfinite(z_max) and z_min <= lvl <= z_max):
                print("Tip: level is not within the SDF value range, skip marching_cubes." \
                      "Possible reasons: insufficient training/fusion, boundary/normalization mismatch, prior and decoder distribution inconsistent." \
                      "You can try: (1) increase frames/coverage; (2) check bound/normalize; (3) temporarily use (min+max)/2 for chimney testing.")
                return

                                
            print("Execute marching cubes algorithm...")
            try:
                             
                x_axis, y_axis, z_axis = grid['xyz']
                original_shape = (len(y_axis), len(x_axis), len(z_axis))
                print(f"Voxel reshaping: Original z array length ={len(z)}, target shape={original_shape}")
                print(f"Calculation verification:{original_shape[0]}message cleaned to English{original_shape[1]}message cleaned to English{original_shape[2]}={original_shape[0]*original_shape[1]*original_shape[2]}")
                
                if version.parse(skimage.__version__) > version.parse('0.15.0'):
                            
                    volume = z.reshape(
                        grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2])
                    
                                 
                    vol_min, vol_max = volume.min(), volume.max()
                    vol_zero_crossings = ((volume[:-1] <= self.level_set) & (volume[1:] > self.level_set)).sum()
                    spacing = (grid['xyz'][0][2] - grid['xyz'][0][1],
                              grid['xyz'][1][2] - grid['xyz'][1][1], 
                              grid['xyz'][2][2] - grid['xyz'][2][1])
                    print(f"Reshaped voxels: shape ={volume.shape}, range=[{vol_min:.6f}, {vol_max:.6f}]")
                    print(f"Zero-crossing estimate ={vol_zero_crossings}, level={self.level_set}, spacing={spacing}")
                    
                                      
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=volume,
                        level=self.level_set,
                        spacing=spacing)
                else:
                            
                    volume = z.reshape(
                        grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2])
                    
                                 
                    vol_min, vol_max = volume.min(), volume.max()
                    vol_zero_crossings = ((volume[:-1] <= self.level_set) & (volume[1:] > self.level_set)).sum()
                    spacing = (grid['xyz'][0][2] - grid['xyz'][0][1],
                              grid['xyz'][1][2] - grid['xyz'][1][1],
                              grid['xyz'][2][2] - grid['xyz'][2][1])
                    print(f"Reshaped voxels: shape ={volume.shape}, range=[{vol_min:.6f}, {vol_max:.6f}]")
                    print(f"Zero-crossing estimate ={vol_zero_crossings}, level={self.level_set}, spacing={spacing}")
                    
                                      
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=volume,
                        level=self.level_set,
                        spacing=spacing)
            except Exception as e:
                print(f'marching_cubes error:{str(e)}. Possibly failed to extract surfaces from level set.')
                return

                     
            print(f"Number of vertices extracted:{len(verts):,}")
            vertices = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

                           
            del grid, points, mask, z, z_list, mask_list
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if color:
                print("Start calculating vertex colors...")
                               
                decoders.block_manager.active_blocks.clear()
                decoders.block_manager.block_usage_time.clear()
                
                                 
                gc.collect()
                torch.cuda.empty_cache()
                
                print(f"GPU memory status - allocated:{torch.cuda.memory_allocated(device) / 1024**3:.2f}GB, cache:{torch.cuda.memory_reserved(device) / 1024**3:.2f}GB")
                
                points = torch.from_numpy(verts).float()
                colors = torch.zeros(points.shape[0], 3)
                
                                 
                batch_size = min(2048, points.shape[0])           
                print(f"Color calculation batch size:{batch_size}, total number of batches:{(points.shape[0] + batch_size - 1) // batch_size}")
                
                for i, pnts in enumerate(torch.split(points, batch_size, dim=0)):
                    print(f"Process color batches{i+1}/{(points.shape[0] + batch_size - 1) // batch_size}, points:{len(pnts)}")
                    
                                  
                    if i > 0:
                        decoders.block_manager.active_blocks.clear()
                        decoders.block_manager.block_usage_time.clear()
                        torch.cuda.empty_cache()
                    
                    try:
                        z_color = self.eval_points(pnts.to(device).float(), all_planes, decoders, need_rgb=True).cpu()[..., :3]
                        colors[i * batch_size:i * batch_size + len(pnts)] = z_color
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"batch{i+1}Out of memory, try smaller batch size...")
                                         
                            sub_batch_size = min(512, len(pnts))
                            for j, sub_pnts in enumerate(torch.split(pnts, sub_batch_size, dim=0)):
                                decoders.block_manager.active_blocks.clear()
                                torch.cuda.empty_cache()
                                z_color = self.eval_points(sub_pnts.to(device).float(), all_planes, decoders, need_rgb=True).cpu()[..., :3]
                                start_idx = i * batch_size + j * sub_batch_size
                                colors[start_idx:start_idx + len(sub_pnts)] = z_color
                        else:
                            raise e
                    
                              
                    if (i + 1) % 3 == 0:          
                        torch.cuda.empty_cache()
                        
                print("Vertex color calculation completed")
                vertex_colors = colors.numpy()
            else:
                vertex_colors = None

                     
            print("Save grid...")
            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh.export(mesh_out_file)
            
                          
            if save_decoder_weights:
                self._save_decoder_and_features(mesh_out_file, decoders, all_planes, device)
            
            if self.verbose:
                print(f'Successfully saved mesh to{mesh_out_file}')
                print(f'Grid Statistics:{len(vertices):,}vertex,{len(faces):,}noodle')

                                       
            try:
                stats = getattr(self, '_near_surface_stats', None)
                if stats and len(stats) > 0:
                    total_count = sum(int(s.get('count', 0)) for s in stats)
                    if total_count > 0:
                                    
                        weighted_bias_sum = sum(float(s.get('bias', 0.0)) * int(s.get('count', 0)) for s in stats)
                        mean_bias = weighted_bias_sum / total_count
                                                                                  
                        import math
                        var_sum = 0.0
                        for s in stats:
                            n = int(s.get('count', 0))
                            if n <= 0:
                                continue
                            b = float(s.get('bias', 0.0))
                            sd = float(s.get('std', 0.0))
                            var_sum += n * (sd * sd + (b - mean_bias) * (b - mean_bias))
                        pooled_std = math.sqrt(var_sum / total_count)
                        print(f"[Mesher.get_mesh] aggregated near-surface stats over {len(stats)} batches: ")
                        print(f"  count={total_count}, mean_bias={mean_bias:.6f}, pooled_std={pooled_std:.6f}")
            except Exception as _:
                pass
