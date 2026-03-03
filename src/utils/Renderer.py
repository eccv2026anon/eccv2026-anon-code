import torch
import os
import time
import numpy as np
from src.common import get_rays, sample_pdf, normalize_3d_coordinate


class Renderer(object):
       

                                                
                       
                                     
                                        
    def __init__(self, cfg, structrecon, ray_batch_size=20000):
                                               
        try:
            rb_cfg = cfg.get('rendering', {}).get('ray_batch_size', None)
            self.ray_batch_size = int(rb_cfg) if rb_cfg is not None else int(ray_batch_size)
        except Exception:
            self.ray_batch_size = int(ray_batch_size)

        self.perturb = cfg['rendering']['perturb']                                                     
        self.n_stratified = cfg['rendering']['n_stratified']              
        self.n_importance = cfg['rendering']['n_importance']                                

                            
        mapping_cfg = cfg.get('mapping', {})
        self.enable_pgis = bool(mapping_cfg.get('enable_pgis', True))            
        self.pgis_tau_s = float(mapping_cfg.get('pgis_tau_s', 1.0))              
        self.pgis_epsilon = float(mapping_cfg.get('pgis_epsilon', 0.001))          
        self.pgis_eta = float(mapping_cfg.get('pgis_eta', 0.5))                
        self.pgis_n_stratified = int(mapping_cfg.get('pgis_n_stratified', 64))              
        self.pgis_n_prior = int(mapping_cfg.get('pgis_n_prior', 64))                  
        self.pgis_min_spacing = float(mapping_cfg.get('pgis_min_spacing', 0.25))              

                                        
        self.amp_enabled = bool(cfg.get('amp', True))
                
        self.verbose = bool(cfg.get('verbose', False)) or bool(getattr(structrecon, 'verbose', False))
                            
        self.log_rgb_stats = bool(cfg.get('rendering', {}).get('log_rgb_stats', False))
        self.log_debug = bool(cfg.get('rendering', {}).get('log_debug', False))
        self.log_weight_stats = bool(cfg.get('rendering', {}).get('log_weight_stats', False))

                                               
        sb = cfg.get('rendering', {}).get('surface_bias', {}) if isinstance(cfg, dict) else {}
        self.surface_bias_enable = bool(sb.get('enable', True))
        self.surface_bias_lambda = float(sb.get('lambda', 0.35))               
        self.surface_bias_topk = int(sb.get('topk', 3))                    
                                               
        self.output_gamma = bool(cfg.get('rendering', {}).get('output_gamma', False))

        self.scale = cfg['scale']                                     
        self.bound = structrecon.bound.to(structrecon.device,
                                    non_blocking=True)                                                                  

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = structrecon.H, structrecon.W, structrecon.fx, structrecon.fy, structrecon.cx, structrecon.cy             

                              
        self.device = structrecon.device
        self._preallocate_tensors()

                            
        prof_cfg = cfg.get('profiling', {}) if isinstance(cfg, dict) else {}
        self.profile_enable = bool(prof_cfg.get('enable', False))
        self.profile_every_n = int(prof_cfg.get('every_n', 1))
        self._profile_step = 0
              
        try:
            out_dir = cfg.get('data', {}).get('output', 'output') if isinstance(cfg, dict) else 'output'
            self.profile_dir = os.path.join(out_dir, 'profile')
            os.makedirs(self.profile_dir, exist_ok=True)
            self.renderer_profile_csv = os.path.join(self.profile_dir, 'renderer_profile.csv')
            if self.profile_enable and (not os.path.exists(self.renderer_profile_csv)):
                with open(self.renderer_profile_csv, 'w', encoding='utf-8') as f:
                    f.write('step,n_rays,coarse_forward_ms,coarse_weight_ms,importance_ms,fine_forward_ms,accum_ms,total_ms\n')
        except Exception:
            self.profile_enable = False

    def _preallocate_tensors(self):
           
                   
        self.t_vals_uni_cache = torch.linspace(0., 1., steps=self.n_stratified, device=self.device)
        self.t_vals_surface_cache = torch.linspace(0., 1., steps=self.n_importance, device=self.device)

                        
        self.ones_cache = {}
        common_sizes = [1, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]
        for size in common_sizes:
            self.ones_cache[size] = torch.ones(size, 1, device=self.device)

    def perturbation(self, z_vals):
           
                                       
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
                                               
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def sample_with_pgis(self, rays_o, rays_d, near, far, Phi_prior, voxel_size, origin_xyz, device):
           
        n_rays = rays_o.shape[0]
        M = self.pgis_n_stratified + self.pgis_n_prior         

                           
        t_vals = torch.linspace(0.0, 1.0, M, device=device)       
        z_candidates = near + (far - near) * t_vals       

                      
        z_candidates_expanded = z_candidates.unsqueeze(0).expand(n_rays, -1)               
        rays_d_expanded = rays_d.unsqueeze(1).expand(-1, M, -1)                  
        rays_o_expanded = rays_o.unsqueeze(1).expand(-1, M, -1)                  

        p_candidates = rays_o_expanded + rays_d_expanded * z_candidates_expanded.unsqueeze(-1)                  

                        
        phi_prior_vals, gradients = self._query_prior_tsdf(p_candidates.view(-1, 3),
                                                          Phi_prior, voxel_size, origin_xyz, device)

        phi_prior_vals = phi_prior_vals.view(n_rays, M)               
        gradients = gradients.view(n_rays, M, 3)                  

                        
                                                                              
        phi_abs = torch.abs(phi_prior_vals)               

                               
                        
        voxel_coords_xyz = (p_candidates.view(-1, 3) - torch.tensor(origin_xyz, device=device)) / voxel_size
        voxel_coords_zyx = voxel_coords_xyz[:, [2, 1, 0]]
        grid_shape = Phi_prior.shape
        valid_coords = (voxel_coords_zyx >= 0) & (voxel_coords_zyx < torch.tensor(grid_shape, device=device))
        m = valid_coords.all(dim=-1).view(n_rays, M)

        grad_norms = torch.norm(gradients, dim=-1)               

                  
        exp_term = torch.exp(-phi_abs / self.pgis_tau_s)
        grad_term = (self.pgis_epsilon + grad_norms) ** self.pgis_eta
        tilde_pi = m.float() * exp_term * grad_term               

                     
        pi = tilde_pi / (torch.sum(tilde_pi, dim=-1, keepdim=True) + 1e-8)               

                           
        N_p = self.pgis_n_prior
        N_s = self.pgis_n_stratified

                                 
        indices = torch.multinomial(pi, N_p, replacement=True)                 

                  
        z_prior = torch.gather(z_candidates_expanded, 1, indices)                 

                    
        t_stratified = torch.linspace(0.0, 1.0, N_s, device=device)
        z_stratified = near + (far - near) * t_stratified         
        z_stratified = z_stratified.unsqueeze(0).expand(n_rays, -1)                 

                
        if self.perturb:
            z_stratified = self.perturbation(z_stratified)

                     
        z_all = torch.cat([z_stratified, z_prior], dim=-1)                       
        z_sorted, _ = torch.sort(z_all, dim=-1)                     

                                     
        z_final = self._enforce_min_spacing(z_sorted, self.pgis_min_spacing * voxel_size)

                                          
        n_total = self.pgis_n_stratified + self.pgis_n_prior
        if z_final.shape[-1] != n_total:
                             
            if z_final.shape[-1] > n_total:
                z_final = z_final[..., :n_total]
            else:
                                  
                last_vals = z_final[..., -1:].expand(-1, n_total - z_final.shape[-1])
                z_final = torch.cat([z_final, last_vals], dim=-1)

        return z_final

    def _query_prior_tsdf(self, points, Phi_prior, voxel_size, origin_xyz, device):
           
                      
        voxel_coords_xyz = (points - torch.tensor(origin_xyz, device=device)) / voxel_size
        voxel_coords = voxel_coords_xyz[:, [2, 1, 0]]

                            
        if isinstance(Phi_prior, np.ndarray):
            Phi_prior = torch.from_numpy(Phi_prior).to(device)
        elif Phi_prior.device != device:
            Phi_prior = Phi_prior.to(device)

                    
        grid_shape = Phi_prior.shape

                      
        voxel_coords_int = torch.round(voxel_coords).long()

                    
        voxel_coords_int = torch.clamp(voxel_coords_int,
                                     torch.tensor([0, 0, 0], device=device),
                                     torch.tensor([grid_shape[0]-1, grid_shape[1]-1, grid_shape[2]-1], device=device))

                 
        phi_vals = Phi_prior[voxel_coords_int[:, 0], voxel_coords_int[:, 1], voxel_coords_int[:, 2]]

                         
        gradients = torch.zeros_like(points, device=device)
                                     
        eps = 1e-3
        for i in range(3):
                         
            offset = torch.zeros(1, 3, device=device)
            offset[0, i] = eps
            pos_offset = points + offset
            neg_offset = points - offset

                         
            pos_coords_xyz = (pos_offset - torch.tensor(origin_xyz, device=device)) / voxel_size
            neg_coords_xyz = (neg_offset - torch.tensor(origin_xyz, device=device)) / voxel_size
            pos_coords = torch.round(pos_coords_xyz[:, [2, 1, 0]]).long()
            neg_coords = torch.round(neg_coords_xyz[:, [2, 1, 0]]).long()

                  
            pos_coords = torch.clamp(pos_coords, torch.tensor([0, 0, 0], device=device),
                                   torch.tensor([grid_shape[0]-1, grid_shape[1]-1, grid_shape[2]-1], device=device))
            neg_coords = torch.clamp(neg_coords, torch.tensor([0, 0, 0], device=device),
                                   torch.tensor([grid_shape[0]-1, grid_shape[1]-1, grid_shape[2]-1], device=device))

            pos_vals = Phi_prior[pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2]]
            neg_vals = Phi_prior[neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2]]

            gradients[:, i] = (pos_vals - neg_vals) / (2 * eps * voxel_size)

        return phi_vals, gradients

    def _enforce_min_spacing(self, z_vals, min_spacing):
           
                    
        z_sorted, _ = torch.sort(z_vals, dim=-1)

                             
        n_rays, n_points = z_sorted.shape
        z_filtered = []

        for ray_idx in range(n_rays):
            z_ray = z_sorted[ray_idx]
                      
            filtered_points = [z_ray[0]]

            for i in range(1, n_points):
                if z_ray[i] - filtered_points[-1] >= min_spacing:
                    filtered_points.append(z_ray[i])

                                 
            while len(filtered_points) < n_points:
                filtered_points.append(filtered_points[-1])

            z_filtered.append(filtered_points)

        return torch.tensor(z_filtered, device=z_vals.device, dtype=z_vals.dtype)

    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None, need_rgb=True,
                        Phi_prior=None, voxel_size=None, origin_xyz=None):
           
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]
                         
        color_surface = None
        depth_surface = None

                                                     
        near = 0.02
                          
        t_vals_uni = self.t_vals_uni_cache
        t_vals_surface = self.t_vals_surface_cache

        gt_mask = torch.zeros(n_rays, dtype=torch.bool, device=device)

                                 
        if gt_depth is not None:
                                         
            gt_depth_flat = gt_depth.reshape(-1)

                                                 
            if gt_depth_flat.shape[0] != n_rays:
                print(f"warning: gt_depth shape({gt_depth_flat.shape[0]}) and n_rays({n_rays}) does not match")

                                       
                if gt_depth_flat.shape[0] < n_rays:
                                           
                    temp_gt_depth = torch.zeros(n_rays, device=device)
                    temp_gt_depth[:gt_depth_flat.shape[0]] = gt_depth_flat
                    gt_depth_flat = temp_gt_depth
                else:
                                              
                    gt_depth_flat = gt_depth_flat[:n_rays]

                                              
            gt_mask = (gt_depth_flat > 0)

                                
        use_pgis = (self.enable_pgis and Phi_prior is not None and
                   voxel_size is not None and origin_xyz is not None)

                     
        if hasattr(decoders, 'use_block_manager') and decoders.use_block_manager:
                             
                            
            if use_pgis:
                             
                near = 0.02
                far = 10.0                       
                z_vals_coarse = self.sample_with_pgis(rays_o, rays_d, near, far,
                                                    Phi_prior, voxel_size, origin_xyz, device)
                                
                n_coarse = z_vals_coarse.shape[-1]
            else:
                        
                z_vals_coarse = torch.empty([n_rays, n_stratified], device=device)

                               
            if gt_mask.any():
                gt_nonezero = gt_depth_flat[gt_mask].unsqueeze(-1)
                if gt_nonezero.numel() > 0:
                                                          
                    near_gt = 0.8 * gt_nonezero
                    far_gt = gt_nonezero + 0.5
                    z_vals_surf = near_gt * (1. - t_vals_uni) + far_gt * t_vals_uni
                    if self.perturb:
                        z_vals_surf = self.perturbation(z_vals_surf)
                    z_vals_coarse[gt_mask] = z_vals_surf

                                
            no_gt_mask = ~gt_mask
            if no_gt_mask.any():
                rays_o_uni = rays_o[no_gt_mask].detach()
                rays_d_uni = rays_d[no_gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)             
                det_rays_d = rays_d_uni.unsqueeze(-1)             
                t = (self.bound.unsqueeze(0) - det_rays_o) / det_rays_d             
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                z_vals_coarse[no_gt_mask] = z_vals_uni

                    
            pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_coarse[..., :,
                                                                       None]                             

                       
            t_total_start = time.perf_counter() if self.profile_enable else None
            with torch.no_grad():
                                                  
                                                 
                                                                     
                                                                         
                from torch.cuda.amp import autocast
                t_coarse_fwd_start = time.perf_counter() if self.profile_enable else None
                with autocast(enabled=self.amp_enabled):
                                                        
                    raw_coarse = decoders(pts_coarse, all_planes, need_rgb=False)
                sdf_coarse = raw_coarse['sdf']
                t_coarse_fwd_end = time.perf_counter() if self.profile_enable else None

                      
                alpha_coarse = self.sdf2alpha(sdf_coarse, decoders.beta)

                                        
                if len(alpha_coarse.shape) == 1:
                    alpha_coarse = alpha_coarse.unsqueeze(-1)

                                
                                                                            

                                    
                                          
                                                                          
                                                 
                ones_tensor = alpha_coarse.new_ones((*alpha_coarse.shape[:-2], 1, alpha_coarse.shape[-1]))

                                                          
                transmittance_with_one = torch.cat([ones_tensor, (1. - alpha_coarse + 1e-10)], dim=-2)

                                                           
                transmittance = torch.cumprod(transmittance_with_one, dim=-2)[..., :-1, :]

                                                              
                weights_coarse = alpha_coarse * transmittance
                                        
                t_coarse_weight_end = time.perf_counter() if self.profile_enable else None

                              
                z_vals_mid = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])

                                                      
                weights_for_sample = weights_coarse[..., 1:-1, :]
                if weights_for_sample.numel() == 0:
                    print("Warning: weights_coarse is empty, use uniform sampling")
                                                        
                    base = z_vals_mid if z_vals_mid.numel() > 0 else z_vals_coarse
                    if base.numel() > 0:
                        z_min = base.min(dim=-1, keepdim=True)[0]
                        z_max = base.max(dim=-1, keepdim=True)[0]
                        t_vals = torch.linspace(0., 1., steps=n_importance, device=device)
                        z_samples = z_min + (z_max - z_min) * t_vals
                    else:
                                       
                        z_samples = torch.linspace(0.1, 1.0, steps=n_importance, device=device)
                        z_samples = z_samples.expand(max(1, z_vals_mid.shape[0]), n_importance)
                else:
                                     
                    if len(weights_for_sample.shape) < 2:
                        print(f"Adjust weights_coarse shape:{weights_for_sample.shape}")
                                              
                        if len(weights_for_sample.shape) == 1:
                            weights_for_sample = weights_for_sample.unsqueeze(0)

                                                            
                    if z_vals_mid.shape[0] != weights_for_sample.shape[0]:
                        print(f"z_vals_mid shape:{z_vals_mid.shape}, weights_for_sample shape:{weights_for_sample.shape}")
                        if z_vals_mid.shape[0] == 1:
                            z_vals_mid = z_vals_mid.expand(weights_for_sample.shape[0], -1)
                        elif weights_for_sample.shape[0] == 1:
                            weights_for_sample = weights_for_sample.expand(z_vals_mid.shape[0], -1)

                                                
                    if weights_for_sample.dim() == 3 and weights_for_sample.shape[-1] == 1:
                        weights_for_sample = weights_for_sample.squeeze(-1)
                    z_samples = sample_pdf(z_vals_mid, weights_for_sample, n_importance, det=False, device=device)

                z_samples = z_samples.detach()

                           
            z_vals, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
            t_importance_end = time.perf_counter() if self.profile_enable else None

                              
                     
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]                                          

                                     
            from torch.cuda.amp import autocast
                                            
            t_fine_fwd_start = time.perf_counter() if self.profile_enable else None
            with autocast(enabled=self.amp_enabled):
                raw = decoders(pts, all_planes, need_rgb=need_rgb)
                                                        
            if self.log_debug:
                try:
                    keys = list(raw.keys()) if isinstance(raw, dict) else []
                    print("[DBG2] raw keys:", keys)
                    if isinstance(raw, dict):
                        for k in keys:
                            v = raw[k]
                            shp = tuple(v.shape) if hasattr(v, 'shape') else type(v)
                            print(f"[DBG2] key={k}, shape={shp}, dtype={getattr(v, 'dtype', None)}, device={getattr(v, 'device', None)}")
                except Exception as e:
                    if self.log_debug:
                        print("[DBG2] inspect raw failed:", e)
            sdf = raw['sdf']
            t_fine_fwd_end = time.perf_counter() if self.profile_enable else None

                                     
            if 'rgb' in raw:
                color_feat = raw['rgb']
            else:
                                                      
                available_keys = list(raw.keys())
                print(f"Warning: There is no 'rgb' key in the raw dictionary. Available keys:{available_keys}")
                color_keys = [k for k in available_keys if k != 'sdf']
                if color_keys:
                    color_feat = raw[color_keys[0]]
                    print(f"use'{color_keys[0]}'Keys as color features")
                else:
                                           
                    print("Error: No color feature key in raw dictionary, use zero tensor instead")
                    color_feat = torch.zeros_like(sdf).expand(*sdf.shape[:-1], 3)

                                                    
                             
            deltas = z_vals[..., 1:] - z_vals[..., :-1]
                                
            last_delta = deltas[..., -1:]
            deltas = torch.cat([deltas, last_delta], dim=-1)          
            deltas = deltas.unsqueeze(-1)             
                                                   
            beta = getattr(decoders, 'beta', 10.0)
            sigma = torch.sigmoid(-sdf * beta)
                                                    
            alpha = 1. - torch.exp(-sigma * beta * deltas)

                             
            if len(alpha.shape) == 1:
                alpha = alpha.unsqueeze(-1)

                            
                                                                        
                                
            ones_tensor = alpha.new_ones((*alpha.shape[:-2], 1, alpha.shape[-1]))
                
            transmittance_with_one = torch.cat([ones_tensor, (1. - alpha + 1e-10)], dim=-2)
                             
            transmittance = torch.cumprod(transmittance_with_one, dim=-2)[..., :-1, :]
                    
            weights = alpha * transmittance
                                          
            if self.log_weight_stats:
                try:
                    wsum = weights.squeeze(-1).sum(-1)
                    small_frac = float((wsum < 1e-3).float().mean().item())
                    print(f"[DBG3] weights small_frac: {small_frac:.3f}")
                except Exception as e:
                    if self.log_debug:
                        print("[DBG3] weight stats failed:", e)

                          
            rendered_rgb_numer = torch.sum(weights * color_feat, -2)
            weight_sum_raw = weights.sum(-2)                   
            weight_sum_rgb = weight_sum_raw.clamp(min=1e-3)
            rendered_rgb = rendered_rgb_numer / weight_sum_rgb
            rendered_depth = torch.sum(weights.squeeze(-1) * z_vals, -1)

                                                          
                                                                        
                              
            depth_var = torch.sum(weights.squeeze(-1) * (z_vals - rendered_depth.unsqueeze(-1))**2, dim=-1)
                                            
            depth_std = torch.sqrt(depth_var + 1e-6)
                                                                               

                                               
            if gt_depth is None and self.surface_bias_enable:
                try:
                    w2 = weights.squeeze(-1)         
                    k = max(1, min(self.surface_bias_topk, w2.shape[-1]))
                    topv, topi = torch.topk(w2, k=k, dim=-1)         
                    topv_norm = topv / (topv.sum(dim=-1, keepdim=True) + 1e-8)
                                     
                    gi = topi.unsqueeze(-1).expand(-1, -1, color_feat.shape[-1])           
                    color_topk = torch.gather(color_feat, dim=-2, index=gi)           
                    color_topk_mix = (topv_norm.unsqueeze(-1) * color_topk).sum(dim=-2)         
                    lam = float(self.surface_bias_lambda)
                    rendered_rgb = (1.0 - lam) * rendered_rgb + lam * color_topk_mix
                except Exception:
                    pass

                                     
            if self.log_debug:
                try:
                                                                 
                    cf = color_feat.detach()
                    rr = rendered_rgb.detach()
                    cf_mean = [float(cf[..., i].mean().item()) for i in range(min(3, cf.shape[-1]))]
                    cf_std = [float(cf[..., i].std().item()) for i in range(min(3, cf.shape[-1]))]
                    rr_mean = [float(rr[..., i].mean().item()) for i in range(min(3, rr.shape[-1]))]
                    rr_std = [float(rr[..., i].std().item()) for i in range(min(3, rr.shape[-1]))]
                    print(f"[DBG2] color_feat mean={cf_mean} std={cf_std}")
                    print(f"[DBG2] rendered_rgb mean={rr_mean} std={rr_std}")
                    wsum_stats = (
                        float(weight_sum_raw.min().item()),
                        float(weight_sum_raw.max().item()),
                        float(weight_sum_raw.mean().item())
                    )
                    print(f"[DBG2] weight_sum_raw min/max/mean={wsum_stats}")
                except Exception as e:
                    if self.log_debug:
                        print("[DBG2] stats failed:", e)

                                                                 
            weight_sum = weights.squeeze(-1).sum(-1)       
            small_mask = weight_sum < 1e-3
            try:
                if self.log_weight_stats and n_rays <= 10000:          
                    frac_small = float(small_mask.float().mean().item())
                    print(
                        f"[Renderer] weights sum: min={float(weight_sum.min().item()):.2e}, max={float(weight_sum.max().item()):.2e}, mean={float(weight_sum.mean().item()):.2e}, small_frac={frac_small:.3f}")
            except Exception:
                pass
            if small_mask.any():
                      
                z = z_vals[small_mask]              
                                                    
                if gt_depth is not None:
                    gt_d_sel = gt_depth[small_mask].unsqueeze(-1)             
                    sigma = max(1e-3, float(truncation))
                    w = torch.exp(-0.5 * ((z - gt_d_sel) / sigma) ** 2)
                    w = w / (w.sum(-1, keepdim=True) + 1e-8)
                else:
                    w = torch.full_like(z, 1.0 / z.shape[-1])
                        
                rendered_depth[small_mask] = (w * z).sum(-1)
                           
                cf = color_feat[small_mask]                 
                rendered_rgb[small_mask] = (w.unsqueeze(-1) * cf).sum(-2)

                                      
            if self.log_debug and (rendered_depth.max() - rendered_depth.min()) < 1e-3:
                print(
                    f"Debugging: Render depth almost constant range={float((rendered_depth.max() - rendered_depth.min()).item()):.6f}, min={float(rendered_depth.min().item()):.6f}, max={float(rendered_depth.max().item()):.6f}")
                    
            try:
                rgb_min = rendered_rgb.amin(
                    dim=0).mean().item() if rendered_rgb.dim() > 1 else rendered_rgb.min().item()
                rgb_max = rendered_rgb.amax(
                    dim=0).mean().item() if rendered_rgb.dim() > 1 else rendered_rgb.max().item()
                rgb_mean = rendered_rgb.mean().item()
                if self.log_rgb_stats:
                    print(f"[Renderer] RGB stats: mean={rgb_mean:.4f}, min{rgb_min:.4f}, max{rgb_max:.4f}")
            except Exception:
                pass

                          
            if self.profile_enable and (self._profile_step % self.profile_every_n == 0):
                try:
                                 
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    coarse_ms = (t_coarse_fwd_end - t_coarse_fwd_start) * 1000.0 if ('t_coarse_fwd_start' in locals() and 't_coarse_fwd_end' in locals()) else 0.0
                    coarse_w_ms = (t_coarse_weight_end - t_coarse_fwd_end) * 1000.0 if ('t_coarse_weight_end' in locals() and 't_coarse_fwd_end' in locals()) else 0.0
                    imp_ms = (t_importance_end - t_coarse_weight_end) * 1000.0 if ('t_importance_end' in locals() and 't_coarse_weight_end' in locals()) else 0.0
                    fine_ms = (t_fine_fwd_end - t_fine_fwd_start) * 1000.0 if ('t_fine_fwd_start' in locals() and 't_fine_fwd_end' in locals()) else 0.0
                    accum_ms = (time.perf_counter() - t_fine_fwd_end) * 1000.0 if 't_fine_fwd_end' in locals() else 0.0
                    total_ms = (time.perf_counter() - t_total_start) * 1000.0 if t_total_start is not None else 0.0
                    with open(self.renderer_profile_csv, 'a', encoding='utf-8') as f:
                        f.write(f"{self._profile_step},{int(n_rays)},{coarse_ms:.3f},{coarse_w_ms:.3f},{imp_ms:.3f},{fine_ms:.3f},{accum_ms:.3f},{total_ms:.3f}\n")
                except Exception:
                    pass
            if self.profile_enable:
                self._profile_step += 1

                         
            result_dict = {
                'depth': rendered_depth,
                'color': rendered_rgb,
                'sdf': sdf,
                'z_vals': z_vals,
                'weight_sum': weight_sum_raw.squeeze(-1),
                'color_surface': color_surface if color_surface is not None else None,
                'depth_surface': depth_surface if depth_surface is not None else None,
                'depth_std': depth_std,                  
            }
        else:
                    
            z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)

                               
            if gt_mask.any():
                gt_nonezero = gt_depth_flat[gt_mask].unsqueeze(-1)
                if gt_nonezero.numel() > 0:
                                                                    
                    gt_depth_surface = gt_nonezero.expand(-1, n_importance)
                    z_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)

                    gt_depth_free = gt_nonezero.expand(-1, n_stratified)
                    z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

                    z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
                    if self.perturb:
                        z_vals_nonzero = self.perturbation(z_vals_nonzero)

                                                            
                    if gt_mask.sum() == z_vals_nonzero.shape[0]:
                        z_vals[gt_mask] = z_vals_nonzero
                    else:
                                                
                        print(
                            f"Warning: Number of True in gt_mask({gt_mask.sum()}) with z_vals_nonzero shape ({z_vals_nonzero.shape[0]}) does not match, use no depth processing")
                        gt_mask.fill_(False)
                else:
                                                  
                    gt_mask.fill_(False)

                                            
                              
            if not gt_mask.any():
                with torch.no_grad():
                    rays_o_uni = rays_o.detach()
                    rays_d_uni = rays_d.detach()
                    det_rays_o = rays_o_uni.unsqueeze(-1)
                    det_rays_d = rays_d_uni.unsqueeze(-1)
                    t = (self.bound.unsqueeze(0) - det_rays_o) / det_rays_d
                    far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    far_bb = far_bb.unsqueeze(-1)
                    far_bb += 0.01
                    z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                    if self.perturb:
                        z_vals_uni = self.perturbation(z_vals_uni)
                    z_vals.fill_(0)       
                    z_vals = z_vals_uni.repeat(1, (n_stratified + n_importance) // n_stratified)

                                
            no_gt_mask = ~gt_mask
            if no_gt_mask.any():
                with torch.no_grad():
                    rays_o_uni = rays_o[no_gt_mask].detach()
                    rays_d_uni = rays_d[no_gt_mask].detach()
                    det_rays_o = rays_o_uni.unsqueeze(-1)             
                    det_rays_d = rays_d_uni.unsqueeze(-1)             
                    t = (self.bound.unsqueeze(0) - det_rays_o) / det_rays_d             
                    far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    far_bb = far_bb.unsqueeze(-1)
                    far_bb += 0.01

                    z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                    if self.perturb:
                        z_vals_uni = self.perturbation(z_vals_uni)
                    pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(
                        -1)                             

                    pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                    sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes)
                    sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                    alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)

                                         
                    if len(alpha_uni.shape) == 1:
                        alpha_uni = alpha_uni.unsqueeze(-1)

                                             
                    ones_shape = list(alpha_uni.shape)
                    ones_shape[-1] = 1
                    ones_tensor = torch.ones(ones_shape, device=device)

                    weights_uni = alpha_uni * torch.cumprod(torch.cat([ones_tensor, (1. - alpha_uni + 1e-10)], -1), -1)[
                                              :, :-1]

                    z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])

                                         
                    weights_for_sample = weights_uni[..., 1:-1]
                    if weights_for_sample.numel() == 0:
                        print("Warning: weights_uni is empty, use uniform sampling")
                                              
                        z_min = z_vals_uni_mid.min(dim=-1, keepdim=True)[0]
                        z_max = z_vals_uni_mid.max(dim=-1, keepdim=True)[0]
                        t_vals = torch.linspace(0., 1., steps=n_importance, device=device)
                        z_samples_uni = z_min + (z_max - z_min) * t_vals
                    else:
                                         
                        if len(weights_for_sample.shape) < 2:
                            print(f"Adjust weights_uni shape:{weights_for_sample.shape}")
                                                  
                            if len(weights_for_sample.shape) == 1:
                                weights_for_sample = weights_for_sample.unsqueeze(0)

                                                                    
                        if z_vals_uni_mid.shape[0] != weights_for_sample.shape[0]:
                            print(
                                f"z_vals_uni_mid shape:{z_vals_uni_mid.shape}, weights_for_sample shape:{weights_for_sample.shape}")
                            if z_vals_uni_mid.shape[0] == 1:
                                z_vals_uni_mid = z_vals_uni_mid.expand(weights_for_sample.shape[0], -1)
                            elif weights_for_sample.shape[0] == 1:
                                weights_for_sample = weights_for_sample.expand(z_vals_uni_mid.shape[0], -1)

                        z_samples_uni = sample_pdf(z_vals_uni_mid, weights_for_sample, n_importance, det=False,
                                                   device=device)

                    z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                    z_vals[no_gt_mask] = z_vals_uni

                   
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]                                          
            raw = decoders(pts, all_planes)
            sdf = raw['sdf']
            color_feat = raw['rgb'] if isinstance(raw, dict) and 'rgb' in raw else raw[..., :-1]

                  
            alpha = self.sdf2alpha(sdf, decoders.beta)

                             
            if len(alpha.shape) == 1:
                alpha = alpha.unsqueeze(-1)

                                 
            ones_shape = list(alpha.shape)
            ones_shape[-1] = 1
            ones_tensor = torch.ones(ones_shape, device=device)

            weights = alpha * torch.cumprod(torch.cat([ones_tensor, (1. - alpha + 1e-10)], -1), -1)[:, :-1]

            rendered_rgb_numer = torch.sum(weights[..., None] * color_feat, -2)
            weight_sum_raw = weights.sum(-1)          
            weight_sum_rgb = weight_sum_raw.unsqueeze(-1).clamp(min=1e-3)
            rendered_rgb = rendered_rgb_numer / weight_sum_rgb
            rendered_depth = torch.sum(weights * z_vals, -1)

                                                          
                                                           
            depth_var = torch.sum(weights * (z_vals - rendered_depth.unsqueeze(-1))**2, dim=-1)
            depth_std = torch.sqrt(depth_var + 1e-6)
                                                                               

                                                    
            if torch.rand(1).item() < 0.001:            
                try:
                    print(f"\n{'='*60}")
                    print(f"[Renderer Debug - SDF2Depth conversion diagnosis]")
                    beta_val = decoders.beta.item() if hasattr(decoders.beta, 'item') else decoders.beta
                    print(f"Beta parameters:{beta_val:.2f}")
                    
                                
                    ray_idx = 0
                    z_sample = z_vals[ray_idx]
                    sdf_sample = sdf[ray_idx] if len(sdf.shape) > 1 else sdf
                    alpha_sample = alpha[ray_idx] if len(alpha.shape) > 1 else alpha
                    weights_sample = weights[ray_idx] if len(weights.shape) > 1 else weights
                    
                    print(f"Number of sampling points:{len(z_sample)}")
                    print(f"z_vals range: [{z_sample.min():.3f}, {z_sample.max():.3f}]rice")
                    print(f"sdf range: [{sdf_sample.min():.4f}, {sdf_sample.max():.4f}]")
                    print(f"alpha range: [{alpha_sample.min():.4f}, {alpha_sample.max():.4f}]")
                    print(f"weights range: [{weights_sample.min():.6f}, {weights_sample.max():.6f}]")
                    print(f"Sum of weights:{weights_sample.sum():.4f}")
                    
                            
                    rd = rendered_depth[ray_idx].item() if rendered_depth.numel() > 1 else rendered_depth.item()
                    print(f"Render depth:{rd:.3f}rice")
                    
                    if gt_depth is not None and gt_depth_flat.numel() > 0:
                        gt_d = gt_depth_flat[ray_idx].item()
                        if gt_d > 0:
                            print(f"GT Depth:{gt_d:.3f}rice")
                            print(f"Depth error:{abs(rd - gt_d):.4f}rice")
                    
                            
                    max_idx = weights_sample.argmax().item()
                    print(f"Weight peak position: index{max_idx}, depth{z_sample[max_idx]:.3f}rice")
                    
                               
                    start_idx = max(0, max_idx - 2)
                    end_idx = min(len(z_sample), max_idx + 3)
                    print(f"Sampling points near the peak:")
                    for i in range(start_idx, end_idx):
                        marker = "peak" if i == max_idx else ""
                        print(f"    [{i}] z={z_sample[i]:.3f}, sdf={sdf_sample[i]:+.4f}, "
                              f"a={alpha_sample[i]:.4f}, w={weights_sample[i]:.6f}{marker}")
                    print(f"{'='*60}\n")
                except Exception as e_debug:
                    print(f"[Renderer Debug] Diagnosis failed:{e_debug}")
                                                                  

                                                 
            weight_sum = weights.sum(-1)
            small_mask = weight_sum < 1e-3
            if small_mask.any():
                z = z_vals[small_mask]
                if gt_depth is not None:
                    gt_d_sel = gt_depth_flat[small_mask].unsqueeze(-1)
                    sigma = max(1e-3, float(truncation))
                    w = torch.exp(-0.5 * ((z - gt_d_sel) / sigma) ** 2)
                    w = w / (w.sum(-1, keepdim=True) + 1e-8)
                else:
                                               
                    w = (z * 0 + 1.0) / z.shape[-1]
                    w = w.unsqueeze(-1)
                    weights = weights.clone()
                    try:
                        weights[small_mask] = w
                    except Exception:
                        pass

                                                     
                color_surface = None
                depth_surface = None
                try:
                                                                         
                    w2 = weights.squeeze(-1)
                    idx = torch.argmax(w2, dim=-1)       
                    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, color_feat.shape[-1])           
                    color_surface = torch.gather(color_feat, dim=-2, index=gather_idx).squeeze(-2)         
                    depth_surface = torch.gather(z_vals, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)       
                    if self.log_debug:
                        try:
                            cs_mean = [float(color_surface[:, i].mean().item()) for i in range(3)]
                            cs_std = [float(color_surface[:, i].std().item()) for i in range(3)]
                            print(f"[DBG2] color_surface mean={cs_mean} std={cs_std}")
                        except Exception:
                            pass
                except Exception as e:
                    if self.log_debug:
                        print("[DBG2] extract color_surface failed:", e)

                        
                rendered_depth[small_mask] = (w * z).sum(-1)
                cf = color_feat[small_mask]
                rendered_rgb[small_mask] = (w.unsqueeze(-1) * cf).sum(-2)

                     
            depth_flat = rendered_depth.reshape(-1)
            color_flat = rendered_rgb.reshape(-1, 3)
                                  
            if gt_depth is None and self.output_gamma:
                color_flat = color_flat.clamp(0, 1) ** (1.0 / 2.2)

                         
            result_dict = {
                'depth': depth_flat,
                'color': color_flat,
                'sdf': sdf,
                'z_vals': z_vals,
                'weight_sum': weight_sum_raw,
                'color_surface': color_surface if color_surface is not None else None,
                'depth_surface': depth_surface if depth_surface is not None else None,
                'depth_std': depth_std,                  
            }

        return result_dict

    def sdf2alpha(self, sdf, beta=10):
           
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None):
           
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size

                              
            if gt_depth is not None:
                               
                gt_depth_shape = gt_depth.shape

                                       
                if len(gt_depth_shape) > 1:
                    gt_depth = gt_depth.reshape(-1)

                        
                total_rays = rays_d.shape[0]

                                        
                if gt_depth.shape[0] != total_rays:
                    print(f"Warning: gt_depth length({gt_depth.shape[0]}) and the total number of rays ({total_rays}) does not match, make adjustments")

                                        
                    if gt_depth.shape[0] < total_rays:
                        temp_gt_depth = torch.zeros(total_rays, device=device)
                        temp_gt_depth[:gt_depth.shape[0]] = gt_depth
                        gt_depth = temp_gt_depth
                    else:
                                         
                        gt_depth = gt_depth[:total_rays]

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i + ray_batch_size]
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                depth_list.append(ret['depth'].float())
                color_list.append(ret['color'])

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color
