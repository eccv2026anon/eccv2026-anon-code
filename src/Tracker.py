import torch
import copy
import os
import time
import numpy as np

from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer


class Tracker(object):
       

    def __init__(self, cfg, args, structrecon):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = structrecon.idx
        self.bound = structrecon.bound
        self.mesher = structrecon.mesher
        self.output = structrecon.output
        self.verbose = structrecon.verbose
        self.renderer = structrecon.renderer
        self.gt_c2w_list = structrecon.gt_c2w_list
        self.mapping_idx = structrecon.mapping_idx
        self.mapping_cnt = structrecon.mapping_cnt
        self.shared_decoders = structrecon.shared_decoders
        self.estimate_c2w_list = structrecon.estimate_c2w_list
        self.truncation = structrecon.truncation

                                             
        self.use_block_manager = structrecon.use_block_manager if hasattr(structrecon, 'use_block_manager') else False
        self.block_manager = structrecon.block_manager if hasattr(structrecon, 'block_manager') else None

                              
        self.shared_planes_xy = structrecon.shared_planes_xy
        self.shared_planes_xz = structrecon.shared_planes_xz
        self.shared_planes_yz = structrecon.shared_planes_yz

        self.shared_c_planes_xy = structrecon.shared_c_planes_xy
        self.shared_c_planes_xz = structrecon.shared_c_planes_xz
        self.shared_c_planes_yz = structrecon.shared_c_planes_yz

        self.cam_lr_T = cfg['tracking']['lr_T']                                            
        self.cam_lr_R = cfg['tracking']['lr_R']                                          
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']                                                      
        self.gt_camera = cfg['tracking']['gt_camera']                                          
        self.tracking_pixels = cfg['tracking']['pixels']                       
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']                                
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.sdf_loss_delta = float(cfg.get('tracking', {}).get('sdf_loss_delta', 0.1))
        self.w_depth = cfg['tracking']['w_depth']              
        self.w_color = cfg['tracking']['w_color']                  
        self.use_odom = bool(cfg.get('tracking', {}).get('use_odom', False))
        self.w_odom = float(cfg.get('tracking', {}).get('w_odom', 10.0))
        self.odom_rot_weight = float(cfg.get('tracking', {}).get('odom_rot_weight', 1.0))
        self.odom_trans_weight = float(cfg.get('tracking', {}).get('odom_trans_weight', 1.0))
        self.odom_huber_delta = float(cfg.get('tracking', {}).get('odom_huber_delta', 0.1))
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']                      
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']                              

        self.every_frame = cfg['mapping']['every_frame']                          
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']                               

        self.prev_mapping_idx = -1                            
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)                                   
        self.n_img = len(self.frame_reader)             
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)            

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'],
                                           inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose,
                                           device=self.device)                                                        

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = structrecon.H, structrecon.W, structrecon.fx, structrecon.fy, structrecon.cx, structrecon.cy

                                                  
        shared_decoders_module = self.shared_decoders.module if hasattr(self.shared_decoders,
                                                                        'is_data_parallel') and self.shared_decoders.is_data_parallel else self.shared_decoders

                                         
                                 
        c_dim = cfg['model']['c_dim']
        truncation = cfg['model']['truncation']
        learnable_beta = cfg['rendering']['learnable_beta']

                           
        from src.networks.decoders import Decoders
        self.decoders = Decoders(c_dim=c_dim, truncation=truncation, learnable_beta=learnable_beta,
                                 device=self.device)
        self.decoders.use_block_manager = bool(self.use_block_manager)
        if not self.use_block_manager:
            self.decoders.block_manager = None

        self.Phi_prior = None
        self.prior_voxel_size = None
        self.prior_origin_xyz = None
        if cfg.get('mapping', {}).get('enable_pgis', False):
            prior_path = cfg.get('model', {}).get('prior_tsdf_path', '')
            if prior_path and os.path.exists(prior_path):
                try:
                    self.Phi_prior = np.load(prior_path)
                    self.prior_voxel_size = cfg.get('model', {}).get('prior_tsdf_voxel_size', 0.06)
                    self.prior_origin_xyz = cfg.get('model', {}).get('prior_tsdf_origin_xyz', [0.0, 0.0, 0.0])
                except Exception:
                    self.Phi_prior = None

                                            
        self.decoders_module = self.decoders

                               
                
        self.planes_xy = []
        self.planes_xz = []
        self.planes_yz = []
        self.c_planes_xy = []
        self.c_planes_xz = []
        self.c_planes_yz = []

                       
        self.planes_loaded = False

                      
        for p in self.decoders.parameters():
            p.requires_grad_(False)

        """The tracking stage does not optimize the scene, but only uses the existing scene to infer the camera pose. Therefore, we need to copy the shared network structure and feature plane (to prevent mapping training from changing them), and set them not to participate in gradient updates (requires_grad=False)."""

    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        vx, vy, vz = v.unbind(dim=-1)
        o = torch.zeros_like(vx)
        row0 = torch.stack([o, -vz, vy], dim=-1)
        row1 = torch.stack([vz, o, -vx], dim=-1)
        row2 = torch.stack([-vy, vx, o], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def _vee(M: torch.Tensor) -> torch.Tensor:
        return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)

    def _so3_log_map(self, R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0)
        theta = torch.acos(cos_theta)

        W = 0.5 * (R - R.transpose(-1, -2))
        w_small = self._vee(W)

        sin_theta = torch.sin(theta)
        scale = theta / (2.0 * sin_theta.clamp_min(eps))
        w_large = self._vee(scale.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-1, -2)))

        small = theta.abs() < 1e-4
        return torch.where(small.unsqueeze(-1), w_small, w_large)

    def _left_jacobian_inv_so3(self, omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
        Omega = self._skew(omega)
        I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(Omega)
        Omega2 = Omega @ Omega

        theta2 = theta * theta
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        coeff = (1.0 / theta2.clamp_min(eps)) - ((1.0 + cos_theta) / (2.0 * theta * sin_theta).clamp_min(eps))
        J_inv_large = I - 0.5 * Omega + coeff.unsqueeze(-1) * Omega2
        J_inv_small = I - 0.5 * Omega + (1.0 / 12.0) * Omega2

        small = theta.squeeze(-1) < 1e-4
        return torch.where(small.unsqueeze(-1).unsqueeze(-1), J_inv_small, J_inv_large)

    def _se3_log_map(self, T: torch.Tensor) -> torch.Tensor:
        R = T[..., :3, :3]
        t = T[..., :3, 3]
        omega = self._so3_log_map(R)
        J_inv = self._left_jacobian_inv_so3(omega)
        rho = (J_inv @ t.unsqueeze(-1)).squeeze(-1)
        return torch.cat([rho, omega], dim=-1)

    def _huber(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        abs_x = x.abs()
        d = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
        return torch.where(abs_x < d, 0.5 * x * x, d * (abs_x - 0.5 * d))

    def _safe_relative_transform(self, prev_pose: torch.Tensor, cur_pose: torch.Tensor) -> torch.Tensor:
        try:
            rel = torch.linalg.solve(prev_pose, cur_pose)
            if torch.isfinite(rel).all():
                return rel
        except Exception:
            pass
        eye = torch.eye(4, dtype=cur_pose.dtype, device=cur_pose.device).unsqueeze(0)
        if cur_pose.dim() == 3 and cur_pose.shape[0] != 1:
            eye = eye.expand(cur_pose.shape[0], -1, -1)
        return eye

    def odom_loss(self, c2w: torch.Tensor, prev_est_c2w: torch.Tensor, odom_delta: torch.Tensor):
        if not self.use_odom or prev_est_c2w is None or odom_delta is None:
            return None

        if c2w.dim() == 2:
            c2w = c2w.unsqueeze(0)
        if prev_est_c2w.dim() == 2:
            prev_est_c2w = prev_est_c2w.unsqueeze(0)
        if odom_delta.dim() == 2:
            odom_delta = odom_delta.unsqueeze(0)

        T_pred = prev_est_c2w @ odom_delta
        T_err = self._safe_relative_transform(T_pred, c2w)
        xi = self._se3_log_map(T_err)
        trans = xi[..., :3]
        rot = xi[..., 3:]

        if self.odom_huber_delta > 0:
            trans_term = self._huber(trans, self.odom_huber_delta).sum(dim=-1)
            rot_term = self._huber(rot, self.odom_huber_delta).sum(dim=-1)
        else:
            trans_term = (trans * trans).sum(dim=-1)
            rot_term = (rot * rot).sum(dim=-1)

        return (self.odom_trans_weight * trans_term + self.odom_rot_weight * rot_term).mean()

    def sdf_losses(self, sdf, z_vals, gt_depth):
           
                     
        if not sdf.requires_grad:
            sdf = sdf.detach().requires_grad_(True)

        if not z_vals.requires_grad:
            z_vals = z_vals.detach().requires_grad_(True)

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

                    
        delta = self.sdf_loss_delta

        def _huber(x):
            abs_x = torch.abs(x)
            return torch.where(abs_x < delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))

        if front_mask.sum() > 0:
            fs_loss = torch.mean(_huber(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        else:
            fs_loss = torch.tensor(0.0, device=sdf.device, requires_grad=True)

        if center_mask.sum() > 0:
            center_loss = torch.mean(_huber(
                (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        else:
            center_loss = torch.tensor(0.0, device=sdf.device, requires_grad=True)

        if tail_mask.sum() > 0:
            tail_loss = torch.mean(_huber(
                (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))
        else:
            tail_loss = torch.tensor(0.0, device=sdf.device, requires_grad=True)

                    
        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def optimize_tracking(self, cam_pose, gt_color, gt_depth, batch_size, optimizer, prev_est_c2w=None, odom_delta=None):
           
                            
        if self.use_block_manager:
                                                 
            all_planes = None
        else:
                            
            all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H,
                                                                                 H - self.ignore_edge_H,
                                                                                 self.ignore_edge_W,
                                                                                 W - self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

                                                                 
        with torch.no_grad():             
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)             
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)             
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / (det_rays_d + 1e-6)             
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)                                                         
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

                          
        if not hasattr(self.decoders_module, 'bound') or self.decoders_module.bound is None:
            self.decoders_module.bound = self.bound.clone()

            
        render_dict = self.renderer.render_batch_ray(
            all_planes, self.decoders, batch_rays_d[inside_mask], batch_rays_o[inside_mask], device, self.truncation,
            gt_depth=batch_gt_depth[inside_mask],
            Phi_prior=self.Phi_prior if self.Phi_prior is not None else None,
            voxel_size=self.prior_voxel_size, origin_xyz=self.prior_origin_xyz
        )
        depth = render_dict['depth']
        color = render_dict['color']
        sdf = render_dict['sdf']
        z_vals = render_dict['z_vals']

              
                            
        sdf_loss = self.sdf_losses(sdf, z_vals, batch_gt_depth[inside_mask])

                          
        if not depth.requires_grad:
            depth = depth.detach().requires_grad_(True)
        if not color.requires_grad:
            color = color.detach().requires_grad_(True)

        depth_loss = torch.mean(torch.abs(depth - batch_gt_depth[inside_mask]))
        color_loss = torch.mean(torch.abs(color - batch_gt_color[inside_mask]))

                    
        loss = self.w_depth * depth_loss + self.w_color * color_loss + sdf_loss
        odom_loss_val = self.odom_loss(c2w, prev_est_c2w, odom_delta)
        if odom_loss_val is not None:
            loss = loss + self.w_odom * odom_loss_val

            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_params_from_mapping(self):
           
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

                          
            shared_decoders_module = self.shared_decoders.module if hasattr(self.shared_decoders,
                                                                            'is_data_parallel') and self.shared_decoders.is_data_parallel else self.shared_decoders

                     
                                           
                          
            with torch.no_grad():
                for target_param, source_param in zip(self.decoders.parameters(), shared_decoders_module.parameters()):
                    target_param.copy_(source_param.detach())

                              
            if hasattr(shared_decoders_module, 'bound') and shared_decoders_module.bound is not None:
                self.decoders.bound = shared_decoders_module.bound.clone()

                                                     
            if self.use_block_manager and self.block_manager is not None:
                self.decoders.set_block_manager(self.block_manager)

                               
            if self.planes_loaded:
                        
                for planes, self_planes in zip(
                        [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                        [self.planes_xy, self.planes_xz, self.planes_yz]):
                    for i, plane in enumerate(planes):
                        self_planes[i].copy_(plane.detach())

                for c_planes, self_c_planes in zip(
                        [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
                        [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
                    for i, c_plane in enumerate(c_planes):
                        self_c_planes[i].copy_(c_plane.detach())

            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
           
        device = self.device

               
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

                                         
        prev_gt_c2w_for_odom = None
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)
            if prev_gt_c2w_for_odom is None:
                odom_delta = torch.eye(4, dtype=gt_c2w.dtype, device=device).unsqueeze(0)
            else:
                odom_delta = self._safe_relative_transform(prev_gt_c2w_for_odom, gt_c2w)

                     
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]                       

                                                            
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                               
                wait_start_time = time.time()
                max_wait_time = 10.0           

                                    
                if idx == 1:
                                                   
                    expected_mapping_idx = 0
                else:
                    expected_mapping_idx = idx - 1

                if self.verbose:
                    print(f"Waiting for the Mapper to process the frame{expected_mapping_idx}, current mapping_idx={self.mapping_idx[0]}")

                while self.mapping_idx[0] != expected_mapping_idx:                  
                    time.sleep(0.001)

                            
                    if time.time() - wait_start_time > max_wait_time:
                        print(
                            f"Warning: Waiting for Mapper to process frame{expected_mapping_idx}Timeout, current mapping_idx={self.mapping_idx[0]}, continue processing")
                                                
                        break

                if self.verbose and self.mapping_idx[0] == expected_mapping_idx:
                    print(f"Mapper processed frame{expected_mapping_idx}, continue to track")

                                
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

                                            
            self.update_params_from_mapping()

            if self.verbose:           
                print(Fore.MAGENTA)
                print("Tracking Frame ", idx.item())
                print(Style.RESET_ALL)

                       
            if not self.planes_loaded:
                self.load_planes()

                                
            if self.use_block_manager:
                                                     
                all_planes = None
            else:
                                
                all_planes = (
                self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

                               
            if idx == 0 or self.gt_camera:
                c2w = gt_c2w                             

                                     
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

            else:
                                                 
                if self.const_speed_assumption and idx - 2 >= 0:
                                                           
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)],
                                            dim=0)                         
                    pre_poses = matrix_to_cam_pose(pre_poses)           
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]           
                else:
                                                          
                    cam_pose = matrix_to_cam_pose(pre_c2w)                

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())                 
                R = torch.nn.Parameter(cam_pose[:, :4].clone())                 
                cam_para_list_T = [T]                               
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam(
                    [{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas': (0.5, 0.999)},
                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas': (0.5, 0.999)}])
                """Use Adam optimizer to optimize pose; Specify different learning rates for the translation parameter T and rotation parameter R respectively; self.cam_lr_T: translation learning rate; self.cam_lr_R: learning rate of rotation; betas=(0.5, 0.999) is the optimizer momentum parameter setting, which is a relatively stable configuration."""
                current_min_loss = torch.tensor(float('inf')).float().to(device)                 
                                                          
                candidate_cam_pose = torch.cat([R, T], -1).clone().detach()
                for cam_iter in range(self.num_cam_iters):                                                   
                                             
                                
                    if not R.requires_grad:
                        R.requires_grad_(True)
                    if not T.requires_grad:
                        T.requires_grad_(True)

                    cam_pose = torch.cat([R, T], -1)

                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes,
                                              self.decoders)                                                        

                    loss = self.optimize_tracking(
                        cam_pose,
                        gt_color,
                        gt_depth,
                        self.tracking_pixels,
                        optimizer_camera,
                        prev_est_c2w=pre_c2w,
                        odom_delta=odom_delta,
                    )
                    if loss < current_min_loss:                             
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()                                           

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(
                0).clone()                                                         
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()                               
            pre_c2w = c2w.clone()                            
            prev_gt_c2w_for_odom = gt_c2w.clone()
            self.idx[0] = idx               

    def load_planes(self):
                                    
        if self.planes_loaded:
            return

        print("Loading feature planes on demand...")

                                
        if self.use_block_manager:
                                      
            if self.block_manager is not None and not hasattr(self.decoders, 'block_manager'):
                self.decoders.set_block_manager(self.block_manager)
            self.planes_loaded = True
            print("Blocked feature plane mode: no need to load global planes")
            return

                                 
        with torch.no_grad():
                    
            for p in self.shared_planes_xy:
                                 
                plane = p.detach().cpu().clone()
                self.planes_xy.append(plane.to(self.device))
                                  
                del plane
                torch.cuda.empty_cache()

            for p in self.shared_planes_xz:
                plane = p.detach().cpu().clone()
                self.planes_xz.append(plane.to(self.device))
                del plane
                torch.cuda.empty_cache()

            for p in self.shared_planes_yz:
                plane = p.detach().cpu().clone()
                self.planes_yz.append(plane.to(self.device))
                del plane
                torch.cuda.empty_cache()

                    
            for p in self.shared_c_planes_xy:
                plane = p.detach().cpu().clone()
                self.c_planes_xy.append(plane.to(self.device))
                del plane
                torch.cuda.empty_cache()

            for p in self.shared_c_planes_xz:
                plane = p.detach().cpu().clone()
                self.c_planes_xz.append(plane.to(self.device))
                del plane
                torch.cuda.empty_cache()

            for p in self.shared_c_planes_yz:
                plane = p.detach().cpu().clone()
                self.c_planes_yz.append(plane.to(self.device))
                del plane
                torch.cuda.empty_cache()

        self.planes_loaded = True
        print("Feature planes loaded successfully.")

    def track_camera(self, batch, is_first_frame=False):
           
                   
        if not self.planes_loaded:
            self.load_planes()

        idx, gt_color, gt_depth, gt_c2w = batch
        device = self.device
        gt_color = gt_color.to(device)
        gt_depth = gt_depth.to(device)
        gt_c2w = gt_c2w.to(device)
        if idx.item() == 0:
            odom_delta = torch.eye(4, dtype=gt_c2w.dtype, device=device).unsqueeze(0)
        else:
            prev_gt = self.gt_c2w_list[idx - 1].unsqueeze(0).to(device)
            if torch.isfinite(prev_gt).all() and torch.abs(prev_gt).sum() > 0:
                odom_delta = self._safe_relative_transform(prev_gt, gt_c2w)
            else:
                odom_delta = torch.eye(4, dtype=gt_c2w.dtype, device=device).unsqueeze(0)

        if is_first_frame or self.gt_camera:
            c2w = gt_c2w
            if not self.no_vis_on_first_frame:
                                    
                if self.use_block_manager:
                                                         
                    all_planes = None
                else:
                                    
                    all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz,
                                  self.c_planes_yz)
                self.visualizer.save_imgs(idx.item(), 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

        else:
            pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)
            if self.const_speed_assumption and idx - 2 >= 0:
                pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                pre_poses = matrix_to_cam_pose(pre_poses)
                cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
            else:
                cam_pose = matrix_to_cam_pose(pre_c2w)

            T = torch.nn.Parameter(cam_pose[:, -3:].clone())
            R = torch.nn.Parameter(cam_pose[:, :4].clone())
            cam_para_list_T = [T]
            cam_para_list_R = [R]
            optimizer_camera = torch.optim.Adam(
                [{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas': (0.5, 0.999)},
                 {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas': (0.5, 0.999)}])

            current_min_loss = torch.tensor(float('inf')).float().to(device)
                                
            if self.use_block_manager:
                                                     
                all_planes = None
            else:
                                
                all_planes = (
                self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

                                                      
            candidate_cam_pose = torch.cat([R, T], -1).clone().detach()

            for cam_iter in range(self.num_cam_iters):
                cam_pose = torch.cat([R, T], -1)

                self.visualizer.save_imgs(idx.item(), cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)
                                                
                if self.visualizer.inside_freq > 0 and cam_iter % self.visualizer.inside_freq == 0:
                    c2w_full = cam_pose_to_matrix(cam_pose)
                    self.visualizer.update_inside_rendering(idx.item(), cam_iter, c2w_full.squeeze(0), gt_color,
                                                            gt_depth, all_planes, self.decoders)

                loss = self.optimize_tracking(
                    cam_pose,
                    gt_color,
                    gt_depth,
                    self.tracking_pixels,
                    optimizer_camera,
                    prev_est_c2w=pre_c2w,
                    odom_delta=odom_delta,
                )
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_pose = cam_pose.clone().detach()

            c2w = cam_pose_to_matrix(candidate_cam_pose)

        self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
        self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
        return c2w
