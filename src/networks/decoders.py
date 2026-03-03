import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate
import numpy as np


class Decoders(nn.Module):

    def __init__(self, c_dim=32, hidden_size=64, truncation=0.06, n_blocks=2, learnable_beta=True,
                 geo_plane_shapes=None, color_plane_shapes=None, device='cuda:0',
                 enable_color_multires=False):

        super().__init__()                       

        self.c_dim = c_dim        
        self.truncation = truncation         
        self.n_blocks = n_blocks         
        self.device = device      
        self.to(device)               
        self.bound = None                          
        self.block_manager = None                 
        self.use_color_multires = enable_color_multires                

                    
                                                      
                                            
                                                                    
        self.use_block_manager = (geo_plane_shapes is None)

                             
                                                  
                                           
        decoder_input_dim = self.c_dim
        decoder_combined_dim = hidden_size + self.c_dim

                          
        self.linears = nn.ModuleList(
            [nn.Linear(decoder_input_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

                             
        self.linears_combined = nn.ModuleList(
            [nn.Linear(decoder_combined_dim, hidden_size * 2)] +
            [nn.Linear(hidden_size * 2, hidden_size * 2) for i in range(n_blocks - 1)])

                               
        self.c_linears = nn.ModuleList(
            [nn.Linear(decoder_input_dim, hidden_size)] +            
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

                    
        if self.use_color_multires:
                        
            self.c_linears_combined = nn.ModuleList(
                [nn.Linear(decoder_combined_dim, hidden_size * 2)] +
                [nn.Linear(hidden_size * 2, hidden_size * 2) for i in range(n_blocks - 1)])
            self.c_output_linear_combined = nn.Linear(hidden_size * 2, 3)         

             
        self.output_linear = nn.Linear(hidden_size, 1)         
        self.output_linear_combined = nn.Linear(hidden_size * 2, 1)          
        self.c_output_linear = nn.Linear(hidden_size, 3)         

                                           
                                       
                                     
        if learnable_beta:
            self.beta = nn.Parameter(30 * torch.ones(1))                             
        else:
            self.beta = 30             

                           
        if geo_plane_shapes is not None and color_plane_shapes is not None:
            self._init_feature_planes(geo_plane_shapes, color_plane_shapes, device)
        else:
                                        
                             
            self.planes_xy = None
            self.planes_xz = None
            self.planes_yz = None
            self.c_planes_xy = None
            self.c_planes_xz = None
            self.c_planes_yz = None

    def set_block_manager(self, block_manager):
                    
        self.block_manager = block_manager
        self.use_block_manager = True
                          
        self.use_color_multires = getattr(self.block_manager, 'use_color_multires', self.use_color_multires)
        print(f"Decoders: Enable block feature plane mode, including thick and thin two-level features, color multi-resolution:{self.use_color_multires}")

                           
        expected_input_dim = self.c_dim
        expected_combined_dim = self.linears[0].out_features + self.c_dim

        actual_input_dim = self.linears[0].in_features

                            
        if actual_input_dim != expected_input_dim:
            print(f"Warning: Decoder input dimensions mismatch (currently:{actual_input_dim}, expect:{expected_input_dim}), reinitialize MLP")

                       
            hidden_size = self.linears[0].out_features
            n_blocks = len(self.linears)

                                  
            self.linears = nn.ModuleList(
                [nn.Linear(expected_input_dim, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

                                     
            self.linears_combined = nn.ModuleList(
                [nn.Linear(expected_combined_dim, hidden_size * 2)] +
                [nn.Linear(hidden_size * 2, hidden_size * 2) for i in range(n_blocks - 1)])

                                       
            self.c_linears = nn.ModuleList(
                [nn.Linear(expected_input_dim, hidden_size)] +              
                [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

                                   
            if self.use_color_multires and not hasattr(self, 'c_linears_combined'):
                self.c_linears_combined = nn.ModuleList(
                    [nn.Linear(expected_combined_dim, hidden_size * 2)] +
                    [nn.Linear(hidden_size * 2, hidden_size * 2) for i in range(n_blocks - 1)])
                self.c_output_linear_combined = nn.Linear(hidden_size * 2, 3)         

                      
            self.output_linear = nn.Linear(hidden_size, 1)         
            self.output_linear_combined = nn.Linear(hidden_size * 2, 1)          
            self.c_output_linear = nn.Linear(hidden_size, 3)         

                        
        if hasattr(self, 'device'):
            self.to(self.device)

    def _init_feature_planes(self, geo_plane_shapes, color_plane_shapes, device):
                     
                
        self.planes_xy = nn.ParameterList([
            nn.Parameter(torch.zeros(geo_plane_shapes['xy'][i]).to(device))
            for i in range(len(geo_plane_shapes['xy']))
        ])
        self.planes_xz = nn.ParameterList([
            nn.Parameter(torch.zeros(geo_plane_shapes['xz'][i]).to(device))
            for i in range(len(geo_plane_shapes['xz']))
        ])
        self.planes_yz = nn.ParameterList([
            nn.Parameter(torch.zeros(geo_plane_shapes['yz'][i]).to(device))
            for i in range(len(geo_plane_shapes['yz']))
        ])

                
        self.c_planes_xy = nn.ParameterList([
            nn.Parameter(torch.zeros(color_plane_shapes['xy'][i]).to(device))
            for i in range(len(color_plane_shapes['xy']))
        ])
        self.c_planes_xz = nn.ParameterList([
            nn.Parameter(torch.zeros(color_plane_shapes['xz'][i]).to(device))
            for i in range(len(color_plane_shapes['xz']))
        ])
        self.c_planes_yz = nn.ParameterList([
            nn.Parameter(torch.zeros(color_plane_shapes['yz'][i]).to(device))
            for i in range(len(color_plane_shapes['yz']))
        ])

    def _sample_from_planes(self, p, planes_xy, planes_xz, planes_yz):
           
        N = p.shape[0]
        device = p.device

        if N == 0 or not planes_xy:
            return torch.zeros(N, 0, device=device)

        feat_list = []

                                     
                                                   
        grid_xy = p[:, [0, 1]].view(1, N, 1, 2)
        grid_xz = p[:, [0, 2]].view(1, N, 1, 2)
        grid_yz = p[:, [1, 2]].view(1, N, 1, 2)

        for i in range(len(planes_xy)):
            xy_plane = planes_xy[i]
            xz_plane = planes_xz[i]
            yz_plane = planes_yz[i]

            if xy_plane.dim() == 3:
                xy_plane = xy_plane.unsqueeze(0)
            if xz_plane.dim() == 3:
                xz_plane = xz_plane.unsqueeze(0)
            if yz_plane.dim() == 3:
                yz_plane = yz_plane.unsqueeze(0)

            xy = F.grid_sample(xy_plane, grid_xy, mode='bilinear',
                               padding_mode='border', align_corners=True)
            xz = F.grid_sample(xz_plane, grid_xz, mode='bilinear',
                               padding_mode='border', align_corners=True)
            yz = F.grid_sample(yz_plane, grid_yz, mode='bilinear',
                               padding_mode='border', align_corners=True)

                                                    
            xy = xy.squeeze(3).squeeze(0).transpose(0, 1)          
            xz = xz.squeeze(3).squeeze(0).transpose(0, 1)
            yz = yz.squeeze(3).squeeze(0).transpose(0, 1)

            feat = xy + xz + yz                 
            feat_list.append(feat)

        return torch.cat(feat_list, dim=-1)                

    def sample_plane_feature(self, p, planes_xy, planes_xz, planes_yz, block_planes=None):
        if hasattr(self, 'device'):
            p = p.to(self.device)

        if block_planes is not None:
            xy_planes_coarse = [block_planes['geo_feat_xy_coarse']]
            xz_planes_coarse = [block_planes['geo_feat_xz_coarse']]
            yz_planes_coarse = [block_planes['geo_feat_yz_coarse']]

            feat_coarse = self._sample_from_planes(p, xy_planes_coarse, xz_planes_coarse, yz_planes_coarse)

            if 'geo_feat_xy_fine' in block_planes:
                xy_planes_fine = [block_planes['geo_feat_xy_fine']]
                xz_planes_fine = [block_planes['geo_feat_xz_fine']]
                yz_planes_fine = [block_planes['geo_feat_yz_fine']]
                feat_fine = self._sample_from_planes(p, xy_planes_fine, xz_planes_fine, yz_planes_fine)
                return torch.cat([feat_coarse, feat_fine], dim=-1)
            else:
                return feat_coarse

        elif self.use_block_manager:
            block_indices = self.block_manager.get_block_index(p)                          
            device = self.device if hasattr(self, 'device') else p.device

            if isinstance(block_indices, list):
                block_indices = torch.tensor(block_indices, device=device)

            sorted_block_idx, sorted_idx = torch.sort(block_indices)
            p_sorted = p[sorted_idx]

            feat_out = torch.zeros(p.shape[0], self.c_dim * 2, device=device, dtype=p.dtype)

            unique_blocks, start_indices = torch.unique_consecutive(sorted_block_idx, return_counts=True)
            start_indices = torch.cumsum(torch.cat([torch.tensor([0], device=device), start_indices[:-1]]), dim=0)

            for block, start in zip(unique_blocks.tolist(), start_indices.tolist()):
                end = (sorted_block_idx == block).nonzero(as_tuple=False)[-1].item() + 1
                p_block = p_sorted[start:end]
                p_block_nor = self.block_manager.normalize_coordinate_by_block(p_block, block)
                block_planes = self.block_manager.get_block_planes(block)

                if block_planes is None:
                    continue

                xy_planes_coarse = [block_planes['geo_feat_xy_coarse']]
                xz_planes_coarse = [block_planes['geo_feat_xz_coarse']]
                yz_planes_coarse = [block_planes['geo_feat_yz_coarse']]
                xy_planes_fine = [block_planes['geo_feat_xy_fine']]
                xz_planes_fine = [block_planes['geo_feat_xz_fine']]
                yz_planes_fine = [block_planes['geo_feat_yz_fine']]

                feat_coarse = self._sample_from_planes(p_block_nor, xy_planes_coarse, xz_planes_coarse,
                                                       yz_planes_coarse)
                feat_fine = self._sample_from_planes(p_block_nor, xy_planes_fine, xz_planes_fine, yz_planes_fine)
                feat = torch.cat([feat_coarse, feat_fine], dim=-1)
                feat_out[sorted_idx[start:end]] = feat

            return feat_out

        else:
            p_nor = normalize_3d_coordinate(p.clone(), self.bound)
            return self._sample_from_planes(p_nor, planes_xy, planes_xz, planes_yz)

    def sample_color_plane_feature(self, p, c_planes_xy, c_planes_xz, c_planes_yz, block_planes=None):
        if hasattr(self, 'device'):
            p = p.to(self.device)

        if block_planes is not None:
                                               
            has_single = ('app_feat_xy' in block_planes) and ('app_feat_xz' in block_planes) and ('app_feat_yz' in block_planes)
            has_multi = ('app_feat_xy_coarse' in block_planes) and ('app_feat_xz_coarse' in block_planes) and ('app_feat_yz_coarse' in block_planes)
            if not has_single and not has_multi:
                feat_dim = self.c_dim * (2 if self.use_color_multires else 1)
                return torch.zeros(p.shape[0], feat_dim, device=p.device, dtype=p.dtype)
                                       
            if 'app_feat_xy_coarse' in block_planes:
                xy_planes_coarse = [block_planes['app_feat_xy_coarse']]
                xz_planes_coarse = [block_planes['app_feat_xz_coarse']]
                yz_planes_coarse = [block_planes['app_feat_yz_coarse']]
            else:
                xy_planes_coarse = [block_planes['app_feat_xy']]
                xz_planes_coarse = [block_planes['app_feat_xz']]
                yz_planes_coarse = [block_planes['app_feat_yz']]
            feat_coarse = self._sample_from_planes(p, xy_planes_coarse, xz_planes_coarse, yz_planes_coarse)

            if self.use_color_multires and 'app_feat_xy_fine' in block_planes:
                xy_planes_fine = [block_planes['app_feat_xy_fine']]
                xz_planes_fine = [block_planes['app_feat_xz_fine']]
                yz_planes_fine = [block_planes['app_feat_yz_fine']]
                feat_fine = self._sample_from_planes(p, xy_planes_fine, xz_planes_fine, yz_planes_fine)
                return torch.cat([feat_coarse, feat_fine], dim=-1)
            else:
                return feat_coarse

        elif self.use_block_manager:
            block_indices = self.block_manager.get_block_index(p)
            device = self.device if hasattr(self, 'device') else p.device

            if isinstance(block_indices, list):
                block_indices = torch.tensor(block_indices, device=device)

            sorted_block_idx, sorted_idx = torch.sort(block_indices)
            p_sorted = p[sorted_idx]

            feat_dim = self.c_dim * (2 if self.use_color_multires else 1)
            feat_out = torch.zeros(p.shape[0], feat_dim, device=device, dtype=p.dtype)

            unique_blocks, start_indices = torch.unique_consecutive(sorted_block_idx, return_counts=True)
            start_indices = torch.cumsum(torch.cat([torch.tensor([0], device=device), start_indices[:-1]]), dim=0)

            for block, start in zip(unique_blocks.tolist(), start_indices.tolist()):
                end = (sorted_block_idx == block).nonzero(as_tuple=False)[-1].item() + 1
                p_block = p_sorted[start:end]
                p_block_nor = self.block_manager.normalize_coordinate_by_block(p_block, block)
                block_planes = self.block_manager.get_block_planes(block)

                if block_planes is None:
                    continue

                                   
                has_single = ('app_feat_xy' in block_planes) and ('app_feat_xz' in block_planes) and ('app_feat_yz' in block_planes)
                has_multi = ('app_feat_xy_coarse' in block_planes) and ('app_feat_xz_coarse' in block_planes) and ('app_feat_yz_coarse' in block_planes)
                if not has_single and not has_multi:
                    feat = torch.zeros(p_block.shape[0], self.c_dim * (2 if self.use_color_multires else 1), device=device, dtype=p.dtype)
                elif self.use_color_multires and 'app_feat_xy_coarse' in block_planes:
                    xy_planes_coarse = [block_planes['app_feat_xy_coarse']]
                    xz_planes_coarse = [block_planes['app_feat_xz_coarse']]
                    yz_planes_coarse = [block_planes['app_feat_yz_coarse']]
                                 
                    if 'app_feat_xy_fine' in block_planes:
                        xy_planes_fine = [block_planes['app_feat_xy_fine']]
                        xz_planes_fine = [block_planes['app_feat_xz_fine']]
                        yz_planes_fine = [block_planes['app_feat_yz_fine']]
                        feat_coarse = self._sample_from_planes(p_block_nor, xy_planes_coarse, xz_planes_coarse, yz_planes_coarse)
                        feat_fine = self._sample_from_planes(p_block_nor, xy_planes_fine, xz_planes_fine, yz_planes_fine)
                        feat = torch.cat([feat_coarse, feat_fine], dim=-1)
                    else:
                        feat = self._sample_from_planes(p_block_nor, xy_planes_coarse, xz_planes_coarse, yz_planes_coarse)
                else:
                    xy_planes = [block_planes['app_feat_xy']]
                    xz_planes = [block_planes['app_feat_xz']]
                    yz_planes = [block_planes['app_feat_yz']]
                    feat = self._sample_from_planes(p_block_nor, xy_planes, xz_planes, yz_planes)

                feat_out[sorted_idx[start:end]] = feat

            return feat_out

        else:
            p_nor = normalize_3d_coordinate(p.clone(), self.bound)
            return self._sample_from_planes(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

    def get_raw_sdf(self, p, all_planes, block_planes=None):
                                    
        if not all_planes or len(all_planes[0]) == 0:
            return torch.zeros(p.shape[0], 1, device=p.device, dtype=p.dtype)

        feat = self.sample_plane_feature(p, all_planes[0], all_planes[1], all_planes[2], block_planes)

        if feat.shape[-1] == self.c_dim * 2:
            feat_coarse, feat_fine = feat.split(self.c_dim, dim=-1)

                           
            h = F.relu(self.linears[0](feat_coarse), inplace=True)
            for l in self.linears[1:]:
                h = F.relu(l(h), inplace=True)

                              
            h_combined = torch.cat([h, feat_fine], dim=-1)
            h_combined = F.relu(self.linears_combined[0](h_combined), inplace=True)
            for l in self.linears_combined[1:]:
                h_combined = F.relu(l(h_combined), inplace=True)

            sdf = self.output_linear_combined(h_combined)
        else:
            h = F.relu(self.linears[0](feat), inplace=True)
            for l in self.linears[1:]:
                h = F.relu(l(h), inplace=True)
            sdf = self.output_linear(h)

        return sdf

    def get_raw_rgb(self, p, all_planes, block_planes=None):
                                        
        if not all_planes or len(all_planes[3]) == 0:
            return torch.zeros(p.shape[0], 3, device=p.device, dtype=p.dtype)

        feat = self.sample_color_plane_feature(p, all_planes[3], all_planes[4], all_planes[5], block_planes)

        if self.use_color_multires and feat.shape[-1] == self.c_dim * 2 and hasattr(self, 'c_linears_combined'):
            feat_coarse, feat_fine = feat.split(self.c_dim, dim=-1)

                           
            h = F.relu(self.c_linears[0](feat_coarse), inplace=True)
            for l in self.c_linears[1:]:
                h = F.relu(l(h), inplace=True)

                              
            h_combined = torch.cat([h, feat_fine], dim=-1)
            h_combined = F.relu(self.c_linears_combined[0](h_combined), inplace=True)
            for l in self.c_linears_combined[1:]:
                h_combined = F.relu(l(h_combined), inplace=True)

            rgb = torch.sigmoid(self.c_output_linear_combined(h_combined))
        else:
            h = F.relu(self.c_linears[0](feat), inplace=True)
            for l in self.c_linears[1:]:
                h = F.relu(l(h), inplace=True)
            rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def forward(self, p, all_planes, need_rgb: bool = True):
                            
                                    
        if all_planes is None:
                               
            if not self.use_block_manager or self.block_manager is None:
                print("Error: all_planes is None but block_manager is not set")
                return {
                    'sdf': torch.zeros(p.shape[0], 1, device=p.device, dtype=p.dtype),
                    'rgb': torch.zeros(p.shape[0], 3, device=p.device, dtype=p.dtype)
                }

                                     
                           
            batch_size = 5000
                                    
            if p.dim() == 3 and p.size(-1) == 3:
                num_pixels = p.shape[0]
                num_samples = p.shape[1]
            else:
                num_pixels = p.shape[0]
                num_samples = None
            num_points = num_pixels
            num_batches = (num_points + batch_size - 1) // batch_size

            device = p.device
                                                
            all_sdf = None
            all_rgb = None
            sdf_initialized = False
            rgb_initialized = False

                                                   
            def _call_get_block_planes(block_idx_local):
                needed_val = ('all' if need_rgb else 'geo')
                              
                mode = getattr(self, '_bm_call_mode', None)
                if mode == 'keyword':
                    return self.block_manager.get_block_planes(block_idx_local, needed=needed_val)
                if mode == 'positional':
                    return self.block_manager.get_block_planes(block_idx_local, needed_val)
                if mode == 'none':
                    return self.block_manager.get_block_planes(block_idx_local)

                         
                try:
                    out = self.block_manager.get_block_planes(block_idx_local, needed=needed_val)
                    self._bm_call_mode = 'keyword'
                    return out
                except TypeError as e:
                    msg = str(e)
                                                                    
                    if ('unexpected keyword' in msg or 'got an unexpected keyword' in msg) and 'needed' in msg:
                        pass
                    else:
                                                    
                        raise
                    try:
                        out = self.block_manager.get_block_planes(block_idx_local, needed_val)
                        self._bm_call_mode = 'positional'
                        return out
                    except TypeError as e2:
                        msg2 = str(e2)
                                                    
                        if ('positional' in msg2 and 'arguments' in msg2) or ('required positional' in msg2):
                            pass
                        else:
                                               
                            raise
                        if not hasattr(self, '_bm_needed_warned') or not getattr(self, '_bm_needed_warned'):
                            fn_dbg = getattr(self.block_manager, 'get_block_planes', None)
                            try:
                                co = getattr(fn_dbg, '__code__', None)
                                varnames = list(getattr(co, 'co_varnames', ())) if co else []
                                argcount = getattr(co, 'co_argcount', None) if co else None
                                co_filename = getattr(co, 'co_filename', None) if co else None
                            except Exception:
                                varnames, argcount, co_filename = [], None, None
                            print(
                                f"warn:{type(self.block_manager).__module__}.{type(self.block_manager).__name__}.get_block_planes does not support needed, the evaluation will load all planes (with higher video memory). \n"
                                f"Debugging: varnames={varnames}, argcount={argcount}, file={co_filename}")
                            self._bm_needed_warned = True
                        self._bm_call_mode = 'none'
                        return self.block_manager.get_block_planes(block_idx_local)

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, num_points)
                batch_p = p[start_idx:end_idx]

                           
                block_indices = self.block_manager.get_block_index(batch_p)

                        
                if isinstance(block_indices, tuple):
                              
                    block_planes = _call_get_block_planes(block_indices)
                    if block_planes is None:
                        continue

                           
                    if batch_p.dim() == 3:
                                            
                        bsz, ss, _ = batch_p.shape
                        p_flat = batch_p.reshape(bsz * ss, 3)
                        p_nor_flat = self.block_manager.normalize_coordinate_by_block(p_flat, block_indices)
                        p_nor = p_nor_flat.reshape(bsz, ss, 3)
                    else:
                        p_nor = self.block_manager.normalize_coordinate_by_block(batch_p, block_indices)

                                         
                    temp_planes = (
                        [block_planes['geo_feat_xy_coarse']],
                        [block_planes['geo_feat_xz_coarse']],
                        [block_planes['geo_feat_yz_coarse']],
                        [block_planes['app_feat_xy'] if 'app_feat_xy' in block_planes else torch.zeros(1)],
                        [block_planes['app_feat_xz'] if 'app_feat_xz' in block_planes else torch.zeros(1)],
                        [block_planes['app_feat_yz'] if 'app_feat_yz' in block_planes else torch.zeros(1)]
                    )

                                                  
                    if p_nor.dim() == 3:
                        bsz, ss, _ = p_nor.shape
                        feat_in = p_nor.reshape(bsz * ss, 3)
                        sdf_flat = self.get_raw_sdf(feat_in, temp_planes, block_planes)
                        rgb_flat = None if not need_rgb else self.get_raw_rgb(feat_in, temp_planes, block_planes)
                        sdf = sdf_flat.reshape(bsz, ss, -1)
                        rgb = None if rgb_flat is None else rgb_flat.reshape(bsz, ss, -1)
                    else:
                        sdf = self.get_raw_sdf(p_nor, temp_planes, block_planes)
                        rgb = None if not need_rgb else self.get_raw_rgb(p_nor, temp_planes, block_planes)

                                             
                    if not sdf_initialized:
                                                
                        if sdf.dim() == 3:             
                            all_sdf = torch.zeros((num_points, sdf.shape[1], sdf.shape[2]), device=device, dtype=p.dtype)
                        else:                                
                            all_sdf = torch.zeros((num_points, 1), device=device, dtype=p.dtype)
                        sdf_initialized = True

                    if need_rgb and not rgb_initialized:
                                                
                        if rgb is not None and rgb.dim() == 3:             
                            all_rgb = torch.zeros((num_points, rgb.shape[1], rgb.shape[2]), device=device, dtype=p.dtype)
                        else:                                
                            all_rgb = torch.zeros((num_points, 3), device=device, dtype=p.dtype)
                        rgb_initialized = True

                                 
                    if sdf.dim() == 3:           
                        all_sdf[start_idx:end_idx] = sdf
                        if need_rgb and rgb is not None:
                            all_rgb[start_idx:end_idx] = rgb
                    else:                         
                        all_sdf[start_idx:end_idx, :] = sdf if sdf.dim() == 2 else sdf.view(-1, 1)
                        if need_rgb and rgb is not None:
                            all_rgb[start_idx:end_idx, :] = rgb if rgb.dim() == 2 else rgb.view(-1, 3)
                else:
                                       
                                                       
                    if isinstance(block_indices, list):
                        block_indices = torch.as_tensor(block_indices, device=device, dtype=torch.long)
                    else:
                        block_indices = block_indices.to(device=device, dtype=torch.long)

                                                                
                    if block_indices.dim() == 3 and block_indices.size(-1) == 3:
                        block_indices_pix = block_indices[:, 0, :]         
                    elif block_indices.dim() == 2 and block_indices.size(-1) == 3:
                        block_indices_pix = block_indices         
                    else:
                        raise ValueError(f"Expected block_indices shape [B,3] or [B,S,3], got {tuple(block_indices.shape)}")

                                        
                                                                       
                    gz = int(self.block_manager.block_grid_size[2].item())
                    gy = int(self.block_manager.block_grid_size[1].item())
                    key_mul_y = gz
                    key_mul_x = gy * gz
                    keys = (block_indices_pix[:, 0] * key_mul_x\
                            + block_indices_pix[:, 1] * key_mul_y\
                            + block_indices_pix[:, 2])

                    sorted_keys, sorted_idx = torch.sort(keys)
                    block_indices_sorted = block_indices_pix[sorted_idx]
                    batch_p_sorted = batch_p[sorted_idx]

                                        
                    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
                    starts = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts[:-1]]), dim=0)

                    for uk, st, ct in zip(unique_keys.tolist(), starts.tolist(), counts.tolist()):
                        end = st + ct
                        sel = slice(st, end)
                        pts_block = batch_p_sorted[sel]

                                   
                        block_triplet = tuple(block_indices_sorted[st].tolist())
                        planes = _call_get_block_planes(block_triplet)
                        if planes is None:
                                                     
                                                 
                            if not sdf_initialized:
                                if pts_block.dim() == 3:
                                    all_sdf = torch.zeros((num_points, pts_block.shape[1], 1), device=device, dtype=p.dtype)
                                else:
                                    all_sdf = torch.zeros((num_points, 1), device=device, dtype=p.dtype)
                                sdf_initialized = True
                            if need_rgb and not rgb_initialized:
                                if pts_block.dim() == 3:
                                    all_rgb = torch.zeros((num_points, pts_block.shape[1], 3), device=device, dtype=p.dtype)
                                else:
                                    all_rgb = torch.zeros((num_points, 3), device=device, dtype=p.dtype)
                                rgb_initialized = True
                                        
                            orig_idx = sorted_idx[sel] + start_idx
                            if pts_block.dim() == 3:
                                         
                                fill_sdf = torch.full((pts_block.shape[0], pts_block.shape[1], 1), self.truncation,
                                                      device=device, dtype=all_sdf.dtype if all_sdf is not None else p.dtype)
                                fill_rgb = torch.zeros((pts_block.shape[0], pts_block.shape[1], 3),
                                                       device=device, dtype=(all_rgb.dtype if (need_rgb and all_rgb is not None) else p.dtype))
                            else:
                                fill_sdf = torch.full((pts_block.shape[0], 1), self.truncation,
                                                      device=device, dtype=all_sdf.dtype if all_sdf is not None else p.dtype)
                                fill_rgb = torch.zeros((pts_block.shape[0], 3),
                                                       device=device, dtype=(all_rgb.dtype if (need_rgb and all_rgb is not None) else p.dtype))
                            all_sdf[orig_idx] = fill_sdf
                            if need_rgb:
                                all_rgb[orig_idx] = fill_rgb
                            continue

                                  
                        if pts_block.dim() == 3:
                            gsz, ss, _ = pts_block.shape
                            pts_flat = pts_block.reshape(gsz * ss, 3)
                            pts_block_nor_flat = self.block_manager.normalize_coordinate_by_block(pts_flat, block_triplet)
                            pts_block_nor = pts_block_nor_flat.reshape(gsz, ss, 3)
                        else:
                            pts_block_nor = self.block_manager.normalize_coordinate_by_block(pts_block, block_triplet)

                                  
                        tmp_planes = (
                            [planes['geo_feat_xy_coarse']],
                            [planes['geo_feat_xz_coarse']],
                            [planes['geo_feat_yz_coarse']],
                            [planes['app_feat_xy'] if 'app_feat_xy' in planes else torch.zeros(1, device=device)],
                            [planes['app_feat_xz'] if 'app_feat_xz' in planes else torch.zeros(1, device=device)],
                            [planes['app_feat_yz'] if 'app_feat_yz' in planes else torch.zeros(1, device=device)]
                        )

                                            
                        if pts_block_nor.dim() == 3:
                            gsz, ss, _ = pts_block_nor.shape
                            feat_in = pts_block_nor.reshape(gsz * ss, 3)
                            sdf_flat = self.get_raw_sdf(feat_in, tmp_planes, planes)
                            rgb_flat = None if not need_rgb else self.get_raw_rgb(feat_in, tmp_planes, planes)
                            sdf_blk = sdf_flat.reshape(gsz, ss, -1)
                            rgb_blk = None if rgb_flat is None else rgb_flat.reshape(gsz, ss, -1)
                        else:
                            sdf_blk = self.get_raw_sdf(pts_block_nor, tmp_planes, planes)
                            rgb_blk = None if not need_rgb else self.get_raw_rgb(pts_block_nor, tmp_planes, planes)

                                                                    
                        if not sdf_initialized:
                            if sdf_blk.dim() == 3:           
                                all_sdf = torch.zeros((num_points, sdf_blk.shape[1], sdf_blk.shape[2]),
                                                      device=device, dtype=sdf_blk.dtype)
                            else:          
                                all_sdf = torch.zeros((num_points, 1), device=device, dtype=sdf_blk.dtype)
                            sdf_initialized = True

                        if need_rgb and not rgb_initialized:
                            if rgb_blk is not None and rgb_blk.dim() == 3:           
                                all_rgb = torch.zeros((num_points, rgb_blk.shape[1], rgb_blk.shape[2]),
                                                      device=device, dtype=rgb_blk.dtype)
                            else:                                       
                                all_rgb = torch.zeros((num_points, 3), device=device, dtype=p.dtype)
                            rgb_initialized = True

                                     
                        orig_idx = sorted_idx[sel] + start_idx
                        if sdf_blk.dim() == 3:
                            all_sdf[orig_idx] = sdf_blk
                            if need_rgb and rgb_blk is not None:
                                all_rgb[orig_idx] = rgb_blk
                        else:
                            all_sdf[orig_idx, :] = sdf_blk if sdf_blk.dim() == 2 else sdf_blk.view(-1, 1)
                            if need_rgb and rgb_blk is not None:
                                all_rgb[orig_idx, :] = rgb_blk if rgb_blk.dim() == 2 else rgb_blk.view(-1, 3)

            return {
                'sdf': all_sdf,
                'rgb': all_rgb if need_rgb else None
            }
