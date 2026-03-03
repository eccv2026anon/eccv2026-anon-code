import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import imageio
import time
import numpy as np             
import torch.nn.functional as F

from colorama import Fore, Style

from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix,
                        get_rays_from_pixels, sample_rays_and_pixels, get_rays)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.cull_mesh import cull_mesh
import matplotlib
from src.networks.decoders import Decoders

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Mapper(object):
       

    def __init__(self, cfg, args, structrecon, renderer=None):
           
        import os
        self.verbose = cfg.get('verbose', False)            
        self.cfg = cfg      
        self.args = args         
        self.output = args.output if args.output is not None else cfg.get('data', {}).get('output',
                                                                                          'output_default')        
                          
        self.loss_log_dir = os.path.join(self.output, 'loss_curves')
        os.makedirs(self.loss_log_dir, exist_ok=True)
        self.loss_history = {'step': [], 'rgb': [], 'depth': [], 'sdf': [], 'total': [], 'frame': []}
        self.loss_csv_path = os.path.join(self.loss_log_dir, 'loss_log.csv')
        try:
            if not os.path.exists(self.loss_csv_path):
                with open(self.loss_csv_path, 'w', encoding='utf-8') as f:
                    f.write('global_step,frame_idx,iter,color_loss,depth_loss,sdf_loss,total_loss\n')
        except Exception:
            pass
        self._loss_global_step = 0
                   
        self.loss_plot_interval = int(cfg.get('mapping', {}).get('loss_plot_interval', 20))
        self.bounds = np.array(cfg['bound']) if 'bound' in cfg else None          
        
                   
        self.truncation = cfg['model']['truncation']
        if not isinstance(self.truncation, (int, float)):
            raise ValueError(f"truncation must be a numeric type, the current type is:{type(self.truncation)}")
        if self.truncation <= 0:
            raise ValueError(f"truncation must be greater than 0, current value:{self.truncation}")
        if self.truncation > 1.0:
            print(f"Warning: truncation value ({self.truncation}) is too large, it is recommended to be within the range of 0.05~0.3")
                                    
        self.edge_guided_ratio = float(cfg.get('mapping', {}).get('edge_guided_pixels_ratio', 0.5))
        self.edge_ignore = int(cfg.get('tracking', {}).get('ignore_edge_W', 0))            
                               
        self.mapper_only_mode = (cfg.get('debug', {}).get('mode', 'all') == 'mapper')
                                                
        warmup_cfg = cfg.get('mapping', {}).get('decoder_warmup', {})
        self.enable_decoder_warmup = warmup_cfg.get('enable', False)
                                           
        _to_float = lambda v, d: (float(str(v).split('#')[0].strip()) if v is not None else float(d)) if not isinstance(
            v, (int, float)) else float(v)
        _to_int = lambda v, d: (
            int(float(str(v).split('#')[0].strip())) if v is not None else int(d)) if not isinstance(v, (
        int, float)) else int(v)
        self.decoder_warmup_iters = _to_int(warmup_cfg.get('iters', 200), 200)
        self.decoder_warmup_lr_mult = _to_float(warmup_cfg.get('lr_mult', 5.0), 5.0)
        self.decoder_warmup_pixels = _to_int(warmup_cfg.get('pixels', cfg['mapping']['pixels'] * 2),
                                             cfg['mapping']['pixels'] * 2)
        self.decoder_warmup_w_color = _to_float(warmup_cfg.get('w_color_scale', 0.0), 0.0)
        self.decoder_warmup_w_depth = _to_float(warmup_cfg.get('w_depth_scale', 1.0), 1.0)
                             
        cb_cfg = warmup_cfg.get('color_bootstrap', {})
        self.color_bootstrap_enable = cb_cfg.get('enable', False)
        self.color_bootstrap_last_iters = _to_int(cb_cfg.get('last_iters', 200), 200)
        self.color_bootstrap_lr_mult = _to_float(cb_cfg.get('lr_mult', 5.0), 5.0)
        self.color_bootstrap_w_color = _to_float(cb_cfg.get('w_color_scale', 1.0), 1.0)

                                                
        import os
        mapping_cfg = cfg.get('mapping', {})
        self.load_decoder_warmup = bool(mapping_cfg.get('load_decoder_warmup', False))
                                                                               
        self.warmup_ckpt_path = warmup_cfg.get('decoder_warmup_ckpt',
                                               os.path.join(self.output, 'checkpoints', 'decoder_warmup.pth'))
        self.warmup_ckpt_exists = self.load_decoder_warmup and os.path.exists(self.warmup_ckpt_path)
                                                                            
        self.freeze_decoder_if_warmup_loaded = bool(mapping_cfg.get('freeze_decoder_if_warmup_loaded', True))
        self.freeze_decoder = bool(self.warmup_ckpt_exists and self.freeze_decoder_if_warmup_loaded)
        
                                       
                                
        self.enable_geometry_regularization = bool(mapping_cfg.get('enable_geometry_regularization', False))
        self.geometry_reg_weight = _to_float(mapping_cfg.get('geometry_reg_weight', 50.0), 50.0)
        self.geometry_reg_start_frame = _to_int(mapping_cfg.get('geometry_reg_start_frame', 0), 0)
        self.geometry_reg_decay = _to_float(mapping_cfg.get('geometry_reg_decay', 1.0), 1.0)
        self.save_reference_geometry_flag = bool(mapping_cfg.get('save_reference_geometry', True))
        self.reference_geo_features = None              
        self.current_geometry_reg_weight = self.geometry_reg_weight                
        
        if self.enable_geometry_regularization:
            print("[Mapper] Geometric regularization enabled:")
            print(f"- Initial weight:{self.geometry_reg_weight}")
            print(f"- Start frame:{self.geometry_reg_start_frame}")
            print(f"- Attenuation coefficient:{self.geometry_reg_decay}")
                                              

                                         
                                     
        self.enable_adaptive_prior_loss = bool(mapping_cfg.get('enable_adaptive_prior_loss', True))
        self.prior_loss_lambda_max = _to_float(mapping_cfg.get('prior_loss_lambda_max', 1.0), 1.0)
        self.prior_loss_gamma = _to_float(mapping_cfg.get('prior_loss_gamma', 1.0), 1.0)
        self.prior_loss_tau_w = _to_float(mapping_cfg.get('prior_loss_tau_w', 1.0), 1.0)
        self.prior_loss_delta = _to_float(mapping_cfg.get('prior_loss_delta', 0.1), 0.1)
        self.prior_loss_w_normal = _to_float(mapping_cfg.get('prior_loss_w_normal', 0.1), 0.1)
        self.prior_loss_max_rays = int(mapping_cfg.get('prior_loss_max_rays', 512))

        if self.enable_adaptive_prior_loss:
            print("[Mapper] Structure-adaptive prior loss enabled:")
            print(f"  - lambda_max: {self.prior_loss_lambda_max}")
            print(f"  - gamma: {self.prior_loss_gamma}")
            print(f"  - tau_w: {self.prior_loss_tau_w}")
            print(f"  - huber_delta: {self.prior_loss_delta}")
            print(f"  - normal_weight: {self.prior_loss_w_normal}")

        dual_cfg = mapping_cfg.get('dual_weight_arbitration', {})
        self.enable_dual_weight_arbitration = bool(dual_cfg.get('enable', False))
        self.prior_weight_gamma = _to_float(dual_cfg.get('prior_weight_gamma', 1.0), 1.0)
        self.dynamic_occlusion_threshold = _to_float(dual_cfg.get('dynamic_occlusion_threshold', 0.5), 0.5)
        self.confidence_threshold_std = _to_float(dual_cfg.get('confidence_threshold_std', 0.1), 0.1)
        self.data_weight_epsilon = _to_float(dual_cfg.get('data_weight_epsilon', 0.1), 0.1)

        if self.enable_dual_weight_arbitration:
            print("[Mapper] Bidirectional arbitration enabled:")
            print(f"  - prior_gamma: {self.prior_weight_gamma}")
            print(f"  - occlusion_threshold: {self.dynamic_occlusion_threshold} m")
            print(f"  - confidence_threshold: {self.confidence_threshold_std}")
            print(f"  - data_epsilon: {self.data_weight_epsilon}")

        self.enable_prior_guided_preoptimization = bool(cfg.get('model', {}).get('enable_prior_initialization', False))
        self.prior_init_iterations = int(cfg.get('model', {}).get('prior_init_iterations', 200))
        self.prior_init_lr_planes = float(cfg.get('model', {}).get('prior_init_lr_planes', 0.01))
        self.prior_init_samples_per_batch = int(cfg.get('model', {}).get('prior_init_samples_per_batch', 16384))

        if self.enable_prior_guided_preoptimization:
            print("[Mapper] Prior-guided pre-optimization enabled:")
            print(f"  - iterations: {self.prior_init_iterations}")
            print(f"  - lr: {self.prior_init_lr_planes}")
            print(f"  - samples_per_batch: {self.prior_init_samples_per_batch}")

        self.Phi_prior = None
        self.voxel_size = None
        self.origin_xyz = None
        self.prior_ready = False

        need_prior = self.enable_adaptive_prior_loss or self.enable_prior_guided_preoptimization or bool(
            mapping_cfg.get('enable_pgis', False)
        )
        if need_prior:
            try:
                import numpy as np
                prior_path = cfg.get('model', {}).get('prior_tsdf_path', '')
                if prior_path and os.path.exists(prior_path):
                    print(f"[Mapper] Loading TSDF prior: {prior_path}")
                    self.Phi_prior = np.load(prior_path)
                    self.voxel_size = cfg.get('model', {}).get('prior_tsdf_voxel_size', 0.06)
                    self.origin_xyz = cfg.get('model', {}).get('prior_tsdf_origin_xyz', [0.0, 0.0, 0.0])
                    self.prior_ready = True
                    print(f"  - shape: {self.Phi_prior.shape}")
                    print(f"  - voxel_size: {self.voxel_size} m")
                    print(f"  - origin_xyz: {self.origin_xyz}")
                else:
                    print(f"[Mapper][WARNING] TSDF prior not found: {prior_path}")
                    self.enable_adaptive_prior_loss = False
                    self.enable_prior_guided_preoptimization = False
            except Exception as e:
                print(f"[Mapper][WARNING] Failed to load TSDF prior: {e}")
                self.enable_adaptive_prior_loss = False
                self.enable_prior_guided_preoptimization = False
        if self.warmup_ckpt_exists:
                          
            if self.enable_decoder_warmup:
                print(f"Decoder Warmup: Existing warmup weights detected{self.warmup_ckpt_path}, skip the warm-up phase")
            self.enable_decoder_warmup = False
            if self.freeze_decoder:
                print("Decoder: Freezing is enabled (freeze_decoder_if_warmup_loaded=true), and the decoder parameters are not trained during the entire Mapping stage.")
                                                                   

        self.idx = structrecon.idx                      
        self.mapping_idx = structrecon.mapping_idx                                  
        self.mapping_cnt = structrecon.mapping_cnt                         
        self.decoders = structrecon.shared_decoders                               

                            
        self.bound = structrecon.bound                            
        self.logger = structrecon.logger                              
        self.mesher = structrecon.mesher                              
        self.renderer = structrecon.renderer                                   

                                                  
        self.decoders_module = self.decoders.module if hasattr(self.decoders,
                                                               'module') else self.decoders

                    
        self.use_block_manager = hasattr(structrecon, 'use_block_manager') and structrecon.use_block_manager
        if self.use_block_manager:
            print("Mapper: Block mode detected, use BlockManager for feature plane management")
            self.block_manager = structrecon.block_manager
                                       
            self.planes_xy = structrecon.shared_planes_xy       
            self.planes_xz = structrecon.shared_planes_xz       
            self.planes_yz = structrecon.shared_planes_yz       
            self.c_planes_xy = structrecon.shared_c_planes_xy       
            self.c_planes_xz = structrecon.shared_c_planes_xz       
            self.c_planes_yz = structrecon.shared_c_planes_yz       
        else:
                        
            print("Mapper: Use traditional global feature plane mode")
            self.planes_xy = structrecon.shared_planes_xy
            self.planes_xz = structrecon.shared_planes_xz
            self.planes_yz = structrecon.shared_planes_yz
            self.c_planes_xy = structrecon.shared_c_planes_xy
            self.c_planes_xz = structrecon.shared_c_planes_xz
            self.c_planes_yz = structrecon.shared_c_planes_yz
            self.block_manager = None

        if self.enable_prior_guided_preoptimization and self.prior_ready:
            if self.use_block_manager:
                print('[Mapper][WARNING] Prior pre-optimization is disabled in block-manager mode.')
            else:
                try:
                    self.preoptimize_from_prior_tsdf(self.Phi_prior, self.voxel_size, self.origin_xyz, self.device)
                except Exception as e:
                    print(f'[Mapper][WARNING] Prior pre-optimization failed: {e}')

        self.estimate_c2w_list = structrecon.estimate_c2w_list                                           
        self.mapping_first_frame = structrecon.mapping_first_frame                                         

        self.scale = cfg['scale']                          
        self.device = cfg['device']                           
        self.keyframe_device = cfg['keyframe_device']                       

                                     
        self.enable_prior_initialization = cfg.get('model', {}).get('enable_prior_initialization', False)
                                                                          
        self.prior_geometry_lr_scale = cfg.get('mapping', {}).get('prior_geometry_lr_scale', 1.0)

        if self.enable_prior_initialization and self.prior_geometry_lr_scale != 1.0:
            print(f"Mapper: OSM prior enabled. Geo-plane LR scale: {self.prior_geometry_lr_scale}")
        elif self.enable_prior_initialization:
                                                                               
            print(f"Mapper: OSM prior enabled. Geo-plane LR scale: 1.0.")

        self.eval_rec = cfg['meshing']['eval_rec']            
                                        
        self.use_init_decoder_for_mesh = bool(cfg.get('meshing', {}).get('use_init_decoder', False))
        self.joint_opt = False                                           
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr']              

                     
        mapping_cfg2 = cfg.get('mapping', {})
        self.export_blocks_instead_of_mesh = bool(mapping_cfg2.get('export_blocks_instead_of_mesh', False))
                                       
        self.blocks_export_root = mapping_cfg2.get('blocks_export_root', os.path.join(self.output, 'blocks_export'))
                  
        try:
            os.makedirs(self.blocks_export_root, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Failed to create block export directory{self.blocks_export_root}: {e}")

        self.mesh_freq = cfg['mapping']['mesh_freq']                  
        self.ckpt_freq = cfg['mapping']['ckpt_freq']                     
        self.mapping_pixels = cfg['mapping']['pixels']               
        self.every_frame = cfg['mapping']['every_frame']                     

        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']                  
        self.w_sdf_center = cfg['mapping']['w_sdf_center']                  
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']                  
        self.sdf_loss_delta = float(cfg.get('mapping', {}).get('sdf_loss_delta', 0.1))
        self.w_depth = cfg['mapping']['w_depth']           
        self.w_color = cfg['mapping']['w_color']           
        self.keyframe_every = cfg['mapping']['keyframe_every']               
        self.mapping_window_size = cfg['mapping']['mapping_window_size']                         
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']                   
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']                    
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']                  
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']                                
                                    
        self.debug_log_every = cfg['mapping'].get('debug_log_every', 10)
                                              
        self.color_hit_weight_thresh = float(cfg.get('mapping', {}).get('color_hit_weight_thresh', 1e-3))
                                                         
        self.rgb_loss_type = str(cfg.get('mapping', {}).get('rgb_loss_type', 'l1')).lower()
        self.rgb_charb_eps = float(cfg.get('mapping', {}).get('rgb_charbonnier_eps', 1e-3))
        self.rgb_loss_in_lab = bool(cfg.get('mapping', {}).get('rgb_loss_in_lab', False))
                                                       
        self.rgb_strict_w_l2 = float(cfg.get('mapping', {}).get('rgb_strict_w_l2', 1.0))
        self.rgb_strict_w_charb = float(cfg.get('mapping', {}).get('rgb_strict_w_charb', 0.5))
        self.rgb_strict_w_cos = float(cfg.get('mapping', {}).get('rgb_strict_w_cos', 0.5))

        self.keyframe_dict = {}                                            
        self.keyframe_list = []                  
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)                  
        self.n_img = len(self.frame_reader)       
                                        
        if self.mapper_only_mode:
            step = 1
        else:
            step = self.every_frame
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, step))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'],
                                           inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

                               
                                                         
        self.visualize_planes_dir = None
        try:
            if bool(cfg.get('meshing', {}).get('visualize_feature_planes', False)):
                self.visualize_planes_dir = os.path.join(self.output, 'feature_planes')
        except Exception:
            self.visualize_planes_dir = None

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = structrecon.H, structrecon.W, structrecon.fx, structrecon.fy, structrecon.cx, structrecon.cy

                                       
        self.amp_enabled = bool(cfg.get('amp', True))

                                      
        try:
            prof_cfg = cfg.get('profiling', {}) if isinstance(cfg, dict) else {}
        except Exception:
            prof_cfg = {}
        self.profile_enable = bool(prof_cfg.get('enable', False))
        try:
            self.profile_every_n = int(prof_cfg.get('every_n', 1))
        except Exception:
            self.profile_every_n = 1
                                                       
        try:
            self._profile_step = 0
            import os as _os
            self.profile_dir = _os.path.join(self.output, 'profile') if hasattr(self, 'output') else 'profile'
                                                                         
            if self.profile_enable:
                _os.makedirs(self.profile_dir, exist_ok=True)
            self.mapping_profile_csv = _os.path.join(self.profile_dir, 'mapping_profile.csv')
        except Exception:
            self._profile_step = 0

                                                    
        self.extras_render_device = str(cfg.get('mapping', {}).get('extras_render_device', self.device))

                            
        try:
            from torch.cuda.amp import GradScaler                
            self.scaler = GradScaler(enabled=self.amp_enabled)
        except Exception:
            class _DummyScaler:
                def __init__(self):
                    self._enabled = False
                def scale(self, x):
                    return x
                def step(self, opt):
                    opt.step()
                def update(self):
                    pass
            self.scaler = _DummyScaler()
        
                                              
                                        
        if self.enable_geometry_regularization and self.save_reference_geometry_flag:
            print("[Mapper] Geometry regularization has been configured and the reference geometry will be saved during the first mapping")
            self._reference_geometry_saved = False           
        else:
            self._reference_geometry_saved = True         
                                                           

    def _save_reference_geometry(self):
                                  
        self.reference_geo_features = {}
        
        try:
            if self.use_block_manager:
                                   
                if not hasattr(self.block_manager, 'active_blocks'):
                    print("Warning: BlockManager has no active_blocks attribute, skips saving reference geometry")
                    return
                
                if len(self.block_manager.active_blocks) == 0:
                    print("Warning: No active blocks currently, skip saving reference geometry")
                    return
                
                for block_idx, block_data in self.block_manager.active_blocks.items():
                    self.reference_geo_features[block_idx] = {}
                    for key in ['geo_feat_xy_coarse', 'geo_feat_xy_fine',
                               'geo_feat_xz_coarse', 'geo_feat_xz_fine',
                               'geo_feat_yz_coarse', 'geo_feat_yz_fine']:
                        if key in block_data:
                                                            
                            self.reference_geo_features[block_idx][key] = block_data[key].detach().clone()
            else:
                               
                self.reference_geo_features['xy'] = [p.detach().clone() for p in self.planes_xy if p is not None]
                self.reference_geo_features['xz'] = [p.detach().clone() for p in self.planes_xz if p is not None]
                self.reference_geo_features['yz'] = [p.detach().clone() for p in self.planes_yz if p is not None]
            
            self._reference_geometry_saved = True          
            
        except Exception as e:
            print(f"Warning: Exception occurred while saving reference geometry:{e}")
            self.reference_geo_features = {}
    
    def _count_reference_features(self):
                         
        if self.reference_geo_features is None:
            return 0
        count = 0
        if self.use_block_manager:
            for block_idx, features in self.reference_geo_features.items():
                count += len(features)
        else:
            for plane_name in ['xy', 'xz', 'yz']:
                if plane_name in self.reference_geo_features:
                    count += len(self.reference_geo_features[plane_name])
        return count
    
    def _compute_geometry_regularization_loss(self):
                             
        if self.reference_geo_features is None or len(self.reference_geo_features) == 0:
            return torch.tensor(0.0, device=self.device)
        
        reg_loss = 0.0
        count = 0
        
        try:
            if self.use_block_manager:
                                      
                if not hasattr(self.block_manager, 'active_blocks'):
                    return torch.tensor(0.0, device=self.device)
                
                for block_idx, ref_features in self.reference_geo_features.items():
                    if block_idx in self.block_manager.active_blocks:
                        current_block = self.block_manager.active_blocks[block_idx]
                        for key, ref_feat in ref_features.items():
                            if key in current_block:
                                current_feat = current_block[key]
                                      
                                reg_loss += torch.mean(torch.abs(current_feat - ref_feat.to(current_feat.device)))
                                count += 1
            else:
                      
                for plane_name in ['xy', 'xz', 'yz']:
                    if plane_name in self.reference_geo_features:
                        ref_planes = self.reference_geo_features[plane_name]
                        current_planes = getattr(self, f'planes_{plane_name}')
                        for ref_p, cur_p in zip(ref_planes, current_planes):
                            if cur_p is not None:
                                      
                                reg_loss += torch.mean(torch.abs(cur_p - ref_p.to(cur_p.device)))
                                count += 1
            
            return reg_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=self.device)
        
        except Exception as e:
            print(f"Warning: Failed to compute geometric regularization loss:{e}")
            return torch.tensor(0.0, device=self.device)

    def _get_extras_render_device(self):
        dev = str(getattr(self, 'extras_render_device', self.device))
        return dev

    def _get_decoders_copy_on_device(self, device_str: str):
           
        try:
            model_cfg = self.cfg.get('model', {})
            decoder_cfg = model_cfg.get('decoder', {})
            dec = Decoders(
                c_dim=model_cfg.get('c_dim', 32),
                hidden_size=decoder_cfg.get('hidden_size', 128),
                truncation=model_cfg.get('truncation', 0.1),
                n_blocks=decoder_cfg.get('n_blocks', 4),
                device=device_str
            ).to(device_str)
            dec.load_state_dict(self.decoders_module.state_dict(), strict=False)
                  
            try:
                dec.bound = getattr(self.decoders_module, 'bound', None)
            except Exception:
                pass
                                                   
            dec.eval()
            for p in dec.parameters():
                p.requires_grad_(False)
            return dec
        except Exception as e:
            print(f"[Warning] On device{device_str}Failed to create a decoder for rendering, and returned to using the training device:{e}")
            return self.decoders_module

    def _to_device_recursive(self, obj, device_str: str):
                                                                    
        import torch as _torch
        if obj is None:
            return None
        if isinstance(obj, _torch.Tensor):
            try:
                return obj.to(device_str, non_blocking=True)
            except Exception:
                return obj.to(device_str)
        if isinstance(obj, dict):
            return {k: self._to_device_recursive(v, device_str) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved_list = [self._to_device_recursive(v, device_str) for v in obj]
            return tuple(moved_list) if isinstance(obj, tuple) else moved_list
                  
        return obj

    def _move_all_planes_to_device(self, all_planes, device_str: str):
                                            
        return self._to_device_recursive(all_planes, device_str)

    def _renderer_push_device(self, device_str: str):
                                                      
        r = self.renderer
        prev = {
            'device': getattr(r, 'device', None),
            't_vals_uni_cache': getattr(r, 't_vals_uni_cache', None),
            't_vals_surface_cache': getattr(r, 't_vals_surface_cache', None),
            'bound': getattr(r, 'bound', None),
            'ones_cache': getattr(r, 'ones_cache', None),
        }
        try:
            if hasattr(r, 't_vals_uni_cache') and r.t_vals_uni_cache is not None:
                r.t_vals_uni_cache = r.t_vals_uni_cache.to(device_str)
            if hasattr(r, 't_vals_surface_cache') and r.t_vals_surface_cache is not None:
                r.t_vals_surface_cache = r.t_vals_surface_cache.to(device_str)
            if hasattr(r, 'bound') and r.bound is not None:
                r.bound = r.bound.to(device_str)
            if hasattr(r, 'ones_cache') and isinstance(r.ones_cache, dict):
                try:
                    r.ones_cache = {k: v.to(device_str) for k, v in r.ones_cache.items()}
                except Exception:
                    pass
            r.device = device_str
        except Exception as e:
            print(f"[Warning] Switch Renderer to device{device_str}fail:{e}")
        return prev

    def _renderer_pop_device(self, prev):
                                          
        r = self.renderer
        try:
            if prev.get('t_vals_uni_cache') is not None:
                            
                cur = getattr(r, 't_vals_uni_cache', None)
                if cur is not None:
                    r.t_vals_uni_cache = cur.to(prev['t_vals_uni_cache'].device)
                else:
                    r.t_vals_uni_cache = prev['t_vals_uni_cache']
            if prev.get('t_vals_surface_cache') is not None:
                cur = getattr(r, 't_vals_surface_cache', None)
                if cur is not None:
                    r.t_vals_surface_cache = cur.to(prev['t_vals_surface_cache'].device)
                else:
                    r.t_vals_surface_cache = prev['t_vals_surface_cache']
            if prev.get('bound') is not None:
                cur = getattr(r, 'bound', None)
                if cur is not None:
                    r.bound = cur.to(prev['bound'].device)
                else:
                    r.bound = prev['bound']
            if prev.get('ones_cache') is not None and isinstance(prev['ones_cache'], dict):
                try:
                                
                    if hasattr(r, 'ones_cache') and isinstance(r.ones_cache, dict):
                        r.ones_cache = {k: v.to(next(iter(prev['ones_cache'].values())).device) for k, v in r.ones_cache.items()}
                    else:
                        r.ones_cache = prev['ones_cache']
                except Exception:
                    r.ones_cache = prev['ones_cache']
            r.device = prev.get('device', r.device)
        except Exception as e:
            print(f"[Warning] Failed to restore Renderer device:{e}")

    def _render_chunked_safe_rgb(self, planes, decoders, rd, ro, device, truncation, H, W, gt_depth_full=None):
                                                                        
        import torch
        colors = []
        surfaces = []
        chunk = min(65536, rd.shape[0])
        prev_n_imp = getattr(self.renderer, 'n_importance', 32)
        tried_fallback = False
        s = 0
        while s < rd.shape[0]:
            try:
                if gt_depth_full is not None:
                    if gt_depth_full.dim() == 2:
                        gt_flat = gt_depth_full.reshape(-1)
                    else:
                        gt_flat = gt_depth_full
                    gt_depth_batch = gt_flat[s:s+chunk]
                else:
                    gt_depth_batch = None

                r = self.renderer.render_batch_ray(planes, decoders,
                                                   rd[s:s+chunk], ro[s:s+chunk],
                                                   device, truncation, gt_depth=gt_depth_batch)
                colors.append(r['color'].detach().cpu())
                try:
                    cs = r.get('color_surface', None)
                    if cs is not None:
                        surfaces.append(cs.detach().cpu())
                except Exception:
                    pass
                torch.cuda.empty_cache()
                s += chunk
            except Exception as e:
                if not tried_fallback:
                    print(f"[Warning] render_batch_ray failed to render in batches, downgraded to n_importance=0 and try again:{e}")
                                                      
                    try:
                        self.renderer.n_importance = 0
                        tried_fallback = True
                                     
                        continue
                    except Exception:
                        pass
                                 
                if chunk > 16384:
                    chunk = chunk // 2
                    print(f"[Warning] Render in chunks and then downgrade, use smaller chunk={chunk}")
                    continue
                else:
                                  
                    print(f"[Error] Rendering chunk failed and cannot be downgraded further, skip this chunk: start={s}, chunk={chunk}")
                    s += chunk
                         
        try:
            self.renderer.n_importance = prev_n_imp
        except Exception:
            pass
        img = torch.cat(colors, dim=0).reshape(H, W, 3).clamp(0, 1).numpy() if len(colors) > 0 else None
        surf = torch.cat(surfaces, dim=0).reshape(H, W, 3).clamp(0, 1).numpy() if len(surfaces) > 0 else None
        return img, surf

    def sdf_losses(self, sdf, z_vals, gt_depth):
           
                        
                                                                                              

                          
        if len(gt_depth.shape) > 1 and gt_depth.shape[-1] != 1:
            gt_depth = gt_depth.reshape(-1)           

                               
        if len(sdf.shape) == 1:
            sdf = sdf.unsqueeze(-1)                       

        if len(z_vals.shape) == 1:
            z_vals = z_vals.unsqueeze(-1)                       

                                
                                    
        if len(sdf.shape) == 3 and sdf.shape[-1] == 1:
                                                               
            sdf = sdf.squeeze(-1)            

                                                      
        if len(sdf.shape) == 2 and sdf.shape[1] == 1 and z_vals.shape[1] > 1:
                                                                                
            sdf = sdf.expand(-1, z_vals.shape[1])

                                 
                     
        gt_depth_expanded = gt_depth.unsqueeze(-1).expand_as(z_vals)

                                 
        front_mask = (z_vals < (gt_depth_expanded - self.truncation))
        back_mask = (z_vals > (gt_depth_expanded + self.truncation))
        center_mask = ((z_vals > (gt_depth_expanded - 0.4 * self.truncation)) &
                       (z_vals < (gt_depth_expanded + 0.4 * self.truncation)))
        tail_mask = ~(front_mask | back_mask | center_mask)

                             
        delta = self.sdf_loss_delta

        def _huber(x):
            abs_x = torch.abs(x)
            return torch.where(abs_x < delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))

        if front_mask.sum() > 0:
            fs_loss = torch.mean(_huber(sdf[front_mask] - 1.0))
        else:
            fs_loss = torch.tensor(0.0, device=sdf.device)

        if center_mask.sum() > 0:
            center_loss = torch.mean(_huber(
                (z_vals + sdf * self.truncation)[center_mask] - gt_depth_expanded[center_mask]))
        else:
            center_loss = torch.tensor(0.0, device=sdf.device)

        if tail_mask.sum() > 0:
            tail_loss = torch.mean(_huber(
                (z_vals + sdf * self.truncation)[tail_mask] - gt_depth_expanded[tail_mask]))
        else:
            tail_loss = torch.tensor(0.0, device=sdf.device)

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def compute_adaptive_prior_loss(self, render_result, gt_depth, rays_o, rays_d, decoders, all_planes,
                                    Phi_prior=None, voxel_size=None, origin_xyz=None):
        if not self.enable_adaptive_prior_loss or Phi_prior is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        device = self.device
        pred_depth = render_result['depth']
        z_vals = render_result['z_vals']
        if z_vals.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero

        if gt_depth is None:
            w_obs = torch.ones_like(pred_depth)
        else:
            depth_residual = torch.abs(pred_depth - gt_depth)
            w_obs = torch.exp(-self.prior_loss_gamma * depth_residual)
            valid_depth = (gt_depth > 0)
            w_obs = w_obs * valid_depth.float()

        n_rays, n_samples = z_vals.shape
        max_rays = min(n_rays, int(self.prior_loss_max_rays))
        if n_rays > max_rays:
            sel = torch.randperm(n_rays, device=device)[:max_rays]
            z_vals = z_vals[sel]
            rays_o = rays_o[sel]
            rays_d = rays_d[sel]
            w_obs = w_obs[sel]

        points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
        points_flat = points.reshape(-1, 3)

        phi_prior, grad_prior = self._query_prior_tsdf_simple(points_flat, Phi_prior, voxel_size, origin_xyz, device)
        valid_mask = self._tsdf_valid_mask(points_flat, Phi_prior.shape, voxel_size, origin_xyz, device)
        valid_mask = valid_mask.float()

        if self.prior_loss_w_normal > 0:
            points_flat = points_flat.detach().requires_grad_(True)

        pred = decoders(points_flat, all_planes, need_rgb=False)
        pred_sdf = pred['sdf'].reshape(-1)

        diff = pred_sdf - phi_prior
        abs_diff = torch.abs(diff)
        delta = self.prior_loss_delta
        huber = torch.where(
            abs_diff < delta,
            0.5 * diff ** 2,
            delta * (abs_diff - 0.5 * delta)
        )

        omega_geo = torch.exp(-torch.abs(phi_prior) / self.prior_loss_tau_w)
        w_obs_expanded = w_obs.repeat_interleave(z_vals.shape[1])
        weight = w_obs_expanded * omega_geo * valid_mask
        weight_sum = weight.sum().clamp_min(1e-6)
        prior_loss = (weight * huber).sum() / weight_sum

        normal_loss = torch.tensor(0.0, device=device)
        if self.prior_loss_w_normal > 0:
            grad_pred = torch.autograd.grad(
                pred_sdf, points_flat, grad_outputs=torch.ones_like(pred_sdf),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            pred_normals = torch.nn.functional.normalize(grad_pred, dim=-1)
            prior_normals = torch.nn.functional.normalize(grad_prior, dim=-1)
            cos_angle = torch.sum(pred_normals * prior_normals, dim=-1).clamp(-1.0, 1.0)
            normal_term = (1.0 - cos_angle)
            normal_loss = (weight * normal_term).sum() / weight_sum

        return prior_loss, normal_loss
    def _query_prior_tsdf_simple(self, points, Phi_prior, voxel_size, origin_xyz, device):
           
                                    
        if hasattr(self.renderer, '_query_prior_tsdf'):
            return self.renderer._query_prior_tsdf(points, Phi_prior, voxel_size, origin_xyz, device)
        else:
                  
            phi_vals = torch.zeros(points.shape[0], device=device)
            gradients = torch.zeros(points.shape[0], 3, device=device)
            return phi_vals, gradients

    def _tsdf_valid_mask(self, points, grid_shape, voxel_size, origin_xyz, device):
        voxel_coords_xyz = (points - torch.tensor(origin_xyz, device=device)) / voxel_size
        voxel_coords = voxel_coords_xyz[:, [2, 1, 0]]
        grid = torch.tensor(grid_shape, device=device)
        valid = (voxel_coords >= 0) & (voxel_coords < grid)
        return valid.all(dim=-1)

    def _load_external_confidence(self, gt_depth, pred_shape):
           
        try:
                                          
                                          
            if hasattr(gt_depth, 'filename'):
                base_name = os.path.splitext(os.path.basename(gt_depth.filename))[0]
                confidence_path = os.path.join(self.output, 'confidence', f"{base_name}_confidence.npy")
                if os.path.exists(confidence_path):
                    confidence = np.load(confidence_path)
                    return torch.from_numpy(confidence)
        except Exception as e:
                             
            pass
        return None

    def preoptimize_from_prior_tsdf(self, Phi_prior, voxel_size, origin_xyz, device):
        if not self.enable_prior_guided_preoptimization or Phi_prior is None:
            return

        if self.use_block_manager:
            print('[Mapper][WARNING] Prior pre-optimization is not supported in block-manager mode.')
            return

        print(f"\n[Mapper] Starting prior-guided pre-optimization ({self.prior_init_iterations} iterations)...")
        start_time = time.time()

        geo_params = []
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            geo_params.extend(planes)

        optimizer = torch.optim.Adam(geo_params, lr=self.prior_init_lr_planes)
        n_samples = self.prior_init_samples_per_batch

        all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz,
            self.c_planes_xy, self.c_planes_xz, self.c_planes_yz
        )

        for iter in range(self.prior_init_iterations):
            points = self._sample_points_for_preoptimization(n_samples, device)
            phi_prior_vals, _ = self._query_prior_tsdf_simple(points, Phi_prior, voxel_size, origin_xyz, device)
            valid_mask = torch.abs(phi_prior_vals) < 1.0
            if valid_mask.sum() == 0:
                continue

            valid_points = points[valid_mask]
            valid_phi_prior = phi_prior_vals[valid_mask]

            pred = self.decoders(valid_points, all_planes, need_rgb=False)
            pred_sdf = pred['sdf'].reshape(-1)

            diff = pred_sdf - valid_phi_prior
            abs_diff = torch.abs(diff)
            delta = self.prior_loss_delta
            huber = torch.where(
                abs_diff < delta,
                0.5 * diff ** 2,
                delta * (abs_diff - 0.5 * delta)
            )

            loss = torch.mean(huber)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter + 1) % 50 == 0:
                print(f"  Iter {iter + 1}/{self.prior_init_iterations}, loss: {loss.item():.6f}")

        elapsed = time.time() - start_time
        print(f"[Mapper] Prior pre-optimization finished in {elapsed:.2f} s")
    def _sample_points_for_preoptimization(self, n_samples, device):
           
                    
        bound_min = self.bounds[0] if self.bounds is not None else [-2.0, -2.0, -2.0]
        bound_max = self.bounds[1] if self.bounds is not None else [2.0, 2.0, 2.0]

                  
        points = torch.rand(n_samples, 3, device=device)
        for i in range(3):
            points[:, i] = points[:, i] * (bound_max[i] - bound_min[i]) + bound_min[i]

        return points

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
           
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0), gt_color.unsqueeze(0), device)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]                              
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0)
        w2cs = torch.inverse(keyframes_c2ws[:-2])                                                

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) *\
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

                                             
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict,
                         keyframe_list, cur_c2w):
           
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device = self.device
        
                                                     
        if self.enable_geometry_regularization and not self._reference_geometry_saved:
            print("[Mapper] The first mapping, saving the pre-optimized geometric features as a regularization reference...")
            self._save_reference_geometry()
            if len(self.reference_geo_features) > 0:
                print(f"[Mapper] Reference geometry features saved (total{self._count_reference_features()}parameters)")
            else:
                print("[Mapper] Warning: Failed to save reference geometry, geometry regularization will be disabled")
                self.enable_geometry_regularization = False
                                                                

                                      
        decoder_module = self.decoders_module

                                 
        joint_kf_list = keyframe_list
        if self.joint_opt and len(keyframe_list) > 4:
            if getattr(self, 'keyframe_selection_method', 'global') == 'overlap':
                try:
                                                                                 
                                                             
                    selected_pos = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w,
                                                                   num_keyframes=int(self.mapping_window_size))
                                           
                    joint_kf_list = []
                    for p in selected_pos:
                        if 0 <= p < len(keyframe_list):
                            kf_id = keyframe_list[p]
                            if kf_id not in joint_kf_list:
                                joint_kf_list.append(kf_id)
                                            
                    for recent in keyframe_list[-2:]:
                        if recent not in joint_kf_list:
                            joint_kf_list.append(recent)
                                                         
                    if len(joint_kf_list) == 0:
                        joint_kf_list = keyframe_list[-int(self.mapping_window_size):]
                except Exception:
                                 
                    joint_kf_list = keyframe_list[-int(self.mapping_window_size):]
            else:
                                         
                joint_kf_list = keyframe_list[-int(self.mapping_window_size):]

                                          
            keyframes_c2w = torch.stack([keyframe_dict[idx]['est_c2w'] for idx in joint_kf_list], dim=0)
            keyframes_c2w_gt = torch.stack([keyframe_dict[idx]['gt_c2w'] for idx in joint_kf_list], dim=0)
            keyframes_c2w = keyframes_c2w.to(device)
            keyframes_c2w_gt = keyframes_c2w_gt.to(device)

                 
        if self.use_block_manager:
                                          
            if self.verbose:
                print("Using BlockManager mode for parameter optimization")

                            
            camera_pos = cur_c2w[:3, 3].cpu().numpy()          
                                        
            self.block_manager.prepare_blocks_for_camera(camera_pos)
                                                                                             
                  
                                                                                    
                                                                                                
                                                                               
                          
                                                   
                                       
                              
                                                
                                           
                                          
                                                                 
                                                                         
                                                                                                                               
                                                                                                             
                                    
                                                                                                                       

                                  
            trainable_params = self.block_manager.get_trainable_parameters()

                   
            params = [
                {'params': trainable_params['geo_planes'],
                 'lr': self.cfg['mapping']['lr']['planes_lr'] * lr_factor *
                       (self.prior_geometry_lr_scale if self.enable_prior_initialization else 1.0)},
                {'params': trainable_params['color_planes'],
                 'lr': self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor}
            ]
                      
            if not self.freeze_decoder:
                params.append({'params': decoder_module.parameters(),
                               'lr': self.cfg['mapping']['lr']['decoders_lr'] * lr_factor})
        else:
                        
            if self.enable_prior_initialization:
                                                
                geo_planes = []
                for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
                    geo_planes.extend(planes)

                color_planes = []
                for planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
                    color_planes.extend(planes)

                                  
                geo_planes_lr = self.cfg['mapping']['lr']['planes_lr'] * self.prior_geometry_lr_scale

                               
                params = [
                    {'params': geo_planes, 'lr': geo_planes_lr * lr_factor},
                    {'params': color_planes, 'lr': self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor}
                ]
                if not self.freeze_decoder:
                    params.append({'params': decoder_module.parameters(),
                                   'lr': self.cfg['mapping']['lr']['decoders_lr'] * lr_factor})

                if self.verbose:
                    print(
                        f"Mapper: Geometric feature plane LR:{geo_planes_lr * lr_factor}, color feature plane LR:{self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor}")
            else:
                                 
                params = [
                    {'params': [p for planes in [self.planes_xy, self.planes_xz, self.planes_yz] for p in planes],
                     'lr': self.cfg['mapping']['lr']['planes_lr'] * lr_factor},
                    {'params': [p for planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz] for p in planes],
                     'lr': self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor}
                ]
                if not self.freeze_decoder:
                    params.append({'params': decoder_module.parameters(),
                                   'lr': self.cfg['mapping']['lr']['decoders_lr'] * lr_factor})

        if self.joint_opt and len(keyframe_list) > 4:
            params.append({'params': [cur_c2w], 'lr': self.joint_opt_cam_lr})
            params.append({'params': [keyframes_c2w], 'lr': self.joint_opt_cam_lr})

        optimizer = torch.optim.Adam(params)

                            
        if self.use_block_manager:
                                            
                                           
            all_planes = None                          
        else:
                            
            all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

               
        from torch.cuda.amp import autocast
        import time
        for iter in range(iters):
            t_total_start = time.perf_counter() if getattr(self, 'profile_enable', False) else None
                      
                                  
            if self.edge_guided_ratio > 0.0:
                def _edge_guided_pixels(color_img, depth_img, n_pick, ratio, ignore):
                                                          
                    with torch.no_grad():
                        rgb = color_img
                        if rgb.dim() == 3 and rgb.shape[-1] == 3:
                            gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
                        else:
                            gray = rgb.squeeze(-1) if rgb.dim() == 3 else rgb
                                      
                        gx = torch.zeros_like(gray)
                        gy = torch.zeros_like(gray)
                        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
                        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
                        mag = torch.sqrt(gx * gx + gy * gy)
                                     
                        Hh, Ww = gray.shape
                        if ignore > 0:
                            mag[:ignore, :] = 0;
                            mag[-ignore:, :] = 0;
                            mag[:, :ignore] = 0;
                            mag[:, -ignore:] = 0
                        if depth_img is not None:
                            mag = mag * (depth_img > 0).float()
                        flat = mag.reshape(-1)
                        n = int(n_pick)
                        n_edge = max(0, min(n, int(n * ratio)))
                                                     
                        k = max(1, int(0.1 * flat.numel()))
                        topk = torch.topk(flat, k, largest=True).indices
                                         
                        if n_edge > 0:
                            sel_edge = topk[torch.randint(high=topk.numel(), size=(n_edge,), device=flat.device)]
                        else:
                            sel_edge = torch.empty(0, dtype=torch.long, device=flat.device)
                        n_rest = n - sel_edge.numel()
                        sel_rest = torch.randint(high=flat.numel(), size=(n_rest,), device=flat.device)
                        sel = torch.cat([sel_edge, sel_rest], dim=0)
                        j = (sel // Ww).float();
                        i = (sel % Ww).float()
                        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1)
                        return (dirs, i, j)

                rays_d_cam = _edge_guided_pixels(cur_gt_color, cur_gt_depth, self.mapping_pixels,
                                                 self.edge_guided_ratio, self.edge_ignore)
            else:
                rays_d_cam = get_rays_from_pixels(H, W, fx, fy, cx, cy, self.mapping_pixels, device)
            target_d, target_rgb, rays_d, rays_o = sample_rays_and_pixels(H, W, fx, fy, cx, cy, cur_gt_depth,
                                                                          cur_gt_color, rays_d_cam, cur_c2w, device)

                   
            with torch.set_grad_enabled(True):
                                       
                if self.use_block_manager:
                                 
                    camera_pos = cur_c2w[:3, 3].cpu().numpy()
                              
                    t_prep_start = time.perf_counter() if self.profile_enable else None
                    self.block_manager.prepare_blocks_for_camera(camera_pos)
                    t_prep_end = time.perf_counter() if self.profile_enable else None
                                          
                    try:
                        if bool(self.cfg.get('mapping', {}).get('debug_export_blocks', False))\
                                and hasattr(self, 'block_manager') and self.block_manager is not None:
                                                                                     
                            try:
                                idx_int = int(idx.item())
                            except Exception:
                                try:
                                    idx_int = int(idx)
                                except Exception:
                                    idx_int = -1
                            debug_snap = f"{idx_int:05d}_after_prepare"
                            os.makedirs(self.blocks_export_root, exist_ok=True)
                            print(f"[Debug] Export blocks after prepare_blocks_for_camera at idx {idx_int} -> snapshot {debug_snap}")
                            self.block_manager.export_all_blocks(self.blocks_export_root, snapshot_name=debug_snap)
                    except Exception as e:
                        if self.verbose:
                            print(f"[Debug][Warning] export_all_blocks after prepare_blocks_for_camera failed at idx {idx}: {e}")

                                
                t_render_start = time.perf_counter() if self.profile_enable else None
                render_dict = self.renderer.render_batch_ray(
                    all_planes, self.decoders, rays_d, rays_o, device,
                    self.truncation, gt_depth=target_d, need_rgb=(self.w_color > 0),
                    Phi_prior=self.Phi_prior if self.prior_ready else None,
                    voxel_size=self.voxel_size, origin_xyz=self.origin_xyz
                )
                depth = render_dict['depth']
                color = render_dict['color']
                sdf = render_dict['sdf']
                z_vals = render_dict['z_vals']
                wsum = render_dict.get('weight_sum', None)
                t_render_end = time.perf_counter() if self.profile_enable else None

                      
                with autocast(enabled=self.amp_enabled):
                    sdf_loss = self.sdf_losses(sdf, z_vals, target_d)

                                                         
                    if self.enable_dual_weight_arbitration:
                        depth_std = render_dict.get('depth_std', None)
                        if depth_std is not None:
                                                      
                                                              
                            depth_residual = torch.abs(depth - target_d)

                                                        
                                                                        
                            is_occlusion = (target_d < (depth - self.dynamic_occlusion_threshold))
                            is_confident = (depth_std < self.confidence_threshold_std)

                                                            
                            w_data = torch.where(
                                is_occlusion & is_confident,
                                torch.tensor(self.data_weight_epsilon, device=device),
                                torch.tensor(1.0, device=device)
                            )

                                                                 
                                                                                         
                            weighted_depth_loss = w_data * depth_residual
                            depth_loss = torch.mean(weighted_depth_loss)

                                         
                            if iter == 0 and self.verbose and torch.rand(1).item() < 0.1:
                                occlusion_ratio = float((is_occlusion & is_confident).float().mean().item())
                                confident_ratio = float(is_confident.float().mean().item())
                                print(f"[Bidirectional Arbitration] Occluded pixel ratio ={occlusion_ratio:.3f}, confidence pixel ratio ={confident_ratio:.3f}")
                        else:
                                                   
                            depth_loss = torch.mean(torch.abs(depth - target_d))
                            if iter == 0 and self.verbose:
                                print("[WARNING] Two-way arbitration enabled but depth_std missing, downgrade to standard loss")
                    else:
                                  
                        depth_loss = torch.mean(torch.abs(depth - target_d))
                                                                         

                                                        
                    if wsum is not None:
                        hit_mask = (wsum > self.color_hit_weight_thresh)
                        if target_d is not None:
                            hit_mask = hit_mask & (target_d > 0)
                        if hit_mask.any():
                            color_loss = self._compute_color_loss(color[hit_mask], target_rgb[hit_mask])
                        else:
                            color_loss = torch.tensor(0.0, device=device)
                    else:
                        color_loss = self._compute_color_loss(color, target_rgb)
                    loss = self.w_depth * depth_loss + self.w_color * color_loss + sdf_loss
                
                                                 
                if self.enable_geometry_regularization and idx >= self.geometry_reg_start_frame:
                    geo_reg_loss = self._compute_geometry_regularization_loss()
                    loss = loss + self.current_geometry_reg_weight * geo_reg_loss

                                      
                    if iter == 0 and self.verbose:
                        print(f"[Mapper] Geometric regularization: loss={geo_reg_loss.item():.6f}, weight={self.current_geometry_reg_weight:.2f}")
                                                          

                                                   
                if self.enable_adaptive_prior_loss and hasattr(self, 'Phi_prior') and self.Phi_prior is not None:
                               
                    prior_loss, normal_loss = self.compute_adaptive_prior_loss(
                        render_dict, target_d, rays_o, rays_d, self.decoders, all_planes,
                        self.Phi_prior, self.voxel_size, self.origin_xyz
                    )

                            
                    current_lambda_prior = self.prior_loss_lambda_max * min(1.0, (idx + 1) / 10.0)         
                    total_prior_loss = current_lambda_prior * (prior_loss + self.prior_loss_w_normal * normal_loss)
                    loss = loss + total_prior_loss

                                     
                    if iter == 0 and self.verbose:
                        print(f"[Mapper] Adaptive prior loss: prior={prior_loss.item():.6f}, normal={normal_loss.item():.6f}, total={total_prior_loss.item():.6f}")
                                                               

                                            
                t_backward_start = time.perf_counter() if self.profile_enable else None
                optimizer.zero_grad()
                                      
                self.scaler.scale(loss).backward()
                total_loss_value = float(loss.item())         
                t_backward_end = time.perf_counter() if self.profile_enable else None

                                  
                del render_dict, depth, color, sdf, z_vals
                if 'wsum' in locals():
                    del wsum

                                             
                if self.joint_opt and len(joint_kf_list) > 4:
                                                                               
                    kf_pixels = int(self.cfg.get('mapping', {}).get('kf_pixels', max(1, self.mapping_pixels // 2)))

                    rays_d_list, rays_o_list = [], []
                    tgt_d_list, tgt_rgb_list = [], []

                    for i, keyframe_idx in enumerate(joint_kf_list):
                        keyframe_gt_depth = keyframe_dict[keyframe_idx]['depth'].to(device)
                        keyframe_gt_color = keyframe_dict[keyframe_idx]['color'].to(device)
                        keyframe_c2w = keyframes_c2w[i]

                                              
                        if self.use_block_manager:
                            kf_camera_pos = keyframe_c2w[:3, 3].cpu().numpy()
                            self.block_manager.prepare_blocks_for_camera(kf_camera_pos)

                                          
                        if self.edge_guided_ratio > 0.0:
                            rays_d_cam = _edge_guided_pixels(keyframe_gt_color, keyframe_gt_depth,
                                                             kf_pixels, self.edge_guided_ratio, self.edge_ignore)
                        else:
                            rays_d_cam = get_rays_from_pixels(H, W, fx, fy, cx, cy, kf_pixels, device)

                        td, trgb, rd, ro = sample_rays_and_pixels(H, W, fx, fy, cx, cy,
                                                                   keyframe_gt_depth, keyframe_gt_color,
                                                                   rays_d_cam, keyframe_c2w, device)
                        tgt_d_list.append(td)
                        tgt_rgb_list.append(trgb)
                        rays_d_list.append(rd)
                        rays_o_list.append(ro)

                                    
                    kf_target_d = torch.cat(tgt_d_list, dim=0)
                    kf_target_rgb = torch.cat(tgt_rgb_list, dim=0)
                    kf_rays_d = torch.cat(rays_d_list, dim=0)
                    kf_rays_o = torch.cat(rays_o_list, dim=0)

                    kf_render = self.renderer.render_batch_ray(
                        all_planes, self.decoders, kf_rays_d, kf_rays_o, device, self.truncation,
                        gt_depth=kf_target_d, need_rgb=(self.w_color > 0),
                        Phi_prior=self.Phi_prior if self.prior_ready else None,
                        voxel_size=self.voxel_size, origin_xyz=self.origin_xyz
                    )
                    kf_depth = kf_render['depth']
                    kf_color = kf_render['color']
                    kf_sdf = kf_render['sdf']
                    kf_z = kf_render['z_vals']
                    kf_wsum = kf_render.get('weight_sum', None)

                                 
                    sdf_loss_kf = self.sdf_losses(kf_sdf, kf_z, kf_target_d)

                                                          
                    if self.enable_dual_weight_arbitration:
                        kf_depth_std = kf_render.get('depth_std', None)
                        if kf_depth_std is not None:
                                                      
                            kf_depth_residual = torch.abs(kf_depth - kf_target_d)

                                                        
                            kf_is_occlusion = (kf_target_d < (kf_depth - self.dynamic_occlusion_threshold))
                            kf_is_confident = (kf_depth_std < self.confidence_threshold_std)
                            kf_w_data = torch.where(
                                kf_is_occlusion & kf_is_confident,
                                torch.tensor(self.data_weight_epsilon, device=device),
                                torch.tensor(1.0, device=device)
                            )

                                                                 
                            depth_loss_kf = torch.mean(kf_weighted_depth_loss)
                        else:
                            depth_loss_kf = torch.mean(torch.abs(kf_depth - kf_target_d))
                    else:
                        depth_loss_kf = torch.mean(torch.abs(kf_depth - kf_target_d))
                                                                         
                    if kf_wsum is not None:
                        hit_mask = (kf_wsum > self.color_hit_weight_thresh)
                        hit_mask = hit_mask & (kf_target_d > 0)
                        color_loss_kf = self._compute_color_loss(kf_color[hit_mask], kf_target_rgb[hit_mask]) if hit_mask.any()\
                            else torch.tensor(0.0, device=device)
                    else:
                        color_loss_kf = self._compute_color_loss(kf_color, kf_target_rgb)
                    loss_kf_total = self.w_depth * depth_loss_kf + self.w_color * color_loss_kf + sdf_loss_kf
                    
                                                          
                    if self.enable_geometry_regularization and idx >= self.geometry_reg_start_frame:
                        geo_reg_loss_kf = self._compute_geometry_regularization_loss()
                        loss_kf_total = loss_kf_total + self.current_geometry_reg_weight * geo_reg_loss_kf
                                                                    

                    self.scaler.scale(loss_kf_total).backward()
                    total_loss_value += float(loss_kf_total.item())

                          
                    del kf_render, kf_depth, kf_color, kf_sdf, kf_z, kf_target_d, kf_target_rgb, kf_rays_d, kf_rays_o

                                   

                                                     
                                   
            try:
                       
                self.loss_history['step'].append(int(self._loss_global_step))
                self.loss_history['frame'].append(int(idx.item() if torch.is_tensor(idx) else int(idx)))
                self.loss_history['rgb'].append(float(color_loss.item()))
                self.loss_history['depth'].append(float(depth_loss.item()))
                self.loss_history['sdf'].append(float(sdf_loss.item()))
                self.loss_history['total'].append(float(total_loss_value))
                        
                with open(self.loss_csv_path, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{self._loss_global_step},{int(self.loss_history['frame'][-1])},{int(iter)},{self.loss_history['rgb'][-1]},{self.loss_history['depth'][-1]},{self.loss_history['sdf'][-1]},{self.loss_history['total'][-1]}\n")
                           
                if self.loss_plot_interval > 0 and (self._loss_global_step % self.loss_plot_interval == 0):
                    self._update_loss_plots()
            except Exception:
                pass
            self._loss_global_step += 1
                                               
                                          
            if (self.debug_log_every is not None) and (self.debug_log_every > 0) and (iter % self.debug_log_every == 0):
                try:
                    def sum_grad_norm(params):
                        total = 0.0
                        cnt = 0
                        for p in params:
                            if p is None:
                                continue
                            g = getattr(p, 'grad', None)
                            if g is not None:
                                total += float(g.norm().item())
                                cnt += 1
                        return total, cnt

                    def sum_param_norm(params):
                        total = 0.0
                        for p in params:
                            if p is not None:
                                total += float(p.data.norm().item())
                        return total

                            
                    if self.use_block_manager:
                        tp = self.block_manager.get_trainable_parameters()
                        geo_params = tp.get('geo_planes', [])
                        color_params = tp.get('color_planes', [])
                    else:
                        geo_params = [p for planes in [self.planes_xy, self.planes_xz, self.planes_yz] for p in planes]
                        color_params = [p for planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz] for p in
                                        planes]
                    decoder_params = list(self.decoders_module.parameters())

                          
                    gn_geo, n_geo = sum_grad_norm(geo_params)
                    gn_col, n_col = sum_grad_norm(color_params)
                    gn_dec, n_dec = sum_grad_norm(decoder_params)

                                
                    pn_geo_before = sum_param_norm(geo_params)
                    pn_col_before = sum_param_norm(color_params)
                    pn_dec_before = sum_param_norm(decoder_params)

                except Exception as e:
                    print(f"[Debug] Exception occurred before statistical gradient/parameter:{e}")
                    gn_geo = gn_col = gn_dec = 0.0
                    n_geo = n_col = n_dec = 0
                    pn_geo_before = pn_col_before = pn_dec_before = 0.0

                             
                try:
                    print(
                        f"[Debug][iter {iter}] loss={float(loss.item()):.6f}, depth={float(depth_loss.item()):.6f}, color={float(color_loss.item()):.6f}")
                    print(
                        f"[Debug][iter {iter}] grad|| geo={gn_geo:.2e} (n={n_geo}), color={gn_col:.2e} (n={n_col}), dec={gn_dec:.2e} (n={n_dec})")
                except Exception:
                    pass

            t_opt_start = time.perf_counter() if self.profile_enable else None
                                      
            self.scaler.step(optimizer)
            self.scaler.update()
            t_opt_end = time.perf_counter() if self.profile_enable else None

                            
            if self.profile_enable and (self._profile_step % self.profile_every_n == 0):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prep_ms = ((t_prep_end - t_prep_start) * 1000.0) if ('t_prep_start' in locals() and 't_prep_end' in locals()) else 0.0
                    render_ms = ((t_render_end - t_render_start) * 1000.0) if ('t_render_start' in locals() and 't_render_end' in locals()) else 0.0
                    bw_ms = ((t_backward_end - t_backward_start) * 1000.0) if ('t_backward_start' in locals() and 't_backward_end' in locals()) else 0.0
                    opt_ms = ((t_opt_end - t_opt_start) * 1000.0) if ('t_opt_start' in locals() and 't_opt_end' in locals()) else 0.0
                    total_ms = (time.perf_counter() - t_total_start) * 1000.0 if t_total_start is not None else 0.0
                    frame_idx = int(idx.item() if torch.is_tensor(idx) else int(idx))
                    with open(self.mapping_profile_csv, 'a', encoding='utf-8') as f:
                        f.write(f"{self._profile_step},{frame_idx},{int(iter)},{prep_ms:.3f},{render_ms:.3f},{bw_ms:.3f},{opt_ms:.3f},{total_ms:.3f}\n")
                except Exception:
                    pass
            if self.profile_enable:
                self._profile_step += 1

                             
            if (self.debug_log_every is not None) and (self.debug_log_every > 0) and (iter % self.debug_log_every == 0):
                try:
                    if self.use_block_manager:
                        tp = self.block_manager.get_trainable_parameters()
                        geo_params = tp.get('geo_planes', [])
                        color_params = tp.get('color_planes', [])
                    else:
                        geo_params = [p for planes in [self.planes_xy, self.planes_xz, self.planes_yz] for p in planes]
                        color_params = [p for planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz] for p in
                                        planes]
                    decoder_params = list(self.decoders_module.parameters())

                    def sum_param_norm(params):
                        s = 0.0
                        for p in params:
                            if p is not None:
                                s += float(p.data.norm().item())
                        return s

                    pn_geo_after = sum_param_norm(geo_params)
                    pn_col_after = sum_param_norm(color_params)
                    pn_dec_after = sum_param_norm(decoder_params)

                    print(
                        f"[Debug][iter {iter}] | param|| geo={(pn_geo_after - pn_geo_before):.2e}, color={(pn_col_after - pn_col_before):.2e}, dec={(pn_dec_after - pn_dec_before):.2e}")
                except Exception as e:
                    print(f"[Debug] Exception when statistical parameters change:{e}")

                       
            if self.visualizer.inside_freq > 0 and iter % self.visualizer.inside_freq == 0:
                                       
                if self.use_block_manager:
                    camera_pos = cur_c2w[:3, 3].cpu().numpy()
                    self.block_manager.prepare_blocks_for_camera(camera_pos)

                self.visualizer.update_inside_rendering(idx, iter, cur_c2w, cur_gt_color, cur_gt_depth, all_planes,
                                                        self.decoders)

                    
        if self.joint_opt and len(joint_kf_list) > 4:
            for i, keyframe_idx in enumerate(joint_kf_list):
                keyframe_dict[keyframe_idx]['est_c2w'] = keyframes_c2w[i].detach().clone()
        
                                         
        if self.enable_geometry_regularization and self.geometry_reg_decay != 1.0:
            old_weight = self.current_geometry_reg_weight
            self.current_geometry_reg_weight *= self.geometry_reg_decay
            if self.verbose and abs(old_weight - self.current_geometry_reg_weight) > 1e-6:
                print(f"[Mapper] Geometric regularization weight decay:{old_weight:.2f} -> {self.current_geometry_reg_weight:.2f}")
                                                 

        return cur_c2w

    def _update_loss_plots(self):
                                                 
        try:
            if len(self.loss_history['step']) < 2:
                return
            steps = self.loss_history['step']
            rgb = self.loss_history['rgb']
            total = self.loss_history['total']
                    
            plt.figure(figsize=(6, 4))
            plt.plot(steps, rgb, label='RGB Loss', color='tab:orange')
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.title('RGB Loss (Mapping)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.loss_log_dir, 'rgb_loss.png'), dpi=150)
            plt.close()
                      
            plt.figure(figsize=(6, 4))
            plt.plot(steps, total, label='Total', color='tab:blue')
            plt.plot(steps, rgb, label='RGB', color='tab:orange', alpha=0.8)
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.title('Loss Curves (Mapping)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.loss_log_dir, 'loss_curves.png'), dpi=150)
            plt.close()
        except Exception:
            pass

                                                    
    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
                                                         
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
                                                     
            in_dtype = rgb.dtype
                            
            srgb = rgb.clamp(0.0, 1.0).to(torch.float32)
            a = 0.055
            lin = torch.where(srgb <= 0.04045,
                              srgb / 12.92,
                              ((srgb + a) / (1 + a)) ** 2.4)
                                     
            M = srgb.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
            XYZ = lin @ M.t()
                                                
            Xn, Yn, Zn = 0.95047, 1.0, 1.08883
            X = XYZ[..., 0] / Xn
            Y = XYZ[..., 1] / Yn
            Z = XYZ[..., 2] / Zn
            eps = 1e-6

            def f(t):
                delta = (6 / 29)
                t3 = t.clamp(min=0.0)
                return torch.where(t3 > delta ** 3, t3.pow(1 / 3), (t3 / (3 * delta ** 2) + 4 / 29))

            fx, fy, fz = f(X + eps), f(Y + eps), f(Z + eps)
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            out = torch.stack([L, a, b], dim=-1)
            return out.to(in_dtype) if out.dtype != in_dtype else out
    

    def _compute_color_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                                                             
        if self.rgb_loss_in_lab:
            pred_c = self._rgb_to_lab(pred)
            tgt_c = self._rgb_to_lab(target)
        else:
            pred_c, tgt_c = pred, target
        diff = pred_c - tgt_c
        if self.rgb_loss_type == 'l2':
            return torch.mean((diff * diff).sum(dim=-1))
        elif self.rgb_loss_type in ('smooth_l1', 'huber'):
            return torch.nn.functional.smooth_l1_loss(pred_c, tgt_c)
        elif self.rgb_loss_type == 'charbonnier':
            eps2 = self.rgb_charb_eps * self.rgb_charb_eps
            return torch.mean(torch.sqrt((diff * diff).sum(dim=-1) + eps2))
        elif self.rgb_loss_type == 'strict':
                                                                             
                             
            eps = 1e-8
                        
            l2_term = (diff * diff).sum(dim=-1)
                                    
            eps2 = self.rgb_charb_eps * self.rgb_charb_eps
            charb_term = torch.sqrt((diff * diff).sum(dim=-1) + eps2)
                                         
            pred_norm = pred_c.norm(dim=-1).clamp_min(eps)
            tgt_norm = tgt_c.norm(dim=-1).clamp_min(eps)
            cos_sim = (pred_c * tgt_c).sum(dim=-1) / (pred_norm * tgt_norm)
            cos_term = 1.0 - cos_sim.clamp(-1.0, 1.0)
            loss = self.rgb_strict_w_l2 * l2_term + self.rgb_strict_w_charb * charb_term + self.rgb_strict_w_cos * cos_term
            return torch.mean(loss)
        else:        
            return torch.mean(torch.abs(diff))

    def _decoder_warmup(self, gt_color, gt_depth, gt_c2w, all_planes):
           
                             
        if self.warmup_ckpt_exists or not self.enable_decoder_warmup or self.decoder_warmup_iters <= 0:
            return
        print(
            f"Decoder Warmup: Start, only optimize the decoder{self.decoder_warmup_iters}eaters, lr{self.decoder_warmup_lr_mult}, pixels={self.decoder_warmup_pixels}")
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
                
        decoder_module = self.decoders_module
        optimizer = torch.optim.Adam([
            {'params': decoder_module.parameters(),
             'lr': self.cfg['mapping']['lr']['decoders_lr'] * self.decoder_warmup_lr_mult}
        ])
                
        cur_c2w = gt_c2w
        total_warm_iters = self.decoder_warmup_iters
                                  
        if self.use_block_manager:
            try:
                self.block_manager.prepare_blocks_for_camera(cur_c2w[:3, 3].detach().cpu().numpy(), view_distance=50.0)
            except Exception:
                pass
                                    
        orig_beta = None
        try:
            if hasattr(decoder_module, 'beta') and isinstance(decoder_module.beta, torch.nn.Parameter):
                orig_beta = decoder_module.beta.detach().clone()
                decoder_module.beta.data = torch.tensor([2.0], device=decoder_module.beta.device,
                                                        dtype=decoder_module.beta.dtype)
        except Exception:
            orig_beta = None
        for it in range(total_warm_iters):
            rays_d_cam = get_rays_from_pixels(H, W, fx, fy, cx, cy, self.decoder_warmup_pixels, device)
            target_d, target_rgb, rays_d, rays_o = sample_rays_and_pixels(H, W, fx, fy, cx, cy, gt_depth, gt_color,
                                                                          rays_d_cam, cur_c2w, device)
                          
            if self.use_block_manager:
                camera_pos = cur_c2w[:3, 3].detach().cpu().numpy()
                self.block_manager.prepare_blocks_for_camera(camera_pos)
            render_dict = self.renderer.render_batch_ray(
                all_planes, self.decoders, rays_d, rays_o, device, self.truncation, gt_depth=target_d,
                Phi_prior=self.Phi_prior if self.prior_ready else None,
                voxel_size=self.voxel_size, origin_xyz=self.origin_xyz
            )
            depth = render_dict['depth']
            color = render_dict['color']
            sdf = render_dict['sdf']
            z_vals = render_dict['z_vals']
                                                        
            sdf_loss = self.sdf_losses(sdf, z_vals, target_d)
            depth_loss = torch.mean(torch.abs(depth - target_d))
            color_loss = torch.mean(torch.abs(color - target_rgb))
            loss = (self.w_depth * self.decoder_warmup_w_depth) * depth_loss
            loss += (self.w_color * self.decoder_warmup_w_color) * color_loss
            loss += sdf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it % max(1, self.decoder_warmup_iters // 5) == 0:
                print(f"Decoder Warmup: iter {it}/{self.decoder_warmup_iters}, loss={float(loss.item()):.6f}")
                              
        if self.color_bootstrap_enable and self.color_bootstrap_last_iters > 0:
            print(
                f"Decoder Warmup (Color Bootstrap): Start, train only color plane + color head{self.color_bootstrap_last_iters} iters")
                           
            for n, p in decoder_module.named_parameters():
                p.requires_grad_(False)
            for n, p in decoder_module.named_parameters():
                if n.startswith('c_') or n in ['c_output_linear.weight', 'c_output_linear.bias',
                                               'c_output_linear_combined.weight', 'c_output_linear_combined.bias']:
                    p.requires_grad_(True)
                                
            param_groups = []
            color_head_params = [p for n, p in decoder_module.named_parameters() if p.requires_grad]
            if len(color_head_params) > 0:
                param_groups.append({'params': color_head_params,
                                     'lr': self.cfg['mapping']['lr']['decoders_lr'] * self.color_bootstrap_lr_mult})
            color_plane_params = []
            if self.use_block_manager:
                tp = self.block_manager.get_trainable_parameters()
                color_plane_params = tp.get('color_planes', [])
            else:
                for planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
                    color_plane_params.extend(list(planes))
            if len(color_plane_params) > 0:
                param_groups.append({'params': color_plane_params,
                                     'lr': self.cfg['mapping']['lr']['c_planes_lr'] * self.color_bootstrap_lr_mult})
            optimizer_c = torch.optim.Adam(param_groups) if len(param_groups) > 0 else torch.optim.Adam(
                color_head_params, lr=self.cfg['mapping']['lr']['decoders_lr'] * self.color_bootstrap_lr_mult)
            for it in range(self.color_bootstrap_last_iters):
                rays_d_cam = get_rays_from_pixels(H, W, fx, fy, cx, cy, self.decoder_warmup_pixels, device)
                target_d, target_rgb, rays_d, rays_o = sample_rays_and_pixels(H, W, fx, fy, cx, cy, gt_depth, gt_color,
                                                                              rays_d_cam, cur_c2w, device)
                ret = self.renderer.render_batch_ray(
                    all_planes, self.decoders, rays_d, rays_o, device, self.truncation, gt_depth=target_d,
                    Phi_prior=self.Phi_prior if self.prior_ready else None,
                    voxel_size=self.voxel_size, origin_xyz=self.origin_xyz
                )
                color = ret['color']
                color_loss = torch.mean(torch.abs(color - target_rgb))
                optimizer_c.zero_grad()
                color_loss.backward()
                if it % max(1, self.color_bootstrap_last_iters // 4) == 0:
                    try:
                        grad_norm_cb = 0.0
                        for n, p in decoder_module.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                grad_norm_cb += float(p.grad.norm().item())
                        grad_norm_planes_app = 0.0
                        for p in color_plane_params:
                            if p.grad is not None:
                                grad_norm_planes_app += float(p.grad.norm().item())
                        print(
                            f"[Mapper][ColorBootstrap] iter {it}/{self.color_bootstrap_last_iters}, color_loss={float(color_loss.item()):.6f}, grad|| head={grad_norm_cb:.2e}, planes_app={grad_norm_planes_app:.2e}")
                    except Exception:
                        pass
                optimizer_c.step()
                      
            for _, p in decoder_module.named_parameters():
                p.requires_grad_(True)
                
        try:
            if orig_beta is not None and hasattr(decoder_module, 'beta'):
                decoder_module.beta.data = orig_beta.data.to(decoder_module.beta.device)
        except Exception:
            pass
        print("Decoder Warmup: End")
                              
        try:
            save_dir = os.path.join(self.output, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            warmup_path = os.path.join(save_dir, 'decoder_warmup.pth')
            torch.save(self.decoders_module.state_dict(), warmup_path)
            print(f"Decoder Warmup: Decoder weights have been saved to{warmup_path}")
        except Exception as e:
            print(f"Warning: Failed to save warm-up decoder weights:{e}")

    def run(self):
           
        cfg = self.cfg        
                            
        if self.use_block_manager:
                                            
                                           
            all_planes = None                          
        else:
                            
            all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]                          
        data_iterator = iter(self.frame_loader)              

                                       
        self.estimate_c2w_list[0] = gt_c2w                   
                                          
        try:
            if self.enable_decoder_warmup:
                self._decoder_warmup(gt_color.squeeze(0).to(self.device),
                                     gt_depth.squeeze(0).to(self.device),
                                     gt_c2w.squeeze(0).to(self.device),
                                     all_planes)
        except Exception as e:
            print(f"Warning: Decoder Warmup failed:{e}")

        init_phase = True           
        prev_idx = -1             
                                        
        if self.mapper_only_mode:
            if self.verbose:
                print("Mapper-Only mode: Complete mapping of all frames in sequence")
            for seq_i in range(self.n_img):
                if self.verbose:
                    print(Fore.GREEN)
                    print("Mapping Frame ", seq_i)
                    print(Style.RESET_ALL)

                _, gt_color, gt_depth, gt_c2w = next(data_iterator)
                gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
                gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
                gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

                                             
                cur_c2w = gt_c2w.clone()

                                 
                if not init_phase:
                    lr_factor = cfg['mapping']['lr_factor']
                    iters = cfg['mapping']['iters']
                else:
                    lr_factor = cfg['mapping']['lr_first_factor']
                    iters = cfg['mapping']['iters_first']

                self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']
                cur_c2w = self.optimize_mapping(iters, lr_factor, torch.tensor(seq_i), gt_color, gt_depth, gt_c2w,
                                                self.keyframe_dict, self.keyframe_list, cur_c2w)
                                                                                                     
                      
                                                                                        
                                                                                                    
                                                               
                                                                             
                                                                                                                        
                                                                                                                 
                                        
                                                                                                                    
                if self.joint_opt:
                    self.estimate_c2w_list[seq_i] = cur_c2w

                       
                if seq_i % self.keyframe_every == 0:
                    if seq_i not in self.keyframe_dict:
                        self.keyframe_list.append(seq_i)
                        self.keyframe_dict[seq_i] = {
                            'gt_c2w': gt_c2w,
                            'idx': torch.tensor(seq_i),
                            'color': gt_color.to(self.keyframe_device),
                            'depth': gt_depth.to(self.keyframe_device),
                            'est_c2w': cur_c2w.clone()
                        }

                init_phase = False
                self.mapping_first_frame[0] = 1
                             
                if ((not (
                        seq_i == 0 and self.no_log_on_first_frame)) and seq_i % self.ckpt_freq == 0) or seq_i == self.n_img - 1:
                    self.logger.log(torch.tensor(seq_i), self.keyframe_list)
                self.mapping_idx[0] = torch.tensor(seq_i)
                self.mapping_cnt[0] += 1
                if (seq_i % self.mesh_freq == 0) and (not (seq_i == 0 and self.no_mesh_on_first_frame)):
                    if self.export_blocks_instead_of_mesh:
                        if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                                                                                               
                            try:
                                for blk_idx in list(self.block_manager.active_blocks.keys()):
                                    self.block_manager.get_block_planes(blk_idx, needed='all')
                                if self.verbose:
                                    print("[PreExport] The required geometry and color planes have been completed for the currently active block")
                            except Exception as e:
                                if self.verbose:
                                    print(f"[PreExport] Failed to complete active block plane:{e}")
                            snap = f"{seq_i:05d}"
                                              
                            intr = {'H': self.H, 'W': self.W, 'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'frustum_depth': 6.0}
                                                       
                            try:
                                c2w_diag = cur_c2w if 'c2w_use' not in locals() else c2w_use
                                R = c2w_diag[:3, :3]
                                t = c2w_diag[:3, 3]
                                detR = torch.det(R).item()
                                fwd = (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_diag.device)).detach().cpu().numpy()
                                p_front = (t + torch.as_tensor(fwd, dtype=t.dtype, device=t.device) * 3.0).detach().cpu()
                                blk_idx = None
                                try:
                                    blk_idx = self.block_manager.get_block_index(p_front)
                                except Exception:
                                    blk_idx = None
                                print(f"[SELFCHK] det(R)={detR:.6f} | fwd={fwd} | front_block={blk_idx}")
                            except Exception as e:
                                print(f"[SELFCHK][warn] failed: {e}")
                            out_dir = self.block_manager.export_all_blocks(
                                self.blocks_export_root,
                                snapshot_name=snap,
                                visualize_camera=True,
                                cam_pose=cur_c2w if 'c2w_use' not in locals() else c2w_use,
                                intrinsics=intr,
                                camera_forward_negative_z=True
                            )
                                                                                                
                            try:
                                extras_dir = os.path.join(out_dir, "_extras")
                                os.makedirs(extras_dir, exist_ok=True)
                                pseudo_mesh = os.path.join(extras_dir, f"{snap}_blocks.ply")
                                                                        
                                self.mesher._save_decoder_and_features(pseudo_mesh, self.decoders, all_planes, self.device)
                                                                          
                                try:
                                    H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
                                                                                      
                                    use_bm = bool(getattr(self.decoders_module, 'use_block_manager', False) or getattr(self, 'use_block_manager', False))
                                    if use_bm:
                                        dev_ex = self.device
                                        dec_ex = self.decoders_module
                                        planes_ex = all_planes
                                        prev_r = None
                                    else:
                                        dev_ex = self._get_extras_render_device()
                                        dec_ex = self._get_decoders_copy_on_device(dev_ex)
                                        planes_ex = self._move_all_planes_to_device(all_planes, dev_ex)
                                        prev_r = self._renderer_push_device(dev_ex)
                                                 
                                    try:
                                        c2w = cur_c2w
                                    except Exception:
                                        try:
                                            c2w = gt_c2w
                                        except Exception:
                                            c2w = self.estimate_c2w_list[min(seq_i, len(self.estimate_c2w_list)-1)]
                                    if isinstance(c2w, torch.Tensor):
                                        c2w_use = c2w.to(dev_ex)
                                    else:
                                        c2w_use = torch.as_tensor(c2w, dtype=torch.float32, device=dev_ex)
                                                                                     
                                    if getattr(self, 'debug_extras_render', True):
                                        dec_mod = self.decoders.module if hasattr(self.decoders, 'module') else self.decoders
                                        print("[DBG1] use_block_manager:", bool(getattr(self, 'use_block_manager', False)),
                                              "dec.use_block_manager:", bool(getattr(dec_mod, 'use_block_manager', False)))
                                        print("[DBG1] has rgb_head:", hasattr(dec_mod, 'rgb_head'),
                                              "n_app_planes:", getattr(dec_mod, 'n_app_planes', None))
                                        if hasattr(self, 'block_manager') and self.block_manager is not None:
                                            bm_attrs = [a for a in dir(self.block_manager) if ('app' in a.lower() or 'appear' in a.lower())]
                                            print("[DBG1] block_manager present. dir hint(app/appearance?):", bm_attrs)
                                        else:
                                            print("[DBG1] block_manager: None")
                                        print("[DBG1] renderer.n_importance:", getattr(self.renderer, 'n_importance', None),
                                              "renderer.n_stratified:", getattr(self.renderer, 'n_stratified', None))
                                        print("[DBG1] H,W,fx,fy,cx,cy:", H, W, fx, fy, cx, cy)

                                                                      
                                    if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                                        try:
                                            cam_pos = c2w_use[:3, 3].detach().cpu().numpy()
                                            self.block_manager.prepare_blocks_for_camera(cam_pos)
                                        except Exception:
                                            pass
                                    rays_o_full, rays_d_full = get_rays(H, W, fx, fy, cx, cy, c2w_use, dev_ex)
                                    rd = rays_d_full.reshape(-1, 3)
                                    ro = rays_o_full.reshape(-1, 3)
                                                                              
                                    img, surf = self._render_chunked_safe_rgb(planes_ex, dec_ex, rd, ro, dev_ex, self.truncation, H, W, gt_depth_full=gt_depth)
                                    if img is not None:
                                        out_png = os.path.join(extras_dir, f"{snap}_render_rgb.png")
                                        imageio.imwrite(out_png, (img * 255).astype(np.uint8))
                                        print(f"Current frame RGB saved:{out_png}")
                                    else:
                                        print("[Warning] RGB rendering of the current frame returns empty and saving has been skipped")
                                    if surf is not None:
                                        out_png2 = os.path.join(extras_dir, f"{snap}_render_rgb_surface.png")
                                        imageio.imwrite(out_png2, (surf * 255).astype(np.uint8))
                                        print(f"Saved surface gets RGB directly:{out_png2}")
                                                                  
                                    if prev_r is not None:
                                        self._renderer_pop_device(prev_r)
                                except Exception as e_img:
                                    print(f"[Warning] Failed to render current frame RGB:{e_img}")
                                    if prev_r is not None:
                                        try:
                                            self._renderer_pop_device(prev_r)
                                        except Exception:
                                            pass
                                                                   
                                if hasattr(self.decoders, 'block_manager') and self.decoders.block_manager is not None:
                                    active_dir = os.path.join(extras_dir, "active_blocks")
                                    self.mesher._visualize_active_feature_planes(self.decoders.block_manager, active_dir,
                                                                                 normalize=True, grid_line_width=2,
                                                                                 block_index_fontsize=8)
                            except Exception as e:
                                print(f"[Warning] Additional export (weight/activation block PNG) failed:{e}")
                        else:
                            print("[Warning] export_blocks_instead_of_mesh=True but BlockManager is not enabled, skips block export and generates mesh instead")
                            mesh_out_file = f'{self.output}/mesh/{seq_i:05d}_mesh.ply'
                            self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                                  visualize_planes_dir=self.visualize_planes_dir,
                                                  use_init_decoder=self.use_init_decoder_for_mesh)
                            cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                                      estimate_c2w_list=self.estimate_c2w_list[:seq_i + 1])
                    else:
                        mesh_out_file = f'{self.output}/mesh/{seq_i:05d}_mesh.ply'
                        self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                              visualize_planes_dir=self.visualize_planes_dir,
                                              use_init_decoder=self.use_init_decoder_for_mesh)
                        cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                                  estimate_c2w_list=self.estimate_c2w_list[:seq_i + 1])

                             
            if self.export_blocks_instead_of_mesh:
                if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                    try:
                        for blk_idx in list(self.block_manager.active_blocks.keys()):
                            self.block_manager.get_block_planes(blk_idx, needed='all')
                        if self.verbose:
                            print("[PreExport] The required geometry and color planes have been completed for the currently active block (final)")
                    except Exception as e:
                        if self.verbose:
                            print(f"[PreExport] (final) Failed to complete active block plane:{e}")
                                  
                    intr_fin = {'H': self.H, 'W': self.W, 'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'frustum_depth': 6.0}
                                        
                    try:
                        c2w_final = self.estimate_c2w_list[-1]
                    except Exception:
                        c2w_final = torch.eye(4, device=self.device)
                                  
                    try:
                        R = c2w_final[:3, :3]
                        t = c2w_final[:3, 3]
                        detR = torch.det(R).item()
                        fwd = (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_final.device)).detach().cpu().numpy()
                        p_front = (t + (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_final.device)) * 3.0).detach().cpu()
                        blk_idx = None
                        try:
                            blk_idx = self.block_manager.get_block_index(p_front)
                        except Exception:
                            blk_idx = None
                        print(f"[SELFCHK][final] det(R)={detR:.6f} | fwd={fwd} | front_block={blk_idx}")
                    except Exception as e:
                        print(f"[SELFCHK][final][warn] failed: {e}")
                    out_dir = self.block_manager.export_all_blocks(
                        self.blocks_export_root, snapshot_name='final',
                        visualize_camera=True, cam_pose=c2w_final, intrinsics=intr_fin,
                        camera_forward_negative_z=True)
                                                                                        
                    try:
                        extras_dir = os.path.join(out_dir, "_extras")
                        os.makedirs(extras_dir, exist_ok=True)
                        pseudo_mesh = os.path.join(extras_dir, "final_blocks.ply")
                        self.mesher._save_decoder_and_features(pseudo_mesh, self.decoders, all_planes, self.device)
                                                            
                        try:
                            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
                            use_bm = bool(getattr(self.decoders_module, 'use_block_manager', False) or getattr(self, 'use_block_manager', False))
                            if use_bm:
                                dev_ex = self.device
                                dec_ex = self.decoders_module
                                planes_ex = all_planes
                                prev_r = None
                            else:
                                dev_ex = self._get_extras_render_device()
                                dec_ex = self._get_decoders_copy_on_device(dev_ex)
                                planes_ex = self._move_all_planes_to_device(all_planes, dev_ex)
                                prev_r = self._renderer_push_device(dev_ex)
                            try:
                                c2w = self.estimate_c2w_list[-1]
                            except Exception:
                                c2w = torch.eye(4)
                            c2w_use = c2w.to(dev_ex) if isinstance(c2w, torch.Tensor) else torch.as_tensor(c2w, dtype=torch.float32, device=dev_ex)
                            if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                                try:
                                    cam_pos = c2w_use[:3, 3].detach().cpu().numpy()
                                    self.block_manager.prepare_blocks_for_camera(cam_pos)
                                except Exception:
                                    pass
                            rays_o_full, rays_d_full = get_rays(H, W, fx, fy, cx, cy, c2w_use, dev_ex)
                            rd = rays_d_full.reshape(-1, 3)
                            ro = rays_o_full.reshape(-1, 3)
                            img = self._render_chunked_safe_rgb(planes_ex, dec_ex, rd, ro, dev_ex, self.truncation, H, W)
                            if img is not None:
                                out_png = os.path.join(extras_dir, "final_render_rgb.png")
                                imageio.imwrite(out_png, (img * 255).astype(np.uint8))
                                print(f"Saved final frame RGB:{out_png}")
                            else:
                                print("[Warning] The final frame RGB rendering returns empty and saving has been skipped")
                            if prev_r is not None:
                                self._renderer_pop_device(prev_r)
                        except Exception as e_img:
                            print(f"[Warning] Failed to render final frame RGB:{e_img}")
                            if prev_r is not None:
                                try:
                                    self._renderer_pop_device(prev_r)
                                except Exception:
                                    pass
                        if hasattr(self.decoders, 'block_manager') and self.decoders.block_manager is not None:
                            active_dir = os.path.join(extras_dir, "active_blocks")
                            self.mesher._visualize_active_feature_planes(self.decoders.block_manager, active_dir,
                                                                         normalize=True, grid_line_width=2,
                                                                         block_index_fontsize=8)
                    except Exception as e:
                        print(f"[Warning] Additional export (weight/activation block PNG) failed:{e}")
                else:
                    print("[Warning] export_blocks_instead_of_mesh=True but BlockManager is not enabled, generates final mesh instead")
                    final_mesh = f"{self.output}/mesh/final_mesh_eval_rec.ply" if self.eval_rec else f"{self.output}/mesh/final_mesh.ply"
                    self.mesher.get_mesh(final_mesh, all_planes, self.decoders, self.keyframe_dict, self.device,
                                         visualize_planes_dir=self.visualize_planes_dir,
                                         use_init_decoder=self.use_init_decoder_for_mesh)
                    cull_mesh(final_mesh, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)
            else:
                final_mesh = f"{self.output}/mesh/final_mesh_eval_rec.ply" if self.eval_rec else f"{self.output}/mesh/final_mesh.ply"
                self.mesher.get_mesh(final_mesh, all_planes, self.decoders, self.keyframe_dict, self.device,
                                     visualize_planes_dir=self.visualize_planes_dir,
                                     use_init_decoder=self.use_init_decoder_for_mesh)
                cull_mesh(final_mesh, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)
            return

                             
        while True:
            while True:
                idx = self.idx[0].clone()                                          
                if idx == self.n_img - 1:          
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:                             
                    break

                time.sleep(0.001)

            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())                
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = next(data_iterator)                         
            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)                       
            gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

            cur_c2w = self.estimate_c2w_list[idx]               

            if not init_phase:                            
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

                                        
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w,
                                            self.keyframe_dict, self.keyframe_list, cur_c2w)                      
                                                                                               
            try:
                if self.export_blocks_instead_of_mesh and self.use_block_manager\
                        and hasattr(self, 'block_manager') and self.block_manager is not None:
                    debug_snap = f"{int(idx.item()):05d}_after_opt"
                    os.makedirs(self.blocks_export_root, exist_ok=True)
                    print(f"[Debug] Export blocks after optimize_mapping at seq {idx.item()} -> snapshot {debug_snap}")
                    intr_dbg = {'H': self.H, 'W': self.W, 'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'frustum_depth': 6.0}
                                                                
                    try:
                        c2w_diag = cur_c2w
                        R = c2w_diag[:3, :3]
                        t = c2w_diag[:3, 3]
                        detR = torch.det(R).item()
                        fwd = (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_diag.device)).detach().cpu().numpy()
                        p_front = (t + (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_diag.device)) * 3.0).detach().cpu()
                        blk_idx = None
                        try:
                            blk_idx = self.block_manager.get_block_index(p_front)
                        except Exception:
                            blk_idx = None
                        print(f"[SELFCHK][after_opt] det(R)={detR:.6f} | fwd={fwd} | front_block={blk_idx}")
                    except Exception as e:
                        print(f"[SELFCHK][after_opt][warn] failed: {e}")
                    out_dir = self.block_manager.export_all_blocks(
                        self.blocks_export_root, snapshot_name=debug_snap,
                        visualize_camera=True, cam_pose=cur_c2w, intrinsics=intr_dbg,
                        camera_forward_negative_z=True)
                                                                                        
                    try:
                        extras_dir = os.path.join(out_dir, "_extras")
                        os.makedirs(extras_dir, exist_ok=True)
                        pseudo_mesh = os.path.join(extras_dir, f"{debug_snap}_blocks.ply")
                        self.mesher._save_decoder_and_features(pseudo_mesh, self.decoders, all_planes, self.device)
                                                                              
                        try:
                            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
                            dev_ex = self._get_extras_render_device()
                            dec_ex = self._get_decoders_copy_on_device(dev_ex)
                            planes_ex = self._move_all_planes_to_device(all_planes, dev_ex)
                            c2w = cur_c2w if isinstance(cur_c2w, torch.Tensor) else torch.as_tensor(cur_c2w, dtype=torch.float32)
                            c2w_use = c2w.to(dev_ex)
                            if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                                try:
                                    cam_pos = c2w_use[:3, 3].detach().cpu().numpy()
                                    self.block_manager.prepare_blocks_for_camera(cam_pos)
                                except Exception:
                                    pass
                            rays_o_full, rays_d_full = get_rays(H, W, fx, fy, cx, cy, c2w_use, dev_ex)
                            rd = rays_d_full.reshape(-1, 3)
                            ro = rays_o_full.reshape(-1, 3)
                            img = self._render_chunked_safe_rgb(planes_ex, dec_ex, rd, ro, dev_ex, self.truncation, H, W)
                            if img is not None:
                                out_png = os.path.join(extras_dir, f"{debug_snap}_render_rgb.png")
                                imageio.imwrite(out_png, (img * 255).astype(np.uint8))
                                print(f"Current frame RGB saved (after optimization):{out_png}")
                            else:
                                print("[Warning] After optimization, frame RGB rendering returns empty and saving has been skipped.")
                            if prev_r is not None:
                                self._renderer_pop_device(prev_r)
                        except Exception as e_img:
                            print(f"[Warning] Failed to render current frame RGB:{e_img}")
                            if prev_r is not None:
                                try:
                                    self._renderer_pop_device(prev_r)
                                except Exception:
                                    pass
                        if hasattr(self.decoders, 'block_manager') and self.decoders.block_manager is not None:
                            active_dir = os.path.join(extras_dir, "active_blocks")
                            self.mesher._visualize_active_feature_planes(self.decoders.block_manager, active_dir,
                                                                         normalize=True, grid_line_width=2,
                                                                         block_index_fontsize=8)
                    except Exception as e:
                        print(f"[Warning] Additional export (weight/activation block PNG) failed:{e}")
            except Exception as e:
                print(f"[Debug][Warning] export_all_blocks after optimize_mapping failed at seq {idx.item()}: {e}")
            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

                                           
            if idx % self.keyframe_every == 0:
                                                                      
                if idx not in self.keyframe_dict:
                    try:
                        self.keyframe_list.append(idx)
                        self.keyframe_dict[idx] = {
                            'gt_c2w': gt_c2w,
                            'idx': idx,
                            'color': gt_color.to(self.keyframe_device),
                            'depth': gt_depth.to(self.keyframe_device),
                            'est_c2w': cur_c2w.clone()
                        }
                        if self.verbose:
                            print(f"Add keyframes{idx}, the current number of keyframes:{len(self.keyframe_list)}")
                    except Exception as e:
                        print(f"Error while adding keyframes:{str(e)}")
                                     
                        try:
                            if idx not in self.keyframe_dict:
                                self.keyframe_list.append(idx)
                                self.keyframe_dict[idx] = {'idx': idx, 'est_c2w': cur_c2w.clone()}
                                print(f"Add keyframes using a simplified approach{idx}")
                        except Exception as e2:
                            print(f"Simplifying adding keyframes also fails:{str(e2)}")

            init_phase = False
            self.mapping_first_frame[0] = 1                                                      

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img - 1:
                self.logger.log(idx, self.keyframe_list)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                if self.export_blocks_instead_of_mesh:
                    if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                        try:
                            for blk_idx in list(self.block_manager.active_blocks.keys()):
                                self.block_manager.get_block_planes(blk_idx, needed='all')
                            if self.verbose:
                                print("[PreExport] The required geometry and color planes (per-idx) have been completed for the currently active block")
                        except Exception as e:
                            if self.verbose:
                                print(f"[PreExport] (per-idx) Failed to complete active block plane:{e}")
                        snap = f"{int(idx.item()):05d}"
                        intr_idx = {'H': self.H, 'W': self.W, 'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy, 'frustum_depth': 6.0}
                                                    
                        try:
                            c2w_diag = cur_c2w
                            R = c2w_diag[:3, :3]
                            t = c2w_diag[:3, 3]
                            detR = torch.det(R).item()
                            fwd = (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_diag.device)).detach().cpu().numpy()
                            p_front = (t + (R @ torch.tensor([0.0, 0.0, -1.0], device=c2w_diag.device)) * 3.0).detach().cpu()
                            blk_idx = None
                            try:
                                blk_idx = self.block_manager.get_block_index(p_front)
                            except Exception:
                                blk_idx = None
                            print(f"[SELFCHK][per-idx] det(R)={detR:.6f} | fwd={fwd} | front_block={blk_idx}")
                        except Exception as e:
                            print(f"[SELFCHK][per-idx][warn] failed: {e}")
                        out_dir = self.block_manager.export_all_blocks(
                            self.blocks_export_root, snapshot_name=snap,
                            visualize_camera=True, cam_pose=cur_c2w, intrinsics=intr_idx,
                            camera_forward_negative_z=True)
                                                                                            
                        try:
                            extras_dir = os.path.join(out_dir, "_extras")
                            os.makedirs(extras_dir, exist_ok=True)
                            pseudo_mesh = os.path.join(extras_dir, f"{snap}_blocks.ply")
                            self.mesher._save_decoder_and_features(pseudo_mesh, self.decoders, all_planes, self.device)
                                                                      
                            try:
                                H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
                                use_bm = bool(getattr(self.decoders_module, 'use_block_manager', False) or getattr(self, 'use_block_manager', False))
                                if use_bm:
                                    dev_ex = self.device
                                    dec_ex = self.decoders_module
                                    planes_ex = all_planes
                                    prev_r = None
                                else:
                                    dev_ex = self._get_extras_render_device()
                                    dec_ex = self._get_decoders_copy_on_device(dev_ex)
                                    planes_ex = self._move_all_planes_to_device(all_planes, dev_ex)
                                    prev_r = self._renderer_push_device(dev_ex)
                                try:
                                    c2w = cur_c2w
                                except Exception:
                                    c2w = self.estimate_c2w_list[min(int(idx.item()), len(self.estimate_c2w_list)-1)]
                                c2w_use = c2w.to(dev_ex) if isinstance(c2w, torch.Tensor) else torch.as_tensor(c2w, dtype=torch.float32, device=dev_ex)
                                if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                                    try:
                                        cam_pos = c2w_use[:3, 3].detach().cpu().numpy()
                                        self.block_manager.prepare_blocks_for_camera(cam_pos)
                                    except Exception:
                                        pass
                                rays_o_full, rays_d_full = get_rays(H, W, fx, fy, cx, cy, c2w_use, dev_ex)
                                rd = rays_d_full.reshape(-1, 3)
                                ro = rays_o_full.reshape(-1, 3)
                                img = self._render_chunked_safe_rgb(planes_ex, dec_ex, rd, ro, dev_ex, self.truncation, H, W)
                                if img is not None:
                                    out_png = os.path.join(extras_dir, f"{snap}_render_rgb.png")
                                    imageio.imwrite(out_png, (img * 255).astype(np.uint8))
                                    print(f"Current frame RGB saved:{out_png}")
                                else:
                                    print("[Warning] RGB rendering of the current frame returns empty and saving has been skipped")
                                if prev_r is not None:
                                    self._renderer_pop_device(prev_r)
                            except Exception as e_img:
                                print(f"[Warning] Failed to render current frame RGB:{e_img}")
                                if prev_r is not None:
                                    try:
                                        self._renderer_pop_device(prev_r)
                                    except Exception:
                                        pass
                            if hasattr(self.decoders, 'block_manager') and self.decoders.block_manager is not None:
                                active_dir = os.path.join(extras_dir, "active_blocks")
                                self.mesher._visualize_active_feature_planes(self.decoders.block_manager, active_dir,
                                                                             normalize=True, grid_line_width=2,
                                                                             block_index_fontsize=8)
                        except Exception as e:
                            print(f"[Warning] Additional export (weight/activation block PNG) failed:{e}")
                    else:
                        print("[Warning] export_blocks_instead_of_mesh=True but BlockManager is not enabled, skips block export and generates mesh instead")
                        mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                        self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                             visualize_planes_dir=self.visualize_planes_dir,
                                             use_init_decoder=self.use_init_decoder_for_mesh)
                        cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                                  estimate_c2w_list=self.estimate_c2w_list[:idx + 1])
                else:
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                         visualize_planes_dir=self.visualize_planes_dir,
                                         use_init_decoder=self.use_init_decoder_for_mesh)
                    cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                              estimate_c2w_list=self.estimate_c2w_list[:idx + 1])

            if idx == self.n_img - 1:
                if self.export_blocks_instead_of_mesh:
                    if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                        try:
                            for blk_idx in list(self.block_manager.active_blocks.keys()):
                                self.block_manager.get_block_planes(blk_idx, needed='all')
                            if self.verbose:
                                print("[PreExport] The required geometry and color planes have been completed for the currently active block (final end)")
                        except Exception as e:
                            if self.verbose:
                                print(f"[PreExport] (final end) Failed to complete active block plane:{e}")
                        out_dir = self.block_manager.export_all_blocks(self.blocks_export_root, snapshot_name='final')
                                                                                            
                        try:
                            extras_dir = os.path.join(out_dir, "_extras")
                            os.makedirs(extras_dir, exist_ok=True)
                            pseudo_mesh = os.path.join(extras_dir, "final_blocks.ply")
                            self.mesher._save_decoder_and_features(pseudo_mesh, self.decoders, all_planes, self.device)
                                                                
                            try:
                                H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
                                try:
                                    c2w = self.estimate_c2w_list[-1]
                                except Exception:
                                    c2w = torch.eye(4, device=self.device)
                                c2w_use = c2w.to(self.device) if isinstance(c2w, torch.Tensor) else torch.as_tensor(c2w, dtype=torch.float32, device=self.device)
                                if self.use_block_manager and hasattr(self, 'block_manager') and self.block_manager is not None:
                                    try:
                                        cam_pos = c2w_use[:3, 3].detach().cpu().numpy()
                                        self.block_manager.prepare_blocks_for_camera(cam_pos)
                                    except Exception:
                                        pass
                                rays_o_full, rays_d_full = get_rays(H, W, fx, fy, cx, cy, c2w_use, self.device)
                                rd = rays_d_full.reshape(-1, 3)
                                ro = rays_o_full.reshape(-1, 3)
                                render = self.renderer.render_batch_ray(all_planes, self.decoders, rd, ro, self.device, self.truncation, need_rgb=True)
                                img = render['color'].detach().reshape(H, W, 3).clamp(0, 1).cpu().numpy()
                                imageio.imwrite(os.path.join(extras_dir, "final_render_rgb.png"), (img * 255).astype(np.uint8))
                            except Exception as e_img:
                                print(f"[Warning] Failed to render final frame RGB:{e_img}")
                            if hasattr(self.decoders, 'block_manager') and self.decoders.block_manager is not None:
                                active_dir = os.path.join(extras_dir, "active_blocks")
                                self.mesher._visualize_active_feature_planes(self.decoders.block_manager, active_dir,
                                                                             normalize=True, grid_line_width=2,
                                                                             block_index_fontsize=8)
                        except Exception as e:
                            print(f"[Warning] Additional export (weight/activation block PNG) failed:{e}")
                    else:
                        print("[Warning] export_blocks_instead_of_mesh=True but BlockManager is not enabled, generates final mesh instead")
                        if self.eval_rec:
                            mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        else:
                            mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                        self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                             visualize_planes_dir=self.visualize_planes_dir,
                                             use_init_decoder=self.use_init_decoder_for_mesh)
                        cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)
                else:
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                    else:
                        mesh_out_file = f'{self.output}/mesh/final_mesh.ply'

                    self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                         visualize_planes_dir=self.visualize_planes_dir,
                                         use_init_decoder=self.use_init_decoder_for_mesh)
                    cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)

                break

            if idx == self.n_img - 1:
                break
