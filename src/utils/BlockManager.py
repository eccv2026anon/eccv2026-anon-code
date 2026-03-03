import torch
import numpy as np
import time
import os
from src.common import normalize_3d_coordinate
import shutil


class BlockManager:
                          

    def __init__(self, cfg, device, bound=None):
           
        self.device = device
        self.cfg = cfg
                                    
        self.verbose = bool(self.cfg.get('verbose', False))
        self.provided_bound = bound                         

                                        
        self.c_dim = self.cfg['model']['c_dim']
        self.c_dim_fine = self.cfg['model']['c_dim_fine']
        self.c_dim_app = self.cfg['model']['c_dim_app']

                      
        adv_cfg = self.cfg.get('model', {}).get('advanced_features', self.cfg.get('advanced_features', {}))
        self.use_color_multires = adv_cfg.get('color_multires', False)
        if self.use_color_multires:
            print(f"BlockManager: Enable color multi-resolution feature")

                                      
        self.presegmented_blocks_dir = None
                              
        if 'model' in self.cfg and 'presegmented_blocks_dir' in self.cfg['model']:
            self.presegmented_blocks_dir = self.cfg['model']['presegmented_blocks_dir']
            print(f"BlockManager: Find the pre-split block directory from the model configuration")
        elif 'preoptimized_blocks_dir' in self.cfg:
            self.presegmented_blocks_dir = self.cfg['preoptimized_blocks_dir']
            print(f"BlockManager: Locate pre-split block directory from root configuration")
        elif 'model' in self.cfg and 'preoptimized_blocks_dir' in self.cfg['model']:
            self.presegmented_blocks_dir = self.cfg['model']['preoptimized_blocks_dir']
            print(f"BlockManager: Find the pre-split blocks directory from the model configuration (using the preoptimized_blocks_dir key)")

        if self.presegmented_blocks_dir:
            print(f"BlockManager: Pre-split blocks will be loaded from the following directory:{self.presegmented_blocks_dir}")
                      
            if not os.path.exists(self.presegmented_blocks_dir):
                print(f"Warning: Pre-split chunk directory does not exist:{self.presegmented_blocks_dir}, this directory will be created")
                try:
                    os.makedirs(self.presegmented_blocks_dir, exist_ok=True)
                except Exception as e:
                    print(f"Error: Unable to create pre-partitioned block directory:{e}")
        else:
            print("BlockManager: No pre-split block directory provided, all new blocks will be initialized randomly.")

                       
                   
        self.geo_voxel_size_coarse = self.cfg.get('planes_res', {}).get('coarse', 0.24)
        self.geo_voxel_size_fine = self.cfg.get('planes_res', {}).get('fine', 0.06)
                   
        self.color_voxel_size_coarse = self.cfg.get('c_planes_res', {}).get('coarse', 0.24)
        self.color_voxel_size_fine = self.cfg.get('c_planes_res', {}).get('fine', 0.03)

        if self.verbose:
            print(f"BlockManager: Read voxel size parameters from configuration file:")
            print(f"- Coarse level of geometric features:{self.geo_voxel_size_coarse}")
            print(f"- Detailed level of geometric features:{self.geo_voxel_size_fine}")
            print(f"- Color feature coarse level:{self.color_voxel_size_coarse}")
            print(f"- Color feature level:{self.color_voxel_size_fine}")

                                         
        self.first_frame_world_offset = None
        self.apply_first_frame_offset_in_block_index = bool(
            self.cfg.get('model', {}).get('apply_first_frame_offset_in_block_index', True)
        )
        try:
            ff_pose_cfg = None
            if 'first_frame_abs_pose' in self.cfg:
                ff_pose_cfg = self.cfg['first_frame_abs_pose']
            elif 'model' in self.cfg and 'first_frame_abs_pose' in self.cfg['model']:
                ff_pose_cfg = self.cfg['model']['first_frame_abs_pose']
            if ff_pose_cfg is not None:
                ff_pose = torch.tensor(ff_pose_cfg, dtype=torch.float32)
                if ff_pose.shape == (4, 4):
                    self.first_frame_world_offset = ff_pose[:3, 3].clone()
                    print(f"BlockManager: Read first_frame_abs_pose translation vector:{self.first_frame_world_offset.tolist()}")
        except Exception as e:
            print(f"Warning: Failed to parse first_frame_abs_pose:{e}")

                                     
                     
        block_size_xy_cfg = self.cfg['model'].get('block_size', 7.68)
                                                      
        if isinstance(block_size_xy_cfg, (list, tuple, np.ndarray)):
            self.block_size_xy = float(block_size_xy_cfg[0]) if len(block_size_xy_cfg) > 0 else 7.68
        elif isinstance(block_size_xy_cfg, torch.Tensor):
            flat = block_size_xy_cfg.flatten()
            self.block_size_xy = float(flat[0].item()) if flat.numel() > 0 else 7.68
        else:
            self.block_size_xy = float(block_size_xy_cfg)
                                 

                                          
                                
        if self.provided_bound is not None:
                                   
            if self.provided_bound.shape == (3, 2):
                                    
                self.global_bound = self.provided_bound.transpose(0, 1).to(device)
                print(f"BlockManager: Use the expanded bounds provided by StructRecon and transpose to [2,3] format")
            elif self.provided_bound.shape == (2, 3):
                          
                self.global_bound = self.provided_bound.to(device)
                print(f"BlockManager: Use the expanded bounds provided by StructRecon, shape [2,3]")
            else:
                print(f"Warning: The bounding shape provided is{self.provided_bound.shape}, try to reinvent...")
                if self.provided_bound.numel() == 6:                 
                    try:
                        reshaped_bound = self.provided_bound.reshape(2, 3).to(device)
                        self.global_bound = reshaped_bound
                        print(f"BlockManager: The provided bounds have been reshaped into [2,3] format")
                    except RuntimeError as e:
                        print(f"Error: Unable to reshape the provided bounds to [2,3] format (runtime error:{e}), will try to read from the configuration")
                        self.provided_bound = None                    
                    except Exception as e:
                        print(f"Error: Boundary reshape failed{type(e).__name__}: {e}, will try to read from the configuration")
                        self.provided_bound = None                    
                else:
                    print(f"Error: The number of boundary elements provided is not 6, cannot be reshaped to [2,3] format, will try to read from configuration")
                    self.provided_bound = None                    
                               
        elif 'bound' in self.cfg:
            self.global_bound = torch.tensor(self.cfg['bound'], device=device)
            print(f"BlockManager: Read global_bound from configuration file root level")
        elif 'mapping' in self.cfg and 'bound' in self.cfg['mapping']:
            self.global_bound = torch.tensor(self.cfg['mapping']['bound'], device=device)
            print(f"BlockManager: Read global_bound from the mapping part")
        else:
            print(f"Error: 'bound' parameter not found in configuration file. Please check the configuration file.")
            raise KeyError("'bound' parameter not found in configuration file")

        if self.verbose:
            print(f"BlockManager.__init__: global_bound type=<class 'torch.Tensor'>, shape={self.global_bound.shape}")
        if self.global_bound.shape != (2, 3):
            if self.global_bound.shape == (3, 2):
                self.global_bound = self.global_bound.transpose(0, 1)
                print("BlockManager: Global_bound has been transposed from [3, 2] to [2, 3] format")
            else:
                print(f"Warning: The shape of global_bound is neither [2, 3] nor [3, 2], but{self.global_bound.shape}")

                                                         
        self.world_min = self.global_bound[0]            
        self.world_max = self.global_bound[1]            

                                         
        if self.verbose:
            print(f"BlockManager: Global bounds successfully loaded with scope:")
            print(f"- Minimum point:{self.world_min}")
            print(f"- Maximum point:{self.world_max}")
            print(f"- Shape:{self.global_bound.shape}")
            print(f"- equipment:{self.global_bound.device}")

                       
        if self.global_bound.device != device:
            self.global_bound = self.global_bound.to(device)
            self.world_min = self.world_min.to(device)
            self.world_max = self.world_max.to(device)
            print(f"BlockManager: Global boundary moved to device{device}")

                               
                               
                                                    
                        
        self.block_size_z_tensor = (self.world_max[2] - self.world_min[2]).to(self.world_max.dtype)
        try:
            _bz = float(self.block_size_z_tensor.detach().cpu().item())
        except Exception:
            _bz = float(self.block_size_z_tensor.reshape(-1)[0].item())
                                             
        self.block_size_z = round(_bz, 8)
        if self.verbose:
            print(f"BlockManager: Set the Z-axis block size to the entire scene height:{self.block_size_z:.4f}")

        try:
                               
                       
            self.geo_n_voxels_coarse = {
                'xy': (int(round(self.block_size_xy / self.geo_voxel_size_coarse)),
                       int(round(self.block_size_xy / self.geo_voxel_size_coarse))),
                'xz': (int(round(self.block_size_xy / self.geo_voxel_size_coarse)),
                       int(round(self.block_size_z / self.geo_voxel_size_coarse))),
                'yz': (int(round(self.block_size_xy / self.geo_voxel_size_coarse)),
                       int(round(self.block_size_z / self.geo_voxel_size_coarse)))
            }

            self.geo_n_voxels_fine = {
                'xy': (int(round(self.block_size_xy / self.geo_voxel_size_fine)),
                       int(round(self.block_size_xy / self.geo_voxel_size_fine))),
                'xz': (int(round(self.block_size_xy / self.geo_voxel_size_fine)),
                       int(round(self.block_size_z / self.geo_voxel_size_fine))),
                'yz': (int(round(self.block_size_xy / self.geo_voxel_size_fine)),
                       int(round(self.block_size_z / self.geo_voxel_size_fine)))
            }

                       
            self.color_n_voxels_coarse = {
                'xy': (int(round(self.block_size_xy / self.color_voxel_size_coarse)),
                       int(round(self.block_size_xy / self.color_voxel_size_coarse))),
                'xz': (int(round(self.block_size_xy / self.color_voxel_size_coarse)),
                       int(round(self.block_size_z / self.color_voxel_size_coarse))),
                'yz': (int(round(self.block_size_xy / self.color_voxel_size_coarse)),
                       int(round(self.block_size_z / self.color_voxel_size_coarse)))
            }

            self.color_n_voxels_fine = {
                'xy': (int(round(self.block_size_xy / self.color_voxel_size_fine)),
                       int(round(self.block_size_xy / self.color_voxel_size_fine))),
                'xz': (int(round(self.block_size_xy / self.color_voxel_size_fine)),
                       int(round(self.block_size_z / self.color_voxel_size_fine))),
                'yz': (int(round(self.block_size_xy / self.color_voxel_size_fine)),
                       int(round(self.block_size_z / self.color_voxel_size_fine)))
            }

            if self.verbose:
                print(f"BlockManager: Number of voxels calculated:")
                print(
                    f"- Coarse level xy of geometric features:{self.geo_n_voxels_coarse['xy']}, xz: {self.geo_n_voxels_coarse['xz']}, yz: {self.geo_n_voxels_coarse['yz']}")
                print(
                    f"- Detailed level of geometric features xy:{self.geo_n_voxels_fine['xy']}, xz: {self.geo_n_voxels_fine['xz']}, yz: {self.geo_n_voxels_fine['yz']}")
                print(
                    f"-Color feature coarse level xy:{self.color_n_voxels_coarse['xy']}, xz: {self.color_n_voxels_coarse['xz']}, yz: {self.color_n_voxels_coarse['yz']}")
                print(
                    f"- Color feature level xy:{self.color_n_voxels_fine['xy']}, xz: {self.color_n_voxels_fine['xz']}, yz: {self.color_n_voxels_fine['yz']}")

        except Exception as e:
                          
            print(f"Warning: Voxel count calculation failed:{e}, the default value will be used")

                 
            default_xy_voxels = 128              
            default_z_voxels = 16             

            self.geo_n_voxels_coarse = {
                'xy': (default_xy_voxels, default_xy_voxels),
                'xz': (default_xy_voxels, default_z_voxels),
                'yz': (default_xy_voxels, default_z_voxels)
            }
            self.geo_n_voxels_fine = {
                'xy': (default_xy_voxels * 4, default_xy_voxels * 4),
                'xz': (default_xy_voxels * 4, default_z_voxels * 4),
                'yz': (default_xy_voxels * 4, default_z_voxels * 4)
            }
            self.color_n_voxels_coarse = self.geo_n_voxels_coarse
            self.color_n_voxels_fine = self.geo_n_voxels_fine

                                         
        self.block_n_voxels = [
            self.geo_n_voxels_coarse['xy'][0],           
            self.geo_n_voxels_coarse['xy'][1],           
            self.geo_n_voxels_coarse['xz'][1]           
        ]

        if self.verbose:
            print(f"BlockManager: Compatibility block_n_voxels is set to:{self.block_n_voxels}")

                        
        voxel_size_xy = torch.tensor([self.geo_voxel_size_coarse, self.geo_voxel_size_coarse]).to(device)
        voxel_size_z = torch.tensor([self.geo_voxel_size_coarse]).to(device)

                     
        self.voxel_size = torch.cat([voxel_size_xy, voxel_size_z]).to(device)
        if self.verbose:
            print(f"BlockManager: The voxel size used (voxel_size) is:{self.voxel_size}")

                                    
        if self.verbose:
            print(
                f"BlockManager: Z-axis block size remains as scene height:{self.block_size_z:.4f}, the xy plane block size is:{self.block_size_xy:.4f}")

                                                                      
        self.compute_block_structure()

                                                              
        self.block_plane_shapes = self._calculate_block_plane_shapes()
                                    
        if self.block_plane_shapes is None:
            print("Fatal error: _calculate_block_plane_shapes method returned None")
                         
            self.block_plane_shapes = {
                'geo_feat_xy_coarse': (self.c_dim, self.geo_n_voxels_coarse['xy'][0], self.geo_n_voxels_coarse['xy'][1]),
                'geo_feat_xz_coarse': (self.c_dim, self.geo_n_voxels_coarse['xz'][0], self.geo_n_voxels_coarse['xz'][1]),
                'geo_feat_yz_coarse': (self.c_dim, self.geo_n_voxels_coarse['yz'][0], self.geo_n_voxels_coarse['yz'][1]),
                'geo_feat_xy_fine': (self.c_dim_fine, self.geo_n_voxels_fine['xy'][0], self.geo_n_voxels_fine['xy'][1]),
                'geo_feat_xz_fine': (self.c_dim_fine, self.geo_n_voxels_fine['xz'][0], self.geo_n_voxels_fine['xz'][1]),
                'geo_feat_yz_fine': (self.c_dim_fine, self.geo_n_voxels_fine['yz'][0], self.geo_n_voxels_fine['yz'][1]),
                'app_feat_xy': (self.c_dim_app, self.color_n_voxels_coarse['xy'][0], self.color_n_voxels_coarse['xy'][1]),
                'app_feat_xz': (self.c_dim_app, self.color_n_voxels_coarse['xz'][0], self.color_n_voxels_coarse['xz'][1]),
                'app_feat_yz': (self.c_dim_app, self.color_n_voxels_coarse['yz'][0], self.color_n_voxels_coarse['yz'][1]),

            }

                                  
        self.active_blocks = {}                          
        self.block_usage_time = {}                           
        self.cached_blocks = {}                        
        self.cached_usage_time = {}                             
        self.transposed_planes = {}                                

                  
        adv_cfg = self.cfg.get('model', {}).get('advanced_features', self.cfg.get('advanced_features', {}))
        self.enable_dynamic_memory = adv_cfg.get('dynamic_memory', True)
        self.max_active_blocks = self.cfg.get('model', {}).get('initial_max_active_blocks', 20)       
                                      
        self.max_cached_blocks = int(self.cfg.get('model', {}).get('max_cached_blocks', 200))
                             
        self.max_prefetch_blocks = int(self.cfg.get('model', {}).get('max_prefetch_blocks', 30))
        self.memory_check_interval = 50                       
        self.memory_check_counter = 0

                
        self.enable_block_consistency = adv_cfg.get('block_consistency', False)
        if self.enable_block_consistency:
            if self.verbose:
                print("BlockManager: Enable inter-block consistency checking")

    def _block_filename(self, idx_tuple):
                       
        return f"block_{idx_tuple[0]}_{idx_tuple[1]}_{idx_tuple[2]}.pth"

    def _tensor_dict_from_params(self, param_dict: dict):
                                                    
        out = {}
        for k, v in param_dict.items():
            try:
                t = v.detach().cpu() if isinstance(v, torch.nn.Parameter) else torch.as_tensor(v).detach().cpu()
            except Exception:
                t = torch.as_tensor(v).detach().cpu()
            out[k] = t
        return out

    def _tensor_dict_from_params_with_transpose_restore(self, param_dict: dict, transposed_planes: set):
           
        out = {}
        for k, v in param_dict.items():
            try:
                t = v.detach().cpu() if isinstance(v, torch.nn.Parameter) else torch.as_tensor(v).detach().cpu()
            except Exception:
                t = torch.as_tensor(v).detach().cpu()
            
                            
            if len(t.shape) == 3 and k in transposed_planes:                                         
                t = t.permute(0, 2, 1)          
            
            out[k] = t
        return out

    def export_all_blocks(self, export_root: str, snapshot_name: str = None,
                          mark_active_in_filename: bool = True,
                          active_marker_suffix: str = "_active",
                          use_distance_filter: bool = False,
                          center: torch.Tensor = None,
                          radius: float = None,
                          visualize_camera: bool = False,
                          cam_pose: torch.Tensor = None,
                          intrinsics: dict = None,
                          vis_out_path: str = None,
                          camera_forward_negative_z: bool = False):
           
                
        out_dir = export_root
        if snapshot_name is not None and len(str(snapshot_name)) > 0:
            out_dir = os.path.join(export_root, str(snapshot_name))
        os.makedirs(out_dir, exist_ok=True)

                    
        grid = self.block_grid_size
        total = int(grid[0].item() * grid[1].item() * grid[2].item())
        cnt = 0
        print(f"BlockManager.export_all_blocks: Start export{total}block arrives{out_dir}")
                  
        if use_distance_filter:
            if center is None or radius is None:
                print("Warning: use_distance_filter=True but no center or radius is provided, distance filtering will be ignored")
                use_distance_filter = False
            else:
                if not isinstance(center, torch.Tensor):
                    center = torch.tensor(center, dtype=torch.float32, device=self.device)
                else:
                    center = center.to(self.device, dtype=torch.float32).reshape(-1)
                radius = float(radius)
                                  
        vis_grid = None
        if visualize_camera:
            try:
                gx, gy, gz = int(grid[0].item()), int(grid[1].item()), int(grid[2].item())
                vis_grid = np.zeros((gy, gx), dtype=np.uint8)              
            except Exception:
                vis_grid = None

        for i in range(int(grid[0].item())):
            for j in range(int(grid[1].item())):
                for k in range(int(grid[2].item())):
                    idx_t = (int(i), int(j), int(k))
                    fname_base = self._block_filename(idx_t)
                                                       
                    mark_as_active = mark_active_in_filename and (idx_t in self.active_blocks) and (len(self.active_blocks[idx_t]) > 0)
                    if mark_as_active and use_distance_filter:
                        try:
                            bb = self.get_block_bound(idx_t)         
                            bb_center = (bb[0] + bb[1]) * 0.5
                                                       
                            dist = torch.norm(bb_center - center)
                            if float(dist.item()) > radius:
                                mark_as_active = False
                        except Exception as e:
                            print(f"Warning: Failed to calculate block center distance{idx_t}: {e}, skip distance filtering")

                    if mark_as_active:
                        if fname_base.endswith('.pth'):
                            fname = fname_base[:-4] + f"{active_marker_suffix}.pth"
                        else:
                            fname = fname_base + f"{active_marker_suffix}"
                    else:
                        fname = fname_base
                    dst_path = os.path.join(out_dir, fname)

                                                                 
                    if idx_t in self.active_blocks and len(self.active_blocks[idx_t]) > 0:
                                                        
                        transposed_planes = self.transposed_planes.get(idx_t, set())
                        tensor_dict = self._tensor_dict_from_params_with_transpose_restore(self.active_blocks[idx_t], transposed_planes)
                        torch.save(tensor_dict, dst_path)
                    elif idx_t in self.cached_blocks and len(self.cached_blocks[idx_t]) > 0:
                                                
                        transposed_planes = self.transposed_planes.get(idx_t, set())
                        tensor_dict = self._tensor_dict_from_params_with_transpose_restore(self.cached_blocks[idx_t], transposed_planes)
                        torch.save(tensor_dict, dst_path)
                    else:
                        src_path = None
                        if self.presegmented_blocks_dir is not None:
                                                             
                            src_path = os.path.join(self.presegmented_blocks_dir, fname_base)
                        if src_path is not None and os.path.exists(src_path):
                            try:
                                shutil.copy2(src_path, dst_path)
                            except Exception as e:
                                print(f"Failed to copy pre-split block{src_path} -> {dst_path}: {e}, a placeholder block will be created")
                                zeros_dict = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in self.block_plane_shapes.items()}
                                torch.save(zeros_dict, dst_path)
                        else:
                                              
                            zeros_dict = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in self.block_plane_shapes.items()}
                            torch.save(zeros_dict, dst_path)

                              
                    if vis_grid is not None and mark_as_active:
                        try:
                            vis_grid[j, i] = 1         
                        except Exception:
                            pass

                    cnt += 1
                    if cnt % 50 == 0:
                        print(f"Exported{cnt}/{total}piece")

        print(f"BlockManager.export_all_blocks: Completed, export directory:{out_dir}")

                       
        if visualize_camera and vis_grid is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Arrow
                        
                if vis_out_path is None or len(str(vis_out_path)) == 0:
                    extras = os.path.join(out_dir, '_extras')
                    os.makedirs(extras, exist_ok=True)
                    vis_out_path = os.path.join(extras, 'active_blocks_with_camera.png')

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(vis_grid, origin='lower', cmap='Greys', alpha=0.6)
                ax.set_title('Active Blocks (XY) with Camera Pose/Frustum')
                ax.set_xlabel('Block X index')
                ax.set_ylabel('Block Y index')

                               
                def world_to_block_xy(xw, yw):
                    bx = float((xw - float(self.world_min[0])) / float(self.block_size_xy))
                    by = float((yw - float(self.world_min[1])) / float(self.block_size_xy))
                    return bx, by

                         
                if cam_pose is not None and isinstance(cam_pose, torch.Tensor) and cam_pose.numel() >= 12:
                    pose = cam_pose.detach().cpu().float().reshape(4, 4)
                    cam_t = pose[:3, 3]
                    bx, by = world_to_block_xy(float(cam_t[0]), float(cam_t[1]))
                    ax.plot([bx], [by], 'ro', label='Camera')
                                                           
                    fwd = -pose[:3, 2] if camera_forward_negative_z else pose[:3, 2]
                    fx, fy = float(fwd[0]), float(fwd[1])
                    ax.add_patch(Arrow(bx, by, fx*0.8, fy*0.8, width=0.3, color='r'))

                    if isinstance(intrinsics, dict):
                        H = int(intrinsics.get('H', 480))
                        W = int(intrinsics.get('W', 640))
                        fx_i = float(intrinsics.get('fx', 525.0))
                        fy_i = float(intrinsics.get('fy', 525.0))
                        cx_i = float(intrinsics.get('cx', W/2))
                        cy_i = float(intrinsics.get('cy', H/2))
                        d = float(intrinsics.get('frustum_depth', 5.0))
                        pix = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
                        cam_pts = []
                        for u,v in pix:
                            x = (u - cx_i) * d / fx_i
                                                                                   
                            if camera_forward_negative_z:
                                y = -(v - cy_i) * d / fy_i
                                z = -d
                            else:
                                y = (v - cy_i) * d / fy_i
                                z = d
                            cam_pts.append([x,y,z])
                        cam_pts = torch.tensor(cam_pts, dtype=torch.float32)
                        R = pose[:3, :3]
                        t = pose[:3, 3]
                        wpts = (R @ cam_pts.T).T + t
                        bx_list, by_list = [], []
                        for p in wpts:
                            bxi, byi = world_to_block_xy(float(p[0]), float(p[1]))
                            bx_list.append(bxi); by_list.append(byi)
                        bx_list.append(bx_list[0]); by_list.append(by_list[0])
                        ax.plot(bx_list, by_list, 'y-', lw=1.5, label='Frustum@d')

                ax.legend(loc='upper right')
                fig.tight_layout()
                fig.savefig(vis_out_path, dpi=150)
                plt.close(fig)
                print(f"BlockManager: View frustum/camera visualization saved:{vis_out_path}")
                
                                                       
                try:
                    import math
                    pose_txt_path = vis_out_path.replace('.png', '_camera_pose.txt')
                    
                              
                    camera_forward = -pose[:3, 2] if camera_forward_negative_z else pose[:3, 2]
                    forward_angle_rad = math.atan2(float(camera_forward[1]), float(camera_forward[0]))
                    forward_angle_deg = math.degrees(forward_angle_rad)
                    if forward_angle_deg < 0:
                        forward_angle_deg += 360
                    
                            
                    with open(pose_txt_path, 'w', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write("StructRecon camera pose information (saved when generated from active_blocks_with_camera.png)\\n")
                        f.write("=" * 80 + "\n\n")
                        
                        f.write("## Coordinate system convention\\n")
                        if camera_forward_negative_z:
                            f.write("Camera coordinate system: OpenCV convention (camera orientation = -Z axis)\\n")
                        else:
                            f.write("Camera coordinate system: OpenGL convention (camera orientation = +Z axis)\\n")
                        f.write("\n")
                        
                        f.write("## Complete pose matrix (4x4)\\n")
                        for i in range(4):
                            row_str = "  ".join([f"{float(pose[i, j]):12.6f}" for j in range(4)])
                            f.write(f"{row_str}\n")
                        f.write("\n")
                        
                        f.write("## Camera position (world coordinates)\\n")
                        f.write(f"X: {float(pose[0, 3]):12.6f}m\n")
                        f.write(f"Y: {float(pose[1, 3]):12.6f}m\n")
                        f.write(f"Z: {float(pose[2, 3]):12.6f}m\n")
                        f.write("\n")
                        
                        f.write("## Rotation matrix (3x3)\\n")
                        for i in range(3):
                            row_str = "  ".join([f"{float(pose[i, j]):12.6f}" for j in range(3)])
                            f.write(f"{row_str}\n")
                        f.write("\n")
                        
                        f.write("## The direction of each axis of the camera coordinate system (in the world coordinate system)\\n")
                        f.write(f"Camera X-axis (right): [{float(pose[0, 0]):9.6f}, {float(pose[1, 0]):9.6f}, {float(pose[2, 0]):9.6f}]\n")
                        f.write(f"Camera Y axis (bottom): [{float(pose[0, 1]):9.6f}, {float(pose[1, 1]):9.6f}, {float(pose[2, 1]):9.6f}]\n")
                        f.write(f"Camera Z-axis (rear): [{float(pose[0, 2]):9.6f}, {float(pose[1, 2]):9.6f}, {float(pose[2, 2]):9.6f}]\n")
                        f.write("\n")
                        
                        f.write("## Actual direction of the camera (arrow direction in visualization)\\n")
                        if camera_forward_negative_z:
                            f.write("Calculation method: -Z axis direction (OpenCV convention)\\n")
                        else:
                            f.write("Calculation method: +Z axis direction (OpenGL convention)\\n")
                        f.write(f"Orientation vector: [{float(camera_forward[0]):9.6f}, {float(camera_forward[1]):9.6f}, {float(camera_forward[2]):9.6f}]\n")
                        f.write(f"Horizontal angle:{forward_angle_deg:9.2f}(0 =east, 90 =north, 180 =west, 270 =south)\n")
                        f.write("\n")
                        
                        f.write("## Visualize arrow information\\n")
                        f.write(f"Arrow starting point (block coordinates): ({bx:.2f}, {by:.2f})\n")
                        f.write(f"Arrow direction (block coordinates): ({fx:.4f}, {fy:.4f})\n")
                        f.write("\n")
                        
                        f.write("=" * 80 + "\n")
                        f.write("Instructions for use:\\n")
                        f.write("1. Compare this file with first_frame_abs_pose\\n saved by osm_pose_selector\\n")
                        f.write("2. Check whether the rotation matrix and translation vector are consistent\\n")
                        f.write("3. Check whether the camera orientation angle is as expected\\n")
                        f.write("4. If the directions are inconsistent, it may be a problem with the coordinate system convention or map rotation compensation\\n")
                        f.write("=" * 80 + "\n")
                    
                    print(f"INFO: Camera pose saved to:{pose_txt_path}")
                    print(f"- Camera position: ({float(pose[0,3]):.2f}, {float(pose[1,3]):.2f}, {float(pose[2,3]):.2f}) rice")
                    print(f"- Camera orientation:{forward_angle_deg:.2f}(horizontal)")
                    
                except Exception as e_pose:
                    print(f"Warning: Failed to save camera pose information:{e_pose}")
                                                                    
                
            except Exception as e:
                print(f"Warning: Failed to draw camera frustum:{e}")

        return out_dir

    def _calculate_block_plane_shapes(self):
           
        try:
                    
            if not hasattr(self, 'c_dim') or not hasattr(self, 'c_dim_fine') or not hasattr(self, 'c_dim_app'):
                print("Error: Missing required dimension parameters: c_dim, c_dim_fine, c_dim_app")
                return None

            if not hasattr(self, 'geo_n_voxels_coarse') or not hasattr(self, 'geo_n_voxels_fine') or\
               not hasattr(self, 'color_n_voxels_coarse') or not hasattr(self, 'color_n_voxels_fine'):
                print("Error: The number of voxels has not been calculated.")
                return None

            block_plane_shapes = {}

                                
            block_plane_shapes.update({
                'geo_feat_xy_coarse': (
                    self.c_dim, self.geo_n_voxels_coarse['xy'][0], self.geo_n_voxels_coarse['xy'][1]),
                'geo_feat_xz_coarse': (
                    self.c_dim, self.geo_n_voxels_coarse['xz'][0], self.geo_n_voxels_coarse['xz'][1]),
                'geo_feat_yz_coarse': (
                    self.c_dim, self.geo_n_voxels_coarse['yz'][0], self.geo_n_voxels_coarse['yz'][1]),
                'geo_feat_xy_fine': (
                    self.c_dim_fine, self.geo_n_voxels_fine['xy'][0], self.geo_n_voxels_fine['xy'][1]),
                'geo_feat_xz_fine': (
                    self.c_dim_fine, self.geo_n_voxels_fine['xz'][0], self.geo_n_voxels_fine['xz'][1]),
                'geo_feat_yz_fine': (
                    self.c_dim_fine, self.geo_n_voxels_fine['yz'][0], self.geo_n_voxels_fine['yz'][1]),
            })

                                      
            if self.use_color_multires:
                block_plane_shapes.update({
                    'app_feat_xy_coarse': (
                        self.c_dim_app, self.color_n_voxels_coarse['xy'][0], self.color_n_voxels_coarse['xy'][1]),
                    'app_feat_xz_coarse': (
                        self.c_dim_app, self.color_n_voxels_coarse['xz'][0], self.color_n_voxels_coarse['xz'][1]),
                    'app_feat_yz_coarse': (
                        self.c_dim_app, self.color_n_voxels_coarse['yz'][0], self.color_n_voxels_coarse['yz'][1]),
                    'app_feat_xy_fine': (
                        self.c_dim_app, self.color_n_voxels_fine['xy'][0], self.color_n_voxels_fine['xy'][1]),
                    'app_feat_xz_fine': (
                        self.c_dim_app, self.color_n_voxels_fine['xz'][0], self.color_n_voxels_fine['xz'][1]),
                    'app_feat_yz_fine': (
                        self.c_dim_app, self.color_n_voxels_fine['yz'][0], self.color_n_voxels_fine['yz'][1]),
                })
                print("BlockManager: Enabled thick and thin levels for color feature planes")
            else:
                block_plane_shapes.update({
                    'app_feat_xy': (
                        self.c_dim_app, self.color_n_voxels_coarse['xy'][0], self.color_n_voxels_coarse['xy'][1]),
                    'app_feat_xz': (
                        self.c_dim_app, self.color_n_voxels_coarse['xz'][0], self.color_n_voxels_coarse['xz'][1]),
                    'app_feat_yz': (
                        self.c_dim_app, self.color_n_voxels_coarse['yz'][0], self.color_n_voxels_coarse['yz'][1]),
                })
                print("BlockManager: Use a single level for color feature planes")

            print(f"BlockManager: Successfully calculated block plane shape, including{len(block_plane_shapes)}feature plane")
            return block_plane_shapes
        except Exception as e:
            print(f"Error: Exception occurred while calculating block plan shape:{e}")
            return None
    def compute_block_structure(self):
                              
                   
        bound_range = self.global_bound[1] - self.global_bound[0]

                  
                                                    
                                                             
        block_size_xy_tensor = torch.tensor(self.block_size_xy, device=self.device, dtype=bound_range.dtype)
        block_size_vec = torch.stack([
            block_size_xy_tensor,
            block_size_xy_tensor,
            self.block_size_z_tensor
        ])

                      
        ratio = bound_range / block_size_vec
                                                 
        eps = torch.tensor(1e-6, dtype=ratio.dtype, device=ratio.device)
        ratio = ratio - eps
        self.block_grid_size = torch.ceil(ratio).int()

        if self.verbose:
            print(f"BlockManager: Scene boundary range:{bound_range}")
            print(f"BlockManager: Use block size vector:{block_size_vec}")
            print(f"BlockManager: Block grid size:{self.block_grid_size}")

                 
        self.total_blocks = self.block_grid_size.prod().item()
        if self.verbose:
            print(f"BlockManager: Total number of blocks:{self.total_blocks}")

    def get_all_possible_block_indices(self):
           
        indices = []
        for i in range(self.block_grid_size[0]):
            for j in range(self.block_grid_size[1]):
                for k in range(self.block_grid_size[2]):
                    indices.append((i, j, k))
        return indices

    def get_block_index(self, p):
           
                      
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, device=self.device, dtype=torch.float32)
        p = p.to(self.device)

                      
        if p.dim() == 1:
            p = p.unsqueeze(0)
            is_single = True
        else:
            is_single = False

        try:
                               
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"Warning: Input point coordinates contain NaN or Inf values:{p}")
                if is_single:
                    return (0, 0, 0)           
                else:
                    return [(0, 0, 0)] * p.shape[0]             

                   
                                         
            p_input = p

            p_clamped = torch.max(torch.min(p_input, self.global_bound[1] - 1e-4), self.global_bound[0] + 1e-4)

                                
            relative_pos = p_clamped - self.global_bound[0]

                                                                 
            block_size_vec = torch.tensor([self.block_size_xy, self.block_size_xy, self.block_size_z],
                                          device=self.device)

                            
            block_idx = (relative_pos / block_size_vec).floor().long()

                         
            if hasattr(self, 'block_grid_size') and isinstance(self.block_grid_size, torch.Tensor):
                max_indices = self.block_grid_size - 1
                block_idx = torch.min(torch.max(block_idx, torch.zeros_like(block_idx)), max_indices)

            if is_single:
                                                                                       
                reshaped_block_idx = block_idx.reshape(block_idx.shape[0], -1)
                list_of_tuples = [tuple(map(int, row)) for row in reshaped_block_idx.cpu().tolist()]
                return list_of_tuples[0]
            else:
                                                                  
                return block_idx

        except Exception as e:
            print(f"Warning: Error calculating block index:{e}")
            if is_single:
                return (0, 0, 0)           
            else:
                return [(0, 0, 0)] * p.shape[0]             

    def get_block_bound(self, block_idx):
           
        block_idx_tensor = torch.tensor(block_idx, device=self.device, dtype=torch.float32)

                                                             
        block_size_vec = torch.tensor([self.block_size_xy, self.block_size_xy, self.block_size_z], device=self.device)

                                                     
        min_bound = self.global_bound[0] + block_idx_tensor * block_size_vec

                                                     
        max_bound = min_bound + block_size_vec

        return torch.stack([min_bound, max_bound])

    def _create_block(self, block_idx, needed: str = 'all'):
           
        block_path = ""
        if self.presegmented_blocks_dir:
            block_path = os.path.join(self.presegmented_blocks_dir,
                                      f'block_{block_idx[0]}_{block_idx[1]}_{block_idx[2]}.pth')

        new_block = {}

                   
                                         
        if needed not in ('all', 'geo', 'color'):
            needed = 'all'
        def _is_needed(name: str) -> bool:
            if needed == 'all':
                return True
            if needed == 'geo':
                return name.startswith('geo_feat_')
            if needed == 'color':
                return name.startswith('app_feat_')
            return True

        if self.presegmented_blocks_dir and os.path.exists(block_path):
            try:
                if self.verbose:
                    print(f"try to start from{block_path}load block{block_idx}")
                block_data = torch.load(block_path, map_location=self.device)
                                                                                        
                      
                                                                                                                         
                                                             
                                                                                                                            
                                                         
                                                                                                                        
                                                                                      
                                        
                                                                                                                       
                                  
                if not block_data:
                    print(f"Warning: Loading pre-split chunks{block_path}is empty. will be initialized randomly.")
                else:
                                                 
                    new_block = {}

                                                           
                    transposed_planes = set()
                    for name, expected_shape in self.block_plane_shapes.items():
                                                    
                        if not _is_needed(name):
                            continue
                        if name not in block_data:
                            if self.verbose:
                                print(f"Warning: Pre-split chunks{block_path}Missing tensor '{name}'. This tensor will be initialized randomly.")
                            continue

                                  
                        actual_shape = block_data[name].shape
                        if actual_shape != expected_shape:
                                                  
                            if (len(actual_shape) == len(expected_shape) and
                                    actual_shape[0] == expected_shape[0] and         
                                    actual_shape[1] * actual_shape[2] == expected_shape[1] * expected_shape[
                                        2]):          

                                if self.verbose:
                                    print(
                                        f"Note: Pre-split blocks{block_path}tensor in '{name}' Dimension order mismatch. Try transposing:{actual_shape} -> {expected_shape}")
                                            
                                try:
                                    block_data[name] = block_data[name].permute(0, 2, 1)
                                    if block_data[name].shape == expected_shape:
                                        if self.verbose:
                                            print(f"Successfully transposed tensor '{name}' Dimensions:{actual_shape} -> {expected_shape}")
                                               
                                        new_block[name] = torch.nn.Parameter(block_data[name])
                                        transposed_planes.add(name)              
                                    else:
                                        if self.verbose:
                                            print(f"The shapes still don't match after transposing:{block_data[name].shape} != {expected_shape}")
                                                      
                                except Exception as e:
                                    if self.verbose:
                                        print(f"Error transposing tensor:{e}")
                                                  
                            else:
                                if self.verbose:
                                    print(
                                        f"Warning: Pre-split chunks{block_path}tensor in '{name}'Shapes do not match. Desired shape:{expected_shape}, actual shape:{actual_shape}. This tensor will be initialized randomly.")
                                              
                        else:
                                        
                            new_block[name] = torch.nn.Parameter(block_data[name])

                    if self.verbose:
                        print(f"from pre-partitioned blocks{block_path}Loaded successfully{len(new_block)}a characteristic plane.")

                                      
                    self.active_blocks[block_idx] = new_block
                    self.transposed_planes[block_idx] = transposed_planes            
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Loading pre-split chunks{block_path}fail:{e}. will be initialized randomly.")

                                                          
        if block_idx not in self.active_blocks:
                                
            new_block = {}
            self.active_blocks[block_idx] = new_block
            if self.verbose:
                print(f"for blocks{block_idx}Create a new feature plane dictionary")
                                                                                                
                  
                                                                                                                     
                                                         
                                                                                                                        
                                                                  
                                                                                                                                 
                                                                                  
                                    
                                                                                                                                

                                                                             
              
                                                                                                                 
                                                     
                                                                                                                    
                                                                    
                                                                                                                                   
                                                                              
                                
                                                                                                                        

                                     
        missing_planes = []
        for name, shape in self.block_plane_shapes.items():
            if not _is_needed(name):
                                 
                continue
            if name not in self.active_blocks[block_idx]:
                missing_planes.append(name)
                self.active_blocks[block_idx][name] = torch.nn.Parameter(torch.randn(*shape, device=self.device) * 0.1)

        if self.verbose:
            if missing_planes:
                print(f"for blocks{block_idx}Randomly initialized{len(missing_planes)}missing feature planes:{', '.join(missing_planes)}")
            else:
                print(f"piece{block_idx}All feature planes are loaded, no random initialization is required")

    def get_block_planes(self, block_idx, needed: str = 'all'):
           
                                                          
        if isinstance(block_idx, torch.Tensor) and block_idx.dim() == 2 and block_idx.shape[1] == 3:
                                                 
            block_planes_list = []
            for i in range(block_idx.shape[0]):
                                
                single_block_idx = tuple(block_idx[i].cpu().numpy().astype(int))
                               
                single_block_planes = self.get_block_planes(single_block_idx, needed=needed)
                block_planes_list.append(single_block_planes)
            return block_planes_list

                                          
        elif isinstance(block_idx, (list, tuple)) and len(block_idx) > 0 and isinstance(block_idx[0], (list, tuple)):
                                    
            block_planes_list = []
            for single_block_idx in block_idx:
                               
                single_block_planes = self.get_block_planes(single_block_idx, needed=needed)
                block_planes_list.append(single_block_planes)
            return block_planes_list

                    
                                           
        if block_idx in self.active_blocks:
                    
            self.block_usage_time[block_idx] = time.time()
            try:
                        
                needed_local = needed if needed in ('all', 'geo', 'color') else 'all'
                def _is_needed_name(nm: str) -> bool:
                    if needed_local == 'all':
                        return True
                    if needed_local == 'geo':
                        return nm.startswith('geo_feat_')
                    if needed_local == 'color':
                        return nm.startswith('app_feat_')
                    return True

                                                 
                missing = []
                for name, shape in self.block_plane_shapes.items():
                    if not _is_needed_name(name):
                        continue
                    if name not in self.active_blocks[block_idx]:
                        self.active_blocks[block_idx][name] = torch.nn.Parameter(
                            torch.randn(*shape, device=self.device) * 0.1
                        )
                        missing.append(name)
                if self.verbose and missing:
                    print(f"is an activated block{block_idx}complete{len(missing)}missing planes:{', '.join(missing)} (needed={needed_local})")
            except Exception as e:
                if self.verbose:
                    print(f"[Debug] Error when filling missing planes with activated blocks:{e}")
            return self.active_blocks[block_idx]

                           
        if block_idx in self.cached_blocks:
                      
            self.cached_usage_time[block_idx] = time.time()
            success = self._load_block_to_gpu(block_idx)
            if success:
                return self.active_blocks[block_idx]

                               
        self._create_block(block_idx, needed=needed)
                                                                                      
              
                                                                                     
                                                                                                                 
                                                     
                          
                                                                                                                    
                                                             
                                                                                                                         
                                                                              
                                
                                                                                                                        

                   
        if block_idx not in self.active_blocks:
            raise KeyError(f"Error: Unable to create block{block_idx}, please check the pre-split chunk path and permissions")

                
        self.block_usage_time[block_idx] = time.time()

                       
        if self.enable_block_consistency:
                   
            adjacent_blocks = self._get_adjacent_blocks(block_idx)
            for adj_block_idx in adjacent_blocks:
                if adj_block_idx in self.active_blocks:
                    self.ensure_block_consistency(block_idx, adj_block_idx)

        return self.active_blocks[block_idx]

    def normalize_coordinate_by_block(self, p, block_idx):
           
                                                          
        if isinstance(block_idx, torch.Tensor) and block_idx.dim() == 2 and block_idx.shape[1] == 3:
                                              
            if p.shape[0] == 1 and block_idx.shape[0] > 1:
                                         
                                      
                single_block_idx = tuple(block_idx[0].cpu().numpy().astype(int))
                return self.normalize_coordinate_by_block(p, single_block_idx)

                                          
                                       
            if p.shape[0] != block_idx.shape[0]:
                raise ValueError(f"Number of points ({p.shape[0]}) and the number of block indexes ({block_idx.shape[0]}) does not match")

            normalized_points = torch.zeros_like(p)
            for i in range(block_idx.shape[0]):
                single_block_idx = tuple(block_idx[i].cpu().numpy().astype(int))
                single_point = p[i:i + 1]               
                normalized_points[i:i + 1] = self.normalize_coordinate_by_block(single_point, single_block_idx)
            return normalized_points

                                    
        elif isinstance(block_idx, (list, tuple)) and len(block_idx) > 0 and isinstance(block_idx[0], (list, tuple)):
                                 
            if p.shape[0] != len(block_idx):
                raise ValueError(f"Number of points ({p.shape[0]}) and the number of block indexes ({len(block_idx)}) does not match")

            normalized_points = torch.zeros_like(p)
            for i, single_block_idx in enumerate(block_idx):
                single_point = p[i:i + 1]               
                normalized_points[i:i + 1] = self.normalize_coordinate_by_block(single_point, single_block_idx)
            return normalized_points

                    
                  
        block_bound = self.get_block_bound(block_idx)

                                          
        from src.common import normalize_3d_coordinate
        return normalize_3d_coordinate(p, block_bound)

    def _load_block_to_gpu(self, block_idx):
                            
        if block_idx not in self.cached_blocks:
            print(f"Error: Attempt to load uncached chunk{block_idx}")
            return False

                     
        self.memory_check_counter += 1
        if self.memory_check_counter >= self.memory_check_interval:
            self.memory_check_counter = 0
            self._adjust_active_blocks_limit()

                              
        if len(self.active_blocks) >= self.max_active_blocks:
            if not self._unload_least_used_block():
                print(f"Warning: Unable to unload blocks to make space")
                return False

                      
        cpu_planes = self.cached_blocks[block_idx]
        gpu_planes = {}

        for name, tensor in cpu_planes.items():
                            
            gpu_planes[name] = torch.nn.Parameter(tensor.to(self.device))

              
        self.active_blocks[block_idx] = gpu_planes
        self.block_usage_time[block_idx] = time.time()

                                
        try:
            del self.cached_blocks[block_idx]
            if block_idx in self.cached_usage_time:
                del self.cached_usage_time[block_idx]
        except Exception:
            pass

        return True

    def get_active_blocks_info(self):
                        
        return {
            'active_blocks': list(self.active_blocks.keys()),
            'cached_blocks': list(self.cached_blocks.keys()),
            'total_blocks': self.total_blocks,
            'max_active_blocks': self.max_active_blocks
        }

    def get_blocks_for_ray(self, ray_o, ray_d, max_dist=10.0):
           
                       
        ray_end = ray_o + ray_d * max_dist
        start_blocks = self.get_block_index(ray_o)
        end_blocks = self.get_block_index(ray_end)

                 
        if isinstance(start_blocks, tuple) and isinstance(end_blocks, tuple):
            if start_blocks == end_blocks:
                return [start_blocks]
            else:
                return [start_blocks, end_blocks]

                 
        blocks_set = set()
        for sb, eb in zip(start_blocks, end_blocks):
            blocks_set.add(sb)
            if sb != eb:
                blocks_set.add(eb)

        return list(blocks_set)

    def get_planes_by_point(self, p):
           
        block_indices = self.get_block_index(p)

                
        if isinstance(block_indices, tuple):
            planes = self.get_block_planes(block_indices)
            return planes, block_indices

                
        planes_list = []
        for idx in block_indices:
            planes_list.append(self.get_block_planes(idx))

        return planes_list, block_indices

    def _adjust_active_blocks_limit(self):
                                    
        if not self.enable_dynamic_memory:
            return

                       
        try:
            import torch
            if hasattr(torch.cuda, 'memory_reserved'):
                                
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                reserved_memory = torch.cuda.memory_reserved(self.device)
                memory_usage = reserved_memory / total_memory
            else:
                               
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                allocated_memory = torch.cuda.memory_allocated(self.device)
                memory_usage = allocated_memory / total_memory

                                        
            if memory_usage > 0.85:              
                         
                new_limit = max(5, self.max_active_blocks - 5)
                if new_limit < self.max_active_blocks:
                    print(
                        f"BlockManager: GPU memory usage{memory_usage:.1%}, reduce the maximum number of active blocks:{self.max_active_blocks} -> {new_limit}")
                    self.max_active_blocks = new_limit

                                  
                    while len(self.active_blocks) > self.max_active_blocks and self._unload_least_used_block():
                        pass
            elif memory_usage < 0.6:              
                         
                new_limit = self.max_active_blocks + 2
                print(
                    f"BlockManager: GPU memory usage{memory_usage:.1%}, increase the maximum number of active blocks:{self.max_active_blocks} -> {new_limit}")
                self.max_active_blocks = new_limit

        except Exception as e:
            print(f"BlockManager: Error while adjusting active block limit:{e}")

    def _are_blocks_adjacent(self, block_i_idx, block_j_idx):
                       
        diff_x = abs(block_i_idx[0] - block_j_idx[0])
        diff_y = abs(block_i_idx[1] - block_j_idx[1])
        diff_z = abs(block_i_idx[2] - block_j_idx[2])

                                   
        return (diff_x <= 1 and diff_y <= 1 and diff_z <= 1) and (diff_x + diff_y + diff_z <= 3)

    def _compute_overlap(self, block_i_idx, block_j_idx):
                          
                  
        bound_i = self.get_block_bound(block_i_idx)
        bound_j = self.get_block_bound(block_j_idx)

                
        overlap_min = torch.max(bound_i[0], bound_j[0])
        overlap_max = torch.min(bound_i[1], bound_j[1])

                  
        if torch.all(overlap_min < overlap_max):
            return overlap_min, overlap_max
        else:
            return None, None

    def ensure_block_consistency(self, block_i_idx, block_j_idx):
           
        if not self._are_blocks_adjacent(block_i_idx, block_j_idx):
            return

                    
        block_i_planes = self.active_blocks[block_i_idx]
        block_j_planes = self.active_blocks[block_j_idx]

                    
        overlap_region = self._calculate_overlap_region(block_i_idx, block_j_idx)
        if overlap_region is None:
            return

        n_samples = 2000              
        sample_points = self._sample_points_in_overlap(overlap_region, n_samples)

                           
        self._blend_features_at_boundary(block_i_planes, block_j_planes, sample_points, overlap_region)

        print(f"BlockManager: Block maintained{block_i_idx}and{block_j_idx}feature consistency between")

    def _calculate_overlap_region(self, block_i_idx, block_j_idx):
           
                    
        i_min = torch.tensor([
            block_i_idx[0] * self.block_size_xy,
            block_i_idx[1] * self.block_size_xy,
            block_i_idx[2] * self.block_size_z
        ], device=self.device)
        i_max = i_min + self.block_size_xy

        j_min = torch.tensor([
            block_j_idx[0] * self.block_size_xy,
            block_j_idx[1] * self.block_size_xy,
            block_j_idx[2] * self.block_size_z
        ], device=self.device)
        j_max = j_min + self.block_size_xy

                
        overlap_min = torch.max(i_min, j_min)
        overlap_max = torch.min(i_max, j_max)

                   
        if torch.all(overlap_max > overlap_min):
            return (overlap_min, overlap_max)
        return None

    def _sample_points_in_overlap(self, overlap_region, n_samples):
                        
        overlap_min, overlap_max = overlap_region

                    
        points = []
        for _ in range(n_samples):
                    
            rand = torch.rand(3, device=self.device)
                          
            point = overlap_min + rand * (overlap_max - overlap_min)
            points.append(point)

        return torch.stack(points)

    def _blend_features_at_boundary(self, block_i_planes, block_j_planes, sample_points, overlap_region):
           
                        
                          
                          
                      
                         

                                   
                          

                                
        pass

    def _unload_least_used_block(self):
                              
        if not self.active_blocks:
            return False

                  
        lru_block_idx = min(self.block_usage_time, key=self.block_usage_time.get)

                       
        block_to_unload = self.active_blocks[lru_block_idx]

                   
        cpu_planes = {}
        for name, tensor_param in block_to_unload.items():
                           
            cpu_planes[name] = tensor_param.data.cpu().clone()

                      
        self.cached_blocks[lru_block_idx] = cpu_planes
        del self.active_blocks[lru_block_idx]
        del self.block_usage_time[lru_block_idx]

        if self.verbose:
            print(f"BlockManager: Unload blocks{lru_block_idx}to CPU cache to free up GPU memory")

                          
        self.cached_usage_time[lru_block_idx] = time.time()
        self._evict_cache_until_under_limit()
        return True

    def _evict_cache_until_under_limit(self):
                                              
        try:
            while self.max_cached_blocks > 0 and len(self.cached_blocks) > self.max_cached_blocks:
                             
                evict_idx = min(self.cached_usage_time, key=self.cached_usage_time.get)
                try:
                    del self.cached_blocks[evict_idx]
                except Exception:
                    pass
                                
                if evict_idx in self.cached_usage_time:
                    del self.cached_usage_time[evict_idx]
                if evict_idx in self.transposed_planes:
                    del self.transposed_planes[evict_idx]
                if self.verbose:
                    print(f"BlockManager: evict blocks from CPU cache{evict_idx} (LRU)")
        except Exception as e:
            if self.verbose:
                print(f"BlockManager: Error evicting cache:{e}")

    def _get_adjacent_blocks(self, block_idx):
           
        x_idx, y_idx, z_idx = block_idx
        adjacent_blocks = []

                     
        adjacent_positions = [
            (x_idx + 1, y_idx, z_idx),
            (x_idx - 1, y_idx, z_idx),
            (x_idx, y_idx + 1, z_idx),
            (x_idx, y_idx - 1, z_idx),
            (x_idx, y_idx, z_idx + 1),
            (x_idx, y_idx, z_idx - 1)
        ]

                     
        for pos in adjacent_positions:
            if self._is_valid_block_idx(pos):
                adjacent_blocks.append(pos)

        return adjacent_blocks

    def _is_valid_block_idx(self, block_idx):
                               
        try:
                            
            if any(idx < -1000 or idx > 1000 for idx in block_idx):
                print(f"Warning: block index{block_idx}beyond reasonable range")
                return False

            x_idx, y_idx, z_idx = block_idx

                       
            if x_idx < 0 or y_idx < 0 or z_idx < 0:
                return False

                                               
            if hasattr(self, 'block_grid_size') and isinstance(self.block_grid_size, torch.Tensor):
                if (x_idx >= self.block_grid_size[0] or
                        y_idx >= self.block_grid_size[1] or
                        z_idx >= self.block_grid_size[2]):
                    return False

                                    
            block_min = torch.tensor([
                x_idx * self.block_size_xy,
                y_idx * self.block_size_xy,
                z_idx * self.block_size_z
            ], device=self.device) + self.global_bound[0]            

            block_max = torch.tensor([
                (x_idx + 1) * self.block_size_xy,
                (y_idx + 1) * self.block_size_xy,
                (z_idx + 1) * self.block_size_z
            ], device=self.device) + self.global_bound[0]            

                           
            overlap_min = torch.max(block_min, self.global_bound[0])
            overlap_max = torch.min(block_max, self.global_bound[1])

                                
            is_valid = torch.all(overlap_max > overlap_min).item()
            return is_valid

        except Exception as e:
            print(f"Warning: Error checking block index validity:{e}")
            return False

    def get_trainable_parameters(self):
           
        geo_planes = []
        color_planes = []

                 
        for block_idx, block_planes in self.active_blocks.items():
                        
            for name, param in block_planes.items():
                if 'geo_feat' in name:
                    geo_planes.append(param)
                elif 'app_feat' in name:
                    color_planes.append(param)

        return {
            'geo_planes': geo_planes,
            'color_planes': color_planes
        }

    def prepare_blocks_for_camera(self, camera_position, view_distance=16.0):
           
        if not isinstance(camera_position, torch.Tensor):
            camera_position = torch.tensor(camera_position, device=self.device, dtype=torch.float32)

                  
        try:
            camera_block_idx = self.get_block_index(camera_position)

                       
            if any(idx < -1000000 or idx > 1000000 for idx in camera_block_idx):
                print(f"Warning: Calculated camera block index{camera_block_idx}If there is an outlier, (0,0,0) will be used as an alternative")
                camera_block_idx = (0, 0, 0)

                           
            self.get_block_planes(camera_block_idx)
        except Exception as e:
            print(f"Warning: Error getting camera block:{e}, will use (0,0,0) as an alternative")
            camera_block_idx = (0, 0, 0)
            self.get_block_planes(camera_block_idx)

                        
                            
        block_radius = min(10, int(np.ceil(view_distance / self.block_size_xy)) + 1)                    

        try:
                            
            if not hasattr(self, 'block_grid_size') or not isinstance(self.block_grid_size, torch.Tensor):
                print("Warning: block_grid_size does not exist or is invalid, a safe value will be used")
                safe_grid_size = torch.tensor([10, 10, 10], device=self.device)         
            else:
                safe_grid_size = self.block_grid_size.clone()

                         
            x_min = max(0, min(int(safe_grid_size[0] - 1), camera_block_idx[0] - block_radius))
            x_max = max(0, min(int(safe_grid_size[0] - 1), camera_block_idx[0] + block_radius))
            y_min = max(0, min(int(safe_grid_size[1] - 1), camera_block_idx[1] - block_radius))
            y_max = max(0, min(int(safe_grid_size[1] - 1), camera_block_idx[1] + block_radius))
            z_min = max(0, min(int(safe_grid_size[2] - 1), camera_block_idx[2] - block_radius))
            z_max = max(0, min(int(safe_grid_size[2] - 1), camera_block_idx[2] + block_radius))

                                      
            max_blocks_to_process = int(self.max_prefetch_blocks)
            blocks_processed = 0

                       
            for x in range(x_min, x_max + 1):
                if blocks_processed >= max_blocks_to_process:
                    break

                for y in range(y_min, y_max + 1):
                    if blocks_processed >= max_blocks_to_process:
                        break

                    for z in range(z_min, z_max + 1):
                        if blocks_processed >= max_blocks_to_process:
                            break

                        block_idx = (x, y, z)
                        try:
                            if self._is_valid_block_idx(block_idx):
                                             
                                block_bound = self.get_block_bound(block_idx)
                                block_center = (block_bound[0] + block_bound[1]) / 2
                                dist = torch.norm(block_center - camera_position)
                                if dist <= view_distance:
                                    self.get_block_planes(block_idx)
                                    self.cached_usage_time[block_idx] = time.time()
                                    blocks_processed += 1
                        except Exception as e:
                            if self.verbose:
                                print(f"preload block{block_idx}An error occurred:{e}")
                            continue

        except Exception as e:
            if self.verbose:
                print(f"Warning: Error preloading blocks within camera field of view:{e}")

                                   
        if self.enable_dynamic_memory:
            self.memory_check_counter += 1
            if self.memory_check_counter >= self.memory_check_interval:
                self._adjust_active_blocks_limit()
                self.memory_check_counter = 0
