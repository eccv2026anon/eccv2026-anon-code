#!/usr/bin/env python3
                     
 
                                    
                                                      
                                     
 
                

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw
import math

                            
       
                            
                                     
SCENE_MIN = [37.9265, -3.5017, -0.2400]                         
SCENE_MAX = [293.2865, 262.8983, 3.1200]                         
SCENE_BOUND = np.array([SCENE_MIN, SCENE_MAX])

                                   
GEO_VOXEL_SIZE_COARSE = 0.24
GEO_VOXEL_SIZE_FINE = 0.06
COLOR_VOXEL_SIZE_COARSE = 0.24
COLOR_VOXEL_SIZE_FINE = 0.03

       
BLOCK_SIZE_XY = 7.68                  

                    
                                    
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.utils.BlockManager import BlockManager

                    
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False          
except (KeyError, ValueError) as e:
    print(f"Warning: Unable to configure Chinese font support ({type(e).__name__}: {e}), will use the English label instead")
except Exception as e:
    print(f"Warning: Unexpected exception occurred while configuring fonts{type(e).__name__}: {e}, will use the English label instead")


def _resample_global_plane_for_block(global_plane, block_plane_shape, block_min_bound, block_max_bound, global_bound,
                                     debug=False):
       
    device = global_plane.device
    _, c_dim, H_b, W_b = block_plane_shape

              
    if debug:
        print(f"Global plane shape:{global_plane.shape}")
        print(f"Block plan shape:{block_plane_shape}")
        print(f"Block minimum bounds:{block_min_bound}")
        print(f"Block maximum boundary:{block_max_bound}")
        print(f"Global boundaries:{global_bound}")

              
    grid_y = torch.linspace(block_min_bound[1], block_max_bound[1], H_b, device=device)
    grid_x = torch.linspace(block_min_bound[0], block_max_bound[0], W_b, device=device)

                                                
    mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing='xy')

                       
    dx = global_bound[1, 0] - global_bound[0, 0]
    dy = global_bound[1, 1] - global_bound[0, 1]

    if dx < 1e-5 or dy < 1e-5:
        print(f"Warning: Global bounds too small: dx={dx}, dy={dy}, use the default range")
        dx = max(dx, 1.0)
        dy = max(dy, 1.0)

                                
    norm_x = 2 * (mesh_x - global_bound[0, 0]) / dx - 1
    norm_y = 2 * (mesh_y - global_bound[0, 1]) / dy - 1

                                            
                                          
    grid = torch.stack((norm_x, norm_y), dim=-1)

                      
                                                                         
                                    
    grid = grid.permute(0, 2, 1, 3) if grid.shape[1] == W_b and grid.shape[2] == H_b else grid

            
    grid = grid.unsqueeze(0)

    if debug:
        print(f"Sampling grid shape:{grid.shape}")
        expected_grid_shape = (1, H_b, W_b, 2)
        if grid.shape != expected_grid_shape:
            print(f"Warning: Sampling grid shape does not match expected: current={grid.shape}, expectation ={expected_grid_shape}")
            print(f"This can cause dimension order issues")

                          
    try:
                             
        if global_plane.dim() == 3:
            global_plane_input = global_plane.unsqueeze(0)          
        else:
            global_plane_input = global_plane

        resampled_plane = F.grid_sample(
            global_plane_input.float(),
            grid.float(),
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )

                  
        if torch.isnan(resampled_plane).any() or torch.isinf(resampled_plane).any():
            print(f"Warning: Resampled results contain NaN or Inf values and will be replaced with zeros")
            resampled_plane = torch.nan_to_num(resampled_plane, nan=0.0, posinf=0.0, neginf=0.0)

                     
        resampled_plane = resampled_plane.squeeze(0)

                    
        expected_shape = (c_dim, H_b, W_b)
        if resampled_plane.shape != expected_shape:
            if debug:
                print(f"The dimensions of the resampling results do not match and need to be adjusted.")
                print(f"Current shape:{resampled_plane.shape}, desired shape:{expected_shape}")

                       
            if resampled_plane.shape[0] == c_dim:
                                    
                if resampled_plane.shape[1:] == (W_b, H_b):
                    resampled_plane = resampled_plane.permute(0, 2, 1)
                    if debug:
                        print(f"Transposed dimensions, new shape:{resampled_plane.shape}")
                else:
                               
                    try:
                        if resampled_plane.numel() == c_dim * H_b * W_b:
                            resampled_plane = resampled_plane.reshape(c_dim, H_b, W_b)
                            if debug:
                                print(f"Reshaped, new shape:{resampled_plane.shape}")
                    except Exception as e:
                        print(f"Reshape failed:{e}")
            else:
                print(f"Channel number mismatch: current={resampled_plane.shape[0]}, expectation ={c_dim}")

        if debug:
            print(f"The final resampling result shape is:{resampled_plane.shape}")
            print(f"Resampling result mean:{resampled_plane.mean()}")
            print(f"Resampling result standard deviation:{resampled_plane.std()}")

        return resampled_plane
    except Exception as e:
        print(f"Resampling failed:{e}")
        print(f"Returns a randomly initialized feature plane")
        return torch.randn(c_dim, H_b, W_b, device=device) * 0.1


def visualize_feature_blocks(block_data_list, block_indices, output_dir, feature_key, normalize=True, grid_line_width=2,
                             grid_line_color=(255, 0, 0), block_index_fontsize=8):
       
    if not block_data_list:
        print(f"No block data available for visualization")
        return

                 
    valid_blocks = []
    valid_indices = []
    for i, block_data in enumerate(block_data_list):
        if feature_key in block_data:
            valid_blocks.append(block_data)
            valid_indices.append(block_indices[i])

    if not valid_blocks:
        print(f"No feature planes included '{feature_key}' block")
        return

                        
    sample_block = valid_blocks[0][feature_key]
    if sample_block.dim() == 3:                
        c_dim, block_height, block_width = sample_block.shape

                                  
        x_indices = [idx[0] for idx in valid_indices]
        y_indices = [idx[1] for idx in valid_indices]
        z_indices = [idx[2] for idx in valid_indices]

        x_min, x_max = min(x_indices), max(x_indices)
        y_min, y_max = min(y_indices), max(y_indices)
        z_min, z_max = min(z_indices), max(z_indices)

                        
        plane_type = feature_key.split('_')[2] if len(feature_key.split('_')) > 2 else "unknown"

                       
        if plane_type == 'xy':
                                  
            grid_rows = y_max - y_min + 1
            grid_cols = x_max - x_min + 1
                                  
            position_map = {(idx[0], idx[1]): (idx[1] - y_min, idx[0] - x_min) for idx in valid_indices}
        elif plane_type == 'xz':
                                  
            grid_rows = z_max - z_min + 1
            grid_cols = x_max - x_min + 1
            position_map = {(idx[0], idx[2]): (idx[2] - z_min, idx[0] - x_min) for idx in valid_indices}
        elif plane_type == 'yz':
                                  
            grid_rows = z_max - z_min + 1
            grid_cols = y_max - y_min + 1
            position_map = {(idx[1], idx[2]): (idx[2] - z_min, idx[1] - y_min) for idx in valid_indices}
        else:
                             
            print(f"Unknown plane type:{plane_type}, using the default grid layout")
            grid_rows = grid_cols = math.ceil(math.sqrt(len(valid_blocks)))
            position_map = {i: (i // grid_cols, i % grid_cols) for i in range(len(valid_blocks))}

        print(f"Visualization{feature_key}: grid size ={grid_rows}x{grid_cols}")

                    
        avg_feature_maps = []
        for block_data in valid_blocks:
            feature_tensor = block_data[feature_key]
                        
            avg_feature = torch.mean(feature_tensor, dim=0).cpu().numpy()
            avg_feature_maps.append(avg_feature)

                                          
        if plane_type == 'xy':
            pos_key = lambda idx: (idx[0], idx[1])
        elif plane_type == 'xz':
            pos_key = lambda idx: (idx[0], idx[2])
        elif plane_type == 'yz':
            pos_key = lambda idx: (idx[1], idx[2])
        else:
                    
            pos_key = lambda idx: len(valid_blocks)                     

                         
        buffer_size = 1                 
        effective_block_height = block_height + 2 * buffer_size
        effective_block_width = block_width + 2 * buffer_size

        full_image_height = grid_rows * effective_block_height + (grid_rows - 1) * grid_line_width
        full_image_width = grid_cols * effective_block_width + (grid_cols - 1) * grid_line_width

                            
        full_image = np.ones((full_image_height, full_image_width)) * np.nan                

                      
        global_min = float('inf')
        global_max = float('-inf')

        if normalize:
            for avg_feature in avg_feature_maps:
                global_min = min(global_min, np.nanmin(avg_feature))
                global_max = max(global_max, np.nanmax(avg_feature))

                    
        for i, (block_data, block_idx, avg_feature) in enumerate(zip(valid_blocks, valid_indices, avg_feature_maps)):
            try:
                          
                key = pos_key(block_idx)
                if key in position_map:
                    grid_row, grid_col = position_map[key]
                else:
                                       
                    print(f"Warning: blocks{block_idx}No location mapping, use default location")
                    grid_row = i // grid_cols
                    grid_col = i % grid_cols

                             
                row_start = grid_row * (effective_block_height + grid_line_width)
                col_start = grid_col * (effective_block_width + grid_line_width)

                                 
                full_image[row_start + buffer_size:row_start + buffer_size + block_height,
                col_start + buffer_size:col_start + buffer_size + block_width] = avg_feature
            except Exception as e:
                print(f"visualization block{block_idx}An error occurred:{e}")

                           
        plt.figure(figsize=(12, 10))

                                  
        if normalize and global_max > global_min:
            vmin, vmax = global_min, global_max
        else:
            vmin, vmax = None, None

        img = plt.imshow(full_image, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)

                                
        cbar = plt.colorbar(img, label='Feature Value', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)             

                         
        plt.title(f'Feature Plane: {feature_key} (Channel Average)')

                    
        for i, (block_data, block_idx) in enumerate(zip(valid_blocks, valid_indices)):
            try:
                key = pos_key(block_idx)
                if key in position_map:
                    grid_row, grid_col = position_map[key]
                                       
                    row_top = grid_row * (effective_block_height + grid_line_width) + buffer_size
                    col_center = grid_col * (effective_block_width + grid_line_width) + buffer_size + block_width // 2

                                        
                    plt.text(col_center, row_top, f"{block_idx[0]},{block_idx[1]},{block_idx[2]}",
                             color='yellow', fontsize=block_index_fontsize, ha='center', va='top')
            except Exception as e:
                print(f"Add block index{block_idx}An error occurred:{e}")

               
        plt.axis('off')

                 
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{feature_key}_avg.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=500)
        plt.close()

        print(f"Feature visualization image saved to:{output_path}")
    else:
        print(f"Unsupported feature plane shapes:{sample_block.shape}")


def main():
    parser = argparse.ArgumentParser(description='StructRecon: Pre-segmented global feature plane script')
    parser.add_argument('config', type=str, help='Configuration file path used for SLAM runs.')
    parser.add_argument('--input_plane_file', type=str,
                        help='(Optional) Enter the global pre-optimized flat file path. If not provided, will be read from the configuration file.')
    parser.add_argument('--output_dir', type=str,
                        help='(Optional) The output directory of the split block files. If not provided, "output/SCENE_NAME/preoptimized_blocks" will be used.')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--visualize', action='store_true', help='Enable feature block visualization, generate PNG image')
    args = parser.parse_args()

              
    print("\\n====== Hyperparameters used =====")
    print(f"Minimum point of scene boundary:{SCENE_MIN}")
    print(f"Maximum point of scene boundary:{SCENE_MAX}")
    print(f"Block size (XY):{BLOCK_SIZE_XY}")
    print(f"Geometric feature coarse level voxel size:{GEO_VOXEL_SIZE_COARSE}")
    print(f"Geometric feature fine-level voxel size:{GEO_VOXEL_SIZE_FINE}")
    print(f"Color feature coarse voxel size:{COLOR_VOXEL_SIZE_COARSE}")
    print(f"Color feature fine-level voxel size:{COLOR_VOXEL_SIZE_FINE}")
    print("=========================\n")

          
    default_config = Path(__file__).resolve().parents[1] / 'configs' / 'structrecon.yaml'
    cfg = config.load_config(args.config, str(default_config))
    print("Configuration loaded successfully.")

    device = cfg.get('device', 'cuda:0')
    debug = args.debug

                      
                                       
    MAX_VISUALIZATION_BLOCKS = -1                 
                
    BLOCK_INDEX_FONTSIZE = 4                

                          
    scene_output_dir = cfg.get('data', {}).get('output', 'output/default_scene')

    input_file = args.input_plane_file or cfg.get('preoptimized_planes_path',
                                                  os.path.join(scene_output_dir, "preoptimized_geometry_planes.pth"))

                                 
    preoptimized_blocks_dir = args.output_dir or os.path.join(scene_output_dir, "preoptimized_blocks")
    cfg['preoptimized_blocks_dir'] = preoptimized_blocks_dir                     

    os.makedirs(preoptimized_blocks_dir, exist_ok=True)

               
    visualization_dir = os.path.join(preoptimized_blocks_dir, "visualization")

    print(f":{input_file}")
    print(f"Output pre-split chunk directory:{preoptimized_blocks_dir}")

    if args.visualize:
        print(f"Visualization output directory:{visualization_dir}")

    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist:{input_file}")
        return

                                
                                                               
    bound_3x2 = np.array([[SCENE_MIN[0], SCENE_MAX[0]],
                          [SCENE_MIN[1], SCENE_MAX[1]],
                          [SCENE_MIN[2], SCENE_MAX[2]]])

               
    if 'mapping' not in cfg:
        cfg['mapping'] = {}
    cfg['mapping']['bound'] = bound_3x2.tolist()
    print(f"Fixed boundaries set:{bound_3x2}")

                   
    if 'planes_res' not in cfg:
        cfg['planes_res'] = {}
    cfg['planes_res']['coarse'] = GEO_VOXEL_SIZE_COARSE
    cfg['planes_res']['fine'] = GEO_VOXEL_SIZE_FINE

    if 'c_planes_res' not in cfg:
        cfg['c_planes_res'] = {}
    cfg['c_planes_res']['coarse'] = COLOR_VOXEL_SIZE_COARSE
    cfg['c_planes_res']['fine'] = COLOR_VOXEL_SIZE_FINE

           
    if 'model' not in cfg:
        cfg['model'] = {}
    cfg['model']['block_size'] = BLOCK_SIZE_XY

                  
    c_dim = cfg['model'].get('c_dim', 32)
    c_dim_fine = cfg['model'].get('c_dim_fine', 32)              
    c_dim_app = cfg['model'].get('c_dim_app', 32)

                  
    adv_cfg = cfg.get('model', {}).get('advanced_features', cfg.get('advanced_features', {}))
    use_color_multires = adv_cfg.get('color_multires', False)

            
    planes_config = {
        'geo_coarse': {'dim': c_dim, 'planes': ['xy', 'xz', 'yz']},
        'geo_fine': {'dim': c_dim_fine, 'planes': ['xy', 'xz', 'yz']},
    }

                          
    if use_color_multires:
        planes_config.update({
            'app_coarse': {'dim': c_dim_app, 'planes': ['xy', 'xz', 'yz']},
            'app_fine': {'dim': c_dim_app, 'planes': ['xy', 'xz', 'yz']}
        })
        print("Enable the color multi-resolution feature, which will generate a thick and thin two-level color plane.")
    else:
        planes_config.update({
            'app': {'dim': c_dim_app, 'planes': ['xy', 'xz', 'yz']}
        })
        print("Use a single-level color feature plane")

                         
    print("Loading global feature plane...")
    try:
        global_planes_dict = torch.load(input_file, map_location=device)

                  
        if not global_planes_dict:
            print("Error: Global flat file is empty!")
            return

        print(f"The global flat file contains the following keys:{list(global_planes_dict.keys())}")

                                
                                           
        special_keys = ['structrecon_bound_at_save_time', 'eslam_bound_at_save_time']

        for key, tensor in global_planes_dict.items():
            if key in special_keys:
                print(f"  - {key}:type={type(tensor)}")
                           
                if isinstance(tensor, np.ndarray):
                                          
                    global_planes_dict[key] = torch.tensor(tensor, device=device)
                    print(f"Converted numpy array to tensor")
            else:
                        
                if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype') and hasattr(tensor, 'device'):
                    print(f"  - {key}: shape={tensor.shape}, type={tensor.dtype}, equipment ={tensor.device}")
                else:
                    print(f"  - {key}:type={type(tensor)}, Unable to display shape/type/device information")

                                    
                    
        plane_keys = [k for k in global_planes_dict.keys() if 'planes' in k.lower()]

                       
        key_mapping = {}

        if plane_keys:
            print(f"The detected feature plane key name format is:{plane_keys[0]}")

                                                              
            if any('planes_xy' in k for k in plane_keys):
                                              
                for plane_type, plane_conf in planes_config.items():
                    feature_type, detail_level = plane_type.split('_') if '_' in plane_type else (plane_type, '')
                    for plane_dir in plane_conf['planes']:
                        src_key = f"{feature_type}_planes_{plane_dir}_{detail_level}" if detail_level else f"{feature_type}_planes_{plane_dir}"
                        if feature_type == 'geo':
                            dst_key = f"geo_feat_{plane_dir}_{'coarse' if detail_level == 'coarse' else 'fine'}"
                        else:       
                            dst_key = f"app_feat_{plane_dir}{'_' + detail_level if detail_level else ''}"

                        key_mapping[src_key] = dst_key
            else:
                                       
                for plane_type, plane_conf in planes_config.items():
                    for plane_dir in plane_conf['planes']:
                        src_key = f"{plane_type}_{plane_dir}"
                        if plane_type.startswith('geo'):
                            dst_key = f"{plane_type.replace('_', '_feat_')}_{plane_dir}"
                        elif plane_type.startswith('app'):
                            if '_' in plane_type:                 
                                suffix = plane_type.split('_')[1]
                                dst_key = f"app_feat_{plane_dir}_{suffix}"
                            else:          
                                dst_key = f"app_feat_{plane_dir}"
                        key_mapping[src_key] = dst_key

                
        print("\\nUse the following key mapping:")
        for src_key, dst_key in key_mapping.items():
            if src_key in global_planes_dict:
                print(f"  {src_key} -> {dst_key}[exist]")
            else:
                print(f"  {src_key} -> {dst_key}[Does not exist!]")

                                
        bound_key = None
        if 'structrecon_bound_at_save_time' in global_planes_dict:
            bound_key = 'structrecon_bound_at_save_time'
        elif 'eslam_bound_at_save_time' in global_planes_dict:
            bound_key = 'eslam_bound_at_save_time'

        if bound_key is not None:
            if bound_key == 'eslam_bound_at_save_time':
                print("Found legacy eslam_bound_at_save_time key, compatible with use as global boundary")
            else:
                print("Found the structrecon_bound_at_save_time key, which will be used as the global boundary")

            structrecon_bound = global_planes_dict[bound_key]
            if isinstance(structrecon_bound, torch.Tensor):
                structrecon_bound = structrecon_bound.cpu().numpy()
            elif isinstance(structrecon_bound, np.ndarray):
                pass              
            else:
                print(f"Warning: structrecon_bound_at_save_time type unknown:{type(structrecon_bound)}")
                structrecon_bound = None

                           
            if structrecon_bound is not None:
                print(f"structrecon_bound shape:{structrecon_bound.shape}, content:{structrecon_bound}")
                                
                if structrecon_bound.shape == (2, 3):
                                      
                    cfg['mapping'] = cfg.get('mapping', {})
                    cfg['mapping']['bound'] = structrecon_bound.tolist()
                    print(f"cfg['mapping']['bound'] has been updated to:{cfg['mapping']['bound']}")

    except Exception as e:
        print(f"Error: Failed to load global flat file:{e}")
        import traceback
        traceback.print_exc()
        return

    print("Loading completed.")

                                       
    try:
        print("Initialize BlockManager to get chunking information...")
        block_manager = BlockManager(cfg, device=device)

                       
        global_bound_min = block_manager.world_min.cpu().numpy()
        global_bound_max = block_manager.world_max.cpu().numpy()

        print(f"Global boundary minimum:{global_bound_min}")
        print(f"Global boundary maximum:{global_bound_max}")

                               
        geo_n_voxels_coarse = block_manager.geo_n_voxels_coarse
        geo_n_voxels_fine = block_manager.geo_n_voxels_fine
        color_n_voxels_coarse = block_manager.color_n_voxels_coarse
        color_n_voxels_fine = block_manager.color_n_voxels_fine

                  
        print("\\nVoxel number information:")
        print(
            f"Geometry feature coarse level - xy:{geo_n_voxels_coarse['xy']}, xz: {geo_n_voxels_coarse['xz']}, yz: {geo_n_voxels_coarse['yz']}")
        print(
            f"Geometric feature detail - xy:{geo_n_voxels_fine['xy']}, xz: {geo_n_voxels_fine['xz']}, yz: {geo_n_voxels_fine['yz']}")
        print(
            f"Color feature coarse level - xy:{color_n_voxels_coarse['xy']}, xz: {color_n_voxels_coarse['xz']}, yz: {color_n_voxels_coarse['yz']}")
        print(
            f"Color feature level - xy:{color_n_voxels_fine['xy']}, xz: {color_n_voxels_fine['xz']}, yz: {color_n_voxels_fine['yz']}")

                    
        if np.any(global_bound_max <= global_bound_min):
            print(f"Error: Invalid global bounds! Minimum value:{global_bound_min}, maximum value:{global_bound_max}")
            return

                                                                    
        all_indices = block_manager.get_all_possible_block_indices()
        print(f"will be{len(all_indices)}Generate pre-split files from each block.")

    except Exception as e:
        print(f"Error: Failed to initialize BlockManager:{e}")
        import traceback
        traceback.print_exc()
        return

                 
    successful_blocks = 0
    failed_blocks = 0
    empty_blocks = 0

                                        
    if not key_mapping:
        print("Error: Unable to determine feature plane keyname mapping, cannot continue")
        return

                  
    all_block_data = []
    all_block_indices = []
    collect_vis_data = args.visualize and (MAX_VISUALIZATION_BLOCKS != 0)

                                                                              
    for block_idx in tqdm(all_indices, desc="Pre-splitting feature planes"):
        block_data = {}
        block_bounds = block_manager.get_block_bound(block_idx)

        if debug:
            print(f"\nProcessing block{block_idx}")
            print(f"Block boundaries:{block_bounds}")

                  
        for src_key, dst_key in key_mapping.items():
            if src_key not in global_planes_dict:
                if debug:
                    print(f"key '{src_key}' does not exist in the global flat dictionary")
                continue

            global_plane = global_planes_dict[src_key]

                            
                                                             
            if "planes_" in src_key:
                                         
                parts = src_key.split("_")
                plane_dir = [p for p in parts if p in ['xy', 'xz', 'yz']][0]
                feature_type = parts[0]             
                detail_level = parts[-1] if parts[-1] in ['coarse', 'fine'] else ''
            else:
                                  
                parts = src_key.split("_")
                plane_dir = parts[-1]
                feature_type = parts[0]
                detail_level = parts[1] if len(parts) > 2 and parts[1] in ['coarse', 'fine'] else ''

                          
            if feature_type == 'geo':
                dim = c_dim if detail_level == 'coarse' else c_dim_fine
            else:       
                dim = c_dim_app

                                                 
            if plane_dir == 'xy':
                dims = [0, 1]
                if feature_type == 'geo':
                    if detail_level == 'coarse':
                        block_shape = (1, dim, geo_n_voxels_coarse['xy'][1], geo_n_voxels_coarse['xy'][0])
                    else:        
                        block_shape = (1, dim, geo_n_voxels_fine['xy'][1], geo_n_voxels_fine['xy'][0])
                else:       
                    if detail_level == 'coarse':
                        block_shape = (1, dim, color_n_voxels_coarse['xy'][1], color_n_voxels_coarse['xy'][0])
                    elif detail_level == 'fine':
                        block_shape = (1, dim, color_n_voxels_fine['xy'][1], color_n_voxels_fine['xy'][0])
                    else:          
                        block_shape = (1, dim, color_n_voxels_coarse['xy'][1], color_n_voxels_coarse['xy'][0])
            elif plane_dir == 'xz':
                dims = [0, 2]
                if feature_type == 'geo':
                    if detail_level == 'coarse':
                        block_shape = (1, dim, geo_n_voxels_coarse['xz'][1], geo_n_voxels_coarse['xz'][0])
                    else:        
                        block_shape = (1, dim, geo_n_voxels_fine['xz'][1], geo_n_voxels_fine['xz'][0])
                else:       
                    if detail_level == 'coarse':
                        block_shape = (1, dim, color_n_voxels_coarse['xz'][1], color_n_voxels_coarse['xz'][0])
                    elif detail_level == 'fine':
                        block_shape = (1, dim, color_n_voxels_fine['xz'][1], color_n_voxels_fine['xz'][0])
                    else:          
                        block_shape = (1, dim, color_n_voxels_coarse['xz'][1], color_n_voxels_coarse['xz'][0])
            else:      
                dims = [1, 2]
                if feature_type == 'geo':
                    if detail_level == 'coarse':
                        block_shape = (1, dim, geo_n_voxels_coarse['yz'][1], geo_n_voxels_coarse['yz'][0])
                    else:        
                        block_shape = (1, dim, geo_n_voxels_fine['yz'][1], geo_n_voxels_fine['yz'][0])
                else:       
                    if detail_level == 'coarse':
                        block_shape = (1, dim, color_n_voxels_coarse['yz'][1], color_n_voxels_coarse['yz'][0])
                    elif detail_level == 'fine':
                        block_shape = (1, dim, color_n_voxels_fine['yz'][1], color_n_voxels_fine['yz'][0])
                    else:          
                        block_shape = (1, dim, color_n_voxels_coarse['yz'][1], color_n_voxels_coarse['yz'][0])

                          
            try:
                global_bound_tensor = torch.tensor(
                    [[global_bound_min[d] for d in dims], [global_bound_max[d] for d in dims]],
                    device=device
                )

                                        
                if global_plane.dim() == 4 and global_plane.shape[0] == 1:
                                        
                    input_plane = global_plane.squeeze(0)
                else:
                                 
                    input_plane = global_plane

                resampled_plane = _resample_global_plane_for_block(
                    input_plane,
                    block_shape,
                    block_bounds[0][dims],
                    block_bounds[1][dims],
                    global_bound_tensor,
                    debug=debug
                )

                               
                if resampled_plane.shape != tuple(block_shape[1:]):
                    if debug:
                        print(
                            f"Warning: resampled plane shape mismatch: expected={tuple(block_shape[1:])},actual={resampled_plane.shape}")

                                                 
                    if resampled_plane.dim() == 3 and resampled_plane.shape[0] == block_shape[1]:
                                            
                        try:
                                           
                            total_elements = resampled_plane.numel()
                            expected_elements = block_shape[1] * block_shape[2] * block_shape[3]

                            if total_elements == expected_elements:
                                             
                                resampled_plane = resampled_plane.reshape(block_shape[1:])
                                if debug:
                                    print(f"Successful reshaping into target shape:{resampled_plane.shape}")
                            else:
                                print(
                                    f"Error: Unable to reshape plane, total number of elements does not match: current={total_elements}, expectation ={expected_elements}")
                                             
                                resampled_plane = torch.randn(block_shape[1:], device=device) * 0.01
                        except Exception as e:
                            print(f"Reshape plane failed:{e}")
                                         
                            resampled_plane = torch.randn(block_shape[1:], device=device) * 0.01
                    else:
                        print(f"Error: Unable to adjust plane shape, shape difference is too large")
                                     
                        resampled_plane = torch.randn(block_shape[1:], device=device) * 0.01

                block_data[dst_key] = resampled_plane.cpu()

            except Exception as e:
                print(f"Error: resampling block{block_idx}plane{src_key}fails when:{e}")
                             
                block_data[dst_key] = torch.randn(block_shape[1:], device='cpu') * 0.01
                failed_blocks += 1

                   
        if not block_data:
            print(f"Warning: blocks{block_idx}There is no feature plane data")
            empty_blocks += 1
            continue

                       
        has_nan = False
        for key, tensor in block_data.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: blocks{block_idx}of{key}Contains NaN or Inf values")
                block_data[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                has_nan = True

                    
        if collect_vis_data and (MAX_VISUALIZATION_BLOCKS < 0 or len(all_block_data) < MAX_VISUALIZATION_BLOCKS):
            all_block_data.append(block_data)
            all_block_indices.append(block_idx)

                                                   
        block_filename = os.path.join(preoptimized_blocks_dir,
                                      f"block_{block_idx[0]}_{block_idx[1]}_{block_idx[2]}.pth")

        try:
            torch.save(block_data, block_filename)
            if debug:
                print(f"saved block{block_idx}arrive{block_filename}")
                print(f"The block contains the following keys:{list(block_data.keys())}")
                for key, tensor in block_data.items():
                    print(
                        f"  - {key}: shape={tensor.shape}, mean={tensor.mean().item():.4f}, standard deviation={tensor.std().item():.4f}")
            successful_blocks += 1
        except Exception as e:
            print(f"Error: saving block{block_idx}fail:{e}")
            failed_blocks += 1

          
    print(f"\nPre-split completed!")
    print(f"Total number of blocks:{len(all_indices)}")
    print(f"Number of successful blocks:{successful_blocks}")
    print(f"Number of failed blocks:{failed_blocks}")
    print(f"Number of empty blocks:{empty_blocks}")

            
    if args.visualize and all_block_data:
        print("\\nGenerating feature block visualization...")
                      
        all_feature_keys = set()
        for block_data in all_block_data:
            all_feature_keys.update(block_data.keys())

                      
        for feature_key in all_feature_keys:
            print(f"Visualize feature planes:{feature_key}")
            visualize_feature_blocks(all_block_data, all_block_indices, visualization_dir, feature_key,
                                     block_index_fontsize=BLOCK_INDEX_FONTSIZE)

            
    print("\\nVerifying output file...")
    for block_idx in all_indices[:min(5, len(all_indices))]:           
        block_filename = os.path.join(preoptimized_blocks_dir,
                                      f"block_{block_idx[0]}_{block_idx[1]}_{block_idx[2]}.pth")
        if os.path.exists(block_filename):
            file_size = os.path.getsize(block_filename)
            try:
                block_data = torch.load(block_filename, map_location='cpu')
                print(f"piece{block_idx}: size={file_size / 1024:.1f}KB, number of keys ={len(block_data)}")
                if len(block_data) > 0:
                    sample_key = next(iter(block_data.keys()))
                    sample_tensor = block_data[sample_key]
                    print(
                        f"sample tensor{sample_key}: shape={sample_tensor.shape}, mean={sample_tensor.mean().item():.4f}")
                else:
                    print(f"Warning: Block file is empty!")
            except Exception as e:
                print(f"piece{block_idx}: size={file_size / 1024:.1f}KB, failed to load:{e}")
        else:
            print(f"piece{block_idx}: file does not exist")


if __name__ == "__main__":
    main()

                                                                                       
