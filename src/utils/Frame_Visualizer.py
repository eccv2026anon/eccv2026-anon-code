import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.common import cam_pose_to_matrix


class Frame_Visualizer(object):
       

    def __init__(self, freq, inside_freq, vis_dir, renderer, truncation, verbose, device='cuda:0'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.truncation = truncation
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def save_imgs(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, all_planes, decoders):
           
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
                gt_color_np = gt_color.squeeze(0).cpu().numpy()

                if c2w_or_camera_tensor.shape[-1] > 4:        
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
                else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

                                           
                renderer_H, renderer_W = self.renderer.H, self.renderer.W
                gt_depth_tensor = gt_depth.clone()

                                                
                if gt_depth.shape[-2:] != (renderer_H, renderer_W):
                    print(f"Convert the GT depth map from{gt_depth.shape[-2:]}Adjust to{renderer_H}x{renderer_W}")
                                            
                    if len(gt_depth.shape) == 2:
                        gt_depth_tensor = gt_depth_tensor.unsqueeze(0)

                                 
                    gt_depth_tensor = torch.nn.functional.interpolate(
                        gt_depth_tensor.unsqueeze(0),          
                        size=(renderer_H, renderer_W),
                        mode='nearest'
                    ).squeeze(0)          

                                            
                if hasattr(decoders, 'use_block_manager') and decoders.use_block_manager:
                                                
                    if decoders.block_manager is None:
                        print("Warning: Decoder is in blocking mode but block_manager is not set")
                    else:
                                     
                        camera_pos = c2w[:3, 3].cpu().numpy()
                                  
                        decoders.block_manager.prepare_blocks_for_camera(camera_pos)

                           
                depth, color = self.renderer.render_img(all_planes, decoders, c2w, self.truncation,
                                                        self.device, gt_depth=gt_depth_tensor)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()

                                                                                  
                if depth_np.shape != gt_depth_np.shape:
                    depth_np = cv2.resize(depth_np, (gt_depth_np.shape[1], gt_depth_np.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                                  
                if color_np.shape[:2] != gt_color_np.shape[:2]:
                    color_np = cv2.resize(color_np, (gt_color_np.shape[1], gt_color_np.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)

                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                                               
                max_depth = float(np.max(gt_depth_np))
                if not np.isfinite(max_depth) or max_depth <= 0:
                    max_depth = float(np.max(depth_np)) if np.isfinite(np.max(depth_np)) else 1.0

                axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                                                 
                axs[1, 0].imshow(gt_color_np)
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np)
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual)
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
                plt.cla()
                plt.clf()

                if self.verbose:
                    print(
                        f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')

    def update_inside_rendering(self, idx, iter, c2w, gt_color_full, gt_depth_full, all_planes, decoders):
           
                       
        if (idx % self.freq != 0) or (iter % self.inside_freq != 0):
            return
        try:
            H, W = self.renderer.H, self.renderer.W
                                                             
            gt_rgb = gt_color_full
            gt_d = gt_depth_full
            if isinstance(gt_rgb, torch.Tensor) and gt_rgb.dim() == 4:
                gt_rgb = gt_rgb.squeeze(0)
            if isinstance(gt_d, torch.Tensor) and gt_d.dim() == 3:
                gt_d = gt_d.squeeze(0)

                                             
                  
            if gt_d.dim() == 1:
                              
                gt_d_img = torch.zeros((H, W), device=self.renderer.device, dtype=torch.float32)
            else:
                if gt_d.shape[-2:] != (H, W):
                    d = gt_d
                    if d.dim() == 2:
                        d = d.unsqueeze(0).unsqueeze(0)
                    elif d.dim() == 3 and d.shape[-1] == 1:
                        d = d.permute(2, 0, 1).unsqueeze(0)
                    elif d.dim() == 3:
                        d = d.unsqueeze(0).unsqueeze(0)
                    d_resized = torch.nn.functional.interpolate(d, size=(H, W), mode='nearest')
                    gt_d_img = d_resized.squeeze().to(self.renderer.device)
                else:
                    gt_d_img = gt_d.to(self.renderer.device)

                  
            if gt_rgb.dim() == 2 and gt_rgb.shape[-1] == 3:
                              
                gt_rgb_img = torch.zeros((H, W, 3), device=self.renderer.device, dtype=torch.float32)
            else:
                if gt_rgb.shape[-2:] != (H, W):
                    rgb = gt_rgb
                    if rgb.dim() == 3 and rgb.shape[-1] == 3:
                        rgb = rgb.permute(2, 0, 1).unsqueeze(0)             
                    elif rgb.dim() == 3:
                        rgb = rgb.unsqueeze(0)
                    elif rgb.dim() == 2:
                        rgb = rgb.unsqueeze(0).unsqueeze(0)
                    rgb_resized = torch.nn.functional.interpolate(rgb, size=(H, W), mode='bilinear', align_corners=True)
                    if rgb_resized.shape[1] == 3:
                        gt_rgb_img = rgb_resized.squeeze(0).permute(1, 2, 0).to(self.renderer.device)
                    else:
                                      
                        gt_rgb_img = torch.zeros((H, W, 3), device=self.renderer.device, dtype=torch.float32)
                else:
                    gt_rgb_img = gt_rgb.to(self.renderer.device)

                                          
                self.save_imgs(
                    idx=idx,
                    iter=iter,
                    gt_depth=gt_d_img.unsqueeze(0),
                    gt_color=gt_rgb_img.unsqueeze(0),
                    c2w_or_camera_tensor=c2w,
                    all_planes=all_planes,
                    decoders=decoders
                )
        except Exception as e:
            if self.verbose:
                print(f"An error occurred during visualization:{e}")
                try:
                    print(f"GT RGB shape:{gt_color_full.shape}, GT depth shape:{gt_depth_full.shape}")
                except Exception:
                    pass
                print(f"Renderer size:{self.renderer.H}x{self.renderer.W}")
