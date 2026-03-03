import numpy as np
import torch
import random

try:
    from pytorch3d.transforms import (
        matrix_to_quaternion as _p3d_matrix_to_quaternion,
        quaternion_to_matrix as _p3d_quaternion_to_matrix,
    )
except ImportError:
    _p3d_matrix_to_quaternion = None
    _p3d_quaternion_to_matrix = None


def _normalize_quaternion(quaternion, eps=1e-8):
    norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(eps)
    return quaternion / norm


def _matrix_to_quaternion_fallback(matrix):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix with shape (..., 3, 3), got {matrix.shape}.")
    if not matrix.is_floating_point():
        matrix = matrix.float()

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    q_abs = torch.sqrt(torch.clamp(q_abs, min=0.0))
    eps = torch.finfo(matrix.dtype).eps

    quat_candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    denom = 2.0 * q_abs.unsqueeze(-1).clamp_min(eps)
    quat_candidates = quat_candidates / denom

    max_ind = torch.argmax(q_abs, dim=-1)
    gather_index = max_ind[..., None, None].expand(*max_ind.shape, 1, 4)
    quat = torch.gather(quat_candidates, dim=-2, index=gather_index).squeeze(-2)
    return _normalize_quaternion(quat)


def _quaternion_to_matrix_fallback(quaternion):
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with shape (..., 4), got {quaternion.shape}.")
    if not quaternion.is_floating_point():
        quaternion = quaternion.float()

    q = _normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    matrix = torch.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )
    return matrix.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    if _p3d_matrix_to_quaternion is not None:
        return _p3d_matrix_to_quaternion(matrix)
    return _matrix_to_quaternion_fallback(matrix)


def quaternion_to_matrix(quaternion):
    if _p3d_quaternion_to_matrix is not None:
        return _p3d_quaternion_to_matrix(quaternion)
    return _quaternion_to_matrix_fallback(quaternion)

def setup_seed(seed):
       
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def as_intrinsics_matrix(intrinsics):
       
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
       
                     
    if weights.shape[0] == 0 or weights.numel() == 0:
                            
        N_rays = bins.shape[0]
        samples = torch.linspace(0., 1., steps=N_samples, device=device).expand(N_rays, N_samples)
        return samples
    
                  
    if len(weights.shape) == 1:
        weights = weights.unsqueeze(0)          
    
                           
    if bins.shape[0] != weights.shape[0]:
        if bins.shape[0] == 1:
            bins = bins.expand(weights.shape[0], -1)
        elif weights.shape[0] == 1:
            weights = weights.expand(bins.shape[0], -1)
        else:
            raise ValueError(f"Incompatible shapes: bins {bins.shape}, weights {weights.shape}")

                                      
    eps = 1e-8           
    weights = torch.clamp(weights, min=0.0)               
    weights_sum = torch.sum(weights, -1, keepdim=True)
                         
    weights_sum = torch.where(weights_sum < eps, torch.ones_like(weights_sum), weights_sum)
    pdf = weights / weights_sum

    cdf = torch.cumsum(pdf, -1)
                        
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

                          
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

                
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)                         

              
    if inds_g.shape[0] == 0 or cdf.shape[0] == 0 or bins.shape[0] == 0:
                 
        N_rays = max(inds_g.shape[0], cdf.shape[0], bins.shape[0])
        if N_rays == 0:
            N_rays = 1             
        samples = torch.linspace(0., 1., steps=N_samples, device=device).expand(N_rays, N_samples)
        return samples

            
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    
              
    try:
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    except RuntimeError as e:
        print(f"Error in sample_pdf: {e}")
        print(f"inds_g shape: {inds_g.shape}, cdf shape: {cdf.shape}, bins shape: {bins.shape}")
        print(f"matched_shape: {matched_shape}")
                    
        samples = torch.linspace(bins.min().item(), bins.max().item(), steps=N_samples, device=device)
        samples = samples.expand(bins.shape[0], N_samples)
        return samples

               
    eps = 1e-8
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
                                              
    denom = torch.clamp(denom, min=eps)
    t = torch.clamp((u - cdf_g[..., 0]) / denom, min=0.0, max=1.0)                
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def random_select(l, k):
       
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device):
       
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)
    dirs = dirs.unsqueeze(-2)
                                                                
                                                            
    rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def select_uv(i, j, n, b, depths, colors, device='cuda:0'):
       
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n * b,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]           
    j = j[indices]           

    indices = indices.reshape(b, -1)
    i = i.reshape(b, -1)
    j = j.reshape(b, -1)

    depths = depths.reshape(b, -1)
    colors = colors.reshape(b, -1, 3)

    depths = torch.gather(depths, 1, indices)          
    colors = torch.gather(colors, 1, indices.unsqueeze(-1).expand(-1, -1, 3))             

    return i, j, depths, colors

def get_sample_uv(H0, H1, W0, W1, n, b, depths, colors, device='cuda:0'):
       
    depths = depths[:, H0:H1, W0:W1]
    colors = colors[:, H0:H1, W0:W1]

    i, j = torch.meshgrid(torch.linspace(W0, W1 - 1, W1 - W0, device=device), torch.linspace(H0, H1 - 1, H1 - H0, device=device))

    i = i.t()             
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, b, depths, colors, device=device)

    return i, j, depth, color

def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2ws, depths, colors, device):
       
    b = c2ws.shape[0]
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, b, depths, colors, device=device)

    rays_o, rays_d = get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), sample_depth.reshape(-1), sample_color.reshape(-1, 3)

def matrix_to_cam_pose(batch_matrices, RT=True):
       
    if RT:
        return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)
    else:
        return torch.cat([batch_matrices[:, :3, 3], matrix_to_quaternion(batch_matrices[:, :3, :3])], dim=-1)

def cam_pose_to_matrix(batch_poses):
       
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]

    return c2w

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
       
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
                                          
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()             
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
                                                                
                                                            
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
       
                    
    if p.dtype != torch.float32:
        p = p.float()
    
                   
    if isinstance(bound, list):
                  
        bound = torch.stack(bound, dim=0)
    
                       
    if bound.dtype != torch.float32:
        bound = bound.float()
    
                         
    if bound.shape[0] == 3 and bound.shape[1] == 2:
                                     
                           
        min_bound = bound[:, 0]        
        max_bound = bound[:, 1]        
    elif bound.shape[0] == 2 and bound.shape[1] == 3:
                                  
        min_bound = bound[0]        
        max_bound = bound[1]        
    else:
        raise ValueError(f"Unexpected bound shape: {bound.shape}. Expected (2, 3) or (3, 2).")
        
    p = p.reshape(-1, 3)
    
                       
    eps = 1e-6            
    range_x = max_bound[0] - min_bound[0]
    range_y = max_bound[1] - min_bound[1]
    range_z = max_bound[2] - min_bound[2]
    
                              
    range_x = torch.where(torch.abs(range_x) < eps, torch.tensor(eps, device=range_x.device, dtype=range_x.dtype), range_x)
    range_y = torch.where(torch.abs(range_y) < eps, torch.tensor(eps, device=range_y.device, dtype=range_y.dtype), range_y)
    range_z = torch.where(torch.abs(range_z) < eps, torch.tensor(eps, device=range_z.device, dtype=range_z.dtype), range_z)
    
    p[:, 0] = ((p[:, 0] - min_bound[0]) / range_x) * 2 - 1.0
    p[:, 1] = ((p[:, 1] - min_bound[1]) / range_y) * 2 - 1.0
    p[:, 2] = ((p[:, 2] - min_bound[2]) / range_z) * 2 - 1.0
    
    return p

def get_rays_from_pixels(H, W, fx, fy, cx, cy, n_pixels, device):
       
              
    i = torch.randint(0, W, (n_pixels,), device=device).float()           
    j = torch.randint(0, H, (n_pixels,), device=device).float()           
    
            
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)
    
    return dirs, i, j

def sample_rays_and_pixels(H, W, fx, fy, cx, cy, gt_depth, gt_color, rays_d_cam, c2w, device):
       
                   
    dirs, i, j = rays_d_cam
    
            
    i_idx = i.long().clamp(0, W-1)
    j_idx = j.long().clamp(0, H-1)
    
             
    target_d = gt_depth[j_idx, i_idx]
    target_rgb = gt_color[j_idx, i_idx]
    
                   
    rays_d = torch.sum(dirs.unsqueeze(1) * c2w[:3, :3], dim=-1)
    
                  
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return target_d, target_rgb, rays_d, rays_o
