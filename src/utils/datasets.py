import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset, Sampler


class SeqSampler(Sampler):
       

    def __init__(self, n_samples, step, include_last=True):
        self.n_samples = n_samples
        self.step = step
        self.include_last = include_last

    def __iter__(self):
        if self.include_last:
            return iter(list(range(0, self.n_samples, self.step)) + [self.n_samples - 1])
        else:
            return iter(range(0, self.n_samples, self.step))

    def __len__(self) -> int:
        return self.n_samples


def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']

                                 
        self.first_frame_abs_pose = None
        self.apply_pose_transform = False

                                    
        first_frame_abs_pose_cfg = None
        if 'first_frame_abs_pose' in cfg:
            first_frame_abs_pose_cfg = cfg['first_frame_abs_pose']
            print("Message: BaseDataset found 'first_frame_abs_pose' in root configuration")
        elif 'model' in cfg and 'first_frame_abs_pose' in cfg['model']:
            first_frame_abs_pose_cfg = cfg['model']['first_frame_abs_pose']
            print("Message: BaseDataset found 'first_frame_abs_pose' in model configuration")

        if first_frame_abs_pose_cfg is not None:
            try:
                       
                self.first_frame_abs_pose = torch.tensor(first_frame_abs_pose_cfg, dtype=torch.float32)
                self.apply_pose_transform = True
                print(f"Message: Absolute pose transformation will be applied to all frames: \n{self.first_frame_abs_pose}")
            except Exception as e:
                print(f"Warning: Failed to load first_frame_abs_pose:{e}")
                self.first_frame_abs_pose = None
                self.apply_pose_transform = False

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
                              
        depth_data = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                                 
        if depth_data is None:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
                                                                     
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.

                                                 
        if depth_data is None:
            raise RuntimeError(f"Unable to read depth image:{depth_path}")
        if len(depth_data.shape) > 2:
                              
            if depth_data.dtype == np.uint8 and depth_data.shape[2] >= 2:
                ch = depth_data
                                       
                candidates = []
                          
                candidates.append((ch[:, :, 0].astype(np.uint16) + (ch[:, :, 1].astype(np.uint16) << 8)))
                          
                if ch.shape[2] > 2:
                    candidates.append((ch[:, :, 1].astype(np.uint16) + (ch[:, :, 2].astype(np.uint16) << 8)))
                          
                if ch.shape[2] > 2:
                    candidates.append((ch[:, :, 2].astype(np.uint16) + (ch[:, :, 0].astype(np.uint16) << 8)))
                             
                ranges = [np.percentile(c[c > 0], 99) if np.any(c > 0) else 0 for c in candidates]
                best_idx = int(np.argmax(ranges))
                depth_u16 = candidates[best_idx]
                depth_data = depth_u16
                print(f"Deep reconstruction: using candidates{best_idx}Synthetic 16-bit depth (99th percentile ={ranges[best_idx]:.2f})")
            else:
                                       
                best_c = 0
                best_range = 0
                for i in range(depth_data.shape[2]):
                    non_zero = depth_data[:, :, i][depth_data[:, :, i] > 0]
                    rng = (np.percentile(non_zero, 99) - np.percentile(non_zero, 1)) if non_zero.size > 0 else 0
                    if rng > best_range:
                        best_range = rng
                        best_c = i
                print(f"Warning: Depth is multi-channel, use channels{best_c}As a depth source (range{best_range:.2f})")
                depth_data = depth_data[:, :, best_c]

             
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale

        if self.crop_size is not None:
                                                                            
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
                                                                                     
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        pose[:3, 3] *= self.scale

                    
        if self.apply_pose_transform and self.first_frame_abs_pose is not None:
                                         
                                                
            if index == 0:
                pose = self.first_frame_abs_pose.clone()
            else:
                                
                            
                first_frame_orig_pose = self.poses[0]
                                   
                                                                   
                                                                        
                                                     
                first_frame_inv = torch.inverse(first_frame_orig_pose)
                relative_pose = first_frame_inv @ pose
                                 
                transformed_pose = self.first_frame_abs_pose @ relative_pose
                pose = transformed_pose

            if index == 0 or index == 1:
                print(f"frame{index}After applying pose transformation: \n{pose}")

                                                                                                                                                             
        return index, color_data, depth_data, pose


class Replica(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, args, scale, device)
                                                                       
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
                              
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
                                              
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and\
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
                                                  
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
                                                 
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class ICPARK(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(ICPARK, self).__init__(cfg, args, scale, device)
                                       
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4])) or sorted(
            glob.glob(os.path.join(
                self.input_folder, 'color', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))

                  
        if not self.color_paths:
            print(f"Warning: in{os.path.join(self.input_folder, 'color')}RGB image file not found in")
            print("Create a dummy RGB image for testing...")
                        
            os.makedirs(os.path.join(self.input_folder, 'color'), exist_ok=True)
            test_img_path = os.path.join(self.input_folder, 'color', '0.png')
                         
            test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128        
            cv2.imwrite(test_img_path, test_img)
            self.color_paths = [test_img_path]

        if not self.depth_paths:
            print(f"Warning: in{os.path.join(self.input_folder, 'depth')}Depth image file not found in")
            print("Create a virtual depth image for testing...")
                         
            os.makedirs(os.path.join(self.input_folder, 'depth'), exist_ok=True)
            test_depth_path = os.path.join(self.input_folder, 'depth', '0.png')
                          
            test_depth = np.ones((480, 640), dtype=np.uint16) * 1000        
            cv2.imwrite(test_depth_path, test_depth)
            self.depth_paths = [test_depth_path]

                                  
        pose_path = os.path.join(self.input_folder, 'pose')
        poses_path = os.path.join(self.input_folder, 'poses')

        if os.path.exists(pose_path) and os.listdir(pose_path):
            self.load_poses(pose_path)
            print(f"Use the pose directory:{pose_path}")
        elif os.path.exists(poses_path) and os.listdir(poses_path):
            self.load_poses(poses_path)
            print(f"Use poses directory:{poses_path}")
        else:
                                   
            print(f"Warning: Pose folder not found, using identity matrix as default pose")
            self.poses = [torch.eye(4) for _ in range(len(self.color_paths))]

                        
            os.makedirs(pose_path, exist_ok=True)
            test_pose_path = os.path.join(pose_path, '0.txt')
            with open(test_pose_path, 'w') as f:
                f.write("1.0 0.0 0.0 0.0\n")
                f.write("0.0 1.0 0.0 0.0\n")
                f.write("0.0 0.0 1.0 0.0\n")
                f.write("0.0 0.0 0.0 1.0\n")

        self.n_img = min(len(self.color_paths), len(self.depth_paths))
        print(f"ICPARK dataset: loaded{self.n_img}Frame RGBD data")
        print(
            f"Number of image files:{len(self.color_paths)}, number of depth files:{len(self.depth_paths)}, number of pose files:{len(self.poses)}")

    def load_poses(self, path):
                                        
        self.poses = []

                           
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))

        if not pose_paths:
            print(f"Warning: in{path}Pose file not found in")
                          
            self.poses = [torch.eye(4) for _ in range(len(self.color_paths))]
            return

        print(f"turn up{len(pose_paths)}pose files")

                     
        if len(pose_paths) > 0:
            print(f"Example of pose file:{os.path.basename(pose_paths[0])}")
        if len(self.color_paths) > 0:
            print(f"RGB image example:{os.path.basename(self.color_paths[0])}")

        for pose_path in pose_paths:
            try:
                with open(pose_path, "r") as f:
                    lines = f.readlines()

                                   
                ls = []
                for line in lines:
                    l = list(map(float, line.split(' ')))
                    ls.append(l)

                c2w = np.array(ls).reshape(4, 4)

                                    
                c2w[:3, 1] *= -1        
                c2w[:3, 2] *= -1        

                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
            except Exception as e:
                print(f"Warning: Unable to parse pose file{pose_path}: {e}")
                self.poses.append(torch.eye(4))

                                       
        if len(self.poses) < len(self.color_paths):
            print(f"Warning: Number of pose files ({len(self.poses)}) is less than the number of image files ({len(self.color_paths)})")
            print(f"Pose file range:{os.path.basename(pose_paths[0])}arrive{os.path.basename(pose_paths[-1])}")
            print(f"RGB image range:{os.path.basename(self.color_paths[0])}arrive{os.path.basename(self.color_paths[-1])}")

                      
            while len(self.poses) < len(self.color_paths):
                self.poses.append(torch.eye(4))

    def __getitem__(self, index):
                                       
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

                 
        color_data = cv2.imread(color_path)
        if color_data is None:
            print(f"Error: Unable to read RGB image{color_path}")
            color_data = np.zeros((480, 640, 3), dtype=np.uint8)          

                                             
        depth_data = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_data is None:
            print(f"Error: Unable to read depth image{depth_path}")
            depth_data = np.zeros((480, 640), dtype=np.uint16)           

                    
        if len(depth_data.shape) > 2:
            print(f"Warning: depth images{depth_path}The dimensions are{depth_data.shape}, try to reconstruct 16-bit depth")
            if depth_data.dtype == np.uint8 and depth_data.shape[2] >= 2:
                ch = depth_data
                candidates = []
                candidates.append((ch[:, :, 0].astype(np.uint16) + (ch[:, :, 1].astype(np.uint16) << 8)))
                if ch.shape[2] > 2:
                    candidates.append((ch[:, :, 1].astype(np.uint16) + (ch[:, :, 2].astype(np.uint16) << 8)))
                    candidates.append((ch[:, :, 2].astype(np.uint16) + (ch[:, :, 0].astype(np.uint16) << 8)))
                ranges = [np.percentile(c[c > 0], 99) if np.any(c > 0) else 0 for c in candidates]
                best_idx = int(np.argmax(ranges))
                depth_data = candidates[best_idx]
                print(f"Deep reconstruction: using candidates{best_idx}Synthetic 16-bit depth (99th percentile ={ranges[best_idx]:.2f})")
            else:
                               
                best_c = 0
                best_range = 0
                for i in range(depth_data.shape[2]):
                    non_zero = depth_data[:, :, i][depth_data[:, :, i] > 0]
                    rng = (np.percentile(non_zero, 99) - np.percentile(non_zero, 1)) if non_zero.size > 0 else 0
                    if rng > best_range:
                        best_range = rng
                        best_c = i
                print(f"Deep multi-channel degradation processing: using channels{best_c}As a depth source (range{best_range:.2f})")
                depth_data = depth_data[:, :, best_c]

                    
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.

                  
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale

                        
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

                      
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale

              
        if self.crop_size is not None:
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

                
        pose = self.poses[index]
        pose[:3, 3] *= self.scale

                                           
                                     
                                  
                
                                       
                                          
                                      
        coord_transform = torch.tensor([
            [1.,  0.,  0., 0.],
            [0.,  0., -1., 0.],                         
            [0.,  1.,  0., 0.],                        
            [0.,  0.,  0., 1.]
        ], dtype=pose.dtype, device=pose.device)
        
                                                                  
        pose = coord_transform @ pose @ torch.inverse(coord_transform)
        
        if index == 0 or index == 1:
            print(f"ICPARK dataset: frames{index}Y/Z coordinate system transformation is applied (Y front Z front, Z up Y down)")
                                                  

                                               
                    
        if self.apply_pose_transform and self.first_frame_abs_pose is not None:
                                         
                                                
            if index == 0:
                pose = self.first_frame_abs_pose.clone()
                if index == 0:
                    print(f"ICPARK dataset: frames{index}first_frame_abs_pose pose transformation applied")
            else:
                                
                            
                first_frame_orig_pose = self.poses[0]
                                   
                first_frame_inv = torch.inverse(first_frame_orig_pose)
                relative_pose = first_frame_inv @ pose
                                 
                transformed_pose = self.first_frame_abs_pose @ relative_pose
                pose = transformed_pose

                if index == 1:
                    print(f"ICPARK dataset: frames{index}Relative pose transformation applied")

        return index, color_data, depth_data, pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
    "ICPARK": ICPARK
}
