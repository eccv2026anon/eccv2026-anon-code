import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer
from src.utils.BlockManager import BlockManager

torch.multiprocessing.set_sharing_strategy('file_system')

class StructRecon():
       

    @staticmethod
    def _normalize_device(device, role=""):
        dev = str(device)
        if dev.startswith("cuda"):
                                                                                 
            if not torch.cuda.is_available():
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                raise RuntimeError(
                    f"[StructRecon] {role} device is '{dev}' but torch.cuda.is_available() is False. "
                    f"Check that the GPU is available inside the container and that your "
                    f"PyTorch has CUDA enabled. CUDA_VISIBLE_DEVICES={cuda_visible!r}."
                )

            n = int(torch.cuda.device_count())
            if n <= 0:
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                raise RuntimeError(
                    f"[StructRecon] {role} device is '{dev}' but torch.cuda.device_count()==0. "
                    f"CUDA_VISIBLE_DEVICES={cuda_visible!r}. If this is a CPU instance, set device='cpu'."
                )

            idx = 0
            if dev != "cuda" and ":" in dev:
                try:
                    idx = int(dev.split(":", 1)[1])
                except Exception:
                    idx = 0

            if idx < 0 or idx >= n:
                                                                              
                print(
                    f"[StructRecon] WARNING: {role} device '{dev}' out of range; "
                    f"{n} CUDA devices visible. Falling back to 'cuda:0'."
                )
                idx = 0

            return f"cuda:{idx}"

        return dev

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']
        mapping_cfg = cfg.get('mapping', {})
        tracking_cfg = cfg.get('tracking', {})
        meshing_cfg = cfg.get('meshing', {})
        self.mapping_device = mapping_cfg.get('device', cfg.get('device', 'cuda:0'))
        self.tracking_device = tracking_cfg.get('device', cfg.get('tracking_device', self.mapping_device))
        self.meshing_device = meshing_cfg.get('device', self.mapping_device)
        self.mapping_device = self._normalize_device(self.mapping_device, role="mapping")
        self.tracking_device = self._normalize_device(self.tracking_device, role="tracking")
        self.meshing_device = self._normalize_device(self.meshing_device, role="meshing")
        self.device = self.mapping_device
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg)
        self.shared_decoders = model

        self.scale = cfg['scale']

        self.load_bound(cfg)

        self.use_block_manager = bool(cfg.get('model', {}).get('use_block_manager', False))
        self.block_manager = None
        decoders_module = self.shared_decoders.module if hasattr(self.shared_decoders, 'module') else self.shared_decoders
        if self.use_block_manager:
            self.block_manager = BlockManager(cfg, device=self.device, bound=self.bound.clone())
            if hasattr(decoders_module, 'set_block_manager'):
                decoders_module.set_block_manager(self.block_manager)
        else:
            if hasattr(decoders_module, 'use_block_manager'):
                decoders_module.use_block_manager = False
            if hasattr(decoders_module, 'block_manager'):
                decoders_module.block_manager = None

        self.init_planes(cfg)

                           
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()

                                                         
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()                       
        self.mapping_cnt.share_memory_()
        
                      
                                     
        self.mapper_done_flag = torch.zeros((1)).int()
        self.mapper_done_flag.share_memory_()
                      

                                                                      
        for shared_planes in [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz]:
            for i, plane in enumerate(shared_planes):
                plane = plane.to(self.device)
                plane.share_memory_()
                shared_planes[i] = plane

        for shared_c_planes in [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz]:
            for i, plane in enumerate(shared_c_planes):
                plane = plane.to(self.device)
                plane.share_memory_()
                shared_c_planes[i] = plane

        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        
                      
        self.mapper_done_flag = self.mapper_done_flag
                      
        
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpts/")

    def update_cam(self):
           
                                                                               
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

                                                                  
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
           

                                                             
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()
        bound_dividable = cfg['planes_res']['bound_dividable']
                                                                          
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]
        self.shared_decoders.bound = self.bound

    def init_planes(self, cfg):
           
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']

        self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        self.fine_c_planes_res = cfg['c_planes_res']['fine']

        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

                                                
        planes_xy, planes_xz, planes_yz = [], [], []
        c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
        planes_res = [self.coarse_planes_res, self.fine_planes_res]
        c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]

        planes_dim = c_dim
        for grid_res in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for grid_res in c_planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        self.shared_planes_xy = planes_xy
        self.shared_planes_xz = planes_xz
        self.shared_planes_yz = planes_yz

        self.shared_c_planes_xy = c_planes_xy
        self.shared_c_planes_xz = c_planes_xz
        self.shared_c_planes_yz = c_planes_yz

    def tracking(self, rank):
           

                                                                  
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
           

        self.mapper.run()

    def run(self):
           

        processes = []
        for rank in range(0, 2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

                                                
if __name__ == '__main__':
    pass
