import os
import torch

class Logger(object):
       

    def __init__(self, structrecon):
        self.verbose = structrecon.verbose                                                                      
        self.ckptsdir = structrecon.ckptsdir                                         
        self.gt_c2w_list = structrecon.gt_c2w_list                                                    
        self.shared_decoders = structrecon.shared_decoders                      
        self.estimate_c2w_list = structrecon.estimate_c2w_list               

    def log(self, idx, keyframe_list):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
