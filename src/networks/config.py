from src.networks.decoders import Decoders
import torch
from torch import nn

              
def get_model(cfg, geo_plane_shapes=None, color_plane_shapes=None, device='cuda:0'):
    c_dim = cfg['model']['c_dim']                                       
    truncation = cfg['model']['truncation']                  
    learnable_beta = cfg['rendering']['learnable_beta']              

             
    decoder = Decoders(c_dim=c_dim, truncation=truncation, learnable_beta=learnable_beta,
                      geo_plane_shapes=geo_plane_shapes, color_plane_shapes=color_plane_shapes, device=device)

                            
    if torch.cuda.device_count() > 1 and cfg.get('use_multi_gpu', False):
        print(f"use{torch.cuda.device_count()}Block GPU for DataParallel parallelism")
                                               
                                    
        decoder = nn.DataParallel(decoder)
                                     
        decoder.is_data_parallel = True
    else:
        decoder.is_data_parallel = False

    return decoder
