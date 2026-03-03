import yaml
from src import networks

def load_config(path, default_path=None):
       
                                         
    with open(path, 'r', encoding='utf-8') as f:
        cfg_special = yaml.full_load(f)

                                              
    inherit_from = cfg_special.get('inherit_from')

                                               
                                 
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r', encoding='utf-8') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

                                
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
       
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


        
def get_model(cfg, geo_plane_shapes=None, color_plane_shapes=None, device='cuda:0'):
       

    model = networks.config.get_model(cfg, geo_plane_shapes=geo_plane_shapes, 
                                     color_plane_shapes=color_plane_shapes, 
                                     device=device)

    return model
