from torch import nn
from layer import get_layer

def make_layer(config):
    if isinstance(config, dict):
        # single module
        return get_layer(config['type'])(**dict(config['args']))
    else:
        # multiple modules
        layer = nn.Sequential()
        for cfg in config:
            layer.add_module(cfg['type'], get_layer(cfg['type'])(**dict(cfg['args'])))
        return layer

# def get_model(emb_cfg, core_cfg, readout_cfg):
#     model = []

#     # define embedding
#     if isinstance(emb_cfg, dict):
#         embedding = make_layer(emb_cfg)
#         model.append(embedding)
#     else:
#         embeddings = []
#         for cfg in emb_cfg:
#             embedding = make_layer(cfg)
#             embeddings.append(embedding)
        
#         embeddings = nn.ModuleList(embeddings)
#         model.append(embeddings)

#     # define core
#     core = make_layer(core_cfg)
#     model.append(core)

#     # define readout
#     if isinstance(readout_cfg, dict):
#         readout = make_layer(readout_cfg)
#         model.append(readout)
#     else:
#         readouts = []
#         for cfg in readout_cfg:
#             readout = make_layer(cfg)
#             readouts.append(readout)
        
#         readouts = nn.ModuleList(readouts) 
#         model.append(readouts)

#     return model

def get_model(emb_cfg, core_cfg, readout_cfg):    
    
    if isinstance(emb_cfg, dict):
        emb_cfg = [emb_cfg]

    if isinstance(readout_cfg, dict):
        readout_cfg = [readout_cfg]

    # define embedding
    embeddings = []
    for cfg in emb_cfg:
        embedding = make_layer(cfg)
        embeddings.append(embedding)

    embeddings = nn.ModuleList(embeddings)

    # define core
    core = make_layer(core_cfg)

    # define readout
    readouts = []
    for cfg in readout_cfg:
        readout = make_layer(cfg)
        readouts.append(readout)
    
    readouts = nn.ModuleList(readouts) 

    return embeddings, core, readouts
