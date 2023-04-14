import torch

def get_optimizer(embeddings, core, readouts, embedding_cfg, core_cfg, readout_cfg):
    optim_embeddings = []
    optimizer = getattr(torch.optim, embedding_cfg['type'])
    optim_args = dict(embedding_cfg['args'])
    for embedding in embeddings:
        optim_embeddings.append(optimizer(embedding.parameters(), **optim_args))

    optimizer = getattr(torch.optim, core_cfg['type'])
    optim_args = dict(core_cfg['args'])
    optim_core = optimizer(core.parameters(), **optim_args)
    
    optim_readouts = []
    optimizer = getattr(torch.optim, readout_cfg['type'])
    optim_args = dict(readout_cfg['args'])
    for readout in readouts:
        optim_readouts.append(optimizer(readout.parameters(), **optim_args)) 

    return optim_embeddings, optim_core, optim_readouts


    # optim_embeddings = []
    # optimizer = getattr(torch.optim, config['optimizer']['embedding']['type'])
    # optim_args = dict(config['optimizer']['embedding']['args'])
    # for embedding in embeddings:
    #     optim_embeddings.append(optimizer(embedding.parameters(), **optim_args))

    # optimizer = getattr(torch.optim, config['optimizer']['core']['type'])
    # optim_args = dict(config['optimizer']['core']['args'])
    # optim_core = optimizer(core.parameters(), **optim_args)
    
    # optim_readouts = []
    # optimizer = getattr(torch.optim, config['optimizer']['readout']['type'])
    # optim_args = dict(config['optimizer']['readout']['args'])
    # for readout in readouts:
    #     optim_readouts.append(optimizer(readout.parameters(), **optim_args)) 
