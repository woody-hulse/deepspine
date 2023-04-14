# This code is based on https://github.com/victoresque/pytorch-template
# AUTHOR: MINJU JUNG

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model
from optimizer import get_optimizer
from data import get_loader
import masked_loss
import trainers


import argparse
import collections
# import os
# import random
import time


# import utils
# import itertools
# import progressbar
import numpy as np
import scipy.stats as scistats
import json

def main(config):
    # # set save_dir where trained model and log will be saved.
    # save_dir = Path(self.config['trainer']['save_dir'])
    # timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''

    # exper_name = self.config['name']
    # self._save_dir = save_dir / 'models' / exper_name / timestamp
    # self._log_dir = save_dir / 'log' / exper_name / timestamp

    # self.save_dir.mkdir(parents=True, exist_ok=True)
    # self.log_dir.mkdir(parents=True, exist_ok=True)
    
    device = config['device']['type']
    seed = config['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    emb_cfg = config['arch']['embedding']
    core_cfg = config['arch']['core']
    readout_cfg = config['arch']['readout']

    # get train loader(s) and valid loader(s)
    train_loaders = get_loader(config['data_loader']['train_loader_args'])
    valid_loaders = get_loader(config['data_loader']['valid_loader_args'])

    embeddings, core, readouts = get_model(emb_cfg, core_cfg, readout_cfg)
    optim_embeddings, optim_core, optim_readouts = get_optimizer(embeddings, core, readouts, **dict(config['optimizer']))

    import ipdb; ipdb.set_trace()
    # import pdb
    # pdb.set_trace()
    # print(sum(p.numel() for p in core.parameters() if p.requires_grad))

    # # multi-GPU
    # if device in ['cuda', 'gpu']:
    #     if torch.cuda.device_count() > 1:
    #         embeddings = nn.DataParallel(embeddings)
    #         core = nn.DataParallel(core)
    #         readouts = nn.DataParallel(readouts)

    embeddings = embeddings.to(device)
    core = core.to(device)
    readouts = readouts.to(device)

    # loss
    criterion = getattr(masked_loss, config['loss'])


    trainer = getattr(trainers, config['trainer']['type']).Trainer(
        model={
            'embeddings': embeddings,
            'core': core,
            'readouts': readouts,
        },
        criterion=criterion,
        optimizer={
            'embeddings': optim_embeddings,
            'core': optim_core,
            'readouts': optim_readouts,
        },
        data_loader={
            'train': train_loaders,
            'valid': valid_loaders,
        },
        config=config
    )

    trainer.train()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    # argparser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    # argparser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    args = argparser.parse_args()
    with open(args.config) as config_json:
        config = json.load(config_json)


    main(config)

    # # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
    #     CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    # ]
    
    # config = ConfigParser(argparser, options)
    # main(config) 
