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
import time, os
import numpy as np
import scipy.stats as scistats
import json
from tqdm import tqdm

class MLP(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(n_in, 128),
      nn.ReLU(),
      nn.Linear(128, n_out),
    )

class GRU(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.flatten = nn.Flatten()
    self.gru = nn.GRU(n_in, 8, 3, batch_first=True)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(8, n_out)

  def forward(self, x):
    x = self.flatten(x)
    x, hidden = self.gru(x)
    x = self.relu(x)
    x = self.linear(x)

    return x


def main(config):
    
    device = config['device']['type']
    seed = config['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    # get train loader(s) and valid loader(s)
    train_loaders = get_loader(config['data_loader']['train_loader_args'])
    valid_loaders = get_loader(config['data_loader']['valid_loader_args'])

    # loss
    criterion = getattr(masked_loss, config['loss'])
    MAX_EPOCHS = 1000

    model = GRU(n_in = 400 * 3, n_out = 400 * 2) 
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(MAX_EPOCHS)):
        for kinematics, emg, ees, _ in train_loaders[0]:

            optimizer.zero_grad()

            xin = torch.cat([kinematics, ees], dim=1)
            xin = xin.flatten(start_dim=1)
            xin = xin.to(device)

            xout = model(xin)
            xout = xout.view(-1, 2, 400) # shape of output EMG

            emg = emg.to(device)
            loss = criterion(xout, emg)

            loss.backward()
            optimizer.step()

        print('Epoch {}, loss={}'.format(epoch, loss.item()))

    torch.save(model.state_dict(), 'outputs/mlp_beta.pth')

def main_test(config):
    device = config['device']['type']
    seed = config['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    # get train loader(s) and valid loader(s)
    train_loaders = get_loader(config['data_loader']['train_loader_args'])
    valid_loaders = get_loader(config['data_loader']['valid_loader_args'])

    # get the model and its parameters
    model = GRU(n_in = 400 * 3, n_out = 400 * 2) 
    # model.load_state_dict(torch.load('outputs/mlp_beta.pth'))
    model = model.to(device)
    model = model.eval()

    # loss
    criterion = F.smooth_l1_loss
 
    list_emg, list_predicted_emg, list_meta, list_loss = [], [], [], []

    batch_id = 0
    for kinematics, emg, ees, meta in valid_loaders[0]:
        print('BATCH: {}'.format(batch_id))
        batch_id += 1

        # [batch x dim x time]
        kinematics = kinematics.to(device)
        emg = emg.to(device)
        ees = ees.to(device)

        xin = torch.cat([kinematics, ees], dim=1)
        xin = xin.flatten(start_dim=1)
        xin = xin.to(device)

        with torch.no_grad():
            xout = model(xin)
            xout = xout.view(-1, 2, 400) # shape of output EMG

        loss = criterion(xout, emg, reduction='none')

        list_emg.append(emg)
        list_predicted_emg.append(xout)
        list_loss.append(loss)
        list_meta.append(meta)

    loss = torch.cat(list_loss, dim=0)
    emg = torch.cat(list_emg, dim=0)                        # [batch x dim x time]
    predicted_emg = torch.cat(list_predicted_emg, dim=0)    # [batch x dim x time]
    metas = torch.cat(list_meta, dim=0)

    state = {
        'predicted_emg': predicted_emg,
        'emg': emg,
        'loss': loss,
        'meta': metas
    }

    save_dir = os.path.join('outputs', 'MLPeval.pth')
    torch.save(state, save_dir)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')

    args = argparser.parse_args()
    with open(args.config) as config_json:
        config = json.load(config_json)

    main(config)
    # main_test(config)

