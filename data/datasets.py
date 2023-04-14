import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, Dataset
from scipy.signal import decimate
import os

from .transforms import get_transform
from .utils import make_path, read_memmap

class SyntheticDataset(Dataset):
    def __init__(self, 
            file_path,
            input_type, 
            output_type,
            is_ees=False,
            indices_path=None,
            ratio=1,
            is_train=True,
            input_transform=None, 
            target_transform=None, 
            share_transform=None):

        mode = 'train' if is_train else 'valid'
        input_path = make_path('%s.dat' % (input_type), directory=os.path.join(file_path, mode))
        output_path = make_path('%s.dat' % (output_type), directory=os.path.join(file_path, mode))
        meta_path = make_path('meta.dat', directory=os.path.join(file_path, mode))


        self.inputs = read_memmap(input_path)
        self.outputs = read_memmap(output_path)
        self.meta = read_memmap(meta_path)[:, np.newaxis, :]

        if input_type == "kinematics":
            self.inputs = np.squeeze(self.inputs, axis=3)
            
        if output_type == "emg":
            self.outputs = np.abs(self.outputs)

        self.is_ess = is_ees
        if self.is_ess:
            ees_path = make_path('ees.dat', directory=os.path.join(file_path, mode))
            self.ees = read_memmap(ees_path)

        self.input_transform = get_transform(input_transform) if input_transform is not None else None
        self.target_transform = get_transform(target_transform) if target_transform is not None else None
        self.share_transform = get_transform(share_transform) if share_transform is not None else None
        
        self.n_samples = int(self.outputs.shape[0])
        if (indices_path is not None) and (ratio < 1):
                indices = torch.load(indices_path)
                self.indices = indices[:int(indices.size(0)*ratio)]
                self.n_samples = int(self.indices.shape[0])
        else:
            self.indices = None
        
        print(self.n_samples)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        input = self.inputs[index]
        output = self.outputs[index]

        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()
        
        if self.is_ess:
            ees = self.ees[index]
            ees = torch.from_numpy(ees).float()
            meta = torch.from_numpy(self.meta[index]).float()

        if self.share_transform is not None:
            if self.is_ess:
                input, output, ees = self.share_transform(input, output, ees)
            else:
                input, output = self.share_transform(input, output)

        if self.input_transform is not None:
            input = self.input_transform(input)

        if self.target_transform is not None:
            output = self.target_transform(output)
        
        if self.is_ess:
            return input, output, ees, meta
        else:
            return input, output

    def __len__(self):
        return self.n_samples
