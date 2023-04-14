import numpy as np
import torch
import random
from torchvision.transforms import Compose

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *argv):
        for t in self.transforms:
            argv = t(*argv)
            if len(argv) == 1:
                argv = argv[0]
        return argv

class Normalize(object):
    def __init__(self, mean, std, axis=1):
        self.mean = torch.Tensor(mean).unsqueeze(axis)
        self.std = torch.Tensor(std).unsqueeze(axis)

    def __call__(self, x):
        # TODO
        ## handling the exception when mean and std dimensions are smaller than x
        return (x - self.mean) / self.std 

class SelectLastTimeStep(object):
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, x):
        offset = x.shape[self.axis]
        return x.index_select(self.axis, torch.arange(offset-1, offset))


class RandomTemporalCrop(object):
    def __init__(self, window_size, stride=1, axis=1):
        self.window_size = window_size
        self.stride = stride
        self.axis = axis

    def __call__(self, *argv):
        x = argv[0]
        high = int((x.size(self.axis) - self.window_size) / self.stride)
        offset = self.stride * random.randint(0, high)

        crop_argv = []

        for arg in argv:
            crop_argv.append(arg.index_select(self.axis, torch.arange(offset, offset+self.window_size)))
        return crop_argv


class TemporalCrop(object):
    def __init__(self, window_size, start_idx, axis):
        self.window_size = window_size
        self.start_idx = start_idx

        self.axis = axis

    def __call__(self, *argv):
        offset = self.start_idx

        crop_argv = []

        for arg in argv:
            crop_argv.append(arg.index_select(self.axis, torch.arange(offset, offset+self.window_size)))
        return crop_argv


def get_transform(transform_cfg):
    list_transform = []
    for t in transform_cfg:
        list_transform.append(globals()[t['type']](**dict(t['args'])))

    return Compose(list_transform)

# if __name__ == '__main__':
#     a=get_transform(['RandomTemporalCrop', 'Normalize'])
#     import pdb
#     pdb.set_trace()