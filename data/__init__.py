from data import datasets
from torch.utils.data import DataLoader
def get_loader(config):
    dataset = getattr(datasets, config['dataset']['type'])(**dict(config['dataset']['args']))
    loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=config['shuffle'])

    # compability
    loaders = [loader]
    return loaders