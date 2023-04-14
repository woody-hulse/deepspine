from .feedforward import ConvNet1d, SensoryEncoder
from .recurrent import StackedGRU, StackedLayerNormGRU
from .spine import SpinalCordCircuit

def get_layer(name):
    return {
        "ConvNet1d": ConvNet1d,
        'SensoryEncoder': SensoryEncoder,
        "GRU": StackedGRU,
        "LayerNormGRU": StackedLayerNormGRU,
        "SCC": SpinalCordCircuit,
    }[name]