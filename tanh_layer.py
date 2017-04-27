import numpy as np

from layer_interface import LayerInterface
from transfer_functions import tanh

class FullyConnected(LayerInterface):

    def __init__(self, layer_dim):
        self.layer_dim = layer_dim
        self.outputs = np.zeros(layer_dim, 1)

    def forward(self, inputs):
        self.outputs = tanh(inputs)

    def backward(self, inputs, output_errors):
        return np.multiply(np.ones(self.layer_dim, 1) - np.multiply(self.outputs, self.outputs), output_errors)

    def update_parameters(self):
        pass
