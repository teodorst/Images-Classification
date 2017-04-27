import numpy as np

from layer_interface import LayerInterface
from transfer_functions import tanh

class FullyConnected(LayerInterface):

    def __init__(self, layer_dim):


    def forward(self, inputs):
        exk = np.power(np.e, inputs)
        self.outputs = exk / np.sum(exk)
        return self.outputs

    def backward(self, inputs, output_errors):
        Z  = np.sum(np.multiply(inputs, output_errors))
        delta = np.dot(inputs, output_errors - np.full(output_errors.shape, Z))
        return delta

    def update_parameters(self):
        pass
