import numpy as np
from time import sleep

from layer_interface import LayerInterface
from transfer_functions import tanh

class Tanh(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = tanh(inputs)
        return self.outputs

    def backward(self, inputs, output_errors):
        return np.multiply(1 - np.multiply(self.outputs, self.outputs), output_errors)

    def update_parameters(self, learning_rate, momentum):
        pass
