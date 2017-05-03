import numpy as np
from time import sleep

from layer_interface import LayerInterface
from transfer_functions import tanh

class SoftMax(LayerInterface):

    def __init__(self):
        pass

    def forward(self, x):
        # print x
        exk = np.exp(x)
        exk_sum = np.sum(exk)
        self.outputs = exk / exk_sum
        return self.outputs

    def backward(self, inputs, output_errors):
        Z  = np.sum(np.multiply(output_errors, self.outputs))
        deltax = np.multiply(self.outputs, output_errors - Z)
        return deltax


    def update_parameters(self, learning_rate, momentum):
        pass
