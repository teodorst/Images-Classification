import numpy as np
from time import sleep

from layer_interface import LayerInterface
from transfer_functions import tanh

class Tanh(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = tanh(inputs)

        # print 'Tanh Forward'
        # print self.outputs
        # sleep(1)
        return self.outputs

    def backward(self, inputs, output_errors):
        # print output_errors.shape
        # print np.multiply(1 - np.multiply(self.outputs, self.outputs), output_errors).shape
        return np.multiply(1 - np.multiply(self.outputs, self.outputs), output_errors)

    def update_parameters(self, learning_rate):
        pass
