import numpy as np
from time import sleep

from layer_interface import LayerInterface
from transfer_functions import tanh

class SoftMax(LayerInterface):

    def __init__(self):
        pass

    def forward(self, x):
        # exk = np.exp(inputs)
        # exk_sum = np.sum(exk)
        ex = np.exp(x - np.max(x))
        self.outputs = ex / ex.sum()
        # print 'Softmax Forward'
        # print self.outputs
        # sleep(1)
        return self.outputs

    def backward(self, inputs, output_errors):
        Z  = np.sum(np.multiply(output_errors, inputs))
        # print output_errors.shape
        deltax = np.multiply(self.outputs, output_errors - Z)
        # print deltax.shape
        return deltax
        # return inputs * (output_errors - inputs * output_errors)


    def update_parameters(self, learning_rate):
        pass
