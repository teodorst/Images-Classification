import numpy as np
import math

from layer_interface import LayerInterface

class MaxPoolingLayer(LayerInterface):

    def __init__(self, stride):
        # Dimensions: stride
        self.stride = stride

        # # indexes of max activations
        # self.switches = {}

    def forward(self, inputs):
        (d, w, h) = inputs.shape
        s = self.stride
        width = int(w / s)
        height = int(h / s)
        self.outputs = inputs.reshape(d, width, s, height, s).max(axis=(2, 4))
        return self.outputs

    def backward(self, inputs, output_errors):
        (d, w, h) = inputs.shape

        s = self.stride
        result = np.zeros(inputs.shape)

        for i in range(d):
            for j in range(w):
                for k in range(h):
                    if inputs[i, j, k] == output_errors[i, j // 2, k // 2]:
                        result[i, j, k] = inputs[i, j, k]

        return result

    def to_string(self):
        return "[MP (%s x %s)]" % (self.stride, self.stride)

