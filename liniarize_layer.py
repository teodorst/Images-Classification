import numpy as np
from time import sleep

from layer_interface import LayerInterface

class LinearizeLayer(LayerInterface):

    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth


    def forward(self, inputs):
        self.outputs = np.matrix(inputs.flatten()).T
        # print 'Liniarized Forward'
        # print self.outputs
        # sleep(1)
        return self.outputs

    def backward(self, inputs, output_errors):
        return np.reshape(np.array(output_errors), (self.height, self.width, self.depth))

    def update_paramters(self, learning_rate):
        pass

    def to_string(self):
        return "[Lin ((%s, %s, %s) -> %s)]" % (self.depth, self.height, self.width, self.depth * self.height * self.width)


class LinearizeLayerReverse(LayerInterface):
    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth

    def forward(self, inputs):
        self.outputs = np.reshape(np.array(inputs), (self.height, self.width, self.depth))
        return self.outputs

    def backward(self, inputs, output_errors):
        return np.matrix(output_errors.flatten()).T

    def update_paramters(self, learning_rate):
        pass

    def to_string(self):
        return "[Lin (%s -> (%s, %s, %s))]" % (self.depth * self.height * self.width, self.depth, self.height, self.width)

