import numpy as np

from layer_interface import LayerInterface

class LinearizeLayer(LayerInterface):

    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth


    def forward(self, inputs):
        return np.matrix(inputs.flatten()).T

    def backward(self, inputs, output_errors):
        return np.reshape(np.array(output_errors), (self.height, self.width, self.depth))

    def to_string(self):
        return "[Lin ((%s, %s, %s) -> %s)]" % (self.depth, self.height, self.width, self.depth * self.height * self.width)


class LinearizeLayerReverse(LayerInterface):
    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.depth = depth

    def forward(self, inputs):
        return np.reshape(np.array(inputs), (self.height, self.width, self.depth))

    def backward(self, inputs, output_errors):
        return np.matrix(output_errors.flatten()).T

    def to_string(self):
        return "[Lin (%s -> (%s, %s, %s))]" % (self.depth * self.height * self.width, self.depth, self.height, self.width)

