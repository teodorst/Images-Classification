import numpy as np
from transfer_functions import relu

from layer_interface import LayerInterface

class ReluLayer(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        # TODO 2
        self.outputs = relu(inputs)

        return self.outputs

    def backward(self, inputs, output_errors):
        # TODO 2
        return output_errors * relu(inputs, True)

    def to_string(self):
        return "[Relu]"
