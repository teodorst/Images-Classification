import numpy as np
from time import sleep

from layer_interface import LayerInterface

class FeedForward:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        # print inputs
        last_input = inputs
        cnt = 0
        for layer in self.layers:
            # print cnt
            cnt += 1
            last_input = layer.forward(last_input)
        # print 'Feed Forward'
        # print last_input
        return last_input

    def backward(self, inputs, output_errors):
        crt_error = output_errors
        cnt = 0
        for layer_no in range(len(self.layers)-1, 0, -1):
            # print cnt
            cnt += 1
            crt_layer = self.layers[layer_no]
            prev_layer = self.layers[layer_no-1]
            crt_error = crt_layer.backward(prev_layer.outputs, crt_error)
        self.layers[0].backward(inputs, crt_error)

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def zero_gradients(self):
        for layer in self.layers:
            layer.zero_gradients()

    def to_string(self):
        return " -> ".join(map(lambda l: l.to_string(), self.layers))

