import numpy as np

from layer_interface import LayerInterface

class FullyConnected(LayerInterface):
    """docstring for Full"""
    def __init__(self, inputs_no, outputs_no, transfer_function):
        self.inputs_no = inputs_no
        self.outputs_no = outputs_no
        self.f = transfer_function

        # Xavier Initialize Parameters
        self.weights = np.random.normal(0, np.sqrt(2.0 / (inputs_no + outputs_no)),
            (self.outputs_no, self.inputs_no))

        self.biases = np.random.normal(0, np.sqrt(2.0 / (inputs_no + outputs_no)),
            (self.outputs_no, 1))

        self.a = np.zeros((self.outputs_no, 1))
        self.outputs = np.zeros((self.outputs_no, 1))

        # Gradients
        self.g_weights = np.zeros((self.outputs_no, self.inputs_no))
        self.g_biases = np.zeros((self.outputs_no, 1))

    def forward(self, inputs):
        self.a = np.dot(self.weights, inputs) + self.biases
        sefl.outputs =  self.f(self.a)

        return self.outputs

    def backward(self, inputs, output_errors):
        z = inputs
        fd_value = self.f(z, True)
        delta = np.dot(self.weights.T, output_errors) * fd_value
        self.g_biases = delta
        self.g_weights = np.dot(inputs, output_errors.T).T

        return delta

    def update_parameters(self, learning_rate):
        self.biases -= self.g_biases * learning_rate
        self.weights -= self.g_weights * learning_rate

    def to_string(self):
        return "[FC (%s -> %s)]" % (self.inputs_no, self.outputs_no)
