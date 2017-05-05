import numpy as np

from layer_interface import LayerInterface
from utils import im2col_indices, im2col, col2im

class ConvolutionalLayer(LayerInterface):

    def __init__(self, inputs_depth, inputs_height, inputs_width, outputs_depth, k, stride):
        # Number of inputs, number of outputs, filter size, stride

        self.inputs_depth = inputs_depth
        self.inputs_height = inputs_height
        self.inputs_width = inputs_width

        self.k = k
        self.stride = stride

        self.outputs_depth = outputs_depth
        self.outputs_height = int((self.inputs_height - self.k) / self.stride + 1)
        self.outputs_width = int((self.inputs_width - self.k) / self.stride + 1)


        # Layer's parameters Xavier method
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + self.inputs_depth + self.k + self.k)),
            (self.outputs_depth, self.inputs_depth, self.k, self.k)
        )
        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + 1)),
            (self.outputs_depth, 1)
        )

        # Computed values
        self.outputs = np.zeros((self.outputs_depth, self.outputs_height, self.outputs_width))

        # Gradients
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)
        self.v_weights = np.zeros(self.weights.shape)

        # im2col indicies
        self.X_col_indices = im2col_indices(
            np.arange(inputs_depth * inputs_height * inputs_width).reshape(
                (inputs_depth, inputs_height, inputs_width)),
            (self.k, self.k),
            self.stride)

    def forward(self, inputs):
        assert(inputs.shape == (self.inputs_depth, self.inputs_height, self.inputs_width))

        self.X_col = im2col(inputs, self.X_col_indices)
        W_row = self.weights.reshape((self.outputs_depth, self.k * self.k * self.inputs_depth))
        res = np.dot(W_row, self.X_col.T) + self.biases
        self.outputs = res.reshape((self.outputs_depth, self.outputs_height, self.outputs_width))

        return self.outputs

    def backward(self, inputs, output_errors):

        # Biases gradients
        self.g_biases += output_errors.sum(axis=(1,2)).reshape(output_errors.shape[0], 1)

        # Weights gradients
        erros_reshaped = output_errors.reshape(self.outputs_depth, -1)
        self.g_weights = np.dot(erros_reshaped, self.X_col)
        self.g_weights = self.g_weights.reshape(self.weights.shape)

        # Inputs gradients
        W_reshaped = self.weights.reshape(self.outputs_depth, -1)
        g_input_col = np.dot(W_reshaped.T, erros_reshaped)
        g_input = col2im(g_input_col, inputs.shape, self.X_col_indices.T)

        return g_input

    def update_parameters(self, learning_rate, momentum):
        self.biases -= self.g_biases * learning_rate
        self.weights -= self.g_weights * learning_rate

    def zero_gradients(self):
        # Gradients
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)
        self.v_weights = np.zeros(self.weights.shape)
        self.v_biases = np.zeros(self.biases.shape)



    def to_string(self):
        return "[C ((%s, %s, %s) -> (%s, %s ) -> (%s, %s, %s)]" % (self.inputs_depth, self.inputs_height, self.inputs_width, self.k, self.stride, self.outputs_depth, self.outputs_height, self.outputs_width)
