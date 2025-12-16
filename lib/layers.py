import numpy as np


class Layer:
    """
    Base class for all network layers.
    """

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params(self):
        return []

    def get_grads(self):
        return []


class Dense(Layer):
    """
    Fully connected layer: y = xW + b
    """

    def __init__(self, input_dim, output_dim):
        # Xavier-like initialization
        limit = np.sqrt(1 / input_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

        self.inputs = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, inputs):
        """
        inputs: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b

    def backward(self, grad_output):
        """
        grad_output: dL/dY, shape (batch_size, output_dim)

        Computes:
        - dW = X^T * dL/dY
        - db = sum(dL/dY)
        - dX = dL/dY * W^T
        """
        batch_size = grad_output.shape[0]

    def backward(self, grad_output):
        self.dW = np.dot(self.inputs.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)

        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def get_params(self):
        return [self.W, self.b]

    def get_grads(self):
        return [self.dW, self.db]