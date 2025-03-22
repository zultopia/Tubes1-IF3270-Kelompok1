import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation, initializer):
        self.weights = initializer((input_dim, output_dim))
        self.biases = np.zeros((1, output_dim))
        self.activation = activation
    
    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output