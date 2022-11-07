import numpy as np

class Activation:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative

    def f(self, z):
        return self.function(z)

    def df(self, z):
        return self.derivative(z)

linear = Activation(
    'Linear',
    lambda z: z,
    lambda z: 1.0
)

relu = Activation(
    'ReLU',
    lambda z: np.maximum(z, 0),
    lambda z: (z > 0.0)
)

lrelu = Activation(
    'LReLU',
    lambda z: np.maximum(z, z * 0.01),
    lambda z: 0.01 * (z <= 0.0) + 1.0 * (z > 0.0)
)

sigmoid = Activation(
    'Sigmoid',
    lambda z: 1.0 / (1.0 + np.exp(-z)),
    lambda z: (1.0 / (1.0 + np.exp(-z))) * (1.0 - (1.0 / (1.0 + np.exp(-z))))
)

tanh = Activation(
    'tanh',
    lambda z: np.tanh(z),
    lambda z: 1.0 - np.power(np.tanh(z), 2.0)
)