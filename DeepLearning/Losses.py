import numpy as np


class Loss:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def f(self, y, yhat):
        return self.function(y, yhat)

    def df(self, y, yhat):
        return self.derivative(y, yhat)


binary_cross_entropy = Loss(
    lambda y, yhat: -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)),
    lambda y, yhat: -(yhat-y)/((yhat-y) * yhat)
)

squared_loss = Loss(
    lambda y, yhat: np.sum(np.power(y - yhat, 2.0), axis=0, keepdims=True),
    lambda y, yhat: 2 * np.sum(yhat - y, axis=0, keepdims=True)
)
