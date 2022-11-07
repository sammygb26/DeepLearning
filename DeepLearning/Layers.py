import numpy as np
from DeepLearning import Activations
from DeepLearning.ParameterTrees import ParameterTree

class Layer:
    def __init__(self, n, activation=Activations.relu):
        self.n_out = n
        self.activation = activation

        self.name = None
        self.n_in = None
        self.b = None
        self.w = None
        self.z = None
        self.a_in = None

    def __str__(self):
        return f'<{self.name} {self.n_in}->{self.n_out} ({(self.n_in + 1) * self.n_out} parameters)>'

    def compile(self,name, n_in):
        self.name = name
        self.n_in = n_in

        self.w = np.random.rand(self.n_out, self.n_in) / self.n_out
        self.b = np.zeros((self.n_out, 1))

    def forward(self, a_in):
        self.a_in = a_in

        z = np.dot(self.w, a_in) + self.b
        self.z = z

        a = self.activation.f(z)
        return a

    def backward(self, da_out):
        m = max(da_out.shape[1], 1)

        dz = self.activation.df(self.z) * da_out

        dw = (1/m) * np.dot(dz, self.a_in.transpose())
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        da_in = np.dot(self.w.transpose(), da_out)

        d = ParameterTree({'w': dw, 'b': db})

        return d, da_in

    def get_trainable_variables(self):
        return ParameterTree({'w': self.w, 'b':self.b})

    def set_trainable_variables(self, pt):
        self.w = pt['w']
        self.b = pt['b']

    def apply_gradients(self, d):
        dw = d['w']
        db = d['b']

        self.w -= dw
        self.b -= db


