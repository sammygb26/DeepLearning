from abc import ABC
import abc
import numpy as np
from DeepLearning.ParameterTrees import ParameterTree

class Sequential:
    def __init__(self, layers):
        self.layers = layers

        self.layer_indices = None
        self.n_in = None

    def compile(self, n_in):
        self.n_in = n_in
        self.layer_indices = {}

        for layer in self.layers:
            layer.compile(f'dense{self.layers.index(layer)}', n_in)
            self.layer_indices[layer.name] = self.layers.index(layer)

            n_in = layer.n_out

    def forward(self, a_in):
        for layer in self.layers:
            a_in = layer.forward(a_in)
        return a_in

    def backward(self, da_out):
        delta = {}
        for layer in reversed(self.layers):
            d, da_out = layer.backward(da_out)
            delta[layer.name] = d
        return ParameterTree(delta), da_out

    def get_trainable_variables(self):
        theta = {}
        for layer in self.layers:
            t = layer.get_trainable_variables()
            theta[layer.name] = t
        return ParameterTree(theta)

    def set_trainable_variables(self, theta):
        for name in theta.keys():
            t = theta[name]
            layer = self.layers[self.layer_indices[name]]
            layer.set_trainable_variables(t)

    def apply_gradients(self, delta=ParameterTree({})):
        for layer_name in delta.keys():
            d = delta[layer_name]
            layer = self.layers[self.layer_indices[layer_name]]
            layer.apply_gradients(d)

