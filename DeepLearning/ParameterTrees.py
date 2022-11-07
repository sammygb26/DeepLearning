import numpy as np


class ParameterTreeIterator:
    def __init__(self, pt):
        self.upper = []
        self.upper_names = []
        self.at = (pt.keys(), pt)

    def __next__(self):
        val = None
        names, pt = self.at

        while type(pt) == ParameterTree and len(names) != 0 and val is None:
            name = names.pop(0)

            if type(pt[name]) != ParameterTree:
                val_at = self.upper_names + [name]
                val = pt[name]
            else:
                self.upper.append(self.at)
                self.upper_names.append(name)
                pt = pt[name]
                names = pt.keys()
                self.at = (names, pt)

        while len(names) == 0 and self.upper:
            self.at = self.upper.pop(-1)
            self.upper_names.pop(-1)
            names, pt = self.at

        if len(names) == 0 and val is None:
            raise StopIteration

        return val_at, val

class ParameterTree:
    def __init__(self, parameter_dict):
        if type(parameter_dict) != dict:
            raise ValueError('parameter_dict argument to ParameterTree isn\'t a of type dict')
        self.parameter_dict = parameter_dict


    def __iter__(self):
        return ParameterTreeIterator(self)

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f'{key} not present in {type(self)} {self}')
        if type(key) == str:
            self.parameter_dict[key] = value
        elif type(key) == list:
            at = self
            for name in key:
                if key.index(name) == len(key) - 1:
                    at[name] = value
                else:
                    at = at[name]


    def __getitem__(self, item):
        if type(item) == str:
            at = self.parameter_dict[item]
        elif type(item) == list:
            at = self
            for name in item:
                at = at.parameter_dict[name]

        return at

    def __contains__(self, item):
        if type(item) == str:
            return item in self.parameter_dict
        elif type(item) == list:
            at = self
            for name in item:
                if name in at:
                    at = at[name]
                else:
                    return False
            return True
        return False

    def math_op(self, other, f):
        result = self.copy()

        if type(other) == ParameterTree:
            is_tree = True
        elif type(other) is int or type(other) is float or type(other) is np.ndarray:
            is_tree = False
        else:
            raise ValueError(f'adding object of type {type} to {type(self)}')

        for key, v in self:
            if is_tree:
                if key not in other:
                    raise ValueError(f'adding two {type(self)} with different structures differing with key {key}')
                result[key] = f(self[key], other[key])
            else:
                result[key] = f(self[key],  other)

        return result

    def __add__(self, other):
        return self.math_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.math_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.math_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.math_op(other, lambda x, y: x / y)

    def copy(self):
        parameter_dict_clone = {}
        for name in self.keys():
            sub = self[name]
            if type(sub) == ParameterTree:
                parameter_dict_clone[name] = sub.copy()
            elif type(sub) == float or type(sub) == int:
                parameter_dict_clone[name] = sub
            else:
                parameter_dict_clone[name] = sub.copy()
        return ParameterTree(parameter_dict_clone)

    def keys(self):
        return sorted(self.parameter_dict.keys())

    def flatten(self):
        arr = []
        for _, v in self:
            if type(v) is int or type(v) is float:
                arr.append(v)
            else:
                arr += v.flatten().tolist()
        return np.array(arr)

