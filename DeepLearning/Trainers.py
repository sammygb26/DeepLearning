import numpy as np
from tqdm import trange
import time

class GradientDescent:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, model, gradients):
        theta = model.get_trainable_variables()
        theta -= gradients * self.alpha
        model.set_trainable_variables()

def optimize(x, y, model, trainer, loss):
    yhat = model.forward(x)
    da_out = loss.df(y, yhat)
    delta, _ = model.backward(da_out)
    trainer(model, delta)

    return loss.f(y, yhat)

def get_batches(x, y, batch_size):
    p = np.random.permutation(x.shape[1])

    batch_cuts = range(batch_size, x.shape[1], batch_size)

    x_batches = np.split(x[:, p], indices_or_sections=batch_cuts, axis=1)
    y_batches = np.split(y[:, p], indices_or_sections=batch_cuts, axis=1)

    return zip(x_batches, y_batches)

def gradientDescent(model, x, y, loss, alpha=0.01, B=1024):
    return stochasticGradientDescent(model, x, y, loss=loss, batch_size=x.shape[1], alpha=alpha, B=B)



def stochasticGradientDescent(model, x, y, loss, batch_size=1024, alpha=0.001, B=1024):
    cost_record = []
    time_record = []
    start = time.time()

    with trange(B) as t:
        for _ in t:
            for x_batch, y_batch in get_batches(x, y, batch_size):
                yhat = model.forward(x_batch)

                cost = loss.f(y, yhat)
                t.set_postfix(loss=cost)

                cost_record.append(cost)
                time_record.append(time.time() - start)

                da_out = loss.df(y_batch,yhat)
                delta, _ = model.backward(da_out)

                model.apply_gradients(delta * alpha)

    return cost_record, time_record

def momentum(model, x, y, loss, batch_size=1024, alpha=0.001, beta=0.9, B=1024):
    v = model.get_trainable_variables() * 0.0
    cost_record = []
    time_record = []
    start = time.time()

    with trange(B) as t:
        for _ in t:
            for x_batch, y_batch in get_batches(x, y, batch_size):
                yhat = model.forward(x_batch)

                cost = np.average(loss.f(y_batch, yhat), axis=1)
                t.set_postfix(loss=cost)

                cost_record.append(cost)
                time_record.append(time.time() - start)

                da_out = loss.df(y_batch, yhat)
                delta, _ = model.backward(da_out)

                v = (v * beta) + (delta * (1-beta))

                model.apply_gradients(v * alpha)

    return cost_record, time_record


def RMSprop(model, x, y, loss, batch_size=1024, alpha=0.001, beta=0.99, B=1024):
    pass
