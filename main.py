from typing import Tuple, Any

import DeepLearning as dl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

model = dl.Models.Sequential([
    dl.Layers.Layer(256, activation=dl.Activations.tanh),
    dl.Layers.Layer(256, activation=dl.Activations.tanh),
    dl.Layers.Layer(256, activation=dl.Activations.tanh),
    dl.Layers.Layer(1, activation=dl.Activations.linear)
])

model.compile(1)

x = np.array([np.arange(-3,3,0.1)])
y = np.power(x, 3) + np.power(x, 2) + np.sin(x*4)

theta = model.get_trainable_variables()
model.set_trainable_variables(theta)

while True:
    cost1, time1 = dl.Trainers.momentum(model, x, y, loss=dl.Losses.squared_loss, alpha=0.005, B=1024, batch_size=128)

    print(f'cost1 -> {cost1}')

    plt.plot(time1, cost1, color='red')
    plt.show()

    yhat = model.forward(x)

    plt.plot(x.flatten(), y.flatten(), color='blue', linestyle='--')
    plt.plot(x.flatten(), yhat.flatten(), color='red')
    plt.show()
