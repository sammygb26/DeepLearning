import numpy as np
from tqdm import tqdm
import DeepLearning.Losses as losses
import matplotlib.pyplot as plt

def grad_check(model, x, y, loss=losses.squared_loss, epsilon=1e-7):
    m = max(x.shape[1], 1)
    theta = model.get_trainable_variables()

    yhat = model.forward(x)
    da_out = loss.df(y, yhat)
    delta, _ = model.backward(da_out)

    delta_approx = delta.copy()
    for key, v in delta_approx:
        with tqdm(np.ndenumerate(v), desc=f'grad_check on {key}', total=v.size) as t:
            for idx, _ in t:
                t.set_postfix(at=idx)

                t_theta = theta.copy()
                t_theta[key][idx] += epsilon
                model.set_trainable_variables(t_theta)
                yhat1 = model.forward(x)

                t_theta = theta.copy()
                t_theta[key][idx] -= epsilon
                model.set_trainable_variables(t_theta)
                yhat2 = model.forward(x)

                l1 = loss.f(y, yhat1)
                l2 = loss.f(y, yhat2)

                d = (l1 - l2) / (2.0 * epsilon)
                delta_approx[key][idx] = (1/m) * np.sum(d, axis=1)

                if np.abs(delta_approx[key][idx] - delta[key][idx]) > 10.0 * epsilon:

                    print(f'{key} {idx} -> '
                          f'd {delta[key][idx]} '
                          f'da {delta_approx[key][idx]} - {delta_approx[key][idx] - delta[key][idx]}')
                    delta_approx[key][idx] = delta[key][idx]


                    #plt.plot(x.flatten(), y.flatten(), color='blue', linestyle='--')
                    #plt.plot(x.flatten(), yhat1.flatten(), color='red')
                    #plt.plot(x.flatten(), yhat2.flatten(), color='green')
                    #plt.show()

    delta_f = delta.flatten()
    delta_approx_f = delta_approx.flatten()
    difference = (delta_approx - delta).flatten()

    score = np.linalg.norm(difference) / (np.linalg.norm(delta_f) + np.linalg.norm(delta_approx_f))

    return delta, delta_approx, score


