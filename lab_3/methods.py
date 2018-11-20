import numpy as _np


def enumerative(x_list, y_list, z_list):
    index = _np.argmin(z_list)
    return (x_list[index], y_list[index])


def simulated_annealing():
    pass
