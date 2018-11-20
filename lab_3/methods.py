import numpy as _np


def enumerative(x_list, y_list, z_list):
    index = _np.argmin(z_list)
    return (x_list[index], y_list[index])


def random_search(x_list, y_list, z_list, x_0, y_0):
    pass


def pattern_search(x_list, y_list, z_list, x_0, y_0):
    pass


def simulated_annealing(x_list, y_list, z_list, x_0, y_0):
    pass


def genetic_search(x_list, y_list, z_list, x_0, y_0):
    pass
