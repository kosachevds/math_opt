import numpy as np
from matplotlib import pyplot as pp


def minimize(func, begin, end, eps):
    count = 0
    # count = 2
    # if func(begin) < func(begin + eps):
    #     return begin, count
    while abs(end - begin) > eps:
        step = (end - begin) / 4.0
        x_prev = begin
        previous = func(begin)
        count += 1
        for x in np.linspace(begin + step, end, num=4):
            current = func(x)
            count += 1
            if current < previous:
                previous = current
                x_prev = x
                continue
            begin = x
            end = x_prev
            break
        if x == end:
            return x, count
    return (begin + end) / 2.0, count


def numeric_nuton(func, x_0, eps):
    x = x_0
    count = 0
    while True:
        f_1 = func(x + eps)
        f_2 = func(x - eps)
        count += 2
        dx1 = (f_1 - f_2) / (2 * eps)
        if abs(dx1) <= eps:
            return (x, count)
        f_0 = func(x)
        count += 1
        if count > 1000:
            return (None, None)
        dx2 = (f_1 - 2 * f_0 + f_2) / (eps * eps)
        if dx2 == 0:
            return (None, None)
        x = x - dx1 / dx2


def _plot_func(func, begin, end):
    x_list = np.linspace(begin, end, num=100)
    f_list = [func(x) for x in x_list]
    pp.plot(x_list, f_list)
    pp.show()
