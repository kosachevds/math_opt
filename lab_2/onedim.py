import numpy as _np
from matplotlib import pyplot as _pp


def minimize(func, begin, end, eps):
    count = 2
    if func(begin) < func(begin + eps):
        return begin, count
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = func(begin)
        count += 1
        for x in _np.linspace(begin + step, end, num=4):
            current = func(x)
            count += 1
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
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
        x = x - dx1 / dx2

# def _find_end(func, begin):
