import numpy


def minimize(func, begin, end, eps):
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = func(begin)
        for x in numpy.linspace(begin + step, end, num=4):
            current = func(x)
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
    return (begin + end) / 2.0
