import numpy
from matplotlib import pyplot as pp


def minimize(func, begin, end, eps):
    # if __debug__:
    #     points = numpy.linspace(begin - end, end)
    #     pp.plot(points, [func(x) for x in points])
    #     pp.show()
    count = 2
    if func(begin) < func(begin + eps):
        return begin, count
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = func(begin)
        count += 1
        for x in numpy.linspace(begin + step, end, num=4):
            current = func(x)
            count += 1
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
    return (begin + end) / 2.0, count
