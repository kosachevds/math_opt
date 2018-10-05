import methods
import numericmethods as nm
import numpy as np
from matplotlib import pyplot as pp

# Max3 / min2 * abs(x0 - x*) < 1


def main():
    x_min = 0
    x_begin = x_min - 5
    x_end = x_min + 5
    x_list = np.linspace(x_begin, x_end)
    for i, item in enumerate(["nuton", "marquardt", "nuton-rafson"]):
        pp.subplot(1, 3, i + 1)
        pp.plot(x_list, function(x_list))
        pp.title(item)
        pp.xlabel("x")
        pp.ylabel("y")
    eps = 1e-5
    for x_0 in x_list:
        pp.subplot(1, 3, 1)
        # result = nm.nuton(function, x_0, eps)[0]
        result = methods.nuton(derivative1, derivative2, x_0, eps)[0]
        if result is None:
            pp.plot(x_0, function(x_0), marker="x", color="r")
        else:
            pp.plot(x_0, function(x_0), marker="o", color="g")

        pp.subplot(1, 3, 2)
        result = methods.marquardt(derivative1, derivative2, x_0, eps)[0]
        if result is None:
            pp.plot(x_0, function(x_0), marker="x", color="r")
        else:
            pp.plot(x_0, function(x_0), marker="o", color="g")

        pp.subplot(1, 3, 3)
        result = methods.nuton_rafson(derivative1, derivative2, x_0, eps)[0]
        if result is None:
            pp.plot(x_0, function(x_0), marker="x", color="r")
        else:
            pp.plot(x_0, function(x_0), marker="o", color="g")
    pp.show()


def function(x):
    return np.arctan(x) * x - np.log(x ** 2 + 1) / 2


def derivative1(x):
    return np.arctan(x)


def derivative2(x):
    return 1 / (1 + x ** 2)


if __name__ == '__main__':
    main()
