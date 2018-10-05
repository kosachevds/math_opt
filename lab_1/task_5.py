import methods
import numericmethods as nm
import numpy as np
from matplotlib import pyplot as pp


def main():
    x_min = 0
    x_list = np.linspace(x_min - 2, x_min + 2)
    pp.plot(x_list, function(x_list))
    eps = 1e-5
    for x_0 in np.linspace(x_min - 2, x_min + 2):
        # result = methods.nuton(derivative1, derivative2, x_0, eps)[0]
        result = nm.nuton(function, x_0, eps)[0]
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