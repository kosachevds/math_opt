import methods
import numericmethods as nm
import numpy as np
from matplotlib import pyplot as pp

# Max3 / min2 * abs(x0 - x*) < 1


def main():
    x_min = 0
    eps = 1e-5
    for i, item in enumerate(["nuton", "marquardt", "nuton-rafson"]):
        pp.subplot(1, 3, i + 1)
        pp.title(item)
        pp.xlabel("x")
        pp.ylabel("y")

    x_0 = 0
    step = 0.01
    pp.subplot(1, 3, 1)
    while True:
        result = methods.nuton(derivative1, derivative2, x_0, eps)[0]
        if result is not None:
            pp.plot(x_0, function(x_0), "go", markersize=2)
            x_0 += step
            continue
        pp.plot(x_0, function(x_0), "rx")
        x_list = np.linspace(x_min, x_0 + step)
        pp.plot(x_list, function(x_list))
        pp.title("nuton: " + str(x_0))
        break

    x_0 = 0
    pp.subplot(1, 3, 2)
    while True:
        result = methods.marquardt(derivative1, derivative2, x_0, eps)[0]
        if result is not None:
            pp.plot(x_0, function(x_0), "go", markersize=2)
            x_0 += step
            continue
        pp.plot(x_0, function(x_0), "rx")
        x_list = np.linspace(x_min, x_0 + step)
        pp.plot(x_list, function(x_list))
        pp.title("marquardt: " + str(x_0))
        break

    x_0 = 0
    pp.subplot(1, 3, 3)
    while True:
        result = methods.nuton_rafson(derivative1, derivative2, x_0, eps)[0]
        if result is not None:
            pp.plot(x_0, function(x_0), "go", markersize=2)
            x_0 += step
            continue
        pp.plot(x_0, function(x_0), "rx")
        x_list = np.linspace(x_min, x_0 + step)
        pp.plot(x_list, function(x_list))
        pp.title("nuton-rafson: " + str(x_0))
        break

    pp.show()


def function(x):
    return np.arctan(x) * x - np.log(x ** 2 + 1) / 2


def derivative1(x):
    return np.arctan(x)


def derivative2(x):
    return 1 / (1 + x ** 2)


if __name__ == '__main__':
    main()
