import numpy as np
from matplotlib import pyplot as pp


def main():
    epsilon = 1e-4
    x_list = np.linspace(1, 12, 100)
    pp.subplot(2, 1, 1)
    pp.xlabel("X")
    pp.ylabel("f(X)")
    pp.plot(x_list, function1(x_list))
    for x in enumerative(function1, x_list[0], x_list[-1], epsilon):
        pp.plot(x, function1(x), marker="o", color="g")

    x_list = np.linspace(0, 4, 100)
    pp.subplot(2, 1, 2)
    pp.xlabel("X")
    pp.ylabel("f(X)")
    pp.plot(x_list, function2(x_list))
    for x in enumerative(function2, x_list[0], x_list[-1], epsilon):
        pp.plot(x, function2(x), marker="o", color="g")

    pp.show()


def function1(x):
    return np.cos(x) / x ** 2


def function2(x):
    return 0.1 * x + 2 * np.sin(4 * x)


def enumerative(function, begin, end, eps):
    previous = function(begin)
    upward = False
    result = []
    for x in np.arange(begin + eps, end, eps):
        current = function(x)
        if upward:
            upward = current > previous
        elif current > previous:
            result.append(x)
            upward = True
        previous = current
    return result


if __name__ == '__main__':
    main()

