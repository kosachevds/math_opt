import numpy as np
import methods
from matplotlib import pyplot as pp

X_BEGIN = 0
X_END = 2


def main():
    # eps_list = np.linspace(1e-2, 1e-6)
    # counts = [methods.golden_ratio(function_, X_BEGIN, X_END, eps)[1] for eps in eps_list]
    # pp.plot(eps_list, counts, label='golden ratio')
    #
    # # counts = [nm.nuton(function_, X_BEGIN, eps)[1] for eps in eps_list]
    # counts = [methods.nuton(derivative1, derivative2, X_BEGIN, eps)[1] for eps in eps_list]
    # pp.plot(eps_list, counts, label='nuton')
    # pp.ylabel("N")
    # pp.xlabel("epsilon")
    # pp.legend()
    # pp.gca().invert_xaxis()

    x_list = np.linspace(0, 2)
    pp.plot(x_list, 56 * (x_list - 1) ** 6)

    pp.show()


def function_(x):
    return (x - 1) ** 8


def derivative1(x):
    return 8 * (x - 1) ** 7


def derivative2(x):
    return 56 * (x - 1) ** 6


if __name__ == "__main__":
    main()
