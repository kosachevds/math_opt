from matplotlib import pyplot as pp
from methods import *
import numericmethods as nm
import numpy as np

X_BEGIN = -1.0
X_END = 1.5


def main():
    # pp.figure(0)
    # x_list = np.linspace(X_BEGIN, X_END)
    # pp.plot(x_list, function(x_list))
    # pp.xlabel("x")
    # pp.ylabel("f(x)")
    # pp.title("function")

    # plot_perfomance(1, 2)

    plot_numeric_vs_analytic(3)

    pp.show()


def get_count(func, eps):
    return func(function, X_BEGIN, X_END, eps)[1]


def plot_perfomance(figure1, figure2):
    eps_list = np.linspace(1e-2, 1e-4, num=30)
    pp.figure(figure1)
    counts = [get_count(enumerative, eps) for eps in eps_list]
    pp.plot(eps_list, counts)
    pp.ylabel("N")
    pp.xlabel("epsilon")
    pp.title("perfomance: enumerative")
    pp.gca().invert_xaxis()

    pp.figure(figure2)
    counts = [get_count(radix, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="radix")

    counts = [get_count(dichotomy, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="dichotomy")

    counts = [get_count(golden_ratio, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="golden_ratio")

    counts = [get_count(parabolic, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="parabolic")

    counts = [middle_point(derivative1, X_BEGIN, X_END, eps)[1]
              for eps in eps_list]
    pp.plot(eps_list, counts, label="middle point")

    counts = [chords(derivative1, X_BEGIN, X_END, eps)[1]
              for eps in eps_list]
    pp.plot(eps_list, counts, label="chords")

    counts = [nuton(derivative1, derivative2, X_BEGIN, eps)[1]
              for eps in eps_list]
    pp.plot(eps_list, counts, label="nuton")

    pp.ylabel("N")
    pp.xlabel("epsilon")
    pp.title("perfomance")
    pp.legend()
    pp.gca().invert_xaxis()


def function(x):
    """ x^2 - 2x + e^(-x)
    """
    return x ** 2 - 2 * x + np.exp(-x)


def derivative1(x):
    """ 2x - 2 - x * e^(-x)
    """
    return 2 * x - 2 - x * np.exp(-x)


def derivative2(x):
    """ x^2 * e^(-x) + 2
    """
    return x ** 2 * np.exp(-x) + 2


def plot_numeric_vs_analytic(figure_id):
    columns = 3
    eps_list = np.linspace(1e-2, 1e-6, 20)

    pp.figure(figure_id)
    counts = [middle_point(derivative1, X_BEGIN, X_END, eps)[1]
              for eps in eps_list]
    pp.subplot(1, columns, 1)
    pp.plot(eps_list, counts)
    counts = [nm.middle_point(function, X_BEGIN, X_END, eps)[1]
              for eps in eps_list]
    pp.plot(eps_list, counts)
    pp.ylabel("N")
    pp.xlabel("epsilon")
    pp.title("middle point")
    pp.gca().invert_xaxis()

    counts = [chords(derivative1, X_BEGIN, X_END, eps)[1] for eps in eps_list]
    pp.subplot(1, columns, 2)
    pp.plot(eps_list, counts)
    counts = [nm.chords(function, X_BEGIN, X_END, eps)[1] for eps in eps_list]
    pp.plot(eps_list, counts)
    pp.ylabel("N")
    pp.xlabel("epsilon")
    pp.title("chords")
    pp.gca().invert_xaxis()

    counts = [nuton(derivative1, derivative2, X_BEGIN, eps)[1]
              for eps in eps_list]
    pp.subplot(1, columns, 3)
    pp.plot(eps_list, counts)
    counts = [nm.nuton(function, X_BEGIN, eps)[1] for eps in eps_list]
    pp.plot(eps_list, counts)
    pp.ylabel("N")
    pp.xlabel("epsilon")
    pp.title("nuton")
    pp.gca().invert_xaxis()


if __name__ == "__main__":
    main()
