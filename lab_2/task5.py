import numpy as np
from matplotlib import pyplot as pp
import methods

X_0 = np.array([-1, 1])
X_MIN = np.array([1, 1])


def main():
    eps_list = np.logspace(-3, -5, 2)
    onedim_list = np.logspace(-4, -8, 3)
    for i, eps in enumerate(eps_list):
        pp.figure(i - 1)
        pp.title("eps = %f" % eps)
        pp.ylabel("count")
        pp.xlabel("onedim eps")

        plot_method(methods.steepest_descent, True, eps, onedim_list,
                    "Steepest descent")
        plot_method(methods.conjugate_gradient, True, eps, onedim_list,
                    "Conj. gradients")
        plot_method(methods.alternating_variable, False, eps, onedim_list,
                    "Alter. variables")

        pp.gca().invert_xaxis()
        pp.legend()
    pp.show()


def plot_method(method, with_gradient, eps, eps_1d_list, label=None):
    if with_gradient:
        pairs = [method(rosenbrock, rosenbrock_gradient, X_0, eps, eps_1d)
                 for eps_1d in eps_1d_list]
    else:
        pairs = [method(rosenbrock, X_0, eps, eps_1d)
                 for eps_1d in eps_1d_list]
    counts = [p[1] for p in pairs]
    eps_1d_list = np.log10(eps_1d_list)
    pp.plot(eps_1d_list, counts, label=label)
    for i, eps_1d in enumerate(eps_1d_list):
        x_min_i = pairs[i][0]
        marker_format = "go"
        if np.linalg.norm(x_min_i - X_MIN) > eps:
            marker_format = "rx"
        count_i = pairs[i][1]
        pp.plot(eps_1d, count_i, marker_format)


def rosenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


def rosenbrock_gradient(x):
    return np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1),
                     -200 * (x[0] ** 2 - x[1])])


if __name__ == "__main__":
    main()
