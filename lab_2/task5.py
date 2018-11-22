import numpy as np
from matplotlib import pyplot as pp
import methods

X_0 = np.array([-1, 1])
X_MIN = np.array([1, 1])


def main():
    eps_list = np.logspace(-3, -5, 2)
    eps_1d_list = np.logspace(-4, -8, 5)
    # plot_counts(eps_list, eps_1d_list)
    plot_errors(eps_list, eps_1d_list)

    # TODO: передавать X_MIN в метод;
    # останавливать работу если достигнута необходимая точность до X_MIN
    # построить графики counts для этого
    # TODO: определить, для каких eps метод не достигает заданной точности

    pp.show()


def plot_counts(eps_list, eps_1d_list):
    for i, eps in enumerate(eps_list):
        eps_log = np.log10(eps)
        pp.figure(i - 1)
        pp.title("eps^(%f)" % eps_log)
        pp.ylabel("count")
        pp.xlabel("onedim eps")

        plot_method_counts(methods.steepest_descent, True, eps, eps_1d_list,
                           "Steepest descent")
        plot_method_counts(methods.conjugate_gradient, True, eps, eps_1d_list,
                           "Conj. gradients")
        plot_method_counts(methods.alternating_variable, False, eps,
                           eps_1d_list, "Alter. variables")

        pp.gca().invert_xaxis()
        pp.legend()


def plot_errors(eps_list, eps_1d_list):
    for i, eps in enumerate(eps_list):
        eps_log = np.log10(eps)
        pp.figure(i - 1)
        pp.title("eps^(%f)" % eps_log)
        pp.ylabel("error")
        pp.xlabel("onedim eps")

        plot_method_error(methods.steepest_descent, True, eps, eps_1d_list,
                          "Steepest descent")
        plot_method_error(methods.conjugate_gradient, True, eps, eps_1d_list,
                          "Conj. gradients")
        plot_method_error(methods.alternating_variable, False, eps,
                          eps_1d_list, "Alter. variables")

        pp.gca().invert_xaxis()
        pp.legend()


def plot_method_counts(method, with_gradient, eps, eps_1d_list, label=None):
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


def plot_method_error(method, with_gradient, eps, eps_1d_list, label=None):
    if with_gradient:
        x_list = [method(rosenbrock, rosenbrock_gradient, X_0, eps, eps_1d)[0]
                  for eps_1d in eps_1d_list]
    else:
        x_list = [method(rosenbrock, X_0, eps, eps_1d)[0]
                  for eps_1d in eps_1d_list]
    errors = np.log10([np.linalg.norm(X_MIN - x_min) for x_min in x_list])
    pp.plot(np.log10(eps_1d_list), errors, label=label)


def rosenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


def rosenbrock_gradient(x):
    return np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1),
                     -200 * (x[0] ** 2 - x[1])])


if __name__ == "__main__":
    main()
