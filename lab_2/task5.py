import numpy as np
from matplotlib import pyplot as pp
import methods

X_0 = np.array([-1, 1])


def main():
    eps_list = np.logspace(-3, -5, 2)
    onedim_list = np.logspace(-4, -8, 3)
    for i, eps in enumerate(eps_list):
        pp.figure(i - 1)
        pp.title("eps = %f" % eps)
        pp.ylabel("count")
        pp.xlabel("onedim eps")
        labels_counts = {}

        counts = [methods.steepest_descent(resenbrock, rosenbrock_gradient,
                                           X_0, eps, onedim_eps)[1]
                  for onedim_eps in onedim_list]
        labels_counts["Steepest descent"] = counts

        counts = [methods.conjugate_gradient(resenbrock, rosenbrock_gradient,
                                             X_0, eps, onedim_eps[1])
                  for onedim_eps in onedim_list]
        labels_counts["Conj. gradients"] = counts

        counts = [methods.alternating_variable(resenbrock, X_0, eps, eps_1d)
                  for eps_1d in onedim_list]
        labels_counts["Alter. variables"]

        for label, counts in labels_counts:
            pp.plot(np.log10(onedim_list), counts, label=label)
        pp.gca().invert_axis()
        pp.legend()



def resenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


def rosenbrock_gradient(x):
    return _np.array([400 * (x[0] ** 2 - x[1])])


if __name__ == "__main__":
    main()
