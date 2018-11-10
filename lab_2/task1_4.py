import numpy as np
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import methods


def _main():
    plot_task2_function(0, 200, 1000)


def task2(x0, first_fig, a_list, eps_list):
    x0 = np.array(x0)
    for i, a in enumerate(a_list):
        def func_wrap(x):
            return task2_function(x[0], x[1], a)

        def gradient_wrap(x):
            return task2_gradient(x, a)

        def hessian_wrap(x):
            return task2_hessian(x, a)

        pp.figure(first_fig + i)

        def get_counts(method):
            return [method(func_wrap, gradient_wrap, x0, eps)[1]
                    for eps in eps_list]

        pp.plot(get_counts(methods.steepest_descent), label="Steepest descent")
        pp.plot(get_counts(methods.conjugate_gradient), label="Conj. gradient")
        pp.plot([methods.nuton(gradient_wrap, hessian_wrap, x0, eps)
                 for eps in eps_list],
                label="Nuton")

        def get_counts_2(method):
            return [method(func_wrap, x0, eps) for eps in eps_list]

        pp.plot(get_counts_2(methods.regular_simplex), label="Regular simplex")
        pp.plot(get_counts_2(methods.alternating_variable),
                label="Altarnating variables")
        pp.plot(get_counts_2(methods.hooke_jeeves), label="Hooke-Jeeves")
        pp.plot(get_counts_2(methods.random_search), label="Random Search")


def plot_task2_function(fig_id, x1_limit, a_param):
    pp.figure(fig_id)
    axes = pp.axes(projection="3d")
    x_1 = np.linspace(-x1_limit, x1_limit)
    x_2 = np.linspace(-3, 3)
    x_1, x_2 = np.meshgrid(x_1, x_2)
    z = task2_function(x_1, x_2, a_param)
    # axes.contour3D(x_1, x_2, z, 64)
    axes.plot_wireframe(x_1, x_2, z)
    axes.set_xlabel("x1")
    axes.set_ylabel("x2")
    axes.set_zlabel("f(x1, x2)")
    axes.set_title("f(x1, x2) = x1 ^ 2 + {0} * x2 ^ 2".format(a_param))
    pp.show()


def task2_function(x1, x2, a):
    return x1 ** 2 + a * x2 ** 2


def task2_gradient(x, a):
    return np.array([2 * x[0], 2 * a * x[1]])


def task2_hessian(x, a):
    return np.array([[2, 0], [0, 2 * a]])


def task3_function(x1, x2):
    return (151 * x1 ** 2 - 300 * x1 * x2 + 151 * x2 ** 2 + 33 * x1 +
            99 * x2 + 48)

if __name__ == "__main__":
    _main()
