import numpy as _np
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import methods


def _main():
    # plot_task2_function(0, 200, 1000)
    task2([10, 10], 0, [1, 250, 1000], _np.logspace(-3, -5, 2))

    # plot_task3_function(0)
    # task3([0, 0], 0, _np.logspace(-3, -5, 3))


def task2(x0, first_fig, param_list, eps_list):
    x0 = _np.array(x0)
    for i, param in enumerate(param_list):
        def func_wrap(x):
            return task2_function(x[0], x[1], param)

        def gradient_wrap(x):
            return task2_gradient(x, param)

        def hessian_wrap(x):
            return task2_hessian(x, param)

        fig_id = first_fig + i
        gradient_methods(func_wrap, gradient_wrap, x0, eps_list, fig_id)
        non_gradient_methods(func_wrap, x0, eps_list, fig_id)
        nuton_counts = [methods.nuton(gradient_wrap, hessian_wrap, x0, eps)[1]
                        for eps in eps_list]
        pp.plot(_np.log10(eps_list), nuton_counts, label="Nuton")

        pp.gca().invert_xaxis()
        pp.legend()
        pp.xlabel("epsilon")
        pp.ylabel("counts")
        pp.title("A = {0}".format(param))
    pp.show()


def task3(x0, fig_id, eps_list):
    x0 = _np.array(x0)

    def func_wrap(x):
        return task3_function(x[0], x[1])

    gradient_methods(func_wrap, task3_gradient, x0, eps_list, fig_id)
    non_gradient_methods(func_wrap, x0, eps_list, fig_id)
    nuton_counts = [methods.nuton(task3_gradient, task3_hessian, x0, eps)[1]
                    for eps in eps_list]
    pp.plot(_np.log10(eps_list), nuton_counts, label="Nuton")

    pp.gca().invert_xaxis()
    pp.legend()
    pp.xlabel("epsilon")
    pp.ylabel("counts")
    pp.show()


def gradient_methods(func, gradient, x0, eps_list, fig_id):
    pp.figure(fig_id)

    def plot_method(method, label):
        counts = [method(func, gradient, x0, eps)[1] for eps in eps_list]
        pp.plot(_np.log10(eps_list), counts, label=label)

    plot_method(methods.steepest_descent, "Steepest descent")
    plot_method(methods.conjugate_gradient, "Conj. gradient")


def non_gradient_methods(func, x0, eps_list, fig_id):
    pp.figure(fig_id)

    def plot_method(method, label):
        counts = [method(func, x0, eps)[1] for eps in eps_list]
        pp.plot(_np.log10(eps_list), counts, label=label)

    plot_method(methods.regular_simplex, "Simplex method")
    plot_method(methods.alternating_variable, "Alternating variables")
    plot_method(methods.hooke_jeeves, "Hooke-Jeeves")
    plot_method(methods.random_search, "Random Search")


def plot_task2_function(fig_id, x1_limit, a_param):
    pp.figure(fig_id)
    axes = pp.axes(projection="3d")
    x_1 = _np.linspace(-x1_limit, x1_limit)
    x_2 = _np.linspace(-3, 3)
    x_1, x_2 = _np.meshgrid(x_1, x_2)
    z = task2_function(x_1, x_2, a_param)
    axes.plot_wireframe(x_1, x_2, z)
    axes.set_xlabel("x1")
    axes.set_ylabel("x2")
    axes.set_zlabel("f(x1, x2)")
    axes.set_title("f(x1, x2) = x1 ^ 2 + {0} * x2 ^ 2".format(a_param))
    pp.show()


def plot_task3_function(fig_id):
    pp.figure(fig_id)
    axes = pp.axes(projection="3d")
    x_1 = _np.linspace(-128, 64)
    x_2 = _np.linspace(-128, 64)
    x_1, x_2 = _np.meshgrid(x_1, x_2)
    z = task3_function(x_1, x_2)
    axes.plot_wireframe(x_1, x_2, z)
    axes.set_xlabel("x1")
    axes.set_ylabel("x2")
    axes.set_zlabel("f(x1, x2)")
    axes.set_title("151 * x1 ** 2 + 33 * x1 + 151 * x2 ** 2 + 99 * x2 "
                   "- 300 * x1 * x2 + 48")
    pp.show()


def task2_function(x1, x2, a):
    return x1 ** 2 + a * x2 ** 2


def task2_gradient(x, a):
    return _np.array([2 * x[0], 2 * a * x[1]])


def task2_hessian(x, a):
    return _np.array([[2.0, 0.0], [0.0, 2.0 * a]])


def task3_function(x1, x2):
    return (151 * x1 ** 2 + 33 * x1 +
            151 * x2 ** 2 + 99 * x2 - 300 * x1 * x2 + 48)


def task3_gradient(x):
    return _np.array([302 * x[0] - 300 * x[1] + 33,
                      -300 * x[0] + 302 * x[1] + 99])


def task3_hessian(x):
    return _np.array([[302, -300], [-300, 302]])


if __name__ == "__main__":
    _main()
