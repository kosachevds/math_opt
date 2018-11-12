import numpy as np
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import methods


def _main():
    # plot_task2_function(0, 200, 1000)
    task2([10, 10], 0, [1000], [1e-3, 1e-5])


def task2(x0, first_fig, param_list, eps_list):
    x0 = np.array(x0)
    for i, param in enumerate(param_list):
        def func_wrap(x):
            return task2_function(x[0], x[1], param)

        def gradient_wrap(x):
            return task2_gradient(x, param)

        def hessian_wrap(x):
            return task2_hessian(x, param)

        plot_perfomance(func_wrap, gradient_wrap, hessian_wrap, x0, eps_list,
                        first_fig + i)

        pp.title("A = {0}".format(param))

 # todo another parameter for gradient methods or make it in two functions
def plot_perfomance(func, gradient, hessian, x0, eps_list, fig_id):
    pp.figure(fig_id)

    # TODO with plotting function instead of get_counts
    def get_counts_1(method):
        """With gradient"""
        return [method(func, gradient, x0, eps)[1]
                for eps in eps_list]

    # pp.plot(eps_list, get_counts_1(methods.steepest_descent), label="Steepest descent")
    # pp.plot(eps_list, get_counts_1(methods.conjugate_gradient), label="Conj. gradient")

    def get_counts_2(method):
        return [method(func, x0, eps)[1] for eps in eps_list]
    # pp.plot(eps_list, get_counts_2(methods.regular_simplex), label="Regular simplex")
    # pp.plot(eps_list, get_counts_2(methods.alternating_variable),
    #         label="Altarnating variables")
    pp.plot(eps_list, get_counts_2(methods.hooke_jeeves), label="Hooke-Jeeves")
    pp.plot(eps_list, get_counts_2(methods.random_search), label="Random Search")

    pp.plot(eps_list,
            [methods.nuton(gradient, hessian, x0, eps)[1] for eps in eps_list],
            label="Nuton")

    pp.gca().invert_xaxis()
    pp.legend()
    pp.xlabel("epsilon")
    pp.ylabel("counts")
    pp.show()


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
    return np.array([[2.0, 0.0], [0.0, 2.0 * a]])


def task3_function(x1, x2):
    return (151 * x1 ** 2 - 300 * x1 * x2 + 151 * x2 ** 2 + 33 * x1 +
            99 * x2 + 48)

if __name__ == "__main__":
    _main()
