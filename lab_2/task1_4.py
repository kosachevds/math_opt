import numpy as np
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D


def _main():
    plot_task2_function(0, 200, 1000)


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


def task3_function(x1, x2):
    return (151 * x1 ** 2 - 300 * x1 * x2 + 151 * x2 ** 2 + 33 * x1 +
            99 * x2 + 48)

if __name__ == "__main__":
    _main()
