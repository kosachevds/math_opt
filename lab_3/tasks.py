import os
import numpy as np
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D


_CURRENT_DIR = os.path.dirname(__file__)


def main():
    func1_filename = os.path.join(_CURRENT_DIR, "data/Funktsia_P2.txt")
    plot_function(func1_filename)
    # func2_filename = os.path.join(_CURRENT_DIR, "data/Funktsia_P4_V4.txt")
    # plot_function(func2_filename)
    pp.show()


def plot_function(filename):
    with open(filename, "rt") as func_file:
        text_lines = [line.split() for line in func_file.readlines()]

    def get_float_column(lines, index):
        return np.array([float(x[index]) for x in lines if x])

    x = get_float_column(text_lines, 0)
    y = get_float_column(text_lines, 1)
    z = get_float_column(text_lines, 2)
    axes = pp.axes(projection="3d")
    axes.scatter3D(x, y, z)
    # x_grid = transform(x)
    # y_grid = transform(y)
    # z_grid = transform(z)
    # axes.plot_surface(x_grid, y_grid, z_grid)


def transform(var_list):
    matrix = []
    size = int(np.sqrt(len(var_list)))
    for i in range(size):
        matrix.append([])
        for j in range(size):
            matrix[i].append(var_list[i * size + j])
    return np.array(matrix)


if __name__ == "__main__":
    main()