import os
import numpy as np
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import methods


_CURRENT_DIR = os.path.dirname(__file__)


def main():
    # func2_filename = os.path.join(_CURRENT_DIR, "data/Funktsia_P4_V4.txt")
    # plot_function(func2_filename, False)
    common_task()
    pp.show()


def common_task():
    func_filename = os.path.join(_CURRENT_DIR, "data/Funktsia_P2.txt")
    x, y, z = read_function(func_filename, True)
    # i_min, j_min, _ = methods.simulated_annealing(z, 10, 10)
    i_min, j_min = methods.genetic_search(z, 10)
    test_genetic_search(z, 10, 100)
    # i_min, j_min = methods.random_search(z, 10, 10)
    axes = Axes3D(pp.figure())
    axes.plot_wireframe(x, y, z)
    # plot_function(func_filename, False)
    axes.scatter(x[i_min, j_min], y[i_min, j_min], z[i_min, j_min], c="r",
                 s=100)


def plot_function(filename, surface):
    x, y, z = read_function(filename, surface)
    axes = pp.axes(projection="3d")
    if surface:
        axes.plot_surface(x, y, z)
    else:
        axes.scatter3D(x, y, z)


def read_function(filename, as_grids):
    with open(filename, "rt") as func_file:
        text_lines = [line.split() for line in func_file.readlines()]

    def get_float_column(lines, index):
        return np.array([float(x[index]) for x in lines if x])

    x = get_float_column(text_lines, 0)
    y = get_float_column(text_lines, 1)
    z = get_float_column(text_lines, 2)
    if not as_grids:
        return x, y, z
    x_grid = transform(x)
    y_grid = transform(y)
    z_grid = transform(z)
    return x_grid, y_grid, z_grid


def transform(var_list):
    matrix = []
    size = int(np.sqrt(len(var_list)))
    for i in range(size):
        matrix.append([])
        for j in range(size):
            matrix[i].append(var_list[i * size + j])
    return np.array(matrix)


def test_genetic_search(z_grid, population_size, launch_count):
    ref_point = methods.enumerative(z_grid)
    ref_z = z_grid[ref_point]
    errors = []
    for _ in range(launch_count):
        x_max = methods.genetic_search(z_grid, population_size)
        errors.append(abs(ref_z - z_grid[x_max]))
    print(np.mean(errors), np.std(errors))

if __name__ == "__main__":
    main()
