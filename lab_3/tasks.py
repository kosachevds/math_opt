import os
import numpy as _np
from matplotlib import pyplot as _pp
from mpl_toolkits.mplot3d import Axes3D


_CURRENT_DIR = os.path.dirname(__file__)


def main():
    func1_filename = os.path.join(_CURRENT_DIR, "data/Funktsia_P2.txt")
    plot_function(func1_filename)
    # func2_filename = os.path.join(_CURRENT_DIR, "data/Funktsia_P4_V4.txt")
    # plot_function(func2_filename)
    _pp.show()


def plot_function(filename):
    with open(filename, "rt") as func_file:
        text_lines = [line.split() for line in func_file.readlines()]

    def get_float_column(lines, index):
        return _np.array([float(x[index]) for x in lines if x])

    x = get_float_column(text_lines, 0)
    y = get_float_column(text_lines, 1)
    z = get_float_column(text_lines, 2)
    size = len(x)
    noise = _np.random.rand(size) - 0.5
    x += noise
    y += noise
    axes = _pp.axes(projection="3d")
    axes.scatter3D(x, y, z)



if __name__ == "__main__":
    main()