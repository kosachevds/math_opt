# TODO: x_list to x_grid and y and z
import numpy as np


def enumerative(x_list, y_list, z_list):
    index = np.argmax(z_list)
    return (x_list[index], y_list[index])


def simulated_annealing(z_grid, i_0, j_0):
    # max_k = 1000
    max_t = 1.1
    min_t = 0.001
    grid_size = len(z_grid)

    def temperature(iteration):
        return max_t * 0.1 / (iteration + 1)

    d_k = grid_size / 4
    t_k = max_t
    i_k, j_k = i_0, j_0
    f_k = z_grid[i_k, j_k]
    count = 1
    k = 0
    # while k < max_k:
    while t_k > min_t and d_k > 1:
        new_i = _get_new_index(i_k, d_k, grid_size)
        new_j = _get_new_index(j_k, d_k, grid_size)
        new_f = z_grid[new_i, new_j]
        count += 1
        d_f = new_f - f_k
        if d_f >= 0 or _do_annearling_jump(t_k, abs(d_f)):
            i_k = new_i
            j_k = new_j
            f_k = new_f
        k += 1
        t_k = temperature(k)
        # if k % 1000 == 0:
        #     d_k /= 2
    return i_k, j_k, count


def genetic_search(x_grid, y_grid, z_grid, i_0, j_0):
    pass


def random_search(x_grid, y_grid, z_grid, i_0, j_0):
    pass


def pattern_search(x_grid, y_grid, z_grid, i_0, j_0):
    pass


def _get_new_index(initial, width, max_size):
    step = round(np.random.uniform(-width / 2, width / 2))
    return min(max(0, initial + step), max_size - 1)


def _do_annearling_jump(t_i, d_f):
    factor_d = 1
    probability = np.exp(-factor_d * d_f / t_i)
    return np.random.rand() <= probability
