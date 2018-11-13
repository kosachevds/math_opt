# TODO: odnedim with one point
import numpy as _np
from numpy import linalg as _la
import onedim as _onedim
from simplex import Simplex


def steepest_descent(func, gradient, x0, eps):
    # TODO with quadratic function matrices
    x_k = x0
    count = 0
    min_alpha = _np.finfo(_np.float32).eps
    while True:
        grad_k = gradient(x_k)
        count += 1
        if _la.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func(x_k - alpha * grad_k)

        alpha_k, step_count = _onedim.minimize(func_alpha, min_alpha, 1, eps)
        if alpha_k == min_alpha:
            alpha_k = 1
        count += step_count
        x_k = x_k - alpha_k * grad_k


def conjugate_gradient(func, gradient, x0, eps):
    x_k = x0
    grad_k = gradient(x0)
    p_k = -grad_k
    count = 1
    k = 0
    while True:
        if _la.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func(x_k + alpha * p_k)

        alpha_k, step_count = _onedim.minimize(func_alpha, eps, 1, eps)
        count += step_count
        x_k = x_k + alpha_k * p_k
        grad_next = gradient(x_k)
        beta = (_la.norm(grad_next) / _la.norm(grad_k)) ** 2
        if k > 0 and k % len(x0) == 0:
            beta = 0
        k += 1
        grad_k = grad_next
        p_k = -grad_k + beta * p_k


def nuton(gradient, hessian, x0, eps):
    x_k = x0
    count = 0
    while True:
        gradient_k = gradient(x_k)
        count += 1
        if _la.norm(gradient_k) < eps:
            return x_k, count
        inv_hessian = _la.inv(hessian(x_k))
        count += 1
        x_k = x_k - _np.matmul(inv_hessian, gradient_k)


def regular_simplex(func, x0, eps):
    simplex = Simplex(x0, _np.e)
    simplex.apply(func)
    count = len(x0)
    while True:
        simplex.sort()
        if simplex.length < eps:
            return simplex.nodes[0], count
        index = simplex.get_count() - 1
        f_max = simplex.values[index]
        new_x = simplex.get_opposite(index)
        new_f = func(new_x)
        count += 1
        if new_f < f_max:
            simplex.replace_pair(index, new_x, new_f)
            continue
        else:
            simplex.reduction(0.5)
            simplex.apply(func)
            count += simplex.get_count()


def alternating_variable(func, x0, eps):  # FIX
    x_i = x0
    size = len(x0)
    count = 0
    while True:
        x_old = x_i
        for j in range(size):
            e_j = _basic_vector(size, j)

            def func_alpha(alpha):
                return func(x_i - alpha * e_j)

            alpha_j, step_count = _onedim.minimize(func_alpha, 0, eps, eps)
            count += step_count
            x_i = x_i + alpha_j * e_j
        if _la.norm(x_old - x_i) <= eps:
            return x_i, count


def hooke_jeeves(func, x0, eps):  # error with [9.9, 9.9]
    delta = _np.ones(len(x0))
    gamma = 2.0
    x_i = x0
    count = 0
    a_k = 2
    while True:
        new_x, step_count = _research(func, delta, x_i)
        count += step_count
        if not _np.array_equal(new_x, x_i):
            x_i = x_i + a_k * (new_x - x_i)
            continue
        if _la.norm(delta) < eps:
            return x_i, count
        delta /= gamma


def random_search(func, x0, eps):  # FIX: bad result with a == 1000, 250; with a == 1 is near good
    max_steps = 3 * len(x0)
    alpha = 1
    gamma = 2.0
    x_i = x0
    f_i = func(x_i)
    count = 1
    ksi_count = 0
    while True:
        ksi = _np.random.uniform(-1, 1, len(x0))
        ksi_count += 1
        y_i = x_i + alpha * (ksi / _la.norm(ksi))
        f_y = func(y_i)
        count += 1
        if f_y < f_i:
            f_i = f_y
            x_i = y_i
            ksi_count = 0
            continue
        if ksi_count < max_steps:
            continue
        if alpha < eps:
            return x_i, count
        alpha /= gamma


def _research(func, delta, x0):
    x_j = x0
    f_j = func(x_j)
    count = 1
    for j in range(len(x0)):
        e_j = _basic_vector(len(x0), j)
        y = x_j - delta[j] * e_j
        f_y = func(y)
        count += 1
        if f_j <= f_y:
            y = x_j + delta[j] * e_j
            f_y = func(y)
            count += 1
            if f_j <= f_y:
                continue
        x_j = y
        f_j = f_y
    return x_j, count


def _basic_vector(size, index):
    vector = [0] * size
    vector[index] = 1
    return _np.array(vector)
