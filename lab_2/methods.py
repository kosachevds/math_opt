# TODO: odnedim with one point
import numpy as np
from numpy import linalg
import onedim
from simplex import Simplex


def steepest_descent(func, gradient_func, x0, eps):
    x_k = x0
    count = 0
    while True:
        grad_k = gradient_func(x_k)
        count += 1
        if linalg.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k - alpha * grad_k])

        alpha_k = onedim.minimize(func_alpha, eps, 1, eps)
        x_k = x_k - alpha_k * grad_k


def conjugate_gradient(func, gradient, x0, eps):
    x_k = x0
    grad_k = gradient(x0)
    p_k = -grad_k
    count = 1
    k = 0
    while True:
        if linalg.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k + alpha * p_k])

        alpha_k = onedim.minimize(func_alpha, eps, 1, eps)
        x_k = x_k + alpha_k * p_k
        grad_next = gradient(x_k)
        beta = (linalg.norm(grad_next) ** 2) / (linalg.norm(grad_k) ** 2)
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
        if linalg.norm(gradient_k) < eps:
            return x_k, count
        inv_hessian = linalg.inv(hessian(x_k))
        count += 1
        x_k = x_k - inv_hessian * gradient_k


def regular_simplex(func, x0, eps):
    simplex = Simplex(x0, eps)
    simplex.apply(func)
    while True:
        simplex.sort()
        index = simplex.get_count() - 1
        while index >= 0:
            x_max, f_max = simplex.get_pair(index)
            new_x = simplex.get_new_x(x_max)
            new_f = func(new_x)
            if new_f < f_max:
                simplex.replace_pair(index, new_x, new_f)
                break
            index -= 1
        return simplex.nodes[0]


def alternating_variable(func, x0, eps):
    x_i = x0
    size = len(x0)
    while True:
        x_old = x_i
        for j in range(size):
            e_j = _basic_vector(size, j)

            def func_alpha(alpha):
                return func([x_i - alpha * e_j])

            alpha_j = onedim.minimize(func_alpha, 0, eps, eps)
            x_i = x_i + alpha_j * e_j
        if linalg.norm(x_old - x_i) <= eps:
            return x_i


def hooke_jeeves(func, x0, eps):
    delta = np.ones(len(x0)) * 2 * eps
    gamma = 2
    x_i = x0
    count = 0
    a_k = 2
    while True:
        new_x, step_count = _research(func, delta, x_i)
        count += step_count
        if new_x != x_i:
            x_i = x_i + a_k * (new_x - x_i)
            continue
        if linalg.norm(delta) < eps:
            return x_i
        delta /= gamma


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
    return np.array(vector)
