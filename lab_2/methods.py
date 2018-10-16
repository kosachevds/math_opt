import numpy as np
from numpy import linalg as la


def _minimize(func, begin, end, eps):
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = func(begin)
        for x in np.linspace(begin + step, end, num=4):
            current = func(x)
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
    return (begin + end) / 2.0


def steepest_descent(func, gradient_func, x0, eps):
    x_k = x0
    count = 0
    while True:
        grad_k = gradient_func(x_k)
        count += 1
        if la.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k - alpha * grad_k])

        alpha_k = _minimize(func, eps, 1, eps)
        x_k = x_k - alpha_k * grad_k


def conjugate_gradient(func, gradient, x0, eps):
    x_k = x0
    grad_k = gradient(x0)
    p_k = -grad_k
    count = 1
    k = 0
    while True:
        if la.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k + alpha * p_k])

        alpha_k = _minimize(func, eps, 1, eps)
        x_k = x_k + alpha_k * p_k
        grad_next = gradient(x_k)
        beta = (la.norm(grad_next) ** 2) / (la.norm(grad_k) ** 2)
        if k > 0 and k % len(x0) == 0:
            beta = 0
        k += 1
        grad_k = grad_next
        p_k = -grad_k + beta * p_k
