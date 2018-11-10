import numpy as np
from numpy import linalg
import onedim


def _numeric_gradient(func, x, eps):
    """2 calculation for every dim"""
    gradient = []
    for i, _ in enumerate(x):
        eps_vector = np.zeros(len(x))
        eps_vector[i] = eps
        gradient.append(func(x + eps_vector) - func(x - eps_vector))
    return np.array(gradient) / (2 * eps)


def steepest_descent(func, x0, eps):
    x_k = x0
    count = 0
    while True:
        grad_k = _numeric_gradient(func, x_k, eps)
        count += 2 * len(x_k)
        if linalg.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k - alpha * grad_k])

        alpha_k, step_count = onedim.minimize(func, eps, 1, eps)
        count += step_count
        x_k = x_k - alpha_k * grad_k


def conjugate_gradient(func, x0, eps):
    x_k = x0
    grad_k = _numeric_gradient(func, x_k, eps)
    p_k = -grad_k
    count = 2 * len(x_k)
    k = 0
    while True:
        if linalg.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k + alpha * p_k])

        alpha_k, step_count = onedim.minimize(func, eps, 1, eps)
        count += step_count
        x_k = x_k + alpha_k * p_k
        grad_next = _numeric_gradient(func, x_k, eps)
        count += 2 * len(x_k)  # 2 calculation for every dim
        beta = (linalg.norm(grad_next) ** 2) / (linalg.norm(grad_k) ** 2)
        if k > 0 and k % len(x0) == 0:
            beta = 0
        k += 1
        grad_k = grad_next
        p_k = -grad_k + beta * p_k
