import numpy as np
from numpy import linalg
import onedim


def _numeric_gradient(func, x, eps):
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
        count += 2 * len(x_k)  # 2 calculation for every dim
        if linalg.norm(grad_k) < eps:
            return x_k, count

        def func_alpha(alpha):
            return func([x_k - alpha * grad_k])

        alpha_k = onedim.minimize(func, eps, 1, eps)
        x_k = x_k - alpha_k * grad_k
