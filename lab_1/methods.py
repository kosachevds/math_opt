import numpy as np


def enumerative(function, begin, end, eps):
    previous = function(begin)
    count = 1
    for x in np.arange(begin + eps, end, eps):
        current = function(x)
        count += 1
        if current > previous:
            return (x - eps, count)
        previous = current
    return (end, count)


def radix(function, begin, end, eps):
    count = 0
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = function(begin)
        count += 1
        for x in np.linspace(begin + step, end, num=4):
            current = function(x)
            count += 1
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
    return ((begin + end) / 2.0, count)


def dichotomy(function, begin, end, eps):
    delta = eps
    count = 0
    while (end - begin) / 2.0 > eps:
        x1 = (begin + end - delta) / 2.0
        x2 = (begin + end + delta) / 2.0
        if function(x1) <= function(x2):
            end = x2
        else:
            begin = x1
        count += 2
    return ((begin + end) / 2.0, count)


def golden_ratio(function, begin, end, eps):
    tau = (np.sqrt(5) - 1) / 2.0
    x1 = (1 - tau) * (end - begin)
    f1 = function(x1)
    x2 = tau * (end - begin)
    f2 = function(x2)
    count = 2
    while (end - begin) / 2.0 > eps:
        if f1 <= f2:
            end = x2
            x2 = x1
            f2 = f1
            x1 = end - tau * (end - begin)
            f1 = function(x1)
        else:
            begin = x1
            x1 = x2
            f1 = f2
            x2 = end - tau * (end - begin)
            f2 = function(x2)
        count += 1
    return ((begin + end) / 2.0, count)


def parabolic(function, begin, end, eps):
    step = (end - begin) / 4.0
    x1 = begin + step
    f1 = function(x1)
    x2 = begin + 2 * step
    f2 = function(x2)
    x3 = end - step
    f3 = function(x3)
    count = 3
    previous = None
    while True:
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2)
        x_ = 0.5 * (x1 + x2 - a1 / a2)
        f_ = function(x_)
        count += 1
        if x_ < x2:
            if f_ >= f2:
                x1 = x_
                f1 = f_
            else:
                x3 = x2
                f3 = f2
                x2 = x_
                f2 = f_
        else:
            if f_ >= x2:
                x3 = x_
                f3 = f_
            else:
                x1 = x2
                f1 = f2
                x2 = x_
                f2 = f_
        if previous is not None and abs(x_ - previous) < eps:
            return (x_, count)
        previous = x_


def middle_point(derivative, begin, end, eps):
    count = 0
    while True:
        middle = (begin + end) / 2.0
        fm = derivative(middle)
        count += 1
        if abs(fm) < eps:
            return (middle, count)
        if fm > 0:
            end = middle
        else:
            begin = middle


def chords(derivative, begin, end, eps):
    db = derivative(begin)
    de = derivative(end)
    count = 2
    while True:
        x = begin - db / (db - de) * (begin - end)
        dx = derivative(x)
        count += 1
        if abs(dx) < eps:
            return (x, count)
        if dx > 0:
            end = x
            de = dx
        else:
            begin = x
            db = dx


def nuton(derivative1, derivative2, x0, eps):
    x = x0
    count = 0
    while True:
        f_1 = derivative1(x)
        count += 1
        if abs(f_1) <= eps:
            return (x, count)
        f_2 = derivative2(x)
        count += 1
        if count > 1000:
            return (None, None)
        x = x - f_1 / f_2


def nuton_rafson(derivative1, derivative2, x0, eps):
    x = x0
    count = 0
    while True:
        f_1 = derivative1(x)
        count += 1
        if abs(f_1) <= eps:
            return (x, count)
        f_2 = derivative2(x)
        count += 1
        f_1_ = derivative1(x - f_1 / f_2)
        count += 1
        if count > 1000:
            return (None, None)
        tau = (f_1 ** 2) / (f_1 ** 2 + f_1_ ** 2)
        x = x - tau * f_1 / f_2


def marquardt(derivative1, derivative2, x0, eps):
    x = x0
    count = 0
    mu = None
    while True:
        f_1 = derivative1(x)
        count += 1
        if abs(f_1) <= eps:
            return (x, count)
        f_2 = derivative2(x)
        count += 1
        if count > 1000:
            return (None, None)
        if mu is None:
            mu = 10 * abs(f_2)
        elif f_1 > 0:
            mu /= mu
        else:
            mu *= 2
        x = x - f_1 / (f_2 + mu)
