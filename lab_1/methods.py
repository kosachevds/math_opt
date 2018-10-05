import numpy as np


def enumerative(function, begin, end, eps):
    previous = function(begin)
    n = 1
    for x in np.arange(begin + eps, end, eps):
        current = function(x)
        n += 1
        if current > previous:
            return (x - eps, n)
        previous = current
    return (end, n)


def radix(function, begin, end, eps):
    n = 0
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = function(begin)
        n += 1
        for x in np.linspace(begin + step, end, num=4):
            current = function(x)
            n += 1
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
    return ((begin + end) / 2.0, n)


def dichotomy(function, begin, end, eps):
    delta = eps
    n = 0
    while (end - begin) / 2.0 > eps:
        x1 = (begin + end - delta) / 2.0
        x2 = (begin + end + delta) / 2.0
        if function(x1) <= function(x2):
            end = x2
        else:
            begin = x1
        n += 2
    return ((begin + end) / 2.0, n)


def golden_ratio(function, begin, end, eps):
    tau = (np.sqrt(5) - 1) / 2.0
    x1 = (1 - tau) * (end - begin)
    f1 = function(x1)
    x2 = tau * (end - begin)
    f2 = function(x2)
    n = 2
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
        n += 1
    return ((begin + end) / 2.0, n)


def parabolic(function, begin, end, eps):
    step = (end - begin) / 4.0
    x1 = begin + step
    f1 = function(x1)
    x2 = begin + 2 * step
    f2 = function(x2)
    x3 = end - step
    f3 = function(x3)
    n = 3
    previous = None
    while True:
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2)
        x_ = 0.5 * (x1 + x2 - a1 / a2)
        f_ = function(x_)
        n += 1
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
            return (x_, n)
        previous = x_


def middle_point(derivative, begin, end, eps):
    n = 0
    while True:
        middle = (begin + end) / 2.0
        fm = derivative(middle)
        n += 1
        if abs(fm) < eps:
            return (middle, n)
        if fm > 0:
            end = middle
        else:
            begin = middle


def chords(derivative, begin, end, eps):
    db = derivative(begin)
    de = derivative(end)
    n = 2
    while True:
        x = begin - db / (db - de) * (begin - end)
        dx = derivative(x)
        n += 1
        if abs(dx) < eps:
            return (x, n)
        if dx > 0:
            end = x
            de = dx
        else:
            begin = x
            db = dx


def nuton(derivative1, derivative2, x0, eps):
    x = x0
    n = 0
    while True:
        f_1 = derivative1(x)
        n += 1
        if abs(f_1) <= eps:
            return (x, n)
        f_2 = derivative2(x)
        n += 1
        x = x - f_1 / f_2
