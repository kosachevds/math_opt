from matplotlib import pyplot as pp
import numpy as np

X_BEGIN = -1.0
X_END = 1.5


def main():
    # pp.figure(0)
    # x_list = np.linspace(X_BEGIN, X_END)
    # pp.plot(x_list, function(x_list))
    # pp.xlabel("x")
    # pp.ylabel("f(x)")

    plot_perfomance(1)

    pp.show()


def plot_perfomance(figure_id):
    pp.figure(figure_id)
    eps_list = np.linspace(1e-2, 1e-4, num=30)

    def get_count(func, eps):
        return func(function, X_BEGIN, X_END, eps)[1]

    # counts = [get_count(enumerative, eps) for eps in eps_list]
    # pp.plot(eps_list, counts, label="enumerative")

    counts = [get_count(radix, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="radix")

    counts = [get_count(dichotomy, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="dichotomy")

    counts = [get_count(golden_ratio, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="golden_ratio")

    counts = [get_count(parabolic, eps) for eps in eps_list]
    pp.plot(eps_list, counts, label="parabolic")

    # counts = [get_count(parabolic, eps) for eps in eps_list]
    # pp.plot(eps_list, counts, label="parabolic")

    # counts = [get_count(parabolic, eps) for eps in eps_list]
    # pp.plot(eps_list, counts, label="parabolic")

    pp.ylabel("N")
    pp.xlabel("epsilon")
    pp.legend()
    pp.gca().invert_xaxis()
    pp.show()


def function(x):
    """ x^2 - 2x + e^(-x)
    """
    return x ** 2 - 2 * x + np.exp(-x)


def diff_func_1(x):
    """ 2x - 2 - x * e^(-x)
    """
    return 2 * x - 2 - x * np.exp(-x)


def diff_func_2(x):
    """ x^2 * e^(-x) + 2
    """
    return x ** 2 * np.exp(-x) + 2


def enumerative(func, begin, end, eps):
    previous = func(begin)
    n = 1
    for x in np.arange(begin + eps, end, eps):
        current = func(x)
        n += 1
        if current > previous:
            return (x - eps, n)
        previous = current
    return (end, n)


def radix(func, begin, end, eps):
    n = 0
    while end - begin > eps:
        step = (end - begin) / 4.0
        previous = func(begin)
        n += 1
        for x in np.linspace(begin + step, end, num=4):
            current = func(x)
            n += 1
            if current < previous:
                previous = current
                continue
            end = x
            begin = x - step
            break
    return ((begin + end) / 2.0, n)


def dichotomy(func, begin, end, eps):
    delta = eps
    n = 0
    while (end - begin) / 2.0 > eps:
        x1 = (begin + end - delta) / 2.0
        x2 = (begin + end + delta) / 2.0
        if func(x1) <= func(x2):
            end = x2
        else:
            begin = x1
        n += 2
    return ((begin + end) / 2.0, n)


def golden_ratio(func, begin, end, eps):
    tau = (np.sqrt(5) - 1) / 2.0
    x1 = (1 - tau) * (end - begin)
    f1 = func(x1)
    x2 = tau * (end - begin)
    f2 = func(x2)
    n = 2
    while (end - begin) / 2.0 > eps:
        if f1 <= f2:
            end = x2
            x2 = x1
            f2 = f1
            x1 = end - tau * (end - begin)
            f1 = func(x1)
        else:
            begin = x1
            x1 = x2
            f1 = f2
            x2 = end - tau * (end - begin)
            f2 = func(x2)
        n +=1
    return ((begin + end) / 2.0, n)


def parabolic(func, begin, end, eps):
    step = (end - begin) / 4.0
    x1 = begin + step
    f1 = func(x1)
    x2 = begin + 2 * step
    f2 = func(x2)
    x3 = end - step
    f3 = func(x3)
    n = 3
    previous = None
    while True:
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2)
        x_ = 0.5 * (x1 + x2 - a1 / a2)
        f_ = func(x_)
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
            begin = middle
        else:
            end = middle


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


if __name__ == "__main__":
    main()
