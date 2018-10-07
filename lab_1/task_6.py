import numpy as np
from matplotlib import pyplot as pp


def main():
    epsilon = 1e-4
    x_list = np.linspace(1, 12, 100)
    pp.subplot(2, 1, 1)
    pp.xlabel("X")
    pp.ylabel("f(X)")
    pp.plot(x_list, function1(x_list))
    for x in enumerative(function1, x_list[0], x_list[-1], epsilon):
        pp.plot(x, function1(x), marker="o", color="g")
    x = broken_lines(function1, x_list[0], x_list[-1], epsilon)[0]
    pp.plot(x, function1(x), marker="+", color="r")

    x_list = np.linspace(0, 4, 100)
    pp.subplot(2, 1, 2)
    pp.xlabel("X")
    pp.ylabel("f(X)")
    pp.plot(x_list, function2(x_list))
    for x in enumerative(function2, x_list[0], x_list[-1], epsilon):
        pp.plot(x, function2(x), marker="o", color="g")
    x = broken_lines(function2, x_list[0], x_list[-1], epsilon)[0]
    pp.plot(x, function2(x), marker="+", color="r")

    pp.show()


def function1(x):
    return np.cos(x) / x ** 2


def function2(x):
    return 0.1 * x + 2 * np.sin(4 * x)


def enumerative(func, begin, end, eps):
    previous = func(begin)
    upward = False
    result = []
    for x in np.arange(begin + eps, end, eps):
        current = func(x)
        if upward:
            upward = current > previous
        elif current > previous:
            result.append(x)
            upward = True
        previous = current
    return result


def broken_lines(func, begin, end, eps):
    const_l, count = get_l_const(func, begin, end, 8);
    f_b = func(begin)
    f_e = func(end)
    count += 2
    pairs = [((f_b - f_e + const_l * (begin + end)) / (2.0 * const_l),
              (f_b + f_e + const_l * (begin - end)) / 2.0)]
    while True:
        min_pair = pairs[0]
        for pair in pairs:
            if pair[1] < min_pair[1]:
                min_pair = pair
        pairs.remove(min_pair)
        x = min_pair[0]
        f_x = func(x)
        count += 1
        delta = (f_x - min_pair[1]) / (2 * const_l)
        if 2 * const_l * delta < eps:
            return (x, count)
        new_p = 0.5 * (f_x + min_pair[1])
        pairs.append((x - delta, new_p))
        pairs.append((x + delta, new_p))


def get_l_const(func, begin, end, chords_count):
    chords_x = np.linspace(begin, end, chords_count + 1)
    chords_func = func(chords_x)
    const_l = None
    for i in range(1, len(chords_x)):
        new_ratio = (abs(chords_func[i] - chords_func[i - 1]) /
                     abs(chords_x[i] - chords_x[i - 1]))
        if const_l is None:
            const_l = new_ratio
        elif new_ratio > const_l:
            const_l = new_ratio
    return (const_l, len(chords_func))


if __name__ == '__main__':
    main()
