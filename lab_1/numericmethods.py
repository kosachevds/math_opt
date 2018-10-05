def middle_point(function, begin, end, eps):
    count = 0
    while True:
        middle = (begin + end) / 2.0
        dxm = (function(middle + eps) - function(middle - eps)) / (2 * eps)
        count += 2
        if abs(dxm) < eps:
            return (middle, count)
        if dxm > 0:
            end = middle
        else:
            begin = middle


def chords(function, begin, end, eps):
    dxb = (function(begin + eps) - function(begin - eps)) / (2 * eps)
    dxe = (function(end + eps) - function(end - eps)) / (2 * eps)
    count = 4
    while True:
        x = begin - dxb / (dxb - dxe) * (begin - end)
        dxx = (function(x + eps) - function(x - eps)) / (2 * eps)
        count += 2
        if abs(dxx) < eps:
            return (x, count)
        if dxx > 0:
            end = x
            dxe = dxx
        else:
            begin = x
            dxb = dxx


def nuton(function, x0, eps):
    x = x0
    count = 0
    while True:
        f_1 = function(x + eps)
        f_2 = function(x - eps)
        count += 2
        dx1 = (f_1 - f_2) / (2 * eps)
        if abs(dx1) <= eps:
            return (x, count)
        f_0 = function(x)
        count += 1
        dx2 = (f_1 - 2 * f_0 + f_2) / (eps * eps)
        x = x - dx1 / dx2
