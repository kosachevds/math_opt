import numpy as _np

X_0 = _np.array([-1, 1])


def main():
    pass


def resenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


if __name__ == "__main__":
    main()
