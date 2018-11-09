import numpy as np


class Simplex():
    def __init__(self, center, length):
        self.values = None
        self.length = length
        self.nodes = []
        count = len(center) + 1
        for i in range(count):
            new_node = [0] * len(center)
            for j, _ in enumerate(new_node):
                if j < i - 1:
                    new_node[j] = center[j]
                elif j == i - 1:
                    new_node[j] = (
                        center[j] + length * np.sqrt(j / (2 * (j + 1))))
                else:
                    new_node[j] = center[j] - length / np.sqrt(2 * j * (j + 1))
            self.nodes.append(new_node)

    def get_count(self):
        return len(self.nodes)

    def get_pair(self, index):
        return (self.nodes[index], self.values[index])

    def apply(self, func):
        self.values = [func(x) for x in self.nodes]

    def sort(self):
        if self.values is None:
            return
        pairs = sorted(zip(self.nodes, self.values), key=(lambda x: x[1]))
        self.nodes = [p[0] for p in pairs]
        self.values = [p[1] for p in pairs]

    def get_new_x(self, x_opposite):
        return np.sum(self.nodes) * 2 / len(self.nodes) - x_opposite

    def replace_pair(self, index, new_x, new_f):
        self.nodes[index] = new_x
        self.values[index] = new_f
