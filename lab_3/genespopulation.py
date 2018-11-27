import numpy as np


class GenesPopulation():
    __grid = None
    __genes = None
    __values = None
    __grid_size = None
    __size = None

    def __init__(self, grid, population_size):
        self.__grid = grid
        self.__grid_size = len(grid)
        self.__size = population_size
        self.__genes = np.random.randint(self.__grid_size,
                                         size=(population_size, 2))
        self.__genes = [tuple(g) for g in self.__genes]
        self.__calculate_values()
        self.__sort()

    def get_best_gene(self):
        return self.__genes[-1]

    def get_best_value(self):
        return self.__values[-1]

    def values_gap(self):
        return self.__values[-1] - self.__values[0]

    def new_generation(self, mutation_ratio):
        if self.__count_possible_pairs() < self.__size:
            return False
        childs = self.__get_childs()
        childs = self.__mutation(childs, mutation_ratio, self.__grid_size)
        self.__genes += childs
        self.__calculate_values()
        self.__sort()
        self.__genes = self.__genes[-self.__size:]
        self.__values = self.__values[-self.__size:]
        return True

    def remake(self):
        self.__genes[-1] = self.__small_mutation(self.__genes[-1])
        self.__mutation_not_unique()
        self.__calculate_values()
        self.__sort()

    def __get_childs(self):
        if self.__count_possible_pairs() < self.__size:
            return None
        f_sum = sum(self.__values)
        weights = [item / f_sum for item in self.__values]
        parents = set()
        while len(parents) < self.__size:
            new_pair = self.__get_parents(self.__genes, weights)
            parents.add(new_pair)
        return [self.__get_child(p) for p in parents]

    def __calculate_values(self):
        self.__values = [self.__grid[g] for g in self.__genes]

    def __sort(self):
        pairs = list(zip(self.__genes, self.__values))
        pairs.sort(key=(lambda x: x[1]))
        self.__genes, self.__values = [list(x) for x in zip(*pairs)]

    def __mutation_not_unique(self):
        for i in range(self.__size):
            for j in range(i + 1, self.__size):
                if not np.array_equal(self.__genes[j], self.__genes[i]):
                    continue
                self.__genes[j] = self.__mutation_gene(self.__genes[j],
                                                       self.__grid_size)

    def __small_mutation(self, gene):
        add = 1
        if bool(np.random.randint(2)):
            add = -1
        index = np.random.randint(2)
        new_item = gene[index] + add
        if new_item >= self.__grid_size or new_item < 0:
            add *= -1
        gene = list(gene)
        gene[index] += add
        return tuple(gene)

    def __count_unique_genes(self):
        return len(set(tuple(g) for g in self.__genes))

    def __count_possible_pairs(self):
        unique_count = self.__count_unique_genes()
        if unique_count < 2:
            return 0
        factorial = np.math.factorial
        return factorial(unique_count) / (factorial(unique_count - 2) * 2)

    @staticmethod
    def __get_child(gene_pair):
        first_item_parent = np.random.randint(2)
        other_item_parent = int(not first_item_parent)
        return (gene_pair[first_item_parent][0],
                gene_pair[other_item_parent][1])

    @staticmethod
    def __get_parents(genes, weights):
        indices = np.random.choice(len(genes), size=2, replace=False,
                                   p=weights)
        indices.sort()
        return tuple(genes[i] for i in indices)

    @staticmethod
    def __mutation(genes, proportion, grid_size):
        mutant_count = int(proportion * len(genes))
        indices = np.random.choice(len(genes), size=mutant_count,
                                   replace=False)
        for index in indices:
            genes[index] = GenesPopulation.__mutation_gene(genes[index],
                                                           grid_size)
        return genes

    @staticmethod
    def __mutation_gene(gene, grid_size):
        item_index = np.random.randint(2)
        new_value = np.random.randint(grid_size)
        gene = list(gene)
        gene[item_index] = new_value
        return tuple(gene)
