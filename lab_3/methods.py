# TODO: x_list to x_grid and y and z
import numpy as np


def enumerative(z_grid):
    return np.unravel_index(np.argmax(z_grid), z_grid.shape)


def simulated_annealing(z_grid, i_0, j_0):
    # max_k = 1000
    max_t = 1.1
    min_t = 0.001
    grid_size = len(z_grid)

    def temperature(iteration):
        return max_t * 0.1 / (iteration + 1)

    d_k = grid_size / 4
    t_k = max_t
    i_k, j_k = i_0, j_0
    f_k = z_grid[i_k, j_k]
    count = 1
    k = 0
    # while k < max_k:
    while t_k > min_t and d_k > 1:
        new_i = _get_new_index(i_k, d_k, grid_size)
        new_j = _get_new_index(j_k, d_k, grid_size)
        new_f = z_grid[new_i, new_j]
        count += 1
        d_f = new_f - f_k
        if d_f >= 0 or _do_annearling_jump(t_k, abs(d_f)):
            i_k = new_i
            j_k = new_j
            f_k = new_f
        k += 1
        t_k = temperature(k)
        # if k % 1000 == 0:
        #     d_k /= 2
    return i_k, j_k, count


def genetic_search(z_grid, population_size):
    max_count = 100
    grid_size = len(z_grid)
    population = np.random.randint(grid_size, size=(population_size, 2))
    population = [tuple(gene) for gene in population]
    generation_count = 1
    f_values = [z_grid[gene] for gene in population]
    while generation_count < max_count and _count_unique(f_values) > population_size / 2:
        childs = _get_new_generation(population, f_values)
        _mutation(childs, 1/2, grid_size)
        childs_f = [z_grid[gene] for gene in childs]
        population += childs
        f_values += childs_f
        population, f_values = _sort_population(population, f_values)
        population = population[population_size:]
        f_values = f_values[population_size:]
        if _count_unique(population) <= np.ceil(population_size * 0.5):
            population = _mutation_not_unique(population, grid_size)
            f_values = [z_grid[gene] for gene in population]
            population, f_values = _sort_population(population, f_values)
        generation_count += 1
    return population[-1]


def random_search(z_grid, i_0, j_0):
    grid_size = len(z_grid)
    d_k = grid_size / 2
    max_step = 6
    step_count = 0
    i_k, j_k = i_0, j_0
    f_k = z_grid[i_k, j_k]
    while d_k > 1:
        step_count += 1
        new_i = _get_new_index(i_0, d_k, grid_size)
        new_j = _get_new_index(j_0, d_k, grid_size)
        new_f = z_grid[new_i, new_j]
        if new_f > f_k:
            i_k = new_i
            j_k = new_j
            f_k = new_f
            step_count = 0
        elif step_count == max_step:
            step_count = 0
            d_k /= 2
    return i_k, j_k


def _pattern_search(x_grid, y_grid, z_grid, i_0, j_0):
    pass


def _get_new_index(initial, width, max_size):
    step = round(np.random.uniform(-width / 2, width / 2))
    return min(max(0, initial + step), max_size - 1)


def _do_annearling_jump(t_i, d_f):
    factor_d = 1
    probability = np.exp(-factor_d * d_f / t_i)
    return np.random.rand() <= probability


def _get_gene_pair(genes, weights):
    indices = np.random.choice(len(genes), size=2, replace=False, p=weights)
    indices.sort()
    return tuple(genes[i] for i in indices)


def _get_child(gene_pair):
    first_item_parent = np.random.randint(2)
    other_item_parent = int(not first_item_parent)
    return (gene_pair[first_item_parent][0], gene_pair[other_item_parent][1])


def _get_new_generation(genes, f_values):
    population_size = len(genes)
    f_sum = sum(f_values)
    weights = [item / f_sum for item in f_values]
    pairs = set()
    while len(pairs) < population_size:
        new_pair = _get_gene_pair(genes, weights)
        pairs.add(new_pair)
    return [_get_child(p) for p in pairs]


def _mutation(genes, proportion, grid_size):
    mutant_count = int(proportion * len(genes))
    indices = np.random.choice(len(genes), size=mutant_count, replace=False)
    for index in indices:
        genes[index] = _mutation_gene(genes[index], grid_size)
    return genes


def _mutation_gene(gene, grid_size):
    item_index = np.random.randint(2)
    new_value = np.random.randint(grid_size)
    gene = list(gene)
    gene[item_index] = new_value
    return tuple(gene)


def _count_unique(values):
    return len(set(values))


def _mutation_not_unique(genes, grid_size):
    count = len(genes)
    for i in range(count):
        for j in range(i + 1, count):
            if genes[j] == genes[i]:
                genes[j] = _mutation_gene(genes[j], grid_size)
    return genes


def _sort_population(population, f_values):
    pairs = list(zip(population, f_values))
    pairs.sort(key=(lambda p: p[1]))
    return [list(x) for x in zip(*pairs)]