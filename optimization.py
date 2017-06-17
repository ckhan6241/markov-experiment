import time
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from random import sample, choice
from defs import *
from transition_probability import TransitionProbability, random_perturbation, NO_STATIONARY_PROB


def mutate(transition_probability, restriction_manager, mutate_prob, perturbation_func):
    for s in restriction_manager.all_rows():
        if random.random() < mutate_prob:
            d1, d2 = restriction_manager.random_cell_pair(s)
            transition_probability.mutate(s, d1, d2, perturbation_func)


def cross(transition_probability1, transition_probability2, restriction_manager):
    new_tp = transition_probability1.create_copy()
    for i in restriction_manager.all_rows():
        if random.choice([0, 1]):
            new_tp.replace(i, transition_probability2.get_prob(i))
    return new_tp


def mean_square_error_cost(transition_probability, target_stationary_prob):
    stationary_prob = transition_probability.get_stationary_probabilities()
    if stationary_prob is NO_STATIONARY_PROB:
        return np.inf
    else:
        cost = np.mean(np.square(stationary_prob - target_stationary_prob))
        return cost


def stratified_sampling(trans, size):
    cost, trans = zip(*trans)
    prob = 1 - np.array(cost)
    prob /= np.sum(prob)
    cdf = np.cumsum(prob)
    vec = (random.random() + np.arange(0.0, float(size))) / float(size+1)
    sample = [trans[i] for i in np.searchsorted(cdf, vec)]
    return sample


class RestrictionManager:
    def __init__(self, size, fixed_transition_probability):
        available_cells = [list(range(size)) for i in range(size)]
        for s, d, p in fixed_transition_probability:
            available_cells[s].remove(d)
        self.available_cells = dict((i, row) for i, row in enumerate(available_cells) if len(row) > 1)
        available_cells_array = [0]*size
        for i, row in self.available_cells.items():
            available_cells_array[i] = len(row)
        self.available_cells_array = available_cells_array

    def random_cell_pair(self, s):
        d1, d2 = random.choice(self.available_cells[s], size=2, replace=False)
        return d1, d2

    def all_rows(self):
        return self.available_cells.keys()

    def complete_restriction(self):
        return not self.available_cells

    def get_available_cells_array(self):
        return self.available_cells_array



def _speciate(trans, to_join_func, species=[]):
    def speciate_one(tran):
        for s in species:
            if to_join_func(s[0], tran):
                s.append(tran)
                return
        species.append([tran])

    if not species:
        species.append([trans[0]])
        start = 1
    else:
        start = 0
    for i in range(start, len(trans)):
        speciate_one(trans[i])
    return species

def _distribute_offsprings(total_amt, costs, counts, to_stagnant):
    amt = np.tile(0, len(costs))
    if to_stagnant.all():
        return amt
    costs = (costs * np.log(counts))[to_stagnant == False]
    costs = np.max(costs)*1.1 - costs
    cdf = np.cumsum(costs)
    vec = np.arange(0.0, 1.0, 1.0/total_amt) * cdf[-1]
    idx = np.searchsorted(vec, cdf)
    idx = [idx[0]] + [idx[i+1]-idx[i] for i in range(len(idx)-1)]
    amt[to_stagnant==False] = idx
    return amt


def _distribute_population_size(size, counts):
    counts = counts.astype('float') / np.sum(counts)
    return counts * size

def _create_to_join_func(restriction_manager):
    weights = restriction_manager.get_available_cells_array()
    def to_join_func(tran1, tran2):
        means = np.mean(np.square(tran1-tran2), axis=1)
        value = np.average(means, weights=weights)
        return value < 0.031
    return to_join_func

def _merge_trans(trans, stagnant, lowest_cost, to_join_func):
    previous_best = trans[0][0][0]
    trans = sorted(trans, key=lambda x: x[0][0])
    if trans[0][0][0] < previous_best:
        print 'from {} to {}'.format(previous_best, trans[0][0][0])
    species = [trans[0]]
    new_stagnant = list([stagnant[0]])
    new_lowest_cost = list([lowest_cost[0]])
    def merge(trans, stagnant, lowest_cost):
        for s in species:
            if to_join_func(s[0], trans[0]):
                s.extend(trans)
                return
        species.append(trans)
        new_stagnant.append(stagnant)
        new_lowest_cost.append(lowest_cost)
    for t, s, l, in zip(trans, stagnant, lowest_cost)[1:]:
        merge(t, s, l)
    return species, np.array(new_stagnant), np.array(new_lowest_cost)


class Optimiser:
    def __init__(self, config, size, fixed_transition_probability, stationary_prob):
        rm = RestrictionManager(size, fixed_transition_probability)
        if rm.complete_restriction():
            raise ValueError('Complete restriction, no optimization room')
        self.rm = rm
        trans = [TransitionProbability(size, fixed_transition_probability) for i in range(config[POPULATION_SIZE])]
        trans = [(mean_square_error_cost(transition_probability, stationary_prob), transition_probability) for
                 transition_probability in trans]
        trans = filter(lambda x: x[0] != np.inf, trans)
        trans = sorted(trans, key=lambda x: x[0])
        self.to_join_func = lambda x,y: _create_to_join_func(self.rm)(x[1].trans, y[1].trans)
        self.trans = _speciate(trans, self.to_join_func)
        self.config = config
        self.stationary_prob = stationary_prob
        if config[PERTURBATION] == DEFAULT_PERTURBATION:
            self.perturbation_func = random_perturbation
        else:
            self.perturbation_func = config[PERTURBATION]

    def optimize(self, verbose=False):
        if verbose:
            start = time.time()
            track_costs = []
        lowest_costs = np.array([tran[0][0] for tran in self.trans])
        stagnant = np.tile(0, len(self.trans))
        for i in range(self.config[MAX_ITERATION]):
            if verbose:
                print 'Iteration: {}'.format(i)
                print '\tPopulation size: {}'.format(sum(len(tran) for tran in self.trans))
                print '\tSpecies: {}'.format(len(self.trans))
                for i, s in enumerate(self.trans):
                    print '\t\t{} -- Cost: {} Size: {}'.format(i, s[0][0], len(s))
            if self.trans[0][0] < self.config[TARGET_COST]:
                if verbose:
                    print '\tTarget cost reached\n\tCost: {}'.format(self.trans[0][0])
                break
            if verbose:
                print '\t Cost: {}'.format(min(tran[0][0] for tran in self.trans))
            to_stagnant = stagnant > self.config[STAGNENT]
            elites, new_gen = [], []
            remove = []
            for i, (is_stagnant, species) in enumerate(zip(to_stagnant, self.trans)):
                if is_stagnant and i != 0:
                    remove.append(i)
                    continue
                elite, tran = species[:self.config[ELITISM]], species[self.config[ELITISM]:]
                elites.append(elite)
                for _, a_tran in tran:
                    if random.random() < self.config[MUTATE_TRANS_PROB]:
                        mutate(a_tran, self.rm , self.config[SINGLE_MUTATE_PROB], self.perturbation_func)
            _cross = lambda x: cross(x[0][1], x[1][1], self.rm)
            for i in range(len(self.trans)-1):
                if random.random() < self.config[INTER_SPECIES_CROSS_PROB]:
                    s1, s2 = sample(self.trans, 2)
                    tran1, tran2 = choice(s1), choice(s2)
                    new_gen.append(_cross((tran1, tran2)))
            costs = np.array([np.mean([tran[0] for tran in species]) for species in self.trans])
            counts = np.array([len(species) for species in self.trans])
            offsprings_counts = _distribute_offsprings(self.config[OFFSPRING], costs, counts, to_stagnant)
            species_size = _distribute_population_size(self.config[POPULATION_SIZE], counts)
            for n, m, species in zip(offsprings_counts, species_size, self.trans):
                for s in species:
                    assert isinstance(s[1], TransitionProbability)
                trans = species[self.config[ELITISM]:]
                if len(species) <= 2:
                    new_gen.append(choice(species)[1].create_copy())
                else:
                    new_gen.extend([_cross(sample(species, 2)) for i in range(n)])
                    new_gen.extend(stratified_sampling(trans, m - n - self.config[ELITISM]))
            new_gen = [(mean_square_error_cost(tran, self.stationary_prob), tran) for tran in new_gen]
            new_gen = filter(lambda x: x[0] != np.inf, new_gen)
            trans = _speciate(new_gen, self.to_join_func, species=elites)
            for species in trans:
                species.sort(key=lambda x: x[0])
            lowest_costs = np.delete(lowest_costs, remove)
            stagnant = np.delete(stagnant, remove)
            same_cost = np.array([lowest_costs[i] == trans[i][0][0] for i in range(len(lowest_costs))])
            stagnant[same_cost] += 1
            stagnant[same_cost == False] = 0
            stagnant  = np.concatenate((stagnant, [0]*(len(trans) - len(stagnant))))
            lowest_costs = np.array([tran[0][0] for tran in trans])
            trans, stagnant, lowest_costs = _merge_trans(trans, stagnant, lowest_costs, self.to_join_func)

            self.trans = trans
            if verbose:
                track_costs.append([x[0] for x in self.trans])
        if verbose:
            end = time.time()
            print 'Best cost: {}\nTime elapsed: {}'.format(self.trans[0][0][0], end - start)
            # plt.figure(1)
            # for j in range(0, i - 1, max((i - 1) / 9, 1)):
            #     plt.plot(track_costs[j], label=j)
            # plt.plot(track_costs[-1], label=i)
            # plt.legend()
            #
            # plt.figure(2)
            # plt.plot([cost[0] for cost in track_costs])
            # return plt

    def get_results(self):
        return self.trans[0]


if __name__ == '__main__':
    rm1 = RestrictionManager(3, [])
    assert rm1.all_rows() == list(range(3)) and rm1.get_available_cells_array() == [3, 3, 3]
    assert rm1.random_cell_pair(0) in [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    assert not rm1.complete_restriction()
    rm2 = RestrictionManager(3, [(0, 0, 0.5)])
    assert rm2.random_cell_pair(0) in [(1, 2), (2, 1)] and rm2.get_available_cells_array() == [2, 3, 3]
    rm3 = RestrictionManager(3, [(0, 0, 0.5), (0, 1, 0.3)])
    assert rm3.all_rows() == [1, 2]
    rm4 = RestrictionManager(3, [(0, 0, 0.5), (0, 1, 0.3), (1, 0, 0.5), (1, 1, 0.3), (2, 0, 0.5), (2, 1, 0.3)])
    assert rm4.complete_restriction() and rm4.get_available_cells_array() == [0,0,0]

    trans1 = TransitionProbability(3, [])
    trans2 = trans1.create_copy()
    rm = RestrictionManager(3, [])
    mutate(trans2, rm, 1, random_perturbation)
    for i in range(3):
        assert (trans1.get_prob(i) != trans2.get_prob(i)).any()
    trans3 = cross(trans1, trans2, rm1)
    for i in range(3):
        assert (trans3.get_prob(i) == trans1.get_prob(i)).all() or (trans3.get_prob(i) == trans2.get_prob(i)).all()

    trans4 = TransitionProbability(0, np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]]), to_copy=True)
    assert np.isclose(mean_square_error_cost(trans4, [0.25, 0.5, 0.25]), 0)

    sample = stratified_sampling(zip([5., 5., 5., 2., 2., 2.], [1, 1, 1, 0, 0, 0]), 3)
    assert np.mean(sample) > 0.5

    species = _speciate([4,2,3,4,6,2,3,1,1,3], lambda x,y: x==y)
    assert species == [[4,4],[2,2],[3,3,3],[6],[1,1]]
    species1 = _speciate([2,3,4,1,2,4,4,1,5], lambda x,y: x==y, species=species)
    assert species1 == [[4,4,4,4,4],[2,2,2,2],[3,3,3,3],[6],[1,1,1,1],[5]]

    counts = _distribute_offsprings(150, np.array([0.2, 0.3, 0.1, 0.4]), np.array([2, 4, 5, 1]), np.array([False, False, False, False]))
    assert sum(counts) == 150

    counts2 = _distribute_offsprings(150, np.array([0.2, 0.3, 0.1, 0.4]), np.array([2, 4, 5, 1]), np.array([False, False, False, True]))
    assert counts2[-1] == 0

    counts3 = _distribute_offsprings(160, np.tile(0.25, 4), np.tile(5, 4), np.tile(False, 4))
    assert (counts3 == [40, 40, 40, 40]).all()

    counts4 = _distribute_offsprings(160, np.tile(0.25, 4), np.tile(5, 4), np.tile(True, 4))
    assert (counts4 == [0]*4).all()

    print('test passed')
