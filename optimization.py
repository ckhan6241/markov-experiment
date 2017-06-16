import time
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from random import sample
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
    if stationary_prob == NO_STATIONARY_PROB:
        return np.inf
    else:
        cost = np.mean(np.square(stationary_prob - target_stationary_prob))
        return cost


def stratified_sampling(trans, size):
    cost, trans = zip(*trans)
    prob = 1 - np.array(cost)
    prob /= np.sum(prob)
    cdf = np.cumsum(prob)
    vec = (random.random() + np.arange(0.0, float(size))) / float(size)
    sample = [trans[i] for i in np.searchsorted(cdf, vec)]
    return sample


class RestrictionManager:
    def __init__(self, size, fixed_transition_probability):
        available_cells = [list(range(size)) for i in range(size)]
        for s, d, p in fixed_transition_probability:
            available_cells[s].remove(d)
        self.available_cells = dict((i, row) for i, row in enumerate(available_cells) if len(row) > 1)

    def random_cell_pair(self, s):
        d1, d2 = random.choice(self.available_cells[s], size=2, replace=False)
        return d1, d2

    def all_rows(self):
        return self.available_cells.keys()

    def complete_restriction(self):
        return not self.available_cells


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
        self.trans = sorted(trans, key=lambda x: x[0])
        self.config = config
        self.stationary_prob = stationary_prob
        if config[PERTURBATION] == DEFAULT_PERTURBATION:
            self.perturbation_func = random_perturbation
        else:
            self.perturbation_func = config[PERTURBATION]

    def optimize(self, verbose=False):
        if verbose:
            start = time.time()
            costs = []
        for i in range(self.config[MAX_ITERATION]):
            if verbose:
                print 'Iteration: {}'.format(i)
                print '\tPopulation size: {}'.format(len(self.trans))
            if self.trans[0][0] < self.config[TARGET_COST]:
                if verbose:
                    print '\tTarget cost reached\n\tCost: {}'.format(self.trans[0][0])
                break
            if verbose:
                print '\t Cost: {}'.format(self.trans[0][0])
            elites, trans = self.trans[:self.config[ELITISM]], self.trans[self.config[ELITISM]:]
            for _, tran in trans:
                if random.random() < self.config[MUTATE_TRANS_PROB]:
                    mutate(tran, self.rm, self.config[SINGLE_MUTATE_PROB], self.perturbation_func)
            _cross = lambda x: cross(x[0][1], x[1][1], self.rm)
            offsprings = [_cross(sample(self.trans, 2)) for i in range(self.config[OFFSPRING])]
            survivals = stratified_sampling(trans, self.config[POPULATION_SIZE] - self.config[ELITISM] - self.config[
                OFFSPRING])
            new_gen = survivals + offsprings
            new_gen = [(mean_square_error_cost(tran, self.stationary_prob), tran) for tran in new_gen]
            new_gen = filter(lambda x: x[0] != np.inf, new_gen)
            self.trans = sorted(elites + new_gen, key=lambda x: x[0])
            if verbose:
                costs.append([x[0] for x in self.trans])
        if verbose:
            end = time.time()
            print 'Best cost: {}\nTime elapsed: {}'.format(self.trans[0][0], end - start)
            plt.figure(1)
            for j in range(0, i-1, max((i-1)/9, 1)):
                plt.plot(costs[j], label=j)
            plt.plot(costs[-1], label=i)
            plt.legend()

            plt.figure(2)
            plt.plot([cost[0] for cost in costs])
            return plt

    def get_results(self):
        return self.trans[0]


if __name__ == '__main__':
    rm1 = RestrictionManager(3, [])
    assert rm1.all_rows() == list(range(3))
    assert rm1.random_cell_pair(0) in [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    assert not rm1.complete_restriction()
    rm2 = RestrictionManager(3, [(0, 0, 0.5)])
    assert rm2.random_cell_pair(0) in [(1, 2), (2, 1)]
    rm3 = RestrictionManager(3, [(0, 0, 0.5), (0, 1, 0.3)])
    assert rm3.all_rows() == [1, 2]
    rm4 = RestrictionManager(3, [(0, 0, 0.5), (0, 1, 0.3), (1, 0, 0.5), (1, 1, 0.3), (2, 0, 0.5), (2, 1, 0.3)])
    assert rm4.complete_restriction()

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

    print('test passed')
