import copy
import numpy as np
import numpy.random as random
from numpy import linalg as LA

EMPTY_PROBABILITY = -1.0
NO_STATIONARY_PROB = None

def _fill_probability(probabilities):
    # indices = [i for i, x in enumerate(probabilities) if x == EMPTY_PROBABILITY]
    indices = np.where(np.in1d(probabilities, [EMPTY_PROBABILITY]))
    if len(indices[0]) == 0: return
    remaining_prob = 1 - np.sum(probabilities[np.in1d(probabilities, [EMPTY_PROBABILITY], invert=True)])
    if len(indices[0]) == 1:
        probabilities[indices] = remaining_prob
    else:
        random_numbers = random.random(len(indices[0]))
        random_probabilities = random_numbers / np.sum(random_numbers) * remaining_prob
        probabilities[indices] = random_probabilities


def _calculate_stationary_probabilities(trans):
    trans = np.array(trans)
    w, v = LA.eig(trans.T)
    idx = np.where(np.amin(v, axis=0) >= 0)
    if len(idx[0]) == 0:
        return NO_STATIONARY_PROB
    else:
        unnormalized_prob = v.T[idx][0]
        normalized_prob = unnormalized_prob / np.sum(unnormalized_prob)
        return normalized_prob


def random_perturbation(probabilities, idx1, idx2):
    probabilities[idx1], probabilities[idx2] = EMPTY_PROBABILITY, EMPTY_PROBABILITY
    _fill_probability(probabilities)


def gaussian_perturbation(probabilities, idx1, idx2):
    amount = random.normal(scale=0.25)
    conditional_prob = 1 - np.sum(
        probabilities[np.where(np.in1d(range(len(probabilities)), (idx1, idx2), invert=True))])
    probabilities[idx1] += amount
    probabilities[idx2] -= amount
    correction_minus = max(probabilities[idx1] - conditional_prob, 0 - probabilities[idx2], 0)
    correction_plus = max(0 - probabilities[idx1], probabilities[idx2] - conditional_prob, 0)
    correction = correction_plus - correction_minus
    probabilities[idx1] += correction
    probabilities[idx2] -= correction


class TransitionProbability:
    def __init__(self, size, fixed_transition_probabilities, to_copy=False):
        if to_copy:
            self.trans = fixed_transition_probabilities.copy()
        else:
            trans = np.tile(EMPTY_PROBABILITY, (size, size))
            for s, d, p in fixed_transition_probabilities:
                trans[s][d] = p
            for i in range(size):
                _fill_probability(trans[i])

            self.trans = trans

    def mutate(self, s, d1, d2, mutation_func):
        mutation_func(self.trans[s], d1, d2)

    def replace(self, s, probabilities):
        self.trans[s] = copy.deepcopy(probabilities)

    def get_stationary_probabilities(self):
        stationary_prob = _calculate_stationary_probabilities(np.array(self.trans))
        return stationary_prob

    def create_copy(self):
        return TransitionProbability(0, self.trans, to_copy=True)

    def get_prob(self, s):
        return self.trans[s]


if __name__ == '__main__':
    # random.seed(0)
    filled_probability = np.array([0.1])
    _fill_probability(filled_probability)
    assert filled_probability == [0.1]
    filled_probability = np.array([0.1, 0.2, EMPTY_PROBABILITY])
    _fill_probability(filled_probability)
    assert np.isclose(filled_probability, [0.1, 0.2, 0.7]).all()
    filled_probability = np.array([0.1, EMPTY_PROBABILITY, EMPTY_PROBABILITY])
    _fill_probability(filled_probability)
    assert np.isclose(filled_probability[0], 0.1) and np.isclose(np.sum(filled_probability), 1.0)

    assert np.isclose(
        _calculate_stationary_probabilities(np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])),
        [0.25, 0.5, 0.25]).all()

    trans1, trans2 = TransitionProbability(3, []), TransitionProbability(3, [])
    for i in range(3):
        assert np.isclose(np.sum(trans1.get_prob(i)), 1.0) and (trans1.get_prob(i) >= 0.0).all()
    prob = list(trans1.get_prob(0))
    value_0_0 = prob[0]
    assert np.isclose(prob, trans1.get_prob(0)).all()
    trans1.mutate(0, 1, 2, random_perturbation)
    assert not (prob == trans1.get_prob(0)).all()
    assert np.isclose(trans1.get_prob(0)[0], value_0_0)
    assert np.isclose(np.sum(trans1.get_prob(0)), 1.0) and (trans1.get_prob(0) >= 0.0).all()
    trans1.replace(0, trans2.get_prob(0))
    assert np.isclose(trans1.get_prob(0), trans2.get_prob(0)).all()
    trans3 = trans1.create_copy()
    for i in range(3):
        assert np.isclose(trans1.get_prob(i), trans3.get_prob(i)).all()
    trans4 = TransitionProbability(0, np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]]), to_copy=True)
    assert np.isclose(trans4.get_stationary_probabilities(), [0.25, 0.5, 0.25]).all()
    trans5 = TransitionProbability(3, [(0, 0, 0.5)])
    assert np.isclose(trans5.get_prob(0)[0], 0.5)

    prob1 = np.array([0, 0.4, 0.6])
    prob2 = prob1.copy()
    random_perturbation(prob1, 1, 2)
    assert np.isclose(prob1[0], 0.0) and np.isclose(np.sum(prob1), 1.0) and (prob1 != prob2).any()

    prob3 = prob2.copy()
    gaussian_perturbation(prob3, 1, 2)
    assert np.isclose(prob3[0], 0.0) and np.isclose(np.sum(prob3), 1.0) and (prob3 != prob2).any()

    print('test passed')
