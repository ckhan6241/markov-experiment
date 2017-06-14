import random

EMPTY_PROBABILITY = None

def _fill_probability(probabilities):
    indices = [i for i, x in enumerate(probabilities) if x == EMPTY_PROBABILITY]
    if not indices: return
    remaining_prob = 1 - sum([x for x in probabilities if x != EMPTY_PROBABILITY])
    if len(indices) == 1:
        probabilities[indices[0]] = remaining_prob
    else:
        random_numbers = sorted([random.random()*remaining_prob for i in range(len(indices)-1)]) + [remaining_prob]
        subtracted_numbers = [random_numbers[0]] + [random_numbers[i+1]-random_numbers[i] for i in range(len(indices)-1)]
        random.shuffle(subtracted_numbers)
        for idx, p in zip(indices, subtracted_numbers):
            probabilities[idx] = p

class TransitionProbability:

    def __init__(self, size, fixed_transition_probabilities):
        trans = [[EMPTY_PROBABILITY]*size for i in range(size)]
        for s, d, p in fixed_transition_probabilities:
            trans[s][d] = p
        for i in range(size):
            _fill_probability(trans[i])
        self.trans = trans

    def mutate(self, s, d1, d2):
        self.trans[s][d1], self.trans[s][d2] = EMPTY_PROBABILITY, EMPTY_PROBABILITY
        _fill_probability((self.trans[s]))

    def replace(self, s, probabilities):
        self.trans[s] = probabilities.copy()




if __name__ == '__main__':
    random.seed(0)
    filled_probability = [0.1]
    _fill_probability(filled_probability)
    assert filled_probability == [0.1]
    filled_probability = [0.1, 0.2, EMPTY_PROBABILITY]
    _fill_probability(filled_probability)
    assert filled_probability == [0.1, 0.2, 0.7]
    filled_probability = [0.1, EMPTY_PROBABILITY, EMPTY_PROBABILITY]
    _fill_probability(filled_probability)
    assert filled_probability[0] == 0.1 and sum(filled_probability) == 1.0

    print('test passed')
