import numpy as np
from defs import get_config, PERTURBATION
from optimization import Optimiser
from transition_probability import _calculate_stationary_probabilities, gaussian_perturbation

if __name__ == '__main__':
    # get default config but overwrite perturbation to gaussian
    config = get_config()
    config[PERTURBATION] = gaussian_perturbation

    # have a transition probability like this:
    #  X     0.25   X
    #  X      X    0.5
    # 0.75    X     X
    size = 3
    fixed_transition_probability = [(0, 1, 0.25), (1, 2, 0.5), (2, 0, 0.75)]

    # and target stationary prob
    # 0.6   0.2   0.2
    stationary_prob = [0.6, 0.2, 0.2]

    op = Optimiser(config, size, fixed_transition_probability, stationary_prob)
    op.optimize(verbose=True)

    print np.around(np.array(op.trans[0][1].trans), 3)
    print np.around(_calculate_stationary_probabilities(op.trans[0][1].trans), 3)
