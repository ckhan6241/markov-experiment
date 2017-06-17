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
    fixed_transition_probability = [(0, 1, 0.25), (1, 2, 0.25), (2, 0, 0.75)]

    # and target stationary prob
    # 0.6   0.2   0.2
    stationary_prob = [0.6, 0.2, 0.2]
    # size = 10
    # fixed_transition_probability = [(0, 1, 0.25), (1, 2, 0.5), (2, 0, 0.75), (3, 0, 0.2), (5, 8, 0.1), (9, 3, 0.0), (9, 9, 0.0)]
    #
    # # and target stationary prob
    # # 0.6   0.2   0.2
    # stationary_prob = [0.4, 0.2, 0.15, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.01]

    op = Optimiser(config, size, fixed_transition_probability, stationary_prob)
    plt = op.optimize(verbose=False)

    for i, t in enumerate(op.trans):
        print i
        print np.around(np.array(t[0][1].trans), 3)
        print np.around(_calculate_stationary_probabilities(t[0][1].trans), 3)
        print np.around(t[0][0], 3)

    # plt.show()
