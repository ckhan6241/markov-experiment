import copy

POPULATION_SIZE = 'POPULATION_SIZE'
MAX_ITERATION = 'MAX_ITERATION'
TARGET_COST = 'TARGET_COST'
ELITISM = 'ELITISM'
MUTATE_TRANS_PROB = 'MUTATE_TRANS_PROB'
SINGLE_MUTATE_PROB = 'SINGLE_MUTATE_PROB'
OFFSPRING = 'OFFSPRING'
PERTURBATION = 'PERTURBATION'

DEFAULT_PERTURBATION = 'DEFAULT_PERTURBATION'

DEFAULT_CONFIG = {
    POPULATION_SIZE: 150,
    MAX_ITERATION: 500,
    TARGET_COST: 10**-6,
    ELITISM: 2,
    MUTATE_TRANS_PROB: 0.8,
    SINGLE_MUTATE_PROB: 0.4,
    OFFSPRING: 100,
    PERTURBATION: DEFAULT_PERTURBATION
}


def get_config():
    return copy.deepcopy(DEFAULT_CONFIG)