import torch

# Easy 1D fixed-depth MGDL

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

EXPERIMENT_NAME = "fixed1d"

# data on [-1,1]
N_TRAIN = 10000
N_VAL = 1500
N_TEST = 1500

DOMAIN_LOW = -1.0
DOMAIN_HIGH = 1.0

# network
WIDTH = 120
MAX_GRADES = 4
EPOCHS_PER_GRADE = [750, 1500, 3000, 6750]
BATCH_SIZE = 400
LEARNING_RATE = 1.0e-3
LR_DECAY = 0.9
LR_STEP_SIZE = 400
EVAL_EVERY = 1

# Underlying partition: P equal cells of [-1,1].
# Select one rule: midpoint, trapezoid, simpson, boole.
# Number of nodes is P, P+1, 2P+1, or 4P+1, respectively.
QUADRATURE_RULE = "boole"
P = 768

# grades where sigma_g is searched
SMOOTHING_BY_GRADE = {
    1: False,
    2: False,
    3: False,
    4: False,
}

SIGMA_INTERVAL = {
    1: (1e-5, 1e-2),
    2: (1e-5, 1e-2),
    3: (1e-5, 1e-2),
}

GOLDEN_MAX_ITER = 12
GOLDEN_SIGMA_TOL = 1.0e-5
SIGMA_DECIMALS = 8

# accept sigma_g only if validation error improves
ACCEPT_ONLY_IF_VAL_MSE_IMPROVES = True
ACCEPT_TOL = 0.0

RESULT_ROOT = "results_fixed1d"
