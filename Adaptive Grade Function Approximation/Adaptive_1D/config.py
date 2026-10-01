import torch

# Adaptive 1D MGDL with Boole smoothing

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

EXPERIMENT_NAME = "hard1d_autodepth_680_boole"

# data on [-1, 1]
N_TRAIN = 60000
N_VAL = 30000
N_TEST = 30000
DOMAIN_LOW = -1.0
DOMAIN_HIGH = 1.0

# network
WIDTH = 120
MAX_GRADES = 20
EPOCHS_PER_GRADE = 10000
BATCH_SIZE = 3000
LEARNING_RATE = 1.0e-3
LR_DECAY = 0.9
LR_STEP_SIZE = 680
MIN_VAL_IMPROVEMENT = 0.0
EVAL_EVERY = 1000

# Boole grid for S_sigma: SUPPORT_GRID_SIZE = 4I + 1
SUPPORT_GRID_SIZE = 8193

# No smoothing parameters are fixed initially
FIXED_SMOOTHING_BY_GRADE = {}

# Search Grade 1 first
GOLDEN_SEARCH_GRADE = 1

# Golden-search intervals used for the adaptive 1D experiment
SIGMA_INTERVAL = {
    1: (1e-5, 1e-2),
    2: (1e-5, 7e-3),
    3: (1e-5, 4e-3),
    4: (1e-5, 6e-4),
}

GOLDEN_MAX_ITER = 12
GOLDEN_SIGMA_TOL = 1.0e-5
SIGMA_DECIMALS = 8
ACCEPT_TOL = 0.0

SAVE_GRADE_OUTPUT_TABLES = False
RESULT_ROOT = "results_hard1d_autodepth_680_boole"
