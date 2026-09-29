import torch

# Easy 2D fixed-depth S-MGDL. Original ReLU network.
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

EXPERIMENT_NAME = "easy2d_smooth_loss"

# Original train / validation / test Cartesian grids.
TRAIN_AXIS_SIZE = 500
VAL_AXIS_SIZE = 300
TEST_AXIS_SIZE = 300
VAL_GRID_SHIFT = 1.0 / 3.0
DOMAIN_LOW = -1.0
DOMAIN_HIGH = 1.0

WIDTH = 190
MAX_GRADES = 3
EPOCHS_PER_GRADE = [75, 150, 475]
BATCH_SIZE = 1000
LEARNING_RATE = 1.0e-3
LR_DECAY = 0.9
LR_STEP_SIZE = 52
EVAL_EVERY = 1

# P = equal coarse partitions per axis, as in the paper.
# Support nodes per axis: P, P+1, 2P+1, 4P+1, respectively.
QUADRATURE_RULE = "boole"  # midpoint, trapezoid, simpson, boole
P = 320

# Original Boole fixed-bandwidth run (no search).
# To reproduce the original search runs, see README.md.
FIXED_SMOOTHING_BY_GRADE = {
    1: (0.01568922, 0.01071002),
    2: (0.00733362, 0.00402762),
}
POWELL_SEARCH_GRADE = None

# Decreasing scale-search intervals discussed in the paper.
SIGMA_SCALE_INTERVAL = {
    1: (1.0e-2, 3.0e-2),
    2: (3.0e-3, 1.0e-2),
    3: (7.0e-3, 1.4e-2),
}
SIGMA_RATIO_INTERVAL = {
    1: (1.0 / 3.0, 3.0),
    2: (1.0 / 3.0, 3.0),
    3: (1.0 / 4.0, 4.0),
}
POWELL_MAX_ITER = 80
POWELL_XTOL = 1.0e-4
POWELL_FTOL = 1.0e-14
SIGMA_DECIMALS = 8
ACCEPT_TOL = 0.0

RESULT_ROOT = "results_easy2d_smooth_loss"
