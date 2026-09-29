import torch

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

EXPERIMENT_NAME = "easy3d_smooth"

TRAIN_AXIS_SIZE = 125
VAL_AXIS_SIZE = 80
TEST_AXIS_SIZE = 80
VAL_GRID_SHIFT = 1.0 / 3.0
DOMAIN_LOW = -1.0
DOMAIN_HIGH = 1.0

WIDTH = 145
MAX_GRADES = 3
EPOCHS_PER_GRADE = [400, 250, 150]
BATCH_SIZE = 3000
LEARNING_RATE = 1.0e-3
LR_DECAY = 0.9
LR_STEP_SIZE = 73
EVAL_EVERY = 50

# The only quadrature selector. P is the number of coarse panels per axis.
QUADRATURE_RULE = "midpoint"  # midpoint, trapezoid, simpson, boole
P = 64

if QUADRATURE_RULE == "midpoint":
    SUPPORT_AXIS_SIZE = P
elif QUADRATURE_RULE == "trapezoid":
    SUPPORT_AXIS_SIZE = P + 1
elif QUADRATURE_RULE == "simpson":
    SUPPORT_AXIS_SIZE = 2 * P + 1
elif QUADRATURE_RULE == "boole":
    SUPPORT_AXIS_SIZE = 4 * P + 1
else:
    raise ValueError("Unknown quadrature rule.")

FIXED_SMOOTHING_BY_GRADE = {}
POWELL_SEARCH_GRADE = 1

# Grade 1 exactly matches both uploaded Fixed 3D configurations.
# Grade 2 uses the specified smaller-scale interval for subsequent search.
SIGMA_SCALE_INTERVAL = {
    1: (3.0e-3, 3.0e-2),
    2: (1.0e-3, 1.0e-2),
}
SIGMA_RATIO_INTERVAL = {
    1: (1.0 / 3.0, 3.0),
    2: (1.0 / 3.0, 3.0),
}

POWELL_MAX_ITER = 120
POWELL_XTOL = 1.0e-4
POWELL_FTOL = 1.0e-14
SIGMA_DECIMALS = 8
ACCEPT_TOL = 0.0
SAVE_OUTPUT = True

RESULT_ROOT = "results_easy3d_smooth"
