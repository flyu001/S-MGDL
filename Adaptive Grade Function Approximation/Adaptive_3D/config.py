import torch

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

EXPERIMENT_NAME = "hard3d_autodepth_boole"

TRAIN_AXIS_SIZE = 125
VAL_AXIS_SIZE = 90
TEST_AXIS_SIZE = 90
VAL_GRID_SHIFT = 1.0 / 3.0
DOMAIN_LOW = -1.0
DOMAIN_HIGH = 1.0

WIDTH = 150
MAX_GRADES = 6
EPOCHS_PER_GRADE = 200
BATCH_SIZE = 3125
LEARNING_RATE = 1.0e-3
LR_DECAY = 0.9
LR_STEP_SIZE = 40
MIN_VAL_IMPROVEMENT = 0.0
EVAL_EVERY = 100

# Tensor-product composite Boole support: SUPPORT_AXIS_SIZE = 4I + 1
SUPPORT_AXIS_SIZE = 769

# Start the sequential smoothing search from Grade 1.
FIXED_SMOOTHING_BY_GRADE = {}
POWELL_SEARCH_GRADE = 1

# Scale intervals used in the adaptive 3D experiment.
SIGMA_SCALE_INTERVAL = {
    1: (3e-3, 3e-2),
    2: (1e-3, 1e-2),
    3: (1e-3, 1e-2),
}

SIGMA_RATIO_INTERVAL = {
    1: (1.0 / 3.0, 3.0),
    2: (1.0 / 3.0, 3.0),
    3: (1.0 / 3.0, 3.0),
}

POWELL_MAX_ITER = 120
POWELL_XTOL = 1.0e-4
POWELL_FTOL = 1.0e-14
SIGMA_DECIMALS = 8
SAVE_GRADE_OUTPUT_TABLES = False
RESULT_ROOT = "results_hard3d_autodepth_boole"
