SEED = 0

# Image-denoising settings
IMAGE = "butterfly"
NOISE_LEVEL = 20.0 / 255.0
GRADE = 4
NUM_LAYER = 3
NUM_CHANNEL = 128
EPOCH = 20000
ALPHA = 0.99
ACTIVATION = "relu"
INTERVAL = 100

# MGDL hyperparameter grid.
# The current paper uses 3 x 4 x 4 = 48 combinations.
LR_PARAMS_LIST = [1e-3, 5e-3, 1e-2]
BETA_LIST = [1e-1, 5e-1, 1.0, 5.0]
LAMBD_LIST = [5e-3, 1e-2, 5e-2, 1e-1]

# Clean validation:
# remove the image-dependent border, divide the remaining region into
# 3 x 3 blocks, and use the center of each block as a clean validation point.
VALIDATION_BLOCK = 3
VALIDATION_BORDER = {
    "butterfly": 1,
    "male": 2,
}[IMAGE]

RESULT_ROOT = "results_image_mgdl_hyperparameter_selection"
