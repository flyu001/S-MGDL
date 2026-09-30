SEED = 0

# Senior-code settings
IMAGE = "butterfly"
NOISE_LEVEL = 20.0 / 255.0
GRADE = 4
NUM_LAYER = 3
NUM_CHANNEL = 128
EPOCH = 20000
LR_PARAMS = 5e-3
BETA = 5
LAMBD = 5e-2
ALPHA = 0.99
ACTIVATION = "relu"
INTERVAL = 100

# Clean validation:
# remove the image-dependent border, divide the remaining region into
# 3 x 3 blocks, and use the center of each block as a clean validation point.
VALIDATION_BLOCK = 3
VALIDATION_BORDER = {
    "butterfly": 1,
    "male": 2,
}[IMAGE]

# Fixed-grid Boole output smoothing
DOMAIN_LOW = 0.0
DOMAIN_HIGH = 1.0
SUPPORT_AXIS_SIZE = 1281  # 4I+1

# Hierarchical joint Powell search:
#   sigma_1 = s,
#   sigma_2 = r_1 sigma_1,
#   sigma_3 = r_2 sigma_2,
# with 0 < r_1,r_2 < 1, hence sigma_1 > sigma_2 > sigma_3.
SIGMA1_INTERVAL = (8e-4, 8e-3)
RATIO_INTERVAL = (0.1, 0.999)

POWELL_MAX_ITER = 120
POWELL_MAX_EVAL = 120
POWELL_XTOL = 1.0e-4
POWELL_FTOL = 1.0e-10
SIGMA_DECIMALS = 8

# If the best positive-sigma candidate does not improve validation PSNR,
# retain the nonsmoothed baseline.
ACCEPT_TOL = 0.0

RESULT_ROOT = "results_image_cleanval_powell_20_80e4_80e3"
