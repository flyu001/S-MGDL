SEED = 0

# Senior-code settings
IMAGE = "butterfly"
NOISE_LEVEL = 10.0 / 255.0
GRADE = 4
NUM_LAYER = 3
NUM_CHANNEL = 128
EPOCH = 20000
LR_PARAMS = 5.0e-3
BETA = 5e-1
LAMBD = 1.0e-2
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

# Fixed smoothing parameters.
# Replace these three numbers by the validation-selected row from
# powell_joint_clean_validation.csv if your selected values differ.
SIGMA1 = 0.0047182
SIGMA2 = 0.00302055  
SIGMA3 = 0.00134056

RESULT_ROOT = "results_image_fixedsigma_10"
