# Multigrade Deep Learning with Smoothing (S-MGDL)

This repository accompanies the paper **"Multigrade Deep Learning with Smoothing"** by Fuliang Lyu, Ronglong Fang, and Yuesheng Xu.

## Code

The repository contains the numerical experiments presented in the paper.

### Fixed-Grade Function Approximation

The folder `Fixed Grade Function Approximation/` contains the fixed-grade synthetic function approximation experiments:

- Fixed-grade 1D function approximation
- Fixed-grade 2D function approximation
- Fixed-grade 3D function approximation

### Image Denoising

The folder `Image Denoising/` contains the image-denoising experiments.

The implementation follows a three-stage experimental workflow:

1. **MGDL hyperparameter selection** using validation PSNR.
2. **S-MGDL smoothing-bandwidth selection** using validation PSNR, with the MGDL hyperparameters fixed.
3. **Final evaluation** of MGDL and S-MGDL using the selected parameters.

The three stages are provided separately to keep parameter selection and final evaluation clearly separated and reproducible.
