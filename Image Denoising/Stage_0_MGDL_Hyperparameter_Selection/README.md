# Stage 0: MGDL Hyperparameter Selection

This folder contains the validation-based hyperparameter selection code for MGDL in the image-denoising experiments.

The training parameters

\[
(\eta,\beta,\lambda)
\]

are selected by final validation PSNR.

For each candidate parameter combination, the complete four-grade MGDL model is trained. The combination with the highest final validation PSNR is selected and then used as the fixed MGDL configuration in Stage 1.

No smoothing is used in this stage.
