# Stage 2: Final Evaluation

This folder contains the final evaluation code for the image-denoising experiments.

The MGDL training parameters

\[
(\eta,\beta,\lambda)
\]

are fixed to the values selected in Stage 0.

The S-MGDL smoothing bandwidths

\[
(\sigma_1,\sigma_2,\sigma_3)
\]

are fixed to the values selected in Stage 1.

No hyperparameter or bandwidth selection is performed in this stage.

MGDL and S-MGDL are rerun using the fixed selected parameters, and the final test PSNR is reported for comparison.

Under the same experimental configuration, the fixed-bandwidth S-MGDL reconstruction in this stage reproduces the corresponding S-MGDL candidate obtained in Stage 1.
