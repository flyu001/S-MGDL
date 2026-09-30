# Stage 1: S-MGDL Bandwidth Selection

This folder contains the validation-based smoothing-bandwidth selection code for S-MGDL in the image-denoising experiments.

The MGDL training parameters

\[
(\eta,\beta,\lambda)
\]

are fixed to the values selected in Stage 0 and are not retuned in this stage.

Smoothing is applied after Grades 1--3. The bandwidths

\[
(\sigma_1,\sigma_2,\sigma_3)
\]

are selected jointly using final validation PSNR.

The same bandwidth is used in both coordinate directions at each grade. Powell's method is used for the joint bandwidth search.

The nonsmoothed MGDL baseline is also evaluated under the same configuration and reproduces the Stage 0 MGDL result.
