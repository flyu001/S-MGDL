# Image Denoising

This folder contains the code for the image-denoising experiments in the S-MGDL paper.

The experiments are organized into three stages to clearly separate hyperparameter selection, smoothing-bandwidth selection, and final test evaluation.

## Experimental Workflow

### Stage 0: MGDL Hyperparameter Selection

`Stage_0_MGDL_Hyperparameter_Selection/`

The MGDL training parameters

\[
(\eta,\beta,\lambda)
\]

are selected using the final validation PSNR.

Each candidate parameter combination is used to train the complete four-grade MGDL model, and the combination with the highest validation PSNR is selected.

No smoothing is used in this stage.

---

### Stage 1: S-MGDL Bandwidth Selection

`Stage_1_S_MGDL_Bandwidth_Selection/`

The MGDL parameters selected in Stage 0 are kept fixed.

Smoothing is applied after Grades 1--3, and the S-MGDL bandwidths

\[
(\sigma_1,\sigma_2,\sigma_3)
\]

are selected jointly using final validation PSNR.

The same bandwidth is used in both coordinate directions at each grade.

The nonsmoothed MGDL baseline in this stage reproduces the MGDL result from Stage 0 under the same experimental configuration.

---

### Stage 2: Final Evaluation

`Stage_2_Final_Evaluation/`

All parameters are fixed before the final evaluation:

- \((\eta,\beta,\lambda)\) are taken from Stage 0.
- \((\sigma_1,\sigma_2,\sigma_3)\) are taken from Stage 1.

No parameter selection is performed in this stage.

The final MGDL and S-MGDL results are evaluated using test PSNR.

## Reproducibility Across Stages

The stages are separated intentionally so that model selection and final evaluation remain independent.

Under the same experimental configuration,

\[
\text{Stage 0 MGDL}
=
\text{Stage 1 nonsmoothed MGDL},
\]

and

\[
\text{Stage 1 S-MGDL}
=
\text{Stage 2 fixed-bandwidth S-MGDL}.
\]

These consistency checks provide an additional verification of the implementation across the three stages.
