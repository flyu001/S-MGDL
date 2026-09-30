import os
from argparse import Namespace

import imageio.v2 as imageio
import jax.numpy as jnp
import numpy as onp

import config as cfg
from mgdlmodel import data_setup, snn, train_model, tv_matrix_form
from smoother import smooth_2d, support_grid
from utils import masked_mse, mse, psnr, save_csv, save_json


def make_opt():
    opt = Namespace()
    opt.noise_level = cfg.NOISE_LEVEL
    opt.image = cfg.IMAGE
    opt.epoch = cfg.EPOCH
    opt.num_channel = cfg.NUM_CHANNEL
    opt.num_layer = cfg.NUM_LAYER
    opt.activation = cfg.ACTIVATION
    opt.lr_params = cfg.LR_PARAMS
    opt.beta = cfg.BETA
    opt.lambd = cfg.LAMBD
    opt.grade = cfg.GRADE
    opt.alpha = cfg.ALPHA
    opt.interval = cfg.INTERVAL
    return opt


def feature_map(params, inputs, opt):
    _, _, feature_fn = snn(1, 1.0, 0.0, opt)
    return feature_fn(params, inputs)


def grade_correction(params, inputs, scale_factor, opt):
    _, model_fn, _ = snn(1, scale_factor, 0.0, opt)
    return model_fn(params, inputs)



def save_image(values, path):
    values = onp.asarray(values)
    imageio.imwrite(
        path,
        onp.uint8(onp.clip(values, 0.0, 1.0) * 255.0),
    )


def fixed_sigma_dict():
    sigma1 = float(cfg.SIGMA1)
    sigma2 = float(cfg.SIGMA2)
    sigma3 = float(cfg.SIGMA3)

    if not (sigma1 > sigma2 > sigma3 > 0.0):
        raise ValueError("Require SIGMA1 > SIGMA2 > SIGMA3 > 0.")

    return {
        1: (sigma1, sigma1),
        2: (sigma2, sigma2),
        3: (sigma3, sigma3),
    }



def run_mgdl(opt, data, sigma_by_grade, result_dir):
    """Run one four-grade path with fixed smoothing parameters.

    The training/noise/validation/smoothing mathematics is the same as the
    clean-validation Powell paper version.
    """
    train_features = data["train_x"]
    train_y = data["train_y"]
    accumulation_img = jnp.zeros_like(data["img"])
    u = jnp.zeros_like(tv_matrix_form(data["img"]))
    scale_factor = 1.0

    support_axis = None
    support_features = None
    if sigma_by_grade:
        support_axis, support_features = support_grid(dtype=data["train_x"].dtype)

    grade_rows = []

    for grade in range(1, opt.grade + 1):
        Phi_before = accumulation_img
        scale_before = scale_factor

        history = train_model(
            grade,
            data,
            scale_before,
            [train_features, train_y],
            Phi_before,
            u,
            opt,
        )

        train_features = history["train_features"]
        u = history["u"]
        senior_raw_scale_next = history["NewscaleFactor"]

        raw_total = history["accumulation_img"]
        u_raw = raw_total - Phi_before
        params = history["params"]

        if grade in sigma_by_grade:
            sigma_x, sigma_y = sigma_by_grade[grade]
            support_raw = grade_correction(
                params,
                support_features,
                scale_before,
                opt,
            )
            u_used = smooth_2d(
                support_raw,
                jnp.linspace(0.0, 1.0, train_y.shape[1], endpoint=False),
                jnp.linspace(0.0, 1.0, train_y.shape[0], endpoint=False),
                sigma_x,
                sigma_y,
                support_axis,
            )
            smoothed = True
        else:
            sigma_x = 0.0
            sigma_y = 0.0
            u_used = u_raw
            smoothed = False

        accumulation_img = Phi_before + u_used

        # Same no-boundary-prox rule as the search version.
        if smoothed:
            scale_factor = jnp.sqrt(
                masked_mse(
                    accumulation_img,
                    train_y,
                    data["train_mask"],
                )
            )
        else:
            scale_factor = senior_raw_scale_next

        if support_features is not None:
            support_features = feature_map(params, support_features, opt)

        grade_rows.append({
            "grade": int(grade),
            "smoothed": bool(smoothed),
            "sigma": float(sigma_x),
            "senior_raw_scaleFactor_next": float(senior_raw_scale_next),
            "used_scaleFactor_next": float(scale_factor),
            "val_psnr_after_grade": psnr(
                accumulation_img,
                data["img"],
                data["val_mask"],
            ),
            "test_psnr_after_grade": psnr(
                accumulation_img,
                data["img"],
                data["train_mask"],
            ),
            "full_psnr_after_grade": psnr(
                accumulation_img,
                data["img"],
            ),
        })


    result = {
        "final_val_mse": float(
            masked_mse(accumulation_img, data["img"], data["val_mask"])
        ),
        "final_val_psnr": psnr(
            accumulation_img, data["img"], data["val_mask"]
        ),
        "final_test_mse": float(
            masked_mse(accumulation_img, data["img"], data["train_mask"])
        ),
        "final_test_psnr": psnr(
            accumulation_img, data["img"], data["train_mask"]
        ),
        # Keep full-image PSNR so the fixed-sigma rerun can be checked directly
        # against the previous Powell-search CSV.
        "final_full_mse": float(mse(accumulation_img, data["img"])),
        "final_full_psnr": psnr(accumulation_img, data["img"]),
        "reconstruction": accumulation_img,
        "grade_rows": grade_rows,
    }

    os.makedirs(result_dir, exist_ok=True)
    save_csv(grade_rows, os.path.join(result_dir, "grade_summary.csv"))
    save_json(
        {
            "final_val_mse": result["final_val_mse"],
            "final_val_psnr": result["final_val_psnr"],
            "final_test_mse": result["final_test_mse"],
            "final_test_psnr": result["final_test_psnr"],
            "final_full_mse": result["final_full_mse"],
            "final_full_psnr": result["final_full_psnr"],
        },
        os.path.join(result_dir, "final_result.json"),
    )
    onp.save(
        os.path.join(result_dir, "reconstruction.npy"),
        onp.asarray(accumulation_img),
    )

    return result


def main():
    opt = make_opt()
    data = data_setup(opt)

    noise_text = int(round(255.0 * cfg.NOISE_LEVEL))
    result_root = os.path.join(
        cfg.RESULT_ROOT,
        "%s_noise%d_fixedsigma" % (cfg.IMAGE, noise_text),
    )
    os.makedirs(result_root, exist_ok=True)

    figures = os.path.join(result_root, "paper_figures")
    os.makedirs(figures, exist_ok=True)

    # The "test" mask is exactly the complement of the clean validation mask.
    noisy_val_mse = float(
        masked_mse(data["train_y"], data["img"], data["val_mask"])
    )
    noisy_val_psnr = psnr(
        data["train_y"], data["img"], data["val_mask"]
    )
    noisy_test_mse = float(
        masked_mse(data["train_y"], data["img"], data["train_mask"])
    )
    noisy_test_psnr = psnr(
        data["train_y"], data["img"], data["train_mask"]
    )
    noisy_full_mse = float(mse(data["train_y"], data["img"]))
    noisy_full_psnr = psnr(data["train_y"], data["img"])

    save_image(
        data["train_y"],
        os.path.join(figures, "noisy.png"),
    )

    # 1. Nonsmoothed MGDL under exactly the same data split.
    nonsmooth = run_mgdl(
        opt,
        data,
        {},
        os.path.join(result_root, "nonsmooth"),
    )
    save_image(
        nonsmooth["reconstruction"],
        os.path.join(figures, "nonsmooth.png"),
    )

    # 2. Fixed selected smoothing parameters.
    sigmas = fixed_sigma_dict()
    smooth = run_mgdl(
        opt,
        data,
        sigmas,
        os.path.join(result_root, "smooth"),
    )
    save_image(
        smooth["reconstruction"],
        os.path.join(figures, "smooth.png"),
    )

    sigma1 = float(cfg.SIGMA1)
    sigma2 = float(cfg.SIGMA2)
    sigma3 = float(cfg.SIGMA3)

    rows = [
        {
            "method": "noisy",
            "sigma1": "",
            "sigma2": "",
            "sigma3": "",
            "val_psnr": noisy_val_psnr,
            "test_psnr": noisy_test_psnr,
            "full_psnr": noisy_full_psnr,
        },
        {
            "method": "nonsmooth",
            "sigma1": 0.0,
            "sigma2": 0.0,
            "sigma3": 0.0,
            "val_psnr": nonsmooth["final_val_psnr"],
            "test_psnr": nonsmooth["final_test_psnr"],
            "full_psnr": nonsmooth["final_full_psnr"],
        },
        {
            "method": "smooth",
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma3": sigma3,
            "val_psnr": smooth["final_val_psnr"],
            "test_psnr": smooth["final_test_psnr"],
            "full_psnr": smooth["final_full_psnr"],
        },
    ]
    save_csv(rows, os.path.join(result_root, "paper_metrics.csv"))

    save_json(
        {
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma3": sigma3,
            "ratio1": sigma2 / sigma1,
            "ratio2": sigma3 / sigma2,
            "noisy_val_psnr": noisy_val_psnr,
            "noisy_test_psnr": noisy_test_psnr,
            "noisy_full_psnr": noisy_full_psnr,
            "nonsmooth_val_psnr": nonsmooth["final_val_psnr"],
            "nonsmooth_test_psnr": nonsmooth["final_test_psnr"],
            "nonsmooth_full_psnr": nonsmooth["final_full_psnr"],
            "smooth_val_psnr": smooth["final_val_psnr"],
            "smooth_test_psnr": smooth["final_test_psnr"],
            "smooth_full_psnr": smooth["final_full_psnr"],
        },
        os.path.join(result_root, "paper_summary.json"),
    )

    print("")
    print("fixed sigmas")
    print("sigma1 =", sigma1)
    print("sigma2 =", sigma2)
    print("sigma3 =", sigma3)
    print("")
    print("noisy test PSNR = %.6f" % noisy_test_psnr)
    print("nonsmooth test PSNR = %.6f" % nonsmooth["final_test_psnr"])
    print("smooth test PSNR = %.6f" % smooth["final_test_psnr"])
    print("")
    print("smooth full PSNR = %.6f" % smooth["final_full_psnr"])
    print("(use full PSNR to check replication against the previous search CSV)")


if __name__ == "__main__":
    main()
