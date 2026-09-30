import math
import os
import shutil
from argparse import Namespace

import jax.numpy as jnp
import numpy as onp
from scipy.optimize import minimize

import config as cfg
from mgdlmodel import data_setup, snn, train_model, tv_matrix_form
from smoother import smooth_2d, support_grid
from utils import masked_mse, mse, psnr, save_csv, save_json


def round_sigma(value):
    return round(float(value), cfg.SIGMA_DECIMALS)


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


def normalize_sigma_dict(sigma_by_grade, total_grades):
    active = {}
    for g, pair in sigma_by_grade.items():
        g = int(g)
        if g >= int(total_grades):
            raise ValueError("Only Grades 1,...,G-1 may be smoothed; sigma_G=0.")

        sigma_x = float(pair[0])
        sigma_y = float(pair[1])
        if sigma_x < 0.0 or sigma_y < 0.0:
            raise ValueError("Smoothing sigmas must be nonnegative.")
        if sigma_x == 0.0 and sigma_y == 0.0:
            continue
        if sigma_x == 0.0 or sigma_y == 0.0:
            raise ValueError("Use either (0,0) for S_0=I or positive sigma_x,sigma_y.")
        active[g] = (sigma_x, sigma_y)
    return active


def run_mgdl(opt, data, sigma_by_grade, result_dir=None):
    """The ratio-Powell base run with clean held-out validation.

    The noisy image is generated exactly as in the supplied
    residualtrajectory_powell_ratio_final base.

    Validation locations are the clean centers of 3 x 3 blocks after removing
    the image-dependent border.  Their noisy labels do not enter the fidelity
    term.  The senior TV/proximal term is unchanged.

    At a smoothed grade g,

        f_g^raw  = Phi_g^raw - Phi_{g-1},
        f_g^used = S_{sigma_g} f_g^raw,
        Phi_g^used = Phi_{g-1} + f_g^used.

    No proximal update is inserted at a grade boundary.  Grade 4 is terminal
    and is not smoothed.
    """
    if int(opt.grade) != 4:
        raise ValueError(
            "This experiment is fixed to four grades: "
            "sigma_1,sigma_2,sigma_3, sigma_4=0."
        )

    sigma_by_grade = normalize_sigma_dict(sigma_by_grade, opt.grade)

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
        accumulation_before = accumulation_img
        scale_before = scale_factor

        history = train_model(
            grade,
            data,
            scale_before,
            [train_features, train_y],
            accumulation_before,
            u,
            opt,
        )

        # Same inherited senior states as the supplied ratio-Powell base.
        train_features = history["train_features"]
        u = history["u"]
        senior_raw_scale_next = history["NewscaleFactor"]

        raw_total = history["accumulation_img"]
        raw_correction = raw_total - accumulation_before
        params = history["params"]

        if grade in sigma_by_grade:
            sigma_x, sigma_y = sigma_by_grade[grade]
            support_raw_correction = grade_correction(
                params,
                support_features,
                scale_before,
                opt,
            )
            used_correction = smooth_2d(
                support_raw_correction,
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
            used_correction = raw_correction
            smoothed = False

        accumulation_img = accumulation_before + used_correction

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
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "senior_raw_scaleFactor_next": float(senior_raw_scale_next),
            "used_scaleFactor_next": float(scale_factor),
            "val_psnr_after_grade": psnr(
                accumulation_img,
                data["img"],
                data["val_mask"],
            ),
            "eval_psnr_after_grade": psnr(
                accumulation_img,
                data["img"],
            ),
        })

    result = {
        "final_val_mse": float(
            masked_mse(
                accumulation_img,
                data["img"],
                data["val_mask"],
            )
        ),
        "final_val_psnr": psnr(
            accumulation_img,
            data["img"],
            data["val_mask"],
        ),
        "final_eval_mse": float(mse(accumulation_img, data["img"])),
        "final_eval_psnr": psnr(accumulation_img, data["img"]),
        "reconstruction": accumulation_img,
        "grade_rows": grade_rows,
    }

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        save_csv(
            grade_rows,
            os.path.join(result_dir, "grade_summary.csv"),
        )
        save_json(
            {
                "final_val_mse": result["final_val_mse"],
                "final_val_psnr": result["final_val_psnr"],
                "final_eval_mse": result["final_eval_mse"],
                "final_eval_psnr": result["final_eval_psnr"],
            },
            os.path.join(result_dir, "final_result.json"),
        )
        onp.save(
            os.path.join(result_dir, "reconstruction.npy"),
            onp.asarray(result["reconstruction"]),
        )

    return result


def search_bounds():
    sigma_low, sigma_high = cfg.SIGMA1_INTERVAL
    ratio_low, ratio_high = cfg.RATIO_INTERVAL

    if not (0.0 < sigma_low < sigma_high):
        raise ValueError("SIGMA1_INTERVAL must satisfy 0 < low < high.")
    if not (0.0 < ratio_low < ratio_high < 1.0):
        raise ValueError("RATIO_INTERVAL must satisfy 0 < low < high < 1.")

    return [
        (math.log(sigma_low), math.log(sigma_high)),
        (ratio_low, ratio_high),
        (ratio_low, ratio_high),
    ]


def search_start():
    sigma_low, sigma_high = cfg.SIGMA1_INTERVAL
    ratio_low, ratio_high = cfg.RATIO_INTERVAL

    sigma_start = math.sqrt(sigma_low * sigma_high)
    ratio_start = 0.5 * (ratio_low + ratio_high)

    return onp.asarray(
        [math.log(sigma_start), ratio_start, ratio_start],
        dtype=float,
    )


def hierarchical_sigmas(x):
    """Map Powell variables to sigma_1 > sigma_2 > sigma_3 > 0."""
    s = math.exp(float(x[0]))
    r1 = float(x[1])
    r2 = float(x[2])

    sigma1 = round_sigma(s)
    sigma2 = round_sigma(s * r1)
    sigma3 = round_sigma(s * r1 * r2)

    unit = 10.0 ** (-cfg.SIGMA_DECIMALS)
    sigma2 = min(sigma2, sigma1 - unit)
    sigma3 = min(sigma3, sigma2 - unit)

    if not (sigma1 > sigma2 > sigma3 > 0.0):
        raise ValueError("Rounded hierarchical sigmas lost strict ordering.")

    sigma_dict = {
        1: (sigma1, sigma1),
        2: (sigma2, sigma2),
        3: (sigma3, sigma3),
    }
    return sigma_dict, sigma1, r1, sigma2, r2, sigma3


def save_best_artifacts(result, result_root):
    save_csv(
        result["grade_rows"],
        os.path.join(result_root, "best_joint_grade_summary.csv"),
    )
    onp.save(
        os.path.join(result_root, "best_joint_reconstruction.npy"),
        onp.asarray(result["reconstruction"]),
    )


def joint_powell_search(opt, data, result_root, baseline):
    baseline_row = {
        "eval": 0,
        "sigma_1": 0.0,
        "ratio_1": "",
        "sigma_2": 0.0,
        "ratio_2": "",
        "sigma_3": 0.0,
        "final_val_psnr": baseline["final_val_psnr"],
        "final_val_mse": baseline["final_val_mse"],
        "final_eval_psnr": baseline["final_eval_psnr"],
        "final_eval_mse": baseline["final_eval_mse"],
    }

    table = [baseline_row]
    csv_path = os.path.join(result_root, "powell_joint_clean_validation.csv")
    save_csv(table, csv_path)

    cache = {}
    best = None
    count = 0

    def objective(x):
        nonlocal best, count

        sigma_dict, sigma1, r1, sigma2, r2, sigma3 = hierarchical_sigmas(x)
        key = (sigma1, sigma2, sigma3)
        if key in cache:
            return cache[key]

        count += 1
        result = run_mgdl(opt, data, sigma_dict)

        row = {
            "eval": count,
            "sigma_1": sigma1,
            "ratio_1": r1,
            "sigma_2": sigma2,
            "ratio_2": r2,
            "sigma_3": sigma3,
            "final_val_psnr": result["final_val_psnr"],
            "final_val_mse": result["final_val_mse"],
            "final_eval_psnr": result["final_eval_psnr"],
            "final_eval_mse": result["final_eval_mse"],
        }
        table.append(row)
        save_csv(table, csv_path)

        if best is None or result["final_val_psnr"] > best["final_val_psnr"]:
            best = dict(row)
            save_best_artifacts(result, result_root)
            save_json(
                best,
                os.path.join(result_root, "best_joint_so_far.json"),
            )

        # scipy.optimize.minimize minimizes, so maximize validation PSNR
        # by minimizing its negative.
        value = -result["final_val_psnr"]
        cache[key] = value
        return value

    minimize(
        objective,
        search_start(),
        method="Powell",
        bounds=search_bounds(),
        options={
            "maxiter": cfg.POWELL_MAX_ITER,
            "maxfev": cfg.POWELL_MAX_EVAL,
            "xtol": cfg.POWELL_XTOL,
            "ftol": cfg.POWELL_FTOL,
            "disp": False,
        },
    )

    return best


def main():
    opt = make_opt()
    data = data_setup(opt)

    noise_text = int(round(255.0 * cfg.NOISE_LEVEL))
    result_root = os.path.join(
        cfg.RESULT_ROOT,
        f"{cfg.IMAGE}_noise{noise_text}_cleanval_powell_ratio",
    )
    os.makedirs(result_root, exist_ok=True)

    print("image =", cfg.IMAGE)
    print("noise level =", cfg.NOISE_LEVEL)
    print("training noisy pixels =", int(jnp.sum(data["train_mask"])))
    print("clean validation pixels =", int(jnp.sum(data["val_mask"])))
    print("LR_PARAMS =", cfg.LR_PARAMS)
    print("BETA =", cfg.BETA)
    print("LAMBD =", cfg.LAMBD)

    # Nonsmoothed baseline under exactly the same train/validation split.
    baseline = run_mgdl(
        opt,
        data,
        {},
        os.path.join(result_root, "baseline_sigma000"),
    )

    baseline_row = {
        "sigma_1": 0.0,
        "ratio_1": 0.0,
        "sigma_2": 0.0,
        "ratio_2": 0.0,
        "sigma_3": 0.0,
        "final_val_psnr": baseline["final_val_psnr"],
        "final_val_mse": baseline["final_val_mse"],
        "final_eval_psnr": baseline["final_eval_psnr"],
        "final_eval_mse": baseline["final_eval_mse"],
    }
    save_json(
        baseline_row,
        os.path.join(result_root, "baseline_sigma000.json"),
    )

    print("nonsmooth validation PSNR =", baseline["final_val_psnr"])
    print("nonsmooth evaluation PSNR =", baseline["final_eval_psnr"])

    best = joint_powell_search(
        opt,
        data,
        result_root,
        baseline,
    )
    save_json(
        best,
        os.path.join(result_root, "powell_joint_clean_validation_best.json"),
    )

    selected = dict(best)
    selected["selection"] = "joint_positive"
    selected_source = os.path.join(
        result_root,
        "best_joint_reconstruction.npy",
    )

    if baseline["final_val_psnr"] + cfg.ACCEPT_TOL >= best["final_val_psnr"]:
        selected = dict(baseline_row)
        selected["selection"] = "nonsmooth_baseline_sigma000"
        selected_source = os.path.join(
            result_root,
            "baseline_sigma000",
            "reconstruction.npy",
        )

    save_json(
        selected,
        os.path.join(result_root, "selected_by_clean_validation.json"),
    )

    if os.path.exists(selected_source):
        shutil.copyfile(
            selected_source,
            os.path.join(result_root, "selected_reconstruction.npy"),
        )


if __name__ == "__main__":
    main()
