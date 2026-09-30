import os
from argparse import Namespace

import jax.numpy as jnp
import numpy as onp

import config as cfg
from mgdlmodel import data_setup, train_model, tv_matrix_form
from utils import masked_mse, mse, psnr, save_csv, save_json


def make_opt(lr_params, beta, lambd):
    opt = Namespace()
    opt.noise_level = cfg.NOISE_LEVEL
    opt.image = cfg.IMAGE
    opt.epoch = cfg.EPOCH
    opt.num_channel = cfg.NUM_CHANNEL
    opt.num_layer = cfg.NUM_LAYER
    opt.activation = cfg.ACTIVATION
    opt.lr_params = lr_params
    opt.beta = beta
    opt.lambd = lambd
    opt.grade = cfg.GRADE
    opt.alpha = cfg.ALPHA
    opt.interval = cfg.INTERVAL
    return opt


def run_mgdl(opt, data, result_dir=None):
    train_features = data["train_x"]
    train_y = data["train_y"]
    accumulation_img = jnp.zeros_like(data["img"])
    u = jnp.zeros_like(tv_matrix_form(data["img"]))
    scale_factor = 1.0

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

        train_features = history["train_features"]
        u = history["u"]
        scale_factor = history["NewscaleFactor"]

        # Keep the same nonsmoothed accumulation path as Stage 1.
        # The subtraction/addition order is intentionally preserved so that
        # a Stage 0 candidate can be reproduced by the Stage 1 baseline.
        raw_total = history["accumulation_img"]
        raw_correction = raw_total - accumulation_before
        accumulation_img = accumulation_before + raw_correction

        grade_rows.append({
            "grade": int(grade),
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


def save_best_artifacts(result, result_root):
    save_csv(
        result["grade_rows"],
        os.path.join(result_root, "best_mgdl_grade_summary.csv"),
    )
    onp.save(
        os.path.join(result_root, "best_mgdl_reconstruction.npy"),
        onp.asarray(result["reconstruction"]),
    )


def grid_search(data, result_root):
    table = []
    best = None
    count = 0
    csv_path = os.path.join(result_root, "mgdl_hyperparameter_grid.csv")

    for lr_params in cfg.LR_PARAMS_LIST:
        for beta in cfg.BETA_LIST:
            for lambd in cfg.LAMBD_LIST:
                count += 1
                opt = make_opt(lr_params, beta, lambd)
                result = run_mgdl(opt, data)

                row = {
                    "eval": count,
                    "lr_params": float(lr_params),
                    "beta": float(beta),
                    "lambd": float(lambd),
                    "final_val_psnr": result["final_val_psnr"],
                    "final_val_mse": result["final_val_mse"],
                    "final_eval_psnr": result["final_eval_psnr"],
                    "final_eval_mse": result["final_eval_mse"],
                }
                table.append(row)
                save_csv(table, csv_path)

                print(
                    "eval = %d, lr = %.6g, beta = %.6g, lambda = %.6g, val PSNR = %.8f"
                    % (
                        count,
                        lr_params,
                        beta,
                        lambd,
                        result["final_val_psnr"],
                    )
                )

                if best is None or result["final_val_psnr"] > best["final_val_psnr"]:
                    best = dict(row)
                    save_best_artifacts(result, result_root)
                    save_json(
                        best,
                        os.path.join(result_root, "best_mgdl_so_far.json"),
                    )

    return best


def main():
    base_opt = make_opt(
        cfg.LR_PARAMS_LIST[0],
        cfg.BETA_LIST[0],
        cfg.LAMBD_LIST[0],
    )
    data = data_setup(base_opt)

    noise_text = int(round(255.0 * cfg.NOISE_LEVEL))
    result_root = os.path.join(
        cfg.RESULT_ROOT,
        "%s_noise%d_mgdl_grid" % (cfg.IMAGE, noise_text),
    )
    os.makedirs(result_root, exist_ok=True)

    print("image =", cfg.IMAGE)
    print("noise level =", cfg.NOISE_LEVEL)
    print("training noisy pixels =", int(jnp.sum(data["train_mask"])))
    print("clean validation pixels =", int(jnp.sum(data["val_mask"])))
    print("LR_PARAMS_LIST =", cfg.LR_PARAMS_LIST)
    print("BETA_LIST =", cfg.BETA_LIST)
    print("LAMBD_LIST =", cfg.LAMBD_LIST)
    print(
        "number of combinations =",
        len(cfg.LR_PARAMS_LIST) * len(cfg.BETA_LIST) * len(cfg.LAMBD_LIST),
    )

    best = grid_search(data, result_root)

    save_json(
        best,
        os.path.join(result_root, "selected_by_clean_validation.json"),
    )

    print("")
    print("best MGDL hyperparameters selected by validation PSNR")
    print("lr_params =", best["lr_params"])
    print("beta =", best["beta"])
    print("lambd =", best["lambd"])
    print("validation PSNR =", best["final_val_psnr"])


if __name__ == "__main__":
    main()
