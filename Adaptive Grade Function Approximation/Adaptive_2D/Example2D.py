import math
import os
import shutil
import time

import numpy as np
import torch
from scipy.optimize import minimize

import config as cfg
# Adaptive 2D Example 1
from functions import target_hard_2d as target_function

# Adaptive 2D Example 2
# from functions import target_hard2_2d as target_function
from mgdl import AutoDepthMGDL2D
from trainer import train_autodepth
from utils import (
    cartesian_grid_2d,
    midpoint_grid_2d,
    shifted_grid_2d,
    ensure_dir,
    normalize_sigmas,
    save_csv,
    save_json,
    set_seed,
    sigma_pair_name,
    sigmas_for_json,
)





def round_sigma(sigma):
    return round(float(sigma), int(cfg.SIGMA_DECIMALS))


def pair_to_scale_ratio(sigma_x, sigma_y):
    sigma = math.sqrt(float(sigma_x) * float(sigma_y))
    rho = float(sigma_x) / float(sigma_y)
    return sigma, rho


def scale_ratio_to_pair(sigma, rho):
    sigma_x = float(sigma) * math.sqrt(float(rho))
    sigma_y = float(sigma) / math.sqrt(float(rho))
    return round_sigma(sigma_x), round_sigma(sigma_y)


def train_Phi_autodepth(data, smoothing_by_grade, result_root, run_label):
    set_seed(cfg.SEED)

    model = AutoDepthMGDL2D(width=cfg.WIDTH, input_dim=2).to(
        device=data["x_train"].device,
        dtype=cfg.DTYPE,
    )

    result_dir = os.path.join(result_root, run_label)
    result = train_autodepth(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        data["x_test"],
        data["y_test"],
        smoothing_by_grade,
        result_dir,
    )
    result["run_label"] = run_label
    result["smoothing_by_grade"] = sigmas_for_json(smoothing_by_grade)
    save_json(result, os.path.join(result_dir, "final_metrics.json"))
    return result


def powell_bounds(grade):
    sigma_left, sigma_right = cfg.SIGMA_SCALE_INTERVAL[int(grade)]
    rho_left, rho_right = cfg.SIGMA_RATIO_INTERVAL[int(grade)]
    return [
        (math.log(float(sigma_left)), math.log(float(sigma_right))),
        (math.log(float(rho_left)), math.log(float(rho_right))),
    ]


def powell_start(grade):
    bounds = powell_bounds(grade)
    return np.array(
        [
            0.5 * (bounds[0][0] + bounds[0][1]),
            0.5 * (bounds[1][0] + bounds[1][1]),
        ],
        dtype=np.float64,
    )


def sigma_from_powell_vector(z):
    sigma = math.exp(float(z[0]))
    rho = math.exp(float(z[1]))
    sigma_x, sigma_y = scale_ratio_to_pair(sigma, rho)
    sigma, rho = pair_to_scale_ratio(sigma_x, sigma_y)
    return sigma_x, sigma_y, sigma, rho


def better(result_a, result_b):
    if result_b is None:
        return True
    score_a = (float(result_a["final_val_mse"]), float(result_a["final_val_max"]))
    score_b = (float(result_b["final_val_mse"]), float(result_b["final_val_max"]))
    return score_a < score_b


def powell_search_sigma_for_grade(data, result_root, fixed_sigmas, grade):
    table = []
    cache = {}
    best = None
    train_count = {"n": 0}

    work_root = os.path.join(result_root, "powell_grade_%d_work" % int(grade))
    ensure_dir(work_root)

    def objective(z):
        sigma_x, sigma_y, sigma, rho = sigma_from_powell_vector(z)
        key = (sigma_x, sigma_y)
        if key in cache:
            return cache[key]

        train_count["n"] += 1
        smoothing_by_grade = dict(fixed_sigmas)
        smoothing_by_grade[int(grade)] = {"sigma_x": sigma_x, "sigma_y": sigma_y}

        run_label = "eval_%03d__g%d__%s" % (
            train_count["n"],
            int(grade),
            sigma_pair_name(sigma_x, sigma_y),
        )

        print("")
        print("=" * 70)
        print("Candidate for sigma_%d: sigma_x = %.10e, sigma_y = %.10e" % (
            int(grade),
            sigma_x,
            sigma_y,
        ))
        print("scale = %.10e, ratio = %.10e" % (sigma, rho))
        print("=" * 70)

        result = train_Phi_autodepth(data, smoothing_by_grade, work_root, run_label)
        result["searched_grade"] = int(grade)
        result["searched_sigma_x"] = sigma_x
        result["searched_sigma_y"] = sigma_y
        result["searched_sigma_scale"] = sigma
        result["searched_sigma_ratio"] = rho

        row = {
            "eval": train_count["n"],
            "grade": int(grade),
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_scale": sigma,
            "sigma_ratio": rho,
            "final_val_mse": float(result["final_val_mse"]),
            "final_val_max": float(result["final_val_max"]),
            "final_test_mse": float(result["final_test_mse"]),
            "final_test_max": float(result["final_test_max"]),
            "accepted_grades": int(result["accepted_grades"]),
            "stop_reason": result["stop_reason"],
            "result_dir": result["result_dir"],
        }
        table.append(row)
        save_csv(table, os.path.join(result_root, "powell_grade_%d.csv" % int(grade)))

        nonlocal best
        if better(result, best):
            best = result

        cache[key] = float(result["final_val_mse"])
        return cache[key]

    minimize(
        objective,
        powell_start(grade),
        method="Powell",
        bounds=powell_bounds(grade),
        options={
            "maxiter": int(cfg.POWELL_MAX_ITER),
            "maxfev": int(cfg.POWELL_MAX_ITER),
            "xtol": float(cfg.POWELL_XTOL),
            "ftol": float(cfg.POWELL_FTOL),
            "disp": True,
        },
    )

    selected_sigmas = dict(fixed_sigmas)
    selected_sigmas[int(grade)] = {
        "sigma_x": float(best["searched_sigma_x"]),
        "sigma_y": float(best["searched_sigma_y"]),
    }

    run_label = "powell_g%d__%s" % (
        int(grade),
        sigma_pair_name(best["searched_sigma_x"], best["searched_sigma_y"]),
    )
    selected = train_Phi_autodepth(data, selected_sigmas, result_root, run_label)

    selected["searched_grade"] = int(grade)
    selected["searched_sigma_x"] = float(best["searched_sigma_x"])
    selected["searched_sigma_y"] = float(best["searched_sigma_y"])
    selected["searched_sigma_scale"] = float(best["searched_sigma_scale"])
    selected["searched_sigma_ratio"] = float(best["searched_sigma_ratio"])
    selected["powell_trained_candidates"] = int(train_count["n"])

    save_json(selected, os.path.join(result_root, "powell_grade_%d_selected.json" % int(grade)))

    if os.path.isdir(work_root):
        shutil.rmtree(work_root)

    return selected


def make_data():
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    x_train = cartesian_grid_2d(cfg.TRAIN_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)
    x_val = shifted_grid_2d(cfg.VAL_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, cfg.VAL_GRID_SHIFT, device, cfg.DTYPE)
    x_test = midpoint_grid_2d(cfg.TEST_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)

    return {
        "x_train": x_train,
        "y_train": target_function(x_train),
        "x_val": x_val,
        "y_val": target_function(x_val),
        "x_test": x_test,
        "y_test": target_function(x_test),
    }



def main():
    if cfg.SUPPORT_AXIS_SIZE < 5 or (cfg.SUPPORT_AXIS_SIZE - 1) % 4 != 0:
        raise ValueError("SUPPORT_AXIS_SIZE must have the form 4I + 1 for Boole weights.")

    total_start = time.perf_counter()
    result_root = cfg.RESULT_ROOT
    ensure_dir(result_root)

    data = make_data()
    fixed_sigmas = normalize_sigmas(cfg.FIXED_SMOOTHING_BY_GRADE)
    stage_rows = []

    save_json(
        {
            "experiment_name": cfg.EXPERIMENT_NAME,
            "seed": cfg.SEED,
            "target": "selected in Example2D.py",
            "model": "auto-depth MGDL with ThreeAffineBlock",
            "smoother": "tensor-product Boole Gaussian smoothing",
            "fixed_smoothing_by_grade": sigmas_for_json(fixed_sigmas),
            "powell_search_grade": cfg.POWELL_SEARCH_GRADE,
        },
        os.path.join(result_root, "run_info.json"),
    )

    if cfg.POWELL_SEARCH_GRADE is None:
        label = "no_smoothing" if len(fixed_sigmas) == 0 else "fixed_sigmas"
        final_model = train_Phi_autodepth(data, fixed_sigmas, result_root, label)
        final_sigmas = dict(fixed_sigmas)
        stage_rows.append(
            {
                "stage": label,
                "searched_grade": "",
                "searched_sigma_x": "",
                "searched_sigma_y": "",
                "decision": "reference",
                "final_val_mse": final_model["final_val_mse"],
                "final_val_max": final_model["final_val_max"],
                "final_test_mse": final_model["final_test_mse"],
                "final_test_max": final_model["final_test_max"],
                "accepted_grades": final_model["accepted_grades"],
                "result_dir": final_model["result_dir"],
            }
        )
    else:
        reference = None
        if len(fixed_sigmas) > 0:
            reference = train_Phi_autodepth(data, fixed_sigmas, result_root, "fixed_sigmas")
            stage_rows.append(
                {
                    "stage": "fixed_sigmas",
                    "searched_grade": "",
                    "searched_sigma_x": "",
                    "searched_sigma_y": "",
                    "decision": "reference",
                    "final_val_mse": reference["final_val_mse"],
                    "final_val_max": reference["final_val_max"],
                    "final_test_mse": reference["final_test_mse"],
                    "final_test_max": reference["final_test_max"],
                    "accepted_grades": reference["accepted_grades"],
                    "result_dir": reference["result_dir"],
                }
            )

        grade = int(cfg.POWELL_SEARCH_GRADE)
        best = powell_search_sigma_for_grade(data, result_root, fixed_sigmas, grade)

        final_sigmas = dict(fixed_sigmas)
        final_sigmas[grade] = {
            "sigma_x": float(best["searched_sigma_x"]),
            "sigma_y": float(best["searched_sigma_y"]),
        }
        final_model = best

        stage_rows.append(
            {
                "stage": "powell_search",
                "searched_grade": grade,
                "searched_sigma_x": float(best["searched_sigma_x"]),
                "searched_sigma_y": float(best["searched_sigma_y"]),
                "decision": "selected_by_validation",
                "final_val_mse": best["final_val_mse"],
                "final_val_max": best["final_val_max"],
                "final_test_mse": best["final_test_mse"],
                "final_test_max": best["final_test_max"],
                "accepted_grades": best["accepted_grades"],
                "result_dir": best["result_dir"],
            }
        )

    comparison_rows = [
        {
            "model": "final_selected",
            "smoothing_by_grade": str(sigmas_for_json(final_sigmas)),
            "final_val_mse": final_model["final_val_mse"],
            "final_val_max": final_model["final_val_max"],
            "final_test_mse": final_model["final_test_mse"],
            "final_test_max": final_model["final_test_max"],
            "accepted_grades": final_model["accepted_grades"],
            "stop_reason": final_model["stop_reason"],
            "result_dir": final_model["result_dir"],
        }
    ]

    save_csv(stage_rows, os.path.join(result_root, "smoothing_stage_decisions.csv"))
    save_csv(comparison_rows, os.path.join(result_root, "reference_vs_final_selected.csv"))
    save_json(sigmas_for_json(final_sigmas), os.path.join(result_root, "selected_smoothing_by_grade.json"))

    print("")
    print("Final selected smoothing by grade:", sigmas_for_json(final_sigmas))
    print("Final ValMSE = %.6e" % final_model["final_val_mse"])
    print("Final TestMSE = %.6e" % final_model["final_test_mse"])
    print("Runtime = %.2f seconds" % (time.perf_counter() - total_start))


if __name__ == "__main__":
    main()
