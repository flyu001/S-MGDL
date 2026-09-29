import math
import os
import shutil
import time

import numpy as np
import torch
from scipy.optimize import minimize

import config as cfg
from functions import target_easy_2d
from mgdl import FixedDepthMGDL2D
from trainer import train_full_mgdl
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

def train_Phi_G(data, sigma_by_grade, result_root, run_label, save_grade_outputs=True):
    set_seed(cfg.SEED)

    model = FixedDepthMGDL2D(
        width=cfg.WIDTH,
        max_grades=cfg.MAX_GRADES,
    ).to(device=data["x_train"].device, dtype=cfg.DTYPE)

    result_dir = os.path.join(result_root, run_label)
    result = train_full_mgdl(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        data["x_test"],
        data["y_test"],
        sigma_by_grade,
        result_dir,
        save_grade_outputs,
    )
    result["label"] = run_label
    result["sigma_by_grade"] = sigmas_for_json(sigma_by_grade)
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

def powell_search_sigma_for_grade(data, result_root, accepted_sigmas, grade):
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
        sigma_by_grade = dict(accepted_sigmas)
        sigma_by_grade[int(grade)] = {"sigma_x": sigma_x, "sigma_y": sigma_y}

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

        result = train_Phi_G(data, sigma_by_grade, work_root, run_label, save_grade_outputs=False)
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

    selected_sigmas = dict(accepted_sigmas)
    selected_sigmas[int(grade)] = {
        "sigma_x": float(best["searched_sigma_x"]),
        "sigma_y": float(best["searched_sigma_y"]),
    }

    run_label = "powell_g%d__%s" % (
        int(grade),
        sigma_pair_name(best["searched_sigma_x"], best["searched_sigma_y"]),
    )
    selected = train_Phi_G(data, selected_sigmas, result_root, run_label, save_grade_outputs=True)

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
        "y_train": target_easy_2d(x_train),
        "x_val": x_val,
        "y_val": target_easy_2d(x_val),
        "x_test": x_test,
        "y_test": target_easy_2d(x_test),
    }

def run_name():
    if cfg.POWELL_SEARCH_GRADE is None:
        search_text = "none"
    else:
        search_text = str(int(cfg.POWELL_SEARCH_GRADE))

    if len(cfg.FIXED_SMOOTHING_BY_GRADE) == 0:
        fixed_text = "none"
    else:
        fixed_text = "-".join(str(int(g)) for g in sorted(cfg.FIXED_SMOOTHING_BY_GRADE))

    return "%s__%s_P%d__G%d__w%d__fixed_%s__search_%s" % (
        cfg.EXPERIMENT_NAME,
        cfg.QUADRATURE_RULE,
        cfg.P,
        cfg.MAX_GRADES,
        cfg.WIDTH,
        fixed_text,
        search_text,
    )

def main():
    total_start = time.perf_counter()

    if len(cfg.EPOCHS_PER_GRADE) != cfg.MAX_GRADES:
        raise ValueError("EPOCHS_PER_GRADE must have length MAX_GRADES.")
    if cfg.P < 1 or cfg.QUADRATURE_RULE not in ("midpoint", "trapezoid", "simpson", "boole"):
        raise ValueError("Choose a valid QUADRATURE_RULE and a positive P.")

    result_root = os.path.join(cfg.RESULT_ROOT, run_name())
    ensure_dir(result_root)

    data = make_data()
    accepted_sigmas = normalize_sigmas(cfg.FIXED_SMOOTHING_BY_GRADE)
    stage_rows = []

    if cfg.POWELL_SEARCH_GRADE is None:
        label = "no_smoothing" if len(accepted_sigmas) == 0 else "fixed_sigmas"
        current = train_Phi_G(data, accepted_sigmas, result_root, label, save_grade_outputs=True)
    else:
        current = None
        if len(accepted_sigmas) > 0:
            current = train_Phi_G(data, accepted_sigmas, result_root, "fixed_sigmas", save_grade_outputs=True)

        grade = int(cfg.POWELL_SEARCH_GRADE)
        best = powell_search_sigma_for_grade(data, result_root, accepted_sigmas, grade)

        previous_val_mse = "" if current is None else float(current["final_val_mse"])
        previous_val_max = "" if current is None else float(current["final_val_max"])

        if current is None:
            accepted = True
        else:
            accepted = float(best["final_val_mse"]) < float(current["final_val_mse"]) - float(cfg.ACCEPT_TOL)


        if accepted:
            decision = "accepted"
            accepted_sigmas[grade] = {
                "sigma_x": float(best["searched_sigma_x"]),
                "sigma_y": float(best["searched_sigma_y"]),
            }
            current = best
        else:
            decision = "skipped"

        stage_rows.append(
            {
                "grade": grade,
                "decision": decision,
                "selected_sigma_x": float(best["searched_sigma_x"]),
                "selected_sigma_y": float(best["searched_sigma_y"]),
                "selected_sigma_scale": float(best["searched_sigma_scale"]),
                "selected_sigma_ratio": float(best["searched_sigma_ratio"]),
                "previous_final_val_mse": previous_val_mse,
                "previous_final_val_max": previous_val_max,
                "candidate_final_val_mse": float(best["final_val_mse"]),
                "candidate_final_val_max": float(best["final_val_max"]),
                "candidate_result_dir": best["result_dir"],
            }
        )

    save_csv(stage_rows, os.path.join(result_root, "smoothing_stage_decisions.csv"))
    save_json(sigmas_for_json(accepted_sigmas), os.path.join(result_root, "selected_sigmas.json"))

    comparison_rows = [
        {
            "model": "selected_model",
            "accepted_sigmas": str(sigmas_for_json(accepted_sigmas)),
            "final_val_mse": float(current["final_val_mse"]),
            "final_val_max": float(current["final_val_max"]),
            "final_test_mse": float(current["final_test_mse"]),
            "final_test_max": float(current["final_test_max"]),
            "result_dir": current["result_dir"],
        }
    ]
    save_csv(comparison_rows, os.path.join(result_root, "smooth_vs_nonsmooth_comparison.csv"))

    save_json(
        {
            "total_runtime_sec": time.perf_counter() - total_start,
            "result_root": result_root,
            "accepted_sigmas": sigmas_for_json(accepted_sigmas),
        },
        os.path.join(result_root, "run_info.json"),
    )

    print("")
    print("Finished.")
    print("Result directory: %s" % result_root)

if __name__ == "__main__":
    main()
