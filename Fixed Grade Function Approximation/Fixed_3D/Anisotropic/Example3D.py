import math
import os
import time

import numpy as np
import torch
from scipy.optimize import minimize

import config as cfg
from functions import target_easy_3d
from mgdl import FixedDepthMGDL3D
from trainer import train_full_mgdl
from utils import (
    cartesian_grid_3d,
    ensure_dir,
    midpoint_grid_3d,
    normalize_sigmas,
    save_csv,
    save_json,
    set_seed,
    shifted_grid_3d,
    sigma_triple_name,
    sigmas_for_json,
)

def round_sigma(sigma):
    return round(float(sigma), int(cfg.SIGMA_DECIMALS))

def triple_to_search_coordinates(sigma_x, sigma_y, sigma_z):
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    sigma_z = float(sigma_z)

    sigma_scale = (sigma_x * sigma_y * sigma_z) ** (1.0 / 3.0)
    sigma_ratio_xy = sigma_x / sigma_y
    sigma_ratio_xy_to_z = math.sqrt(sigma_x * sigma_y) / sigma_z

    return sigma_scale, sigma_ratio_xy, sigma_ratio_xy_to_z

def search_coordinates_to_triple(sigma_scale, sigma_ratio_xy, sigma_ratio_xy_to_z):
    sigma_scale = float(sigma_scale)
    sigma_ratio_xy = float(sigma_ratio_xy)
    sigma_ratio_xy_to_z = float(sigma_ratio_xy_to_z)

    sigma_x = sigma_scale * sigma_ratio_xy ** 0.5 * sigma_ratio_xy_to_z ** (1.0 / 3.0)
    sigma_y = sigma_scale * sigma_ratio_xy ** (-0.5) * sigma_ratio_xy_to_z ** (1.0 / 3.0)
    sigma_z = sigma_scale * sigma_ratio_xy_to_z ** (-2.0 / 3.0)

    return round_sigma(sigma_x), round_sigma(sigma_y), round_sigma(sigma_z)

def train_Phi_G(data, sigma_by_grade, result_root, run_label, searched_grade=None):
    set_seed(cfg.SEED)

    model = FixedDepthMGDL3D(
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
        searched_grade,
    )

    result["label"] = run_label
    result["sigma_by_grade"] = sigmas_for_json(sigma_by_grade)
    save_json(result, os.path.join(result_dir, "final_metrics.json"))
    return result

def better(result_a, result_b):
    if result_b is None:
        return True
    score_a = (float(result_a["final_val_mse"]), float(result_a["final_val_max"]))
    score_b = (float(result_b["final_val_mse"]), float(result_b["final_val_max"]))
    return score_a < score_b

def make_search_row(search_eval, result, grade):

    return {
        "search_eval": int(search_eval),
        "trained_eval": int(result["trained_eval"]),
        "grade": int(grade),
        "sigma_x": float(result["searched_sigma_x"]),
        "sigma_y": float(result["searched_sigma_y"]),
        "sigma_z": float(result["searched_sigma_z"]),
        "sigma_scale": float(result["searched_sigma_scale"]),
        "sigma_ratio_xy": float(result["searched_sigma_ratio_xy"]),
        "sigma_ratio_xy_to_z": float(result["searched_sigma_ratio_xy_to_z"]),
        "final_val_mse": float(result["final_val_mse"]),
        "final_val_max": float(result["final_val_max"]),
        "final_test_mse": float(result["final_test_mse"]),
        "final_test_max": float(result["final_test_max"]),
        "is_selected": 0,
        "result_dir": result["result_dir"],
    }

def powell_search_sigma_for_grade(data, result_root, accepted_sigmas, grade):
    grade = int(grade)
    work_root = os.path.join(result_root, "powell_grade_%d_work" % grade)
    ensure_dir(work_root)

    cache = {}
    search_rows = []
    seen = set()
    trained_count = {"n": 0}

    def evaluate_sigma(sigma_x, sigma_y, sigma_z):
        sigma_x = round_sigma(sigma_x)
        sigma_y = round_sigma(sigma_y)
        sigma_z = round_sigma(sigma_z)
        key = (sigma_x, sigma_y, sigma_z)

        if key in cache:
            return cache[key]

        trained_count["n"] += 1
        sigma_by_grade = dict(accepted_sigmas)
        sigma_by_grade[grade] = {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
        }

        run_label = "eval_%03d__g%d__%s" % (
            trained_count["n"],
            grade,
            sigma_triple_name(sigma_x, sigma_y, sigma_z),
        )

        sigma_scale, sigma_ratio_xy, sigma_ratio_xy_to_z = triple_to_search_coordinates(
            sigma_x,
            sigma_y,
            sigma_z,
        )

        print("")
        print("=" * 76)
        print(
            "Candidate for sigma_%d: sigma_x = %.10e, sigma_y = %.10e, sigma_z = %.10e"
            % (grade, sigma_x, sigma_y, sigma_z)
        )
        print(
            "scale = %.10e, ratio_xy = %.10e, ratio_xy_to_z = %.10e"
            % (sigma_scale, sigma_ratio_xy, sigma_ratio_xy_to_z)
        )
        print("=" * 76)

        result = train_Phi_G(
            data,
            sigma_by_grade,
            work_root,
            run_label,
            searched_grade=grade,
        )

        result["trained_eval"] = int(trained_count["n"])
        result["searched_grade"] = grade
        result["searched_sigma_x"] = sigma_x
        result["searched_sigma_y"] = sigma_y
        result["searched_sigma_z"] = sigma_z
        result["searched_sigma_scale"] = sigma_scale
        result["searched_sigma_ratio_xy"] = sigma_ratio_xy
        result["searched_sigma_ratio_xy_to_z"] = sigma_ratio_xy_to_z

        cache[key] = result
        return result

    def record(result):
        key = (
            float(result["searched_sigma_x"]),
            float(result["searched_sigma_y"]),
            float(result["searched_sigma_z"]),
        )
        if key in seen:
            return

        seen.add(key)
        search_rows.append(make_search_row(len(search_rows) + 1, result, grade))
        save_csv(
            search_rows,
            os.path.join(result_root, "powell_grade_%d.csv" % grade),
        )

    def objective(z):
        sigma_scale = math.exp(float(z[0]))
        sigma_ratio_xy = math.exp(float(z[1]))
        sigma_ratio_xy_to_z = math.exp(float(z[2]))

        sigma_x, sigma_y, sigma_z = search_coordinates_to_triple(
            sigma_scale,
            sigma_ratio_xy,
            sigma_ratio_xy_to_z,
        )
        result = evaluate_sigma(sigma_x, sigma_y, sigma_z)
        record(result)
        return float(result["final_val_mse"])

    scale_left, scale_right = cfg.SIGMA_SCALE_INTERVAL[grade]
    ratio_left, ratio_right = cfg.SIGMA_RATIO_INTERVAL[grade]

    bounds = [
        (math.log(float(scale_left)), math.log(float(scale_right))),
        (math.log(float(ratio_left)), math.log(float(ratio_right))),
        (math.log(float(ratio_left)), math.log(float(ratio_right))),
    ]
    start = np.array(
        [
            0.5 * (bounds[0][0] + bounds[0][1]),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    objective(start)
    minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": int(cfg.POWELL_MAX_ITER),
            "maxfev": int(cfg.POWELL_MAX_ITER),
            "xtol": float(cfg.POWELL_XTOL),
            "ftol": float(cfg.POWELL_FTOL),
            "disp": True,
        },
    )

    selected = None
    for result in cache.values():
        if better(result, selected):
            selected = result

    for row in search_rows:
        if (
            float(row["sigma_x"]) == float(selected["searched_sigma_x"])
            and float(row["sigma_y"]) == float(selected["searched_sigma_y"])
            and float(row["sigma_z"]) == float(selected["searched_sigma_z"])
        ):
            row["is_selected"] = 1

    save_csv(search_rows, os.path.join(result_root, "powell_grade_%d.csv" % grade))

    selected["powell_trained_candidates"] = int(trained_count["n"])
    save_json(selected, os.path.join(result_root, "powell_grade_%d_selected.json" % grade))
    return selected

def make_data():
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    x_train = cartesian_grid_3d(
        cfg.TRAIN_AXIS_SIZE,
        cfg.DOMAIN_LOW,
        cfg.DOMAIN_HIGH,
        device,
        cfg.DTYPE,
    )
    x_val = shifted_grid_3d(
        cfg.VAL_AXIS_SIZE,
        cfg.DOMAIN_LOW,
        cfg.DOMAIN_HIGH,
        cfg.VAL_GRID_SHIFT,
        device,
        cfg.DTYPE,
    )
    x_test = midpoint_grid_3d(
        cfg.TEST_AXIS_SIZE,
        cfg.DOMAIN_LOW,
        cfg.DOMAIN_HIGH,
        device,
        cfg.DTYPE,
    )

    return {
        "x_train": x_train,
        "y_train": target_easy_3d(x_train),
        "x_val": x_val,
        "y_val": target_easy_3d(x_val),
        "x_test": x_test,
        "y_test": target_easy_3d(x_test),
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

    return "%s__%s__G%d__w%d__P%d__fixed_%s__search_%s" % (
        cfg.EXPERIMENT_NAME,
        cfg.QUADRATURE_RULE,
        cfg.MAX_GRADES,
        cfg.WIDTH,
        cfg.P,
        fixed_text,
        search_text,
    )

def main():
    total_start = time.perf_counter()

    if int(cfg.P) < 1:
        raise ValueError("P must be positive.")
    expected = {
        "midpoint": cfg.P,
        "trapezoid": cfg.P + 1,
        "simpson": 2 * cfg.P + 1,
        "boole": 4 * cfg.P + 1,
    }[cfg.QUADRATURE_RULE]
    if cfg.SUPPORT_AXIS_SIZE != expected:
        raise ValueError("SUPPORT_AXIS_SIZE does not match P and the rule.")

    result_root = os.path.join(cfg.RESULT_ROOT, run_name())
    ensure_dir(result_root)

    data = make_data()
    accepted_sigmas = normalize_sigmas(cfg.FIXED_SMOOTHING_BY_GRADE)
    stage_rows = []

    if cfg.POWELL_SEARCH_GRADE is None:
        label = "no_smoothing" if len(accepted_sigmas) == 0 else "fixed_sigmas"
        current = train_Phi_G(data, accepted_sigmas, result_root, label, searched_grade=None)
    else:
        current = None
        if len(accepted_sigmas) > 0:
            current = train_Phi_G(data, accepted_sigmas, result_root, "fixed_sigmas", searched_grade=None)

        grade = int(cfg.POWELL_SEARCH_GRADE)
        best = powell_search_sigma_for_grade(data, result_root, accepted_sigmas, grade)

        if current is None:
            accepted = True
        else:
            accepted = float(best["final_val_mse"]) < float(current["final_val_mse"]) - float(cfg.ACCEPT_TOL)


        if accepted:
            decision = "accepted"
            accepted_sigmas[grade] = {
                "sigma_x": float(best["searched_sigma_x"]),
                "sigma_y": float(best["searched_sigma_y"]),
                "sigma_z": float(best["searched_sigma_z"]),
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
                "selected_sigma_z": float(best["searched_sigma_z"]),
                "selected_sigma_scale": float(best["searched_sigma_scale"]),
                "selected_sigma_ratio_xy": float(best["searched_sigma_ratio_xy"]),
                "selected_sigma_ratio_xy_to_z": float(best["searched_sigma_ratio_xy_to_z"]),
                "final_val_mse": float(best["final_val_mse"]),
                "final_val_max": float(best["final_val_max"]),
                "final_test_mse": float(best["final_test_mse"]),
                "final_test_max": float(best["final_test_max"]),
                "result_dir": best["result_dir"],
            }
        )

    save_csv(stage_rows, os.path.join(result_root, "smoothing_stage_decisions.csv"))
    save_json(sigmas_for_json(accepted_sigmas), os.path.join(result_root, "selected_smoothing_by_grade.json"))
    save_json(
        {
            "total_runtime_sec": time.perf_counter() - total_start,
            "result_root": result_root,
            "selected_smoothing_by_grade": sigmas_for_json(accepted_sigmas),
            "final_result": current,
        },
        os.path.join(result_root, "run_info.json"),
    )

    print("")
    print("Final selected smoothing by grade:", sigmas_for_json(accepted_sigmas))
    print("Final ValMSE = %.6e" % current["final_val_mse"])
    print("Final TestMSE = %.6e" % current["final_test_mse"])

if __name__ == "__main__":
    main()
