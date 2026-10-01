import math
import os
import shutil
import time

import numpy as np
import torch
from scipy.optimize import minimize

import config as cfg
from functions import target_hard_3d
from mgdl import AutoDepthMGDL3D
from trainer import train_autodepth
from utils import (
    cartesian_grid_3d, ensure_dir, midpoint_grid_3d, normalize_sigmas,
    save_csv, save_json, set_seed, shifted_grid_3d, sigma_triple_name,
    sigmas_for_json,
)


def round_sigma(sigma):
    return round(float(sigma), int(cfg.SIGMA_DECIMALS))


def triple_to_search_coordinates(sx, sy, sz):
    scale = (float(sx) * float(sy) * float(sz)) ** (1.0 / 3.0)
    ratio_xy = float(sx) / float(sy)
    ratio_xy_to_z = math.sqrt(float(sx) * float(sy)) / float(sz)
    return scale, ratio_xy, ratio_xy_to_z


def search_coordinates_to_triple(scale, ratio_xy, ratio_xy_to_z):
    sx = scale * ratio_xy ** 0.5 * ratio_xy_to_z ** (1.0 / 3.0)
    sy = scale * ratio_xy ** (-0.5) * ratio_xy_to_z ** (1.0 / 3.0)
    sz = scale * ratio_xy_to_z ** (-2.0 / 3.0)
    return round_sigma(sx), round_sigma(sy), round_sigma(sz)


def train_Phi_autodepth(data, smoothing_by_grade, result_root, run_label):
    set_seed(cfg.SEED)
    model = AutoDepthMGDL3D(cfg.WIDTH, input_dim=3).to(
        device=data["x_train"].device, dtype=cfg.DTYPE,
    )
    result_dir = os.path.join(result_root, run_label)
    result = train_autodepth(
        model, data["x_train"], data["y_train"], data["x_val"], data["y_val"],
        data["x_test"], data["y_test"], smoothing_by_grade, result_dir,
    )
    result["run_label"] = run_label
    result["smoothing_by_grade"] = sigmas_for_json(smoothing_by_grade)
    save_json(result, os.path.join(result_dir, "final_metrics.json"))
    return result


def better(a, b):
    if b is None:
        return True
    return (float(a["final_val_mse"]), float(a["final_val_max"])) < (
        float(b["final_val_mse"]), float(b["final_val_max"])
    )


def powell_search_sigma_for_grade(data, result_root, fixed_sigmas, grade):
    grade = int(grade)
    table, cache = [], {}
    best = None
    count = {"n": 0}
    work_root = os.path.join(result_root, "powell_grade_%d_work" % grade)
    ensure_dir(work_root)

    scale_left, scale_right = cfg.SIGMA_SCALE_INTERVAL[grade]
    ratio_left, ratio_right = cfg.SIGMA_RATIO_INTERVAL[grade]
    bounds = [
        (math.log(scale_left), math.log(scale_right)),
        (math.log(ratio_left), math.log(ratio_right)),
        (math.log(ratio_left), math.log(ratio_right)),
    ]

    def objective(z):
        nonlocal best
        scale, ratio_xy, ratio_xy_to_z = map(math.exp, map(float, z))
        sx, sy, sz = search_coordinates_to_triple(scale, ratio_xy, ratio_xy_to_z)
        key = (sx, sy, sz)
        if key in cache:
            return cache[key]
        count["n"] += 1
        scale, ratio_xy, ratio_xy_to_z = triple_to_search_coordinates(sx, sy, sz)
        smoothing = dict(fixed_sigmas)
        smoothing[grade] = {"sigma_x": sx, "sigma_y": sy, "sigma_z": sz}
        label = "eval_%03d__g%d__%s" % (count["n"], grade, sigma_triple_name(sx, sy, sz))
        print("\nCandidate g%d: sx=%.10e sy=%.10e sz=%.10e" % (grade, sx, sy, sz))
        print("scale=%.10e ratio_xy=%.10e ratio_xy_to_z=%.10e" % (scale, ratio_xy, ratio_xy_to_z))
        result = train_Phi_autodepth(data, smoothing, work_root, label)
        result.update({
            "searched_grade": grade, "searched_sigma_x": sx, "searched_sigma_y": sy,
            "searched_sigma_z": sz, "searched_sigma_scale": scale,
            "searched_sigma_ratio_xy": ratio_xy,
            "searched_sigma_ratio_xy_to_z": ratio_xy_to_z,
        })
        row = {
            "eval": count["n"], "grade": grade, "sigma_x": sx, "sigma_y": sy,
            "sigma_z": sz, "sigma_scale": scale, "sigma_ratio_xy": ratio_xy,
            "sigma_ratio_xy_to_z": ratio_xy_to_z,
            "final_val_mse": result["final_val_mse"], "final_val_max": result["final_val_max"],
            "final_test_mse": result["final_test_mse"], "final_test_max": result["final_test_max"],
            "accepted_grades": result["accepted_grades"],
            "stop_reason": result["stop_reason"], "result_dir": result["result_dir"],
        }
        table.append(row)
        save_csv(table, os.path.join(result_root, "powell_grade_%d.csv" % grade))
        if better(result, best):
            best = result
        cache[key] = float(result["final_val_mse"])
        return cache[key]

    start = np.array([0.5 * (a + b) for a, b in bounds], dtype=np.float64)
    minimize(objective, start, method="Powell", bounds=bounds, options={
        "maxiter": int(cfg.POWELL_MAX_ITER), "maxfev": int(cfg.POWELL_MAX_ITER),
        "xtol": float(cfg.POWELL_XTOL), "ftol": float(cfg.POWELL_FTOL), "disp": True,
    })
    selected_sigmas = dict(fixed_sigmas)
    selected_sigmas[grade] = {
        "sigma_x": best["searched_sigma_x"], "sigma_y": best["searched_sigma_y"],
        "sigma_z": best["searched_sigma_z"],
    }
    selected = train_Phi_autodepth(
        data, selected_sigmas, result_root,
        "powell_g%d__%s" % (grade, sigma_triple_name(
            best["searched_sigma_x"], best["searched_sigma_y"], best["searched_sigma_z"])),
    )
    selected.update({k: v for k, v in best.items() if k.startswith("searched_")})
    selected["powell_trained_candidates"] = count["n"]
    save_json(selected, os.path.join(result_root, "powell_grade_%d_selected.json" % grade))
    if os.path.isdir(work_root):
        shutil.rmtree(work_root)
    return selected


def make_data():
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    x_train = cartesian_grid_3d(cfg.TRAIN_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)
    x_val = shifted_grid_3d(cfg.VAL_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, cfg.VAL_GRID_SHIFT, device, cfg.DTYPE)
    x_test = midpoint_grid_3d(cfg.TEST_AXIS_SIZE, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)
    return {
        "x_train": x_train, "y_train": target_hard_3d(x_train),
        "x_val": x_val, "y_val": target_hard_3d(x_val),
        "x_test": x_test, "y_test": target_hard_3d(x_test),
    }


def main():
    if cfg.SUPPORT_AXIS_SIZE < 5 or (cfg.SUPPORT_AXIS_SIZE - 1) % 4 != 0:
        raise ValueError("SUPPORT_AXIS_SIZE must have the form 4I + 1 for Boole weights.")
    start = time.perf_counter()
    ensure_dir(cfg.RESULT_ROOT)
    data = make_data()
    fixed = normalize_sigmas(cfg.FIXED_SMOOTHING_BY_GRADE)
    if cfg.POWELL_SEARCH_GRADE is None:
        label = "no_smoothing" if not fixed else "fixed_sigmas"
        final = train_Phi_autodepth(data, fixed, cfg.RESULT_ROOT, label)
        final_sigmas = fixed
    else:
        grade = int(cfg.POWELL_SEARCH_GRADE)
        final = powell_search_sigma_for_grade(data, cfg.RESULT_ROOT, fixed, grade)
        final_sigmas = dict(fixed)
        final_sigmas[grade] = {
            "sigma_x": final["searched_sigma_x"], "sigma_y": final["searched_sigma_y"],
            "sigma_z": final["searched_sigma_z"],
        }
    save_json(sigmas_for_json(final_sigmas), os.path.join(cfg.RESULT_ROOT, "selected_smoothing_by_grade.json"))
    print("\nFinal selected smoothing:", sigmas_for_json(final_sigmas))
    print("Final ValMSE = %.6e" % final["final_val_mse"])
    print("Final TestMSE = %.6e" % final["final_test_mse"])
    print("Runtime = %.2f seconds" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
