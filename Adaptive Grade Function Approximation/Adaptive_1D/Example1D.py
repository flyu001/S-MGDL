import os
import shutil

import torch

import config as cfg
# Adaptive 1D Example 1
from functions import target_hard_1d as target_function

# Adaptive 1D Example 2
# from functions import target_hard2_1d as target_function
from mgdl import AutoDepthMGDL1D
from trainer import train_autodepth
from utils import data_grid_1d, ensure_dir, float_name, midpoint_grid_1d, save_csv, save_json, set_seed

def round_sigma(sigma):
    return round(float(sigma), int(cfg.SIGMA_DECIMALS))

def train_Phi_autodepth(data, smoothing_by_grade, result_root, run_label):
    set_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)
    model = AutoDepthMGDL1D(width=cfg.WIDTH, input_dim=1).to(device=device, dtype=cfg.DTYPE)

    result_dir = os.path.join(result_root, run_label)
    ensure_dir(result_dir)

    out = train_autodepth(
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

    out["run_label"] = run_label
    out["smoothing_by_grade"] = dict(smoothing_by_grade)
    return out

def golden_search_sigma_for_grade(grade, fixed_smoothing_by_grade, reference, data, result_root):
    if grade not in cfg.SIGMA_INTERVAL:
        raise ValueError("Missing SIGMA_INTERVAL for grade %d." % grade)

    a, b = cfg.SIGMA_INTERVAL[grade]
    rho = (5.0 ** 0.5 - 1.0) / 2.0

    c = b - rho * (b - a)
    d = a + rho * (b - a)

    candidate_results = []
    evaluated = {}

    def evaluate_sigma(sigma_raw):
        sigma = round_sigma(sigma_raw)

        if sigma in evaluated:
            return evaluated[sigma]

        smoothing_by_grade = dict(fixed_smoothing_by_grade)
        smoothing_by_grade[int(grade)] = sigma

        run_label = "golden_g%d_sigma_%s" % (grade, float_name(sigma))
        out = train_Phi_autodepth(data, smoothing_by_grade, result_root, run_label)
        out["searched_grade"] = int(grade)
        out["searched_sigma"] = sigma
        evaluated[sigma] = out
        candidate_results.append(out)
        return out

    out_c = evaluate_sigma(c)
    out_d = evaluate_sigma(d)

    iteration = 0
    while iteration < cfg.GOLDEN_MAX_ITER and abs(b - a) > cfg.GOLDEN_SIGMA_TOL:
        if out_c["final_val_mse"] <= out_d["final_val_mse"]:
            b = d
            d = c
            out_d = out_c
            c = b - rho * (b - a)
            out_c = evaluate_sigma(c)
        else:
            a = c
            c = d
            out_c = out_d
            d = a + rho * (b - a)
            out_d = evaluate_sigma(d)

        iteration += 1

    best = candidate_results[0]
    for out in candidate_results[1:]:
        if out["final_val_mse"] < best["final_val_mse"]:
            best = out
        elif out["final_val_mse"] == best["final_val_mse"] and out["final_val_max"] < best["final_val_max"]:
            best = out

    rows = []
    for out in candidate_results:
        rows.append(
            {
                "searched_grade": int(grade),
                "searched_sigma": float(out["searched_sigma"]),
                "final_train_mse": out["final_train_mse"],
                "final_train_max": out["final_train_max"],
                "final_val_mse": out["final_val_mse"],
                "final_val_max": out["final_val_max"],
                "final_test_mse": out["final_test_mse"],
                "final_test_max": out["final_test_max"],
                "accepted_grades": out["accepted_grades"],
                "stop_reason": out["stop_reason"],
                "is_best": int(out is best),
                "result_dir": out["result_dir"],
            }
        )

    save_csv(rows, os.path.join(result_root, "golden_grade_%d.csv" % grade))

    best_dir = os.path.abspath(best["result_dir"])
    for out in candidate_results:
        candidate_dir = os.path.abspath(out["result_dir"])
        if candidate_dir != best_dir and os.path.isdir(candidate_dir):
            shutil.rmtree(candidate_dir)

    return best, candidate_results

def main():
    if cfg.SUPPORT_GRID_SIZE < 5 or (cfg.SUPPORT_GRID_SIZE - 1) % 4 != 0:
        raise ValueError("SUPPORT_GRID_SIZE must have the form 4I + 1 for Boole weights.")

    if cfg.GOLDEN_SEARCH_GRADE is not None:
        if cfg.GOLDEN_SEARCH_GRADE in cfg.FIXED_SMOOTHING_BY_GRADE:
            raise ValueError("GOLDEN_SEARCH_GRADE is already fixed in FIXED_SMOOTHING_BY_GRADE.")

    device = torch.device(cfg.DEVICE)
    set_seed(cfg.SEED)

    x_train = data_grid_1d(cfg.N_TRAIN, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device=device, dtype=cfg.DTYPE)
    x_val = data_grid_1d(cfg.N_VAL, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device=device, dtype=cfg.DTYPE)
    x_test = midpoint_grid_1d(cfg.N_TEST, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device=device, dtype=cfg.DTYPE)

    y_train = target_function(x_train)
    y_val = target_function(x_val)
    y_test = target_function(x_test)

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    result_root = cfg.RESULT_ROOT
    ensure_dir(result_root)

    save_json(
        {
            "experiment_name": cfg.EXPERIMENT_NAME,
            "seed": cfg.SEED,
            "target": "selected in Example1D.py",
            "model": "auto-depth MGDL with ThreeAffineBlock",
            "smoothing_quadrature": "composite Boole rule",
            "seed_protocol": "set_seed(SEED) once at the beginning of each run",
            "fixed_smoothing_by_grade": cfg.FIXED_SMOOTHING_BY_GRADE,
            "golden_search_grade": cfg.GOLDEN_SEARCH_GRADE,
        },
        os.path.join(result_root, "run_info.json"),
    )

    fixed_smoothing_by_grade = {}
    for key, value in cfg.FIXED_SMOOTHING_BY_GRADE.items():
        fixed_smoothing_by_grade[int(key)] = round_sigma(value)
    final_smoothing_by_grade = dict(fixed_smoothing_by_grade)
    stage_rows = []

    if cfg.GOLDEN_SEARCH_GRADE is None:
        reference = train_Phi_autodepth(data, fixed_smoothing_by_grade, result_root, "reference")
        final_model = reference
        stage_rows.append(
            {
                "stage": "reference",
                "searched_grade": "",
                "searched_sigma": "",
                "decision": "reference",
                "final_val_mse": reference["final_val_mse"],
                "final_val_max": reference["final_val_max"],
                "final_test_mse": reference["final_test_mse"],
                "final_test_max": reference["final_test_max"],
                "accepted_grades": reference["accepted_grades"],
                "result_dir": reference["result_dir"],
            }
        )
    else:
        reference = None
        best, candidate_results = golden_search_sigma_for_grade(
            cfg.GOLDEN_SEARCH_GRADE,
            fixed_smoothing_by_grade,
            reference,
            data,
            result_root,
        )
        final_model = best
        final_smoothing_by_grade[int(cfg.GOLDEN_SEARCH_GRADE)] = float(best["searched_sigma"])

        stage_rows.append(
            {
                "stage": "golden_search",
                "searched_grade": int(cfg.GOLDEN_SEARCH_GRADE),
                "searched_sigma": float(best["searched_sigma"]),
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
            "smoothing_by_grade": str(final_smoothing_by_grade),
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
    save_json(final_smoothing_by_grade, os.path.join(result_root, "selected_smoothing_by_grade.json"))

    print("\nFinal selected smoothing by grade:", final_smoothing_by_grade)
    print("Final ValMSE = %.6e" % final_model["final_val_mse"])
    print("Final TestMSE = %.6e" % final_model["final_test_mse"])

if __name__ == "__main__":
    main()
