import os
import shutil
import time

import torch

import config as cfg
from functions import target_easy_1d
from mgdl import FixedDepthMGDL1D
from trainer import train_full_mgdl
from utils import data_grid_1d, ensure_dir, float_name, midpoint_grid_1d, save_csv, save_json, set_seed

def round_sigma(sigma):
    return round(float(sigma), int(cfg.SIGMA_DECIMALS))

def train_Phi_G(data, sigma_by_grade, result_root, run_label):
    set_seed(cfg.SEED)

    model = FixedDepthMGDL1D(
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
    )
    result["label"] = run_label
    result["sigma_by_grade"] = dict(sigma_by_grade)
    save_json(result, os.path.join(result_dir, "final_metrics.json"))
    return result

def golden_search_sigma_for_grade(data, result_root, accepted_sigmas, grade, current_reference):
    left, right = cfg.SIGMA_INTERVAL[int(grade)]
    left = float(left)
    right = float(right)
    golden_ratio = (5.0 ** 0.5 - 1.0) / 2.0

    c = right - golden_ratio * (right - left)
    d = left + golden_ratio * (right - left)

    candidate_results = []
    evaluated = {}
    golden_rows = []
    iteration = 0

    while True:
        if iteration == 0:
            new_sigmas = [("c", c), ("d", d)]
        elif last_side == "left":
            new_sigmas = [("c", c)]
        else:
            new_sigmas = [("d", d)]

        for role, sigma_raw in new_sigmas:
            sigma = round_sigma(sigma_raw)

            if sigma in evaluated:
                result = evaluated[sigma]
            else:
                sigma_by_grade = dict(accepted_sigmas)
                sigma_by_grade[int(grade)] = sigma
                run_label = "golden_g%d_sigma_%s" % (grade, float_name(sigma))

                print("")
                print("=" * 70)
                print("Candidate for sigma_%d: %.10e" % (grade, float(sigma)))
                print("Sigmas used in this Phi_G run: %s" % str(sigma_by_grade))
                print("=" * 70)

                result = train_Phi_G(data, sigma_by_grade, result_root, run_label)
                result["searched_grade"] = int(grade)
                result["searched_sigma"] = sigma
                evaluated[sigma] = result
                candidate_results.append(result)

            if current_reference is None:
                reference_val_mse = ""
                delta_val_mse = ""
            else:
                reference_val_mse = float(current_reference["final_val_mse"])
                delta_val_mse = float(result["final_val_mse"]) - reference_val_mse

            golden_rows.append(
                {
                    "iteration": iteration,
                    "role": role,
                    "interval_left": left,
                    "interval_right": right,
                    "searched_grade": int(grade),
                    "candidate_sigma": sigma,
                    "final_val_mse_of_Phi_G": float(result["final_val_mse"]),
                    "final_val_max_of_Phi_G": float(result["final_val_max"]),
                    "final_test_mse_of_Phi_G": float(result["final_test_mse"]),
                    "final_test_max_of_Phi_G": float(result["final_test_max"]),
                    "current_reference_final_val_mse": reference_val_mse,
                    "delta_final_val_mse_vs_reference": delta_val_mse,
                    "result_dir": result["result_dir"],
                }
            )

            if role == "c":
                result_c = result
            else:
                result_d = result

        c_score = (float(result_c["final_val_mse"]), float(result_c["final_val_max"]))
        d_score = (float(result_d["final_val_mse"]), float(result_d["final_val_max"]))

        if iteration >= cfg.GOLDEN_MAX_ITER or abs(right - left) <= cfg.GOLDEN_SIGMA_TOL:
            break

        iteration += 1
        if c_score <= d_score:
            right = d
            d = c
            result_d = result_c
            c = right - golden_ratio * (right - left)
            last_side = "left"
        else:
            left = c
            c = d
            result_c = result_d
            d = left + golden_ratio * (right - left)
            last_side = "right"

    best = candidate_results[0]
    for result in candidate_results[1:]:
        result_score = (float(result["final_val_mse"]), float(result["final_val_max"]))
        best_score = (float(best["final_val_mse"]), float(best["final_val_max"]))
        if result_score < best_score:
            best = result

    save_csv(golden_rows, os.path.join(result_root, "golden_grade_%d.csv" % grade))
    save_json(best, os.path.join(result_root, "golden_grade_%d_selected.json" % grade))

    best_dir = os.path.abspath(best["result_dir"])
    for result in candidate_results:
        candidate_dir = os.path.abspath(result["result_dir"])
        if candidate_dir != best_dir and os.path.isdir(candidate_dir):
            shutil.rmtree(candidate_dir)

    return best

def main():
    total_start = time.perf_counter()

    if len(cfg.EPOCHS_PER_GRADE) != cfg.MAX_GRADES:
        raise ValueError("EPOCHS_PER_GRADE must have length MAX_GRADES.")

    search_grades = []
    for grade in range(1, cfg.MAX_GRADES + 1):
        if cfg.SMOOTHING_BY_GRADE.get(grade, False):
            search_grades.append(grade)
            if grade not in cfg.SIGMA_INTERVAL:
                raise ValueError("Missing SIGMA_INTERVAL for searched grade %d." % grade)

    if len(search_grades) == 0:
        search_text = "none"
    else:
        search_text = "-".join(str(grade) for grade in search_grades)

    result_root = os.path.join(
        cfg.RESULT_ROOT,
        "%s__%s__P%d__G%d__w%d__search_grades_%s" % (cfg.EXPERIMENT_NAME, cfg.QUADRATURE_RULE, cfg.P, cfg.MAX_GRADES, cfg.WIDTH, search_text),
    )
    ensure_dir(result_root)

    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    x_train = data_grid_1d(cfg.N_TRAIN, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)
    x_val = data_grid_1d(cfg.N_VAL, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)
    x_test = midpoint_grid_1d(cfg.N_TEST, cfg.DOMAIN_LOW, cfg.DOMAIN_HIGH, device, cfg.DTYPE)
    data = {
        "x_train": x_train,
        "y_train": target_easy_1d(x_train),
        "x_val": x_val,
        "y_val": target_easy_1d(x_val),
        "x_test": x_test,
        "y_test": target_easy_1d(x_test),
    }

    accepted_sigmas = {}
    current_reference = None
    stage_rows = []

    if len(search_grades) == 0:
        print("")
        print("=" * 70)
        print("Single run: no smoothing")
        print("=" * 70)
        current_reference = train_Phi_G(data, {}, result_root, "no_smoothing")
    else:
        for grade in search_grades:
            print("")
            print("#" * 70)
            print("Golden search for sigma_%d" % grade)
            if current_reference is None:
                print("No reference model is trained in this search run.")
            else:
                print("Current selected final ValMSE(Phi_G) = %.10e" % float(current_reference["final_val_mse"]))
            print("#" * 70)

            if current_reference is None:
                previous_val_mse = ""
                previous_val_max = ""
            else:
                previous_val_mse = float(current_reference["final_val_mse"])
                previous_val_max = float(current_reference["final_val_max"])

            best = golden_search_sigma_for_grade(
                data,
                result_root,
                accepted_sigmas,
                grade,
                current_reference,
            )

            if current_reference is None:
                accepted_sigmas[int(grade)] = float(best["searched_sigma"])
                decision = "selected_by_validation"
                delta_val_mse = ""
                current_reference = best
            elif float(best["final_val_mse"]) < float(current_reference["final_val_mse"]) - float(cfg.ACCEPT_TOL):
                accepted_sigmas[int(grade)] = float(best["searched_sigma"])
                decision = "accepted"
                delta_val_mse = float(best["final_val_mse"]) - float(current_reference["final_val_mse"])
                current_reference = best
            elif not cfg.ACCEPT_ONLY_IF_VAL_MSE_IMPROVES:
                accepted_sigmas[int(grade)] = float(best["searched_sigma"])
                decision = "accepted_without_val_improvement"
                delta_val_mse = float(best["final_val_mse"]) - float(current_reference["final_val_mse"])
                current_reference = best
            else:
                decision = "skipped"
                delta_val_mse = float(best["final_val_mse"]) - float(current_reference["final_val_mse"])

            stage_rows.append(
                {
                    "grade": int(grade),
                    "decision": decision,
                    "selected_sigma_candidate": float(best["searched_sigma"]),
                    "previous_final_val_mse": previous_val_mse,
                    "previous_final_val_max": previous_val_max,
                    "candidate_final_val_mse": float(best["final_val_mse"]),
                    "candidate_final_val_max": float(best["final_val_max"]),
                    "candidate_final_test_mse": float(best["final_test_mse"]),
                    "candidate_final_test_max": float(best["final_test_max"]),
                    "delta_candidate_final_val_mse": delta_val_mse,
                    "accepted_sigmas_after_decision": str(dict(accepted_sigmas)),
                    "best_candidate_result_dir": best["result_dir"],
                    "grade_output_tables": os.path.join(best["result_dir"], "grade_outputs"),
                }
            )

    save_csv(stage_rows, os.path.join(result_root, "smoothing_stage_decisions.csv"))
    save_json({str(k): float(v) for k, v in accepted_sigmas.items()}, os.path.join(result_root, "selected_sigmas.json"))

    if len(search_grades) == 0:
        comparison_rows = [
            {
                "model": "no_smoothing",
                "accepted_sigmas": "{}",
                "final_val_mse": float(current_reference["final_val_mse"]),
                "final_val_max": float(current_reference["final_val_max"]),
                "final_test_mse": float(current_reference["final_test_mse"]),
                "final_test_max": float(current_reference["final_test_max"]),
                "result_dir": current_reference["result_dir"],
            }
        ]
    else:
        comparison_rows = [
            {
                "model": "selected_smoothing",
                "accepted_sigmas": str(dict(accepted_sigmas)),
                "final_val_mse": float(current_reference["final_val_mse"]),
                "final_val_max": float(current_reference["final_val_max"]),
                "final_test_mse": float(current_reference["final_test_mse"]),
                "final_test_max": float(current_reference["final_test_max"]),
                "result_dir": current_reference["result_dir"],
            }
        ]
    save_csv(comparison_rows, os.path.join(result_root, "smooth_vs_nonsmooth_comparison.csv"))

    lines = []
    if len(search_grades) == 0:
        lines.append("No-smoothing run")
    else:
        lines.append("Selected smoothing run")
    lines.append("")
    lines.append("Final ValMSE: %.12e" % float(current_reference["final_val_mse"]))
    lines.append("Final TestMSE: %.12e" % float(current_reference["final_test_mse"]))
    lines.append("Selected sigmas: %s" % str(dict(accepted_sigmas)))

    with open(os.path.join(result_root, "smooth_vs_nonsmooth_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    save_json(
        {
            "total_runtime_sec": time.perf_counter() - total_start,
            "result_root": result_root,
            "accepted_sigmas": accepted_sigmas,
            "reference_run_included": int(len(search_grades) == 0),
        },
        os.path.join(result_root, "run_info.json"),
    )

    print("")
    print("Finished.")
    print("Result directory: %s" % result_root)

if __name__ == "__main__":
    main()
