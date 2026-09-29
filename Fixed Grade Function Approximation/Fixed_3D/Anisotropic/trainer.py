import csv
import os
import time

import torch

import config as cfg
from smoother import GaussianSmoother3D, grade_output_on_support_grid
from utils import ensure_dir, mse_and_max_error, save_csv, save_json, support_axis_1d


class StepDecayLR:

    def __init__(self, optimizer, eta0, decay, step_size):
        self.optimizer = optimizer
        self.eta0 = float(eta0)
        self.decay = float(decay)
        self.step_size = int(step_size)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        eta = self.eta0 * self.decay ** (self.step_count // self.step_size)
        for group in self.optimizer.param_groups:
            group["lr"] = eta


def smoothed_grade_output(x_query, values_on_support, sigma_triple):
    smoother = GaussianSmoother3D(x_query, sigma_triple)
    return smoother.apply(values_on_support)


def train_one_grade(
    model,
    grade,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    Phi_train_before,
    Phi_val_before,
    Phi_test_before,
    sigma_triple,
    history,
    result_dir,
    searched_grade,
):
    model.freeze_for_grade(grade)

    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.LEARNING_RATE)
    scheduler = StepDecayLR(optimizer, cfg.LEARNING_RATE, cfg.LR_DECAY, cfg.LR_STEP_SIZE)

    if sigma_triple is None:
        support_axis = None
    else:
        support_axis = support_axis_1d(
            cfg.SUPPORT_AXIS_SIZE,
            cfg.DOMAIN_LOW,
            cfg.DOMAIN_HIGH,
            device=x_train.device,
            dtype=x_train.dtype,
        )

    residual_train_before = y_train - Phi_train_before
    n_train = x_train.shape[0]
    epochs = int(cfg.EPOCHS_PER_GRADE[grade - 1])
    grade_start = time.perf_counter()

    u_used_train_final = None
    u_used_val_final = None
    u_used_test_final = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train, device=x_train.device)

        for start in range(0, n_train, int(cfg.BATCH_SIZE)):
            batch_index = permutation[start:start + int(cfg.BATCH_SIZE)]
            x_batch = x_train[batch_index]
            residual_batch = residual_train_before[batch_index]

            optimizer.zero_grad()
            u_raw_batch = model.u_raw(x_batch, grade)
            loss = torch.mean((u_raw_batch - residual_batch).pow(2))
            loss.backward()
            optimizer.step()

        scheduler.step()

        if sigma_triple is None:
            evaluate_now = epoch == epochs - 1 or ((epoch + 1) % int(cfg.EVAL_EVERY) == 0)
        else:
            evaluate_now = epoch == epochs - 1

        if not evaluate_now:
            continue

        model.eval()
        with torch.no_grad():
            if sigma_triple is None:
                u_used_train = model.u_raw(x_train, grade)
                u_used_val = model.u_raw(x_val, grade)
                u_used_test = model.u_raw(x_test, grade)
            else:
                values_on_support = grade_output_on_support_grid(model, grade, support_axis)
                u_used_train = smoothed_grade_output(x_train, values_on_support, sigma_triple)
                u_used_val = smoothed_grade_output(x_val, values_on_support, sigma_triple)
                u_used_test = smoothed_grade_output(x_test, values_on_support, sigma_triple)
                del values_on_support

            Phi_train_after = Phi_train_before + u_used_train
            Phi_val_after = Phi_val_before + u_used_val
            Phi_test_after = Phi_test_before + u_used_test

            train_mse, train_max = mse_and_max_error(Phi_train_after, y_train)
            val_mse, val_max = mse_and_max_error(Phi_val_after, y_val)
            test_mse, test_max = mse_and_max_error(Phi_test_after, y_test)

        if sigma_triple is None:
            sigma_x = ""
            sigma_y = ""
            sigma_z = ""
        else:
            sigma_x = float(sigma_triple["sigma_x"])
            sigma_y = float(sigma_triple["sigma_y"])
            sigma_z = float(sigma_triple["sigma_z"])

        history.append(
            {
                "grade": int(grade),
                "epoch": epoch + 1,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "sigma_z": sigma_z,
                "train_mse": train_mse,
                "train_max": train_max,
                "val_mse": val_mse,
                "val_max": val_max,
                "test_mse": test_mse,
                "test_max": test_max,
                "grade_elapsed_sec": time.perf_counter() - grade_start,
            }
        )

        print(
            "G%d | epoch %d/%d | train %.6e | val %.6e | test %.6e"
            % (grade, epoch + 1, epochs, train_mse, val_mse, test_mse)
        )

        if epoch == epochs - 1:
            u_used_train_final = u_used_train
            u_used_val_final = u_used_val
            u_used_test_final = u_used_test

    Phi_train_after = Phi_train_before + u_used_train_final
    Phi_val_after = Phi_val_before + u_used_val_final
    Phi_test_after = Phi_test_before + u_used_test_final

    if bool(cfg.SAVE_OUTPUT) and searched_grade is None:
        save_grade_output_tables(
            model,
            grade,
            sigma_triple,
            x_train,
            y_train,
            Phi_train_before,
            u_used_train_final,
            x_val,
            y_val,
            Phi_val_before,
            u_used_val_final,
            x_test,
            y_test,
            Phi_test_before,
            u_used_test_final,
            result_dir,
        )

    return Phi_train_after, Phi_val_after, Phi_test_after



def save_grade_output_tables(
    model,
    grade,
    sigma_triple,
    x_train,
    y_train,
    Phi_train_before,
    u_used_train,
    x_val,
    y_val,
    Phi_val_before,
    u_used_val,
    x_test,
    y_test,
    Phi_test_before,
    u_used_test,
    result_dir,
):
    output_dir = os.path.join(result_dir, "grade_outputs")
    ensure_dir(output_dir)

    if sigma_triple is None:
        sigma_x = ""
        sigma_y = ""
        sigma_z = ""
    else:
        sigma_x = float(sigma_triple["sigma_x"])
        sigma_y = float(sigma_triple["sigma_y"])
        sigma_z = float(sigma_triple["sigma_z"])

    split_data = [
        ("train", x_train, y_train, Phi_train_before, u_used_train),
        ("val", x_val, y_val, Phi_val_before, u_used_val),
        ("test", x_test, y_test, Phi_test_before, u_used_test),
    ]

    model.eval()
    chunk_size = int(cfg.BATCH_SIZE)

    for split_name, x, y_true, Phi_before, u_used in split_data:
        path = os.path.join(
            output_dir,
            "grade_%d_%s_output.csv" % (grade, split_name),
        )

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "split",
                    "grade",
                    "sigma_x",
                    "sigma_y",
                    "sigma_z",
                    "x1",
                    "x2",
                    "x3",
                    "y_true",
                    "Phi_before_" + split_name,
                    "u_raw_" + split_name,
                    "u_used_" + split_name,
                    "Phi_after_" + split_name,
                    "residual_after_" + split_name,
                ]
            )

            for start in range(0, x.shape[0], chunk_size):
                end = min(start + chunk_size, x.shape[0])

                with torch.no_grad():
                    if sigma_triple is None:
                        u_raw_chunk = u_used[start:end]
                    else:
                        u_raw_chunk = model.u_raw(x[start:end], grade)

                x_cpu = x[start:end].detach().cpu().numpy()
                y_cpu = y_true[start:end].detach().cpu().reshape(-1).numpy()
                Phi_before_cpu = Phi_before[start:end].detach().cpu().reshape(-1).numpy()
                u_raw_cpu = u_raw_chunk.detach().cpu().reshape(-1).numpy()
                u_used_cpu = u_used[start:end].detach().cpu().reshape(-1).numpy()
                Phi_after_cpu = Phi_before_cpu + u_used_cpu
                residual_cpu = y_cpu - Phi_after_cpu

                writer.writerows(
                    (
                        split_name,
                        int(grade),
                        sigma_x,
                        sigma_y,
                        sigma_z,
                        float(x_cpu[i, 0]),
                        float(x_cpu[i, 1]),
                        float(x_cpu[i, 2]),
                        float(y_cpu[i]),
                        float(Phi_before_cpu[i]),
                        float(u_raw_cpu[i]),
                        float(u_used_cpu[i]),
                        float(Phi_after_cpu[i]),
                        float(residual_cpu[i]),
                    )
                    for i in range(end - start)
                )

def train_full_mgdl(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    sigma_by_grade,
    result_dir,
    searched_grade=None,
):
    ensure_dir(result_dir)

    Phi_train = torch.zeros_like(y_train)
    Phi_val = torch.zeros_like(y_val)
    Phi_test = torch.zeros_like(y_test)

    history = []
    run_start = time.perf_counter()

    for grade in range(1, cfg.MAX_GRADES + 1):
        sigma_triple = sigma_by_grade.get(grade, None)

        if sigma_triple is None:
            print("Training grade %d/%d; sigma = none" % (grade, cfg.MAX_GRADES))
        else:
            print(
                "Training grade %d/%d; sigma_x = %.10e; sigma_y = %.10e; sigma_z = %.10e"
                % (
                    grade,
                    cfg.MAX_GRADES,
                    float(sigma_triple["sigma_x"]),
                    float(sigma_triple["sigma_y"]),
                    float(sigma_triple["sigma_z"]),
                )
            )

        Phi_train, Phi_val, Phi_test = train_one_grade(
            model,
            grade,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            Phi_train,
            Phi_val,
            Phi_test,
            sigma_triple,
            history,
                    result_dir,
            searched_grade,
        )

    final_train_mse, final_train_max = mse_and_max_error(Phi_train, y_train)
    final_val_mse, final_val_max = mse_and_max_error(Phi_val, y_val)
    final_test_mse, final_test_max = mse_and_max_error(Phi_test, y_test)

    result = {
        "final_train_mse": final_train_mse,
        "final_train_max": final_train_max,
        "final_val_mse": final_val_mse,
        "final_val_max": final_val_max,
        "final_test_mse": final_test_mse,
        "final_test_max": final_test_max,
        "trained_grades": cfg.MAX_GRADES,
        "runtime_sec": time.perf_counter() - run_start,
        "result_dir": result_dir,
    }

    save_csv(history, os.path.join(result_dir, "training_history.csv"))
    save_json(result, os.path.join(result_dir, "final_metrics.json"))
    return result
