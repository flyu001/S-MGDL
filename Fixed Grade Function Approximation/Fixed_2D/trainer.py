import os
import time

import torch

import config as cfg
from smoother import GaussianSmoother2D
from utils import ensure_dir, mse_and_max_error, save_csv, save_json, support_grid_2d

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

def raw_and_used_grade_output(model, grade, x_query, u_raw_support, sigma_pair):
    u_raw_query = model.u_raw(x_query, grade)

    if sigma_pair is None:
        return u_raw_query, u_raw_query

    smoother = GaussianSmoother2D(x_query, sigma_pair)
    u_used_query = smoother.apply(u_raw_support)
    return u_raw_query, u_used_query

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
    sigma_pair,
    history,
    result_dir,
    save_grade_outputs,
):
    model.freeze_for_grade(grade)

    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.LEARNING_RATE)
    scheduler = StepDecayLR(optimizer, cfg.LEARNING_RATE, cfg.LR_DECAY, cfg.LR_STEP_SIZE)

    if sigma_pair is None:
        support_nodes = None
    else:
        support_nodes = support_grid_2d(
            cfg.P,
            cfg.QUADRATURE_RULE,
            cfg.DOMAIN_LOW,
            cfg.DOMAIN_HIGH,
            device=x_train.device,
            dtype=x_train.dtype,
        )

    residual_train_before = y_train - Phi_train_before
    n_train = x_train.shape[0]
    epochs = int(cfg.EPOCHS_PER_GRADE[grade - 1])
    grade_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train, device=x_train.device)

        for start in range(0, n_train, cfg.BATCH_SIZE):
            batch_index = permutation[start:start + cfg.BATCH_SIZE]
            x_batch = x_train[batch_index]
            residual_batch = residual_train_before[batch_index]

            optimizer.zero_grad()
            u_raw_batch = model.u_raw(x_batch, grade)
            loss = torch.mean((u_raw_batch - residual_batch).pow(2))
            loss.backward()
            optimizer.step()

        scheduler.step()

        if not (epoch == 0 or epoch == epochs - 1 or ((epoch + 1) % cfg.EVAL_EVERY == 0)):
            continue

        model.eval()
        with torch.no_grad():
            if sigma_pair is None:
                u_raw_support = None
            else:
                u_raw_support = model.u_raw(support_nodes, grade)

            _, u_used_train = raw_and_used_grade_output(model, grade, x_train, u_raw_support, sigma_pair)
            _, u_used_val = raw_and_used_grade_output(model, grade, x_val, u_raw_support, sigma_pair)
            _, u_used_test = raw_and_used_grade_output(model, grade, x_test, u_raw_support, sigma_pair)

            Phi_train_after = Phi_train_before + u_used_train
            Phi_val_after = Phi_val_before + u_used_val
            Phi_test_after = Phi_test_before + u_used_test

            train_mse, train_max = mse_and_max_error(Phi_train_after, y_train)
            val_mse, val_max = mse_and_max_error(Phi_val_after, y_val)
            test_mse, test_max = mse_and_max_error(Phi_test_after, y_test)

        if sigma_pair is None:
            sigma_x = ""
            sigma_y = ""
        else:
            sigma_x = float(sigma_pair["sigma_x"])
            sigma_y = float(sigma_pair["sigma_y"])

        history.append(
            {
                "grade": int(grade),
                "epoch": epoch + 1,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "train_mse": train_mse,
                "train_max": train_max,
                "val_mse": val_mse,
                "val_max": val_max,
                "test_mse": test_mse,
                "test_max": test_max,
                "grade_elapsed_sec": time.perf_counter() - grade_start,
            }
        )

    model.eval()
    with torch.no_grad():
        if sigma_pair is None:
            u_raw_support = None
        else:
            u_raw_support = model.u_raw(support_nodes, grade)

        u_raw_train, u_used_train = raw_and_used_grade_output(model, grade, x_train, u_raw_support, sigma_pair)
        u_raw_val, u_used_val = raw_and_used_grade_output(model, grade, x_val, u_raw_support, sigma_pair)
        u_raw_test, u_used_test = raw_and_used_grade_output(model, grade, x_test, u_raw_support, sigma_pair)

    Phi_train_after = Phi_train_before + u_used_train
    Phi_val_after = Phi_val_before + u_used_val
    Phi_test_after = Phi_test_before + u_used_test

    if save_grade_outputs:
        save_grade_output_tables(
            grade,
            sigma_pair,
            x_train,
            y_train,
            Phi_train_before,
            u_raw_train,
            u_used_train,
            x_val,
            y_val,
            Phi_val_before,
            u_raw_val,
            u_used_val,
            x_test,
            y_test,
            Phi_test_before,
            u_raw_test,
            u_used_test,
            result_dir,
        )

    return Phi_train_after, Phi_val_after, Phi_test_after

def save_grade_output_tables(
    grade,
    sigma_pair,
    x_train,
    y_train,
    Phi_train_before,
    u_raw_train,
    u_used_train,
    x_val,
    y_val,
    Phi_val_before,
    u_raw_val,
    u_used_val,
    x_test,
    y_test,
    Phi_test_before,
    u_raw_test,
    u_used_test,
    result_dir,
):
    ensure_dir(os.path.join(result_dir, "grade_outputs"))

    if sigma_pair is None:
        sigma_x = ""
        sigma_y = ""
    else:
        sigma_x = float(sigma_pair["sigma_x"])
        sigma_y = float(sigma_pair["sigma_y"])

    split_data = [
        ("train", x_train, y_train, Phi_train_before, u_raw_train, u_used_train),
        ("val", x_val, y_val, Phi_val_before, u_raw_val, u_used_val),
        ("test", x_test, y_test, Phi_test_before, u_raw_test, u_used_test),
    ]

    for split_name, x, y_true, Phi_before, u_raw, u_used in split_data:
        Phi_after = Phi_before + u_used
        residual_after = y_true - Phi_after

        x_cpu = x.detach().cpu()
        y_cpu = y_true.detach().cpu().reshape(-1)
        Phi_before_cpu = Phi_before.detach().cpu().reshape(-1)
        u_raw_cpu = u_raw.detach().cpu().reshape(-1)
        u_used_cpu = u_used.detach().cpu().reshape(-1)
        Phi_after_cpu = Phi_after.detach().cpu().reshape(-1)
        residual_cpu = residual_after.detach().cpu().reshape(-1)

        rows = []
        for i in range(x_cpu.shape[0]):
            row = {
                "split": split_name,
                "grade": int(grade),
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "x1": float(x_cpu[i, 0]),
                "x2": float(x_cpu[i, 1]),
                "y_true": float(y_cpu[i]),
            }
            row["Phi_before_" + split_name] = float(Phi_before_cpu[i])
            row["u_raw_" + split_name] = float(u_raw_cpu[i])
            row["u_used_" + split_name] = float(u_used_cpu[i])
            row["Phi_after_" + split_name] = float(Phi_after_cpu[i])
            row["residual_after_" + split_name] = float(residual_cpu[i])
            rows.append(row)

        path = os.path.join(result_dir, "grade_outputs", "grade_%d_%s_output.csv" % (grade, split_name))
        save_csv(rows, path)

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
    save_grade_outputs=True,
):
    ensure_dir(result_dir)

    Phi_train = torch.zeros_like(y_train)
    Phi_val = torch.zeros_like(y_val)
    Phi_test = torch.zeros_like(y_test)

    history = []
    run_start = time.perf_counter()

    for grade in range(1, cfg.MAX_GRADES + 1):
        sigma_pair = sigma_by_grade.get(grade, None)

        if sigma_pair is None:
            print("Training grade %d/%d; sigma = none" % (grade, cfg.MAX_GRADES))
        else:
            print(
                "Training grade %d/%d; sigma_x = %.10e; sigma_y = %.10e"
                % (grade, cfg.MAX_GRADES, float(sigma_pair["sigma_x"]), float(sigma_pair["sigma_y"]))
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
            sigma_pair,
            history,
            result_dir,
            save_grade_outputs,
        )

    final_train_mse, final_train_max = mse_and_max_error(Phi_train, y_train)
    final_val_mse, final_val_max = mse_and_max_error(Phi_val, y_val)
    final_test_mse, final_test_max = mse_and_max_error(Phi_test, y_test)

    save_csv(history, os.path.join(result_dir, "training_history.csv"))
    save_json(
        {
            "final_train_mse": final_train_mse,
            "final_train_max": final_train_max,
            "final_val_mse": final_val_mse,
            "final_val_max": final_val_max,
            "final_test_mse": final_test_mse,
            "final_test_max": final_test_max,
            "trained_grades": cfg.MAX_GRADES,
            "runtime_sec": time.perf_counter() - run_start,
            "result_dir": result_dir,
        },
        os.path.join(result_dir, "final_metrics.json"),
    )

    return {
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
