import os
import time

import torch

import config as cfg
from smoother import GaussianSmoother1D
from utils import ensure_dir, mse_and_max_error, save_csv, quadrature_grid_1d

class StepDecayLR:

    def __init__(self, optimizer, eta0, decay, step_size):
        self.optimizer = optimizer
        self.eta0 = float(eta0)
        self.decay = float(decay)
        self.step_size = int(step_size)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        factor = self.decay ** (self.step_count // self.step_size)
        eta = self.eta0 * factor
        for group in self.optimizer.param_groups:
            group["lr"] = eta

def raw_and_used_grade_output(model, grade, x_query, support_grid, sigma):
    u_raw_query = model.u_raw(x_query, grade)

    if sigma is None:
        return u_raw_query, u_raw_query

    u_raw_support = model.u_raw(support_grid, grade)
    smoother = GaussianSmoother1D(x_query, support_grid, sigma)
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
    sigma,
    history,
    result_dir,
):
    model.freeze_for_grade(grade)

    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.LEARNING_RATE)
    scheduler = StepDecayLR(
        optimizer=optimizer,
        eta0=cfg.LEARNING_RATE,
        decay=cfg.LR_DECAY,
        step_size=cfg.LR_STEP_SIZE,
    )

    support_grid = quadrature_grid_1d(
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
            u_used_train = raw_and_used_grade_output(model, grade, x_train, support_grid, sigma)[1]
            u_used_val = raw_and_used_grade_output(model, grade, x_val, support_grid, sigma)[1]
            u_used_test = raw_and_used_grade_output(model, grade, x_test, support_grid, sigma)[1]

            Phi_train_after = Phi_train_before + u_used_train
            Phi_val_after = Phi_val_before + u_used_val
            Phi_test_after = Phi_test_before + u_used_test

            train_mse, train_max = mse_and_max_error(Phi_train_after, y_train)
            val_mse, val_max = mse_and_max_error(Phi_val_after, y_val)
            test_mse, test_max = mse_and_max_error(Phi_test_after, y_test)

            if sigma is None:
                sigma_value = ""
            else:
                sigma_value = float(sigma)

            history.append(
                {
                    "grade": int(grade),
                    "epoch": epoch + 1,
                    "sigma": sigma_value,
                    "Phi_after_grade_train_mse": train_mse,
                    "Phi_after_grade_train_max": train_max,
                    "Phi_after_grade_val_mse": val_mse,
                    "Phi_after_grade_val_max": val_max,
                    "Phi_after_grade_test_mse": test_mse,
                    "Phi_after_grade_test_max": test_max,
                    "grade_elapsed_sec": time.perf_counter() - grade_start,
                }
            )

    model.eval()
    with torch.no_grad():
        u_raw_train, u_used_train = raw_and_used_grade_output(model, grade, x_train, support_grid, sigma)
        u_raw_val, u_used_val = raw_and_used_grade_output(model, grade, x_val, support_grid, sigma)
        u_raw_test, u_used_test = raw_and_used_grade_output(model, grade, x_test, support_grid, sigma)

    Phi_train_after = Phi_train_before + u_used_train
    Phi_val_after = Phi_val_before + u_used_val
    Phi_test_after = Phi_test_before + u_used_test

    ensure_dir(os.path.join(result_dir, "grade_outputs"))
    split_data = [
        ("train", x_train, y_train, Phi_train_before, u_raw_train, u_used_train),
        ("val", x_val, y_val, Phi_val_before, u_raw_val, u_used_val),
        ("test", x_test, y_test, Phi_test_before, u_raw_test, u_used_test),
    ]

    if sigma is None:
        sigma_value = ""
    else:
        sigma_value = float(sigma)

    for split_name, x, y_true, Phi_before, u_raw, u_used in split_data:
        Phi_after = Phi_before + u_used
        residual_after = y_true - Phi_after

        x_cpu = x.detach().cpu().reshape(-1)
        y_cpu = y_true.detach().cpu().reshape(-1)
        Phi_before_cpu = Phi_before.detach().cpu().reshape(-1)
        u_raw_cpu = u_raw.detach().cpu().reshape(-1)
        u_used_cpu = u_used.detach().cpu().reshape(-1)
        Phi_after_cpu = Phi_after.detach().cpu().reshape(-1)
        residual_cpu = residual_after.detach().cpu().reshape(-1)

        rows = []
        for i in range(x_cpu.numel()):
            row = {
                "split": split_name,
                "grade": int(grade),
                "sigma": sigma_value,
                "x": float(x_cpu[i]),
                "y_true": float(y_cpu[i]),
            }
            row["Phi_before_" + split_name] = float(Phi_before_cpu[i])
            row["u_raw_" + split_name] = float(u_raw_cpu[i])
            row["u_used_" + split_name] = float(u_used_cpu[i])
            row["Phi_after_" + split_name] = float(Phi_after_cpu[i])
            row["residual_after_" + split_name] = float(residual_cpu[i])
            rows.append(row)

        save_csv(rows, os.path.join(result_dir, "grade_outputs", "grade_%d_%s_output.csv" % (grade, split_name)))

    return Phi_train_after, Phi_val_after, Phi_test_after

def train_full_mgdl(model, x_train, y_train, x_val, y_val, x_test, y_test, sigma_by_grade, result_dir):
    ensure_dir(result_dir)

    Phi_train = torch.zeros_like(y_train)
    Phi_val = torch.zeros_like(y_val)
    Phi_test = torch.zeros_like(y_test)

    history = []
    run_start = time.perf_counter()

    for grade in range(1, cfg.MAX_GRADES + 1):
        if grade in sigma_by_grade:
            sigma = sigma_by_grade[grade]
        else:
            sigma = None

        if sigma is None:
            sigma_text = "none"
        else:
            sigma_text = "%.10e" % float(sigma)
        print("Training grade %d/%d; sigma = %s" % (grade, cfg.MAX_GRADES, sigma_text))

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
            sigma,
            history,
            result_dir,
        )

    final_train_mse, final_train_max = mse_and_max_error(Phi_train, y_train)
    final_val_mse, final_val_max = mse_and_max_error(Phi_val, y_val)
    final_test_mse, final_test_max = mse_and_max_error(Phi_test, y_test)

    save_csv(history, os.path.join(result_dir, "training_history.csv"))

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
