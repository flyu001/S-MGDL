import os
import time

import torch

import config as cfg
from smoother import BooleGaussianSmoother2D
from utils import (
    ensure_dir,
    mse_and_max_error,
    save_csv,
    save_json,
    support_grid_2d,
)


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

    smoother = BooleGaussianSmoother2D(x_query, sigma_pair)
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
    start_epoch,
    result_dir,
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
            cfg.SUPPORT_AXIS_SIZE,
            cfg.DOMAIN_LOW,
            cfg.DOMAIN_HIGH,
            device=x_train.device,
            dtype=x_train.dtype,
        )

    residual_train_before = y_train - Phi_train_before
    residual_val_before = y_val - Phi_val_before
    residual_test_before = y_test - Phi_test_before

    n_train = x_train.shape[0]
    epochs = int(cfg.EPOCHS_PER_GRADE)
    history = []
    grade_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
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

            u_raw_train, u_used_train = raw_and_used_grade_output(model, grade, x_train, u_raw_support, sigma_pair)
            u_raw_val, u_used_val = raw_and_used_grade_output(model, grade, x_val, u_raw_support, sigma_pair)
            u_raw_test, u_used_test = raw_and_used_grade_output(model, grade, x_test, u_raw_support, sigma_pair)

            Phi_train_after = Phi_train_before + u_used_train
            Phi_val_after = Phi_val_before + u_used_val
            Phi_test_after = Phi_test_before + u_used_test

            train_mse, train_max = mse_and_max_error(Phi_train_after, y_train)
            val_mse, val_max = mse_and_max_error(Phi_val_after, y_val)
            test_mse, test_max = mse_and_max_error(Phi_test_after, y_test)

            residual_train_mse, _ = mse_and_max_error(u_raw_train, residual_train_before)
            residual_val_mse, _ = mse_and_max_error(u_raw_val, residual_val_before)
            residual_test_mse, _ = mse_and_max_error(u_raw_test, residual_test_before)

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
                "global_epoch": start_epoch + epoch + 1,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "train_mse": train_mse,
                "train_max": train_max,
                "val_mse": val_mse,
                "val_max": val_max,
                "test_mse": test_mse,
                "test_max": test_max,
                "residual_train_mse": residual_train_mse,
                "residual_val_mse": residual_val_mse,
                "residual_test_mse": residual_test_mse,
                "epoch_time_sec": time.perf_counter() - epoch_start,
                "grade_elapsed_sec": time.perf_counter() - grade_start,
            }
        )

        print(
            "G%d | epoch %d/%d | global %d | residual val mse %.3e | full val mse %.3e | full test mse %.3e"
            % (
                grade,
                epoch + 1,
                epochs,
                start_epoch + epoch + 1,
                residual_val_mse,
                val_mse,
                test_mse,
            )
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

    if len(history) == 0 or history[-1]["epoch"] != epochs:
        train_mse, train_max = mse_and_max_error(Phi_train_after, y_train)
        val_mse, val_max = mse_and_max_error(Phi_val_after, y_val)
        test_mse, test_max = mse_and_max_error(Phi_test_after, y_test)
        residual_train_mse, _ = mse_and_max_error(u_raw_train, residual_train_before)
        residual_val_mse, _ = mse_and_max_error(u_raw_val, residual_val_before)
        residual_test_mse, _ = mse_and_max_error(u_raw_test, residual_test_before)
        history.append(
            {
                "grade": int(grade),
                "epoch": epochs,
                "global_epoch": start_epoch + epochs,
                "sigma_x": "" if sigma_pair is None else float(sigma_pair["sigma_x"]),
                "sigma_y": "" if sigma_pair is None else float(sigma_pair["sigma_y"]),
                "train_mse": train_mse,
                "train_max": train_max,
                "val_mse": val_mse,
                "val_max": val_max,
                "test_mse": test_mse,
                "test_max": test_max,
                "residual_train_mse": residual_train_mse,
                "residual_val_mse": residual_val_mse,
                "residual_test_mse": residual_test_mse,
                "epoch_time_sec": 0.0,
                "grade_elapsed_sec": time.perf_counter() - grade_start,
            }
        )


    if cfg.SAVE_GRADE_OUTPUT_TABLES:
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

    return Phi_train_after, Phi_val_after, Phi_test_after, history


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


def train_autodepth(model, x_train, y_train, x_val, y_val, x_test, y_test, smoothing_by_grade, result_dir):
    ensure_dir(result_dir)

    Phi_train = torch.zeros_like(y_train)
    Phi_val = torch.zeros_like(y_val)
    Phi_test = torch.zeros_like(y_test)

    accepted_history = []
    attempted_history = []
    decision_rows = []
    accepted_grade_epochs = []

    accepted_val_mse = None
    start_epoch = 0
    run_start = time.perf_counter()
    stop_reason = None
    stopped_after_attempted_grade = 0

    for grade in range(1, cfg.MAX_GRADES + 1):
        print("")
        print("=" * 70)
        print("Adding / training Grade %d" % grade)
        print("=" * 70)

        model.add_grade()
        model.to(device=x_train.device, dtype=x_train.dtype)

        sigma_pair = smoothing_by_grade.get(grade, None)
        grade_start = time.perf_counter()

        Phi_train_candidate, Phi_val_candidate, Phi_test_candidate, grade_history = train_one_grade(
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
            start_epoch,
            result_dir,
        )
        grade_time = time.perf_counter() - grade_start

        attempted_history.extend(grade_history)
        last_row = grade_history[-1]
        candidate_val_mse = float(last_row["val_mse"])

        if grade == 1:
            improvement = ""
            accepted = True
            decision_reason = "accept first grade"
        else:
            improvement = float(accepted_val_mse) - candidate_val_mse
            accepted = improvement > cfg.MIN_VAL_IMPROVEMENT
            if accepted:
                decision_reason = "accepted: validation MSE improved"
            else:
                decision_reason = "rejected: validation MSE did not improve enough"

        decision_rows.append(
            {
                "grade": int(grade),
                "last_epoch": int(last_row["epoch"]),
                "last_global_epoch": int(last_row["global_epoch"]),
                "previous_accepted_val_mse": accepted_val_mse if accepted_val_mse is not None else "",
                "candidate_val_mse": candidate_val_mse,
                "improvement": improvement,
                "min_val_improvement": cfg.MIN_VAL_IMPROVEMENT,
                "accepted": int(accepted),
                "decision_reason": decision_reason,
                "grade_time_sec": grade_time,
            }
        )

        if accepted:
            accepted_history.extend(grade_history)
            accepted_grade_epochs.append(int(cfg.EPOCHS_PER_GRADE))
            accepted_val_mse = candidate_val_mse
            start_epoch += int(cfg.EPOCHS_PER_GRADE)

            Phi_train = Phi_train_candidate
            Phi_val = Phi_val_candidate
            Phi_test = Phi_test_candidate

            save_csv(accepted_history, os.path.join(result_dir, "accepted_history.csv"))
            save_csv(attempted_history, os.path.join(result_dir, "attempted_history.csv"))
            save_csv(decision_rows, os.path.join(result_dir, "grade_decisions.csv"))

            print("Accepted Grade %d | last val mse %.6e" % (grade, candidate_val_mse))
        else:
            model.remove_last_grade()
            stopped_after_attempted_grade = grade
            stop_reason = "stopped after grade %d: last val mse improvement %.6e <= threshold %.6e" % (
                grade,
                improvement,
                cfg.MIN_VAL_IMPROVEMENT,
            )
            print(stop_reason)
            break

    if stop_reason is None:
        stopped_after_attempted_grade = model.num_grades()
        if model.num_grades() == cfg.MAX_GRADES:
            stop_reason = "reached max_grades=%d" % cfg.MAX_GRADES
        else:
            stop_reason = "training finished"

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
        "accepted_grades": model.num_grades(),
        "accepted_last_val_mse": accepted_val_mse,
        "accepted_grade_epochs": accepted_grade_epochs,
        "stopped_after_attempted_grade": stopped_after_attempted_grade,
        "stop_reason": stop_reason,
        "runtime_sec": time.perf_counter() - run_start,
        "result_dir": result_dir,
    }

    save_csv(accepted_history, os.path.join(result_dir, "accepted_history.csv"))
    save_csv(attempted_history, os.path.join(result_dir, "attempted_history.csv"))
    save_csv(decision_rows, os.path.join(result_dir, "grade_decisions.csv"))
    save_json(result, os.path.join(result_dir, "final_metrics.json"))

    return result
