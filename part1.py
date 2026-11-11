"""
Part 1 — Estimation, inference and a regression baseline
Capstone Project — PhonePe Banking Decision Intelligence
"""

import csv
import math
import os
import numpy as np


def run_part1():
    # -------------------------------------------------------------
    # Step 0: Read data/records.csv
    # -------------------------------------------------------------
    records_path = os.path.join("data", "records.csv")
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"Could not find {records_path}")

    # Columns:
    # 0: record_id
    # 1: account_age_months (X1)
    # 2: failed_txn_count_90d (X2)
    # 3: credit_utilisation_ratio (X3)
    # 4: avg_monthly_inflow_inr_thousands (X4)
    # 5: repayment_score (X5)
    # 6: irregular_salary_flag (X6)
    # 7: autopay_flag (X7)
    # 8: loan_default (outcome)
    # 9: monthly_spend_inr_thousands (value)

    raw_data = []
    with open(records_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            raw_data.append([float(x) for x in row[1:]])

    data_arr = np.array(raw_data, dtype=np.float64)

    X1 = data_arr[:, 0]
    X2 = data_arr[:, 1]
    X3 = data_arr[:, 2]
    X4 = data_arr[:, 3]
    X5 = data_arr[:, 4]
    X6 = data_arr[:, 5]
    X7 = data_arr[:, 6]
    outcome = data_arr[:, 7]
    value = data_arr[:, 8]

    n = len(value)

    # -------------------------------------------------------------
    # Task 1.1: Estimate with an interval
    # -------------------------------------------------------------
    mean_value = float(np.sum(value) / n)
    # Sample SD: divide by n - 1
    sd_value = float(np.sqrt(np.sum((value - mean_value) ** 2) / (n - 1)))
    se_value = float(sd_value / np.sqrt(n))
    ci95_low = float(mean_value - 1.96 * se_value)
    ci95_high = float(mean_value + 1.96 * se_value)

    # -------------------------------------------------------------
    # Task 1.2: Test a planning claim
    # H0: mu = 8.0 vs H1: mu != 8.0 at alpha = 0.05
    # -------------------------------------------------------------
    z_stat = float((mean_value - 8.0) / se_value)
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # -------------------------------------------------------------
    # Task 1.3: Size a future sample
    # E = 0.5; smallest integer n >= (1.96 * sd_value / E)^2
    # -------------------------------------------------------------
    E = 0.5
    n_required = int(math.ceil((1.96 * sd_value / E) ** 2))

    # -------------------------------------------------------------
    # Task 1.4: Simple regression by the closed form
    # Regress value on X4: reg_slope = cov(X4, value) / var(X4) (divisor n - 1)
    # -------------------------------------------------------------
    mean_X4 = float(np.sum(X4) / n)
    cov_X4_val = float(np.sum((X4 - mean_X4) * (value - mean_value)) / (n - 1))
    var_X4 = float(np.sum((X4 - mean_X4) ** 2) / (n - 1))
    reg_slope = cov_X4_val / var_X4
    reg_intercept = mean_value - reg_slope * mean_X4

    fitted = reg_slope * X4 + reg_intercept
    tss = float(np.sum((value - mean_value) ** 2))
    ess = float(np.sum((fitted - mean_value) ** 2))
    rss = float(np.sum((value - fitted) ** 2))

    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < (1e-6 * tss) else "no"
    r2 = float(1.0 - rss / tss)
    rmse = float(np.sqrt(rss / n))
    mae = float(np.sum(np.abs(value - fitted)) / n)

    # -------------------------------------------------------------
    # Task 1.5: Fit the same line by gradient descent
    # Standardise X4 into z = (X4 - mean X4) / SD of X4 (sample SD)
    # Model: y_hat = m * z + c, start m = 0, c = 0
    # Cost J = (1/n) sum e^2, e = y - y_hat
    # Gradients: dJ/dm = -(2/n) sum(z * e), dJ/dc = -(2/n) sum(e)
    # Update with eta = 0.1, 500 updates on all 800 rows
    # -------------------------------------------------------------
    sd_X4 = float(np.sqrt(var_X4))
    z = (X4 - mean_X4) / sd_X4

    m = 0.0
    c = 0.0
    eta = 0.1

    def compute_cost(cur_m, cur_c):
        y_hat = cur_m * z + cur_c
        e = value - y_hat
        return float(np.sum(e ** 2) / n)

    gd_J0 = compute_cost(m, c)
    gd_J1 = None
    gd_J10 = None
    gd_J100 = None
    gd_J500 = None

    for step in range(1, 501):
        y_hat = m * z + c
        e = value - y_hat
        grad_m = float(- (2.0 / n) * np.sum(z * e))
        grad_c = float(- (2.0 / n) * np.sum(e))

        m = m - eta * grad_m
        c = c - eta * grad_c

        if step == 1:
            gd_J1 = compute_cost(m, c)
        elif step == 10:
            gd_J10 = compute_cost(m, c)
        elif step == 100:
            gd_J100 = compute_cost(m, c)
        elif step == 500:
            gd_J500 = compute_cost(m, c)

    gd_slope = float(m / sd_X4)
    gd_intercept = float(c - m * mean_X4 / sd_X4)
    gd_slope_gap = float(abs(gd_slope - reg_slope))

    # -------------------------------------------------------------
    # Task 1.6: Check for a funnel, then apply the log remedy
    # Sort 800 rows by fitted value of Task 1.4 ascending (stable sort)
    # Lower group: first 400; Upper group: last 400
    # -------------------------------------------------------------
    residuals_raw = value - fitted
    sort_order_raw = np.argsort(fitted, kind="stable")
    lower_idx_raw = sort_order_raw[:400]
    upper_idx_raw = sort_order_raw[400:]

    res_lower_raw = residuals_raw[lower_idx_raw]
    res_upper_raw = residuals_raw[upper_idx_raw]

    # Sample SD (divide by 400 - 1)
    resid_sd_lower_raw = float(np.sqrt(np.sum((res_lower_raw - np.mean(res_lower_raw)) ** 2) / (len(res_lower_raw) - 1)))
    resid_sd_upper_raw = float(np.sqrt(np.sum((res_upper_raw - np.mean(res_upper_raw)) ** 2) / (len(res_upper_raw) - 1)))
    resid_sd_ratio_raw = float(resid_sd_upper_raw / resid_sd_lower_raw)

    # Log-log regression: ln(value) on ln(X4)
    ln_val = np.log(value)
    ln_X4 = np.log(X4)
    mean_ln_val = float(np.sum(ln_val) / n)
    mean_ln_X4 = float(np.sum(ln_X4) / n)

    cov_ln = float(np.sum((ln_X4 - mean_ln_X4) * (ln_val - mean_ln_val)) / (n - 1))
    var_ln_X4 = float(np.sum((ln_X4 - mean_ln_X4) ** 2) / (n - 1))

    loglog_slope = float(cov_ln / var_ln_X4)
    loglog_intercept = float(mean_ln_val - loglog_slope * mean_ln_X4)

    fitted_log = loglog_slope * ln_X4 + loglog_intercept
    residuals_log = ln_val - fitted_log

    tss_log = float(np.sum((ln_val - mean_ln_val) ** 2))
    rss_log = float(np.sum((ln_val - fitted_log) ** 2))
    loglog_r2 = float(1.0 - rss_log / tss_log)
    loglog_rmse = float(np.sqrt(rss_log / n))

    # Split rows of log model into lower and upper halves by its own fitted values
    sort_order_log = np.argsort(fitted_log, kind="stable")
    res_lower_log = residuals_log[sort_order_log[:400]]
    res_upper_log = residuals_log[sort_order_log[400:]]

    resid_sd_lower_log = float(np.sqrt(np.sum((res_lower_log - np.mean(res_lower_log)) ** 2) / (len(res_lower_log) - 1)))
    resid_sd_upper_log = float(np.sqrt(np.sum((res_upper_log - np.mean(res_upper_log)) ** 2) / (len(res_upper_log) - 1)))
    resid_sd_ratio_log = float(resid_sd_upper_log / resid_sd_lower_log)

    # Task 1.7 adjusted R2 for log-log model (p = 1)
    # adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    loglog_adj_r2 = float(1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1))

    # -------------------------------------------------------------
    # Task 1.7: Multiple regression by the normal equation
    # Regress ln(value) on 8 columns:
    # 1, ln(X4), X1, X2, X3, X5, X6, X7
    # beta_hat = (X^T X)^(-1) X^T Y
    # -------------------------------------------------------------
    ones = np.ones(n, dtype=np.float64)
    X_mat = np.column_stack([ones, ln_X4, X1, X2, X3, X5, X6, X7])
    Y_vec = ln_val

    # Normal equations: (X^T X) beta = X^T Y
    XtX = np.dot(X_mat.T, X_mat)
    XtY = np.dot(X_mat.T, Y_vec)
    betas = np.linalg.solve(XtX, XtY)

    mr_beta_0 = float(betas[0])
    mr_beta_1 = float(betas[1])
    mr_beta_2 = float(betas[2])
    mr_beta_3 = float(betas[3])
    mr_beta_4 = float(betas[4])
    mr_beta_5 = float(betas[5])
    mr_beta_6 = float(betas[6])
    mr_beta_7 = float(betas[7])

    fitted_mr = np.dot(X_mat, betas)
    rss_mr = float(np.sum((Y_vec - fitted_mr) ** 2))
    tss_mr = tss_log  # Y is ln(value)
    mr_r2 = float(1.0 - rss_mr / tss_mr)
    p_mr = 7
    mr_adj_r2 = float(1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1))

    # -------------------------------------------------------------
    # Write outputs/part1_results.txt
    # -------------------------------------------------------------
    results = [
        ("n", f"{n}"),
        ("mean_value", f"{mean_value:.4f}"),
        ("sd_value", f"{sd_value:.4f}"),
        ("se_value", f"{se_value:.4f}"),
        ("ci95_low", f"{ci95_low:.4f}"),
        ("ci95_high", f"{ci95_high:.4f}"),
        ("z_stat", f"{z_stat:.4f}"),
        ("reject_h0", f"{reject_h0}"),
        ("n_required", f"{n_required}"),
        ("reg_slope", f"{reg_slope:.4f}"),
        ("reg_intercept", f"{reg_intercept:.4f}"),
        ("tss", f"{tss:.4f}"),
        ("ess", f"{ess:.4f}"),
        ("rss", f"{rss:.4f}"),
        ("tss_equals_ess_plus_rss", f"{tss_equals_ess_plus_rss}"),
        ("r2", f"{r2:.4f}"),
        ("rmse", f"{rmse:.4f}"),
        ("mae", f"{mae:.4f}"),
        ("gd_J0", f"{gd_J0:.4f}"),
        ("gd_J1", f"{gd_J1:.4f}"),
        ("gd_J10", f"{gd_J10:.4f}"),
        ("gd_J100", f"{gd_J100:.4f}"),
        ("gd_J500", f"{gd_J500:.4f}"),
        ("gd_slope", f"{gd_slope:.4f}"),
        ("gd_intercept", f"{gd_intercept:.4f}"),
        ("gd_slope_gap", f"{gd_slope_gap:.4f}"),
        ("resid_sd_lower_raw", f"{resid_sd_lower_raw:.4f}"),
        ("resid_sd_upper_raw", f"{resid_sd_upper_raw:.4f}"),
        ("resid_sd_ratio_raw", f"{resid_sd_ratio_raw:.4f}"),
        ("loglog_intercept", f"{loglog_intercept:.4f}"),
        ("loglog_slope", f"{loglog_slope:.4f}"),
        ("loglog_r2", f"{loglog_r2:.4f}"),
        ("loglog_rmse", f"{loglog_rmse:.4f}"),
        ("resid_sd_lower_log", f"{resid_sd_lower_log:.4f}"),
        ("resid_sd_upper_log", f"{resid_sd_upper_log:.4f}"),
        ("resid_sd_ratio_log", f"{resid_sd_ratio_log:.4f}"),
        ("mr_beta_0", f"{mr_beta_0:.4f}"),
        ("mr_beta_1", f"{mr_beta_1:.4f}"),
        ("mr_beta_2", f"{mr_beta_2:.4f}"),
        ("mr_beta_3", f"{mr_beta_3:.4f}"),
        ("mr_beta_4", f"{mr_beta_4:.4f}"),
        ("mr_beta_5", f"{mr_beta_5:.4f}"),
        ("mr_beta_6", f"{mr_beta_6:.4f}"),
        ("mr_beta_7", f"{mr_beta_7:.4f}"),
        ("mr_r2", f"{mr_r2:.4f}"),
        ("mr_adj_r2", f"{mr_adj_r2:.4f}"),
        ("loglog_adj_r2", f"{loglog_adj_r2:.4f}")
    ]

    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "part1_results.txt")
    with open(out_file, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Successfully generated {out_file} with {len(results)} keys.")
    for k, v in results:
        print(f"{k} = {v}")


if __name__ == "__main__":
    run_part1()
