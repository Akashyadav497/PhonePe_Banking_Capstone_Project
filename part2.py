"""
Part 2 — Classification and evaluation
Capstone Project — PhonePe Banking Decision Intelligence
"""

import os
import math
import numpy as np
from common import load_dataset, stratified_split, standardise_train_test, compute_metrics


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y, p):
    # loss = - (1/n) * sum( y * ln(p) + (1-y) * ln(1-p) )
    eps = 1e-15
    p_clamped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clamped) + (1.0 - y) * np.log(1.0 - p_clamped)))


def train_logistic_regression(X_train_std, y_train, eta=0.1, n_epochs=1000, checkpoint_epochs=None):
    n, d = X_train_std.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    checkpoints = {}
    if checkpoint_epochs is None:
        checkpoint_epochs = []

    # Checkpoint epoch 0
    p0 = sigmoid(np.dot(X_train_std, w) + b)
    if 0 in checkpoint_epochs:
        checkpoints[0] = bce_loss(y_train, p0)

    for epoch in range(1, n_epochs + 1):
        z = np.dot(X_train_std, w) + b
        p = sigmoid(z)
        err = p - y_train  # (y_hat - y)

        # Gradients: dL/dw_j = (1/n) sum (y_hat - y) x_j, dL/db = (1/n) sum (y_hat - y)
        grad_w = np.dot(X_train_std.T, err) / n
        grad_b = np.sum(err) / n

        w -= eta * grad_w
        b -= eta * grad_b

        if epoch in checkpoint_epochs:
            p_cur = sigmoid(np.dot(X_train_std, w) + b)
            checkpoints[epoch] = bce_loss(y_train, p_cur)

    return w, b, checkpoints


def predict_knn_proba(X_ref, y_ref, X_query, K):
    """
    K-nearest neighbours from scratch.
    Distance is Euclidean distance on standardised features.
    Ranked by (distance, then position in file order among reference rows; stable sort).
    Returns probability: share of positive neighbours among K.
    """
    n_queries = len(X_query)
    probas = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        diff = X_ref - X_query[i]
        dists = np.sum(diff ** 2, axis=1)  # squared Euclidean preserves ranking
        # stable sort keeps original reference row order for ties
        nn_indices = np.argsort(dists, kind="stable")[:K]
        probas[i] = np.mean(y_ref[nn_indices])

    return probas


def run_part2():
    X, y, value = load_dataset()
    train_idx, test_idx = stratified_split(X, y)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    train_rows = len(y_train)
    train_positives = int(np.sum(y_train))
    test_rows = len(y_test)
    test_positives = int(np.sum(y_test))

    # Standardise
    X_train_std, X_test_std, mean_tr, sd_tr = standardise_train_test(X_train_raw, X_test_raw)

    # -------------------------------------------------------------
    # Task 2.2: Logistic regression from scratch (1000 updates, eta = 0.1)
    # -------------------------------------------------------------
    w, b, checkpoints = train_logistic_regression(
        X_train_std, y_train, eta=0.1, n_epochs=1000,
        checkpoint_epochs=[0, 1, 10, 100, 1000]
    )

    loss_epoch_0 = checkpoints[0]
    loss_epoch_1 = checkpoints[1]
    loss_epoch_10 = checkpoints[10]
    loss_epoch_100 = checkpoints[100]
    loss_epoch_1000 = checkpoints[1000]

    weight_X1, weight_X2, weight_X3, weight_X4, weight_X5, weight_X6, weight_X7 = w
    bias = b

    test_probs = sigmoid(np.dot(X_test_std, w) + b)
    test_bce = bce_loss(y_test, test_probs)

    # -------------------------------------------------------------
    # Task 2.3: Learning-rate effect (200 updates from zero with eta in 0.01, 0.1, 1.0)
    # -------------------------------------------------------------
    _, _, cp_001 = train_logistic_regression(X_train_std, y_train, eta=0.01, n_epochs=200, checkpoint_epochs=[200])
    _, _, cp_01 = train_logistic_regression(X_train_std, y_train, eta=0.1, n_epochs=200, checkpoint_epochs=[200])
    _, _, cp_10 = train_logistic_regression(X_train_std, y_train, eta=1.0, n_epochs=200, checkpoint_epochs=[200])

    loss_epoch_200_eta_001 = cp_001[200]
    loss_epoch_200_eta_01 = cp_01[200]
    loss_epoch_200_eta_10 = cp_10[200]

    # -------------------------------------------------------------
    # Task 2.4: Odds ratios = exp(weight)
    # -------------------------------------------------------------
    odds_ratio_X1 = float(np.exp(weight_X1))
    odds_ratio_X2 = float(np.exp(weight_X2))
    odds_ratio_X3 = float(np.exp(weight_X3))
    odds_ratio_X4 = float(np.exp(weight_X4))
    odds_ratio_X5 = float(np.exp(weight_X5))
    odds_ratio_X6 = float(np.exp(weight_X6))
    odds_ratio_X7 = float(np.exp(weight_X7))

    # -------------------------------------------------------------
    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    # Positive when p >= tau
    # -------------------------------------------------------------
    m_05 = compute_metrics(y_test, (test_probs >= 0.5).astype(int))
    m_03 = compute_metrics(y_test, (test_probs >= 0.3).astype(int))

    blind_preds = np.zeros(test_rows, dtype=int)
    m_blind = compute_metrics(y_test, blind_preds)
    blind_accuracy = m_blind["accuracy"]

    # -------------------------------------------------------------
    # Task 2.6: Threshold sweep for tau in [0.1, ..., 0.9]
    # Write outputs/part2_thresholds.csv
    # -------------------------------------------------------------
    thresholds_rows = []
    for tau_step in range(1, 10):
        t_val = round(tau_step * 0.1, 1)
        m_t = compute_metrics(y_test, (test_probs >= t_val).astype(int))
        tpr = m_t["recall"]
        fpr = m_t["fp"] / (m_t["fp"] + m_t["tn"]) if (m_t["fp"] + m_t["tn"]) > 0 else 0.0
        thresholds_rows.append((t_val, m_t["tp"], m_t["fp"], m_t["tn"], m_t["fn"], tpr, fpr))

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part2_thresholds.csv"), "w", newline="") as f:
        f.write("tau,tp,fp,tn,fn,tpr,fpr\n")
        for row in thresholds_rows:
            f.write(f"{row[0]:.1f},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]:.4f},{row[6]:.4f}\n")

    # -------------------------------------------------------------
    # Task 2.7: K-nearest neighbours with 5-fold cross-validation
    # fold = index % 5 in 560 training rows
    # -------------------------------------------------------------
    odd_Ks = list(range(1, 23, 2))
    cv_f1_scores = {K: [] for K in odd_Ks}

    # Pre-assign folds
    fold_assignment = np.array([i % 5 for i in range(train_rows)])

    for fold in range(5):
        val_mask = (fold_assignment == fold)
        ref_mask = ~val_mask

        X_ref_cv = X_train_std[ref_mask]
        y_ref_cv = y_train[ref_mask]

        X_val_cv = X_train_std[val_mask]
        y_val_cv = y_train[val_mask]

        for K in odd_Ks:
            probas_val = predict_knn_proba(X_ref_cv, y_ref_cv, X_val_cv, K)
            # positive when >= (K + 1) / (2 * K)
            preds_val = (probas_val >= (K + 1) / (2.0 * K)).astype(int)
            m_val = compute_metrics(y_val_cv, preds_val)
            cv_f1_scores[K].append(m_val["f1"])

    mean_cv_f1 = {K: float(np.mean(cv_f1_scores[K])) for K in odd_Ks}

    with open(os.path.join("outputs", "part2_knn_cv.csv"), "w", newline="") as f:
        f.write("K,mean_cv_f1\n")
        for K in odd_Ks:
            f.write(f"{K},{mean_cv_f1[K]:.4f}\n")

    # Best K: highest mean F1, ties go to smaller K
    knn_best_k = odd_Ks[0]
    best_score = -1.0
    for K in odd_Ks:
        if mean_cv_f1[K] > best_score:
            best_score = mean_cv_f1[K]
            knn_best_k = K

    # Refit with all 560 training rows as reference and test on 240 test rows
    knn_test_probas = predict_knn_proba(X_train_std, y_train, X_test_std, knn_best_k)
    knn_test_preds = (knn_test_probas >= (knn_best_k + 1) / (2.0 * knn_best_k)).astype(int)
    m_knn = compute_metrics(y_test, knn_test_preds)

    # -------------------------------------------------------------
    # Task 2.8: Choose the alert threshold with money
    # Cost = 50 * fp + 400 * fn
    # tau = k / 20 for k = 1 ... 19 on TRAINING rows
    # -------------------------------------------------------------
    train_probs = sigmoid(np.dot(X_train_std, w) + b)
    best_cost_train = float("inf")
    cost_tau = 0.05

    for k in range(1, 20):
        cand_tau = round(k / 20.0, 2)
        m_cand = compute_metrics(y_train, (train_probs >= cand_tau).astype(int))
        cost_cand = 50 * m_cand["fp"] + 400 * m_cand["fn"]
        if cost_cand < best_cost_train:
            best_cost_train = cost_cand
            cost_tau = cand_tau

    cost_train_at_tau = best_cost_train

    # Test costs
    m_test_at_tau = compute_metrics(y_test, (test_probs >= cost_tau).astype(int))
    cost_test_at_tau = 50 * m_test_at_tau["fp"] + 400 * m_test_at_tau["fn"]

    m_test_at_05 = compute_metrics(y_test, (test_probs >= 0.5).astype(int))
    cost_test_at_05 = 50 * m_test_at_05["fp"] + 400 * m_test_at_05["fn"]

    # Cost flagging nobody on test rows (every positive missed: fn = test_positives, fp = 0)
    cost_test_flag_nobody = 50 * 0 + 400 * test_positives

    # -------------------------------------------------------------
    # Write outputs/part2_results.txt
    # -------------------------------------------------------------
    results = [
        ("train_rows", f"{train_rows}"),
        ("train_positives", f"{train_positives}"),
        ("test_rows", f"{test_rows}"),
        ("test_positives", f"{test_positives}"),
        ("loss_epoch_0", f"{loss_epoch_0:.4f}"),
        ("loss_epoch_1", f"{loss_epoch_1:.4f}"),
        ("loss_epoch_10", f"{loss_epoch_10:.4f}"),
        ("loss_epoch_100", f"{loss_epoch_100:.4f}"),
        ("loss_epoch_1000", f"{loss_epoch_1000:.4f}"),
        ("weight_X1", f"{weight_X1:.4f}"),
        ("weight_X2", f"{weight_X2:.4f}"),
        ("weight_X3", f"{weight_X3:.4f}"),
        ("weight_X4", f"{weight_X4:.4f}"),
        ("weight_X5", f"{weight_X5:.4f}"),
        ("weight_X6", f"{weight_X6:.4f}"),
        ("weight_X7", f"{weight_X7:.4f}"),
        ("bias", f"{bias:.4f}"),
        ("test_bce", f"{test_bce:.4f}"),
        ("loss_epoch_200_eta_0.01", f"{loss_epoch_200_eta_001:.4f}"),
        ("loss_epoch_200_eta_0.1", f"{loss_epoch_200_eta_01:.4f}"),
        ("loss_epoch_200_eta_1.0", f"{loss_epoch_200_eta_10:.4f}"),
        ("odds_ratio_X1", f"{odds_ratio_X1:.4f}"),
        ("odds_ratio_X2", f"{odds_ratio_X2:.4f}"),
        ("odds_ratio_X3", f"{odds_ratio_X3:.4f}"),
        ("odds_ratio_X4", f"{odds_ratio_X4:.4f}"),
        ("odds_ratio_X5", f"{odds_ratio_X5:.4f}"),
        ("odds_ratio_X6", f"{odds_ratio_X6:.4f}"),
        ("odds_ratio_X7", f"{odds_ratio_X7:.4f}"),
        ("tau_0.5_tp", f"{m_05['tp']}"),
        ("tau_0.5_fp", f"{m_05['fp']}"),
        ("tau_0.5_tn", f"{m_05['tn']}"),
        ("tau_0.5_fn", f"{m_05['fn']}"),
        ("tau_0.5_accuracy", f"{m_05['accuracy']:.4f}"),
        ("tau_0.5_precision", f"{m_05['precision']:.4f}"),
        ("tau_0.5_recall", f"{m_05['recall']:.4f}"),
        ("tau_0.5_f1", f"{m_05['f1']:.4f}"),
        ("tau_0.5_specificity", f"{m_05['specificity']:.4f}"),
        ("tau_0.3_tp", f"{m_03['tp']}"),
        ("tau_0.3_fp", f"{m_03['fp']}"),
        ("tau_0.3_tn", f"{m_03['tn']}"),
        ("tau_0.3_fn", f"{m_03['fn']}"),
        ("tau_0.3_accuracy", f"{m_03['accuracy']:.4f}"),
        ("tau_0.3_precision", f"{m_03['precision']:.4f}"),
        ("tau_0.3_recall", f"{m_03['recall']:.4f}"),
        ("tau_0.3_f1", f"{m_03['f1']:.4f}"),
        ("tau_0.3_specificity", f"{m_03['specificity']:.4f}"),
        ("blind_accuracy", f"{blind_accuracy:.4f}"),
        ("knn_best_k", f"{knn_best_k}"),
        ("knn_tp", f"{m_knn['tp']}"),
        ("knn_fp", f"{m_knn['fp']}"),
        ("knn_tn", f"{m_knn['tn']}"),
        ("knn_fn", f"{m_knn['fn']}"),
        ("knn_accuracy", f"{m_knn['accuracy']:.4f}"),
        ("knn_precision", f"{m_knn['precision']:.4f}"),
        ("knn_recall", f"{m_knn['recall']:.4f}"),
        ("knn_f1", f"{m_knn['f1']:.4f}"),
        ("cost_tau", f"{cost_tau:.2f}"),
        ("cost_train_at_tau", f"{cost_train_at_tau}"),
        ("cost_test_at_tau", f"{cost_test_at_tau}"),
        ("cost_test_at_0.5", f"{cost_test_at_05}"),
        ("cost_test_flag_nobody", f"{cost_test_flag_nobody}")
    ]

    out_file = os.path.join("outputs", "part2_results.txt")
    with open(out_file, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Successfully generated {out_file} with {len(results)} keys.")
    for k, v in results[:10]:
        print(f"{k} = {v}")
    print("...")
    for k, v in results[-5:]:
        print(f"{k} = {v}")


if __name__ == "__main__":
    run_part2()
