"""
Part 4 — Model-choice and architecture memo
Capstone Project — PhonePe Banking Decision Intelligence
"""

import os
import math
import csv
import numpy as np
from common import load_dataset, stratified_split, standardise_train_test, compute_metrics
from part2 import train_logistic_regression, predict_knn_proba, sigmoid


def run_part4():
    # Load dataset
    X, y, value = load_dataset()
    train_idx, test_idx = stratified_split(X, y)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    train_rows = len(y_train)  # 560
    test_rows = len(y_test)    # 240

    # -------------------------------------------------------------
    # Task 4.1: Which split is purest?
    # 560 training rows in original unstandardised units
    # S1: X6 == 1
    # S2: X7 == 1
    # S3: X3 > 0.5
    # S4: X1 <= 24
    # -------------------------------------------------------------
    def compute_entropy_gini(labels):
        n_node = len(labels)
        if n_node == 0:
            return 0.0, 0.0
        p1 = np.sum(labels) / n_node
        p0 = 1.0 - p1

        # Entropy: - sum p_i log2(p_i)
        ent = 0.0
        if p1 > 0:
            ent -= p1 * math.log2(p1)
        if p0 > 0:
            ent -= p0 * math.log2(p0)

        # Gini: 1 - sum p_i^2
        gini = 1.0 - (p0 ** 2 + p1 ** 2)
        return float(ent), float(gini)

    root_entropy, root_gini = compute_entropy_gini(y_train)

    splits_defs = [
        ("S1", X_train_raw[:, 5] == 1.0),
        ("S2", X_train_raw[:, 6] == 1.0),
        ("S3", X_train_raw[:, 2] > 0.5),
        ("S4", X_train_raw[:, 0] <= 24.0)
    ]

    splits_results = []
    for s_name, yes_mask in splits_defs:
        no_mask = ~yes_mask

        y_yes = y_train[yes_mask]
        y_no = y_train[no_mask]

        n_yes = len(y_yes)
        pos_yes = int(np.sum(y_yes))
        n_no = len(y_no)
        pos_no = int(np.sum(y_no))

        ent_yes, gini_yes = compute_entropy_gini(y_yes)
        ent_no, gini_no = compute_entropy_gini(y_no)

        w_ent = (n_yes / train_rows) * ent_yes + (n_no / train_rows) * ent_no
        w_gini = (n_yes / train_rows) * gini_yes + (n_no / train_rows) * gini_no

        info_gain = float(root_entropy - w_ent)
        gini_gain = float(root_gini - w_gini)

        splits_results.append({
            "split": s_name,
            "n_yes": n_yes,
            "positives_yes": pos_yes,
            "n_no": n_no,
            "positives_no": pos_no,
            "info_gain": info_gain,
            "gini_gain": gini_gain,
            "yes_mask": yes_mask
        })

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part4_splits.csv"), "w", newline="") as f:
        f.write("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain\n")
        for sr in splits_results:
            f.write(f"{sr['split']},{sr['n_yes']},{sr['positives_yes']},{sr['n_no']},{sr['positives_no']},{sr['info_gain']:.4f},{sr['gini_gain']:.4f}\n")

    best_split_info = max(splits_results, key=lambda x: x["info_gain"])
    best_split_gini = max(splits_results, key=lambda x: x["gini_gain"])
    best_split_by_info_gain = best_split_info["split"]
    best_split_by_gini_gain = best_split_gini["split"]

    # -------------------------------------------------------------
    # Task 4.2: A one-question model and two ways to combine models
    # The stump is the best split by info gain.
    # Predicted probability is share of positive training rows on same side.
    # -------------------------------------------------------------
    stump_rate_yes = float(best_split_info["positives_yes"] / best_split_info["n_yes"])
    stump_rate_no = float(best_split_info["positives_no"] / best_split_info["n_no"])

    # Standardise features for logistic and KNN
    X_train_std, X_test_std, _, _ = standardise_train_test(X_train_raw, X_test_raw)

    # 1. Logistic model (from Task 2.2)
    w_lr, b_lr, _ = train_logistic_regression(X_train_std, y_train, eta=0.1, n_epochs=1000)
    p_logistic = sigmoid(np.dot(X_test_std, w_lr) + b_lr)
    pred_logistic = (p_logistic >= 0.3).astype(int)

    # 2. KNN model (Task 2.7 model with chosen K = 5)
    K_best = 5
    p_knn = predict_knn_proba(X_train_std, y_train, X_test_std, K_best)
    # positive when at least (K + 1) / 2 neighbours are positive
    pred_knn = (p_knn >= (K_best + 1) / (2.0 * K_best)).astype(int)

    # 3. Stump predictions on test rows
    # Best split was S3: X3 > 0.5 (column index 2 of original X)
    test_yes_mask = (X_test_raw[:, 2] > 0.5) if best_split_by_info_gain == "S3" else None
    if best_split_by_info_gain == "S1":
        test_yes_mask = (X_test_raw[:, 5] == 1.0)
    elif best_split_by_info_gain == "S2":
        test_yes_mask = (X_test_raw[:, 6] == 1.0)
    elif best_split_by_info_gain == "S4":
        test_yes_mask = (X_test_raw[:, 0] <= 24.0)

    p_stump = np.where(test_yes_mask, stump_rate_yes, stump_rate_no)
    pred_stump = (p_stump >= 0.3).astype(int)

    # 4. Hard vote: positive when at least 2 of 3 predict positive
    votes_sum = pred_logistic + pred_knn + pred_stump
    pred_hard_vote = (votes_sum >= 2).astype(int)

    # 5. Soft vote: positive when mean of 3 probabilities >= 0.3
    p_soft = (p_logistic + p_knn + p_stump) / 3.0
    pred_soft_vote = (p_soft >= 0.3).astype(int)

    voting_models = [
        ("logistic", compute_metrics(y_test, pred_logistic)),
        ("knn", compute_metrics(y_test, pred_knn)),
        ("stump", compute_metrics(y_test, pred_stump)),
        ("hard_vote", compute_metrics(y_test, pred_hard_vote)),
        ("soft_vote", compute_metrics(y_test, pred_soft_vote))
    ]

    with open(os.path.join("outputs", "part4_voting.csv"), "w", newline="") as f:
        f.write("model,tp,fp,tn,fn,precision,recall,f1\n")
        for m_name, m_dict in voting_models:
            f.write(f"{m_name},{m_dict['tp']},{m_dict['fp']},{m_dict['tn']},{m_dict['fn']},{m_dict['precision']:.4f},{m_dict['recall']:.4f},{m_dict['f1']:.4f}\n")

    # -------------------------------------------------------------
    # Task 4.3: Two rounds of AdaBoost by hand
    # -------------------------------------------------------------
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    adaboost_rows = []

    # Round 1
    w1_before = np.ones(8) / 8.0
    err_mask1 = (h1 != y_ada)
    eps1 = float(np.sum(w1_before[err_mask1]))
    alpha1 = float(0.5 * math.log((1.0 - eps1) / eps1))
    w1_unnorm = w1_before * np.exp(-alpha1 * y_ada * h1)
    w1_after = w1_unnorm / np.sum(w1_unnorm)

    for pt in range(8):
        adaboost_rows.append((1, pt + 1, w1_before[pt], w1_unnorm[pt], w1_after[pt]))

    # Round 2
    w2_before = w1_after
    err_mask2 = (h2 != y_ada)
    eps2 = float(np.sum(w2_before[err_mask2]))
    alpha2 = float(0.5 * math.log((1.0 - eps2) / eps2))
    w2_unnorm = w2_before * np.exp(-alpha2 * y_ada * h2)
    w2_after = w2_unnorm / np.sum(w2_unnorm)

    for pt in range(8):
        adaboost_rows.append((2, pt + 1, w2_before[pt], w2_unnorm[pt], w2_after[pt]))

    round1_error = eps1
    round1_alpha = alpha1
    round2_error = eps2
    round2_alpha = alpha2

    # Ensemble predictions on the 8 points: sign(alpha1 * h1 + alpha2 * h2)
    ensemble_preds = np.sign(alpha1 * h1 + alpha2 * h2)
    ensemble_accuracy = float(np.mean(ensemble_preds == y_ada))

    with open(os.path.join("outputs", "part4_adaboost.csv"), "w", newline="") as f:
        f.write("round,point,weight_before,weight_unnormalised,weight_after\n")
        for r_num, pt_num, wb, wu, wa in adaboost_rows:
            f.write(f"{r_num},{pt_num},{wb:.4f},{wu:.4f},{wa:.4f}\n")

    # -------------------------------------------------------------
    # Task 4.4: Out-of-bag share
    # (1 - 1/560)^560
    # -------------------------------------------------------------
    oob_fraction = float((1.0 - 1.0 / 560.0) ** 560)

    # -------------------------------------------------------------
    # Task 4.5: Size the neural networks a later project would need
    # Model A:
    # Option 1: MLP 784 -> 16 -> 16 -> 10
    # Option 2: CNN
    # -------------------------------------------------------------
    # MLP:
    # mlp_hidden1: (784 * 16) + 16 = 12560
    # mlp_hidden2: (16 * 16) + 16 = 272
    # mlp_output: (16 * 10) + 10 = 170
    mlp_h1 = 784 * 16 + 16
    mlp_h2 = 16 * 16 + 16
    mlp_out = 16 * 10 + 10
    mlp_total = mlp_h1 + mlp_h2 + mlp_out

    # CNN:
    # input: 28 x 28 x 1
    # conv1: 5x5 filters, 6 filters, stride 1, pad 2
    # output spatial: (28 - 5 + 2*2)/1 + 1 = 28 -> shape: 28 x 28 x 6
    # params conv1: 5 * 5 * 1 * 6 + 6 = 156
    # pool1: 2x2 stride 2 -> 14 x 14 x 6 (0 params)
    # conv2: 5x5 filters, 16 filters, stride 1, pad 0
    # output spatial: (14 - 5 + 0)/1 + 1 = 10 -> shape: 10 x 10 x 16
    # params conv2: 5 * 5 * 6 * 16 + 16 = 2416
    # pool2: 2x2 stride 2 -> 5 x 5 x 16 = 400 features
    # flatten: 400
    # dense1: 400 -> 32 -> (400 * 32) + 32 = 12832
    # dense2: 32 -> 10 -> (32 * 10) + 10 = 330
    cnn_c1 = 5 * 5 * 1 * 6 + 6
    cnn_c2 = 5 * 5 * 6 * 16 + 16
    cnn_d1 = (5 * 5 * 16) * 32 + 32
    cnn_d2 = 32 * 10 + 10
    cnn_total = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

    architecture_rows = [
        ("mlp_hidden1", mlp_h1),
        ("mlp_hidden2", mlp_h2),
        ("mlp_output", mlp_out),
        ("cnn_conv1", cnn_c1),
        ("cnn_conv2", cnn_c2),
        ("cnn_dense1", cnn_d1),
        ("cnn_dense2", cnn_d2)
    ]

    with open(os.path.join("outputs", "part4_architecture.csv"), "w", newline="") as f:
        f.write("layer,parameters\n")
        for lyr, p_count in architecture_rows:
            f.write(f"{lyr},{p_count}\n")

    # Scaling comparison: 128 x 128 x 3 input
    # FC layer into 1,024 neurons: (128 * 128 * 3 * 1024) + 1024
    fc_layer_params = (128 * 128 * 3 * 1024) + 1024
    # Conv layer: 64 filters of size 3 x 3: (3 * 3 * 3 * 64) + 64
    conv_layer_params = (3 * 3 * 3 * 64) + 64

    # Model B: RNN
    # X_t: 64-dim, H_t: 128-dim, Y: 16 classes
    # H_t = tanh(W_xh X_t + W_hh H_{t-1} + b_h)
    # W_xh: 128 x 64 -> 128 * 64 = 8192
    # W_hh: 128 x 128 -> 128 * 128 = 16384
    # b_h: 128
    # Y = W_hy H_T + b_y
    # W_hy: 16 x 128 -> 16 * 128 = 2048
    # b_y: 16
    rnn_total = (128 * 64) + (128 * 128) + 128 + (16 * 128) + 16

    # -------------------------------------------------------------
    # Write outputs/part4_results.txt
    # -------------------------------------------------------------
    results = [
        ("root_entropy", f"{root_entropy:.4f}"),
        ("root_gini", f"{root_gini:.4f}"),
        ("best_split_by_info_gain", f"{best_split_by_info_gain}"),
        ("best_split_by_gini_gain", f"{best_split_by_gini_gain}"),
        ("stump_rate_yes", f"{stump_rate_yes:.4f}"),
        ("stump_rate_no", f"{stump_rate_no:.4f}"),
        ("round1_error", f"{round1_error:.4f}"),
        ("round1_alpha", f"{round1_alpha:.4f}"),
        ("round2_error", f"{round2_error:.4f}"),
        ("round2_alpha", f"{round2_alpha:.4f}"),
        ("ensemble_accuracy", f"{ensemble_accuracy:.4f}"),
        ("oob_fraction", f"{oob_fraction:.4f}"),
        ("mlp_total", f"{mlp_total}"),
        ("cnn_total", f"{cnn_total}"),
        ("fc_layer_params", f"{fc_layer_params}"),
        ("conv_layer_params", f"{conv_layer_params}"),
        ("rnn_total", f"{rnn_total}")
    ]

    out_file = os.path.join("outputs", "part4_results.txt")
    with open(out_file, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Successfully generated {out_file} with {len(results)} keys.")
    for k, v in results:
        print(f"{k} = {v}")


if __name__ == "__main__":
    run_part4()
