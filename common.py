"""
Common utilities and dataset loader for PhonePe Banking Decision Intelligence.
Shared between part2.py and part4.py without side effects upon import.
"""

import csv
import os
import math
import numpy as np


def load_dataset(csv_path=os.path.join("data", "records.csv")):
    """
    Loads data/records.csv and returns:
    records: list of raw dicts or lists
    X: 800 x 7 numpy array of predictors (X1 to X7)
    y: 800-element numpy array of binary outcomes (loan_default)
    value: 800-element numpy array of continuous values (monthly_spend_inr_thousands)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_data = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # row: record_id, c1, c2, c3, c4, c5, b1, b2, outcome, value
            raw_data.append([float(x) for x in row[1:]])

    data_arr = np.array(raw_data, dtype=np.float64)
    # X1 to X7 are columns 0 to 6
    X = data_arr[:, 0:7]
    y = data_arr[:, 7]
    value = data_arr[:, 8]
    return X, y, value


def stratified_split(X, y, value=None):
    """
    Task 2.1 Stratified split:
    Take the rows with outcome 0 in file order; the first floor(7 * count / 10) go to training, rest to test.
    Do the same for outcome 1. Both sets keep file order.
    Returns:
    train_indices, test_indices
    """
    n = len(y)
    idx_0 = [i for i in range(n) if y[i] == 0]
    idx_1 = [i for i in range(n) if y[i] == 1]

    n_train_0 = (7 * len(idx_0)) // 10
    n_train_1 = (7 * len(idx_1)) // 10

    train_idx_set = set(idx_0[:n_train_0] + idx_1[:n_train_1])
    test_idx_set = set(idx_0[n_train_0:] + idx_1[n_train_1:])

    # Both sets keep original file order
    train_idx = [i for i in range(n) if i in train_idx_set]
    test_idx = [i for i in range(n) if i in test_idx_set]

    return np.array(train_idx), np.array(test_idx)


def standardise_train_test(X_train, X_test):
    """
    Standardise using TRAINING mean and sample SD (divide by n_train - 1).
    Apply those parameters to test rows.
    """
    n_train = len(X_train)
    mean_train = np.mean(X_train, axis=0)
    # sample SD: divide by n_train - 1
    sd_train = np.sqrt(np.sum((X_train - mean_train) ** 2, axis=0) / (n_train - 1))

    X_train_std = (X_train - mean_train) / sd_train
    X_test_std = (X_test - mean_train) / sd_train

    return X_train_std, X_test_std, mean_train, sd_train


def compute_metrics(y_true, y_pred):
    """
    Computes tp, fp, tn, fn, accuracy, precision, recall, f1, specificity.
    Handles zero denominators as specified in assignment.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }
