"""
Part 3 — Segmentation and dimensionality reduction
Capstone Project — PhonePe Banking Decision Intelligence
"""

import os
import math
import csv
import numpy as np
from common import load_dataset


def run_part3():
    # Load dataset
    # Columns: X1 to X7, outcome, value
    X, y, value = load_dataset()
    n = len(X)  # 800

    # Task 3.1 uses continuous predictors X1 to X5
    X_cont = X[:, 0:5]

    # Standardise over all 800 rows using sample standard deviation (divide by n - 1)
    mean_X_cont = np.mean(X_cont, axis=0)
    sd_X_cont = np.sqrt(np.sum((X_cont - mean_X_cont) ** 2, axis=0) / (n - 1))
    X_std = (X_cont - mean_X_cont) / sd_X_cont

    # -------------------------------------------------------------
    # Task 3.1: K-means from scratch for K = 2, 3, 4, 5, 6
    # -------------------------------------------------------------
    wcss_dict = {}
    assignments_dict = {}
    centroids_dict = {}

    for K in [2, 3, 4, 5, 6]:
        step_val = 800 // K
        init_indices = [k * step_val for k in range(K)]
        centroids = X_std[init_indices].copy()

        assignments = np.zeros(n, dtype=int)

        while True:
            # Assignment pass
            new_assignments = np.zeros(n, dtype=int)
            for i in range(n):
                row = X_std[i]
                dists = np.sum((centroids - row) ** 2, axis=1)
                # ties go to the lower-numbered centroid
                best_cluster = int(np.argmin(dists))
                new_assignments[i] = best_cluster

            # Check if any assignment changed
            changed = not np.array_equal(new_assignments, assignments)
            assignments = new_assignments

            if not changed:
                # Loop stops at first assignment pass in which no row changes cluster
                break

            # Update centroids
            for k in range(K):
                members = X_std[assignments == k]
                if len(members) > 0:
                    centroids[k] = np.mean(members, axis=0)

        # Compute WCSS: sum over all rows of squared distance to row's final centroid
        wcss = 0.0
        for i in range(n):
            c_idx = assignments[i]
            wcss += float(np.sum((X_std[i] - centroids[c_idx]) ** 2))

        wcss_dict[K] = wcss
        assignments_dict[K] = assignments
        centroids_dict[K] = centroids

    wcss_k2 = wcss_dict[2]
    wcss_k3 = wcss_dict[3]
    wcss_k4 = wcss_dict[4]
    wcss_k5 = wcss_dict[5]
    wcss_k6 = wcss_dict[6]

    # -------------------------------------------------------------
    # Task 3.2: Choose K with the elbow
    # Smallest K in {3, 4, 5} for which:
    # WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    # -------------------------------------------------------------
    elbow_k = None
    for K in [3, 4, 5]:
        drop_curr = wcss_dict[K] - wcss_dict[K + 1]
        drop_prev = wcss_dict[K - 1] - wcss_dict[K]
        if drop_curr < 0.5 * drop_prev:
            elbow_k = K
            break

    # If none found (though one should match), fallback to 3
    if elbow_k is None:
        elbow_k = 3

    # -------------------------------------------------------------
    # Task 3.3: Profile the K = 3 segments
    # Order the three clusters by ascending mean of X3 (in original units)
    # Call them segments A, B, C
    # -------------------------------------------------------------
    assign_k3 = assignments_dict[3]
    cluster_means_X3 = []
    for k in range(3):
        mask = (assign_k3 == k)
        cluster_means_X3.append((np.mean(X[mask, 2]), k))

    # Sort ascending by mean of X3
    cluster_means_X3.sort(key=lambda x: x[0])
    ordered_cluster_indices = [item[1] for item in cluster_means_X3]
    segment_names = ["A", "B", "C"]

    segments_data = []
    sizes = {}

    for s_name, c_idx in zip(segment_names, ordered_cluster_indices):
        mask = (assign_k3 == c_idx)
        sz = int(np.sum(mask))
        sizes[s_name] = sz
        m_x1 = float(np.mean(X[mask, 0]))
        m_x2 = float(np.mean(X[mask, 1]))
        m_x3 = float(np.mean(X[mask, 2]))
        m_x4 = float(np.mean(X[mask, 3]))
        m_x5 = float(np.mean(X[mask, 4]))
        out_rate = float(np.mean(y[mask]))
        m_val = float(np.mean(value[mask]))
        segments_data.append((s_name, sz, m_x1, m_x2, m_x3, m_x4, m_x5, out_rate, m_val))

    size_segment_A = sizes["A"]
    size_segment_B = sizes["B"]
    size_segment_C = sizes["C"]

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part3_segments.csv"), "w", newline="") as f:
        f.write("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value\n")
        for row in segments_data:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f},{row[7]:.4f},{row[8]:.4f}\n")

    # Map each row's assignment to segment label A, B, or C
    row_to_segment = {}
    for s_name, c_idx in zip(segment_names, ordered_cluster_indices):
        for i in range(n):
            if assign_k3[i] == c_idx:
                row_to_segment[i] = s_name

    # -------------------------------------------------------------
    # Task 3.4: Hierarchical clustering on supplied table
    # Cohorts E1 to E6
    # -------------------------------------------------------------
    cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ], dtype=np.float64)

    def run_agglomerative(rule):
        # Initial clusters: [{0}, {1}, ..., {5}]
        clusters = [[i] for i in range(6)]
        merges = []

        for merge_step in range(1, 6):
            best_dist = float("inf")
            best_pair = (-1, -1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1 = clusters[i]
                    c2 = clusters[j]
                    sub_dists = [dist_matrix[u, v] for u in c1 for v in c2]

                    if rule == "single":
                        d_val = min(sub_dists)
                    elif rule == "complete":
                        d_val = max(sub_dists)
                    elif rule == "average":
                        d_val = sum(sub_dists) / len(sub_dists)

                    if d_val < best_dist:
                        best_dist = d_val
                        best_pair = (i, j)

            i, j = best_pair
            c1 = clusters[i]
            c2 = clusters[j]

            # Format cluster names: sorted ascending member names joined by '+'
            c1_str = "+".join(sorted([cohort_names[idx] for idx in c1]))
            c2_str = "+".join(sorted([cohort_names[idx] for idx in c2]))

            # cluster_a is the cluster whose smallest member number is the smaller one
            min_c1 = min(c1)
            min_c2 = min(c2)
            if min_c1 < min_c2:
                cl_a_str, cl_b_str = c1_str, c2_str
            else:
                cl_a_str, cl_b_str = c2_str, c1_str

            merges.append((rule, merge_step, cl_a_str, cl_b_str, best_dist))

            # Merge clusters
            new_cl = sorted(c1 + c2)
            new_clusters = [clusters[idx] for idx in range(len(clusters)) if idx not in (i, j)]
            new_clusters.append(new_cl)
            clusters = new_clusters

        # Cut dendrogram: four gaps between consecutive merge heights (merge 2 - 1, ..., merge 5 - 4)
        # cut lies in largest gap; number of clusters = 6 - (number of merges below cut)
        heights = [m[4] for m in merges]
        gaps = [heights[g] - heights[g - 1] for g in range(1, 5)]
        max_gap_idx = int(np.argmax(gaps))  # 0 corresponds to merge 2 - merge 1, i.e., 1 merge below cut
        merges_below_cut = max_gap_idx + 1
        clusters_at_cut = 6 - merges_below_cut

        return merges, clusters_at_cut

    single_merges, single_clusters_at_cut = run_agglomerative("single")
    complete_merges, complete_clusters_at_cut = run_agglomerative("complete")
    average_merges, average_clusters_at_cut = run_agglomerative("average")

    all_hier_rows = single_merges + complete_merges + average_merges
    with open(os.path.join("outputs", "part3_hierarchical.csv"), "w", newline="") as f:
        f.write("linkage,merge_no,cluster_a,cluster_b,height\n")
        for row in all_hier_rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.4f}\n")

    # -------------------------------------------------------------
    # Task 3.5: PCA by the seven steps
    # Standardised 800 x 5 matrix of Task 3.1
    # -------------------------------------------------------------
    # (1) mean vector: all zeros
    # (2) centre data: X_std is already centred
    # (3) covariance matrix: S = (1 / (n - 1)) X^T X
    S = np.dot(X_std.T, X_std) / (n - 1)

    # (4) eigenvalues and eigenvectors using numpy.linalg.eigh
    eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(S)

    # (5) sort by eigenvalue, largest first
    sort_order = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[sort_order]
    eigenvectors = eigenvectors_raw[:, sort_order]

    # Sign convention: flip each eigenvector so entry of largest absolute value is positive
    for col in range(5):
        v = eigenvectors[:, col]
        max_abs_idx = int(np.argmax(np.abs(v)))
        if v[max_abs_idx] < 0:
            eigenvectors[:, col] = -v

    eigenvalue_sum = float(np.sum(eigenvalues))
    eigenvalue_1 = float(eigenvalues[0])
    eigenvalue_2 = float(eigenvalues[1])
    eigenvalue_3 = float(eigenvalues[2])
    eigenvalue_4 = float(eigenvalues[3])
    eigenvalue_5 = float(eigenvalues[4])

    explained_ratios = eigenvalues / eigenvalue_sum
    explained_ratio_1 = float(explained_ratios[0])
    explained_ratio_2 = float(explained_ratios[1])
    explained_ratio_3 = float(explained_ratios[2])
    explained_ratio_4 = float(explained_ratios[3])
    explained_ratio_5 = float(explained_ratios[4])

    cumulative_ratios = np.cumsum(explained_ratios)
    cumulative_ratio_1 = float(cumulative_ratios[0])
    cumulative_ratio_2 = float(cumulative_ratios[1])
    cumulative_ratio_3 = float(cumulative_ratios[2])
    cumulative_ratio_4 = float(cumulative_ratios[3])
    cumulative_ratio_5 = float(cumulative_ratios[4])

    # Smallest number of components with cumulative ratio >= 0.8
    components_for_80pct = int(np.where(cumulative_ratios >= 0.8)[0][0] + 1)

    # Loadings of PC1 and PC2
    pc1_loading_X1 = float(eigenvectors[0, 0])
    pc1_loading_X2 = float(eigenvectors[1, 0])
    pc1_loading_X3 = float(eigenvectors[2, 0])
    pc1_loading_X4 = float(eigenvectors[3, 0])
    pc1_loading_X5 = float(eigenvectors[4, 0])

    pc2_loading_X1 = float(eigenvectors[0, 1])
    pc2_loading_X2 = float(eigenvectors[1, 1])
    pc2_loading_X3 = float(eigenvectors[2, 1])
    pc2_loading_X4 = float(eigenvectors[3, 1])
    pc2_loading_X5 = float(eigenvectors[4, 1])

    # Projection: Z = X_std @ W (top 2 eigenvectors)
    W2 = eigenvectors[:, :2]
    Z = np.dot(X_std, W2)

    row1_pc1, row1_pc2 = float(Z[0, 0]), float(Z[0, 1])
    row2_pc1, row2_pc2 = float(Z[1, 0]), float(Z[1, 1])
    row3_pc1, row3_pc2 = float(Z[2, 0]), float(Z[2, 1])

    # Mean PC1 score for each Task 3.3 segment
    mean_pc1_segment_A = float(np.mean([Z[i, 0] for i in range(n) if row_to_segment[i] == "A"]))
    mean_pc1_segment_B = float(np.mean([Z[i, 0] for i in range(n) if row_to_segment[i] == "B"]))
    mean_pc1_segment_C = float(np.mean([Z[i, 0] for i in range(n) if row_to_segment[i] == "C"]))

    # -------------------------------------------------------------
    # Write outputs/part3_results.txt
    # -------------------------------------------------------------
    results = [
        ("wcss_k2", f"{wcss_k2:.4f}"),
        ("wcss_k3", f"{wcss_k3:.4f}"),
        ("wcss_k4", f"{wcss_k4:.4f}"),
        ("wcss_k5", f"{wcss_k5:.4f}"),
        ("wcss_k6", f"{wcss_k6:.4f}"),
        ("elbow_k", f"{elbow_k}"),
        ("size_segment_A", f"{size_segment_A}"),
        ("size_segment_B", f"{size_segment_B}"),
        ("size_segment_C", f"{size_segment_C}"),
        ("single_clusters_at_cut", f"{single_clusters_at_cut}"),
        ("complete_clusters_at_cut", f"{complete_clusters_at_cut}"),
        ("average_clusters_at_cut", f"{average_clusters_at_cut}"),
        ("eigenvalue_sum", f"{eigenvalue_sum:.4f}"),
        ("eigenvalue_1", f"{eigenvalue_1:.4f}"),
        ("eigenvalue_2", f"{eigenvalue_2:.4f}"),
        ("eigenvalue_3", f"{eigenvalue_3:.4f}"),
        ("eigenvalue_4", f"{eigenvalue_4:.4f}"),
        ("eigenvalue_5", f"{eigenvalue_5:.4f}"),
        ("explained_ratio_1", f"{explained_ratio_1:.4f}"),
        ("explained_ratio_2", f"{explained_ratio_2:.4f}"),
        ("explained_ratio_3", f"{explained_ratio_3:.4f}"),
        ("explained_ratio_4", f"{explained_ratio_4:.4f}"),
        ("explained_ratio_5", f"{explained_ratio_5:.4f}"),
        ("cumulative_ratio_1", f"{cumulative_ratio_1:.4f}"),
        ("cumulative_ratio_2", f"{cumulative_ratio_2:.4f}"),
        ("cumulative_ratio_3", f"{cumulative_ratio_3:.4f}"),
        ("cumulative_ratio_4", f"{cumulative_ratio_4:.4f}"),
        ("cumulative_ratio_5", f"{cumulative_ratio_5:.4f}"),
        ("components_for_80pct", f"{components_for_80pct}"),
        ("pc1_loading_X1", f"{pc1_loading_X1:.4f}"),
        ("pc1_loading_X2", f"{pc1_loading_X2:.4f}"),
        ("pc1_loading_X3", f"{pc1_loading_X3:.4f}"),
        ("pc1_loading_X4", f"{pc1_loading_X4:.4f}"),
        ("pc1_loading_X5", f"{pc1_loading_X5:.4f}"),
        ("pc2_loading_X1", f"{pc2_loading_X1:.4f}"),
        ("pc2_loading_X2", f"{pc2_loading_X2:.4f}"),
        ("pc2_loading_X3", f"{pc2_loading_X3:.4f}"),
        ("pc2_loading_X4", f"{pc2_loading_X4:.4f}"),
        ("pc2_loading_X5", f"{pc2_loading_X5:.4f}"),
        ("row1_pc1", f"{row1_pc1:.4f}"),
        ("row1_pc2", f"{row1_pc2:.4f}"),
        ("row2_pc1", f"{row2_pc1:.4f}"),
        ("row2_pc2", f"{row2_pc2:.4f}"),
        ("row3_pc1", f"{row3_pc1:.4f}"),
        ("row3_pc2", f"{row3_pc2:.4f}"),
        ("mean_pc1_segment_A", f"{mean_pc1_segment_A:.4f}"),
        ("mean_pc1_segment_B", f"{mean_pc1_segment_B:.4f}"),
        ("mean_pc1_segment_C", f"{mean_pc1_segment_C:.4f}")
    ]

    out_file = os.path.join("outputs", "part3_results.txt")
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
    run_part3()
