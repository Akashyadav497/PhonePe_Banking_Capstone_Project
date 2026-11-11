Domain: Banking

## How to run

To reproduce all results from a clean repository clone, execute the following commands in order from the repository root:

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (if data/records.csv is not present)
python generate_data.py

# 3. Execute Part 1 (Estimation, inference, and regression baseline)
python part1.py

# 4. Execute Part 2 (Classification, ROC sweep, KNN CV, and cost optimisation)
python part2.py

# 5. Execute Part 3 (K-means, hierarchical clustering, and PCA)
python part3.py

# 6. Execute Part 4 (Entropy/Gini splits, voting ensemble, AdaBoost, and architecture sizing)
python part4.py
```

All generated text results and CSV files will be written to the `outputs/` directory.

## Part 1 findings

- **R1.1 Hypothesis Test & Planning Claim**: The null hypothesis H0: μ = 8.0 is rejected at significance level α = 0.05 because the test statistic |z_stat| = 2.8930 exceeds the two-tailed critical value of 1.96. In domain terms, a Type I error would occur if the risk operations team falsely concludes that average monthly customer spend differs from the planned benchmark of INR 8,000 when the true population mean is indeed INR 8,000, triggering an unnecessary revision of lending limits and financial projections. The observed difference mean_value - 8.0 = 0.9382 column units represents an excess spend of INR 938.20 per borrower; in digital lending portfolios operating on tight margins, a ~11.7% deviation is practically meaningful and warrants updating underwriting assumptions.
- **R1.2 Sample Sizing**: The minimum required sample size to estimate the mean spend within a margin of error E = 0.5 (INR 500) at 95% confidence is n_required = 1293. Because the available book contains only 800 rows (800 < 1293), the planning team's current dataset is underpowered to guarantee the desired precision of ±INR 500, implying that broader confidence bounds must be accepted until ~500 additional accounts are observed.
- **R1.3 Gradient Descent Verification**: The absolute slope gap between closed-form OLS and gradient descent is `gd_slope_gap = 0.0000`. Gradient descent reached the exact closed-form solution because the objective function for linear regression is strictly convex with a single global minimum, and feature standardisation ensured circular contours for smooth, rapid convergence within 500 full-batch steps at learning rate η = 0.1.
- **R1.4 Residual Funnel & Log Transformation**: The ratio of residual standard deviations in the raw regression model is resid_sd_ratio_raw = 3.7525 (upper half 8.6891 vs. lower half 2.3155), displaying severe heteroskedasticity (a widening residual funnel where variance grows with fitted spend). Applying the log-log transformation reduces this ratio to resid_sd_ratio_log = 1.0396 (upper 0.5369 vs. lower 0.5165), effectively eliminating the funnel and restoring homoskedastic residual behaviour.
- **R1.5 Multiple Regression Coefficients & Adjusted R²**: In Task 1.7, the predictors whose coefficients have an absolute value exceeding the mechanical threshold of 0.05 are ln(X4) (1.1576), X2 (0.0940), X3 (0.1067), and X6 (0.1518). Comparing the adjusted R² of the multiple regression model (mr_adj_r2 = 0.7403) against the bivariate log-log model (loglog_adj_r2 = 0.6561) demonstrates an improvement of ~8.4 percentage points of explained variance, confirming that the additional behavioural friction and repayment predictors earn their place in the model.

## Part 2 findings

- **R2.1 Learning Rate Comparison**: Ordering the training losses after 200 epochs from lowest to highest yields η = 1.0 (0.3703) < η = 0.1 (0.3776) < η = 0.01 (0.5091). This demonstrates that an excessively small learning rate (η = 0.01) progresses sluggishly and fails to converge within 200 epochs, whereas larger rates advance substantially faster along smooth convex surfaces without destabilising when the surface is well-conditioned.
- **R2.2 Key Predictors & Odds Ratios**: The three predictors with the largest absolute weights are X3 (credit_utilisation_ratio, weight = 0.9013), X1 (account_age_months, weight = -0.5554), and X2 (failed_txn_count_90d, weight = 0.5505). The largest odds ratio belongs to X3 (odds_ratio_X3 = 2.4629): for every increase of one training standard deviation in credit utilisation ratio (~0.23), the odds of loan default multiply by 2.4629, holding all other borrower attributes fixed.
- **R2.3 Classification Threshold Dynamics**: On the test partition, the baseline naive classifier yields blind_accuracy = 0.7875 (predicting no defaults for all 240 rows), while logistic regression at τ = 0.5 achieves an accuracy of 0.8042. Blind accuracy illustrates why accuracy is deceptive for imbalanced books (~21.25% defaults): a completely uninformative model achieves nearly 79% accuracy yet detects 0 defaults. Lowering τ from 0.5 to 0.3 increases recall dramatically from 0.4706 to 0.7451 (detecting 38 defaults instead of 24), while precision drops moderately from 0.5455 to 0.4634.
- **R2.4 Model Comparison (Logistic vs. KNN)**: Comparing the logistic regression model at τ = 0.3 with the cross-validated K-NN model (K = 5) on the test set, logistic regression achieves an F1 of 0.5714 and a recall of 0.7451, whereas K-NN attains an F1 of 0.3696 and a recall of only 0.3333. Logistic regression heavily outperforms K-NN by capturing linear risk trends that metric distance struggles to separate in sparse 7-dimensional space.
- **R2.5 Policy Deployment & Threshold Selection**: Deploying logistic regression at the cost-optimal threshold cost_tau = 0.15 yields an expected test operational cost of INR 6050, compared to INR 11800 at default τ = 0.5 and INR 20400 for flagging nobody (cost_test_flag_nobody). Threshold cost_tau was chosen strictly on the training set because calibrating decision thresholds on test data introduces data snooping, resulting in overly optimistic cost expectations that fail to generalise out-of-sample.

## Part 3 findings

- **R3.1 K-means WCSS Drops & Elbow Selection**: The four successive WCSS drops are:
  - WCSS(2) - WCSS(3) = 2262.6753 - 1602.8471 = 659.8282
  - WCSS(3) - WCSS(4) = 1602.8471 - 1494.7732 = 108.0739
  - WCSS(4) - WCSS(5) = 1494.7732 - 1323.7157 = 171.0575
  - WCSS(5) - WCSS(6) = 1323.7157 - 1278.1691 = 45.5466
  
  The mathematical elbow rule identifies K = 3 because the drop from K=3 to K=4 (108.0739) is less than half the preceding drop (0.5 * 659.8282 = 329.9141). The non-monotonic drop at K=5 highlights that pinned deterministic starting rows yield reproducible partitions rather than guaranteed global optima.
- **R3.2 Segment Profiling & Business Strategy**:
  - **Segment A ("Prime Established Borrowers", size = 322)**: Longest account age (47.50 months), lowest friction (1.95 failed transactions), low credit utilisation (0.2250), high repayment score (62.16), and lowest default rate (1.86%). *Action*: Offer automated pre-approved prime credit limit increases and premium digital payment reward tiers.
  - **Segment B ("High-Volume Growth Borrowers", size = 255)**: Highest monthly inflow (INR 28,719) and spend (INR 17,125), moderate tenure (28.61 months), elevated transaction friction (6.07 failed txns), and moderate default rate of 23.53%. *Action*: Provide proactive liquidity buffers and smart repayment scheduling linked to salary credits.
  - **Segment C ("High-Risk Subprime Borrowers", size = 223)**: Highest default rate at 46.64%, highest credit utilisation (0.7118), shortest account tenure (14.39 months), and lowest repayment score (38.52). *Action*: Implement strict underwriting caps, daily micro-repayments, and mandatory auto-debit covenants.
- **R3.3 Hierarchical Clustering Cut Discrepancy**: At the dendrogram cut, single linkage forms 4 clusters ({E1}, {E4}, {E5}, and {E2, E3, E6}), whereas complete linkage forms 2 distinct macro-clusters ({E1, E4, E5} and {E2, E3, E6}). Single linkage merges clusters based on the minimum distance between any single pair of points, so the small distance between E2 and E3+E6 (1.4000) causes an early merge, leaving the large jump from merge 2 (1.4) to merge 3 (2.8) of gap 1.4 as the largest gap before E4, E5, and E1 merge. In contrast, complete linkage requires all members of merged clusters to be close (maximum pairwise distance), causing merges 3 and 4 to remain at low heights (2.8 and 2.9) until the final merge between {E1, E4, E5} and {E2, E3, E6} jumps to 7.9 (gap of 5.0), cleanly isolating the two natural 3-cohort borrower clusters.
- **R3.4 Principal Component Analysis**: After standardising X1 to X5, the mean vector is approximately [0, 0, 0, 0, 0]. The first principal component (PC1) accounts for 52.27% of total variance (explained_ratio_1 = 0.5227) and is driven in the positive direction by X1 (+0.5317) and X5 (+0.4418), and in the negative direction by X3 (-0.5187) and X2 (-0.4100), reflecting a distinct "Credit Health vs. Credit Distress" axis. Retaining 3 components is required to capture at least 80% of total variance (cumulative_ratio_3 = 0.8616, components_for_80pct = 3). The segment PC1 means (mean_pc1_segment_A = +1.7773, mean_pc1_segment_B = -0.7831, mean_pc1_segment_C = -1.6709) cleanly separate the borrower segments along this axis, validating the cluster profiles.

## Part 4 findings

- **R4.1 Tree Splits & Regularisation**: Split S4 (X1 <= 24.0 months) is the purest split, yielding the highest information gain (0.1252) and Gini gain (0.0606). A decision tree grown until every leaf is pure memorises sample noise, idiosyncrasies, and outliers, causing high variance and poor generalisation on unseen records. Common stopping regularisers include establishing a minimum leaf sample count (e.g., `min_samples_leaf >= 20`) or capping maximum tree depth.
- **R4.2 Model Evaluation & Ensemble Comparison**: Among the five models evaluated in Task 4.2, the logistic regression model (τ = 0.3) is the superior model to ship, achieving an F1 of 0.5714 and a recall of 0.7451. The decision stump achieved a recall of 0.7647 and F1 of 0.5532. Ensembling models did not improve performance: hard voting dropped F1 to 0.5344 and recall to 0.6863, while soft voting achieved F1 = 0.5507 and recall = 0.7451, primarily because the weak KNN model (F1 = 0.3696, recall = 0.3333) degraded the ensemble consensus.
- **R4.3 AdaBoost Round Dynamics**: After round 1 normalisation, the two misclassified points (Points 3 and 7) have a combined weight of 0.2500 + 0.2500 = 0.5000. Round 2 assigns a higher voting confidence α2 = 0.8047 to hypothesis h2 compared to α1 = 0.5493 for h1 because h2 achieves a lower weighted error (round2_error = 0.1667 vs. round1_error = 0.2500) on the reweighted dataset.
- **R4.4 Deep Learning Architectural Sizing**:
  - **Model A (Option 2 CNN Layer Breakdown)**:

| CNN Layer | Output Shape | Parameters |
|---|---|---|
| Conv1 + ReLU | 28 × 28 × 6 | 156 |
| MaxPool1 | 14 × 14 × 6 | 0 |
| Conv2 + ReLU | 10 × 10 × 16 | 2,416 |
| MaxPool2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 + ReLU | 32 | 12,832 |
| Dense2 | 10 | 330 |

Total parameters = 15,734 (`cnn_total = 15734`).

  - **Model B Recurrent Parameters**: The weight shapes are W_xh (128 × 64), W_hh (128 × 128), and W_hy (16 × 128). The total parameter count (rnn_total = 26768) does not scale with sequence length because recurrent weights are shared across all unrolled time steps.
  - **Classification Settings & Vanishing Gradients**: For 10-class digit recognition, the hidden activation should be ReLU and the output activation Softmax paired with Categorical Cross-Entropy loss. In very deep networks, repeated backpropagation multiplications by small weight matrices cause gradients to decay exponentially toward zero, halting early-layer learning. Skip connections remedy this by providing identity gradient highways that allow gradients to flow backwards unimpeded.

## Model-choice memo

**To:** Head of Lending-Risk Operations  
**From:** Decision Intelligence Lead  
**Date:** 11 November 2026  
**Subject:** Production Delinquency Model Recommendation  

I recommend deploying the **Logistic Regression trained from scratch** operating at alert threshold **τ = 0.15** for production collections intervention.

On the independent test partition of 240 borrowers, this policy flags potential defaults with high operational sensitivity, capturing 45 out of 51 true defaults (test recall = 0.8824 or 88.24%, TP = 45, FP = 73, TN = 116, FN = 6). In rigorous financial terms under PhonePe's loss structure (INR 50 per false alarm, INR 400 per missed default), this policy incurs a total portfolio recovery cost of **INR 6,050** (`cost_test_at_tau = 6050`). This represents a dramatic reduction from the unmanaged default loss of **INR 20,400** (`cost_test_flag_nobody = 20400`) and heavily outperforms the conventional τ = 0.5 rule (**INR 11,800**, `cost_test_at_0.5 = 11800`), saving INR 5,750 per cohort.

Ensemble alternatives were evaluated in Task 4.2: KNN demonstrated poor performance (recall = 33.33%, F1 = 0.3696), dragging down soft voting (recall = 74.51%, F1 = 0.5507) and hard voting (recall = 68.63%, F1 = 0.5344). Logistic regression delivers superior operational reliability and clear interpretability (credit utilisation multiplies default odds by 2.4629, `odds_ratio_X3 = 2.4629`).

A key operational risk is that this threshold was calibrated on a single synthetic cross-sectional split; macroeconomic shifts or festival spending shocks may alter default dynamics. Neither Model A (cheque digit OCR) nor Model B (payment remarks NLP) is worth building before this structured scoring engine is in production, as automated collections interventions on core banking predictors deliver immediate, quantifiable loss reductions exceeding INR 14,000 per cohort.

## Limitations

1. **Synthetic Data Generation**: The underlying records originate from a parametric generator. Real-world Indian digital payments encompass complex multimodal distributions, merchant settlement latencies, and regional credit behaviours not fully captured here.
2. **Single Cross-Sectional Split**: The models and thresholds are validated on a single 70/30 stratified train-test split rather than rolling out-of-time (OOT) payment windows, leaving sensitivity to temporal drift unmeasured.
3. **Small Positive Test Volume**: With only 51 positive default cases in the test set, cell counts in the confusion matrix are subject to integer sensitivity where a single misclassification moves precision or recall by ~2 percentage points.
