# IV. Results (IEEE draft)

Figures referenced below are generated in `figures/` by the experimental pipeline. Insert them in the IEEE manuscript as Figs. 1–6 with captions.

## A. Cohort Summary and Class Imbalance

The raw BRFSS-derived table contains 253,680 adults. Restricting to women yields 141,974 respondents. Fig. 1 (`class_balance_women.png`) shows the diabetes-status distribution in this cohort: the large majority report no diabetes, a smaller fraction report diabetes, and prediabetes is comparatively rare. This imbalance is the dominant statistical context for all subsequent metrics. Models that predict “no diabetes” almost everywhere can still achieve deceptively high accuracy; therefore we emphasize macro-averaged scores and class-wise confusion.

## B. Traditional Machine Learning Performance

Table I summarizes test-set metrics for baseline and tuned scikit-learn models from the controlled experiment run.

**Table I. Traditional ML test performance (women cohort)**

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | ROC-AUC (OvR) |
|-------|----------|---------|---------|----------|---------------|
| Random Forest (tuned) | 0.831 | 0.439 | 0.456 | **0.447** | 0.759 |
| Logistic Regression (tuned) | 0.655 | 0.448 | 0.524 | 0.430 | 0.777 |
| Logistic Regression (baseline) | 0.655 | 0.447 | 0.524 | 0.430 | 0.777 |
| Gradient Boosting (tuned) | **0.860** | 0.479 | 0.392 | 0.405 | **0.797** |
| Gradient Boosting (baseline) | 0.860 | 0.477 | 0.392 | 0.405 | 0.797 |
| Random Forest (baseline) | 0.854 | 0.460 | 0.387 | 0.398 | 0.750 |

Under the pre-declared primary metric (macro-F1), **tuned random forest** is the best traditional model (macro-F1 ≈ 0.447). Gradient boosting attains the highest accuracy (≈ 0.860) and OvR AUC (≈ 0.797) but a lower macro-F1 (≈ 0.405), consistent with prioritizing majority-class correctness over balanced recall across the three labels. Logistic regression trades accuracy for improved macro-recall (≈ 0.52), illustrating the classic precision–recall tension under imbalance.

Fig. 2 (`cm_best_traditional.png`) presents the confusion matrix for the best traditional model. Most mass lies on the no-diabetes diagonal, with nontrivial spillover between adjacent glycemic states. Fig. 3 (`roc_best_traditional.png`) shows one-versus-rest ROC curves; separation is strongest for the diabetes versus rest contrast and weakest for prediabetes, matching prevalence and label noise expectations for an intermediate state.

Fig. 4 (`learning_curve_rf.png`) reports a random-forest learning curve on a 25,000-row stratified subsample for computational tractability. Macro-F1 rises with additional training data and shows a persistent gap between training and cross-validation scores, indicating that variance/imbalance—not merely sample size—limits generalization.

## C. Deep Learning Performance

Table II reports the three MLP configurations trained with class weights, early stopping, and `tf.data` pipelines on identically split, standardized features.

**Table II. Deep learning test performance (women cohort)**

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | ROC-AUC (OvR) |
|-------|----------|---------|---------|----------|---------------|
| **MLP 64–32** | 0.617 | 0.443 | **0.526** | **0.417** | 0.773 |
| MLP 128–64–32 | 0.598 | 0.431 | 0.505 | 0.400 | 0.762 |
| MLP 256–128–64 | 0.566 | 0.454 | 0.498 | 0.399 | 0.775 |

The lighter **MLP 64–32** achieves the best deep macro-F1 (≈ 0.417) and the strongest deep macro-recall (≈ 0.526). Wider networks did not improve macro-F1 under the same early-stopping regime, suggesting that additional capacity mainly fits majority patterns or validation noise rather than minority structure in this feature space.

Fig. 5 (`cm_best_dl.png` and `roc_best_dl.png`) shows the best MLP’s confusion matrix and OvR ROC curves. Relative to the tuned forest, the MLP operates at lower accuracy but competitive ranking quality (AUC ≈ 0.77), again reflecting class-weight-driven recall orientation. Fig. 6 (`dl_training_curves.png`) depicts training/validation loss and accuracy for the selected MLP: validation loss plateaus and triggers early stopping, limiting overfitting while confirming that further epochs yield diminishing returns.

## D. Head-to-Head Comparison

**Table III. Best traditional vs. best deep model**

| Paradigm | Selected model | Accuracy | Macro-F1 | ROC-AUC (OvR) |
|----------|----------------|----------|----------|---------------|
| Traditional ML | Random Forest (tuned) | 0.831 | **0.447** | 0.759 |
| Deep Learning | MLP 64–32 | 0.617 | 0.417 | 0.773 |

On macro-F1, the tuned random forest outperforms the best MLP by approximately three points (0.447 vs. 0.417). The MLP remains within a similar AUC band and offers higher macro-recall than the boosting models, but it does not dominate tabular baselines. This outcome aligns with literature arguing that strong tree ensembles remain highly competitive on medium-width survey tables.

## E. Compact Deployment Artifact

For Google Drive delivery, the winning forest family was re-tuned under depth and estimator caps. The compact forest uses `n_estimators=100`, `max_depth=14`, `min_samples_leaf=2`, and `max_features=sqrt`, achieving test accuracy 0.759, macro-F1 **0.443**, and OvR AUC 0.761—within about 0.004 macro-F1 of the unconstrained tuned forest (0.447)—while shrinking the `.joblib` from roughly 655 MB to about 64 MB. The Keras `.keras` weights remain small (~90 KB) and are packaged with `dl_standard_scaler.joblib`.

## F. Qualitative Error Observations

Across models, prediabetes is the hardest class: it is rare, clinically intermediate, and likely the noisiest self-report category. Confusions between prediabetes and no-diabetes dominate minority errors, whereas diabetes versus no-diabetes separation is comparatively stronger when cardiometabolic burden features are elevated. Mean `MetabolicBurden` increases across true glycemic severity in the test slice, supporting the face validity of the engineered features even when discrete classification of the middle class remains difficult.
