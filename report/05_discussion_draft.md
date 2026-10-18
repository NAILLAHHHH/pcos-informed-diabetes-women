# V. Discussion

## A. Why Models Diverge: Hyperparameters, Algorithms, and Imbalance

The experimental pattern is coherent once macro-F1—not accuracy—is treated as the decision criterion. **Gradient boosting** maximized accuracy (≈0.86) and OvR AUC (≈0.80) by concentrating correct predictions on the dominant “no diabetes” class, which lowers macro-recall. **Class-weighted random forests** sacrificed some accuracy to recover minority recall, winning macro-F1. **Logistic regression** with balanced weights behaved similarly to a high-recall linear separator: lower accuracy, stronger macro-recall. Among MLPs, the **lighter 64–32 network** beat wider variants, consistent with capacity control under noisy, imbalanced survey labels: extra width mainly fits majority patterns before early stopping restores a similar validation loss.

These outcomes match tabular-learning literature: gradient-boosted trees and forests remain formidable on heterogeneous tables, and MLPs are competitive but not automatically superior [13], [14]. They also echo BRFSS diabetes studies where high AUC coexists with limited sensitivity unless the metric/threshold explicitly prioritizes minority detection [7], [8].

## B. Learning Curves, Overfitting, and Bias–Variance

The random-forest learning curve (Fig. 4) shows macro-F1 rising with training size while a gap remains between training and cross-validation scores. That pattern indicates **variance plus irreducible label noise**, not pure underfitting: additional data helps, but self-report prediabetes labels and feature ceilings bound performance. Deep-learning training curves (Fig. 6) show validation loss plateauing and triggering early stopping; train/validation accuracy diverge modestly rather than catastrophically, suggesting regularization (dropout, weight decay, class weights) constrained overfitting. In bias–variance terms, boosting appears **low-bias/high-majority-fit**, while class-weighted forests/MLPs accept higher majority error to reduce minority bias—trading global accuracy for balanced risk.

## C. Confusion Matrices and Clinically Meaningful Errors

Confusion matrices for the best forest and best MLP (Figs. 2 and 5) concentrate errors on the **prediabetes** row/column. Prediabetes is rare, clinically intermediate, and likely the noisiest self-report category in BRFSS. Adjacent-state confusions (prediabetes ↔ no diabetes) dominate; diabetes versus no-diabetes separation is stronger when `MetabolicBurden` and BMI are elevated (see `error_analysis_by_class.csv`). For a screening-oriented women’s-health narrative, false negatives on intermediate risk are costlier than extra false positives on the majority class—hence macro-F1 and recall-oriented discussion rather than accuracy championship.

ROC OvR curves reinforce the same story: diabetes-versus-rest ranks more easily than prediabetes-versus-rest. Application context therefore favors reporting **class-wise** metrics and cautioning against deployment as a diagnostic tool.

## D. Feature Engineering Ablation

Comparing logistic regression with and without engineered metabolic composites (experiment table, ablation stage) tests whether PCOS-motivated features add signal beyond raw indicators. In our runs the macro-F1 difference was **near-null** (≈0.4305 without vs ≈0.4295 with engineered features), indicating that linear models already capture much of the same cardiometabolic information through raw BMI, blood pressure, and cholesterol. We retain the composites for **interpretability and literature alignment** [1]–[5] and for nonlinear models that can exploit interactions, while openly reporting the ablation rather than claiming performance magic.

## E. Dataset Limitations and Authenticity

Self-report diabetes labels may disagree with laboratory diagnosis; BRFSS lacks HbA1c, fasting glucose, medications, and **PCOS diagnosis fields** [6]. Cross-sectional design precludes incidence modeling. U.S. 2015 adults may not transfer internationally or temporally. These limits explain mid-0.4 macro-F1 values without indicting the pipeline: the task is intrinsically hard under survey noise and severe imbalance.

## F. Ethics and Practical Artifact Hygiene

Artifacts are educational, not clinical decision support. Transparency about missing PCOS labels is an ethical requirement against overselling women’s-health AI. Compact model serialization for Drive delivery preserves nearly identical macro-F1 at practical file sizes—an engineering detail that affects reproducibility as much as accuracy tables.
