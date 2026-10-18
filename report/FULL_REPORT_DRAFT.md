---
# PCOS-Informed Multi-Class Diabetes Classification in Women Using BRFSS Health Indicators

**Required header links**
- GitHub repository: https://github.com/NAILLAHHHH/pcos-informed-diabetes-women
- Google Drive folder: `TODO`
- Presentation video: `TODO` (script: `report/VIDEO_SCRIPT.md`)

**Artifacts:** `figures/` · `report/experiment_results/experiment_table.csv` · offline W&B under `wandb/`
---

# Introduction (IEEE draft)

**Working title:** PCOS-Informed Multi-Class Diabetes Classification in Women Using BRFSS Health Indicators: A Comparative Study of Traditional Machine Learning and Deep Learning

**Repository / deliverable links (replace before submission):**
- GitHub repository: https://github.com/NAILLAHHHH/pcos-informed-diabetes-women
- Google Drive folder: `TODO`
- Presentation video: `TODO`

---

Diabetes mellitus remains one of the most consequential non-communicable diseases worldwide, with rising prevalence, substantial healthcare costs, and well-documented disparities by sex, age, and metabolic risk profile. Among women, endocrine and metabolic conditions further shape diabetes risk across the life course. Polycystic ovary syndrome (PCOS) is the most common endocrine disorder of reproductive-age women and is strongly associated with insulin resistance, impaired glucose tolerance, and subsequent Type 2 diabetes [1]–[5]. That clinical relationship motivates the present study: if metabolic risk accumulates along pathways that overlap with the PCOS phenotype, then carefully designed machine learning (ML) and deep learning (DL) models applied to large population health surveys may help characterize diabetes status in women and illuminate which routinely collected indicators carry signal [6]–[10].

At the same time, scholarly and pedagogical integrity require a clear separation between *motivation* and *measurement*. Most publicly available PCOS-specific clinical tables are small—often hundreds to a few thousand records—and therefore fail minimum sample-size expectations for rigorous comparative modeling. Large government survey systems such as the U.S. Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) contain hundreds of thousands of adult responses and include diabetes history alongside cardiometabolic and lifestyle indicators, yet they do not typically record a confirmed PCOS diagnosis. Accordingly, this project does **not** claim to detect PCOS, invent PCOS labels, or estimate PCOS-conditional diabetes risk from unavailable fields. Instead, we define an authentic classification task grounded in observed BRFSS labels: predict multi-class diabetes status—no diabetes, prediabetes, or diabetes—among female respondents, while interpreting features and errors in light of the PCOS–diabetes literature.

The empirical substrate of the study is the cleaned BRFSS 2015 diabetes health indicators release commonly distributed as an open research dataset derived from CDC survey microdata. The raw file contains 253,680 adult records prior to cohort filtering, satisfying the requirement for a large, non-toy, labeled classification corpus. We restrict the analytic cohort to women, retain the three-class diabetes target, and engineer transparent metabolic composites (for example, cardiometabolic burden and lifestyle-risk summaries) from existing survey fields rather than from synthetic generators. This design preserves provenance, scale, and label authenticity while still centering a women’s-health research narrative that is clinically meaningful.

Methodologically, the project compares two complementary modeling paradigms under a shared preprocessing and evaluation protocol. On the traditional ML side, we train and tune multiple scikit-learn estimators—including logistic regression, random forests, gradient boosting, and a calibrated linear support vector machine—with attention to class imbalance, stratified splitting, and hyperparameter search. On the deep learning side, we implement multilayer perceptron architectures in TensorFlow/Keras using the Functional API and feed scaled tabular tensors through `tf.data` pipelines with shuffling, batching, and prefetching. Across both paradigms we emphasize metrics suited to imbalanced multi-class problems—macro-averaged precision, recall, and F1, one-versus-rest ROC–AUC, confusion matrices, and learning curves—rather than accuracy alone.

The contributions of this work are fourfold. First, we formulate an assignment-compliant, ethically transparent women’s diabetes classification problem that is clinically motivated by PCOS without misrepresenting dataset contents. Second, we provide a reproducible end-to-end pipeline spanning data validation, feature engineering, traditional ML, and deep learning. Third, we conduct controlled experiments that contrast model families under common splits and report both aggregate metrics and class-wise error structure, with particular attention to the minority prediabetes class. Fourth, we critically discuss dataset limitations—self-report bias, absence of laboratory glucose measures and PCOS diagnoses, and survey-feature ceilings on tabular DL gains—so that performance claims remain proportionate to the evidence.

The remainder of the report is organized as follows. Section II reviews related work on PCOS–diabetes epidemiology and computational approaches to diabetes risk prediction from survey and clinical data. Section III details the dataset, cohort definition, preprocessing, model architectures, training procedures, and evaluation design. Section IV presents experimental results for traditional and deep models. Section V discusses implications, failure modes, and limitations. Section VI concludes and outlines directions for future work, including linkage to clinical registries that actually record PCOS status.

# II. Literature Review

## A. PCOS, Insulin Resistance, and Type 2 Diabetes

Polycystic ovary syndrome (PCOS) is the most common endocrine disorder among reproductive-age women and is tightly coupled to metabolic dysfunction. Using gold-standard euglycaemic–hyperinsulinaemic clamp studies, Cassar *et al.* reported that insulin sensitivity is approximately 27% lower in women with PCOS than in controls, with higher body-mass index (BMI) further worsening insulin resistance [1]. Observational meta-analyses likewise estimate roughly a threefold elevation in type 2 diabetes (T2D) risk among women with PCOS, with obesity amplifying that risk [2]. Foundational physiologic work demonstrated profound peripheral insulin resistance in PCOS independent of obesity [3], and Legro *et al.* documented high rates of impaired glucose tolerance and undiagnosed diabetes in PCOS cohorts, motivating earlier glycemic assessment [4]. International evidence-based guidelines accordingly recommend assessing glycemic status at PCOS diagnosis and reassessing periodically, preferring oral glucose tolerance testing in higher-risk women [5].

These findings matter for machine learning because they identify a **women’s metabolic phenotype**—elevated BMI, hypertension, dyslipidemia, and inactivity—that is both clinically salient in PCOS and measurable in population surveys even when PCOS diagnoses are unavailable. The present project therefore treats PCOS–diabetes epidemiology as theoretical grounding and interpretive context, not as a claim that BRFSS contains PCOS labels.

## B. BRFSS and Computational Diabetes Risk Modeling

The U.S. Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) is a large, ongoing telephone survey of adult health behaviors and chronic conditions [6]. Xie *et al.* built multiple classifiers on 2014 BRFSS data for type 2 diabetes risk prediction, reporting AUCs from about 0.72 to 0.79 and noting that high overall accuracy can coexist with limited sensitivity—an imbalance-aware evaluation lesson directly relevant here [7]. Subsequent BRFSS studies continue to show strong tree-ensemble performance and emphasize class imbalance, self-report labels, and the limits of cross-sectional prediction [8], [9]. Cleaned BRFSS 2015 diabetes-indicator releases used widely in open ML practice provide a reproducible tabular benchmark with hundreds of thousands of rows [10].

A recurring gap in this literature is **sex-specific framing**: many notebooks pool sexes and optimize accuracy, under-discussing women’s endocrine context. Few open assignments simultaneously (i) restrict to women, (ii) keep multi-class glycemic labels including prediabetes, and (iii) refuse to fabricate PCOS labels while still engaging PCOS scholarship. That gap is the niche this project occupies.

## C. Classical Machine Learning for Tabular Health Data

Logistic regression remains a transparent multi-class baseline under standardized features. Ensemble methods—random forests [11] and gradient boosting machines [12]—are strong default learners for mixed discrete/continuous clinical tables because they capture nonlinear interactions with limited preprocessing. In BRFSS diabetes tasks, tree ensembles frequently lead accuracy/AUC tables while linear models remain useful for calibrated, interpretable baselines [7], [8]. Support vector machines are competitive after scaling but can be costly at full BRFSS scale; linear SVM variants are the practical compromise.

## D. Deep Learning on Tabular Data: Promise and Limits

Despite neural networks’ success in vision and language, tabular deep learning remains contested. Borisov *et al.* survey deep models for heterogeneous tables and find gradient-boosted trees still usually outperform neural nets on supervised tabular tasks [13]. Gorishniy *et al.* raise strong neural baselines (ResNet-like MLPs and Transformers) yet conclude there is still no universally superior family versus gradient-boosted decision trees [14]. These results justify implementing Keras MLPs with `tf.data` for the assignment’s deep-learning requirement while **predicting** that carefully tuned forests/boosting may match or beat MLPs on BRFSS indicators—exactly the comparative hypothesis tested in Section IV.

## E. Synthesis and Project Positioning

Across clinical and computational literatures, three tensions recur: (1) PCOS elevates diabetes risk through insulin resistance and obesity pathways [1]–[5]; (2) BRFSS enables large-scale modeling but lacks endocrine labels and laboratory confirmation [6]–[10]; (3) deep learning is not automatically better than classical ensembles on survey tables [13], [14]. This project addresses those tensions by defining an authentic women’s multi-class diabetes task, engineering metabolic composites aligned with PCOS pathways without inventing diagnoses, and comparing scikit-learn models to TensorFlow/Keras MLPs under imbalance-aware metrics with offline experiment logging.

# III. Methodology (IEEE draft)

## A. Problem Formulation

Let each respondent be represented by a feature vector \(\mathbf{x} \in \mathbb{R}^{d}\) derived from BRFSS health indicators and engineered metabolic composites. The target \(y \in \{0,1,2\}\) denotes self-reported diabetes status: \(0\) = no diabetes, \(1\) = prediabetes, and \(2\) = diabetes. The learning goal is to estimate a classifier \(f(\mathbf{x})\) that maps features to class labels (and, where available, class probabilities) for **female** respondents only. Performance is assessed with imbalance-aware metrics on a held-out test partition that is never used for model selection nested inside training folds.

## B. Dataset and Cohort Definition

We use the cleaned BRFSS 2015 diabetes health indicators file containing **253,680** rows and 21 core predictors plus the diabetes target prior to our cohort filter [5]. This raw cardinality satisfies the project’s minimum of 20,000 samples before preprocessing. Sex encoding in the cleaned release uses \(0\) for female and \(1\) for male. After retaining women, the analytic cohort comprises **141,974** rows—still far above the size threshold. The sex variable is used solely for cohort definition and is dropped thereafter to avoid trivial leakage of the filter itself.

Class prevalence in the women cohort remains heavily skewed toward no diabetes, with diabetes as a secondary class and prediabetes as a rare minority. That structure is treated as a first-class experimental factor rather than a nuisance to hide via accuracy-only reporting.

**Authenticity constraint.** No PCOS diagnosis field exists in this corpus. Feature engineering therefore uses only observed survey variables. Clinical discussion may reference PCOS pathways, but labels and claims remain diabetes-status classification in women.

## C. Preprocessing and Feature Engineering

We begin with an automated **data-quality audit** (`diagnose_data_quality`) that records missingness, dtypes, and unique-value counts for every column, plus BMI quantile and outlier flags. Although the cleaned BRFSS indicators release is largely complete, we still **drop rows with missing targets or required predictors** and **clip BMI** to a plausible survey range \([12, 60]\) to limit extreme coding artifacts while preserving ordinal/binary questionnaire fields. Tukey IQR fences for BMI are reported for transparency even when clinical clipping is the applied transform. All steps are serialized to `report/experiment_results/data_quality_audit.csv` and `cleaning_report.json` for replication.

Features are cast to numeric types for modeling. We construct three transparent composites:

1. **MetabolicBurden** — sum of high blood pressure, high cholesterol, and obesity indicator (\(\mathrm{BMI}\ge 30\)).
2. **LifestyleRisk** — combination of physical inactivity and scaled general-health score.
3. **BMI_x_HighBP** — interaction between BMI and hypertension.

These fields are motivated by cardiometabolic patterns frequently discussed in PCOS–diabetes research [1]–[5]; they are **not** surrogate PCOS labels. The final feature matrix uses the original indicators plus the three engineered columns (\(d=23\) in the reported runs). An **ablation** comparing logistic regression with and without engineered features is logged in the experiment table to justify retention empirically rather than by narrative alone.

We apply stratified splits: 70% train, 15% validation, 15% test (implemented as a 15% test holdout from the full cohort, then a validation fraction from the remainder), with `random_state=42`. Continuous scaling via `StandardScaler` is fit on the training partition only and applied to validation/test. Tree models can tolerate unscaled inputs, but a shared scaled representation keeps linear models and neural networks comparable; pipelines encapsulate scaling for scikit-learn estimators.

## D. Traditional Machine Learning Models

We train four algorithm families implemented in scikit-learn:

1. **Logistic regression** with balanced class weights and increased iteration limits.
2. **Random forest** with balanced subsample weights.
3. **Gradient boosting** (stage-wise additive trees).
4. **Linear SVM** with probability calibration (available in the notebook; optional in timed batch runs because of cost at full sample size).

Baselines use reasonable defaults, followed by `RandomizedSearchCV` on train+validation with macro-F1 scoring and three-fold stratified CV. Search spaces cover regularization strength (linear models), tree depth/estimators/leaf size (forests and boosting), and SVM margin parameter \(C\). The selected traditional model for deployment artifacts is the estimator with best held-out macro-F1; a compact random-forest configuration is retained for Drive delivery to avoid multi-hundred-megabyte weight files when depth/estimators explode.

## E. Deep Learning Models and `tf.data` Pipelines

Deep models are multilayer perceptrons built with the Keras Functional API. An input layer of dimension \(d\) feeds one or more `Dense` blocks with ReLU activations, \(\ell_2\) kernel regularization, batch normalization, and dropout. Softmax output produces three-class probabilities trained under sparse categorical cross-entropy.

We evaluate at least three capacity/regularization settings, for example:

- MLP‑64‑32 (lighter),
- MLP‑128‑64‑32 (medium),
- MLP‑256‑128‑64 (wider, stronger dropout, lower learning rate).

Minibatches are assembled with `tf.data.Dataset.from_tensor_slices`, optional shuffling (training only), `batch`, and `prefetch(AUTOTUNE)`. Class weights inversely proportional to training frequencies counteract majority dominance. Optimization uses Adam with ReduceLROnPlateau and EarlyStopping on validation loss (best weights restored).

## F. Evaluation Protocol

Primary selection metric: **macro-F1**. Secondary metrics: accuracy, macro precision/recall, and **macro one-versus-rest ROC–AUC**. We report confusion matrices for best traditional and best deep models, ROC curves per class, random-forest learning curves (macro-F1 versus training size), and neural training/validation loss–accuracy trajectories. Error analysis stratifies mistakes by true class and examines metabolic-feature means among errors to connect quantitative failures to clinical interpretability.

## G. Reproducibility, W&B Offline Logs, and Artifact Handling

Code is organized as importable modules under `src/` plus a documented Jupyter notebook and a batch runner (`scripts/run_experiments.py`). Dependencies are listed in `requirements.txt` (including `wandb`). Every experimental run is logged with **Weights & Biases in offline mode** (`WANDB_MODE=offline`), writing local run directories under `wandb/` and appending a row to `report/experiment_results/experiment_table.csv` with model name, hyperparameters, split sizes, metrics, and a short observation. TensorBoard-style epoch curves for MLPs are also captured via W&B step logs of train/val loss and accuracy. Raw CSVs and model binaries are gitignored; the Google Drive folder stores the zipped raw dataset and best model weights. Random seeds are fixed (`random_state=42`, Keras seed utility).

# IV. Results (IEEE draft)

Figures referenced below are generated in `figures/` by the experimental pipeline. Insert them in the IEEE manuscript as Figs. 1–6 with captions.

## A. Cohort Summary and Class Imbalance

The raw BRFSS-derived table contains 253,680 adults. Restricting to women yields 141,974 respondents. Fig. 1 (`class_balance_women.png`) shows the diabetes-status distribution in this cohort: the large majority report no diabetes, a smaller fraction report diabetes, and prediabetes is comparatively rare. This imbalance is the dominant statistical context for all subsequent metrics. Models that predict “no diabetes” almost everywhere can still achieve deceptively high accuracy; therefore we emphasize macro-averaged scores and class-wise confusion.

## B. Master Experiment Table and W&B Traceability

All runs—including feature ablations, classical baselines/tunings, and MLP architecture variants—are recorded in `report/experiment_results/experiment_table.csv` and mirrored by offline W&B runs under `wandb/`. Each row includes `run_id`, stage, approach, hyperparameters (JSON), stratified split sizes, accuracy, macro precision/recall/F1, OvR ROC–AUC, and a short observation explaining what changed relative to the previous stage. This section summarizes that table; another researcher can regenerate it via `python scripts/run_experiments.py` with `WANDB_MODE=offline`.

## C. Traditional Machine Learning Performance

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

# VI. Conclusion (IEEE draft)

This study developed a reproducible, assignment-compliant classification pipeline for **multi-class diabetes status among women** using a large open BRFSS-derived dataset (253,680 raw rows; 141,974 women after cohort filtering). The work is clinically motivated by the established association between PCOS and Type 2 diabetes risk, yet it remains authentic to the data: PCOS diagnoses are unavailable and are not synthesized. That authenticity is itself a contribution: it shows how to keep a women’s-health narrative without violating dataset provenance rules or inventing labels.

Across controlled experiments, a **tuned random forest** delivered the best traditional macro-F1 (approximately 0.447), while a compact **MLP (64–32)** was the strongest deep model (macro-F1 approximately 0.417) among tested architectures. A size-constrained forest used for Drive delivery preserved essentially the same macro-F1 (approximately 0.443) at a practical artifact size. Gradient boosting achieved higher accuracy and OvR AUC but weaker balanced recognition of minority classes. Confusion matrices, ROC curves, and learning curves consistently identified prediabetes as the principal failure mode under severe imbalance and self-report noise.

The comparative lesson is clear: for medium-width tabular survey features, carefully tuned traditional ensembles remain highly competitive with neural MLPs, even when the latter are trained with class weights and modern `tf.data` pipelines. Students and practitioners should therefore treat paradigm comparison as an empirical question, not a prestige contest. Method choices—stratified splits, class weights, macro-F1 selection, and explicit limitation statements—matter as much as architecture diagrams.

Future work should seek clinically linked datasets that record both glycemic outcomes and confirmed PCOS status, incorporate laboratory biomarkers, and extend fairness evaluation across intersecting social determinants—while preserving the same discipline about what labels do and do not exist. Until then, PCOS-informed diabetes classification in women on open BRFSS indicators remains a rigorous, honest, and instructive benchmark for contrasting traditional machine learning with deep learning.

# References (IEEE)

[1] S. Cassar *et al.*, “Insulin resistance in polycystic ovary syndrome: A systematic review and meta-analysis of euglycaemic–hyperinsulinaemic clamp studies,” *Hum. Reprod.*, vol. 31, no. 11, pp. 2619–2631, Nov. 2016, doi: 10.1093/humrep/dew243.

[2] “Risk of type 2 diabetes mellitus in polycystic ovary syndrome is associated with obesity: A meta-analysis of observational studies,” *Endocrine*, vol. 74, no. 2, pp. 245–253, 2021, doi: 10.1007/s12020-021-02801-2.

[3] A. Dunaif, K. R. Segal, W. Futterweit, and A. Dobrjansky, “Profound peripheral insulin resistance, independent of obesity, in polycystic ovary syndrome,” *Diabetes*, vol. 38, no. 9, pp. 1165–1174, 1989.

[4] R. S. Legro, A. R. Kunselman, W. C. Dodson, and A. Dunaif, “Prevalence and predictors of risk for type 2 diabetes mellitus and impaired glucose tolerance in polycystic ovary syndrome: A prospective, controlled study,” *J. Clin. Endocrinol. Metab.*, vol. 84, no. 1, pp. 165–169, 1999.

[5] H. J. Teede *et al.*, “Recommendations from the international evidence-based guideline for the assessment and management of polycystic ovary syndrome,” *Hum. Reprod.*, vol. 33, no. 9, pp. 1602–1618, 2018, doi: 10.1093/humrep/dey256.

[6] Centers for Disease Control and Prevention, “Behavioral Risk Factor Surveillance System (BRFSS),” Atlanta, GA, USA. [Online]. Available: https://www.cdc.gov/brfss

[7] Z. Xie, O. Nikolayeva, J. Luo, and D. Li, “Building risk prediction models for type 2 diabetes using machine learning techniques,” *Prev. Chronic Dis.*, vol. 16, E130, 2019, doi: 10.5888/pcd16.190109.

[8] M. M. Chowdhury, R. S. Ayon, and M. S. Hossain, “An investigation of machine learning algorithms and data augmentation techniques for diabetes diagnosis using class imbalanced BRFSS dataset,” *Healthcare Analytics*, vol. 5, Art. no. 100297, 2024, doi: 10.1016/j.health.2023.100297.

[9] “Development and evaluation of machine learning models for diabetes risk prediction using BRFSS 2023 data,” in *Proc. IEEE DASA*, 2025, doi: 10.1109/dasa68193.2025.11498954.

[10] A. Teboul, “Diabetes Health Indicators Dataset (BRFSS 2015),” Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

[11] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[12] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.

[13] V. Borisov *et al.*, “Deep neural networks and tabular data: A survey,” *IEEE Trans. Neural Netw. Learn. Syst.*, doi: 10.1109/TNNLS.2022.3225573; arXiv:2110.01889.

[14] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for tabular data,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021.
