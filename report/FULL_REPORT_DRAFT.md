---
IEEE-style manuscript draft (paste into Word/LaTeX; polish citations)
**Required header links — replace before submission**
- GitHub repository: `TODO`
- Google Drive folder: `TODO`
- Presentation video: `TODO`
---

# Introduction (IEEE draft)

**Working title:** PCOS-Informed Multi-Class Diabetes Classification in Women Using BRFSS Health Indicators: A Comparative Study of Traditional Machine Learning and Deep Learning

**Repository / deliverable links (replace before submission):**
- GitHub repository: `TODO`
- Google Drive folder: `TODO`
- Presentation video: `TODO`

---

Diabetes mellitus remains one of the most consequential non-communicable diseases worldwide, with rising prevalence, substantial healthcare costs, and well-documented disparities by sex, age, and metabolic risk profile. Among women, endocrine and metabolic conditions further shape diabetes risk across the life course. Polycystic ovary syndrome (PCOS) is the most common endocrine disorder of reproductive-age women and is strongly associated with insulin resistance, impaired glucose tolerance, and subsequent Type 2 diabetes. That clinical relationship motivates the present study: if metabolic risk accumulates along pathways that overlap with the PCOS phenotype, then carefully designed machine learning (ML) and deep learning (DL) models applied to large population health surveys may help characterize diabetes status in women and illuminate which routinely collected indicators carry signal.

At the same time, scholarly and pedagogical integrity require a clear separation between *motivation* and *measurement*. Most publicly available PCOS-specific clinical tables are small—often hundreds to a few thousand records—and therefore fail minimum sample-size expectations for rigorous comparative modeling. Large government survey systems such as the U.S. Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) contain hundreds of thousands of adult responses and include diabetes history alongside cardiometabolic and lifestyle indicators, yet they do not typically record a confirmed PCOS diagnosis. Accordingly, this project does **not** claim to detect PCOS, invent PCOS labels, or estimate PCOS-conditional diabetes risk from unavailable fields. Instead, we define an authentic classification task grounded in observed BRFSS labels: predict multi-class diabetes status—no diabetes, prediabetes, or diabetes—among female respondents, while interpreting features and errors in light of the PCOS–diabetes literature.

The empirical substrate of the study is the cleaned BRFSS 2015 diabetes health indicators release commonly distributed as an open research dataset derived from CDC survey microdata. The raw file contains 253,680 adult records prior to cohort filtering, satisfying the requirement for a large, non-toy, labeled classification corpus. We restrict the analytic cohort to women, retain the three-class diabetes target, and engineer transparent metabolic composites (for example, cardiometabolic burden and lifestyle-risk summaries) from existing survey fields rather than from synthetic generators. This design preserves provenance, scale, and label authenticity while still centering a women’s-health research narrative that is clinically meaningful.

Methodologically, the project compares two complementary modeling paradigms under a shared preprocessing and evaluation protocol. On the traditional ML side, we train and tune multiple scikit-learn estimators—including logistic regression, random forests, gradient boosting, and a calibrated linear support vector machine—with attention to class imbalance, stratified splitting, and hyperparameter search. On the deep learning side, we implement multilayer perceptron architectures in TensorFlow/Keras using the Functional API and feed scaled tabular tensors through `tf.data` pipelines with shuffling, batching, and prefetching. Across both paradigms we emphasize metrics suited to imbalanced multi-class problems—macro-averaged precision, recall, and F1, one-versus-rest ROC–AUC, confusion matrices, and learning curves—rather than accuracy alone.

The contributions of this work are fourfold. First, we formulate an assignment-compliant, ethically transparent women’s diabetes classification problem that is clinically motivated by PCOS without misrepresenting dataset contents. Second, we provide a reproducible end-to-end pipeline spanning data validation, feature engineering, traditional ML, and deep learning. Third, we conduct controlled experiments that contrast model families under common splits and report both aggregate metrics and class-wise error structure, with particular attention to the minority prediabetes class. Fourth, we critically discuss dataset limitations—self-report bias, absence of laboratory glucose measures and PCOS diagnoses, and survey-feature ceilings on tabular DL gains—so that performance claims remain proportionate to the evidence.

The remainder of the report is organized as follows. Section II reviews related work on PCOS–diabetes epidemiology and computational approaches to diabetes risk prediction from survey and clinical data. Section III details the dataset, cohort definition, preprocessing, model architectures, training procedures, and evaluation design. Section IV presents experimental results for traditional and deep models. Section V discusses implications, failure modes, and limitations. Section VI concludes and outlines directions for future work, including linkage to clinical registries that actually record PCOS status.

# II. Literature Review (IEEE draft)

## A. PCOS and Diabetes Risk in Women

Polycystic ovary syndrome (PCOS) is a heterogeneous endocrine disorder characterized by ovulatory dysfunction, hyperandrogenism, and/or polycystic ovarian morphology. Beyond reproductive morbidity, PCOS is tightly linked to metabolic disease. Multiple epidemiological syntheses report elevated rates of insulin resistance, impaired glucose tolerance, and Type 2 diabetes mellitus among women with PCOS relative to age- and BMI-comparable controls [1], [2]. Proposed mechanisms include androgen excess, visceral adiposity, chronic low-grade inflammation, and disrupted insulin signaling. Clinical guidance therefore often recommends earlier and more frequent glycemic screening in this population [3].

Importantly for computational work, the PCOS–diabetes association is not reducible to a single biomarker. Risk concentrates among women who also exhibit cardiometabolic features—elevated body mass index (BMI), hypertension, dyslipidemia, and low physical activity—that are routinely captured in population surveys even when PCOS itself is not. This creates a methodological opportunity: large public health datasets can support women’s diabetes classification models whose feature space overlaps pathways emphasized in PCOS research, provided authors do not over-claim PCOS detection from missing labels.

## B. Population Surveys and Diabetes Prediction

The CDC Behavioral Risk Factor Surveillance System (BRFSS) is a long-running telephone survey of U.S. adults that records self-reported chronic conditions, preventive behaviors, and sociodemographic factors [4]. Cleaned derivatives of BRFSS diabetes modules have become common benchmarks for tabular classification, including binary and three-class diabetes outcomes with features such as blood pressure, cholesterol, BMI, general health, and lifestyle indicators [5]. Strengths include scale (hundreds of thousands of rows), open access, and demographic breadth. Limitations are equally structural: self-report error, limited laboratory confirmation, coarse age bands, and absence of many clinical endocrinology variables (including PCOS diagnosis, androgen panels, and oral glucose tolerance results).

Prior ML studies on BRFSS-style diabetes indicators frequently report high overall accuracy that is driven by majority-class “no diabetes” predictions, with weaker minority-class performance for prediabetes [5], [6]. That pattern motivates evaluation designs centered on macro-averaged F1, precision/recall, and ROC–AUC rather than accuracy alone—especially in women’s health analyses where under-detection of intermediate risk states has screening implications.

## C. Traditional Machine Learning for Tabular Health Data

Logistic regression remains a transparent baseline for multi-class health outcomes and provides calibrated linear decision boundaries when features are standardized. Ensemble tree methods—random forests and gradient boosting—often dominate tabular biomedical prediction because they capture nonlinear interactions and mixed discrete/continuous encodings without heavy architecture search [7], [8]. Support vector machines can be competitive after scaling, though linear formulations are typically preferred at BRFSS scale for computational tractability. Across these families, class weighting, stratified resampling, and hyperparameter search are standard responses to imbalance.

## D. Deep Learning on Tabular Clinical/Survey Features

Deep neural networks have transformed imaging and sequence tasks, yet their advantage on medium-dimensional tabular survey data is less consistent. Multilayer perceptrons (MLPs) with regularization, batch normalization, and class-weighted losses can match strong tree baselines, but gains are often modest when inputs are sparse questionnaire encodings rather than rich sensor or EHR timelines [9], [10]. Modern training stacks (TensorFlow/Keras with `tf.data`) improve throughput and reproducibility for large row counts, while early stopping and learning-rate schedules reduce overfitting to majority classes. Comparative studies therefore treat DL not as an automatic upgrade but as a second paradigm whose value must be demonstrated under identical splits and metrics.

## E. Positioning of the Present Study

This project sits at the intersection of women’s metabolic health and reproducible ML pedagogy. Unlike PCOS ultrasound or small clinic tables, our corpus satisfies large-sample requirements via BRFSS-derived indicators. Unlike generic diabetes notebooks that ignore sex-specific framing, we restrict the cohort to women and interpret engineered metabolic features through the PCOS–diabetes literature. Unlike studies that conflate motivation with labels, we explicitly state that PCOS status is unavailable. The empirical contribution is a controlled comparison of scikit-learn ensembles/linear models against Keras MLPs for three-class diabetes status in women, accompanied by learning-curve and error analyses that foreground dataset ceilings.

## References (seed list — expand/verify in final bibliography)

[1] R. S. Legro et al., related PCOS metabolic reviews / Endocrine Society guidance (replace with exact citation).  
[2] Peer-reviewed meta-analyses on PCOS and Type 2 diabetes risk (insert DOI).  
[3] Clinical screening recommendations for glycemic testing in PCOS.  
[4] CDC BRFSS overview and 2015 data documentation.  
[5] A. Teboul (curator), Diabetes Health Indicators Dataset (BRFSS 2015 cleaned release), Kaggle.  
[6] Prior BRFSS diabetes classification studies emphasizing class imbalance.  
[7] L. Breiman, “Random Forests,” *Machine Learning*, 2001.  
[8] J. H. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine,” *Annals of Statistics*, 2001.  
[9] Surveys on deep learning for tabular data (e.g., comparative MLP vs. GBDT discussions).  
[10] TensorFlow/`tf.data` pipeline guidance for scalable input pipelines.

# III. Methodology (IEEE draft)

## A. Problem Formulation

Let each respondent be represented by a feature vector \(\mathbf{x} \in \mathbb{R}^{d}\) derived from BRFSS health indicators and engineered metabolic composites. The target \(y \in \{0,1,2\}\) denotes self-reported diabetes status: \(0\) = no diabetes, \(1\) = prediabetes, and \(2\) = diabetes. The learning goal is to estimate a classifier \(f(\mathbf{x})\) that maps features to class labels (and, where available, class probabilities) for **female** respondents only. Performance is assessed with imbalance-aware metrics on a held-out test partition that is never used for model selection nested inside training folds.

## B. Dataset and Cohort Definition

We use the cleaned BRFSS 2015 diabetes health indicators file containing **253,680** rows and 21 core predictors plus the diabetes target prior to our cohort filter [5]. This raw cardinality satisfies the project’s minimum of 20,000 samples before preprocessing. Sex encoding in the cleaned release uses \(0\) for female and \(1\) for male. After retaining women, the analytic cohort comprises **141,974** rows—still far above the size threshold. The sex variable is used solely for cohort definition and is dropped thereafter to avoid trivial leakage of the filter itself.

Class prevalence in the women cohort remains heavily skewed toward no diabetes, with diabetes as a secondary class and prediabetes as a rare minority. That structure is treated as a first-class experimental factor rather than a nuisance to hide via accuracy-only reporting.

**Authenticity constraint.** No PCOS diagnosis field exists in this corpus. Feature engineering therefore uses only observed survey variables. Clinical discussion may reference PCOS pathways, but labels and claims remain diabetes-status classification in women.

## C. Preprocessing and Feature Engineering

Missingness in the cleaned indicators release is already limited; we validate required columns at load time and abort if the schema drifts. Features are cast to numeric types for modeling. We construct three transparent composites:

1. **MetabolicBurden** — sum of high blood pressure, high cholesterol, and obesity indicator (\(\mathrm{BMI}\ge 30\)).
2. **LifestyleRisk** — combination of physical inactivity and scaled general-health score.
3. **BMI_x_HighBP** — interaction between BMI and hypertension.

These fields are motivated by cardiometabolic patterns frequently discussed in PCOS–diabetes research; they are **not** surrogate PCOS labels. The final feature matrix uses the original indicators plus the three engineered columns (\(d=23\) in the reported runs).

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

## G. Reproducibility and Artifact Handling

Code is organized as importable modules under `src/` plus a documented Jupyter notebook and a batch runner (`scripts/run_experiments.py`). Dependencies are pinned in spirit via `requirements.txt`. Raw CSVs and model binaries are gitignored; the assignment’s Google Drive folder stores (i) the zipped raw dataset, (ii) the best traditional model (`.joblib`), and (iii) the best Keras model (`.keras`) with its scaler. Random seeds are fixed where libraries allow (`random_state=42`, Keras seed utility).

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

# V. Discussion (IEEE draft)

## A. Interpretation of Model Behavior

The experimental pattern is coherent: **accuracy-favoring ensembles** (gradient boosting) and **balance-favoring models** (class-weighted forests, logistic regression, class-weighted MLPs) optimize different slices of the same impossibility triangle created by rare prediabetes labels. Declaring macro-F1 as the primary metric makes the tuned random forest the principled traditional winner, even though boosting looks stronger if one only cites accuracy. For a screening-oriented narrative in women’s health, missing intermediate risk states is costly; hence the metric choice is not cosmetic.

Deep learning’s failure to clearly surpass trees should not be read as a failure of the DL implementation. The input representation is a modest set of questionnaire-derived numerics. In that regime, inductive biases of axis-aligned ensembles are often well matched to the data-generating process, whereas MLPs must rediscover interactions through optimization under heavy class imbalance. The lighter MLP’s relative success among neural variants further suggests that regularization and calibration to minority recall matter more here than width.

## B. What the PCOS Framing Does—and Does Not—Buy

The PCOS-informed framing is valuable as **scientific motivation and interpretive lens**. It justifies studying women specifically, highlights metabolic composites, and connects errors to a real clinical discourse about insulin resistance. It does **not** authorize causal claims about PCOS, personalized PCOS risk scores, or diagnostic substitution for endocrine evaluation. Any reader conflating “PCOS-informed” with “PCOS-labeled” would misread both the dataset and the ethics of the work; the manuscript therefore repeats the authenticity statement across Introduction, Methodology, and Discussion.

## C. Dataset Limitations

Several limitations bound external validity:

1. **Self-report diabetes labels** may disagree with laboratory diagnosis; prediabetes is especially vulnerable to under-ascertainment.
2. **No PCOS field** prevents direct estimation of PCOS-conditional diabetes risk.
3. **Cross-sectional survey design** precludes incidence modeling and treatment-effect claims.
4. **Feature ceiling**: absent labs (HbA1c, fasting glucose), medications, and reproductive endocrine markers, model discrimination has a structural upper bound.
5. **Geographic/temporal scope**: BRFSS 2015 U.S. adults may not transfer to other countries, care systems, or post-pandemic behavioral baselines.

These limits explain why macro-F1 values in the mid‑0.4 range can still be experimentally meaningful: they reflect hard minority recognition under noisy labels, not merely weak modeling.

## D. Ethical and Practical Considerations

Population-scale classifiers can reinforce surveillance or stigmatization if deployed carelessly. The present artifacts are educational and analytical; they are not cleared clinical decision support. Fairness audits beyond sex filtering (race/ethnicity, income, education) remain future work. Transparency about missing PCOS labels is itself an ethical practice against overselling women’s-health AI.

## E. Implications for the ML vs. DL Requirement

Meeting the assignment’s dual-paradigm mandate does not require DL superiority. The pedagogically stronger result is comparative honesty: both paradigms were implemented with appropriate tooling (`Pipeline` + search; Keras Functional API + `tf.data`), evaluated under shared splits, and interpreted against dataset constraints. That is a more scholarly outcome than forcing a neural network to “win” through metric shopping.

## F. Practical Deployment Notes

Even as an academic deliverable, artifact hygiene matters. An unconstrained forest serialized to hundreds of megabytes is awkward for sharing and versioning; the compact forest retained nearly identical macro-F1 with an order-of-magnitude smaller file. Likewise, storing the DL scaler beside the `.keras` file is essential for correct inference. These engineering details rarely appear in accuracy tables, yet they determine whether collaborators can actually reproduce predictions from the Drive folder.

## G. Toward Stronger Future Designs

If a future study obtained linked EHR data with confirmed PCOS codes, oral glucose tolerance results, and medication history, the same dual-paradigm template could be reapplied with a sharper scientific question: short-term conversion from euglycemia to dysglycemia among women with PCOS, or differential calibration of risk scores by PCOS status. Until such labels exist in an open, sufficiently large table, authentic projects should prefer honest women’s diabetes classification over synthetic PCOS fabrication.

# VI. Conclusion (IEEE draft)

This study developed a reproducible, assignment-compliant classification pipeline for **multi-class diabetes status among women** using a large open BRFSS-derived dataset (253,680 raw rows; 141,974 women after cohort filtering). The work is clinically motivated by the established association between PCOS and Type 2 diabetes risk, yet it remains authentic to the data: PCOS diagnoses are unavailable and are not synthesized. That authenticity is itself a contribution: it shows how to keep a women’s-health narrative without violating dataset provenance rules or inventing labels.

Across controlled experiments, a **tuned random forest** delivered the best traditional macro-F1 (approximately 0.447), while a compact **MLP (64–32)** was the strongest deep model (macro-F1 approximately 0.417) among tested architectures. A size-constrained forest used for Drive delivery preserved essentially the same macro-F1 (approximately 0.443) at a practical artifact size. Gradient boosting achieved higher accuracy and OvR AUC but weaker balanced recognition of minority classes. Confusion matrices, ROC curves, and learning curves consistently identified prediabetes as the principal failure mode under severe imbalance and self-report noise.

The comparative lesson is clear: for medium-width tabular survey features, carefully tuned traditional ensembles remain highly competitive with neural MLPs, even when the latter are trained with class weights and modern `tf.data` pipelines. Students and practitioners should therefore treat paradigm comparison as an empirical question, not a prestige contest. Method choices—stratified splits, class weights, macro-F1 selection, and explicit limitation statements—matter as much as architecture diagrams.

Future work should seek clinically linked datasets that record both glycemic outcomes and confirmed PCOS status, incorporate laboratory biomarkers, and extend fairness evaluation across intersecting social determinants—while preserving the same discipline about what labels do and do not exist. Until then, PCOS-informed diabetes classification in women on open BRFSS indicators remains a rigorous, honest, and instructive benchmark for contrasting traditional machine learning with deep learning.

## Appendix A — Compact deployment forest (Drive artifact)

- Params: n_estimators=100, max_depth=14, min_samples_leaf=2, max_features=sqrt
- Test accuracy / macro-F1 / OvR-AUC: 0.759 / 0.443 / 0.761
- Artifact ~64 MB `.joblib` (Drive traditional-ML deliverable)

## Appendix B — Figure file map

| Figure | File |
|--------|------|
| Class balance | figures/class_balance_women.png |
| CM traditional | figures/cm_best_traditional.png |
| ROC traditional | figures/roc_best_traditional.png |
| Learning curve | figures/learning_curve_rf.png |
| CM / ROC deep | figures/cm_best_dl.png , figures/roc_best_dl.png |
| DL curves | figures/dl_training_curves.png |

## Appendix C — Drive package contents

Upload `deliverables/google_drive/ML_Summative_Drive_Package.zip` (or the two inner zips) to a folder shared as “Anyone with the link can view”.
