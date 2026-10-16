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
