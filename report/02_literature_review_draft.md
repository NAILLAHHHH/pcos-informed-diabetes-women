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
