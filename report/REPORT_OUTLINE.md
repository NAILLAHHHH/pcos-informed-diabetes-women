# IEEE Report Outline (draft scaffold)

Replace `TODO` links on the title page before submission.

## Title page / header (required links)

- GitHub repository: `TODO`
- Google Drive folder: `TODO`
- Presentation video: `TODO`

## Working title

PCOS-Informed Multi-Class Diabetes Classification in Women Using BRFSS Health Indicators: A Comparative Study of Traditional Machine Learning and Deep Learning

## Authenticity paragraph (use early in Introduction)

This study addresses diabetes status classification among women using open CDC BRFSS-derived survey data. The clinical motivation is the well-documented association between polycystic ovary syndrome (PCOS) and Type 2 diabetes risk. The dataset does **not** contain confirmed PCOS diagnoses; therefore models are not claimed to detect PCOS. Features overlapping the metabolic phenotype commonly discussed in PCOS research (e.g., BMI, blood pressure, cholesterol, physical activity) are emphasized in feature engineering and discussion.

## Suggested section targets (for 3,500–5,000 words excl. refs)

1. **Introduction** (~600–800) — problem, motivation, contributions, authenticity statement
2. **Literature Review** (~700–900) — PCOS–diabetes epidemiology; ML/DL for diabetes risk; survey-data limitations
3. **Methodology** (~900–1,100) — data, cohort definition, preprocessing, ML models, DL architecture, `tf.data`, tuning, metrics
4. **Results** (~700–900) — tables + figures (learning curves, confusion matrices, ROC); ML vs DL
5. **Discussion** (~500–700) — errors, imbalance, no PCOS label, generalizability, ethics
6. **Conclusion** (~250–400) — takeaways + future work (e.g., linked clinical PCOS registries)

## Figures to integrate from `figures/`

- Class balance (women cohort)
- Confusion matrices (best ML, best DL)
- ROC OvR curves
- Learning curves
- Optional: training history (loss/accuracy) for the MLP
