# PCOS-Informed Diabetes Classification in Women

Multi-class diabetes risk classification among **women**, using open CDC BRFSS-derived health indicator data. The project compares **traditional machine learning** (scikit-learn) and **deep learning** (TensorFlow/Keras + `tf.data`).

**GitHub:** https://github.com/NAILLAHHHH/pcos-informed-diabetes-women

## Authenticity statement (important)

This project is **not** a PCOS-labeled prediction task. Public PCOS tabular datasets are typically far below the assignment’s 20,000-row threshold.

Instead:

- **Clinical motivation:** Women with polycystic ovary syndrome (PCOS) have elevated risk of insulin resistance and Type 2 diabetes. That literature motivates studying diabetes status in women and metabolic risk features that overlap the PCOS phenotype (e.g., BMI, hypertension, cholesterol, physical activity, age).
- **Target label:** Diabetes status from BRFSS (`Diabetes_012`: no diabetes / prediabetes / diabetes).
- **Population:** Female respondents only (`Sex == 0` in the cleaned BRFSS 2015 indicators file).
- **Limitation (stated clearly in the report):** BRFSS does not include a confirmed PCOS diagnosis field. Models estimate diabetes class in women; they do **not** diagnose PCOS.

## Dataset

| Item | Detail |
|------|--------|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, cleaned diabetes health indicators release |
| Common open mirror | [Diabetes Health Indicators Dataset (Kaggle)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| Raw file used | `diabetes_012_health_indicators_BRFSS2015.csv` |
| Raw size | **253,680** rows (before filtering/preprocessing) |
| After female filter | ~140k+ rows (still ≫ 20,000) |
| Target | `Diabetes_012` — 0 = no diabetes, 1 = prediabetes, 2 = diabetes |
| License / provenance | CDC public health survey data; cite CDC BRFSS and the cleaned release you download |

**Do not commit raw CSV/ZIP files to GitHub.** Keep them in `data/raw/` locally and in the Google Drive deliverable folder.

### Download

```bash
# Option A — Kaggle Hub (requires Kaggle API credentials)
python scripts/download_data.py

# Option B — Manual
# 1. Download the Kaggle or CDC-derived CSV
# 2. Place diabetes_012_health_indicators_BRFSS2015.csv in data/raw/
```

## Project structure

```text
pcos-informed-diabetes-women/
├── notebooks/
│   └── 01_ml_dl_classification.ipynb   # Main experimental notebook
├── src/                                # Reusable pipeline modules
├── scripts/download_data.py
├── data/raw/                           # Raw data (gitignored)
├── data/processed/                     # Intermediate splits (gitignored)
├── models/                             # Saved .joblib / .keras (gitignored)
├── figures/                            # Plots for the report
├── report/                             # IEEE-style report draft materials
├── requirements.txt
└── README.md
```

## Methods (high level)

1. **EDA & filtering** — female subset; missingness/outlier audit; class imbalance inspection.
2. **Feature engineering** — metabolic risk composites (+ ablation vs baseline features).
3. **Traditional ML** — Logistic Regression, Random Forest, Gradient Boosting; hyperparameter search; learning curves.
4. **Deep learning** — Keras Functional MLP; `tf.data`; early stopping / class weights; architecture search.
5. **Evaluation** — macro-F1, precision/recall, confusion matrices, ROC-AUC (OvR), learning curves, error analysis.
6. **Experiment tracking** — W&B **offline** logs + `report/experiment_results/experiment_table.csv`.

```bash
export WANDB_MODE=offline
python scripts/run_experiments.py
```

## Deliverables checklist

- [x] Public/shared GitHub repo (this project) + clear README
- [ ] Google Drive folder (Anyone with the link): zipped raw dataset + best traditional ML model + best DL weights
- [ ] IEEE-style report (3,500–5,000 words) with working links on page 1
- [ ] 5–10 minute presentation video

### Links

| Deliverable | Link |
|-------------|------|
| GitHub repository | https://github.com/NAILLAHHHH/pcos-informed-diabetes-women |
| Google Drive folder | `TODO` |
| Presentation video | `TODO` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_ml_dl_classification.ipynb
```

Colab: upload/clone the repo, install `requirements.txt`, mount Drive or upload `data/raw/` CSV.

## Git history and intentional backdating (rubric note)

This repository’s commit **author/committer timestamps were intentionally backdated** as part of the Machine Learning summative rubric item on demonstrating git backdating. Dates were set with `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` (environment variables) when creating the commits—not by changing the system clock permanently.

| Intended timeline (Africa/Johannesburg, UTC+2) | Focus of commit |
|-----------------------------------------------|-----------------|
| Monday 2026-10-12 | Project scaffold, config, ignore rules |
| Tuesday 2026-10-13 | Data loading, preprocessing, download script |
| Wednesday 2026-10-14 | Traditional ML + evaluation utilities |
| Thursday 2026-10-15 | Deep learning (`tf.data`/Keras) + notebook |
| Friday 2026-10-16 | Experiment figures + report section drafts |
| Saturday 2026-10-17 | Full report assembly, README links, backdating disclosure |

**When the backdating was performed:** Saturday, 17 October 2026 (same day as the initial push), after the codebase and experiments already existed locally. The git graph is therefore a **documented, pedagogical reconstruction of a weekly workflow**, not a claim that each file was first written on those calendar days in isolation.

To inspect dates locally:

```bash
git log --pretty=fuller
```

## Citation notes

When writing the report, cite:

1. CDC BRFSS (survey year and data documentation URL).
2. The cleaned diabetes indicators release you actually used (e.g., Teboul BRFSS 2015 diabetes health indicators on Kaggle), plus any peer-reviewed PCOS–diabetes epidemiology papers in the literature review.

## Author

Machine Learning module summative project.
