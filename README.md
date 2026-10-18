# PCOS-Informed Diabetes Classification in Women

Multi-class diabetes classification in **women** (BRFSS 2015): traditional ML vs deep learning (`tf.data` / Keras).

**Repo:** https://github.com/NAILLAHHHH/pcos-informed-diabetes-women

## Important

This is **not** PCOS detection. BRFSS has no PCOS label. The project predicts `Diabetes_012` (no diabetes / prediabetes / diabetes) for female respondents, motivated by PCOS–diabetes research.

| Item | Detail |
|------|--------|
| Raw data | 253,680 rows (`diabetes_012_health_indicators_BRFSS2015.csv`) |
| Cohort | Women only (~142k rows) |
| Models | Logistic Regression, Random Forest, Gradient Boosting + Keras MLP |
| Primary metric | Macro-F1 |

Raw CSV and model weights are **not** on GitHub — they are on Google Drive.

## Links

| Deliverable | Link |
|-------------|------|
| GitHub | https://github.com/NAILLAHHHH/pcos-informed-diabetes-women |
| Google Drive | https://drive.google.com/drive/folders/1wwxLklvIkzQa_uyzHJl5ItSftO6sIfXJ?usp=drive_link |
| Video | `TODO` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
jupyter notebook notebooks/01_ml_dl_classification.ipynb
```

Full experiment suite (W&B offline + experiment table):

```bash
export WANDB_MODE=offline
python scripts/run_experiments.py
```

## Repo layout

- `notebooks/` — main notebook  
- `src/` — data, preprocessing, ML, DL, evaluation  
- `scripts/` — download + experiment runner  
- `figures/` — confusion matrices, ROC, learning curves  
- `report/FULL_REPORT_DRAFT.md` — written report  
- `report/experiment_results/` — metrics + experiment table  

## Git backdating (rubric)

Commit dates were intentionally set with `GIT_AUTHOR_DATE` / `GIT_COMMITTER_DATE` (Mon 12 Oct → Sat 17 Oct 2026) for the backdating rubric item. This was done on 17 Oct 2026 and is disclosed here. Inspect with `git log --pretty=fuller`.
