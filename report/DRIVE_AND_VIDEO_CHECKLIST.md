# Google Drive & Video Checklist

## Google Drive folder (Anyone with the link can view)

Create one folder, e.g. `ML-Summative-PCOS-Informed-Diabetes`, and upload:

1. **Raw dataset (zipped)**  
   - Zip `diabetes_012_health_indicators_BRFSS2015.csv` (full raw file, not only the women subset).
2. **Best traditional ML model**  
   - `best_traditional_ml.joblib` from `models/` after running the notebook.
3. **Best deep learning model**  
   - `best_deep_learning.keras` (and optionally `dl_standard_scaler.joblib`).
4. Optional: exported figures used in the report.

Paste the share link into the report header and README.

## Presentation video (5–10 minutes)

Suggested arc:

1. Problem & authenticity (women’s diabetes; PCOS as motivation, not a label) — ~1 min  
2. Dataset (BRFSS origin, size, female filter, imbalance) — ~1.5 min  
3. Methodology (features, ML suite, Keras/`tf.data`) — ~2 min  
4. Experiments & results (tables + confusion/ROC/learning curves) — ~3 min  
5. Limitations & takeaways — ~1–2 min  

Host on Drive / YouTube (unlisted) and put the link on page 1 of the report.
