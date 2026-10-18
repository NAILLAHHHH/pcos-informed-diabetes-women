# Demo Video Script (5–8 minutes, ~5 rubric points)

**Goal:** Clear problem → data → methods → experiments → takeaways.  
**Visuals:** Share screen with GitHub README, notebook, `experiment_table.csv`, figures, and report.

## 0:00–0:45 — Hook & authenticity
“This project classifies diabetes status in **women** using CDC BRFSS data. It’s **PCOS-informed**—motivated by the link between PCOS and type 2 diabetes—but BRFSS has **no PCOS label**, so we do not pretend to detect PCOS.”

Show README authenticity statement.

## 0:45–2:00 — Problem & dataset
- Multi-class: no diabetes / prediabetes / diabetes  
- Raw rows: 253,680; women cohort: ~142k  
- Why BRFSS: open, large, relevant metabolic features  
- Class imbalance chart (`class_balance_women.png`)

## 2:00–3:30 — Methodology
- Preprocessing: missingness audit, BMI clipping, stratified 70/15/15, standardization  
- Feature engineering: MetabolicBurden, LifestyleRisk, BMI×HighBP  
- Traditional ML: logreg, RF, gradient boosting + RandomizedSearchCV  
- Deep learning: Keras Functional MLP + `tf.data`, class weights, early stopping  
- W&B **offline** logs + experiment table

Show `report/experiment_results/experiment_table.csv` and a `wandb/` run folder.

## 3:30–5:30 — Results & error analysis
- Table: RF tuned best macro-F1; boosting best accuracy/AUC; MLP 64–32 best DL  
- Confusion matrices + ROC  
- Learning curve / DL curves → mild overfitting, prediabetes hardest  
- Trade-off: accuracy vs balanced recognition

## 5:30–6:30 — Takeaways
- Trees still competitive on tabular survey data  
- Metric choice changes the ‘winner’  
- Dataset limits: self-report, no PCOS field  
- Links: GitHub / Drive / report

## Recording tips
- 1080p, clear mic, no music  
- Unlisted YouTube or Drive video link on report page 1  
- Keep under 10 minutes
