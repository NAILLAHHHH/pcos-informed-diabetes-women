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
