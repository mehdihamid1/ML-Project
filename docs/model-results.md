# Model comparison

Mode: full experiment.
CV folds: 10. Dataset SHA-256: `d13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242`.

| Model | CV AUC mean ± std | CV accuracy mean ± std | Mean fit seconds |
| --- | --- | --- | --- |
| LightGBM | 0.998219 ± 0.000558 | 0.985741 ± 0.002176 | 7.681 |
| XGBoost | 0.997909 ± 0.000608 | 0.983926 ± 0.002242 | 8.795 |
| Random Forest | 0.997520 ± 0.000663 | 0.982370 ± 0.002802 | 11.011 |
| CatBoost | 0.996644 ± 0.000905 | 0.981131 ± 0.002118 | 10.288 |
| PyTorch MLP | 0.995044 ± 0.000818 | 0.972518 ± 0.001943 | 10.710 |
| Logistic Regression | 0.988586 ± 0.001440 | 0.957683 ± 0.004278 | 11.173 |
| Decision Tree | 0.980194 ± 0.003112 | 0.975658 ± 0.003284 | 8.535 |

Selected model: **LightGBM**.
Hold-out AUC: 0.997892; accuracy: 0.984330.
Hold-out samples: 8679. Threshold: 0.5.

Confusion matrix: rows = true class; columns = predicted class.

| True class | Predicted goodware (0) | Predicted malware (1) |
| --- | --- | --- |
| Goodware (0) | 4159 | 61 |
| Malware (1) | 75 | 4384 |
