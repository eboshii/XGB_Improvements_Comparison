# XGB_Improvements_Comparison

Controlled comparison of extension patterns on top of XGBoost using Numerai `v6.0` (strict chronological split with 16-week purge gaps and a 267-date out-of-sample holdout).

## Scripts
- `numerai_xgb_val.py` — initial single-model baseline (`small` features)
- `numerai_xgb_benchmark.py` — controlled comparison (`small` features): early-stopped control, interaction constraints, `XGBRanker`, adversarial validation weighting, era boosting, multi-target rank blending, post-hoc neutralization, and ACI position sizing
- `numerai_xgb_all_quantized.py` — streaming `QuantileDMatrix` (`DataIter`) on all `1,040` features: control, deep-and-slow regularization, era boosting, 23-factor neutralization, and ACI sizing
