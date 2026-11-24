"""Controlled benchmark of XGBoost fixes and meta-techniques on Numerai v6.0.

All experiments share the exact same chronological split with 16-week purge gaps:
  Train (2003..2018) -> Purge (16w) -> Early-Stop Val (100w) -> Purge (16w) -> Holdout (~267w)
"""

import gc
import json
import time
from pathlib import Path

import numerapi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("/content/data/v6.0")
FEATURE_SET = "small"  # 101 features (fits 4.33M train rows + holdout comfortably in 12.7 GB Colab RAM)
PURGE_DATES = 16
EARLY_STOP_DATES = 100
TARGET_COL = "target"
MULTI_TARGETS = [
    "target",
    "target_factor_neutral_20",
    "target_camille_20",
    "target_sydney_20",
    "target_irina_60",
]
RANDOM_STATE = 42

BASE_XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=100,
    reg_lambda=10.0,
    subsample=0.8,
    colsample_bytree=0.1,
    tree_method="hist",
    device="cuda",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=RANDOM_STATE,
)


def ensure_data() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    napi = numerapi.NumerAPI()
    paths: dict[str, Path] = {}
    for fname in ["features.json", "train.parquet", "validation.parquet"]:
        dest = DATA_DIR / fname
        if not dest.exists():
            print(f"Downloading v6.0/{fname}...")
            napi.download_dataset(f"v6.0/{fname}", dest_path=str(dest))
        paths[fname.split(".")[0]] = dest
    return paths


def load_data(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    with paths["features"].open("r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["feature_sets"][FEATURE_SET]
    neutralize_features = [c for c in meta["feature_sets"].get("signals", []) if c in features]

    train_schema = set(pq.ParquetFile(paths["train"]).schema.names)
    val_schema = set(pq.ParquetFile(paths["validation"]).schema.names)
    targets = [t for t in MULTI_TARGETS if t in train_schema and t in val_schema]
    cols = ["date"] + targets + [c for c in features if c in train_schema]

    print(f"Loading train.parquet ({len(features)} features, {len(targets)} targets)...")
    train_df = pd.read_parquet(paths["train"], columns=cols)
    train_df = train_df.dropna(subset=[TARGET_COL]).sort_values("date").reset_index(drop=True)

    print("Loading validation.parquet...")
    val_cols = cols + (["data_type"] if "data_type" in val_schema else [])
    val_full = pd.read_parquet(paths["validation"], columns=val_cols)
    if "data_type" in val_full.columns:
        val_full = val_full[val_full["data_type"] == "validation"].drop(columns=["data_type"])
    val_full = val_full.dropna(subset=[TARGET_COL]).sort_values("date").reset_index(drop=True)

    # Cast non-numeric features (e.g. feature_country) to shared categorical dtype
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]
    for col in cat_cols:
        dtype = pd.CategoricalDtype(categories=train_df[col].dropna().unique())
        train_df[col] = train_df[col].astype(dtype)
        val_full[col] = val_full[col].astype(dtype)

    # Strict chronological split with purge gaps
    val_dates = sorted(val_full["date"].unique())
    es_start = PURGE_DATES
    es_end = PURGE_DATES + EARLY_STOP_DATES
    holdout_start = es_end + PURGE_DATES

    es_dates = set(val_dates[es_start:es_end])
    holdout_dates = set(val_dates[holdout_start:])

    es_df = val_full[val_full["date"].isin(es_dates)].reset_index(drop=True)
    holdout_df = val_full[val_full["date"].isin(holdout_dates)].reset_index(drop=True)
    del val_full
    gc.collect()

    print(
        f"Split summary:\n"
        f"  Train         : {len(train_df):,} rows, {train_df['date'].nunique()} dates "
        f"({train_df['date'].min()} .. {train_df['date'].max()})\n"
        f"  Purge 1       : {PURGE_DATES} dates dropped\n"
        f"  Early-Stop Val: {len(es_df):,} rows, {es_df['date'].nunique()} dates "
        f"({es_df['date'].min()} .. {es_df['date'].max()})\n"
        f"  Purge 2       : {PURGE_DATES} dates dropped\n"
        f"  Holdout (OOS) : {len(holdout_df):,} rows, {holdout_df['date'].nunique()} dates "
        f"({holdout_df['date'].min()} .. {holdout_df['date'].max()})"
    )
    return train_df, es_df, holdout_df, features, neutralize_features


def build_interaction_constraints(features: list[str]) -> list[list[str]]:
    """Group features by Numerai family so trees only interact within coherent groups."""
    groups: dict[str, list[str]] = {}
    for col in features:
        parts = col.split("_")
        family = parts[1] if len(parts) > 2 and parts[0] == "feature" else "factors"
        groups.setdefault(family, []).append(col)
    return [cols for cols in groups.values() if cols]


def neutralize_predictions(
    df: pd.DataFrame,
    pred: np.ndarray,
    neutralize_cols: list[str],
    proportion: float = 0.5,
) -> np.ndarray:
    """Per-date linear neutralization of predictions against factor/signal features."""
    if not neutralize_cols or proportion <= 0.0:
        return pred

    num_cols = [c for c in neutralize_cols if pd.api.types.is_numeric_dtype(df[c])]
    out = np.empty_like(pred, dtype=np.float64)
    tmp = pd.DataFrame({"date": df["date"].values, "pred": pred}, index=df.index)

    for _, idx in tmp.groupby("date", sort=False).groups.items():
        y = tmp.loc[idx, "pred"].to_numpy(dtype=np.float64)
        X = df.loc[idx, num_cols].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        X = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        exposures = X @ coef
        out[df.index.get_indexer(idx)] = y - proportion * exposures
    return out


def per_date_rank(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Convert predictions to per-date percentile ranks in [0, 1]."""
    s = pd.Series(pred, index=df.index)
    return s.groupby(df["date"], sort=False).rank(pct=True).to_numpy(dtype=np.float64)


def evaluate_holdout(name: str, holdout_df: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """Compute per-date Pearson correlation, Sharpe, hit rate, max drawdown, and RMSE."""
    eval_df = pd.DataFrame(
        {
            "date": holdout_df["date"].values,
            "target": holdout_df[TARGET_COL].values,
            "pred": pred,
        }
    )
    era_corrs = eval_df.groupby("date", sort=True).apply(
        lambda g: g["pred"].corr(g["target"]),
        include_groups=False,
    ).dropna()

    mean_corr = float(era_corrs.mean())
    std_corr = float(era_corrs.std(ddof=1))
    sharpe = mean_corr / std_corr if std_corr > 0 else 0.0
    hit_rate = float((era_corrs > 0).mean())

    cum = era_corrs.cumsum()
    running_max = cum.cummax()
    max_dd = float((cum - running_max).min())

    # Standardize pred before RMSE if rank-based/unscaled so RMSE reflects calibration where applicable
    rmse = float(np.sqrt(np.mean((eval_df["target"] - eval_df["pred"]) ** 2)))
    return {
        "Experiment": name,
        "MeanCorr": mean_corr,
        "StdCorr": std_corr,
        "Sharpe": sharpe,
        "HitRate": hit_rate,
        "MaxDD": max_dd,
        "RMSE": rmse,
    }


def run_aci_sizing(
    holdout_df: pd.DataFrame,
    pred: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.05,
) -> dict[str, float]:
    """Adaptive Conformal Inference (ACI) dynamic gross exposure sizing across holdout dates."""
    eval_df = pd.DataFrame(
        {
            "date": holdout_df["date"].values,
            "target": holdout_df[TARGET_COL].values,
            "pred": pred,
        }
    )

    alpha_t = alpha
    q_history: list[float] = []
    unscaled_rets: list[float] = []
    aci_weights: list[float] = []

    for _, g in eval_df.groupby("date", sort=True):
        p = g["pred"].to_numpy(dtype=np.float64)
        y = g["target"].to_numpy(dtype=np.float64)
        # Cross-sectional z-scored long-short portfolio return (equals correlation)
        p_z = (p - p.mean()) / (p.std() + 1e-8)
        y_c = y - y.mean()
        ret = float(np.mean(p_z * y_c))
        unscaled_rets.append(ret)

        # Residual scale for this date
        resid = np.abs(y - p)
        q_t = float(np.quantile(resid, np.clip(1.0 - alpha_t, 0.05, 0.99)))
        if not q_history:
            w_t = 1.0
        else:
            baseline_q = float(np.median(q_history[-20:]))
            w_t = float(np.clip(baseline_q / (q_t + 1e-8), 0.25, 1.5))
        aci_weights.append(w_t)
        q_history.append(q_t)

        # ACI online miscoverage update
        err_t = float(np.mean(resid > q_t))
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err_t), 0.01, 0.5))

    unscaled = np.array(unscaled_rets)
    weights = np.array(aci_weights)
    weights = weights / weights.mean()  # normalize average gross leverage to 1.0
    scaled = unscaled * weights

    def port_stats(r: np.ndarray) -> tuple[float, float, float]:
        mean_r = float(r.mean())
        std_r = float(r.std(ddof=1))
        sharpe = mean_r / std_r if std_r > 0 else 0.0
        cum = np.cumsum(r)
        mdd = float((cum - np.maximum.accumulate(cum)).min())
        return mean_r, sharpe, mdd

    u_mean, u_sharpe, u_mdd = port_stats(unscaled)
    s_mean, s_sharpe, s_mdd = port_stats(scaled)
    return {
        "unscaled_mean": u_mean,
        "unscaled_sharpe": u_sharpe,
        "unscaled_mdd": u_mdd,
        "aci_mean": s_mean,
        "aci_sharpe": s_sharpe,
        "aci_mdd": s_mdd,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }


def main() -> None:
    t0 = time.time()
    paths = ensure_data()
    train_df, es_df, holdout_df, features, neut_cols = load_data(paths)

    X_train, y_train = train_df[features], train_df[TARGET_COL]
    X_es, y_es = es_df[features], es_df[TARGET_COL]
    X_holdout = holdout_df[features]

    results: list[dict[str, float]] = []

    # 1. Fixed Control Baseline
    print("\n[1/7] Training Fixed Control XGBRegressor...")
    ctrl_model = xgb.XGBRegressor(**BASE_XGB_PARAMS)
    ctrl_model.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=250)
    ctrl_pred = ctrl_model.predict(X_holdout)
    results.append(evaluate_holdout("1. Control XGB (small, ES)", holdout_df, ctrl_pred))
    print(f"  Best iteration: {ctrl_model.best_iteration}")

    # 2. Feature-Fraction & Interaction Constraints
    print("\n[2/7] Training Interaction-Constrained XGBRegressor...")
    constraints = build_interaction_constraints(features)
    inter_params = {
        **BASE_XGB_PARAMS,
        "colsample_bynode": 0.5,
        "interaction_constraints": constraints,
    }
    inter_model = xgb.XGBRegressor(**inter_params)
    inter_model.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=250)
    inter_pred = inter_model.predict(X_holdout)
    results.append(evaluate_holdout("2. Interaction Constraints", holdout_df, inter_pred))
    del inter_model
    gc.collect()

    # 3. Pairwise Ranking Loss (XGBRanker)
    print("\n[3/7] Training XGBRanker (rank:pairwise, qid=date)...")
    qid_train = train_df["date"].astype("category").cat.codes.to_numpy()
    qid_es = es_df["date"].astype("category").cat.codes.to_numpy()
    y_train_rank = (y_train * 4).round().astype(int)
    y_es_rank = (y_es * 4).round().astype(int)

    rank_params = {k: v for k, v in BASE_XGB_PARAMS.items()}
    rank_params["objective"] = "rank:pairwise"
    rank_model = xgb.XGBRanker(**rank_params)
    rank_model.fit(
        X_train,
        y_train_rank,
        qid=qid_train,
        eval_set=[(X_es, y_es_rank)],
        eval_qid=[qid_es],
        verbose=250,
    )
    rank_pred = rank_model.predict(X_holdout)
    results.append(evaluate_holdout("3. XGBRanker (pairwise)", holdout_df, rank_pred))
    del rank_model
    gc.collect()

    # 4. Adversarial Validation + Sample Weighting
    print("\n[4/7] Running Adversarial Validation (Train vs Early-Stop Val)...")
    n_adv = min(400_000, len(X_train), len(X_es))
    adv_tr = X_train.sample(n=n_adv, random_state=RANDOM_STATE)
    adv_va = X_es.sample(n=n_adv, random_state=RANDOM_STATE)
    X_adv = pd.concat([adv_tr, adv_va], axis=0, ignore_index=True)
    y_adv = np.concatenate([np.zeros(n_adv), np.ones(n_adv)])

    adv_clf = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.3,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        random_state=RANDOM_STATE,
    )
    adv_clf.fit(X_adv, y_adv, verbose=False)
    adv_auc = roc_auc_score(y_adv, adv_clf.predict_proba(X_adv)[:, 1])
    p_val = np.clip(adv_clf.predict_proba(X_train)[:, 1], 0.05, 0.95)
    adv_weights = np.clip(p_val / (1.0 - p_val), 0.2, 5.0)
    adv_weights = adv_weights / adv_weights.mean()
    print(f"  Adversarial Train-vs-Val AUC: {adv_auc:.4f}")

    adv_model = xgb.XGBRegressor(**BASE_XGB_PARAMS)
    adv_model.fit(X_train, y_train, sample_weight=adv_weights, eval_set=[(X_es, y_es)], verbose=250)
    adv_pred = adv_model.predict(X_holdout)
    results.append(evaluate_holdout(f"4. Adv-Weighted (AUC={adv_auc:.2f})", holdout_df, adv_pred))
    del adv_clf, adv_model, X_adv
    gc.collect()

    # 5. Era Boosting (Upweight bottom 30% train dates & warm-start)
    print("\n[5/7] Running Era Boosting...")
    tr_pred = ctrl_model.predict(X_train)
    tr_eval = pd.DataFrame({"date": train_df["date"].values, "y": y_train.values, "p": tr_pred})
    tr_era_corr = tr_eval.groupby("date", sort=False).apply(
        lambda g: g["p"].corr(g["y"]), include_groups=False
    )
    cutoff = tr_era_corr.quantile(0.30)
    hard_dates = set(tr_era_corr[tr_era_corr <= cutoff].index)
    era_weights = np.where(train_df["date"].isin(hard_dates), 3.0, 1.0)
    era_weights = era_weights / era_weights.mean()

    era_params = {**BASE_XGB_PARAMS, "n_estimators": 500, "learning_rate": 0.02}
    era_model = xgb.XGBRegressor(**era_params)
    era_model.fit(
        X_train,
        y_train,
        sample_weight=era_weights,
        eval_set=[(X_es, y_es)],
        xgb_model=ctrl_model.get_booster(),
        verbose=250,
    )
    era_pred = era_model.predict(X_holdout)
    results.append(evaluate_holdout("5. Era Boosting (+500 trees)", holdout_df, era_pred))
    del era_model, tr_eval
    gc.collect()

    # 6. Multi-Target Blending (5 diverse targets, per-date rank averaged)
    print("\n[6/7] Training Multi-Target Blend (5 targets)...")
    rank_preds = [per_date_rank(holdout_df, ctrl_pred)]
    for t_col in MULTI_TARGETS[1:]:
        if t_col not in train_df.columns:
            continue
        mask_tr = train_df[t_col].notna()
        mask_es = es_df[t_col].notna()
        print(f"  Fitting target: {t_col}...")
        m = xgb.XGBRegressor(**BASE_XGB_PARAMS)
        m.fit(
            X_train[mask_tr],
            train_df.loc[mask_tr, t_col],
            eval_set=[(X_es[mask_es], es_df.loc[mask_es, t_col])],
            verbose=False,
        )
        rank_preds.append(per_date_rank(holdout_df, m.predict(X_holdout)))
        del m
        gc.collect()

    multi_pred = np.mean(rank_preds, axis=0)
    results.append(evaluate_holdout("6. Multi-Target Rank Blend (5)", holdout_df, multi_pred))

    # 7. Post-Hoc Feature Neutralization (50% & 100% against signal/factor features)
    print(f"\n[7/7] Running Post-Hoc Feature Neutralization ({len(neut_cols)} signal cols)...")
    neut50_ctrl = neutralize_predictions(holdout_df, ctrl_pred, neut_cols, proportion=0.5)
    neut50_multi = neutralize_predictions(holdout_df, multi_pred, neut_cols, proportion=0.5)
    neut100_multi = neutralize_predictions(holdout_df, multi_pred, neut_cols, proportion=1.0)
    results.append(evaluate_holdout("7a. Control + 50% Neut", holdout_df, neut50_ctrl))
    results.append(evaluate_holdout("7b. MultiTarget + 50% Neut", holdout_df, neut50_multi))
    results.append(evaluate_holdout("7c. MultiTarget + 100% Neut", holdout_df, neut100_multi))

    # Print comparison table
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 92)
    print("HOLDOUT COMPARISON TABLE (OOS ~267 weekly dates)")
    print("=" * 92)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "MeanCorr": "{:.5f}".format,
                "StdCorr": "{:.5f}".format,
                "Sharpe": "{:.4f}".format,
                "HitRate": "{:.2%}".format,
                "MaxDD": "{:.4f}".format,
                "RMSE": "{:.5f}".format,
            },
        )
    )

    # 8. ACI Dynamic Position Sizing on Control & MultiTarget
    print("\n" + "=" * 92)
    print("ACI DYNAMIC POSITION SIZING (Magnitude-Aware Portfolio Metrics on Holdout)")
    print("=" * 92)
    for label, p_arr in [("Control XGB", ctrl_pred), ("Multi-Target Blend", multi_pred)]:
        aci = run_aci_sizing(holdout_df, p_arr)
        print(
            f"{label}:\n"
            f"  Unscaled  -> MeanRet: {aci['unscaled_mean']:.5f} | Sharpe: {aci['unscaled_sharpe']:.4f} | MaxDD: {aci['unscaled_mdd']:.4f}\n"
            f"  ACI-Scaled-> MeanRet: {aci['aci_mean']:.5f} | Sharpe: {aci['aci_sharpe']:.4f} | MaxDD: {aci['aci_mdd']:.4f} "
            f"(weights [{aci['weight_min']:.2f}, {aci['weight_max']:.2f}])"
        )

    res_df.to_csv("/content/benchmark_results.csv", index=False)
    print(f"\nCompleted all experiments in {time.time() - t0:.1f}s. Saved /content/benchmark_results.csv")


if __name__ == "__main__":
    main()
