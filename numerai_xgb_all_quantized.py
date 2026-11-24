"""Streaming QuantileDMatrix XGBoost benchmark on ALL 1,040 Numerai v6.0 features.

Streams train/val/holdout parquet files in batches via xgb.DataIter so all
4.33M rows x 1,040 features fit comfortably in Colab T4 RAM (12.7 GB) and VRAM (15 GB).
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

DATA_DIR = Path("/content/data/v6.0")
FEATURE_SET = "all"  # 1,040 features
PURGE_DATES = 16
EARLY_STOP_DATES = 100
TARGET_COL = "target"
BATCH_SIZE = 150_000
TRAIN_DATE_STRIDE = 4  # non-overlapping 20-day eras (~1.08M rows) to fit 1,040 features in 12.7 GB Colab RAM
ES_DATE_STRIDE = 2  # ~50 early-stop dates (~319k rows)
RANDOM_STATE = 42


class ParquetBatchIter(xgb.DataIter):
    """Stream parquet batches into XGBoost QuantileDMatrix with optional date filtering and weights."""

    def __init__(
        self,
        parquet_path: Path,
        features: list[str],
        target_col: str,
        cat_dtypes: dict[str, pd.CategoricalDtype],
        allowed_dates: set[str] | None = None,
        date_weights: dict[str, float] | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.features = features
        self.target_col = target_col
        self.cat_dtypes = cat_dtypes
        self.allowed_dates = allowed_dates
        self.date_weights = date_weights
        self.batch_size = batch_size

        schema_names = set(pq.ParquetFile(parquet_path).schema.names)
        self.cols = ["date", target_col] + [c for c in features if c in schema_names]
        self._iter = None

    def reset(self) -> None:
        pf = pq.ParquetFile(self.parquet_path)
        self._iter = pf.iter_batches(batch_size=self.batch_size, columns=self.cols)

    def next(self, input_data) -> int:
        while True:
            try:
                batch = next(self._iter)
            except StopIteration:
                return 0

            df = batch.to_pandas()
            if self.allowed_dates is not None:
                df = df[df["date"].isin(self.allowed_dates)]
            df = df.dropna(subset=[self.target_col])
            if df.empty:
                continue

            for col, dtype in self.cat_dtypes.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)

            kwargs = {
                "data": df[self.features],
                "label": df[self.target_col].to_numpy(dtype=np.float32),
            }
            if self.date_weights is not None:
                w = df["date"].map(self.date_weights).fillna(1.0).to_numpy(dtype=np.float32)
                kwargs["weight"] = w

            input_data(**kwargs)
            return 1


def predict_in_batches(
    booster: xgb.Booster,
    parquet_path: Path,
    features: list[str],
    cat_dtypes: dict[str, pd.CategoricalDtype],
    allowed_dates: set[str],
    batch_size: int = 50_000,
) -> np.ndarray:
    """Run zero-copy CPU inplace_predict in small chunks to avoid DMatrix allocation."""
    import pyarrow as pa

    booster.set_param({"device": "cpu"})
    schema_names = set(pq.ParquetFile(parquet_path).schema.names)
    cols = ["date", TARGET_COL] + [c for c in features if c in schema_names]
    pf = pq.ParquetFile(parquet_path)
    preds: list[np.ndarray] = []

    for idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols)):
        df = batch.to_pandas()
        df = df[df["date"].isin(allowed_dates)].dropna(subset=[TARGET_COL])
        if df.empty:
            del df
            continue
        for col, dtype in cat_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        preds.append(booster.inplace_predict(df[features]))
        del df
        if idx % 10 == 0:
            pa.default_memory_pool().release_unused()
            gc.collect()
    pa.default_memory_pool().release_unused()
    gc.collect()
    return np.concatenate(preds)


def neutralize_predictions(
    df: pd.DataFrame,
    pred: np.ndarray,
    neutralize_cols: list[str],
    proportion: float = 0.5,
) -> np.ndarray:
    """Per-date linear neutralization against signal/factor features."""
    if not neutralize_cols or proportion <= 0.0:
        return pred
    num_cols = [c for c in neutralize_cols if pd.api.types.is_numeric_dtype(df[c])]
    out = np.empty_like(pred, dtype=np.float64)
    tmp = pd.DataFrame({"date": df["date"].values, "pred": pred}, index=df.index)

    for _, idx in tmp.groupby("date", sort=False).groups.items():
        y = tmp.loc[idx, "pred"].to_numpy(dtype=np.float64)
        X = np.nan_to_num(df.loc[idx, num_cols].to_numpy(dtype=np.float64), nan=0.0)
        X = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[df.index.get_indexer(idx)] = y - proportion * (X @ coef)
    return out


def evaluate_holdout(name: str, eval_base: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    eval_df = pd.DataFrame(
        {"date": eval_base["date"].values, "target": eval_base[TARGET_COL].values, "pred": pred}
    )
    era_corrs = eval_df.groupby("date", sort=True).apply(
        lambda g: g["pred"].corr(g["target"]), include_groups=False
    ).dropna()

    mean_corr = float(era_corrs.mean())
    std_corr = float(era_corrs.std(ddof=1))
    sharpe = mean_corr / std_corr if std_corr > 0 else 0.0
    hit_rate = float((era_corrs > 0).mean())
    cum = era_corrs.cumsum()
    max_dd = float((cum - cum.cummax()).min())
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


def run_aci_sizing(eval_base: pd.DataFrame, pred: np.ndarray, alpha: float = 0.1, gamma: float = 0.05) -> dict[str, float]:
    eval_df = pd.DataFrame(
        {"date": eval_base["date"].values, "target": eval_base[TARGET_COL].values, "pred": pred}
    )
    alpha_t = alpha
    q_hist: list[float] = []
    unscaled_rets, aci_weights = [], []

    for _, g in eval_df.groupby("date", sort=True):
        p = g["pred"].to_numpy(dtype=np.float64)
        y = g["target"].to_numpy(dtype=np.float64)
        p_z = (p - p.mean()) / (p.std() + 1e-8)
        unscaled_rets.append(float(np.mean(p_z * (y - y.mean()))))

        resid = np.abs(y - p)
        q_t = float(np.quantile(resid, np.clip(1.0 - alpha_t, 0.05, 0.99)))
        w_t = 1.0 if not q_hist else float(np.clip(np.median(q_hist[-20:]) / (q_t + 1e-8), 0.25, 1.5))
        aci_weights.append(w_t)
        q_hist.append(q_t)
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - float(np.mean(resid > q_t))), 0.01, 0.5))

    unscaled = np.array(unscaled_rets)
    weights = np.array(aci_weights)
    weights /= weights.mean()
    scaled = unscaled * weights

    def stats(r: np.ndarray) -> tuple[float, float, float]:
        c = np.cumsum(r)
        return float(r.mean()), float(r.mean() / (r.std(ddof=1) + 1e-12)), float((c - np.maximum.accumulate(c)).min())

    u_m, u_s, u_d = stats(unscaled)
    s_m, s_s, s_d = stats(scaled)
    return {"u_mean": u_m, "u_sharpe": u_s, "u_mdd": u_d, "s_mean": s_m, "s_sharpe": s_s, "s_mdd": s_d}


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    napi = numerapi.NumerAPI()
    for fname in ["features.json", "train.parquet", "validation.parquet"]:
        dest = DATA_DIR / fname
        if not dest.exists():
            print(f"Downloading v6.0/{fname}...")
            napi.download_dataset(f"v6.0/{fname}", dest_path=str(dest))

    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "validation.parquet"

    with (DATA_DIR / "features.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["feature_sets"][FEATURE_SET]
    signal_cols = [c for c in meta["feature_sets"].get("signals", []) if c in features]
    print(f"Loaded feature_set='{FEATURE_SET}' ({len(features)} features, {len(signal_cols)} signal cols).")

    # Build shared categorical dtypes from train.parquet and select strided train dates
    train_meta = pd.read_parquet(train_path, columns=["date", TARGET_COL]).dropna(subset=[TARGET_COL])
    all_train_dates = sorted(train_meta["date"].unique())
    train_dates_set = set(all_train_dates[::TRAIN_DATE_STRIDE])
    del train_meta

    cat_dtypes: dict[str, pd.CategoricalDtype] = {}
    if "feature_country" in features:
        countries = pd.read_parquet(train_path, columns=["feature_country"])["feature_country"].dropna().unique()
        cat_dtypes["feature_country"] = pd.CategoricalDtype(categories=countries)

    # Determine chronological Early-Stop Val and Holdout dates from validation.parquet
    val_meta = pd.read_parquet(val_path, columns=["date", "data_type", TARGET_COL])
    val_meta = val_meta[val_meta["data_type"] == "validation"].dropna(subset=[TARGET_COL])
    val_dates = sorted(val_meta["date"].unique())
    es_dates = set(val_dates[PURGE_DATES : PURGE_DATES + EARLY_STOP_DATES : ES_DATE_STRIDE])
    holdout_dates = set(val_dates[PURGE_DATES + EARLY_STOP_DATES + PURGE_DATES :])
    del val_meta

    # Load lightweight holdout evaluation frame (only date, target, and 23 signal cols)
    holdout_eval_df = pd.read_parquet(val_path, columns=["date", "data_type", TARGET_COL] + signal_cols)
    holdout_eval_df = (
        holdout_eval_df[
            (holdout_eval_df["data_type"] == "validation") & (holdout_eval_df["date"].isin(holdout_dates))
        ]
        .dropna(subset=[TARGET_COL])
        .drop(columns=["data_type"])
        .reset_index(drop=True)
    )
    print(f"Holdout eval rows: {len(holdout_eval_df):,}, dates: {holdout_eval_df['date'].nunique()}")

    # Build streaming QuantileDMatrix on GPU
    print(f"Building quantized train ({len(train_dates_set)}/{len(all_train_dates)} dates) & early-stop DMatrices on CUDA...")
    train_iter = ParquetBatchIter(train_path, features, TARGET_COL, cat_dtypes, allowed_dates=train_dates_set)
    dtrain = xgb.QuantileDMatrix(train_iter, enable_categorical=True)

    es_iter = ParquetBatchIter(val_path, features, TARGET_COL, cat_dtypes, allowed_dates=es_dates)
    des = xgb.QuantileDMatrix(es_iter, ref=dtrain, enable_categorical=True)
    print(f"QuantileDMatrix built: train={dtrain.num_row():,}x{dtrain.num_col()}, es={des.num_row():,}x{des.num_col()}")

    results: list[dict[str, float]] = []

    # 1. Control XGB on ALL 1,040 features
    print("\n[1/4] Training Control XGB (all 1,040 features, depth=5)...")
    ctrl_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 100,
        "lambda": 10.0,
        "subsample": 0.8,
        "colsample_bytree": 0.1,
        "seed": RANDOM_STATE,
    }
    ctrl_booster = xgb.train(
        ctrl_params,
        dtrain,
        num_boost_round=2000,
        evals=[(des, "val")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    print(f"  Best iteration: {ctrl_booster.best_iteration}")
    ctrl_pred = predict_in_batches(ctrl_booster, val_path, features, cat_dtypes, holdout_dates)
    results.append(evaluate_holdout("1. Control XGB (all 1040)", holdout_eval_df, ctrl_pred))
    del ctrl_booster
    gc.collect()

    # 2. Deep & Slow Regularized XGB on ALL 1,040 features
    print("\n[2/4] Training Deep & Slow XGB (depth=7, mcw=1000, colsample=0.05, lr=0.015)...")
    deep_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.015,
        "max_depth": 7,
        "min_child_weight": 1000,
        "lambda": 20.0,
        "subsample": 0.8,
        "colsample_bytree": 0.05,
        "seed": RANDOM_STATE,
    }
    deep_booster = xgb.train(
        deep_params,
        dtrain,
        num_boost_round=2500,
        evals=[(des, "val")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    print(f"  Best iteration: {deep_booster.best_iteration}")
    deep_pred = predict_in_batches(deep_booster, val_path, features, cat_dtypes, holdout_dates)
    results.append(evaluate_holdout("2. Deep & Slow XGB (all 1040)", holdout_eval_df, deep_pred))

    # 3. Era Boosting on Deep & Slow (upweight bottom 30% train dates)
    print("\n[3/4] Running Era Boosting on Deep & Slow...")
    tr_dates = pd.read_parquet(train_path, columns=["date", TARGET_COL]).dropna(subset=[TARGET_COL])
    tr_dates = tr_dates[tr_dates["date"].isin(train_dates_set)].reset_index(drop=True)
    tr_preds = predict_in_batches(deep_booster, train_path, features, cat_dtypes, train_dates_set)
    tr_dates["pred"] = tr_preds
    tr_era_corr = tr_dates.groupby("date", sort=False).apply(
        lambda g: g["pred"].corr(g[TARGET_COL]), include_groups=False
    )
    cutoff = tr_era_corr.quantile(0.30)
    hard_dates = set(tr_era_corr[tr_era_corr <= cutoff].index)
    date_weights = {d: (3.0 if d in hard_dates else 1.0) for d in train_dates_set}
    del tr_dates, tr_preds
    gc.collect()

    del dtrain, des
    gc.collect()
    era_iter = ParquetBatchIter(
        train_path, features, TARGET_COL, cat_dtypes, allowed_dates=train_dates_set, date_weights=date_weights
    )
    dtrain_era = xgb.QuantileDMatrix(era_iter, enable_categorical=True)
    des_era = xgb.QuantileDMatrix(
        ParquetBatchIter(val_path, features, TARGET_COL, cat_dtypes, allowed_dates=es_dates),
        ref=dtrain_era,
        enable_categorical=True,
    )

    deep_booster.set_param({"device": "cuda"})
    era_params = {**deep_params, "learning_rate": 0.01}
    era_booster = xgb.train(
        era_params,
        dtrain_era,
        num_boost_round=400,
        evals=[(des_era, "val")],
        xgb_model=deep_booster,
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    era_pred = predict_in_batches(era_booster, val_path, features, cat_dtypes, holdout_dates)
    results.append(evaluate_holdout("3. Deep+Slow + EraBoost", holdout_eval_df, era_pred))

    # 4. Post-Hoc Feature Neutralization against all 23 signal/factor features
    print(f"\n[4/4] Running Post-Hoc Neutralization against {len(signal_cols)} signal cols...")
    for label, p_arr in [("Deep+Slow", deep_pred), ("Deep+EraBoost", era_pred)]:
        for prop in [0.5, 1.0]:
            p_neut = neutralize_predictions(holdout_eval_df, p_arr, signal_cols, proportion=prop)
            results.append(evaluate_holdout(f"4. {label} + {int(prop*100)}% Neut", holdout_eval_df, p_neut))

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 92)
    print("ALL-1040-FEATURES QUANTIZED HOLDOUT COMPARISON (267 OOS Dates)")
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

    aci = run_aci_sizing(holdout_eval_df, era_pred)
    print(
        f"\nACI Sizing on Deep+EraBoost:\n"
        f"  Unscaled  -> MeanRet: {aci['u_mean']:.5f} | Sharpe: {aci['u_sharpe']:.4f} | MaxDD: {aci['u_mdd']:.4f}\n"
        f"  ACI-Scaled-> MeanRet: {aci['s_mean']:.5f} | Sharpe: {aci['s_sharpe']:.4f} | MaxDD: {aci['s_mdd']:.4f}"
    )

    res_df.to_csv("/content/benchmark_all_results.csv", index=False)
    print(f"\nCompleted in {time.time() - t0:.1f}s. Saved /content/benchmark_all_results.csv")


if __name__ == "__main__":
    main()
