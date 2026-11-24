"""Simple XGBoost Numerai validation predictor using numerapi on Colab GPU."""

import json
from pathlib import Path

import numerapi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

DATA_DIR = Path("/content/data")
DATASET_VERSION = "v6.0"
FEATURE_SET = "small"  # "small", "medium", or "all"
TARGET_COL = "target"
PRED_OUTPUT_PATH = Path("/content/val_predictions.parquet")


def download_datasets(napi: numerapi.NumerAPI, version: str, data_dir: Path) -> dict[str, Path]:
    """Download features.json, train.parquet, and validation.parquet if not already present."""
    files = {
        "features": f"{version}/features.json",
        "train": f"{version}/train.parquet",
        "validation": f"{version}/validation.parquet",
    }
    local_paths: dict[str, Path] = {}
    for key, remote_path in files.items():
        dest = data_dir / remote_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"Downloading {remote_path} -> {dest} ...")
            napi.download_dataset(remote_path, dest_path=str(dest))
        else:
            print(f"Using cached {dest}")
        local_paths[key] = dest
    return local_paths


def load_feature_list(features_path: Path, feature_set: str) -> list[str]:
    """Load the requested feature subset from features.json."""
    with features_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_sets = meta.get("feature_sets", {})
    if feature_set in feature_sets:
        return feature_sets[feature_set]
    fallback = next(iter(feature_sets))
    print(f"Feature set '{feature_set}' not found; falling back to '{fallback}'.")
    return feature_sets[fallback]


def load_split(parquet_path: Path, features: list[str], target_col: str, filter_val: bool = False) -> tuple[pd.DataFrame, str]:
    """Read required columns from a Numerai parquet file and drop rows with missing targets."""
    schema_names = set(pq.ParquetFile(parquet_path).schema.names)
    time_col = "era" if "era" in schema_names else "date"
    cols = [time_col, target_col] + [c for c in features if c in schema_names]
    if filter_val and "data_type" in schema_names:
        cols.append("data_type")

    df = pd.read_parquet(parquet_path, columns=cols)
    if filter_val and "data_type" in df.columns:
        df = df[df["data_type"] == "validation"].drop(columns=["data_type"])
    df = df.dropna(subset=[target_col])
    return df, time_col


def compute_era_metrics(df: pd.DataFrame, time_col: str, target_col: str, pred_col: str = "prediction") -> dict[str, float]:
    """Compute overall RMSE and per-era/date Pearson correlation summary statistics."""
    per_era_corr = df.groupby(time_col, observed=True).apply(
        lambda g: g[pred_col].corr(g[target_col]),
        include_groups=False,
    ).dropna()

    mean_corr = float(per_era_corr.mean())
    std_corr = float(per_era_corr.std(ddof=1))
    sharpe = mean_corr / std_corr if std_corr > 0 else 0.0
    hit_rate = float((per_era_corr > 0).mean())
    rmse = float(np.sqrt(np.mean((df[target_col] - df[pred_col]) ** 2)))

    return {
        "n_eras": int(len(per_era_corr)),
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "rmse": rmse,
    }


def main() -> None:
    napi = numerapi.NumerAPI()
    paths = download_datasets(napi, DATASET_VERSION, DATA_DIR)

    features = load_feature_list(paths["features"], FEATURE_SET)
    print(f"Using {len(features)} features from '{FEATURE_SET}' set.")

    train_df, _ = load_split(paths["train"], features, TARGET_COL)
    val_df, time_col = load_split(paths["validation"], features, TARGET_COL, filter_val=True)
    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")

    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train_df[c])]
    for col in cat_cols:
        dtype = pd.CategoricalDtype(categories=train_df[col].dropna().unique())
        train_df[col] = train_df[col].astype(dtype)
        val_df[col] = val_df[col].astype(dtype)

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        colsample_bytree=0.1,
        subsample=0.8,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        random_state=42,
    )

    print("Training XGBRegressor on CUDA...")
    model.fit(
        train_df[features],
        train_df[TARGET_COL],
        eval_set=[(val_df[features], val_df[TARGET_COL])],
        verbose=100,
    )

    val_df["prediction"] = model.predict(val_df[features])
    metrics = compute_era_metrics(val_df, time_col, TARGET_COL)

    print("\nValidation Metrics:")
    print(f"  Periods ({time_col}) : {metrics['n_eras']}")
    print(f"  Mean Corr      : {metrics['mean_corr']:.5f}")
    print(f"  Std Corr       : {metrics['std_corr']:.5f}")
    print(f"  Sharpe         : {metrics['sharpe']:.4f}")
    print(f"  Hit Rate       : {metrics['hit_rate']:.2%}")
    print(f"  RMSE           : {metrics['rmse']:.5f}")

    val_df[[time_col, TARGET_COL, "prediction"]].to_parquet(PRED_OUTPUT_PATH)
    print(f"\nSaved validation predictions to {PRED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
