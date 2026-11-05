"""Car features. The only feature logic; train.py and service.py import it."""

import numpy as np
import pandas as pd

from app.projects.car_price.text_flags import RULES

CATEGORICAL = ["brand", "model", "fuel", "gearbox", "body", "drive", "region", "condition"]
# No `year`: the 2024 dump and fresh API rows were listed in different years, so the car's
# age at listing time is the comparable quantity (year = listing year − age).
NUMERIC = ["age", "mileage_km", "km_per_year", "engine_l"]
FLAGS = [f"flag_{k}" for k in RULES]
CURRENT_YEAR = 2026
UNKNOWN = "unknown"


def build_features(
    df: pd.DataFrame, *, with_flags: bool = True, model_vocab: set[str] | None = None
) -> pd.DataFrame:
    """df has the CarInput columns (+ optional `posted_at`, flag_* columns)."""
    x = pd.DataFrame(index=df.index)
    listed = (
        pd.to_datetime(df["posted_at"]).dt.year
        if "posted_at" in df
        else pd.Series(CURRENT_YEAR, index=df.index)
    )
    x["age"] = (listed - df["year"].astype(int)).clip(lower=0)
    x["mileage_km"] = df["mileage_km"].astype(float)
    x["km_per_year"] = x["mileage_km"] / np.clip(x["age"], 1, None)
    x["engine_l"] = pd.to_numeric(df["engine_l"], errors="coerce").astype(float)
    for c in CATEGORICAL:
        col = df[c] if c in df else pd.Series(None, index=df.index, dtype=object)
        vals = col.fillna(UNKNOWN).astype(str).str.lower().str.strip()
        if c == "model" and model_vocab is not None:
            vals = vals.where(vals.isin(model_vocab), "other")
        x[c] = vals.astype("category")
    if with_flags:
        for f in FLAGS:
            x[f] = df[f].astype(int) if f in df else 0
    return x


def align_categories(x: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Give every categorical column the training levels, so LightGBM sees identical codes.
    Unseen values become NaN (LightGBM treats them as missing)."""
    x = x.copy()
    for c, levels in meta["categories"].items():
        if c in x:
            x[c] = pd.Categorical(x[c].astype(str), categories=levels)
    return x
