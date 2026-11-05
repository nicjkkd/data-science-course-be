"""uv run python -m app.projects.car_price.train [--quick]

Needs data/processed/listings.parquet (scripts/build_car_dataset.py); --quick falls back to the
committed 200-row sample."""

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.contract import (
    ChartSpec,
    Importance,
    InsightsResponse,
    MetricsResponse,
    ModelRow,
    Point,
    Series,
)
from app.core.evaluate import histogram, pred_vs_actual_chart, regression_metrics
from app.core.project import write_artifacts
from app.core.regions import REGIONS
from app.projects.car_price.features import CATEGORICAL, FLAGS, NUMERIC, UNKNOWN, build_features
from app.projects.car_price.schemas import BODY_LABELS, FEATURE_LABELS
from app.projects.car_price.text_flags import flags_from_text

SLUG = "car-price"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data/processed/listings.parquet"
SAMPLE = HERE / "data/sample/listings_sample.parquet"
MAX_UNKNOWN = 0.5  # a categorical column unknown in more rows than this is not a feature


def load(quick: bool) -> pd.DataFrame:
    if DATA.exists():
        df = pd.read_parquet(DATA)
    elif quick:
        print("using the committed 200-row sample", flush=True)
        df = pd.read_parquet(SAMPLE)
    else:
        raise SystemExit("no data: run scripts/build_car_dataset.py")
    df = df.dropna(subset=["price_usd", "year", "mileage_km", "brand", "model"])
    df = df[df["price_usd"].between(500, 200_000) & (df["year"] >= 1990)]
    df = df[df["mileage_km"] <= 1_500_000]
    flags = df["description"].map(flags_from_text)
    for f in FLAGS:
        key = f.removeprefix("flag_")
        df[f] = flags.map(lambda s, k=key: k in s).astype(int)
    df["condition"] = np.where(
        df["flag_needs_repair"] == 1,
        "needs_repair",
        np.where(df["damage"].astype(bool), "repaired", "undamaged"),
    )
    df = df.reset_index(drop=True)
    return (
        df.sample(5000, random_state=42).reset_index(drop=True) if quick and len(df) > 5000 else df
    )


def main(quick: bool) -> None:
    df = load(quick)
    y = np.log1p(df["price_usd"].to_numpy())
    min_count = 20 if len(df) > 2000 else 3
    counts = df["model"].str.lower().str.strip().value_counts()
    model_vocab = set(counts[counts >= min_count].index)
    x_all = build_features(df, with_flags=True, model_vocab=model_vocab)
    unknown_share = {c: float((x_all[c] == UNKNOWN).mean()) for c in CATEGORICAL}
    cats = [c for c in CATEGORICAL if unknown_share[c] <= MAX_UNKNOWN]
    dropped = [c for c in CATEGORICAL if c not in cats]
    print(f"{len(df)} rows; dropped (mostly unknown): {dropped}", flush=True)
    x_full = x_all[NUMERIC + cats + FLAGS]
    x_noflags = x_full.drop(columns=FLAGS)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    # early stopping on a validation split carved from train, so the test set stays untouched
    idx_fit, idx_val = train_test_split(idx_tr, test_size=0.2, random_state=42)
    y_tr, y_te = y[idx_tr], y[idx_te]
    y_te_usd = np.expm1(y_te)
    n_est = 300 if quick else 3000
    preds: dict[str, np.ndarray] = {}

    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_count),
                cats,
            ),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ]
    )
    ridge = make_pipeline(pre, Ridge(alpha=1.0)).fit(x_noflags.iloc[idx_tr], y_tr)
    preds["Ridge (baseline)"] = np.expm1(ridge.predict(x_noflags.iloc[idx_te]))

    x_codes = x_noflags.copy()
    for c in cats:
        x_codes[c] = x_codes[c].cat.codes
    x_codes = x_codes.fillna(-1)
    rf = RandomForestRegressor(
        n_estimators=100 if quick else 300,
        min_samples_leaf=3,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(x_codes.iloc[idx_tr], y_tr)
    preds["Random Forest"] = np.expm1(rf.predict(x_codes.iloc[idx_te]))
    del rf

    def fit_lgb(x: pd.DataFrame) -> lgb.LGBMRegressor:
        m = lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        m.fit(
            x.iloc[idx_fit],
            y[idx_fit],
            eval_X=x.iloc[idx_val],
            eval_y=y[idx_val],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return m

    lgb_plain = fit_lgb(x_noflags)
    preds["LightGBM"] = np.expm1(lgb_plain.predict(x_noflags.iloc[idx_te]))
    has_text = int(df[FLAGS].to_numpy().sum()) > 0
    final, x_final = lgb_plain, x_noflags
    if has_text:
        lgb_flags = fit_lgb(x_full)
        preds["LightGBM + text flags"] = np.expm1(lgb_flags.predict(x_full.iloc[idx_te]))
        final, x_final = lgb_flags, x_full

    scores = {k: regression_metrics(y_te_usd, p) for k, p in preds.items()}
    best = min(scores, key=lambda k: scores[k]["MAE"])
    rows = [
        ModelRow(name=k, is_baseline=k.startswith("Ridge"), is_best=k == best, metrics=m)
        for k, m in scores.items()
    ]

    # quantile models for the range (p10 / p90) on the final feature set
    quantiles = {}
    for q in (0.1, 0.9):
        quantiles[str(q)] = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=150 if quick else 600,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            verbose=-1,
        ).fit(x_final.iloc[idx_tr], y_tr)
    p10 = np.expm1(quantiles["0.1"].predict(x_final.iloc[idx_te]))
    p90 = np.expm1(quantiles["0.9"].predict(x_final.iloc[idx_te]))
    coverage = float(np.mean((y_te_usd >= p10) & (y_te_usd <= p90)))

    gain = final.booster_.feature_importance("gain")
    imp = sorted(
        ((f, v) for f, v in zip(x_final.columns, gain, strict=True) if v > 0), key=lambda t: -t[1]
    )[:15]
    total = sum(v for _, v in imp) or 1
    importances = [
        Importance(feature=f, label=FEATURE_LABELS.get(f, f), value=float(v / total))
        for f, v in imp
    ]

    final_name = "LightGBM + text flags" if has_text else "LightGBM"
    p_final = preds[final_name]
    charts = [
        pred_vs_actual_chart(y_te_usd, p_final),
        histogram(
            np.clip(p_final - y_te_usd, -15_000, 15_000),
            id="residuals_hist",
            title="Розподіл похибок",
            x_label="Прогноз − факт, USD (обрізано ±15 000)",
        ),
        mae_by_brand_chart(df.iloc[idx_te], y_te_usd, p_final),
    ]
    metrics = MetricsResponse(
        task="regression",
        trained_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        n_train=len(idx_tr),
        n_test=len(idx_te),
        models=rows,
        importances=importances,
        charts=charts,
    )
    top_brands = df["brand"].value_counts().index[:40]
    names = df.groupby("brand")["brand_name"].agg(lambda s: s.mode().iat[0])
    bodies = df["body"].dropna().str.lower().value_counts().index[:12]
    meta = {
        "model_version": f"{SLUG}@{dt.date.today().isoformat()}",
        "git_sha": git_sha(),
        "with_flags": bool(has_text),
        "features": list(x_final.columns),
        "dropped_features": dropped,
        "categories": {c: list(x_final[c].cat.categories) for c in cats},
        "model_vocab": sorted(model_vocab),
        "mae_usd": scores[final_name]["MAE"],
        "range_coverage": round(coverage, 3),
        "brands": {b: str(names.get(b, b)) for b in top_brands},  # most common first
        # form options; when body is not a model feature, offer the full list
        "bodies": sorted(bodies) if "body" in cats else list(BODY_LABELS),
        "regions": list(REGIONS),
        "n_api_rows": int((df["source"] == "api").sum()),
    }
    insights = build_insights(df, dropped)
    d = write_artifacts(SLUG, {"final": final, "quantiles": quantiles}, meta, metrics, insights)
    print({k: round(v["MAE"]) for k, v in scores.items()}, f"p10–p90 coverage {coverage:.2f} →", d)


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def mae_by_brand_chart(df_te: pd.DataFrame, y, p) -> ChartSpec:
    err = pd.Series(np.abs(p - y), index=df_te.index)
    top = df_te["brand"].value_counts().index[:12]
    names = df_te.groupby("brand")["brand_name"].first()
    pts = [Point(x=str(names[b]), y=round(float(err[df_te["brand"] == b].mean()))) for b in top]
    return ChartSpec(
        id="mae_by_brand",
        title="MAE за марками",
        kind="bar",
        y_label="USD",
        series=[Series(name="MAE", points=pts)],
    )


def build_insights(df: pd.DataFrame, dropped: list[str]) -> InsightsResponse:
    per_year = df.groupby("year")["price_usd"].agg(["median", "count"])
    by_year = per_year[per_year["count"] >= 30]["median"].loc[2000:]  # skip thin years
    by_region = df.groupby("region")["price_usd"].median().sort_values(ascending=False)[:15]
    names = df.groupby("brand")["brand_name"].first()
    brand_share = df["brand"].value_counts(normalize=True)[:12] * 100
    sample = df.sample(min(500, len(df)), random_state=42)
    n_api = int((df["source"] == "api").sum())
    n_rows = f"{len(df):,}".replace(",", " ")
    note = (
        f"{n_rows} оголошень AUTO.RIA: публічний дамп (січень 2024) + {n_api} свіжих через API. "
        "Ціни — запитувані в оголошеннях, не ціни угод."
    )
    if dropped:
        labels = ", ".join(FEATURE_LABELS.get(c, c).lower() for c in dropped)
        note += f" Ознаки «{labels}» здебільшого відсутні в даних і не використовуються моделлю."
    return InsightsResponse(
        charts=[
            histogram(
                np.clip(df["price_usd"], 0, 59_999),
                bins=np.arange(0, 62_000, 2_000),
                id="price_hist",
                title="Розподіл цін",
                x_label="USD (обрізано на 60 000)",
            ).model_copy(update={"description": note}),
            ChartSpec(
                id="price_by_year",
                title="Медіана ціни за роком випуску",
                kind="line",
                x_label="Рік",
                y_label="USD",
                series=[
                    Series(
                        name="Медіана",
                        points=[Point(x=int(k), y=float(v)) for k, v in by_year.items()],
                    )
                ],
            ),
            ChartSpec(
                id="price_by_region",
                title="Медіана ціни за регіоном",
                kind="bar",
                y_label="USD",
                series=[
                    Series(
                        name="Медіана",
                        points=[
                            Point(x=REGIONS.get(str(k), str(k)), y=float(v))
                            for k, v in by_region.items()
                        ],
                    )
                ],
            ),
            ChartSpec(
                id="mileage_vs_price",
                title="Пробіг і ціна",
                kind="scatter",
                x_label="км",
                y_label="USD",
                series=[
                    Series(
                        name="Оголошення",
                        points=[
                            Point(x=float(r.mileage_km), y=float(r.price_usd))
                            for r in sample.itertuples()
                        ],
                    )
                ],
            ),
            ChartSpec(
                id="brand_share",
                title="Частка марок у вибірці, %",
                kind="bar",
                series=[
                    Series(
                        name="%",
                        points=[
                            Point(x=str(names.get(k, k)), y=round(float(v), 1))
                            for k, v in brand_share.items()
                        ],
                    )
                ],
            ),
        ]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
