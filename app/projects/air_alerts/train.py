"""uv run python -m app.projects.air_alerts.train [--quick]

Needs data/processed/alerts.parquet (scripts/fetch_alerts.py)."""

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from app.core.contract import (
    Cell,
    ChartSpec,
    Importance,
    InsightsResponse,
    MetricsResponse,
    ModelRow,
    Point,
    Series,
)
from app.core.evaluate import calibration_chart, probabilistic_metrics
from app.core.project import artifacts_dir, write_artifacts
from app.core.regions import REGION_ORDER, REGIONS
from app.projects.air_alerts.features import (
    FEATURE_LABELS,
    FEATURES,
    LAG_FEATURES,
    PERMANENT,
    RATE_DAYS,
    TZ,
    add_features,
    hourly_grid,
    to_model_frame,
)

SLUG = "air-alerts"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data/processed/alerts.parquet"
SAMPLE = HERE / "data/sample/alerts_sample.parquet"
TEST_WEEKS = 8
VAL_WEEKS = 8
RECENT_DAYS = 90
DURATION_FEATURES = ["region", "hour", "dow", "month", "cnt_24h", "nat_cnt_6h", "active_now"]
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]


def load(quick: bool) -> tuple[pd.DataFrame, bool]:
    """(alerts, is_sample). Without the processed data, --quick falls back to the sample."""
    is_sample = not DATA.exists()
    if is_sample:
        if not quick:
            raise SystemExit("no data: run scripts/fetch_alerts.py")
        print("using the committed 2-week sample", flush=True)
    a = pd.read_parquet(SAMPLE if is_sample else DATA)
    return a[~a["region"].isin(PERMANENT)].reset_index(drop=True), is_sample


def main(quick: bool) -> None:
    alerts, is_sample = load(quick)
    regions = [r for r in REGION_ORDER if r in set(alerts["region"])]
    end = alerts["started_at"].max().tz_convert("UTC").floor("h")
    first = alerts["started_at"].min().tz_convert("UTC").ceil("h")
    if is_sample:  # smoke run on 2 weeks: days 1–8 train, 9–11 val, 12–14 test
        start, test_from, val_from = first, end - pd.Timedelta(days=3), end - pd.Timedelta(days=6)
    else:
        test_from = end - pd.Timedelta(weeks=TEST_WEEKS)
        val_from = test_from - pd.Timedelta(weeks=VAL_WEEKS)
        train_weeks = 26 if quick else 10_000
        start = max(first, val_from - pd.Timedelta(weeks=train_weeks))
        start -= pd.Timedelta(days=RATE_DAYS)  # warm-up for the 28-day rate, dropped below
    grid = add_features(hourly_grid(alerts, regions, max(start, first), end))
    if not is_sample:
        grid = grid[grid["ts"] >= start + pd.Timedelta(days=RATE_DAYS)]
    tr = grid[grid["ts"] < val_from]
    va = grid[(grid["ts"] >= val_from) & (grid["ts"] < test_from)]
    te = grid[grid["ts"] >= test_from]
    print(f"rows: train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=150 if quick else 600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        to_model_frame(tr, regions),
        tr["label"],
        eval_X=to_model_frame(va, regions),
        eval_y=va["label"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y = te["label"].to_numpy()
    p_model = model.predict_proba(to_model_frame(te, regions))[:, 1]
    p_base = te["rate_28d"].clip(0.001, 0.999).to_numpy()
    scores = {
        "Історична частота (baseline)": probabilistic_metrics(y, p_base),
        "LightGBM": probabilistic_metrics(y, p_model),
    }
    best = min(scores, key=lambda k: scores[k]["Brier"])
    rows = [
        ModelRow(name=k, is_baseline=k.endswith("(baseline)"), is_best=k == best, metrics=m)
        for k, m in scores.items()
    ]

    gain = model.booster_.feature_importance("gain")
    total = float(gain.sum()) or 1.0
    importances = sorted(
        (
            Importance(feature=f, label=FEATURE_LABELS[f], value=float(v / total))
            for f, v in zip(FEATURES, gain, strict=True)
            if v > 0
        ),
        key=lambda i: -i.value,
    )

    duration, duration_mae = fit_duration(alerts, grid, val_from, test_from, regions, quick)
    cal = calibration_chart(y, p_model)
    cal.series[0].name = "LightGBM"
    cal.description = "Чи означає «30 %» справді 30 % — головний графік"
    metrics = MetricsResponse(
        task="timeseries",
        trained_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        n_train=len(tr),
        n_test=len(te),
        models=rows,
        importances=importances,
        charts=[cal, backtest_chart(te, p_model, p_base), roc_chart(y, p_model, p_base)],
    )

    recent_from = end - pd.Timedelta(days=RECENT_DAYS)
    recent_alerts = alerts[alerts["finished_at"] >= recent_from]
    recent_grid = grid[grid["ts"] >= recent_from]
    typicals = recent_grid.groupby("region")[LAG_FEATURES].median().round(2)
    meta = {
        "model_version": f"{SLUG}@{dt.date.today().isoformat()}",
        "git_sha": git_sha(),
        "features": FEATURES,
        "categories": {"region": regions},
        "history_end": end.tz_convert(TZ).isoformat(),
        "region_typicals": typicals.to_dict(orient="index"),
        "permanent": sorted(PERMANENT),
        "duration_mae_min": duration_mae,
    }
    d = write_artifacts(
        SLUG,
        {"proba": model, "duration": duration},
        meta,
        metrics,
        build_insights(alerts, recent_grid, regions),
    )
    recent_alerts.to_parquet(artifacts_dir(SLUG) / "recent_alerts.parquet")
    show = {k: {m: round(v, 4) for m, v in s.items()} for k, s in scores.items()}
    print(show, f"duration MAE {duration_mae} min →", d, flush=True)


def fit_duration(alerts, grid, val_from, test_from, regions, quick):
    """LightGBM on log1p(duration) for alert starts (alerts with a recorded end only)."""
    a = alerts[~alerts["naive"]].copy()
    a["ts"] = a["started_at"].dt.tz_convert("UTC").dt.floor("h")
    a = a.merge(grid[["ts", "region", *DURATION_FEATURES[1:]]], on=["ts", "region"])
    a["minutes"] = (a["finished_at"] - a["started_at"]).dt.total_seconds() / 60
    a = a[a["minutes"].between(1, 24 * 60)]
    x = a[DURATION_FEATURES].copy()
    x["region"] = pd.Categorical(x["region"], categories=regions)
    y = np.log1p(a["minutes"].to_numpy())
    is_tr = (a["ts"] < val_from).to_numpy()
    is_te = (a["ts"] >= test_from).to_numpy()
    if is_tr.sum() < 100 or is_te.sum() < 10:
        is_tr = is_te = np.ones(len(a), dtype=bool)
    m = lgb.LGBMRegressor(
        n_estimators=100 if quick else 400, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, random_state=42, verbose=-1,
    ).fit(x[is_tr], y[is_tr])  # fmt: skip
    mae = float(np.mean(np.abs(np.expm1(m.predict(x[is_te])) - a["minutes"].to_numpy()[is_te])))
    return m, round(mae, 1)


def backtest_chart(te: pd.DataFrame, p_model, p_base) -> ChartSpec:
    week = ((te["ts"] - te["ts"].min()) // pd.Timedelta(weeks=1)).to_numpy()
    df = pd.DataFrame({"w": week, "y": te["label"].to_numpy(), "m": p_model, "b": p_base})
    df["m"] = (df["m"] - df["y"]) ** 2
    df["b"] = (df["b"] - df["y"]) ** 2
    weekly = df.groupby("w")[["m", "b"]].mean()
    n = len(weekly)

    def series(name: str, col: str) -> Series:
        return Series(
            name=name,
            points=[Point(x=f"Т{i + 1}", y=round(float(v), 4)) for i, v in enumerate(weekly[col])],
        )

    return ChartSpec(
        id="backtest",
        title="Бектест по тижнях",
        description=f"Brier score на кожному з {n} тестових тижнів (менше — краще)",
        kind="line",
        x_label="Тиждень",
        y_label="Brier",
        series=[series("LightGBM", "m"), series("Baseline", "b")],
    )


def roc_chart(y, p_model, p_base) -> ChartSpec:
    def curve(name: str, p) -> Series:
        fpr, tpr, _ = skm.roc_curve(y, p)
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 100)).astype(int))
        pts = [Point(x=round(float(fpr[i]), 4), y=round(float(tpr[i]), 4)) for i in idx]
        return Series(name=name, points=pts)

    return ChartSpec(
        id="roc",
        title="ROC-крива",
        description="Тестові тижні",
        kind="line",
        x_label="FPR",
        y_label="TPR",
        series=[
            curve("LightGBM", p_model),
            curve("Baseline", p_base),
            Series(name="Випадково", points=[Point(x=0, y=0), Point(x=1, y=1)]),
        ],
    )


def build_insights(alerts: pd.DataFrame, recent: pd.DataFrame, regions) -> InsightsResponse:
    rate = recent.groupby(["region", "hour"])["label"].mean()
    cells = [
        Cell(x=f"{h:02d}:00", y=REGIONS[r], v=round(float(rate.get((r, h), 0.0)), 3))
        for r in regions
        for h in range(24)
    ]
    local = alerts["started_at"].dt.tz_convert(TZ)
    per_month = local.dt.strftime("%Y-%m").value_counts().sort_index()
    per_month = per_month.iloc[1:-1]  # first and last months are partial
    minutes = (alerts["finished_at"] - alerts["started_at"]).dt.total_seconds() / 60
    minutes = minutes[~alerts["naive"]].clip(upper=359.9)
    edges = np.arange(0, 380, 20)
    counts, _ = np.histogram(minutes, bins=edges)
    days = local.dt.normalize()
    n_days = days.groupby(local.dt.dayofweek).nunique()
    per_dow = local.dt.dayofweek.value_counts() / n_days.reindex(range(7)).replace(0, 1)
    first, last = local.min(), local.max()
    return InsightsResponse(
        charts=[
            ChartSpec(
                id="region_hour_heatmap",
                title="Тривоги за областю і годиною",
                description=(
                    f"Частка годин із активною тривогою за останні {RECENT_DAYS} днів. "
                    "Дані волонтерського каналу eTryvoga (рівень областей), оновлюються при "
                    "перенавчанні; Луганська область і Крим — постійна тривога, не в даних"
                ),
                kind="heatmap",
                x_label="Година",
                y_label="Область",
                cells=cells,
            ),
            ChartSpec(
                id="alerts_per_month",
                title="Тривоги по місяцях",
                description=f"Кількість тривог по всіх областях, {first:%m.%Y}–{last:%m.%Y}",
                kind="line",
                x_label="Місяць",
                y_label="Тривог",
                series=[
                    Series(
                        name="Тривоги",
                        points=[Point(x=k, y=float(v)) for k, v in per_month.items()],
                    )
                ],
            ),
            ChartSpec(
                id="duration_hist",
                title="Тривалість тривог",
                description="Хвилин; хвіст >6 год обрізано",
                kind="histogram",
                x_label="Хвилин",
                y_label="Тривог",
                series=[
                    Series(
                        name="Тривоги",
                        points=[
                            Point(x=f"{int(e)}–{int(e) + 20}", y=float(c))
                            for e, c in zip(edges[:-1], counts, strict=True)
                        ],
                    )
                ],
            ),
            ChartSpec(
                id="by_weekday",
                title="Тривоги за днем тижня",
                description="Середня кількість на день (усі області)",
                kind="bar",
                x_label="День",
                y_label="Тривог",
                series=[
                    Series(
                        name="Тривоги",
                        points=[
                            Point(x=WEEKDAYS[i], y=round(float(per_dow.get(i, 0)), 1))
                            for i in range(7)
                        ],
                    )
                ],
            ),
        ]
    )


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
