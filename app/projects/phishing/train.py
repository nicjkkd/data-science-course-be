"""uv run python -m app.projects.phishing.train [--quick]"""

import argparse
import datetime as dt
import io
import subprocess
import zipfile
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.core.contract import (
    ChartSpec,
    Importance,
    InsightsResponse,
    MetricsResponse,
    ModelRow,
    Point,
    Series,
)
from app.core.evaluate import (
    calibration_chart,
    classification_metrics,
    confusion_chart,
    pr_chart,
    roc_chart,
    threshold_chart,
)
from app.core.project import write_artifacts
from app.projects.phishing.features import (
    FEATURE_LABELS,
    build_features,
    split_url,
    strip_www,
    top_label,
)

SLUG = "phishing"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data/raw/phiusiil.parquet"
SAMPLE = HERE / "data/sample/phiusiil_sample.parquet"
# UCI PhiUSIIL (id 967). The direct zip is used because ucimlrepo's API call hung.
ZIP_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
TLD_SMOOTHING = 20


def download() -> pd.DataFrame:
    print("downloading PhiUSIIL …", flush=True)
    r = httpx.get(ZIP_URL, timeout=300, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name), usecols=["URL", "label"])
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW)
    return df


def load(quick: bool) -> pd.DataFrame:
    if RAW.exists():
        raw = pd.read_parquet(RAW)
    else:
        try:
            raw = download()
        except httpx.HTTPError as e:
            if not quick:
                raise
            print(f"download failed ({e}); using the committed sample", flush=True)
            raw = pd.read_parquet(SAMPLE)
    # PhiUSIIL: label 1 = legitimate (verified: all label-1 URLs are https://www.<domain>).
    df = pd.DataFrame({"url": raw["URL"].astype(str), "y": (raw["label"] == 0).astype(int)})
    # The model sees the host only, so URLs of one host are the same sample: keep one per host,
    # drop hosts that appear with both labels.
    parts = df["url"].map(split_url)
    df["host"] = parts.map(
        lambda s: ".".join(x for x in (strip_www(s["sub"]), s["domain"], s["tld"]) if x)
    )
    df = df.drop_duplicates(["host", "y"])
    df = df[~df.duplicated("host", keep=False)].reset_index(drop=True)
    print(f"{len(raw)} URLs → {len(df)} unique hosts; phishing share {df.y.mean():.2f}", flush=True)
    if not SAMPLE.exists() and RAW.exists():
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        raw.sample(500, random_state=42).to_parquet(SAMPLE)
    if quick and len(df) > 5000:
        df = df.sample(5000, random_state=42).reset_index(drop=True)
    return df


def tld_rates(urls: pd.Series, y: np.ndarray) -> tuple[dict[str, float], float]:
    """Smoothed share of legitimate registered domains per TLD — from the training split only.
    Counted per registered domain (domain + TLD), so one free host with hundreds of phishing
    subdomains (e.g. *.zzz.com.ua) counts once instead of swamping its TLD."""
    parts = urls.map(split_url)
    t = pd.DataFrame(
        {
            "tld": parts.map(lambda s: top_label(s["tld"])).to_numpy(),
            "reg": parts.map(lambda s: f"{s['domain']}.{s['tld']}").to_numpy(),
            "legit": 1 - y,
        }
    )
    t = t.groupby(["tld", "reg"], as_index=False)["legit"].max()
    prior = float(t["legit"].mean())
    g = t.groupby("tld")["legit"].agg(["sum", "count"])
    rates = (g["sum"] + TLD_SMOOTHING * prior) / (g["count"] + TLD_SMOOTHING)
    return {k: round(float(v), 4) for k, v in rates.items()}, prior


def main(quick: bool) -> None:
    df = load(quick)
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=df["y"]
    )
    y = df["y"].to_numpy()
    y_tr, y_te = y[idx_tr], y[idx_te]
    rates, prior = tld_rates(df["url"].iloc[idx_tr], y_tr)
    x = build_features(df["url"], rates, prior)
    x_tr, x_te = x.iloc[idx_tr], x.iloc[idx_te]

    models = {
        "Logistic Regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100 if quick else 300, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200 if quick else 800,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            verbose=-1,
        ),
    }
    probas: dict[str, np.ndarray] = {}
    for name, m in models.items():
        m.fit(x_tr, y_tr)
        probas[name] = m.predict_proba(x_te)[:, 1]
    scores = {k: classification_metrics(y_te, p) for k, p in probas.items()}
    best = max(scores, key=lambda k: scores[k]["ROC_AUC"])
    final_name = "LightGBM"  # served model (SHAP TreeExplainer); usually also the best
    final = models[final_name]
    rows = [
        ModelRow(name=k, is_baseline=k.endswith("(baseline)"), is_best=k == best, metrics=m)
        for k, m in scores.items()
    ]

    gain = final.booster_.feature_importance("gain")
    imp = sorted(
        ((f, v) for f, v in zip(x.columns, gain, strict=True) if v > 0), key=lambda t: -t[1]
    )[:15]
    total = sum(v for _, v in imp) or 1
    importances = [
        Importance(feature=f, label=FEATURE_LABELS.get(f, f), value=float(v / total))
        for f, v in imp
    ]

    p = probas[final_name]
    metrics = MetricsResponse(
        task="classification",
        trained_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        n_train=len(idx_tr),
        n_test=len(idx_te),
        models=rows,
        importances=importances,
        charts=[
            roc_chart(y_te, p),
            pr_chart(y_te, p),
            confusion_chart(y_te, p),
            calibration_chart(y_te, p),
            threshold_chart(y_te, p),
        ],
    )
    te = df.iloc[idx_te].assign(p=p)
    short = te[te["url"].str.len() <= 45]
    meta = {
        "model_version": f"{SLUG}@{dt.date.today().isoformat()}",
        "git_sha": git_sha(),
        "features": list(x.columns),
        "tld_rates": rates,
        "tld_prior": prior,
        "thresholds": {"phishing": 0.7, "suspicious": 0.3},
        "examples_safe": short[(short.y == 0) & (short.p < 0.05)]["url"].head(3).tolist(),
        "examples_phishing": short[(short.y == 1) & (short.p > 0.95)]["url"].head(3).tolist(),
    }
    insights = build_insights(df)
    d = write_artifacts(SLUG, final, meta, metrics, insights)
    print({k: round(v["ROC_AUC"], 4) for k, v in scores.items()}, "→", d, flush=True)


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def pct(share: float) -> float:
    return round(float(share) * 100, 1)


def build_insights(df: pd.DataFrame) -> InsightsResponse:
    names = {1: "Фішинг", 0: "Легітимні"}
    parts = df["url"].map(split_url)
    lengths = df["url"].str.len().clip(upper=200)
    edges = np.arange(0, 210, 10)
    length_series = []
    for cls in (1, 0):
        counts, _ = np.histogram(lengths[df.y == cls], bins=edges)
        pts = [Point(x=int(e), y=float(c)) for e, c in zip(edges[:-1], counts, strict=True)]
        length_series.append(Series(name=names[cls], points=pts))

    tlds = parts.map(lambda s: "." + s["tld"] if s["tld"] else "IP")
    g = df.assign(tld=tlds).groupby("tld")["y"].agg(["mean", "count"])
    top = g[g["count"] >= 200].sort_values("mean", ascending=False).head(12)

    subs = parts.map(lambda s: len(strip_www(s["sub"]).split(".")) if strip_www(s["sub"]) else 0)
    subs = subs.clip(upper=4).map(lambda n: "4+" if n == 4 else str(n))
    sub_series = []
    for cls in (1, 0):
        share = subs[df.y == cls].value_counts(normalize=True) * 100
        pts = [Point(x=k, y=round(float(share.get(k, 0)), 1)) for k in ["0", "1", "2", "3", "4+"]]
        sub_series.append(Series(name=names[cls], points=pts))

    https = parts.map(lambda s: s["scheme"] == "https")
    https_pts = [
        Point(x=names[c], y=round(float(https[df.y == c].mean() * 100), 1)) for c in (1, 0)
    ]

    return InsightsResponse(
        charts=[
            ChartSpec(
                id="url_length_hist",
                title="Довжина URL",
                description=(
                    "Обмеження датасету: легітимні URL у PhiUSIIL — лише головні сторінки "
                    "(https://www.…), тому модель навчено тільки на ознаках домену"
                ),
                kind="histogram",
                x_label="Символів (обрізано на 200)",
                series=length_series,
            ),
            ChartSpec(
                id="top_tlds_phishing",
                title="TLD з найбільшою часткою фішингу",
                description="% фішингових серед доменів з цим TLD (≥200 доменів)",
                kind="bar",
                series=[
                    Series(
                        name="Фішинг",
                        points=[
                            Point(x=k, y=round(float(v * 100), 1)) for k, v in top["mean"].items()
                        ],
                    )
                ],
            ),
            ChartSpec(
                id="subdomain_count",
                title="Кількість піддоменів (без www)",
                description="% доменів у кожному класі",
                kind="bar",
                series=sub_series,
            ),
            ChartSpec(
                id="https_share",
                title="Частка HTTPS",
                description="% URL зі схемою https",
                kind="bar",
                series=[Series(name="HTTPS", points=https_pts)],
            ),
        ]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
