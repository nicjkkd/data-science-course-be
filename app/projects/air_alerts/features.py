"""Hourly alert grid and leak-free lag features. The only feature logic; train.py and
service.py import it.

One row per (region, hour). `label` = 1 if an alert overlaps [t, t+1h). Every lag feature
uses only hours strictly before t.
"""

import numpy as np
import pandas as pd

TZ = "Europe/Kyiv"
LAGS_H = (6, 24, 72)
RATE_DAYS = 28
SINCE_CAP_H = 24 * 30
# Permanent alerts not listed in the dataset (Luhansk since 2022-04-04, Crimea since 2022-12-11).
PERMANENT = {"luhansk", "crimea", "sevastopol"}

FEATURE_LABELS = {
    "hour": "Година доби",
    "dow": "День тижня",
    "month": "Місяць",
    "region": "Область",
    "cnt_6h": "Тривоги тут за 6 год",
    "cnt_24h": "Тривоги тут за 24 год",
    "cnt_72h": "Тривоги тут за 72 год",
    "hours_since_last": "Годин від останньої тривоги",
    "nat_cnt_6h": "Тривоги по Україні за 6 год",
    "rate_28d": "Частота для області й години за 28 днів",
    "active_now": "Тривога триває зараз",
}
FEATURES = list(FEATURE_LABELS)
CATEGORICAL = ["region"]
LAG_FEATURES = ["cnt_6h", "cnt_24h", "cnt_72h", "hours_since_last", "nat_cnt_6h", "active_now"]


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _hours_from(ts: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    return ((ts.dt.tz_convert("UTC") - t0) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def hourly_grid(alerts: pd.DataFrame, regions: list[str], start, end) -> pd.DataFrame:
    """Rows for every (region, hour) in [start, end) (UTC hours) with `label` and `starts`
    (alerts starting in that hour). `alerts`: region, started_at, finished_at (tz-aware)."""
    start, end = (_utc(t).floor("h") for t in (start, end))
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(hours)
    frames = []
    for reg in regions:
        a = alerts[alerts["region"] == reg]
        s = np.floor(_hours_from(a["started_at"], start)).astype(np.int64)
        # an alert ending exactly on the hour does not cover that hour
        e = np.ceil(_hours_from(a["finished_at"], start)).astype(np.int64)
        diff = np.zeros(n + 1, dtype=np.int32)
        np.add.at(diff, np.clip(s, 0, n), 1)
        np.add.at(diff, np.clip(e, 0, n), -1)
        label = (np.cumsum(diff)[:n] > 0).astype(np.int8)
        starts = np.bincount(s[(s >= 0) & (s < n)], minlength=n).astype(np.int16)
        frames.append(pd.DataFrame({"ts": hours, "region": reg, "label": label, "starts": starts}))
    return pd.concat(frames, ignore_index=True)


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Calendar + lag features for every grid row, using only hours before the row."""
    g = grid.sort_values(["region", "ts"]).reset_index(drop=True)
    local = g["ts"].dt.tz_convert(TZ)
    g["hour"] = local.dt.hour.astype(np.int8)
    g["dow"] = local.dt.dayofweek.astype(np.int8)
    g["month"] = local.dt.month.astype(np.int8)
    by_region = g.groupby("region", sort=False)
    prev_starts = by_region["starts"].shift(1, fill_value=0)
    for h in LAGS_H:
        g[f"cnt_{h}h"] = (
            prev_starts.groupby(g["region"], sort=False)
            .rolling(h, min_periods=1).sum().reset_index(level=0, drop=True)
        )  # fmt: skip
    # hours since the last alert start strictly before t
    idx = pd.Series(np.arange(len(g)), index=g.index, dtype=float)
    last = idx.where(g["starts"] > 0).groupby(g["region"], sort=False).ffill()
    last = last.groupby(g["region"], sort=False).shift(1)
    g["hours_since_last"] = (idx - last).clip(upper=SINCE_CAP_H).fillna(SINCE_CAP_H)
    # nationwide alert starts in the previous 6 hours
    nat = g.groupby("ts")["starts"].sum().sort_index()
    nat6 = nat.shift(1, fill_value=0).rolling(6, min_periods=1).sum()
    g["nat_cnt_6h"] = g["ts"].map(nat6).astype(float)
    # same region, same local hour, previous 28 days
    g["rate_28d"] = (
        g.groupby(["region", "hour"], sort=False)["label"]
        .transform(lambda s: s.shift(1).rolling(RATE_DAYS, min_periods=1).mean())
        .fillna(0.0)
    )
    g["active_now"] = by_region["label"].shift(1, fill_value=0).astype(np.int8)
    return g


def to_model_frame(g: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    x = g[FEATURES].copy()
    x["region"] = pd.Categorical(x["region"], categories=regions)
    return x
