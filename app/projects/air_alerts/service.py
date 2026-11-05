from functools import cached_property

import numpy as np
import pandas as pd
import shap

from app.core.contract import (
    Contribution,
    Explanation,
    InsightsResponse,
    MetricsResponse,
    PredictResponse,
    SchemaResponse,
)
from app.core.explain import proportional_scale, raw_shap, sigmoid, to_explanation
from app.core.project import Artifacts, artifacts_dir, load_artifacts
from app.core.regions import REGIONS
from app.projects.air_alerts.features import (
    FEATURE_LABELS,
    LAG_FEATURES,
    PERMANENT,
    RATE_DAYS,
    SINCE_CAP_H,
    TZ,
    add_features,
    hourly_grid,
    to_model_frame,
)
from app.projects.air_alerts.schemas import FIELDS, AlertsInput, AlertsResult, HourlyProbability

HORIZON_H = 6
WEEKDAYS = ["понеділок", "вівторок", "середа", "четвер", "пʼятниця", "субота", "неділя"]
MONTHS = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень",
          "вересень", "жовтень", "листопад", "грудень"]  # fmt: skip


class AirAlertsService:
    slug = "air-alerts"

    @cached_property
    def art(self) -> Artifacts:
        return load_artifacts(self.slug)

    @cached_property
    def explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.art.model["proba"])

    @cached_property
    def recent(self) -> pd.DataFrame:
        """Last 90 days of alerts saved by train.py — the history lag features come from."""
        return pd.read_parquet(artifacts_dir(self.slug) / "recent_alerts.parquet")

    @cached_property
    def history_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.art.meta["history_end"]).tz_convert("UTC")

    def schema(self) -> SchemaResponse:
        return FIELDS

    def metrics(self) -> MetricsResponse:
        return self.art.metrics

    def insights(self) -> InsightsResponse:
        return self.art.insights

    def predict(self, inp: AlertsInput) -> PredictResponse[AlertsResult]:
        t = pd.Timestamp(inp.start).tz_convert("UTC").floor("h")
        hours = [t + pd.Timedelta(hours=k) for k in range(HORIZON_H)]
        version = self.art.meta["model_version"]
        if inp.region in PERMANENT:
            return permanent_response(inp.region, hours, version)

        stale = t > self.history_end + pd.Timedelta(hours=1)
        x = self._rows(inp.region, t, stale)
        regions = self.art.meta["categories"]["region"]
        xm = to_model_frame(x, regions)
        proba = self.art.model["proba"].predict_proba(xm)[:, 1]
        duration = float(np.expm1(self.art.model["duration"].predict(
            xm.iloc[:1][self.art.model["duration"].feature_name_])[0]))  # fmt: skip

        base_raw, values = raw_shap(self.explainer, xm.iloc[:1])
        base_p = sigmoid(base_raw)
        explanation = to_explanation(
            xm.iloc[:1],
            values,
            base_p,
            lambda f, v: describe(f, v, stale),
            scale=proportional_scale(float(proba[0]) - base_p, values),
            digits=3,
        )
        result = AlertsResult(
            region=inp.region,
            hourly=[
                HourlyProbability(ts=iso(h), probability=round(float(p), 3))
                for h, p in zip(hours, proba, strict=True)
            ],
            expected_duration_min=round(duration / 5) * 5,
        )
        return PredictResponse(result=result, explanation=explanation, model_version=version)

    def _rows(self, region: str, t: pd.Timestamp, stale: bool) -> pd.DataFrame:
        """Feature rows for t … t+5h. Lags come from history before t only (predictions are not
        fed back); for hours after the first, lags stay as of t and `hours_since_last` grows.
        Beyond the stored history, lags fall back to the region's typical values."""
        regions = self.art.meta["categories"]["region"]
        anchor = min(t, self.history_end + pd.Timedelta(hours=1))
        past = self.recent[self.recent["started_at"] < anchor]
        lo = anchor - pd.Timedelta(days=RATE_DAYS + 3)
        g = add_features(hourly_grid(past, regions, lo, anchor + pd.Timedelta(days=1)))
        g = g[g["region"] == region].set_index("ts")
        now = g.loc[anchor].copy()
        if stale:
            typ = self.art.meta["region_typicals"][region]
            for f in LAG_FEATURES:
                now[f] = typ[f]
            now["active_now"] = 0
        # same-hour rate over the last 28 days of history, by local hour
        hist = g[g.index < anchor]
        hist_hour = hist.index.tz_convert(TZ).hour
        rows = []
        for k in range(HORIZON_H):
            h = t + pd.Timedelta(hours=k)
            local = h.tz_convert(TZ)
            r = now.copy()
            r["hour"], r["dow"], r["month"] = local.hour, local.dayofweek, local.month
            same = hist["label"][hist_hour == local.hour].tail(RATE_DAYS)
            r["rate_28d"] = float(same.mean()) if len(same) else 0.0
            r["hours_since_last"] = min(float(now["hours_since_last"]) + k, SINCE_CAP_H)
            r["region"] = region
            rows.append(r)
        return pd.DataFrame(rows).reset_index(drop=True)


def iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert(TZ).isoformat()


def describe(feature: str, value, stale: bool) -> tuple[str, str]:
    label = FEATURE_LABELS.get(feature, feature)
    typical = " (типово)" if stale and feature in LAG_FEATURES else ""
    if feature == "hour":
        return f"Година {int(value):02d}:00", str(int(value))
    if feature == "dow":
        return f"День тижня: {WEEKDAYS[int(value)]}", str(int(value))
    if feature == "month":
        return f"Місяць: {MONTHS[int(value) - 1]}", str(int(value))
    if feature == "region":
        return f"Область: {REGIONS.get(str(value), value)}", str(value)
    if feature == "rate_28d":
        return f"{label}: {value * 100:.0f} %", f"{value:.2f}"
    if feature == "active_now":
        return f"{label}: {'так' if value else 'ні'}", str(int(value))
    if feature == "hours_since_last":
        v = float(value)
        text = "понад 30 днів" if v >= SINCE_CAP_H else f"{v:.0f} год"
        return f"Від останньої тривоги: {text}{typical}", f"{v:.0f}"
    return f"{label}: {float(value):.0f}{typical}", f"{float(value):.0f}"


def permanent_response(region: str, hours, version: str) -> PredictResponse[AlertsResult]:
    result = AlertsResult(
        region=region,
        hourly=[HourlyProbability(ts=iso(h), probability=1.0) for h in hours],
        expected_duration_min=None,
    )
    explanation = Explanation(
        base_value=1.0,
        contributions=[
            Contribution(
                feature="permanent",
                label="Постійна тривога з 2022 року (не входить у дані моделі)",
                value="",
                impact=0.0,
            )
        ],
    )
    return PredictResponse(result=result, explanation=explanation, model_version=version)
