from functools import cached_property

import numpy as np
import pandas as pd
import shap

from app.core.contract import InsightsResponse, MetricsResponse, PredictResponse, SchemaResponse
from app.core.explain import proportional_scale, raw_shap, to_explanation
from app.core.llm import extract_structured
from app.core.project import Artifacts, load_artifacts
from app.core.regions import REGIONS
from app.projects.car_price.features import FLAGS, align_categories, build_features
from app.projects.car_price.schemas import (
    BODY_LABELS,
    CONDITION_LABELS,
    DRIVE_LABELS,
    FEATURE_LABELS,
    FUEL_LABELS,
    GEARBOX_LABELS,
    CarInput,
    CarResult,
    ExtractedCar,
    ExtractResponse,
    build_field_specs,
)
from app.projects.car_price.text_flags import flags_from_text

FAIR_BAND_PCT = 7.0


class CarPriceService:
    slug = "car-price"

    @cached_property
    def art(self) -> Artifacts:
        return load_artifacts(self.slug)

    @cached_property
    def explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.art.model["final"])

    def schema(self) -> SchemaResponse:
        return build_field_specs(self.art.meta["brands"], self.art.meta["bodies"])

    def metrics(self) -> MetricsResponse:
        return self.art.metrics

    def insights(self) -> InsightsResponse:
        return self.art.insights

    def _frame(self, inp: CarInput) -> pd.DataFrame:
        meta = self.art.meta
        row = inp.model_dump(exclude={"text_flags", "listing_price_usd"})
        for f in FLAGS:
            row[f] = int(f.removeprefix("flag_") in inp.text_flags)
        x = build_features(
            pd.DataFrame([row]), with_flags=meta["with_flags"], model_vocab=set(meta["model_vocab"])
        )
        return align_categories(x, meta)[meta["features"]]

    def predict(self, inp: CarInput) -> PredictResponse[CarResult]:
        x = self._frame(inp)
        price = float(np.expm1(self.art.model["final"].predict(x)[0]))
        q = self.art.model["quantiles"]
        lo = float(np.expm1(q["0.1"].predict(x)[0]))
        hi = float(np.expm1(q["0.9"].predict(x)[0]))
        lo, hi = min(lo, price), max(hi, price)

        # SHAP is additive in log-price; distribute the USD difference proportionally.
        base_log, values = raw_shap(self.explainer, x)
        base_usd = float(np.expm1(base_log))
        explanation = to_explanation(
            x,
            values,
            base_usd,
            lambda f, v: describe(f, v, inp, self.art.meta["brands"]),
            scale=proportional_scale(price - base_usd, values),
            digits=0,
        )

        delta = verdict = None
        if inp.listing_price_usd:
            delta = (inp.listing_price_usd - price) / price * 100
            verdict = (
                "above" if delta > FAIR_BAND_PCT else "below" if delta < -FAIR_BAND_PCT else "fair"
            )
        result = CarResult(
            price_usd=round(price),
            range_usd=(round(lo), round(hi)),
            listing_price_usd=inp.listing_price_usd,
            delta_pct=round(delta, 1) if delta is not None else None,
            verdict=verdict,
            text_flags=inp.text_flags,
        )
        return PredictResponse(
            result=result, explanation=explanation, model_version=self.art.meta["model_version"]
        )

    def extract(self, text: str) -> ExtractResponse:
        meta = self.art.meta
        system = (
            "Ти витягуєш структуровані дані з українських оголошень про продаж авто. "
            "Повертай лише те, що є в тексті; невідоме лишай null. "
            "brand — slug з переліку: " + ", ".join(meta["brands"]) + ". "
            "model — назва моделі без марки (наприклад «Passat»). "
            "Пробіг у кілометрах (тис. км → ×1000). Ціну в USD; якщо в грн чи EUR — залиш null. "
            "body — одне з: " + ", ".join(BODY_LABELS) + ". "
            "region — slug з переліку: " + ", ".join(meta["regions"]) + ". "
            "condition: undamaged (без пошкоджень), repaired (було ДТП, відремонтоване), "
            "needs_repair. text_flags: accident (ДТП), repaint (фарбування), recent_repair "
            "(нещодавні заміни/ремонт), needs_repair, service_history, imported_us (пригнане зі "
            "США). confidence — наскільки повно й однозначно оголошення описує авто (0–1)."
        )
        ex = extract_structured(system, text, ExtractedCar)
        flags = sorted(set(ex.text_flags) | flags_from_text(text))
        fields = {
            k: v
            for k, v in ex.model_dump(exclude={"text_flags", "confidence"}).items()
            if v is not None
        }
        if fields.get("region") not in REGIONS:
            fields.pop("region", None)
        if fields.get("brand"):
            fields["brand"] = str(fields["brand"]).lower()
        return ExtractResponse(fields=fields, text_flags=flags, confidence=ex.confidence)


def describe(feature: str, value, inp: CarInput, brands: dict[str, str]) -> tuple[str, str]:
    """Bar label (with the value) for one feature, using the user's input for display."""
    raw = getattr(inp, feature, None)
    if feature == "mileage_km":
        return f"Пробіг {int(value):,} км".replace(",", " "), str(int(value))
    if feature == "km_per_year":
        return f"Пробіг за рік {int(value):,} км".replace(",", " "), str(int(value))
    if feature == "age":
        return f"Вік авто {int(value)} р. ({inp.year})", str(int(value))
    if feature == "engine_l":
        return f"Двигун {inp.engine_l:.1f} л", f"{inp.engine_l:.1f}"
    if feature == "brand":
        return f"Марка {brands.get(inp.brand.lower(), inp.brand)}", inp.brand
    if feature == "model":
        rare = " (рідкісна — як «інші»)" if str(value) == "other" else ""
        return f"Модель {inp.model}{rare}", inp.model
    if feature.startswith("flag_"):
        return f"{FEATURE_LABELS[feature]} в описі: {'так' if int(value) else 'ні'}", str(
            int(value)
        )
    names = {
        "fuel": FUEL_LABELS,
        "gearbox": GEARBOX_LABELS,
        "drive": DRIVE_LABELS,
        "body": BODY_LABELS,
        "region": REGIONS,
        "condition": CONDITION_LABELS,
    }
    if feature in names:
        shown = names[feature].get(str(raw), "не вказано") if raw else "не вказано"
        return f"{FEATURE_LABELS[feature]}: {shown.lower() if feature != 'region' else shown}", str(
            raw
        )
    return f"{FEATURE_LABELS.get(feature, feature)}: {value}", str(value)
