from functools import cached_property

import pandas as pd
import shap

from app.core.contract import InsightsResponse, MetricsResponse, PredictResponse, SchemaResponse
from app.core.explain import proportional_scale, raw_shap, sigmoid, to_explanation
from app.core.project import Artifacts, load_artifacts
from app.projects.phishing.features import FEATURE_LABELS, build_features, split_url
from app.projects.phishing.schemas import FIELDS, PhishingInput, PhishingResult, UrlPart

LONG_URL = 75
RARE_TLD = 0.3
# model feature → the rule flag shown for it (only when that rule fired)
FEATURE_FLAG = {
    "is_ip": "ip_host",
    "brand_in_sub": "brand_in_subdomain",
    "n_hyphen": "digits_hyphen_domain",
    "tld_legit_rate": "rare_tld",
    "n_subdomains": "many_subdomains",
    "n_suspicious": "suspicious_keyword",
}


class PhishingService:
    slug = "phishing"

    @cached_property
    def art(self) -> Artifacts:
        return load_artifacts(self.slug)

    @cached_property
    def explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.art.model)

    def schema(self) -> SchemaResponse:
        return FIELDS

    def metrics(self) -> MetricsResponse:
        return self.art.metrics

    def insights(self) -> InsightsResponse:
        return self.art.insights

    def predict(self, inp: PhishingInput) -> PredictResponse[PhishingResult]:
        meta = self.art.meta
        x = build_features(pd.Series([inp.url]), meta["tld_rates"], meta["tld_prior"])
        p = float(self.art.model.predict_proba(x)[0, 1])
        th = meta["thresholds"]
        verdict = (
            "phishing" if p >= th["phishing"] else "suspicious" if p >= th["suspicious"] else "safe"
        )

        parts, fired = url_parts(inp.url, x.iloc[0].to_dict())
        # base = sigmoid(E[log-odds]): then sum(raw SHAP) and (p − base) always share a sign,
        # so the proportional conversion to probability points never flips a bar.
        base_raw, values = raw_shap(self.explainer, x)
        base_p = sigmoid(base_raw)
        explanation = to_explanation(
            x,
            values,
            base_p,
            describe,
            scale=proportional_scale(p - base_p, values),
            digits=3,
            feature_ids={f: flag for f, flag in FEATURE_FLAG.items() if flag in fired},
        )
        result = PhishingResult(probability=round(p, 4), verdict=verdict, parts=parts)
        return PredictResponse(
            result=result, explanation=explanation, model_version=meta["model_version"]
        )


def describe(feature: str, value) -> tuple[str, str]:
    label = FEATURE_LABELS.get(feature, feature)
    if feature == "tld_legit_rate":
        return f"{label}: {value * 100:.0f} %", f"{value:.2f}"
    if feature in ("digit_ratio", "entropy_host"):
        return f"{label}: {value:.2f}", f"{value:.2f}"
    if feature.startswith(("is_", "brand_")):
        return f"{label}: {'так' if value else 'ні'}", str(int(value))
    return f"{label}: {int(value)}", str(int(value))


def url_parts(url: str, f: dict) -> tuple[list[UrlPart], set[str]]:
    """Split the URL for the UI and attach rule-based flags to the part each one concerns."""
    s = split_url(url)
    sub_labels = [x for x in s["sub"].split(".") if x and x != "www"]
    path = s["path"].lower()
    login = "login" in path or "signin" in path
    flags: dict[str, str] = {}  # part kind → flag id (first rule wins)

    def flag(kind: str, fid: str, cond: bool) -> None:
        if cond and kind not in flags:
            flags[kind] = fid

    flag("scheme", "no_https", s["scheme"] != "https")
    flag("domain", "ip_host", bool(f["is_ip"]))
    flag("subdomain", "brand_in_subdomain", bool(f["brand_in_sub"]))
    flag("subdomain", "many_subdomains", len(sub_labels) >= 3)
    domain = s["domain"]
    flag("domain", "digits_hyphen_domain", "-" in domain and any(c.isdigit() for c in domain))
    flag("tld", "rare_tld", bool(s["tld"]) and f["tld_legit_rate"] < RARE_TLD)
    flag("path", "login_in_path", login)
    flag("domain", "suspicious_keyword", f["n_suspicious"] > 0)
    flag("path", "at_symbol", "@" in s["raw"])
    flag("path", "long_url", len(s["raw"]) > LONG_URL)

    parts: list[UrlPart] = [
        UrlPart(text=f"{s['scheme']}://", kind="scheme", flag=flags.get("scheme"))
    ]
    if s["sub"]:
        parts.append(UrlPart(text=s["sub"] + ".", kind="subdomain", flag=flags.get("subdomain")))
    parts.append(UrlPart(text=domain, kind="domain", flag=flags.get("domain")))
    if s["tld"]:
        parts.append(UrlPart(text="." + s["tld"], kind="tld", flag=flags.get("tld")))
    if s["port"] is not None:
        parts.append(UrlPart(text=f":{s['port']}", kind="port", flag=None))
    if s["path"] and s["path"] != "/":
        parts.append(UrlPart(text=s["path"], kind="path", flag=flags.get("path")))
    if s["query"]:
        parts.append(UrlPart(text="?" + s["query"], kind="query", flag=None))
    if s["fragment"]:
        parts.append(UrlPart(text="#" + s["fragment"], kind="fragment", flag=None))
    return parts, set(flags.values())
