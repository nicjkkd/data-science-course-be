import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import shap

from app.core.contract import Contribution, Explanation

MAX_BARS = 8

# describe(feature, raw_value) -> (label shown in the UI, value as a string)
Describe = Callable[[str, Any], tuple[str, str]]


def raw_shap(explainer: shap.TreeExplainer, x: pd.DataFrame) -> tuple[float, np.ndarray]:
    """SHAP for ONE row in the model's raw output space (log-price, log-odds).
    Returns (expected_value, per-feature values). For binary classifiers: the positive class."""
    with warnings.catch_warnings():  # shap warns that LightGBM binary output is a list
        warnings.simplefilter("ignore", UserWarning)
        sv = explainer.shap_values(x)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (rows, features, classes)
        sv = sv[..., -1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return base, sv[0]


def to_explanation(
    x: pd.DataFrame,
    values: np.ndarray,
    base_value: float,
    describe: Describe,
    scale: float = 1.0,
    digits: int | None = None,
    feature_ids: dict[str, str] | None = None,
) -> Explanation:
    """Scale raw SHAP values into display units, keep the MAX_BARS largest and fold the rest
    into «Інші фактори». `feature_ids` renames features (e.g. is_https → no_https flag id)."""
    rnd = (lambda v: round(v, digits)) if digits is not None else (lambda v: v)
    rows = []
    for f, v in zip(x.columns, values, strict=True):
        label, value = describe(f, x.iloc[0][f])
        fid = (feature_ids or {}).get(f, f)
        rows.append(Contribution(feature=fid, label=label, value=value, impact=float(v) * scale))
    rows.sort(key=lambda c: abs(c.impact), reverse=True)
    top, rest = rows[:MAX_BARS], rows[MAX_BARS:]
    if rest:
        top.append(
            Contribution(
                feature="_other",
                label="Інші фактори",
                value="",
                impact=sum(c.impact for c in rest),
            )
        )
    return Explanation(
        base_value=rnd(base_value),
        contributions=[c.model_copy(update={"impact": rnd(c.impact)}) for c in top],
    )


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def proportional_scale(target_delta: float, raw_values: np.ndarray) -> float:
    """SHAP is additive in raw space (log-price / log-odds), not in display units.
    Distribute the display-unit difference (prediction − base) proportionally to raw SHAP."""
    total = float(np.sum(raw_values))
    return target_delta / total if abs(total) > 1e-9 else 0.0
