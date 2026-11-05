import numpy as np
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve

from app.core.contract import Cell, ChartSpec, Point, Series


def regression_metrics(y, p) -> dict[str, float]:
    """Errors in target units (USD), rounded for display."""
    return {
        "MAE": round(float(skm.mean_absolute_error(y, p))),
        "MedAE": round(float(skm.median_absolute_error(y, p))),
        "RMSE": round(float(np.sqrt(skm.mean_squared_error(y, p)))),
        "R2": round(float(skm.r2_score(y, p)), 4),
        "MAPE": round(float(np.mean(np.abs((y - p) / np.clip(y, 1, None))) * 100), 2),
    }


def classification_metrics(y, proba, threshold=0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "ROC_AUC": float(skm.roc_auc_score(y, proba)),
        "PR_AUC": float(skm.average_precision_score(y, proba)),
        "F1": float(skm.f1_score(y, pred, zero_division=0)),
        "Precision": float(skm.precision_score(y, pred, zero_division=0)),
        "Recall": float(skm.recall_score(y, pred, zero_division=0)),
    }


def probabilistic_metrics(y, proba) -> dict[str, float]:
    return {
        "Brier": float(skm.brier_score_loss(y, proba)),
        "LogLoss": float(skm.log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "ROC_AUC": float(skm.roc_auc_score(y, proba)),
    }


def _thin(n: int, k: int = 100) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, k)).astype(int))


def roc_chart(y, proba) -> ChartSpec:
    fpr, tpr, _ = skm.roc_curve(y, proba)
    idx = _thin(len(fpr))
    return ChartSpec(
        id="roc",
        title="ROC-крива",
        kind="line",
        x_label="FPR",
        y_label="TPR",
        series=[
            Series(name="Модель", points=[Point(x=float(fpr[i]), y=float(tpr[i])) for i in idx]),
            Series(name="Випадково", points=[Point(x=0, y=0), Point(x=1, y=1)]),
        ],
    )


def pr_chart(y, proba) -> ChartSpec:
    p, r, _ = skm.precision_recall_curve(y, proba)
    idx = _thin(len(p))
    return ChartSpec(
        id="pr",
        title="Precision–Recall",
        kind="line",
        x_label="Recall",
        y_label="Precision",
        series=[Series(name="Модель", points=[Point(x=float(r[i]), y=float(p[i])) for i in idx])],
    )


def confusion_chart(y, proba, labels=("Легітимний", "Фішинг"), threshold=0.5) -> ChartSpec:
    cm = skm.confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1])
    cells = [
        Cell(x=f"Прогноз: {labels[j]}", y=f"Факт: {labels[i]}", v=float(cm[i, j]))
        for i in range(2)
        for j in range(2)
    ]
    return ChartSpec(
        id="confusion", title="Матриця помилок (поріг 0.5)", kind="heatmap", cells=cells
    )


def calibration_chart(y, proba, bins=10) -> ChartSpec:
    frac, mean = calibration_curve(y, proba, n_bins=bins, strategy="quantile")
    return ChartSpec(
        id="calibration",
        title="Калібрування",
        description="Чи означає «30 %» справді 30 %",
        kind="line",
        x_label="Прогноз",
        y_label="Факт",
        series=[
            Series(
                name="Модель",
                points=[Point(x=float(m), y=float(f)) for m, f in zip(mean, frac, strict=True)],
            ),
            Series(name="Ідеал", points=[Point(x=0, y=0), Point(x=1, y=1)]),
        ],
    )


def threshold_chart(y, proba) -> ChartSpec:
    ts = np.round(np.arange(0.05, 1.0, 0.05), 2)
    prec, rec, f1 = [], [], []
    for t in ts:
        pred = (proba >= t).astype(int)
        prec.append(skm.precision_score(y, pred, zero_division=0))
        rec.append(skm.recall_score(y, pred, zero_division=0))
        f1.append(skm.f1_score(y, pred, zero_division=0))

    def mk(name: str, vals: list[float]) -> Series:
        return Series(
            name=name, points=[Point(x=float(t), y=float(v)) for t, v in zip(ts, vals, strict=True)]
        )

    return ChartSpec(
        id="threshold_curve",
        title="Метрики за порогом",
        kind="line",
        x_label="Поріг",
        series=[mk("precision", prec), mk("recall", rec), mk("f1", f1)],
    )


def histogram(values, bins=30, *, id, title, x_label, name="Кількість") -> ChartSpec:
    counts, edges = np.histogram(values, bins=bins)
    return ChartSpec(
        id=id,
        title=title,
        kind="histogram",
        x_label=x_label,
        series=[
            Series(
                name=name,
                points=[
                    Point(x=round(float(e), 2), y=float(c))
                    for e, c in zip(edges[:-1], counts, strict=True)
                ],
            )
        ],
    )


def pred_vs_actual_chart(y, p, n=400, seed=42) -> ChartSpec:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    return ChartSpec(
        id="pred_vs_actual",
        title="Прогноз vs факт",
        kind="scatter",
        x_label="Факт",
        y_label="Прогноз",
        series=[Series(name="Тест", points=[Point(x=float(y[i]), y=float(p[i])) for i in idx])],
    )
