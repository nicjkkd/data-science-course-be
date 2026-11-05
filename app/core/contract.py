"""The API contract — mirror of web/src/lib/api/contract.ts.

zod's `.optional()` rejects `null`, so models whose optional fields are `.optional()` on the
frontend (FieldSpec, ChartSpec) drop None-valued keys when serialized. Fields that are
`.nullable()` on the frontend (e.g. CarResult.verdict) keep their nulls.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer

FieldType = Literal["number", "select", "radio", "text", "url", "date"]
TaskType = Literal["regression", "classification", "timeseries"]
ChartKind = Literal["line", "bar", "histogram", "scatter", "heatmap"]


class OmitNone(BaseModel):
    @model_serializer(mode="wrap")
    def _omit_none(self, handler: Any) -> dict[str, Any]:
        return {k: v for k, v in handler(self).items() if v is not None}


class FieldOption(BaseModel):
    value: str
    label: str


class FieldSpec(OmitNone):
    name: str
    label: str
    type: FieldType
    required: bool = True
    options: list[FieldOption] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: str | float | None = None
    help: str | None = None


class SchemaResponse(BaseModel):
    fields: list[FieldSpec]


class Contribution(BaseModel):
    feature: str
    label: str
    value: str
    impact: float


class Explanation(BaseModel):
    base_value: float
    contributions: list[Contribution]


class PredictResponse[R: BaseModel](BaseModel):
    result: R
    explanation: Explanation
    model_version: str


class Point(BaseModel):
    x: float | str
    y: float


class Series(BaseModel):
    name: str
    points: list[Point]


class Cell(BaseModel):
    x: str
    y: str
    v: float


class ChartSpec(OmitNone):
    id: str
    title: str
    description: str | None = None
    kind: ChartKind
    x_label: str | None = None
    y_label: str | None = None
    series: list[Series] = Field(default_factory=list)
    cells: list[Cell] | None = None


class ModelRow(BaseModel):
    name: str
    is_baseline: bool
    is_best: bool
    metrics: dict[str, float]


class Importance(BaseModel):
    feature: str
    label: str
    value: float


class MetricsResponse(BaseModel):
    task: TaskType
    trained_at: str
    n_train: int
    n_test: int
    models: list[ModelRow]
    importances: list[Importance]
    charts: list[ChartSpec]


class InsightsResponse(BaseModel):
    charts: list[ChartSpec]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    projects: list[str]
