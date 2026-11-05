import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
from fastapi import APIRouter

from app.core.config import settings
from app.core.contract import InsightsResponse, MetricsResponse, SchemaResponse


class ArtifactsMissingError(RuntimeError):
    """Raised on the first request to a project whose train.py has not been run (→ HTTP 503)."""


class ProjectService(Protocol):
    slug: str

    def schema(self) -> SchemaResponse: ...
    def metrics(self) -> MetricsResponse: ...
    def insights(self) -> InsightsResponse: ...


@dataclass(frozen=True)
class Artifacts:
    """Everything train.py writes and service.py reads."""

    model: Any  # fitted estimator(s) (joblib)
    meta: dict[str, Any]  # model_version, feature names, label maps, thresholds
    metrics: MetricsResponse
    insights: InsightsResponse


def artifacts_dir(slug: str) -> Path:
    if settings.artifacts_dir:
        return Path(settings.artifacts_dir) / slug
    return Path(__file__).resolve().parents[1] / "projects" / slug.replace("-", "_") / "artifacts"


def load_artifacts(slug: str) -> Artifacts:
    d = artifacts_dir(slug)
    if not (d / "model.joblib").exists():
        module = slug.replace("-", "_")
        raise ArtifactsMissingError(
            f"no artifacts for {slug}; run: uv run python -m app.projects.{module}.train"
        )
    return Artifacts(
        model=joblib.load(d / "model.joblib"),
        meta=json.loads((d / "meta.json").read_text()),
        metrics=MetricsResponse.model_validate_json((d / "metrics.json").read_text()),
        insights=InsightsResponse.model_validate_json((d / "insights.json").read_text()),
    )


def write_artifacts(
    slug: str, model: Any, meta: dict, metrics: MetricsResponse, insights: InsightsResponse
) -> Path:
    d = artifacts_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib", compress=3)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (d / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (d / "insights.json").write_text(insights.model_dump_json(indent=2))
    return d


def make_router(service: Any, slug: str, predict_route: APIRouter) -> APIRouter:
    """Mount the three generic GET endpoints plus the project's own predict router."""
    router = APIRouter(prefix=f"/api/{slug}", tags=[slug])

    @router.get("/schema", response_model=SchemaResponse)
    def schema() -> SchemaResponse:
        return service.schema()

    @router.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        return service.metrics()

    @router.get("/insights", response_model=InsightsResponse)
    def insights() -> InsightsResponse:
        return service.insights()

    router.include_router(predict_route)
    return router
