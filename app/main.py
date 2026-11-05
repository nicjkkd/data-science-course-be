from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.contract import HealthResponse
from app.core.project import ArtifactsMissingError
from app.projects.air_alerts.router import build as build_alerts
from app.projects.car_price.router import build as build_car
from app.projects.phishing.router import build as build_phishing

PROJECTS = ["car-price", "phishing", "air-alerts"]

app = FastAPI(title="Data Analysis Systems API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers are built at import time (so they appear in OpenAPI); each service loads its
# artifacts lazily on the first request.
app.include_router(build_car())
app.include_router(build_phishing())
app.include_router(build_alerts())


@app.exception_handler(ArtifactsMissingError)
def artifacts_missing(_: Request, exc: ArtifactsMissingError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", projects=PROJECTS)
