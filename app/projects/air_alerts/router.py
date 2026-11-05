from fastapi import APIRouter

from app.core.contract import PredictResponse
from app.core.project import make_router
from app.projects.air_alerts.schemas import AlertsInput, AlertsResult
from app.projects.air_alerts.service import AirAlertsService


def build() -> APIRouter:
    svc = AirAlertsService()
    r = APIRouter()

    @r.post("/predict", response_model=PredictResponse[AlertsResult])
    def predict(body: AlertsInput) -> PredictResponse[AlertsResult]:
        return svc.predict(body)

    return make_router(svc, "air-alerts", r)
