from fastapi import APIRouter

from app.core.contract import PredictResponse
from app.core.project import make_router
from app.projects.phishing.schemas import PhishingInput, PhishingResult
from app.projects.phishing.service import PhishingService


def build() -> APIRouter:
    svc = PhishingService()
    r = APIRouter()

    @r.post("/predict", response_model=PredictResponse[PhishingResult])
    def predict(body: PhishingInput) -> PredictResponse[PhishingResult]:
        return svc.predict(body)

    return make_router(svc, "phishing", r)
