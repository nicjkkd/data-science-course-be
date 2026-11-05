from fastapi import APIRouter, HTTPException

from app.core.contract import PredictResponse
from app.core.project import make_router
from app.projects.car_price.schemas import CarInput, CarResult, ExtractRequest, ExtractResponse
from app.projects.car_price.service import CarPriceService


def build() -> APIRouter:
    svc = CarPriceService()
    r = APIRouter()

    @r.post("/predict", response_model=PredictResponse[CarResult])
    def predict(body: CarInput) -> PredictResponse[CarResult]:
        return svc.predict(body)

    @r.post("/extract", response_model=ExtractResponse)
    def extract(body: ExtractRequest) -> ExtractResponse:
        try:
            return svc.extract(body.text)
        except RuntimeError as e:  # missing key
            raise HTTPException(503, str(e)) from e

    return make_router(svc, "car-price", r)
