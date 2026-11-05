from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.core.contract import FieldOption, FieldSpec, SchemaResponse
from app.core.regions import REGIONS

# Crimea and Sevastopol have no alert records at all (permanent alert, not in the dataset).
FORM_REGIONS = {k: v for k, v in REGIONS.items() if k not in ("crimea", "sevastopol")}


class AlertsInput(BaseModel):
    region: str
    start: datetime  # hour precision; must carry a timezone offset

    @field_validator("region")
    @classmethod
    def known_region(cls, v: str) -> str:
        if v not in REGIONS:
            raise ValueError(f"unknown region: {v}")
        return v

    @field_validator("start")
    @classmethod
    def has_offset(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("start needs a timezone offset, e.g. 2026-10-18T14:00:00+03:00")
        return v


class HourlyProbability(BaseModel):
    ts: str
    probability: float = Field(ge=0, le=1)


class AlertsResult(BaseModel):
    region: str
    hourly: list[HourlyProbability]
    expected_duration_min: float | None


FIELDS = SchemaResponse(
    fields=[
        FieldSpec(
            name="region",
            label="Область",
            type="select",
            options=[FieldOption(value=k, label=v) for k, v in FORM_REGIONS.items()],
        ),
        FieldSpec(name="start", label="Початок (година)", type="date"),
    ]
)
