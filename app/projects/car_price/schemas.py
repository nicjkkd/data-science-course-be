from typing import Literal

from pydantic import BaseModel, Field

from app.core.contract import FieldOption, FieldSpec, SchemaResponse
from app.core.regions import REGIONS

Condition = Literal["undamaged", "repaired", "needs_repair"]
Fuel = Literal["petrol", "diesel", "hybrid", "electric", "gas"]
Gearbox = Literal["manual", "auto"]
Drive = Literal["fwd", "rwd", "awd"]
TextFlag = Literal[
    "accident", "repaint", "recent_repair", "needs_repair", "service_history", "imported_us"
]

TEXT_FLAG_LABELS: dict[str, str] = {
    "accident": "ДТП",
    "repaint": "фарбування",
    "recent_repair": "нещодавній ремонт",
    "needs_repair": "потребує ремонту",
    "service_history": "сервісна історія",
    "imported_us": "пригнане зі США",
}

FEATURE_LABELS: dict[str, str] = {
    "brand": "Марка",
    "model": "Модель",
    "age": "Вік авто",
    "mileage_km": "Пробіг",
    "km_per_year": "Пробіг за рік",
    "engine_l": "Об'єм двигуна",
    "fuel": "Пальне",
    "gearbox": "КПП",
    "body": "Кузов",
    "drive": "Привід",
    "region": "Регіон",
    "condition": "Стан",
    **{f"flag_{k}": v.capitalize() for k, v in TEXT_FLAG_LABELS.items()},
}

FUEL_LABELS = {"petrol": "Бензин", "diesel": "Дизель", "hybrid": "Гібрид", "electric": "Електро",
               "gas": "Газ/бензин"}  # fmt: skip
GEARBOX_LABELS = {"manual": "Механіка", "auto": "Автомат"}
DRIVE_LABELS = {"fwd": "Передній", "rwd": "Задній", "awd": "Повний"}
BODY_LABELS = {"sedan": "Седан", "wagon": "Універсал", "hatchback": "Хетчбек",
               "suv": "Позашляховик / кросовер", "minivan": "Мінівен", "coupe": "Купе",
               "pickup": "Пікап", "liftback": "Ліфтбек", "convertible": "Кабріолет"}  # fmt: skip
CONDITION_LABELS = {"undamaged": "Непошкоджене", "repaired": "Відремонтоване",
                    "needs_repair": "Потребує ремонту"}  # fmt: skip


class CarInput(BaseModel):
    brand: str
    model: str
    year: int = Field(ge=1990, le=2026)
    mileage_km: int = Field(ge=0, le=1_500_000)
    engine_l: float = Field(ge=0.5, le=8)
    fuel: Fuel
    gearbox: Gearbox
    body: str
    drive: Drive | None = None
    region: str
    condition: Condition
    text_flags: list[TextFlag] = Field(default_factory=list)
    listing_price_usd: float | None = Field(default=None, gt=0)


class CarResult(BaseModel):
    price_usd: float
    range_usd: tuple[float, float]
    listing_price_usd: float | None
    delta_pct: float | None
    verdict: Literal["above", "fair", "below"] | None
    text_flags: list[TextFlag]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20, max_length=8000)


class ExtractedCar(BaseModel):
    """What Claude returns. All optional: unknown stays None."""

    brand: str | None = None
    model: str | None = None
    year: int | None = None
    mileage_km: int | None = None
    engine_l: float | None = None
    fuel: Fuel | None = None
    gearbox: Gearbox | None = None
    body: str | None = None
    drive: Drive | None = None
    region: str | None = None
    condition: Condition | None = None
    listing_price_usd: float | None = None
    text_flags: list[TextFlag] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ExtractResponse(BaseModel):
    fields: dict  # CarInput partial
    text_flags: list[TextFlag]
    confidence: float


def _options(labels: dict[str, str]) -> list[FieldOption]:
    return [FieldOption(value=k, label=v) for k, v in labels.items()]


def build_field_specs(brands: dict[str, str], bodies: list[str]) -> SchemaResponse:
    """`brands`: slug → display name (from the training data)."""
    body_opts = [FieldOption(value=b, label=BODY_LABELS.get(b, b)) for b in bodies]
    return SchemaResponse(
        fields=[
            FieldSpec(name="brand", label="Марка", type="select", options=_options(brands)),
            FieldSpec(name="model", label="Модель", type="text"),
            FieldSpec(name="year", label="Рік", type="number", min=1990, max=2026, step=1),
            FieldSpec(
                name="mileage_km",
                label="Пробіг, км",
                type="number",
                min=0,
                max=1_500_000,
                step=1000,
            ),
            FieldSpec(name="engine_l", label="Об'єм, л", type="number", min=0.5, max=8, step=0.1),
            FieldSpec(name="fuel", label="Пальне", type="select", options=_options(FUEL_LABELS)),
            FieldSpec(name="gearbox", label="КПП", type="select", options=_options(GEARBOX_LABELS)),
            FieldSpec(name="body", label="Кузов", type="select", options=body_opts),
            FieldSpec(
                name="drive",
                label="Привід",
                type="select",
                required=False,
                options=_options(DRIVE_LABELS),
            ),
            FieldSpec(name="region", label="Регіон", type="select", options=_options(REGIONS)),
            FieldSpec(
                name="condition",
                label="Стан",
                type="radio",
                default="undamaged",
                options=_options(CONDITION_LABELS),
            ),
            FieldSpec(
                name="listing_price_usd",
                label="Ціна в оголошенні, $",
                type="number",
                required=False,
                min=1,
            ),
        ]
    )
