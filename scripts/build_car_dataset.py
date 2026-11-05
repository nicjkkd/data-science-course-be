"""Build app/projects/car_price/data/processed/listings.parquet from two AUTO.RIA sources.

    uv run python scripts/build_car_dataset.py [--no-dump]

1. API listings collected by scripts/fetch_autoria.py (data/raw/<auto_id>.json): fresh, with
   body type, drive and the seller's description (→ text flags).
2. A public dump of ~216k AUTO.RIA listings scraped in Jan 2024
   (github.com/fuadhuseynov1/Portfolio-Project-AutoRia, auto_ria_cleaned.rar): no body type,
   drive or description. Downloaded and unpacked with `tar` (bsdtar/libarchive reads RAR; on
   Linux install libarchive-tools). The free API key allows only ~30 requests/hour, so the
   dump is what makes a 200k-row dataset possible.

Rows are deduplicated by auto_id (API rows win). Also writes the committed 200-row sample.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.regions import REGIONS  # noqa: E402

DATA = ROOT / "app/projects/car_price/data"
RAW = DATA / "raw"
DUMP_URL = (
    "https://github.com/fuadhuseynov1/Portfolio-Project-AutoRia/raw/main/auto_ria_cleaned.rar"
)
DUMP_CSV = RAW / "auto_ria_cleaned.csv"
DUMP_POSTED = "2024-01-07"  # date of the dump's CSV inside the archive

COLUMNS = ["auto_id", "brand", "model", "year", "mileage_km", "engine_l", "fuel", "gearbox",
           "body", "drive", "region", "price_usd", "damage", "description", "posted_at",
           "source"]  # fmt: skip

FUEL = {
    "Бензин": "petrol",
    "Дизель": "diesel",
    "Газ пропан-бутан / Бензин": "gas",
    "Газ метан / Бензин": "gas",
    "Газ": "gas",
    "Електро": "electric",
    "Гібрид": "hybrid",
    "Гібрид (HEV)": "hybrid",
    "Гібрид (PHEV)": "hybrid",
    "Гібрид (MHEV)": "hybrid",
}
GEARBOX = {
    "Ручна / Механіка": "manual",
    "Механічна": "manual",
    "Автомат": "auto",
    "Варіатор": "auto",
    "Робот": "auto",
    "Роботизована": "auto",
    "Типтронік": "auto",
    "Редуктор": "auto",
}
DRIVE = {"Передній": "fwd", "Задній": "rwd", "Повний": "awd"}
BODY = {
    "Седан": "sedan",
    "Універсал": "wagon",
    "Хетчбек": "hatchback",
    "Позашляховик / Кросовер": "suv",
    "Мінівен": "minivan",
    "Мікровен": "minivan",
    "Купе": "coupe",
    "Пікап": "pickup",
    "Ліфтбек": "liftback",
    "Кабріолет": "convertible",
    "Родстер": "convertible",
    "Лімузин": "sedan",
    "Фастбек": "liftback",
}
# Cyrillic brand names → latin slugs (the rest are lowercased as is)
BRAND_SLUG = {"ВАЗ / Lada": "lada", "ВАЗ": "lada", "ЗАЗ": "zaz", "ГАЗ": "gaz", "УАЗ": "uaz",
              "Москвич/АЗЛК": "moskvich", "ИЖ": "izh", "Богдан": "bogdan"}  # fmt: skip
LABEL_TO_REGION = {v: k for k, v in REGIONS.items()}

_ENGINE = re.compile(r"(\d+(?:[.,]\d+)?)\s*л")


def region_slug(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip()
    if name.startswith("Київ"):
        return "kyiv_city" if "міськрада" in name or name == "Київ" else "kyiv"
    return LABEL_TO_REGION.get(name.removesuffix(" область").strip())


def split_fuel(text: str | None) -> tuple[str | None, float | None]:
    """'Дизель, 2 л.' → ('diesel', 2.0)."""
    if not text:
        return None, None
    text = text.strip()
    kind = text.split(",")[0].strip()
    m = _ENGINE.search(text)
    engine = float(m.group(1).replace(",", ".")) if m else None
    return FUEL.get(kind), engine


def brand_slug(name: str) -> str:
    name = name.strip()
    return BRAND_SLUG.get(name, name.lower())


def from_api() -> pd.DataFrame:
    rows = []
    for f in RAW.glob("*.json"):
        d = json.loads(f.read_text())
        if not d or not d.get("USD"):
            continue
        a = d.get("autoData", {})
        fuel, engine = split_fuel(a.get("fuelName"))
        rows.append(
            {
                "auto_id": int(a.get("autoId") or f.stem),
                "brand": brand_slug(d.get("markName", "")),
                "brand_name": d.get("markName"),
                "model": d.get("modelName"),
                "year": a.get("year"),
                "mileage_km": (a.get("raceInt") or 0) * 1000,
                "engine_l": engine,
                "fuel": fuel,
                "gearbox": GEARBOX.get(a.get("gearboxName", "")),
                "body": BODY.get(d.get("subCategoryName", "")),
                "drive": DRIVE.get(a.get("driveName", "")),
                "region": region_slug((d.get("stateData") or {}).get("regionName")),
                "price_usd": d.get("USD"),
                "damage": "ДТП" in (d.get("infoBarText") or ""),
                "description": a.get("description"),
                "posted_at": d.get("addDate"),
                "source": "api",
            }
        )
    return pd.DataFrame(rows)


def ensure_dump() -> Path:
    if DUMP_CSV.exists():
        return DUMP_CSV
    archive = RAW / "auto_ria_cleaned.rar"
    if not archive.exists():
        print("downloading", DUMP_URL, flush=True)
        r = httpx.get(DUMP_URL, timeout=300, follow_redirects=True)
        r.raise_for_status()
        archive.write_bytes(r.content)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(RAW)], check=True)
    return DUMP_CSV


def from_dump() -> pd.DataFrame:
    df = pd.read_csv(ensure_dump())
    fuel_engine = df["fuel"].map(split_fuel)
    brand = df["manufacturer"].astype(str).str.strip()
    # model_name is "<brand> <model>  <year>"
    model = [
        re.sub(r"\s+\d{4}\s*$", "", m.strip()).removeprefix(b).strip()
        for m, b in zip(df["model_name"].astype(str), brand, strict=True)
    ]
    return pd.DataFrame(
        {
            "auto_id": df["link"].str.extract(r"_(\d+)\.html$")[0].astype(float),
            "brand": brand.map(brand_slug),
            "brand_name": brand,
            "model": model,
            "year": df["release_year"],
            "mileage_km": df["mileage"],
            "engine_l": fuel_engine.map(lambda t: t[1]),
            "fuel": fuel_engine.map(lambda t: t[0]),
            "gearbox": df["transmission"].astype(str).str.strip().map(GEARBOX),
            "body": None,
            "drive": None,
            "region": df["region"].map(region_slug),
            "price_usd": df["price USD($)"],
            "damage": df["accident"].astype(str).str.strip() == "Був в ДТП",
            "description": None,
            "posted_at": DUMP_POSTED,
            "source": "dump",
        }
    )


def main(use_dump: bool) -> None:
    parts = [from_api()]
    if use_dump:
        parts.append(from_dump())
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df.dropna(subset=["price_usd", "year", "brand", "model"])
    df = df[df["model"].astype(str).str.len() > 0]
    df = df.drop_duplicates("auto_id", keep="first")  # API rows come first
    df["engine_l"] = df["engine_l"].where(df["engine_l"].between(0.5, 8))
    df["posted_at"] = pd.to_datetime(df["posted_at"], format="mixed").dt.strftime("%Y-%m-%d")
    df["year"] = df["year"].astype(int)
    df["mileage_km"] = df["mileage_km"].astype(float)
    df["auto_id"] = df["auto_id"].astype("Int64")
    out = DATA / "processed/listings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[[*COLUMNS, "brand_name"]].to_parquet(out)
    print(df["source"].value_counts().to_dict(), "→", out, flush=True)

    api, dump = df[df["source"] == "api"], df[df["source"] == "dump"]
    n_api = min(len(api), 50)
    n_dump = min(len(dump), 200 - n_api)
    sample = pd.concat([api.sample(n_api, random_state=42), dump.sample(n_dump, random_state=42)])
    sample = sample.drop(columns=["auto_id"]).assign(description=sample["description"].str[:200])
    (DATA / "sample").mkdir(parents=True, exist_ok=True)
    sample.to_parquet(DATA / "sample/listings_sample.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dump", action="store_true", help="API listings only")
    main(not ap.parse_args().no_dump)
