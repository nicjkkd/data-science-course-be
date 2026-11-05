"""Download air-raid alerts and normalize them to data/processed/alerts.parquet.

    uv run python scripts/fetch_alerts.py

Source: github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (MIT), the volunteer dataset
(eTryvoga): oblast level, from 2022-02-25, updated daily. The official dataset switched to
raion-level alerts in Dec 2025, so the volunteer file is the consistent oblast-level series.
Rows without a recorded end have `naive=True` and finished_at = started_at + 30 min.

Luhansk oblast and Crimea are under permanent alerts (since 2022-04-04 and 2022-12-11) that
the dataset does not list; the API treats them as always-on (see air_alerts/service.py).
"""

from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/volunteer_data_en.csv"
)
DATA = Path(__file__).resolve().parents[1] / "app/projects/air_alerts/data"
TZ = "Europe/Kyiv"

NAME_TO_SLUG = {
    "Vinnytska oblast": "vinnytsia",
    "Volynska oblast": "volyn",
    "Dnipropetrovska oblast": "dnipro",
    "Donetska oblast": "donetsk",
    "Zhytomyrska oblast": "zhytomyr",
    "Zakarpatska oblast": "zakarpattia",
    "Zaporizka oblast": "zaporizhzhia",
    "Ivano-Frankivska oblast": "ivano_frankivsk",
    "Kyivska oblast": "kyiv",
    "Kyiv City": "kyiv_city",
    "Kirovohradska oblast": "kirovohrad",
    "Luhanska oblast": "luhansk",
    "Lvivska oblast": "lviv",
    "Mykolaivska oblast": "mykolaiv",
    "Odeska oblast": "odesa",
    "Poltavska oblast": "poltava",
    "Rivnenska oblast": "rivne",
    "Sumska oblast": "sumy",
    "Ternopilska oblast": "ternopil",
    "Kharkivska oblast": "kharkiv",
    "Khersonska oblast": "kherson",
    "Khmelnytska oblast": "khmelnytskyi",
    "Cherkaska oblast": "cherkasy",
    "Chernivetska oblast": "chernivtsi",
    "Chernihivska oblast": "chernihiv",
    "Autonomous Republic of Crimea": "crimea",
    "Sevastopol": "sevastopol",
}


def main() -> None:
    print("downloading", URL, flush=True)
    r = httpx.get(URL, timeout=300, follow_redirects=True)
    r.raise_for_status()
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "raw/volunteer_data_en.csv").write_text(r.text)
    df = pd.read_csv(StringIO(r.text))
    df = df.dropna(subset=["region", "started_at", "finished_at"])
    unknown = set(df["region"]) - set(NAME_TO_SLUG)
    if unknown:
        raise ValueError(f"unknown region names: {sorted(unknown)}")
    out = pd.DataFrame(
        {
            "region": df["region"].map(NAME_TO_SLUG),
            "started_at": pd.to_datetime(df["started_at"], utc=True).dt.tz_convert(TZ),
            "finished_at": pd.to_datetime(df["finished_at"], utc=True).dt.tz_convert(TZ),
            "naive": df["naive"].astype(str).str.lower() == "true",
        }
    )
    out = out[out["finished_at"] > out["started_at"]]
    out = out.drop_duplicates(["region", "started_at"]).sort_values("started_at")
    out = out.reset_index(drop=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA / "processed/alerts.parquet")
    end = out["started_at"].max()
    sample = out[out["started_at"] > end - pd.Timedelta(days=14)]
    (DATA / "sample").mkdir(parents=True, exist_ok=True)
    sample.to_parquet(DATA / "sample/alerts_sample.parquet")
    print(
        f"{len(out)} alerts, {out['started_at'].min():%Y-%m-%d} … {end:%Y-%m-%d %H:%M}, "
        f"naive {out['naive'].mean():.1%}; sample {len(sample)} rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
