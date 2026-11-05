# Backend — FastAPI + ML

Serves the API contract in `docs/overview.md` §3 (mirror: `app/core/contract.py` ↔ `web/src/lib/api/contract.ts`) for three projects: `car-price`, `phishing`, `air-alerts`.

## Setup

```bash
cd backend
uv sync
cp .env.example .env   # then fill in the keys
```

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | `POST /api/car-price/extract` only (model `claude-haiku-5-5`, `app/core/llm.py`). Without it `/extract` returns 503. |
| `AUTORIA_API_KEY` | `scripts/fetch_autoria.py` only (data collection). |
| `CORS_ORIGINS` | Comma-separated allowed origins (default `http://localhost:3000`). |
| `ARTIFACTS_DIR` | Optional override for where models are read/written (default `app/projects/<p>/artifacts`). |

LightGBM needs OpenMP: on macOS `brew install libomp` if `import lightgbm` fails.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

`GET /health`, then per project `GET /api/<p>/schema|metrics|insights` and `POST /api/<p>/predict` (+ `POST /api/car-price/extract`). Routers are built at import time; each service loads its artifacts on the first request (missing artifacts → 503 with the train command to run). Interactive docs: `http://localhost:8000/docs`.

Lint: `uv run ruff check . && uv run ruff format --check .`

## Testing

There are no automated tests; test by hand against a running server. The models are committed, so a fresh clone can serve every endpoint without the data files.

### API keys

Both keys are personal; put them in `backend/.env` (gitignored, never commit it). Restart the server after editing `.env`.

| Key | Where to get it | Needed for |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) → API Keys (requires a funded account; each `/extract` call is billed) | `POST /api/car-price/extract` only |
| `AUTORIA_API_KEY` | [developers.ria.com](https://developers.ria.com/) → sign up → the key is shown in the account (free, ~30 requests/hour) | `scripts/fetch_autoria.py` only (collecting new car listings) |

Everything else (`/schema`, `/metrics`, `/insights`, `/predict` for all three projects) works with no keys.

### Without keys

```bash
B=http://localhost:8000
curl $B/health
for p in car-price phishing air-alerts; do
  for e in schema metrics insights; do curl -s -o /dev/null -w "$p/$e %{http_code}\n" $B/api/$p/$e; done
done

curl -X POST $B/api/car-price/predict -H 'Content-Type: application/json' -d '{
  "brand": "volkswagen", "model": "passat", "year": 2016, "mileage_km": 180000, "engine_l": 2.0,
  "fuel": "diesel", "gearbox": "auto", "body": "wagon", "region": "kyiv",
  "condition": "undamaged", "listing_price_usd": 13500}'

curl -X POST $B/api/phishing/predict -H 'Content-Type: application/json' \
  -d '{"url": "http://paypal.secure-login.xyz/verify"}'

curl -X POST $B/api/air-alerts/predict -H 'Content-Type: application/json' \
  -d '{"region": "kharkiv", "start": "2026-11-05T18:00:00+02:00"}'
```

Valid `brand` / `body` / `region` values are listed in `GET /api/car-price/schema`; `start` must include a timezone offset. Or use `http://localhost:8000/docs` → «Try it out».

### With `ANTHROPIC_API_KEY`

```bash
curl -X POST $B/api/car-price/extract -H 'Content-Type: application/json' -d '{
  "text": "Продам Volkswagen Passat B8 2016, 2.0 TDI, автомат, універсал, пробіг 180 тис. км, Київ. Не бита, не фарбована, сервісна історія. Ціна 13500$"}'
```

Returns the parsed form fields, text flags and a confidence. Without the key it returns 503. Each call spends Anthropic credit.

### With `AUTORIA_API_KEY`

```bash
uv run python scripts/fetch_autoria.py   # Ctrl+C after a few listings appear in data/raw/
```

It writes `app/projects/car_price/data/raw/<id>.json`. A missing key exits with `AUTORIA_API_KEY is not set`. To use the new listings, run `build_car_dataset.py` and retrain (see below).

## Data and training

Trained models are committed under `app/projects/<p>/artifacts/`, so the API runs without retraining. Raw and processed data are not committed (`data/raw/`, `data/processed/` are gitignored, ~110 MB in total); each project commits a small `data/sample/`. After cloning, recreate them with the commands below before retraining. Only `train.py` writes artifacts.

`train.py --quick` trains on a 5k subsample with few trees (smoke run); without the full data it falls back to the committed sample.

### Phishing

```bash
uv run python -m app.projects.phishing.train
```

Downloads [UCI PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (235k URLs; label 1 = legitimate, inverted so 1 = phishing) to `data/raw/phiusiil.parquet`. All legitimate URLs in PhiUSIIL are bare homepages (`https://www.<domain>`), so scheme / `www` / path would only learn that artifact: the model uses **host-level lexical features only** (`features.py`), one row per unique host. Scheme and path signals (no HTTPS, `login` in path, `@`, long URL) are rule-based flags in the URL breakdown, not model inputs. TLD reputation is the smoothed share of legitimate registered domains per top-level label, computed on the training split. Models: logistic regression (baseline), random forest, LightGBM (served).

### Air alerts

```bash
uv run python scripts/fetch_alerts.py      # refresh data (the dataset updates daily)
uv run python -m app.projects.air_alerts.train
```

Source: volunteer (eTryvoga) dataset from [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT), oblast level, from 2022-02-25. The official dataset switched to raion-level alerts in Dec 2025, so it is not used. Hourly grid per oblast with leak-free lags (`features.py`); time split: last 8 weeks test, 8 weeks before validation (early stopping). Baseline: 28-day same-hour frequency. A second LightGBM predicts alert duration (MAE in `meta.json`). Training writes the last 90 days of alerts to `artifacts/recent_alerts.parquet`; `/predict` computes lags from it. For hours after the stored history, lag features fall back to the oblast's typical (median) values and are labelled «(типово)» in the explanation — retrain after `fetch_alerts.py` to refresh. Luhansk oblast and Crimea are under permanent alerts that the dataset does not list: `/predict` returns 100 % for them and the model excludes them.

### Car price

```bash
nohup uv run python scripts/fetch_autoria.py &    # optional, slow (see below)
uv run python scripts/build_car_dataset.py         # → data/processed/listings.parquet
uv run python -m app.projects.car_price.train
```

Two AUTO.RIA sources, deduplicated by listing id:
1. **AUTO.RIA developer API** (`fetch_autoria.py`, needs `AUTORIA_API_KEY`): `/auto/search` → ids, `/auto/info` → `data/raw/<id>.json`. Fresh listings with body type, drive and description. The free key allows only ~30 requests per hour (`HourOverlimit`), so the script sleeps through the limit and is resumable; it collects a few hundred listings a day.
2. **Public dump** of ~216k AUTO.RIA listings scraped in January 2024 ([fuadhuseynov1/Portfolio-Project-AutoRia](https://github.com/fuadhuseynov1/Portfolio-Project-AutoRia), `auto_ria_cleaned.rar`, no license stated; used for coursework). `build_car_dataset.py` downloads it and unpacks it with `tar` (bsdtar/libarchive reads RAR; on Linux install `libarchive-tools`). It has no body type, drive or description.

Because body type and drive are missing in almost all rows, `train.py` drops any categorical column that is unknown in more than half the rows (they stay in the form but do not affect the price). Car age is computed from the listing date (2024 for the dump), not from 2026. Text flags (`text_flags.py`: keyword rules with clause-level negation) come from API descriptions only, so the «+ text flags» model currently matches plain LightGBM. Models: ridge (baseline), random forest, LightGBM on `log1p(price)`; LightGBM quantile models (p10/p90) give the range. Early stopping uses a validation split carved from train. Prices are asking prices, not sale prices.

### Explanations

SHAP (`TreeExplainer`) on the served LightGBM, in raw model space (log-price, log-odds); the service converts to USD or probability points proportionally (`app/core/explain.py`), keeps the 8 largest contributions and folds the rest into «Інші фактори».

## Layout

```
app/
├── main.py                 # FastAPI app, CORS, /health, routers
├── core/                   # config, contract, project helpers, explain, evaluate, llm, regions
└── projects/<p>/           # schemas, features, train, service, router, data/, artifacts/
    └── car_price/text_flags.py
scripts/                    # fetch_autoria.py, build_car_dataset.py, fetch_alerts.py
```
