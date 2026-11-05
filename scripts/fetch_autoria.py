"""Collect used-car listings from the AUTO.RIA developer API.

    uv run python scripts/fetch_autoria.py [--target 20000]

- /auto/search (passenger cars, USD, customs-cleared, located in Ukraine) pages are fetched
  only when no collected id is waiting; ids go to data/raw/_ids.txt, finished pages to
  data/raw/_pages.txt.
- For each id, /auto/info → data/raw/<auto_id>.json. Existing files are skipped, so the
  script is resumable: stop it any time and run it again.

Needs AUTORIA_API_KEY in backend/.env. The free key allows only ~30 requests per hour: on the
hourly limit (HTTP 429, HourOverlimit) the script sleeps until the next hour and continues.
Run it in the background (nohup) and let it go for days, or use the Kaggle fallback.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE = "https://developers.ria.com"
RAW = Path(__file__).resolve().parents[1] / "app/projects/car_price/data/raw"
SEARCH = {"category_id": 1, "countpage": 100, "currency": 1, "custom": 1, "abroad": 2}
SLEEP_S = 0.25
MAX_RETRIES = 5


def api_key() -> str:
    key = os.environ.get("AUTORIA_API_KEY")
    env = Path(__file__).resolve().parents[1] / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("AUTORIA_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("AUTORIA_API_KEY is not set (backend/.env)")
    return key


class Client:
    def __init__(self, key: str):
        self.key = key
        self.http = httpx.Client(base_url=BASE, timeout=30)
        self.requests = 0

    def get(self, path: str, **params) -> dict | None:
        """GET with retries. Returns None for 4xx; sleeps through the hourly limit."""
        attempt = 0
        while attempt < MAX_RETRIES:
            time.sleep(SLEEP_S)
            try:
                r = self.http.get(path, params={"api_key": self.key, **params})
            except httpx.HTTPError as e:
                attempt += 1
                print(f"  network error: {e}; retrying", flush=True)
                time.sleep(5 * attempt)
                continue
            self.requests += 1
            if r.status_code == 429:
                wait = 3600 - time.time() % 3600 + 60
                now = time.strftime("%H:%M")
                print(f"  {now} hourly limit; sleeping {wait / 60:.0f} min", flush=True)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                attempt += 1
                time.sleep(10 * attempt)
                continue
            if r.status_code >= 400:
                print(f"  HTTP {r.status_code} {path} {params}: {r.text[:200]}", flush=True)
                return None
            return r.json()
        return None


class Ids:
    """Collected listing ids plus the search pages they came from (both persisted)."""

    def __init__(self):
        self.ids_file, self.pages_file = RAW / "_ids.txt", RAW / "_pages.txt"
        self.ids = self.ids_file.read_text().split() if self.ids_file.exists() else []
        self.seen = set(self.ids)
        pages = self.pages_file.read_text().split() if self.pages_file.exists() else []
        self.next_page = max((int(p) for p in pages), default=-1) + 1

    def fetch_page(self, c: Client) -> int:
        data = c.get("/auto/search", **SEARCH, page=self.next_page)
        found = (data or {}).get("result", {}).get("search_result", {}).get("ids", [])
        new = [i for i in found if i not in self.seen]
        self.seen.update(new)
        self.ids += new
        with self.ids_file.open("a") as f:
            f.writelines(f"{i}\n" for i in new)
        with self.pages_file.open("a") as f:
            f.write(f"{self.next_page}\n")
        print(f"search page {self.next_page}: +{len(new)} ids (total {len(self.ids)})", flush=True)
        self.next_page += 1
        return len(found)


def main(target: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    c = Client(api_key())
    ids = Ids()
    saved = {p.stem for p in RAW.glob("*.json")}
    print(f"{len(saved)} listings saved, {len(ids.ids)} ids known, target {target}", flush=True)
    while len(saved) < target:
        todo = [i for i in ids.ids if i not in saved]
        if not todo:
            if ids.fetch_page(c) == 0:
                print("search returned no more ids", flush=True)
                break
            continue
        for auto_id in todo:
            if len(saved) >= target:
                break
            data = c.get("/auto/info", auto_id=auto_id)
            # a missing listing is saved as {} so it is not requested again
            (RAW / f"{auto_id}.json").write_text(json.dumps(data or {}, ensure_ascii=False))
            saved.add(auto_id)
            if len(saved) % 50 == 0:
                print(f"{len(saved)} saved ({c.requests} requests this run)", flush=True)
    print(f"done: {len(saved)} saved, {c.requests} requests this run", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20_000)
    main(ap.parse_args().target)
