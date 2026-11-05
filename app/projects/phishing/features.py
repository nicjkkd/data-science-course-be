"""Lexical features of the URL's host. The only feature logic; train.py and service.py import it.

PhiUSIIL's legitimate URLs are all bare homepages (`https://www.<domain>`: 100 % https,
100 % www, no path), so scheme / www / path features would only learn the dataset's collection
artifact. The model therefore sees the host (without `www.`) only; scheme and path signals are
shown as rule-based flags in the URL breakdown (see service.py), not learned.
"""

import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

SUSPICIOUS = [
    "login", "verify", "secure", "update", "account", "confirm", "pay", "bank", "signin",
    "password", "wallet", "support", "auth", "billing", "service",
]  # fmt: skip
BRANDS = [
    "olx", "novaposhta", "privatbank", "privat24", "monobank", "oschad", "rozetka", "google",
    "apple", "microsoft", "paypal", "amazon", "facebook", "instagram", "telegram", "diia",
    "ukrposhta", "netflix", "metamask", "outlook", "office", "binance", "coinbase", "whatsapp",
]  # fmt: skip
# Second-level labels that form a public suffix with a ccTLD: rozetka.com.ua → domain "rozetka".
SECOND_LEVEL = {"com", "net", "org", "gov", "edu", "co", "ac", "in", "kiev", "kyiv", "biz", "info"}

FEATURE_LABELS = {
    "host_len": "Довжина домену",
    "domain_len": "Довжина основного домену",
    "n_subdomains": "Кількість піддоменів",
    "n_digits": "Цифри в домені",
    "digit_ratio": "Частка цифр у домені",
    "n_hyphen": "Дефіси в домені",
    "longest_label": "Найдовша частина домену",
    "is_ip": "IP замість домену",
    "is_punycode": "Punycode (xn--)",
    "tld_len": "Довжина TLD",
    "tld_legit_rate": "Надійність TLD",
    "entropy_host": "Ентропія домену",
    "n_suspicious": "Підозрілі слова в домені",
    "brand_in_sub": "Бренд у піддомені",
    "brand_in_domain": "Бренд усередині чужого домену",
}
NUMERIC = list(FEATURE_LABELS)

_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def split_url(url: str) -> dict:
    u = url.strip()
    u = u if "://" in u else "http://" + u
    p = urlsplit(u)
    host = (p.hostname or "").lower().rstrip(".")
    try:
        port = p.port
    except ValueError:
        port = None
    rest = {"scheme": p.scheme, "host": host, "port": port, "path": p.path, "query": p.query,
            "fragment": p.fragment, "raw": u}  # fmt: skip
    if _IP.match(host):
        return {**rest, "sub": "", "domain": host, "tld": ""}
    labels = host.split(".") if host else []
    n_suffix = 2 if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL else 1
    tld = ".".join(labels[-n_suffix:]) if len(labels) > n_suffix else ""
    domain = labels[-n_suffix - 1] if len(labels) > n_suffix else host
    sub = ".".join(labels[: -n_suffix - 1]) if len(labels) > n_suffix + 1 else ""
    return {**rest, "sub": sub, "domain": domain, "tld": tld}


def strip_www(sub: str) -> str:
    return sub[4:] if sub.startswith("www.") else "" if sub == "www" else sub


def top_label(tld: str) -> str:
    """Reputation is kept per top-level label: com.ua and ua share the .ua rate."""
    return tld.rsplit(".", 1)[-1]


def canonical_host(s: dict) -> str:
    """Host without `www.` — what the model sees."""
    return ".".join(x for x in (strip_www(s["sub"]), s["domain"], s["tld"]) if x)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    return -sum(n / len(s) * math.log2(n / len(s)) for n in c.values())


def url_features(url: str, tld_rates: dict[str, float] | None = None, prior: float = 0.5) -> dict:
    s = split_url(url)
    sub, domain, tld = strip_www(s["sub"]), s["domain"], s["tld"]
    host = canonical_host(s)
    digits = sum(ch.isdigit() for ch in host)
    return {
        "host_len": len(host),
        "domain_len": len(domain),
        "n_subdomains": len(sub.split(".")) if sub else 0,
        "n_digits": digits,
        "digit_ratio": digits / max(len(host), 1),
        "n_hyphen": host.count("-"),
        "longest_label": max((len(x) for x in host.split(".")), default=0),
        "is_ip": int(bool(_IP.match(s["host"]))),
        "is_punycode": int("xn--" in host),
        "tld_len": len(tld),
        "tld_legit_rate": (tld_rates or {}).get(top_label(tld), prior),
        "entropy_host": _entropy(host),
        "n_suspicious": sum(w in host for w in SUSPICIOUS),
        "brand_in_sub": int(any(b in sub for b in BRANDS)),
        "brand_in_domain": int(domain not in BRANDS and any(b in domain for b in BRANDS)),
    }


def build_features(urls: pd.Series, tld_rates: dict[str, float] | None, prior: float = 0.5):
    rows = [url_features(u, tld_rates, prior) for u in urls]
    return pd.DataFrame(rows, index=urls.index)[NUMERIC]
