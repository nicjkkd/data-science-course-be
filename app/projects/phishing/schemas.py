from typing import Literal

from pydantic import BaseModel, Field

from app.core.contract import FieldSpec, SchemaResponse

PartKind = Literal["scheme", "subdomain", "domain", "tld", "port", "path", "query", "fragment"]
# Flag ids must match web/src/lib/phishing-flags.ts.
Flag = Literal[
    "no_https", "brand_in_subdomain", "digits_hyphen_domain", "rare_tld", "login_in_path",
    "ip_host", "many_subdomains", "long_url", "at_symbol", "suspicious_keyword",
]  # fmt: skip


class PhishingInput(BaseModel):
    url: str = Field(min_length=4, max_length=2048)


class UrlPart(BaseModel):
    text: str
    kind: PartKind
    flag: Flag | None


class PhishingResult(BaseModel):
    probability: float = Field(ge=0, le=1)
    verdict: Literal["phishing", "suspicious", "safe"]
    parts: list[UrlPart]


FIELDS = SchemaResponse(
    fields=[
        FieldSpec(
            name="url",
            label="URL",
            type="url",
            help="Вставте посилання повністю, наприклад https://example.com/login",
        )
    ]
)
