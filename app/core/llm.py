"""The only module that imports `anthropic`."""

from anthropic import Anthropic
from pydantic import BaseModel

from app.core.config import settings

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024


def extract_structured[T: BaseModel](system: str, user: str, schema: type[T]) -> T:
    """Force a tool call whose input_schema is the pydantic model; return the parsed instance."""
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = Anthropic(api_key=settings.anthropic_api_key)
    tool = {
        "name": "emit",
        "description": "Return the extracted data.",
        "input_schema": schema.model_json_schema(),
    }
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": "emit"},
        messages=[{"role": "user", "content": user}],
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return schema.model_validate(block.input)
