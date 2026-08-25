import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


RESPONSE_DIRECTORY = Path(__file__).with_name("responses")


def _to_json_value(response: Any) -> Any:
    """Convert an SDK response to its unwrapped JSON-compatible value."""
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return response
    return response


def save_raw_responses(responses: list[Any]) -> Path:
    """Save all raw LLM responses from one agent execution in one JSON file."""
    if not responses:
        raise ValueError("At least one LLM response is required.")

    RESPONSE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = RESPONSE_DIRECTORY / (
        f"{timestamp}_{uuid4().hex[:8]}.json"
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [_to_json_value(response) for response in responses],
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    return output_path