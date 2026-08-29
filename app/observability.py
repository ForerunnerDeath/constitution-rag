import json
import logging
from typing import Any, cast


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event_data: object = getattr(
            record,
            "event_data",
            None,
        )

        if isinstance(event_data, dict):
            payload = cast(dict[str, Any], event_data)
        else:
            payload = {
                "message": record.getMessage(),
            }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


logger = logging.getLogger("constitution_rag")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "event": event,
        **fields,
    }

    logger.info(
        "",
        extra={
            "event_data": payload,
        },
    )
