# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""JSON-safe conversion shared by the RPC reply and saved tool results."""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Best-effort conversion of a tool result to a JSON-safe value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {
            "kind": "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
            "encoding": "base64",
        }
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    return repr(value)
