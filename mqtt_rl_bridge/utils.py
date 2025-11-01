import json
from typing import Any

def safe_json_loads(payload: bytes) -> dict[str, Any]:
    """Robust JSON decode that never raises."""
    try:
        return json.loads(payload.decode())
    except Exception:
        return {}