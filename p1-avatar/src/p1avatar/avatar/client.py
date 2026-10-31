import logging
from pathlib import Path

import httpx

from ..config import get_settings

log = logging.getLogger(__name__)


def render_avatar(wav_path: Path, out_path: Path, timeout: float = 300.0) -> Path | None:
    """Send answer audio to ditto_service and save the lip-synced mp4.

    Returns None when the avatar is disabled or the service is down, so the kiosk falls back to
    audio only instead of failing the whole turn.
    """
    s = get_settings()
    if not s.avatar_enabled:
        return None
    try:
        with open(wav_path, "rb") as f:
            r = httpx.post(f"{s.ditto_url}/render", files={"audio": (wav_path.name, f, "audio/wav")},
                           timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("Avatar render skipped: %s", e)
        return None
    out_path.write_bytes(r.content)
    return out_path


def avatar_available() -> bool:
    s = get_settings()
    if not s.avatar_enabled:
        return False
    try:
        return httpx.get(f"{s.ditto_url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False
