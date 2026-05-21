"""Lightweight GoPro metadata extraction from MP4/GPMF payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import av


_CASN_RECORD_RE = re.compile(rb"CASNc\x01\x00\x0f([A-Z0-9]{14})")
_CASN_FALLBACK_RE = re.compile(rb"\b(C[0-9]{13})\b")
_MINF_RECORD_RE = re.compile(rb"MINFc\x01\x00\x1e([^\x00]{1,30})")


@dataclass(frozen=True, slots=True)
class GoProMetadata:
    """Identity metadata embedded by GoPro cameras."""

    serial_number: str | None = None
    model: str | None = None
    firmware: str | None = None


def _read_bytes(path: Path, max_bytes: int = 128 * 1024 * 1024) -> bytes:
    """Read enough of an MP4 to find GoPro atoms without unbounded memory use."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size <= max_bytes:
            return f.read()

        head = f.read(max_bytes // 2)
        f.seek(max(0, size - max_bytes // 2))
        tail = f.read(max_bytes // 2)
        return head + tail


def read_gopro_metadata(path: Path) -> GoProMetadata:
    """Read GoPro serial/model/firmware metadata when present.

    CASN is not exposed by PyAV/ffprobe as ordinary metadata for current GoPro
    files, but it is present in the embedded GPMF/udta payload as a CASN record.
    """
    data = _read_bytes(path)

    serial_number = None
    match = _CASN_RECORD_RE.search(data)
    if match is not None:
        serial_number = match.group(1).decode("ascii", errors="ignore")
    else:
        fallback = _CASN_FALLBACK_RE.search(data)
        if fallback is not None:
            serial_number = fallback.group(1).decode("ascii", errors="ignore")

    model = None
    model_match = _MINF_RECORD_RE.search(data)
    if model_match is not None:
        model = model_match.group(1).decode("ascii", errors="ignore").strip("\x00 ") or None

    firmware = None
    try:
        container = av.open(str(path))
        try:
            firmware = container.metadata.get("firmware")
        finally:
            container.close()
    except Exception:
        firmware = None

    return GoProMetadata(serial_number=serial_number, model=model, firmware=firmware)


def read_gopro_casn(path: Path) -> str | None:
    """Return the GoPro CASN serial number for a video, if present."""
    return read_gopro_metadata(path).serial_number
