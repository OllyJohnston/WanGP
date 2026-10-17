# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Configuration for external model scan directories.

Defines the ``model_scan_paths`` block stored in ``wgp_config.json``. These are
extra folders WanGP searches (read-only) for diffusion checkpoints, text
encoders and VAEs, so weight files that already exist elsewhere on disk can be
reused without copying them into ``ckpts/``.

The block is intentionally a small, self-describing schema::

    "model_scan_paths": {
        "checkpoints":   ["D:/models/diffusion"],
        "text_encoders": ["D:/models/text_encoders"],
        "vae":           ["D:/models/vae"],
        "loras":         ["D:/models/loras"]
    }

Categories ``checkpoints``, ``text_encoders`` and ``vae`` are appended to the
checkpoint lookup roots (see :func:`checkpoint_lookup_roots`); the *first*
configured checkpoint root keeps its role as the download destination. The
``loras`` category is reserved in the schema but is **not yet consumed** by the
pipeline (native LoRA scanning is not wired up), so it is deliberately not
surfaced in the configuration UI.

This module must stay dependency-free (standard library only) so it can be
imported and unit tested without the heavy ML runtime.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

# Single source of truth for the config key and its categories.
SCAN_PATHS_CONFIG_KEY = "model_scan_paths"
# Boolean toggle: reuse a scanned local weight instead of downloading a remote one.
BIND_LOCAL_CONFIG_KEY = "model_scan_bind_local"
SCAN_CATEGORIES = ("checkpoints", "text_encoders", "vae", "loras")
# Categories that participate in checkpoint filename lookup.
LOOKUP_CATEGORIES = ("checkpoints", "text_encoders", "vae")


def default_scan_paths() -> dict:
    """Return an empty scan-paths block covering every known category."""
    return {category: [] for category in SCAN_CATEGORIES}


def default_bind_local() -> bool:
    """Binding is opt-in: default off so existing behaviour is preserved."""
    return False


def normalize_bind_local(value: Any) -> bool:
    """Coerce a config/UI value into a bool for the bind-local toggle."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default_bind_local()


def bind_local_from_server(server_config: Any) -> bool:
    """Read and normalise the bind-local toggle from a server config mapping."""
    if not isinstance(server_config, Mapping):
        return default_bind_local()
    return normalize_bind_local(server_config.get(BIND_LOCAL_CONFIG_KEY, default_bind_local()))


def has_scan_paths(scan_paths: Any) -> bool:
    """True when a scan-paths block contains at least one configured directory."""
    if not isinstance(scan_paths, Mapping):
        return False
    return any(len(normalize_path_list(scan_paths.get(category))) > 0 for category in SCAN_CATEGORIES)


def _split_entries(value: Any) -> list:
    """Split a config value into a flat list of raw path strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace("\r", "").split("\n")
    if isinstance(value, (list, tuple, set)):
        return [str(one) for one in value]
    return []


def _dedupe_paths(entries: Iterable[Any]) -> list:
    """Strip blanks, drop duplicates (case-insensitive on Windows) and keep order."""
    result: list = []
    seen: set = set()
    for entry in entries:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        # Normalise separators for the dedupe key only; the stored value is kept verbatim.
        key = text.replace("\\", "/").rstrip("/").casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def normalize_path_list(value: Any) -> list:
    """Public helper to turn a Textbox/list value into a clean path list."""
    return _dedupe_paths(_split_entries(value))


def normalize_scan_paths(value: Any) -> dict:
    """Coerce an arbitrary value into a well-formed scan-paths block.

    Unknown categories are dropped and every known category is guaranteed to be
    a list, so callers never need defensive ``.get(..., [])``.
    """
    normalized = default_scan_paths()
    if not isinstance(value, Mapping):
        return normalized
    for category in SCAN_CATEGORIES:
        normalized[category] = normalize_path_list(value.get(category))
    return normalized


def scan_paths_from_server(server_config: Any) -> dict:
    """Read and normalise the block from a server config mapping."""
    if not isinstance(server_config, Mapping):
        return default_scan_paths()
    return normalize_scan_paths(server_config.get(SCAN_PATHS_CONFIG_KEY))


def scan_paths_to_text(scan_paths: Mapping, category: str) -> str:
    """Render one category as newline-separated text for a UI Textbox."""
    if not isinstance(scan_paths, Mapping):
        return ""
    return "\n".join(normalize_path_list(scan_paths.get(category)))


def merged_checkpoint_paths(base_paths: Any, scan_paths: Any) -> list:
    """Return base download roots followed by lookup-only scan directories.

    Order matters: ``files_locator`` treats the first entry as the default
    download destination, so base paths always come first and scan directories
    are appended purely for filename lookup.
    """
    merged = _dedupe_paths(_split_entries(base_paths))
    seen = {_dedupe_key(path) for path in merged}
    block = scan_paths if isinstance(scan_paths, Mapping) else {}
    for category in LOOKUP_CATEGORIES:
        for path in _dedupe_paths(_split_entries(block.get(category))):
            key = _dedupe_key(path)
            if key in seen:
                continue
            seen.add(key)
            merged.append(path)
    return merged


def _dedupe_key(path: str) -> str:
    return str(path).replace("\\", "/").rstrip("/").casefold()


def checkpoint_lookup_roots(server_config: Any, base_paths: Any = None) -> list:
    """Compute the effective checkpoint lookup roots for a server config.

    ``base_paths`` defaults to the config's own ``checkpoints_paths``. When both
    are empty the caller's ``files_locator`` default should be supplied instead
    of relying on a hard-coded fallback here (avoids a second source of truth).
    """
    if base_paths is None and isinstance(server_config, Mapping):
        base_paths = server_config.get("checkpoints_paths")
    return merged_checkpoint_paths(base_paths, scan_paths_from_server(server_config))
