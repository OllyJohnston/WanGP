# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Pure logic for binding a requested (usually remote) weight filename to a
locally scanned copy, so an already-present file is reused instead of
downloaded.

The owner decision is to match by **basename only**, which is deliberately
permissive; the safety therefore lives entirely in the guards below:

* an ambiguous basename (more than one local candidate) never binds;
* ``.safetensors``/``.sft`` candidates are preferred over other containers;
* an optional ``guard`` (the signature/role compatibility check) can veto
  candidates that cannot serve the requested role;
* an optional ``expected_size`` (from remote metadata, when knowable) must match
  exactly.

This module is standard-library only so it can be unit tested without the ML
runtime.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Mapping, Optional

# Container formats preferred when several candidates share a basename.
PREFERRED_EXTENSIONS = (".safetensors", ".sft")

REASON_BOUND = "bound"
REASON_NO_INDEX = "no_index"
REASON_NO_MATCH = "no_match"
REASON_AMBIGUOUS = "ambiguous"
REASON_SIZE_MISMATCH = "size_mismatch"
REASON_INCOMPATIBLE = "incompatible"
REASON_DISABLED = "disabled"


class BindResult(dict):
    """Outcome of a bind attempt (``dict`` for easy logging, with attribute access)."""

    def __init__(self, path: Optional[str], reason: str, candidates: Iterable[str] = (), expected_size: Optional[int] = None):
        super().__init__(path=path, reason=reason, candidates=tuple(candidates), expected_size=expected_size)

    @property
    def path(self) -> Optional[str]:
        return self["path"]

    @property
    def reason(self) -> str:
        return self["reason"]

    @property
    def candidates(self) -> tuple:
        return self["candidates"]

    def __bool__(self) -> bool:
        return self["path"] is not None


def basename_of(requested: str) -> str:
    """Return the plain filename of a URL or path, ignoring query/fragment/alternate suffix."""
    if not requested:
        return ""
    name = str(requested)
    # 'URL|alternate_path' → use the URL part.
    name = name.split("|")[0]
    name = name.split("?")[0].split("#")[0]
    name = name.replace("\\", "/")
    return name.rsplit("/", 1)[-1]


def _extension_of(path: str) -> str:
    return os.path.splitext(path or "")[1].lower()


def entry_path(entry) -> str:
    """Extract a path from a scanner entry (dict) or a plain string."""
    if isinstance(entry, Mapping):
        return entry.get("path") or ""
    return str(entry)


def build_basename_map(entries) -> dict:
    """Map lowercased basename -> list of entries, from scanner index entries.

    Accepts either the ``entries`` mapping (``{normalised_path: entry}``) or a
    plain iterable of entries/strings.
    """
    source = entries.values() if isinstance(entries, Mapping) else entries
    result: dict = {}
    for entry in source or []:
        path = entry_path(entry)
        if not path:
            continue
        key = os.path.basename(path).lower()
        result.setdefault(key, []).append(entry)
    return result


def _sorted_candidates(candidates: list) -> list:
    return sorted(candidates, key=lambda entry: entry_path(entry).casefold())


def _paths(candidates: Iterable) -> tuple:
    return tuple(entry_path(entry) for entry in candidates)


def find_local_bind(
    requested: str,
    basename_map: Optional[Mapping] = None,
    *,
    expected_size: Optional[int] = None,
    role: Optional[str] = None,
    guard: Optional[Callable] = None,
    allow_ambiguous: bool = False,
) -> BindResult:
    """Try to resolve ``requested`` to a single compatible local file.

    Returns a :class:`BindResult`; ``bool(result)`` is True only when a unique,
    compatible candidate was found.
    """
    if not basename_map:
        return BindResult(None, REASON_NO_INDEX)

    name = basename_of(requested).lower()
    if not name:
        return BindResult(None, REASON_NO_MATCH)

    candidates = list(basename_map.get(name, []))
    if not candidates:
        return BindResult(None, REASON_NO_MATCH)

    # 1) Prefer safetensors containers when any are present.
    preferred = [entry for entry in candidates if _extension_of(entry_path(entry)) in PREFERRED_EXTENSIONS]
    if preferred:
        candidates = preferred

    # 2) Optional signature/role compatibility guard (must be able to serve the role).
    if guard is not None:
        compatible = [entry for entry in candidates if guard(requested, entry, role)]
        if not compatible:
            return BindResult(None, REASON_INCOMPATIBLE, _paths(candidates), expected_size)
        candidates = compatible

    # 3) Optional exact size agreement when the remote size is knowable.
    if expected_size is not None:
        matching = [entry for entry in candidates if _entry_size(entry) == expected_size]
        if not matching:
            return BindResult(None, REASON_SIZE_MISMATCH, _paths(candidates), expected_size)
        candidates = matching

    # 4) Ambiguity guard: never bind when more than one candidate survives.
    if len(candidates) > 1 and not allow_ambiguous:
        return BindResult(None, REASON_AMBIGUOUS, _paths(candidates), expected_size)

    chosen = _sorted_candidates(candidates)[0]
    return BindResult(entry_path(chosen), REASON_BOUND, _paths(candidates), expected_size)


def _entry_size(entry) -> Optional[int]:
    if isinstance(entry, Mapping):
        value = entry.get("size")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None
