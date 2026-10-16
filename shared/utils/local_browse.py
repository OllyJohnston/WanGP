# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Pure, gradio-free helpers for browsing local folders and files.

Kept stdlib-only so it can be unit-tested without torch or gradio, and reused by
the Gradio file picker.
"""
from __future__ import annotations

import os
from pathlib import Path


def normalize_extensions(extensions) -> set[str]:
    """Accept '.safetensors' / 'safetensors' / ['.a', '.b'] and return {'.a', '.b'}."""
    return {f".{str(ext).strip().lower().lstrip('.')}" for ext in (extensions or []) if str(ext).strip()}


def _sorted_names(directory: str) -> list[str]:
    try:
        return sorted(os.listdir(directory), key=str.casefold)
    except OSError:
        return []


def list_subfolders(directory) -> list[tuple[str, str]]:
    """Return (display_name, full_path) for the immediate sub-folders, sorted by name."""
    directory = str(directory or "").strip()
    if not directory or not os.path.isdir(directory):
        return []
    result: list[tuple[str, str]] = []
    for name in _sorted_names(directory):
        full = os.path.join(directory, name)
        try:
            if os.path.isdir(full):
                result.append((name, full))
        except OSError:
            continue
    return result


def list_files(directory, extensions=None) -> list[tuple[str, str]]:
    """Return (display_name, full_path) for the immediate matching files, sorted by name."""
    directory = str(directory or "").strip()
    if not directory or not os.path.isdir(directory):
        return []
    allowed = normalize_extensions(extensions)
    result: list[tuple[str, str]] = []
    for name in _sorted_names(directory):
        full = os.path.join(directory, name)
        try:
            if not os.path.isfile(full):
                continue
        except OSError:
            continue
        if allowed and Path(full).suffix.lower() not in allowed:
            continue
        result.append((name, full))
    return result


def folder_of_value(value) -> str | None:
    """Return the folder implied by a path-like value (a file's folder, or the folder itself).

    URLs and empty values return None.
    """
    value = str(value or "").strip().strip('"')
    if not value or "://" in value:
        return None
    candidate = os.path.abspath(os.path.normpath(os.path.expanduser(value)))
    if os.path.isfile(candidate):
        return os.path.dirname(candidate)
    if os.path.isdir(candidate):
        return candidate
    return None
