# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Read-only scanner for external model directories.

Walks configured scan roots and builds a persisted index of the weight files it
finds. For ``.safetensors`` files it reads **only** the JSON header (an 8-byte
little-endian ``uint64`` length prefix followed by that many UTF-8 bytes) and
classifies the file via :mod:`shared.utils.model_signature`. Tensor payloads are
never touched.

Safety guarantees:

* Pickle-based formats (``.ckpt``/``.pt``/``.pth``/``.bin``) are indexed by
  name/size only and are **never** deserialised -- the scanner performs no
  ``torch.load``/``pickle`` call anywhere.
* The header length is validated against a sane ceiling before any read buffer
  is allocated, guarding against corrupted or hostile files.
* The persisted index is keyed by normalised absolute path and stores
  ``mtime_ns`` + size, so unchanged files skip the header read on later scans.

This module must stay dependency-free (standard library only) so it can be
imported and unit tested without the heavy ML runtime.
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Optional

try:  # normal import in a full environment
    from shared.utils import model_signature as ms
except ImportError:  # dependency-light environment (no torch): load sibling directly
    import sys

    _MS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_signature.py")
    _MS_SPEC = importlib.util.spec_from_file_location("model_signature", _MS_PATH)
    ms = importlib.util.module_from_spec(_MS_SPEC)
    sys.modules["model_signature"] = ms  # keep the module discoverable by name
    _MS_SPEC.loader.exec_module(ms)


INDEX_VERSION = 1
INDEX_FILENAME = "model_scan_index.json"

# Sane upper bound for a safetensors JSON header (a real header is a few MB at
# most; the payload dwarfs it). Guards against corrupted/hostile length fields.
DEFAULT_MAX_HEADER_BYTES = 100 * 1024 * 1024

# Upper bound on directory nesting. Combined with the visit guard in
# :func:`walk_files` this makes an infinite walk (symlink/junction loop)
# impossible even where a loop is not detectable as a plain symlink.
DEFAULT_MAX_DEPTH = 32

# Maximum number of warning strings retained in an index document.
_MAX_WARNINGS = 50

SAFETENSORS_EXTENSIONS = (".safetensors", ".sft")
GGUF_EXTENSIONS = (".gguf",)
PICKLE_EXTENSIONS = (".ckpt", ".pt", ".pth", ".bin", ".binn")
ONNX_EXTENSIONS = (".onnx",)
SCANNABLE_EXTENSIONS = SAFETENSORS_EXTENSIONS + GGUF_EXTENSIONS + PICKLE_EXTENSIONS + ONNX_EXTENSIONS

_SKIP_DIR_NAMES = frozenset({"__pycache__"})

FORMAT_SAFETENSORS = "safetensors"
FORMAT_GGUF = "gguf"
FORMAT_PICKLE = "pickle"
FORMAT_ONNX = "onnx"
FORMAT_OTHER = "other"


class SafetensorsHeaderError(Exception):
    """Raised when a safetensors header cannot be read safely."""


def detect_format(path: str) -> str:
    """Classify a file by extension (no file access)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in SAFETENSORS_EXTENSIONS:
        return FORMAT_SAFETENSORS
    if ext in GGUF_EXTENSIONS:
        return FORMAT_GGUF
    if ext in PICKLE_EXTENSIONS:
        return FORMAT_PICKLE
    if ext in ONNX_EXTENSIONS:
        return FORMAT_ONNX
    return FORMAT_OTHER


def read_safetensors_header(path: str, max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES) -> dict:
    """Read and parse only the JSON header of a safetensors file.

    Raises :class:`SafetensorsHeaderError` (with a stable machine-readable code
    as its message) when the file is not a valid, safely-sized header.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:  # pragma: no cover - defensive
        raise SafetensorsHeaderError("stat_failed") from exc

    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise SafetensorsHeaderError("short_prefix")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0:
            raise SafetensorsHeaderError("empty_header")
        if header_len > max_header_bytes:
            raise SafetensorsHeaderError("header_too_large")
        if 8 + header_len > size:
            raise SafetensorsHeaderError("header_exceeds_file")
        blob = handle.read(header_len)
        if len(blob) != header_len:
            raise SafetensorsHeaderError("short_header")

    try:
        parsed = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsHeaderError("bad_json") from exc
    if not isinstance(parsed, dict):
        raise SafetensorsHeaderError("bad_header_type")
    return parsed


def header_keys(header: Mapping) -> list:
    """Return tensor keys from a parsed header (excluding ``__metadata__``)."""
    return [key for key in header if key != "__metadata__"]


def normalise_key(path: str) -> str:
    """Stable index key for a path (absolute, case-normalised on Windows)."""
    return os.path.normcase(os.path.abspath(path))


def inspect_file(
    path: str,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    stat_result: Optional[os.stat_result] = None,
) -> dict:
    """Inspect one file and return its index entry (never loads payloads)."""
    try:
        stat_result = stat_result or os.stat(path)
    except OSError as exc:
        return {
            "path": path,
            "size": None,
            "mtime_ns": None,
            "format": detect_format(path),
            "inspected": False,
            "ok": False,
            "error": f"stat_failed:{exc.errno}",
            "tensor_count": None,
            "signature": None,
            "role": None,
            "confidence": None,
            "matched_by": "none",
            "categories": [],
        }

    fmt = detect_format(path)
    entry = {
        "path": path,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "format": fmt,
        "inspected": False,
        "ok": True,
        "error": None,
        "tensor_count": None,
        "signature": None,
        "role": None,
        "confidence": None,
        "matched_by": "none",
        "categories": [],
    }

    if fmt != FORMAT_SAFETENSORS:
        # Indexed by name only -- pickle formats are deliberately never opened.
        detection = ms.detect_from_filename(path)
        entry.update(
            signature=detection.signature,
            role=detection.role,
            confidence=detection.confidence,
            matched_by=detection.matched_by,
        )
        return entry

    try:
        header = read_safetensors_header(path, max_header_bytes)
    except SafetensorsHeaderError as exc:
        detection = ms.detect_from_filename(path)
        entry.update(
            inspected=True,
            ok=False,
            error=str(exc),
            signature=detection.signature,
            role=detection.role,
            confidence=detection.confidence,
            matched_by=detection.matched_by,
        )
        return entry

    keys = header_keys(header)
    detection = ms.detect_signature(keys, filename=path)
    entry.update(
        inspected=True,
        ok=True,
        tensor_count=len(keys),
        signature=detection.signature,
        role=detection.role,
        confidence=detection.confidence,
        matched_by=detection.matched_by,
    )
    return entry


def _is_reusable(prior: Optional[Mapping], stat_result: os.stat_result, fmt: str) -> bool:
    if not prior or not isinstance(prior, Mapping):
        return False
    if prior.get("size") != stat_result.st_size:
        return False
    if prior.get("mtime_ns") != stat_result.st_mtime_ns:
        return False
    if prior.get("format") != fmt:
        return False
    return prior.get("error") is None


def iter_scan_roots(roots: Any) -> Iterable[tuple]:
    """Yield ``(root_path, category)`` pairs from a list or category mapping."""
    if isinstance(roots, Mapping):
        for category, paths in roots.items():
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            for path in paths or ():
                yield str(path), category
    elif isinstance(roots, (str, os.PathLike)):
        yield str(roots), None
    else:
        for path in roots or ():
            yield str(path), None


def _dir_identity(path: str, stat_result: os.stat_result):
    """Loop-detection identity: ``(st_dev, st_ino)`` when available, else realpath.

    On some Windows filesystems ``st_ino`` is 0, so we fall back to the
    canonicalised real path (junctions/symlinks resolved).
    """
    inode = getattr(stat_result, "st_ino", 0)
    if inode:
        return (getattr(stat_result, "st_dev", 0), inode)
    return os.path.normcase(os.path.realpath(path))


def walk_files(
    root: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    on_warning: Optional[Callable[[str, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    deadline: Optional[float] = None,
) -> Iterable[tuple]:
    """Yield ``(dirpath, filenames)`` under ``root`` with degrade-safe guards.

    Symlinked *files* are included (matching ``os.walk(followlinks=False)``), and
    unreadable directories/files are skipped with a warning rather than aborting
    the whole scan. Directory loops (symlinks/junctions) are stopped by a visited
    identity set plus a hard ``max_depth`` ceiling, and ``deadline`` (a
    ``time.monotonic`` value) bounds the walk.
    """
    def _warn(path: str, reason: str) -> None:
        if on_warning is not None:
            try:
                on_warning(path, reason)
            except Exception:
                pass

    visited = set()
    stack = [(root, 0)]
    while stack:
        if should_cancel is not None and should_cancel():
            return
        if deadline is not None and time.monotonic() > deadline:
            _warn(root, "timeout")
            return
        path, depth = stack.pop()
        try:
            stat_result = os.stat(path)
        except OSError as exc:
            _warn(path, f"stat:{type(exc).__name__}")
            continue
        identity = _dir_identity(path, stat_result)
        if identity in visited:
            _warn(path, "loop")
            continue
        visited.add(identity)
        try:
            with os.scandir(path) as iterator:
                scanned = list(iterator)
        except OSError as exc:
            _warn(path, f"listdir:{type(exc).__name__}")
            continue
        subdirs: list = []
        files: list = []
        for entry in scanned:
            if entry.name in _SKIP_DIR_NAMES:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
                elif entry.is_file(follow_symlinks=True):
                    files.append(entry.name)
            except OSError as exc:  # unreadable entry -> skip, keep scanning
                _warn(entry.path, f"stat:{type(exc).__name__}")
        subdirs.sort(key=str.casefold)
        files.sort(key=str.casefold)
        yield path, files
        if depth >= max_depth:
            if subdirs:
                _warn(path, "max_depth")
            continue
        for subdir in reversed(subdirs):
            stack.append((subdir, depth + 1))


def empty_index() -> dict:
    """Return a fresh, empty index document."""
    return {"version": INDEX_VERSION, "scanned_at": None, "roots": [], "stats": {}, "entries": {}}


def load_index(cache_path: Optional[str]) -> dict:
    """Load a persisted index; returns an empty index when absent/invalid."""
    if not cache_path or not os.path.isfile(cache_path):
        return empty_index()
    try:
        with open(cache_path, encoding="utf-8") as reader:
            data = json.load(reader)
    except (OSError, json.JSONDecodeError):
        return empty_index()
    if not isinstance(data, dict) or data.get("version") != INDEX_VERSION:
        return empty_index()
    if not isinstance(data.get("entries"), dict):
        return empty_index()
    data.setdefault("roots", [])
    data.setdefault("stats", {})
    return data


def save_index(index: Mapping, cache_path: Optional[str]) -> None:
    """Persist an index atomically (temp file + replace)."""
    if not cache_path:
        return
    directory = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(directory, exist_ok=True)
    text = json.dumps(index, indent=1, sort_keys=True)
    fd, temporary = tempfile.mkstemp(prefix=".model_scan.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            writer.write(text)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scan_roots(
    roots: Any,
    cache_path: Optional[str] = None,
    previous: Optional[Mapping] = None,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    extensions: Iterable[str] = SCANNABLE_EXTENSIONS,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[str, dict], None]] = None,
    now: Optional[float] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    deadline: Optional[float] = None,
    on_warning: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """Walk ``roots`` and return an index, reusing unchanged cached entries.

    ``roots`` may be a list of directories or a ``{category: [paths]}`` mapping
    (as produced by ``model_scan_config``). ``previous``/``cache_path`` supply
    the prior index used to skip header reads for unchanged files.

    The walk is degrade-safe: unreadable directories/files are skipped with a
    warning, directory loops are bounded by ``max_depth`` + a visited-identity
    set, and ``deadline`` (a ``time.monotonic`` value) bounds the total scan.
    Warnings and a ``timed_out`` flag are recorded in the returned ``stats``.
    """
    base = previous if previous is not None else load_index(cache_path)
    prior_entries = base.get("entries", {}) if isinstance(base, Mapping) else {}
    allowed = {ext.lower() for ext in extensions}

    entries: dict = {}
    roots_list: list = []
    warnings: list = []
    inspected = reused = scanned_files = skipped = duplicates = 0
    seen_names: set = set()
    cancelled = False
    timed_out = False

    def _record_warning(path: str, reason: str) -> None:
        if len(warnings) < _MAX_WARNINGS:
            warnings.append(f"{reason}: {path}")
        if on_warning is not None:
            try:
                on_warning(path, reason)
            except Exception:
                pass

    def _out_of_time() -> bool:
        return deadline is not None and time.monotonic() > deadline

    for root, category in iter_scan_roots(roots):
        roots_list.append({"path": root, "category": category})
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        if _out_of_time():
            timed_out = True
            _record_warning(root, "timeout")
            break
        if not root or not os.path.isdir(root):
            continue
        for dirpath, filenames in walk_files(
            root, max_depth=max_depth, on_warning=_record_warning,
            should_cancel=should_cancel, deadline=deadline,
        ):
            for name in filenames:
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                if _out_of_time():
                    timed_out = True
                    break
                if os.path.splitext(name)[1].lower() not in allowed:
                    continue
                # De-duplicate by filename: the same name reached through several
                # scan roots (e.g. NTFS junctions to one folder) is a single model.
                name_key = name.casefold()
                if name_key in seen_names:
                    duplicates += 1
                    continue
                full = os.path.join(dirpath, name)
                try:
                    stat_result = os.stat(full)
                except OSError as exc:
                    skipped += 1
                    _record_warning(full, f"stat:{type(exc).__name__}")
                    continue
                scanned_files += 1
                key = normalise_key(full)
                fmt = detect_format(full)
                prior = prior_entries.get(key) if isinstance(prior_entries, Mapping) else None
                if _is_reusable(prior, stat_result, fmt):
                    entry = dict(prior)
                    reused += 1
                else:
                    entry = inspect_file(full, max_header_bytes, stat_result)
                    inspected += 1
                categories = set(entry.get("categories") or ())
                if category:
                    categories.add(category)
                entry["categories"] = sorted(categories)
                entries[key] = entry
                seen_names.add(name_key)
                if on_progress is not None:
                    on_progress(full, entry)
            if cancelled or timed_out:
                break
        if cancelled or timed_out:
            break
        if _out_of_time():
            timed_out = True

    index = {
        "version": INDEX_VERSION,
        "scanned_at": now if now is not None else time.time(),
        "roots": roots_list,
        "stats": {
            "files": scanned_files,
            "inspected": inspected,
            "reused": reused,
            "cancelled": cancelled,
            "timed_out": timed_out,
            "skipped": skipped,
            "duplicates": duplicates,
            "warnings": warnings,
        },
        "entries": entries,
    }
    save_index(index, cache_path)
    return index


def entries_by_signature(index: Mapping, signature: str, role: Optional[str] = None) -> list:
    """Return index entries matching ``signature`` (and optionally ``role``)."""
    matches = []
    for entry in (index.get("entries", {}) if isinstance(index, Mapping) else {}).values():
        if entry.get("signature") != signature:
            continue
        if role is not None and entry.get("role") != role:
            continue
        matches.append(entry)
    return matches


def index_summary(index: Mapping) -> dict:
    """Aggregate counts by role/signature for logging and UI."""
    summary: dict = {"files": 0, "by_role": {}, "by_signature": {}, "errors": 0}
    for entry in (index.get("entries", {}) if isinstance(index, Mapping) else {}).values():
        summary["files"] += 1
        role = entry.get("role") or "unknown"
        summary["by_role"][role] = summary["by_role"].get(role, 0) + 1
        signature = entry.get("signature") or "unknown"
        summary["by_signature"][signature] = summary["by_signature"].get(signature, 0) + 1
        if entry.get("error"):
            summary["errors"] += 1
    return summary
