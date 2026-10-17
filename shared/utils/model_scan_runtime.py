# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Runtime for the external model scan feature.

Builds and caches the scan index (via :mod:`model_scanner`), exposes a
conservative basename resolver that plugs into ``files_locator`` as a
last-resort ``filename_resolver``, and turns the guarded
:func:`model_local_bind.find_local_bind` into the download-path interceptor.

The resolver is only active when at least one scan directory is configured
*and* the opt-in ``model_scan_bind_local`` toggle is set, so default behaviour
is unchanged. Everything degrades safely: any failure to build or read the
index simply yields "no bind".

Standard-library only; sibling modules are imported with an importlib fallback
so the module can be unit tested without the ML runtime.
"""
from __future__ import annotations

import html
import importlib
import importlib.util
import os
import sys
import threading
import time
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name):
    """Load a sibling module by path, keeping a single identity under both the
    canonical ``shared.utils.<name>`` key and the short ``<name>`` alias."""
    canonical = "shared.utils." + name
    if canonical in sys.modules:
        return sys.modules[canonical]
    existing = sys.modules.get(name)
    if existing is not None:
        # Adopt an already-loaded short-name module so we never create a shadow copy.
        sys.modules[canonical] = existing
        return existing
    module_path = os.path.join(_THIS_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(canonical, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = module
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _import(name):
    """Prefer the canonical package module (identity with the app), else load by path.

    Only an inability to import the package module triggers the path fallback; a
    real error inside a sibling (e.g. ``SyntaxError``) propagates instead of being
    masked by a shadow module. Both paths leave the module reachable under the
    canonical name and the short-name alias, pointing at the same object.
    """
    try:
        module = importlib.import_module("shared.utils." + name)
    except ImportError:
        module = _load_sibling(name)
    sys.modules.setdefault(name, module)
    return module


model_scanner = _import("model_scanner")
model_local_bind = _import("model_local_bind")
model_scan_config = _import("model_scan_config")
model_architecture_map = _import("model_architecture_map")
files_locator = _import("files_locator")

# Convenience re-exports so callers do not need to import the map themselves.
ROLE_CHECKPOINT = model_architecture_map.ROLE_CHECKPOINT
ROLE_TEXT_ENCODER = model_architecture_map.ROLE_TEXT_ENCODER
ROLE_VAE = model_architecture_map.ROLE_VAE

INDEX_CACHE_FILENAME = "model_scan_index.json"
# The index is a regenerate-able cache, so it is kept inside a dedicated
# sub-folder (ignored by git) rather than beside wgp_config.json.
INDEX_CACHE_DIR = "cache"


def default_cache_path(base_dir: str, cache_dir: str = INDEX_CACHE_DIR) -> str:
    """Return the default index location under ``base_dir`` (``<base_dir>/cache/model_scan_index.json``)."""
    return os.path.join(base_dir, cache_dir, INDEX_CACHE_FILENAME)

# Bound for a synchronous cold scan, so a slow/unreachable network path can never
# hang UI initialisation. A scan that exceeds this keeps running in the
# background; the caller receives whatever has been indexed so far.
DEFAULT_SCAN_TIMEOUT = 20.0

_MAX_WARNINGS = 50

_lock = threading.RLock()
_state = {
    "server_config": None,
    "cache_path": None,
    "scan_paths": None,
    "enabled": False,
    "bind": False,
    "index": None,
    "basename_map": None,
    "resolver_installed": False,
    "timeout": DEFAULT_SCAN_TIMEOUT,
    "scan_thread": None,
    "partial_entries": {},
    "warnings": [],
    "timed_out": False,
    "generation": 0,
}
_hf_repo_size_cache: dict = {}


def reset():
    """Drop all cached state (used by tests and on config reload)."""
    with _lock:
        _state.update(
            server_config=None, cache_path=None, scan_paths=None, enabled=False,
            bind=False, index=None, basename_map=None, resolver_installed=False,
            scan_thread=None, partial_entries={}, warnings=[], timed_out=False,
            generation=_state["generation"] + 1,
        )
        _hf_repo_size_cache.clear()


def configure(server_config, cache_path=None, timeout=None):
    """(Re)configure from a server config mapping; resets the cached index."""
    with _lock:
        scan_paths = model_scan_config.scan_paths_from_server(server_config)
        _state["server_config"] = server_config
        _state["cache_path"] = cache_path
        _state["scan_paths"] = scan_paths
        _state["enabled"] = model_scan_config.has_scan_paths(scan_paths)
        _state["bind"] = model_scan_config.bind_local_from_server(server_config)
        _state["index"] = None
        _state["basename_map"] = None
        _state["warnings"] = []
        _state["timed_out"] = False
        _state["generation"] += 1  # invalidate any in-flight scan
        if timeout is not None:
            _state["timeout"] = timeout
        return status()


def reload(server_config):
    """Re-read the config (e.g. after a settings save) keeping the index cache path."""
    return configure(server_config, cache_path=_state["cache_path"])


def status() -> dict:
    with _lock:
        index = _state["index"]
        stats = index.get("stats", {}) if isinstance(index, dict) else {}
        thread = _state["scan_thread"]
        return {
            "enabled": _state["enabled"],
            "bind": _state["bind"],
            "resolver_installed": _state["resolver_installed"],
            "scan_paths": dict(_state["scan_paths"] or {}),
            "cache_path": _state["cache_path"],
            "warnings": list(_state["warnings"]),
            "timed_out": _state["timed_out"],
            "scanning": bool(thread is not None and thread.is_alive()),
            "stats": dict(stats),
        }


def is_enabled() -> bool:
    return bool(_state["enabled"])


def is_bind_enabled() -> bool:
    return bool(_state["enabled"] and _state["bind"])


def scan_roots() -> dict:
    """Scan-directory lookup categories as ``{category: [paths]}`` (empty entries dropped)."""
    block = _state["scan_paths"] or {}
    roots = {}
    for category in model_scan_config.LOOKUP_CATEGORIES:
        paths = model_scan_config.normalize_path_list(block.get(category))
        if paths:
            roots[category] = paths
    return roots


def _run_scan(roots, cache_path, generation):
    """Worker body: build the index off the UI thread; commit only if current."""
    previous = None
    if cache_path:
        try:
            previous = model_scanner.load_index(cache_path)
        except Exception:
            previous = None

    def on_progress(full, entry):
        with _lock:
            if _state["generation"] == generation:
                _state["partial_entries"][model_scanner.normalise_key(full)] = entry

    def on_warning(path, reason):
        with _lock:
            if _state["generation"] == generation and len(_state["warnings"]) < _MAX_WARNINGS:
                _state["warnings"].append(f"{reason}: {path}")

    try:
        index = model_scanner.scan_roots(
            roots, cache_path=cache_path, previous=previous,
            on_progress=on_progress, on_warning=on_warning,
        )
    except Exception:
        index = None
    with _lock:
        if _state["generation"] != generation:
            return
        if index is not None:
            _state["index"] = index
            _state["basename_map"] = model_local_bind.build_basename_map(index.get("entries", {}))
        _state["scan_thread"] = None


def _start_scan(roots):
    """Start (or join) the background scan for the current configuration."""
    with _lock:
        thread = _state["scan_thread"]
        if thread is not None and thread.is_alive():
            return thread
        _state["partial_entries"] = {}
        _state["warnings"] = []
        _state["index"] = None
        _state["basename_map"] = None
        _state["timed_out"] = False
        _state["generation"] += 1
        thread = threading.Thread(
            target=_run_scan, args=(roots, _state["cache_path"], _state["generation"]), daemon=True
        )
        _state["scan_thread"] = thread
        thread.start()
        return thread


def _partial_index():
    with _lock:
        entries = dict(_state["partial_entries"])
    return {
        "version": model_scanner.INDEX_VERSION,
        "scanned_at": None,
        "roots": [],
        "stats": {"partial": True},
        "entries": entries,
    }


def _current_index():
    with _lock:
        index = _state["index"]
    return index if index is not None else _partial_index()


def get_index(force: bool = False, timeout: Optional[float] = None, wait: bool = True):
    """Return the scan index, building it on first use (or when ``force``).

    The scan always runs on a background thread. With ``wait`` (default) the
    caller waits up to ``timeout`` seconds (configurable, defaults to
    :data:`DEFAULT_SCAN_TIMEOUT`); if it is still running a partial index is
    returned and the scan continues in the background, so a slow path never
    blocks the caller indefinitely.
    """
    with _lock:
        if _state["index"] is not None and not force:
            return _state["index"]
        roots = scan_roots()
        if not roots:
            _state["index"] = model_scanner.empty_index()
            _state["basename_map"] = {}
            return _state["index"]
        thread = _start_scan(roots)
        effective_timeout = _state["timeout"] if timeout is None else timeout
    if not wait:
        return _current_index()
    thread.join(effective_timeout)
    if thread.is_alive():
        with _lock:
            _state["timed_out"] = True
        return _current_index()
    with _lock:
        index = _state["index"]
    return index if index is not None else _current_index()


def get_basename_map() -> dict:
    index = get_index()
    with _lock:
        mapping = _state["basename_map"]
        if mapping is not None:
            return mapping
    mapping = model_local_bind.build_basename_map(index.get("entries", {}))
    with _lock:
        _state["basename_map"] = mapping
    return mapping


def warm(timeout: Optional[float] = None):
    """Build/rebuild the index with a bounded wait (used by the config UI save)."""
    return get_index(force=True, timeout=timeout)


def status_html() -> str:
    """Short HTML summary of the folder scan for the configuration UI status line."""
    info = status()
    if not info["enabled"]:
        return "<div style='color:#888;'>Folder scan: no scan folders configured.</div>"
    stats = info.get("stats") or {}
    parts = []
    if stats.get("files") is not None:
        parts.append(f"{stats.get('files', 0)} file(s)")
    if stats.get("inspected") is not None:
        parts.append(f"{stats.get('inspected', 0)} inspected")
    if stats.get("reused") is not None:
        parts.append(f"{stats.get('reused', 0)} cached")
    parts.append("local reuse ON" if info.get("bind") else "local reuse off")
    summary = ", ".join(parts) if parts else "not scanned yet"
    warnings = list(stats.get("warnings") or []) + list(info.get("warnings") or [])
    colour = "#888"
    extra = ""
    if info.get("timed_out") or stats.get("timed_out"):
        colour = "#c60"
        extra += " &mdash; scan timed out, still running in the background"
    if warnings:
        colour = "#c60"
        extra += f" &mdash; {len(warnings)} warning(s); see the terminal log"
    return f"<div style='color:{colour};'>Folder scan: {html.escape(summary)}{extra}.</div>"


# The scan category a file was found under is authoritative for its role, so we
# never have to guess the type from the filename or tensor signature.
_CATEGORY_ROLES = {
    "checkpoints": ROLE_CHECKPOINT,
    "text_encoders": ROLE_TEXT_ENCODER,
    "vae": ROLE_VAE,
}


def _entry_roles(entry) -> set:
    return {_CATEGORY_ROLES[category] for category in (entry.get("categories") or []) if category in _CATEGORY_ROLES}


def _role_guard(requested, entry, role):
    """Require the local file to live in the scan category that matches the needed role."""
    if role is None:
        return True
    roles = _entry_roles(entry)
    if not roles:
        return True
    return role in roles


def _category_guard(requested, entry, role):
    """Locator-path guard: the request carries no role, so require the entry to be a
    recognised weight file (checkpoints / text_encoders / vae), never an untyped one."""
    return bool(_entry_roles(entry))


def resolve_basename(file_name) -> Optional[str]:
    """Conservative locator resolver: unique basename match only, no download side effects.

    Unlike :func:`attempt_bind` this path has no caller-supplied role, so it is
    deliberately strict: only header-safe weight containers are eligible, the
    entry must belong to a recognised weight category, and when the request is a
    remote URL its size must match before it will bind.
    """
    if not is_bind_enabled():
        return None
    name = model_local_bind.basename_of(file_name)
    if os.path.splitext(name)[1].lower() not in model_local_bind.BINDABLE_EXTENSIONS:
        return None
    mapping = get_basename_map()
    if not mapping:
        return None
    expected_size = remote_file_size(file_name) if _looks_remote(file_name) else None
    result = model_local_bind.find_local_bind(
        file_name, mapping, expected_size=expected_size, guard=_category_guard
    )
    return result.path


def attempt_bind(requested, role=None, expected_size=None):
    """Full guarded bind for the download path. Returns a BindResult."""
    if not is_bind_enabled():
        return model_local_bind.BindResult(None, model_local_bind.REASON_DISABLED)
    mapping = get_basename_map()
    if not mapping:
        return model_local_bind.BindResult(None, model_local_bind.REASON_NO_INDEX)
    if expected_size is None and _looks_remote(requested):
        expected_size = remote_file_size(requested)
    return model_local_bind.find_local_bind(
        requested, mapping, expected_size=expected_size, role=role, guard=_role_guard
    )


def install_locator_resolver():
    """Register the basename resolver on files_locator so nested files resolve everywhere."""
    with _lock:
        if _state["resolver_installed"]:
            return
        files_locator.register_filename_resolver(resolve_basename)
        _state["resolver_installed"] = True


def uninstall_locator_resolver():
    with _lock:
        if not _state["resolver_installed"]:
            return
        files_locator.unregister_filename_resolver(resolve_basename)
        _state["resolver_installed"] = False


def _looks_remote(value) -> bool:
    return isinstance(value, str) and value.startswith("http")


def parse_hf_url(url):
    """Return ``(repo_id, file_path)`` for a huggingface.co file URL, else ``(None, None)``."""
    if not _looks_remote(url):
        return None, None
    base = url.split("|")[0].split("?")[0].split("#")[0]
    prefix = "https://huggingface.co/"
    if not base.startswith(prefix):
        return None, None
    remainder = base[len(prefix):]
    for marker in ("/resolve/", "/blob/"):
        if marker in remainder:
            repo, _, rest = remainder.partition(marker)
            # rest looks like 'main/<path>' (or a revision); drop the first segment.
            parts = rest.split("/", 1)
            file_path = parts[1] if len(parts) == 2 else ""
            return (repo or None), (file_path or None)
    return None, None


def remote_file_size(url) -> Optional[int]:
    """Best-effort remote file size via the HuggingFace API (cached per repo). None if unknown."""
    repo_id, file_path = parse_hf_url(url)
    if not repo_id or not file_path:
        return None
    sizes = _hf_repo_sizes(repo_id)
    if not sizes:
        return None
    return sizes.get(file_path)


def _hf_repo_sizes(repo_id):
    if repo_id in _hf_repo_size_cache:
        return _hf_repo_size_cache[repo_id]
    sizes = None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        sizes = {sibling.rfilename: sibling.size for sibling in info.siblings if sibling.size is not None}
    except Exception:
        sizes = None
    _hf_repo_size_cache[repo_id] = sizes
    return sizes
