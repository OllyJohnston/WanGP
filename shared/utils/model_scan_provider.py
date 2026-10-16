# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Turn scanned local checkpoints into selectable, in-memory model definitions.

The provider is deliberately stdlib-only and free of any WanGP import so it can
be unit tested without torch. Everything that needs the runtime (handler
inheritance via ``init_model_def``) is injected through the ``initialiser``
callable.

Scanned entries are shaped exactly like finetunes: they carry a synthetic
``path`` of ``finetunes/<id>.json``. This makes the whole existing finetune
machinery (hierarchy placement, ``*`` marker, delete guard, editor) treat them
correctly, and it means "promote to disk" is simply writing that same file.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from typing import Callable, Dict, Mapping, Optional

try:  # pragma: no cover - exercised indirectly
    from shared.utils import model_architecture_map as amap
except Exception:  # pragma: no cover - importlib fallback for tests without package context
    _AMAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_architecture_map.py")
    _AMAP_NAME = "model_architecture_map"
    if _AMAP_NAME in sys.modules:
        amap = sys.modules[_AMAP_NAME]
    else:
        _spec = importlib.util.spec_from_file_location(_AMAP_NAME, _AMAP_PATH)
        amap = importlib.util.module_from_spec(_spec)
        sys.modules[_AMAP_NAME] = amap
        _spec.loader.exec_module(amap)

ROLE_CHECKPOINT = amap.ROLE_CHECKPOINT

# Marker written into ``model_def["metadata"]`` so the UI can recognise entries
# that only exist because of a directory scan.
SCANNED_METADATA_KEY = "scanned"
# Reserved key holding the real absolute path of the discovered weight file.
SCAN_PATH_KEY = "scan_path"
# Namespace for dynamic ids so they can never shadow a disk definition by accident.
SCANNED_ID_PREFIX = "scan_"
# Scanned entries masquerade as finetunes; this is the folder their synthetic path lives in.
SCANNED_PATH_DIR = "finetunes"
# Filename appended to a scanned entry id when two scanned files share a stem.
_MAX_ID_ATTEMPTS = 100


def slugify(value) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value


def dynamic_id(stem, prefix: str = SCANNED_ID_PREFIX) -> str:
    slug = slugify(stem)
    return f"{prefix}{slug}" if slug else ""


def scanned_json_path(model_id: str) -> str:
    """Synthetic (not yet written) finetune path for a scanned model id."""
    return os.path.join(SCANNED_PATH_DIR, slugify(model_id) + ".json")


def is_scanned_model_def(model_def) -> bool:
    if not isinstance(model_def, dict):
        return False
    if SCAN_PATH_KEY in model_def:
        return True
    metadata = model_def.get("metadata")
    return bool(isinstance(metadata, dict) and metadata.get(SCANNED_METADATA_KEY))


def _is_checkpoint_entry(entry) -> bool:
    categories = entry.get("categories") or []
    return "checkpoints" in categories


def _allocate_id(base: str, taken: set) -> Optional[str]:
    if base not in taken:
        return base
    for attempt in range(2, _MAX_ID_ATTEMPTS):
        candidate = f"{base}_{attempt}"
        if candidate not in taken:
            return candidate
    return None


def _human_name(stem: str) -> str:
    return str(stem or "").strip()


def model_infos_from_models_def(models_def: Mapping[str, Mapping]) -> Dict[str, "amap.DefinitionInfo"]:
    """Build the architecture index the resolver expects from a live ``models_def``."""
    infos = {}
    for model_type, model_def in models_def.items():
        if not isinstance(model_def, dict):
            continue
        infos[model_type] = amap.DefinitionInfo(
            model_type=model_type,
            architecture=model_def.get("architecture"),
            path=model_def.get("path"),
        )
    return infos


def build_scanned_model_defs(
    index,
    model_infos: Mapping[str, "amap.DefinitionInfo"],
    disk_model_types=None,
    *,
    prefix: str = SCANNED_ID_PREFIX,
    initialiser: Optional[Callable[[str, dict], dict]] = None,
) -> Dict[str, dict]:
    """Return ``{model_id: model_def}`` for scanned checkpoints resolvable to an installed model.

    Only entries found in the ``checkpoints`` scan category whose signature maps
    to an installed architecture (``available``) are registered. Disk ids always
    win: a scanned entry whose id would collide with ``disk_model_types`` is
    skipped. Two scanned files sharing a stem are disambiguated with a numeric
    suffix.
    """
    entries = index.get("entries", {}) if isinstance(index, dict) else index
    if isinstance(entries, dict):
        entries = entries.values()
    disk = set(disk_model_types or ())
    taken = set(disk)
    result: Dict[str, dict] = {}
    for entry in sorted(entries, key=lambda item: str(item.get("path", "")).casefold()):
        path = entry.get("path")
        if not path or not _is_checkpoint_entry(entry):
            continue
        resolution = amap.resolve_checkpoint(
            entry.get("signature"), ROLE_CHECKPOINT, model_infos, filename=path
        )
        if resolution.status != amap.STATUS_AVAILABLE or not resolution.architecture:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        base_id = dynamic_id(stem, prefix)
        if not base_id or base_id in disk:
            # No usable id, or a real definition already owns this id -> disk wins.
            continue
        model_id = _allocate_id(base_id, taken)
        if model_id is None:
            continue
        taken.add(model_id)
        absolute = os.path.abspath(path)
        model_def = {
            "name": _human_name(stem),
            "architecture": resolution.architecture,
            "description": f"Local checkpoint discovered in a scanned folder: {absolute}",
            "URLs": [absolute],
            "finetune_source_model": resolution.model_type,
            SCAN_PATH_KEY: absolute,
        }
        if initialiser is not None:
            model_def = initialiser(model_id, model_def)
        result[model_id] = model_def
    return result


def merge_scanned_models(models_def: Mapping[str, dict], scanned_defs: Mapping[str, dict]) -> Dict[str, dict]:
    """Merge scanned defs into a model registry; existing (disk) entries always win."""
    merged = dict(models_def)
    for model_id, model_def in scanned_defs.items():
        merged.setdefault(model_id, model_def)
    return merged


def strip_scanned_models(models_def) -> int:
    """Remove every previously injected scanned entry from a registry, in place.

    Called at the start of every refresh so dynamic entries are always rebuilt
    from a clean slate instead of accumulating across cycles. Returns the number
    of entries removed.
    """
    if not isinstance(models_def, dict):
        return 0
    removed = 0
    for model_id in [mid for mid, model_def in list(models_def.items()) if is_scanned_model_def(model_def)]:
        models_def.pop(model_id, None)
        removed += 1
    return removed
