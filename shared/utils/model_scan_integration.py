# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Torch-free glue between ``wgp.py`` and the model-scan feature.

``wgp.py`` pulls in torch and gradio, so it cannot be imported by the stdlib test
suite. This module holds the handful of integration decisions that used to live
inline in ``wgp.py`` -- role mapping, synthetic model initialisation, scan status
reporting, registry injection and local-bind interception -- expressed as pure
functions with the runtime/provider modules injected. ``wgp.py`` then becomes
thin delegation, and the behaviour is testable with fakes.
"""

from __future__ import annotations

import os
from typing import Optional


def role_for_file_type(file_type, runtime):
    """Scan role for a ``download_models`` ``file_type`` (2 == text encoder)."""
    return runtime.ROLE_TEXT_ENCODER if file_type == 2 else runtime.ROLE_CHECKPOINT


def role_for_asset_prop(prop, runtime):
    """Scan role for a recursive asset list; only VAE weights are role-constrained."""
    return runtime.ROLE_VAE if prop == "VAE_URLs" else None


def initialise_scanned_model(model_id, model_def, *, provider, init_model_def):
    """Give a scanned checkpoint definition the synthetic finetune shape the UI expects.

    Sets a synthetic ``finetunes/<id>.json`` path, runs the handler initialiser
    (inheritance), then tags the definition as a scanned finetune carrying its
    real on-disk weight path. The input mapping is never mutated.
    """
    model_def = dict(model_def)
    model_def["path"] = provider.scanned_json_path(model_id)
    model_def = init_model_def(model_id, model_def)
    metadata = model_def.setdefault("metadata", {})
    metadata["finetune"] = True
    metadata[provider.SCANNED_METADATA_KEY] = True
    metadata["local_path"] = model_def.get(provider.SCAN_PATH_KEY, "")
    return model_def


def report_scan_status(index, *, printer=print, limit: int = 10) -> list:
    """Print (and return) the scan's skipped paths / timeout notice for the terminal log."""
    stats = index.get("stats", {}) if isinstance(index, dict) else {}
    lines = []
    if stats.get("timed_out"):
        lines.append("Model scan: the folder scan is taking longer than expected and is continuing in the background.")
    warnings = stats.get("warnings") or []
    if warnings:
        lines.append(f"Model scan: {len(warnings)} path(s) skipped during the scan:")
        for warning in warnings[:limit]:
            lines.append(f"  - {warning}")
        if len(warnings) > limit:
            lines.append(f"  - ... and {len(warnings) - limit} more")
    for line in lines:
        printer(line)
    return lines


def register_scanned_models(models_def, *, runtime, provider, initialiser=None, printer=print) -> list:
    """Refresh the in-memory scanned model registry from the (non-blocking) scan index.

    When the feature is disabled any previously injected entries are stripped so
    the registry cannot retain stale scanned models; otherwise the cached index is
    used and the scan thread is never joined. Returns the list of scanned ids added.
    """
    try:
        if not runtime.is_enabled():
            provider.strip_scanned_models(models_def)
            return []
        index = runtime.get_index(force=False, wait=False)
        added = provider.inject_scanned_models(models_def, index, initialiser=initialiser)
        if index:
            report_scan_status(index, printer=printer)
        return added
    except Exception as scan_error:
        printer(f"Model scan: unable to register discovered models: {scan_error}")
        return []


def try_bind_local(model_filename, role=None, *, runtime, printer=print) -> Optional[str]:
    """Return the path of a local weight to reuse in place of a remote download, or ``None``.

    Binding is opt-in (``model_scan_bind_local``) and only succeeds for a unique,
    container- and signature-compatible local file, so a wrong file never silently
    replaces a download.
    """
    if not model_filename or not isinstance(model_filename, str):
        return None
    try:
        if not runtime.is_bind_enabled():
            return None
        result = runtime.attempt_bind(model_filename, role=role)
        if result.path:
            printer(f"Reusing local file for '{os.path.basename(model_filename)}': {result.path}")
            return result.path
        if result.reason not in ("disabled", "no_index", "no_match"):
            printer(
                f"Local file not reused for '{os.path.basename(model_filename)}' "
                f"({result.reason}); candidates: {list(result.candidates)}"
            )
    except Exception as scan_error:
        printer(f"Local scan bind failed for '{model_filename}': {scan_error}")
    return None
