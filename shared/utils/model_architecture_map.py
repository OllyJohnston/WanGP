# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Maps scanner signatures onto live WanGP model definitions.

The scanner (:mod:`shared.utils.model_scanner`) tells us *what* a file is (a
``krea2`` checkpoint, a ``t5`` text encoder, a ``vae_wan`` autoencoder, ...).
This module answers the next question: *which WanGP model does it satisfy?*

Design rules:

* **Runtime ids are discovered, never assumed.** ``discover_model_definitions``
  reads the ``defaults/*.json`` (and any plugin defaults) actually present, so a
  mapping is only reported as :data:`STATUS_AVAILABLE` when the target
  ``model_type`` really exists. Anything else is :data:`STATUS_UNSUPPORTED`
  (known family, no installed handler) or :data:`STATUS_UNKNOWN`.
* **Candidates are ordered preferences, not truth.** The candidate tables below
  only say "prefer these model ids, in this order"; discovery decides whether
  any of them is usable.
* **Components are handler-owned.** Text encoders and VAEs are usually created
  by the architecture handler rather than declared in the JSON def, so the
  component table is a small, evidence-referenced compatibility hint used as a
  *guard* (never as the sole basis for binding a file).

This module must stay dependency-free (standard library only).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple

try:  # normal import in a full environment
    from shared.utils import model_signature as ms
except ImportError:  # dependency-light environment (no torch): load sibling directly
    import sys

    _MS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_signature.py")
    _MS_SPEC = importlib.util.spec_from_file_location("model_signature", _MS_PATH)
    ms = importlib.util.module_from_spec(_MS_SPEC)
    sys.modules["model_signature"] = ms
    _MS_SPEC.loader.exec_module(ms)


ROLE_CHECKPOINT = ms.ROLE_CHECKPOINT
ROLE_TEXT_ENCODER = ms.ROLE_TEXT_ENCODER
ROLE_VAE = ms.ROLE_VAE

STATUS_AVAILABLE = "available"
STATUS_UNSUPPORTED = "unsupported"
STATUS_UNKNOWN = "unknown"


class DefinitionInfo(NamedTuple):
    """A model definition discovered on disk."""

    model_type: str
    architecture: str
    path: str


class ModelResolution(NamedTuple):
    """Result of resolving a scanned checkpoint signature to a model id."""

    signature: Optional[str]
    role: Optional[str]
    status: str
    model_type: Optional[str]
    architecture: Optional[str]
    candidates: Tuple[str, ...]
    note: str


# --- Runtime discovery -----------------------------------------------------

def load_model_definition(path: str) -> Optional[dict]:
    """Return the ``model`` subtree of a definition file, or ``None``.

    ``None`` means the file is not a model definition (so it is not a
    selectable model) -- e.g. a profile or a non-JSON-shaped file.
    """
    try:
        with open(path, encoding="utf-8") as reader:
            data = json.load(reader)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    if not isinstance(model, dict):
        return None
    return model


def discover_model_definitions(roots: Iterable[str]) -> Dict[str, DefinitionInfo]:
    """Discover installed model ids from one or more defaults roots.

    Later roots win on a ``model_type`` clash, matching ``refresh_model_defs``
    precedence (defaults first, then plugin overrides).
    """
    models: Dict[str, DefinitionInfo] = {}
    for root in roots or ():
        if not root:
            continue
        for path in sorted(glob.glob(os.path.join(root, "*.json"))):
            model = load_model_definition(path)
            if model is None:
                continue
            model_type = os.path.basename(path)[:-5]
            architecture = model.get("architecture") or model_type
            models[model_type] = DefinitionInfo(model_type, architecture, path)
    return models


def discover_architectures(models: Mapping[str, DefinitionInfo]) -> set:
    """Return the set of architecture ids referenced by discovered models."""
    return {info.architecture for info in models.values() if info.architecture}


# --- Checkpoint signature -> model id --------------------------------------

# Ordered candidate model ids per scanned checkpoint signature. The first entry
# is the preferred default; discovery validates which candidates are installed.
CHECKPOINT_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "krea2": ("krea2_turbo", "krea2_raw", "krea2_turbo_edit", "krea2_raw_edit"),
    "z_image": ("z_image", "z_image_base", "z_image_control", "z_image_twinflow_turbo"),
    "flux": (
        "flux",
        "flux_dev_kontext",
        "flux_schnell",
        "flux_chroma",
        "flux_krea",
        "flux2_dev",
        "flux2_klein_9b",
    ),
    "wan": ("t2v_2_2", "t2v", "i2v_2_2", "i2v", "flf2v_720p"),
    "sd15": ("stable_diffusion_1_4", "stable_diffusion_1_5"),
    "sdxl": ("stable_diffusion_xl", "stable_diffusion_xl_turbo"),
    "minimax_h3": ("minimax_h3_fl2va_pruned",),
}

# MiniMax-H3 ships several tasks sharing the ``minimax_h3`` filename signature;
# refine by sub-token (order matters: check the most specific first).
_MINIMAX_H3_SUBTYPES: Tuple[Tuple[str, str], ...] = (
    ("tts", "minimax_h3_tts_ref2va_pruned"),
    ("ref2va", "minimax_h3_ref2va_pruned"),
    ("fl2va", "minimax_h3_fl2va_pruned"),
    ("control", "minimax_h3_control_pruned"),
    ("vdn", "minimax_h3_vdn_pruned"),
)


def candidate_model_types(signature: Optional[str], filename: str = "") -> Tuple[str, ...]:
    """Ordered candidate model ids for a checkpoint signature."""
    if not signature:
        return ()
    if signature == "minimax_h3" and filename:
        base = os.path.basename(filename).lower()
        for token, model_type in _MINIMAX_H3_SUBTYPES:
            if token in base:
                return (model_type,)
    return CHECKPOINT_CANDIDATES.get(signature, ())


def resolve_checkpoint(
    signature: Optional[str],
    role: Optional[str] = ROLE_CHECKPOINT,
    models: Optional[Mapping[str, DefinitionInfo]] = None,
    filename: str = "",
) -> ModelResolution:
    """Resolve a scanned checkpoint signature to an installed model id."""
    models = models or {}
    if role != ROLE_CHECKPOINT:
        return ModelResolution(signature, role, STATUS_UNKNOWN, None, None, (), "not a checkpoint")
    if not signature:
        return ModelResolution(None, role, STATUS_UNKNOWN, None, None, (), "no checkpoint signature detected")

    candidates = candidate_model_types(signature, filename)
    if not candidates:
        return ModelResolution(
            signature, role, STATUS_UNKNOWN, None, None, (), f"no known model mapping for '{signature}'"
        )
    for model_type in candidates:
        info = models.get(model_type)
        if info is not None:
            return ModelResolution(
                signature,
                role,
                STATUS_AVAILABLE,
                model_type,
                info.architecture,
                candidates,
                f"mapped to installed model '{model_type}'",
            )
    return ModelResolution(
        signature,
        role,
        STATUS_UNSUPPORTED,
        None,
        None,
        candidates,
        "family recognised but no installed model provides it (a plugin may be required)",
    )


# --- Component (text encoder / VAE) compatibility --------------------------

# Evidence references (handler source), used as a compatibility *hint*:
#   * z_image  -> qwen3   : models/z_image/z_image_handler.py ("qwen3_bf16.safetensors")
#   * krea2*   -> qwen_vl : models/krea2/krea2_handler.py ("Qwen3-VL-4B-Instruct_text_bf16")
#   * ideogram4-> qwen_vl : models/ideogram4/ideogram4_handler.py
#   * flux*    -> t5,clip_l; vae_standard : models/ideogram4 handlers / Flux repo
#   * wan (t2v/i2v/flf2v) -> umt5; vae_wan : Wan uses UMT5-XXL + Wan VAE
#   * sd15/sdxl -> clip_l; vae_standard    : classic CLIP-L + SD VAE
ARCHITECTURE_COMPONENTS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "z_image": {"text_encoder": ("qwen3",)},
    "z_image_base": {"text_encoder": ("qwen3",)},
    "krea2_turbo": {"text_encoder": ("qwen_vl",)},
    "krea2_raw": {"text_encoder": ("qwen_vl",)},
    "qwen_image_20B": {"text_encoder": ("qwen_vl",), "vae": ("vae_wan",)},
    "qwen_image_21_7B": {"text_encoder": ("qwen_vl",), "vae": ("vae_wan",)},
    "flux": {"text_encoder": ("t5", "clip_l"), "vae": ("vae_standard",)},
    "flux_schnell": {"text_encoder": ("t5", "clip_l"), "vae": ("vae_standard",)},
    "flux_chroma": {"text_encoder": ("t5", "clip_l"), "vae": ("vae_standard",)},
    "t2v_2_2": {"text_encoder": ("umt5",), "vae": ("vae_wan",)},
    "t2v": {"text_encoder": ("umt5",), "vae": ("vae_wan",)},
    "i2v_2_2": {"text_encoder": ("umt5",), "vae": ("vae_wan",)},
    "i2v": {"text_encoder": ("umt5",), "vae": ("vae_wan",)},
    "flf2v_720p": {"text_encoder": ("umt5",), "vae": ("vae_wan",)},
    "stable_diffusion_1_4": {"text_encoder": ("clip_l",), "vae": ("vae_standard",)},
    "stable_diffusion_1_5": {"text_encoder": ("clip_l",), "vae": ("vae_standard",)},
    "stable_diffusion_xl": {"text_encoder": ("clip_l",), "vae": ("vae_standard",)},
}


def compatible_component_model_types(
    signature: Optional[str],
    role: Optional[str],
    models: Optional[Mapping[str, DefinitionInfo]] = None,
    table: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> Tuple[str, ...]:
    """Installed model ids whose ``role`` component accepts ``signature``.

    Example: a scanned ``clip_l`` text encoder is compatible with every
    installed model whose architecture lists ``clip_l`` in
    :data:`ARCHITECTURE_COMPONENTS`.
    """
    if not signature or role not in (ROLE_TEXT_ENCODER, ROLE_VAE):
        return ()
    table = ARCHITECTURE_COMPONENTS if table is None else table
    models = models or {}
    result = []
    for model_type, info in models.items():
        components = table.get(info.architecture)
        if not components:
            continue
        if signature in components.get(role, ()):
            result.append(model_type)
    return tuple(sorted(result))

