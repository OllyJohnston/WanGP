# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Tensor-signature classifier for safetensors model files.

The classifier inspects only the dotted key *prefixes* of a safetensors header
(never tensor payloads) and maps them to a model family plus its component role
(``checkpoint``, ``text_encoder`` or ``vae``). When the header keys are
ambiguous or were stripped, it falls back to keyword matching on the filename.

The rules are data, not code: add a :class:`SignatureRule` to :data:`ALL_RULES`
to teach the scanner a new family. Rules are evaluated in descending
``priority``; the first rule whose key conditions all hold wins, so
higher-priority (more specific) families must be listed above broader ones.

Probe prefixes below were derived from real checkpoint headers (ComfyUI layout):
``conditioner.embedders`` => SDXL, ``cond_stage_model.transformer`` => SD 1.5,
``double_blocks``/``single_blocks`` => Flux, ``blocks.N.attn.gate`` +
``qknorm``/``txtfusion`` => Krea 2, ``noise_refiner``/``cap_embedder`` => Z-Image
and ``patch_embedding``/``text_embedding`` => Wan.

This module must stay dependency-free (standard library only).
"""
from __future__ import annotations

import os
from typing import Iterable, NamedTuple, Optional, Sequence


class SignatureRule(NamedTuple):
    """A single classification rule.

    ``all_of``/``any_of``/``none_of`` are dotted key-prefix probes. ``all_of``
    must all be present, ``any_of`` needs at least one, and ``none_of`` must all
    be absent. At least one of ``all_of``/``any_of`` is required, otherwise the
    rule would match every file.
    """

    id: str
    role: str
    all_of: tuple = ()
    any_of: tuple = ()
    none_of: tuple = ()
    filename_markers: tuple = ()
    priority: int = 0


class Detection(NamedTuple):
    """Result of a classification attempt."""

    signature: Optional[str]
    role: Optional[str]
    confidence: Optional[str]  # "high" | "medium" | "low" | None
    matched_by: str  # "header" | "filename" | "none"


NO_MATCH = Detection(None, None, None, "none")

# Leading wrappers stripped once before prefix extraction so that, e.g., both
# ``blocks.0.attn`` and ``model.diffusion_model.blocks.0.attn`` resolve to the
# same probe.
_WRAPPER_PREFIXES = ("model.diffusion_model.", "model.", "diffusion_model.", "transformer.")

# --- Component roles -------------------------------------------------------
ROLE_CHECKPOINT = "checkpoint"
ROLE_TEXT_ENCODER = "text_encoder"
ROLE_VAE = "vae"


# Diffusion checkpoints (required families first, then adjacent families).
_CHECKPOINT_RULES = (
    SignatureRule(
        id="flux",
        role=ROLE_CHECKPOINT,
        all_of=("double_blocks", "single_blocks"),
        any_of=("img_in", "txt_in", "vector_in", "guidance_in"),
        none_of=("cond_type_embedding",),  # HunyuanVideo MMDiT marker
        filename_markers=("flux",),
        priority=95,
    ),
    SignatureRule(
        id="krea2",
        role=ROLE_CHECKPOINT,
        all_of=("blocks",),
        any_of=("txtfusion", "qknorm"),
        filename_markers=("krea2", "krea-2", "krea_2", "krea"),
        priority=90,
    ),
    SignatureRule(
        id="z_image",
        role=ROLE_CHECKPOINT,
        all_of=("layers",),
        any_of=("noise_refiner", "cap_embedder", "context_refiner"),
        filename_markers=("z-image", "zimage", "z_image", "zia"),
        priority=88,
    ),
    SignatureRule(
        id="wan",
        role=ROLE_CHECKPOINT,
        all_of=("patch_embedding", "text_embedding"),
        any_of=("blocks.0.self_attn", "time_projection", "time_embedding"),
        filename_markers=("wan2", "wan_2", "wan-2", "wan2.1", "wan2.2"),
        priority=85,
    ),
    SignatureRule(
        id="minimax_h3",
        role=ROLE_CHECKPOINT,
        # Filename-only rule (real MiniMax-H3 weights are GGUF, never
        # header-inspected). Hyphenated markers only, so the underscore-named
        # H3 autoencoders (minimax_h3_video_vae / _audio_vae) stay VAEs.
        filename_markers=("minimax-h3-fl2va", "minimax-h3-ref2va", "minimax-h3-control", "minimax-h3-vdn", "minimax-h3-tts"),
        priority=84,
    ),
    SignatureRule(
        id="sdxl",
        role=ROLE_CHECKPOINT,
        all_of=("conditioner.embedders",),
        filename_markers=("sdxl", "pony", "illustrious"),
        priority=80,
    ),
    SignatureRule(
        id="sd15",
        role=ROLE_CHECKPOINT,
        all_of=("cond_stage_model.transformer",),
        none_of=("conditioner.embedders",),
        filename_markers=("sd1.5", "v1-5", "v1_5", "1.5"),
        priority=70,
    ),
)

# Text encoders.
_TEXT_ENCODER_RULES = (
    SignatureRule(
        id="qwen_vl",
        role=ROLE_TEXT_ENCODER,
        any_of=("model.language_model", "visual.blocks", "model.visual"),
        all_of=("model",),
        filename_markers=("qwen2.5-vl", "qwen2_5_vl", "qwen3-vl", "qwen3vl", "qwen_vl", "qwen-vl"),
        priority=68,
    ),
    SignatureRule(
        id="umt5",
        role=ROLE_TEXT_ENCODER,
        all_of=("encoder.block",),
        any_of=("spiece_model", "scale_weight"),
        filename_markers=("umt5",),
        priority=64,
    ),
    SignatureRule(
        id="t5",
        role=ROLE_TEXT_ENCODER,
        all_of=("encoder.block",),
        any_of=("shared",),
        none_of=("visual",),
        filename_markers=("t5xxl", "t5_xxl", "t5-xxl", "t5"),
        priority=60,
    ),
    SignatureRule(
        id="clip_l",
        role=ROLE_TEXT_ENCODER,
        all_of=("text_model.encoder",),
        filename_markers=("clip_l", "clip-l", "clip_large", "clip-l"),
        priority=56,
    ),
    SignatureRule(
        id="qwen3",
        role=ROLE_TEXT_ENCODER,
        all_of=("model.layers",),
        any_of=("model.embed_tokens", "model.norm"),
        none_of=("visual", "model.visual", "model.language_model", "vision_tower", "multi_modal_projector"),
        filename_markers=("qwen3", "qwen_3"),
        priority=52,
    ),
)

# Autoencoders.
_VAE_RULES = (
    SignatureRule(
        id="vae_wan",
        role=ROLE_VAE,
        any_of=("decoder.upsamples", "encoder.downsamples"),
        priority=44,
    ),
    SignatureRule(
        id="vae_standard",
        role=ROLE_VAE,
        # Classic SD/Flux autoencoders use ``decoder.up.N``/``encoder.down.N``;
        # newer ones use ``up_blocks``/``down_blocks``. Both map here.
        any_of=("decoder.up_blocks", "encoder.down_blocks", "decoder.up", "encoder.down"),
        priority=40,
    ),
)

ALL_RULES: tuple = _CHECKPOINT_RULES + _TEXT_ENCODER_RULES + _VAE_RULES


def build_probe_set(keys: Iterable[str]) -> set:
    """Return every dotted token sequence present in ``keys`` (O(1) lookups).

    Each key contributes all of its contiguous dot-delimited token runs, for the
    raw key and for the key with one leading module wrapper removed. This makes
    a probe match whether the token run is at the start of a key or nested
    inside it (e.g. ``qknorm`` inside ``blocks.0.attn.qknorm.knorm``).
    """
    probes: set = set()
    for key in keys:
        if not isinstance(key, str) or key == "__metadata__":
            continue
        candidates = [key]
        for wrapper in _WRAPPER_PREFIXES:
            if key.startswith(wrapper):
                candidates.append(key[len(wrapper):])
                break
        for candidate in candidates:
            tokens = candidate.split(".")
            for start in range(len(tokens)):
                for end in range(start + 1, len(tokens) + 1):
                    probes.add(".".join(tokens[start:end]))
    return probes


# Backwards-compatible alias; the probe set supersedes plain prefix matching.
build_prefixes = build_probe_set


def _matches(rule: SignatureRule, probes: set) -> Optional[str]:
    if rule.all_of and not all(probe in probes for probe in rule.all_of):
        return None
    if rule.any_of and not any(probe in probes for probe in rule.any_of):
        return None
    if rule.none_of and any(probe in probes for probe in rule.none_of):
        return None
    if not rule.all_of and not rule.any_of:
        return None
    # ``all_of`` rules are anchored on a required key, so they are more certain.
    return "high" if rule.all_of else "medium"


def _ordered_rules(rules: Sequence[SignatureRule]) -> list:
    return sorted(rules, key=lambda rule: -rule.priority)


def detect_from_filename(filename: Optional[str], rules: Sequence[SignatureRule] = ALL_RULES) -> Detection:
    """Keyword fallback used when no header rule matched."""
    if not filename:
        return NO_MATCH
    base = os.path.basename(str(filename)).lower()
    if not base:
        return NO_MATCH
    for rule in _ordered_rules(rules):
        for marker in rule.filename_markers:
            if marker and marker in base:
                return Detection(rule.id, rule.role, "low", "filename")
    return NO_MATCH


def detect_signature(
    keys: Optional[Iterable[str]] = None,
    filename: Optional[str] = None,
    rules: Sequence[SignatureRule] = ALL_RULES,
    probes: Optional[set] = None,
) -> Detection:
    """Classify a model file from its header ``keys`` (or ``filename`` alone).

    Header inspection always wins; the filename is consulted only when no rule
    matched the keys, so a misleading filename cannot override real evidence.
    """
    if probes is None:
        probes = build_probe_set(keys or ())
    for rule in _ordered_rules(rules):
        confidence = _matches(rule, probes)
        if confidence is not None:
            return Detection(rule.id, rule.role, confidence, "header")
    return detect_from_filename(filename, rules)
