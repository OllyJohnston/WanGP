# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fill missing model fields from the live model definitions.

Some models (notably dynamically scanned ones) have no JSON file on disk, so a
form built only from the file would be empty even though the effective model
definition is known at runtime. This helper merges those values in.
"""
from __future__ import annotations

DEFAULT_INHERITED_FIELDS = ("URLs", "URLs2", "text_encoder_URLs", "name", "description")


def seed_missing_fields(model_values, model_types, get_model_def, fields=DEFAULT_INHERITED_FIELDS) -> dict:
    """Return a copy of ``model_values`` with empty fields filled from ``model_types``.

    For each field, the first model type whose definition provides a truthy value
    wins. Existing truthy values are never overwritten.
    """
    seeded = dict(model_values or {})
    for field in fields:
        if seeded.get(field):
            continue
        for model_type in model_types:
            if not model_type:
                continue
            candidate = (get_model_def(model_type) or {}).get(field)
            if candidate:
                seeded[field] = candidate
                break
    return seeded
