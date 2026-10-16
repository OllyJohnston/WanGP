# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Unit tests for the signature -> model-id mapping.

Loaded directly from file when the heavy ``shared.utils`` package cannot be
imported (no torch); the module itself is standard-library only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

try:  # normal import in a full environment
    from shared.utils import model_architecture_map as mam
except ImportError:  # dependency-light test environment (no torch)
    _MODULE_PATH = Path(__file__).resolve().parents[1] / "model_architecture_map.py"
    _SPEC = importlib.util.spec_from_file_location("model_architecture_map", _MODULE_PATH)
    mam = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mam)


CORE_DEFS = {
    "flux": "flux",
    "krea2_turbo": "krea2_turbo",
    "z_image": "z_image",
    "t2v_2_2": "t2v_2_2",
    "minimax_h3_fl2va_pruned": "minimax_h3_fl2va_pruned",
}


def _write_def(root: str, model_type: str, architecture: str) -> None:
    payload = {"model": {"name": model_type, "architecture": architecture}}
    with open(os.path.join(root, model_type + ".json"), "w", encoding="utf-8") as writer:
        json.dump(payload, writer)


def _build_models(root: str, defs: dict) -> dict:
    for model_type, architecture in defs.items():
        _write_def(root, model_type, architecture)
    return mam.discover_model_definitions([root])


class DiscoveryTests(unittest.TestCase):
    def test_discovers_models_and_architectures(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = _build_models(tmp, CORE_DEFS)
            self.assertEqual(set(models), set(CORE_DEFS))
            self.assertEqual(models["flux"].architecture, "flux")
            self.assertIn("krea2_turbo", mam.discover_architectures(models))

    def test_non_model_json_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_models(tmp, CORE_DEFS)
            with open(os.path.join(tmp, "profile.json"), "w", encoding="utf-8") as writer:
                json.dump({"profile": 3}, writer)
            with open(os.path.join(tmp, "broken.json"), "w", encoding="utf-8") as writer:
                writer.write("{not json")
            models = mam.discover_model_definitions([tmp])
            self.assertNotIn("profile", models)
            self.assertNotIn("broken", models)

    def test_later_root_overrides(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _build_models(first, {"z_image": "z_image"})
            _build_models(second, {"z_image": "z_image_override"})
            models = mam.discover_model_definitions([first, second])
            self.assertEqual(models["z_image"].architecture, "z_image_override")


class ResolveCheckpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models = _build_models(self._tmp.name, CORE_DEFS)

    def tearDown(self):
        self._tmp.cleanup()

    def _resolve(self, signature, filename=""):
        return mam.resolve_checkpoint(signature, mam.ROLE_CHECKPOINT, self.models, filename=filename)

    def test_krea2_maps_to_krea2_turbo(self):
        result = self._resolve("krea2")
        self.assertEqual(result.status, mam.STATUS_AVAILABLE)
        self.assertEqual(result.model_type, "krea2_turbo")
        self.assertEqual(result.architecture, "krea2_turbo")

    def test_z_image_maps_to_z_image(self):
        self.assertEqual(self._resolve("z_image").model_type, "z_image")

    def test_flux_maps_to_flux(self):
        self.assertEqual(self._resolve("flux").model_type, "flux")

    def test_wan_prefers_t2v_2_2(self):
        result = self._resolve("wan")
        self.assertEqual(result.model_type, "t2v_2_2")
        self.assertEqual(result.candidates[0], "t2v_2_2")

    def test_sd15_unsupported_without_plugin(self):
        result = self._resolve("sd15")
        self.assertEqual(result.status, mam.STATUS_UNSUPPORTED)
        self.assertIsNone(result.model_type)
        self.assertEqual(result.candidates[0], "stable_diffusion_1_4")

    def test_sdxl_unsupported_without_plugin(self):
        self.assertEqual(self._resolve("sdxl").status, mam.STATUS_UNSUPPORTED)

    def test_sd15_available_when_plugin_installed(self):
        with tempfile.TemporaryDirectory() as plugin:
            _build_models(plugin, {"stable_diffusion_1_4": "stable_diffusion_1_4"})
            combined = dict(self.models)
            combined.update(mam.discover_model_definitions([plugin]))
            result = mam.resolve_checkpoint("sd15", mam.ROLE_CHECKPOINT, combined)
            self.assertEqual(result.status, mam.STATUS_AVAILABLE)
            self.assertEqual(result.model_type, "stable_diffusion_1_4")

    def test_unknown_signature_is_unknown(self):
        self.assertEqual(self._resolve("mystery").status, mam.STATUS_UNKNOWN)

    def test_non_checkpoint_role_is_unknown(self):
        result = mam.resolve_checkpoint("t5", mam.ROLE_TEXT_ENCODER, self.models)
        self.assertEqual(result.status, mam.STATUS_UNKNOWN)

    def test_minimax_fl2va_refinement(self):
        result = self._resolve("minimax_h3", "MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf")
        self.assertEqual(result.model_type, "minimax_h3_fl2va_pruned")
        self.assertEqual(result.status, mam.STATUS_AVAILABLE)

    def test_minimax_ref2va_refinement_is_unsupported_when_absent(self):
        result = self._resolve("minimax_h3", "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf")
        self.assertEqual(result.status, mam.STATUS_UNSUPPORTED)
        self.assertEqual(result.candidates, ("minimax_h3_ref2va_pruned",))

    def test_minimax_without_filename_falls_back_to_default(self):
        result = self._resolve("minimax_h3")
        self.assertEqual(result.candidates[0], "minimax_h3_fl2va_pruned")


class ComponentCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models = _build_models(self._tmp.name, CORE_DEFS)

    def tearDown(self):
        self._tmp.cleanup()

    def test_qwen3_compatible_with_z_image(self):
        result = mam.compatible_component_model_types("qwen3", mam.ROLE_TEXT_ENCODER, self.models)
        self.assertEqual(result, ("z_image",))

    def test_qwen_vl_compatible_with_krea2(self):
        result = mam.compatible_component_model_types("qwen_vl", mam.ROLE_TEXT_ENCODER, self.models)
        self.assertEqual(result, ("krea2_turbo",))

    def test_umt5_compatible_with_wan(self):
        result = mam.compatible_component_model_types("umt5", mam.ROLE_TEXT_ENCODER, self.models)
        self.assertEqual(result, ("t2v_2_2",))

    def test_vae_wan_compatible_with_wan(self):
        result = mam.compatible_component_model_types("vae_wan", mam.ROLE_VAE, self.models)
        self.assertEqual(result, ("t2v_2_2",))

    def test_clip_l_compatible_with_flux(self):
        # Flux uses CLIP-L alongside T5, so clip_l matches the installed flux model.
        self.assertEqual(mam.compatible_component_model_types("clip_l", mam.ROLE_TEXT_ENCODER, self.models), ("flux",))

    def test_clip_l_adds_sd_when_installed(self):
        with tempfile.TemporaryDirectory() as plugin:
            _build_models(plugin, {"stable_diffusion_xl": "stable_diffusion_xl"})
            combined = dict(self.models)
            combined.update(mam.discover_model_definitions([plugin]))
            result = mam.compatible_component_model_types("clip_l", mam.ROLE_TEXT_ENCODER, combined)
            self.assertEqual(result, ("flux", "stable_diffusion_xl"))

    def test_unknown_role_returns_empty(self):
        self.assertEqual(mam.compatible_component_model_types("clip_l", mam.ROLE_CHECKPOINT, self.models), ())

class RealDefaultsTests(unittest.TestCase):
    def test_repository_defaults(self):
        root = Path(__file__).resolve().parents[3] / "defaults"
        if not root.is_dir():
            self.skipTest("defaults directory not present")
        models = mam.discover_model_definitions([str(root)])
        for model_type in ("flux", "krea2_turbo", "z_image", "t2v_2_2", "i2v_2_2", "minimax_h3_fl2va_pruned"):
            self.assertIn(model_type, models)
        # SD 1.5 / SDXL are plugin-provided, so they must NOT resolve from core defaults.
        self.assertNotIn("stable_diffusion_1_4", models)
        self.assertEqual(mam.resolve_checkpoint("krea2", mam.ROLE_CHECKPOINT, models).model_type, "krea2_turbo")
        self.assertEqual(mam.resolve_checkpoint("wan", mam.ROLE_CHECKPOINT, models).model_type, "t2v_2_2")
        self.assertEqual(mam.resolve_checkpoint("sd15", mam.ROLE_CHECKPOINT, models).status, mam.STATUS_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
