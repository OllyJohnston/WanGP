# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Unit tests for the tensor-signature classifier.

Loaded directly from file when the heavy ``shared.utils`` package cannot be
imported (no torch); the module itself is standard-library only.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:  # normal import in a full environment
    from shared.utils import model_signature as ms
except ImportError:  # dependency-light test environment (no torch)
    _MODULE_PATH = Path(__file__).resolve().parents[1] / "model_signature.py"
    _SPEC = importlib.util.spec_from_file_location("model_signature", _MODULE_PATH)
    ms = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(ms)


def detect(keys, filename=None):
    return ms.detect_signature(keys, filename=filename)


class DiffusionCheckpointTests(unittest.TestCase):
    def test_sdxl_from_conditioner_embedders(self):
        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight",
            "conditioner.embedders.1.transformer.text_model.encoder.layers.0.layer_norm1.weight",
            "first_stage_model.decoder.conv_in.weight",
        ]
        result = detect(keys)
        self.assertEqual(result.signature, "sdxl")
        self.assertEqual(result.role, ms.ROLE_CHECKPOINT)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.matched_by, "header")

    def test_sd15_from_cond_stage_model(self):
        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight",
            "first_stage_model.decoder.conv_in.weight",
            "alphas_cumprod",
        ]
        result = detect(keys)
        self.assertEqual(result.signature, "sd15")
        self.assertEqual(result.matched_by, "header")

    def test_sdxl_wins_over_sd15_when_both_markers_present(self):
        keys = ["conditioner.embedders.0.x.weight", "cond_stage_model.transformer.x.weight"]
        self.assertEqual(detect(keys).signature, "sdxl")

    def test_flux(self):
        keys = [
            "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "img_in.weight",
            "txt_in.weight",
            "vector_in.weight",
            "guidance_in.weight",
        ]
        result = detect(keys)
        self.assertEqual(result.signature, "flux")
        self.assertEqual(result.role, ms.ROLE_CHECKPOINT)

    def test_flux_rejects_hunyuan_video_signature(self):
        keys = ["double_blocks.0.img_attn_qkv.weight", "single_blocks.0.linear1.weight",
                "img_in.weight", "txt_in.weight", "cond_type_embedding.weight"]
        self.assertNotEqual(detect(keys).signature, "flux")

    def test_krea2_bare_keys(self):
        keys = [
            "blocks.0.attn.gate.weight",
            "blocks.0.attn.qknorm.knorm.scale",
            "txtfusion.layerwise_blocks.0.x.weight",
            "tmlp.0.weight",
        ]
        self.assertEqual(detect(keys).signature, "krea2")

    def test_krea2_wrapped_keys(self):
        keys = [
            "model.diffusion_model.blocks.0.attn.gate.weight",
            "model.diffusion_model.blocks.0.attn.qknorm.knorm.scale",
        ]
        self.assertEqual(detect(keys).signature, "krea2")

    def test_z_image(self):
        keys = [
            "layers.0.attn.weight",
            "noise_refiner.0.x.weight",
            "context_refiner.0.x.weight",
            "cap_embedder.0.weight",
            "cap_pad_token",
        ]
        self.assertEqual(detect(keys).signature, "z_image")
        self.assertEqual(detect(keys).confidence, "high")

    def test_wan(self):
        keys = [
            "patch_embedding.weight",
            "text_embedding.0.weight",
            "time_embedding.0.weight",
            "blocks.0.self_attn.q.weight",
        ]
        self.assertEqual(detect(keys).signature, "wan")

    def test_wan_through_wrapper(self):
        keys = [
            "model.diffusion_model.patch_embedding.weight",
            "model.diffusion_model.text_embedding.0.weight",
            "model.diffusion_model.blocks.0.self_attn.q.weight",
        ]
        self.assertEqual(detect(keys).signature, "wan")


class TextEncoderTests(unittest.TestCase):
    def test_clip_l(self):
        keys = ["text_model.encoder.layers.0.layer_norm1.weight", "text_model.embeddings.token_embedding.weight"]
        self.assertEqual(detect(keys).signature, "clip_l")
        self.assertEqual(detect(keys).role, ms.ROLE_TEXT_ENCODER)

    def test_t5(self):
        keys = ["encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"]
        self.assertEqual(detect(keys).signature, "t5")

    def test_umt5_beats_t5(self):
        keys = ["encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight", "spiece_model"]
        self.assertEqual(detect(keys).signature, "umt5")

    def test_qwen_vl(self):
        keys = ["model.layers.0.self_attn.q_proj.weight", "visual.blocks.0.mlp.fc1.weight"]
        self.assertEqual(detect(keys).signature, "qwen_vl")

    def test_qwen3_without_vision(self):
        keys = ["model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight", "model.norm.weight"]
        self.assertEqual(detect(keys).signature, "qwen3")

    def test_mistral_vlm_is_not_qwen3(self):
        keys = [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "vision_tower.transformer.layers.0.x",
            "multi_modal_projector.linear_1.weight",
        ]
        self.assertIsNone(detect(keys, filename="ministral-3-3b.safetensors").signature)


class VaeTests(unittest.TestCase):
    def test_wan_vae(self):
        keys = ["decoder.upsamples.0.resample.1.weight", "encoder.downsamples.0.weight", "conv1.weight"]
        result = detect(keys)
        self.assertEqual(result.signature, "vae_wan")
        self.assertEqual(result.role, ms.ROLE_VAE)

    def test_standard_vae(self):
        keys = ["decoder.up_blocks.0.weight", "encoder.down_blocks.0.weight", "decoder.mid_block.x.weight"]
        self.assertEqual(detect(keys).signature, "vae_standard")

    def test_classic_vae_uses_up_down(self):
        keys = ["decoder.up.0.block.0.norm1.weight", "encoder.down.0.block.0.norm1.weight",
                "decoder.conv_in.weight"]
        self.assertEqual(detect(keys).signature, "vae_standard")


class FallbackTests(unittest.TestCase):
    def test_filename_fallback_when_keys_unknown(self):
        result = detect(["totally.unknown.key"], filename="flux1-dev-fp8.safetensors")
        self.assertEqual(result.signature, "flux")
        self.assertEqual(result.confidence, "low")
        self.assertEqual(result.matched_by, "filename")

    def test_header_wins_over_misleading_filename(self):
        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.0.x.weight",
            "conditioner.embedders.1.x.weight",
        ]
        result = detect(keys, filename="flux1-dev.safetensors")
        self.assertEqual(result.signature, "sdxl")
        self.assertEqual(result.matched_by, "header")

    def test_unknown_is_no_match(self):
        result = detect(["some.other.tensors.weight"], filename="mystery.safetensors")
        self.assertIsNone(result.signature)
        self.assertEqual(result.matched_by, "none")

    def test_bare_xl_marker_no_longer_matches(self):
        self.assertIsNone(detect(["x"], filename="some_xl_thing.safetensors").signature)

    def test_metadata_key_ignored(self):
        keys = ["__metadata__"]
        self.assertIsNone(detect(keys).signature)


class MiniMaxH3Tests(unittest.TestCase):
    def test_fl2va_gguf_by_filename(self):
        result = detect([], filename="MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf")
        self.assertEqual(result.signature, "minimax_h3")
        self.assertEqual(result.role, ms.ROLE_CHECKPOINT)
        self.assertEqual(result.matched_by, "filename")

    def test_ref2va_gguf_by_filename(self):
        self.assertEqual(detect([], filename="MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf").signature, "minimax_h3")

    def test_h3_underscore_autoencoders_are_not_checkpoints(self):
        # Underscore-named H3 autoencoders must never be misclassified as H3 checkpoints.
        self.assertIsNone(ms.detect_from_filename("minimax_h3_video_vae_fp16.safetensors").signature)
        self.assertIsNone(ms.detect_from_filename("minimax_h3_audio_vae_fp32.safetensors").signature)


if __name__ == "__main__":
    unittest.main()
