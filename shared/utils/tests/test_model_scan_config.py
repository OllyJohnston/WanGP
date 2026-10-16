# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Unit tests for the model scan-path configuration block.

The production package ``shared.utils`` imports the ML runtime through its
``__init__`` chain, so a plain ``import`` fails where torch is absent. We load
the target module directly from its file when the normal import is not
possible; the module itself is standard-library only, so both paths behave
identically.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:  # normal import in a full environment
    from shared.utils import model_scan_config as msc
except ImportError:  # dependency-light test environment (no torch)
    _MODULE_PATH = Path(__file__).resolve().parents[1] / "model_scan_config.py"
    _SPEC = importlib.util.spec_from_file_location("model_scan_config", _MODULE_PATH)
    msc = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(msc)


class ScanPathDefaultsTests(unittest.TestCase):
    def test_default_block_covers_every_category(self):
        block = msc.default_scan_paths()
        self.assertEqual(set(block), set(msc.SCAN_CATEGORIES))
        self.assertTrue(all(value == [] for value in block.values()))

    def test_normalize_rejects_non_mappings(self):
        for value in (None, 5, "a path", ["a", "b"]):
            block = msc.normalize_scan_paths(value)
            self.assertEqual(block, msc.default_scan_paths())

    def test_normalize_fills_missing_categories_and_drops_unknown(self):
        block = msc.normalize_scan_paths({"checkpoints": ["D:/a"], "bogus": ["x"]})
        self.assertEqual(set(block), set(msc.SCAN_CATEGORIES))
        self.assertEqual(block["checkpoints"], ["D:/a"])
        self.assertEqual(block["text_encoders"], [])
        self.assertNotIn("bogus", block)

    def test_normalize_accepts_textbox_multiline_string(self):
        block = msc.normalize_scan_paths({"vae": "D:/v1\r\nD:/v2\n\n"})
        self.assertEqual(block["vae"], ["D:/v1", "D:/v2"])


class PathListTests(unittest.TestCase):
    def test_strips_blanks_and_preserves_order(self):
        self.assertEqual(msc.normalize_path_list(["  a  ", "", "b"]), ["a", "b"])

    def test_dedupe_is_case_insensitive_and_separator_agnostic(self):
        # Windows-style duplicates differing only by case or slash direction.
        value = ["D:/Models", "d:\\models", "D:\\Models\\"]
        self.assertEqual(msc.normalize_path_list(value), ["D:/Models"])

    def test_scalar_value_becomes_single_entry(self):
        self.assertEqual(msc.normalize_path_list("D:/only"), ["D:/only"])
        self.assertEqual(msc.normalize_path_list(None), [])


class ServerConfigTests(unittest.TestCase):
    def test_scan_paths_from_server_reads_block(self):
        config = {msc.SCAN_PATHS_CONFIG_KEY: {"checkpoints": ["D:/ck"]}}
        self.assertEqual(msc.scan_paths_from_server(config)["checkpoints"], ["D:/ck"])

    def test_scan_paths_from_server_tolerates_missing_key(self):
        self.assertEqual(msc.scan_paths_from_server({}), msc.default_scan_paths())
        self.assertEqual(msc.scan_paths_from_server(None), msc.default_scan_paths())

    def test_scan_paths_to_text_round_trips(self):
        block = msc.normalize_scan_paths({"loras": ["D:/l1", "D:/l2"]})
        self.assertEqual(msc.scan_paths_to_text(block, "loras"), "D:/l1\nD:/l2")


class MergedPathsTests(unittest.TestCase):
    def test_base_paths_come_first_then_scan_directories(self):
        block = {"checkpoints": ["D:/ck"], "text_encoders": ["D:/te"], "vae": ["D:/vae"]}
        merged = msc.merged_checkpoint_paths(["ckpts", "."], block)
        self.assertEqual(merged, ["ckpts", ".", "D:/ck", "D:/te", "D:/vae"])

    def test_duplicates_across_base_and_scan_are_collapsed(self):
        block = {"checkpoints": ["ckpts", "D:/new"], "text_encoders": ["D:/new"], "vae": []}
        merged = msc.merged_checkpoint_paths(["ckpts", "."], block)
        self.assertEqual(merged, ["ckpts", ".", "D:/new"])

    def test_loras_category_is_not_merged_into_checkpoint_roots(self):
        block = {"loras": ["D:/loras"]}
        self.assertEqual(msc.merged_checkpoint_paths(["ckpts"], block), ["ckpts"])

    def test_empty_base_yields_only_garbage_free_result(self):
        block = {"text_encoders": ["D:/te"]}
        self.assertEqual(msc.merged_checkpoint_paths([], block), ["D:/te"])

    def test_download_root_is_never_moved(self):
        # The first entry must remain the first entry regardless of scan dirs.
        block = {"checkpoints": ["A", "B"]}
        merged = msc.merged_checkpoint_paths(["ckpts", "."], block)
        self.assertEqual(merged[0], "ckpts")


class LookupRootsTests(unittest.TestCase):
    def test_uses_config_checkpoints_paths_when_base_not_given(self):
        config = {
            "checkpoints_paths": ["ckpts"],
            msc.SCAN_PATHS_CONFIG_KEY: {"checkpoints": ["D:/ck"]},
        }
        self.assertEqual(msc.checkpoint_lookup_roots(config), ["ckpts", "D:/ck"])

    def test_explicit_base_overrides_config_value(self):
        config = {"checkpoints_paths": ["ignored"], msc.SCAN_PATHS_CONFIG_KEY: {}}
        self.assertEqual(msc.checkpoint_lookup_roots(config, base_paths=["D:/base"]), ["D:/base"])

    def test_missing_config_returns_empty(self):
        self.assertEqual(msc.checkpoint_lookup_roots(None), [])
        self.assertEqual(msc.checkpoint_lookup_roots({}), [])


if __name__ == "__main__":
    unittest.main()
