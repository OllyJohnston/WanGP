# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import importlib.util
import os
import sys
import tempfile
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.dirname(_TESTS_DIR)


def _load(name):
    try:
        module = importlib.import_module(f"shared.utils.{name}")
        return module
    except Exception:
        path = os.path.join(_UTILS_DIR, name + ".py")
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]


provider = _load("model_scan_provider")
amap = _load("model_architecture_map")


def _entry(path, signature, categories=("checkpoints",)):
    return {"path": path, "signature": signature, "categories": list(categories)}


def _model_infos():
    return {
        "krea2_turbo": amap.DefinitionInfo("krea2_turbo", "krea2_turbo", "defaults/krea2_turbo.json"),
        "z_image": amap.DefinitionInfo("z_image", "z_image", "defaults/z_image.json"),
        "t2v_2_2": amap.DefinitionInfo("t2v_2_2", "t2v_2_2", "defaults/t2v_2_2.json"),
    }


class ProviderHelpersTests(unittest.TestCase):
    def test_slugify_matches_editor_sanitizer(self):
        self.assertEqual(provider.slugify("my model!! v2"), "my_model_v2")
        self.assertEqual(provider.slugify("krea2SATDirtyrealism_v4_fp8"), "krea2SATDirtyrealism_v4_fp8")

    def test_dynamic_id_prefixes_stem(self):
        self.assertEqual(provider.dynamic_id("myModel"), "scan_myModel")
        self.assertEqual(provider.dynamic_id("   "), "")

    def test_scanned_json_path_is_under_finetunes(self):
        path = provider.scanned_json_path("scan_foo")
        self.assertEqual(os.path.basename(os.path.dirname(path)).casefold(), "finetunes")
        self.assertEqual(os.path.basename(path), "scan_foo.json")

    def test_is_scanned_model_def(self):
        self.assertTrue(provider.is_scanned_model_def({provider.SCAN_PATH_KEY: "x"}))
        self.assertTrue(provider.is_scanned_model_def({"metadata": {provider.SCANNED_METADATA_KEY: True}}))
        self.assertFalse(provider.is_scanned_model_def({"metadata": {}}))
        self.assertFalse(provider.is_scanned_model_def(None))

    def test_claimed_scan_paths_ignores_remote_and_scanned_defs(self):
        claimed = provider.claimed_scan_paths(
            {
                "cloud": {"URLs": ["https://h/x/y.safetensors"]},
                "local": {"URLs": ["D:/models/foo.safetensors"]},
                "scanned": {
                    provider.SCAN_PATH_KEY: "D:/models/bar.safetensors",
                    "URLs": ["D:/models/bar.safetensors"],
                },
            }
        )
        self.assertIn(os.path.normcase(os.path.abspath("D:/models/foo.safetensors")), claimed)
        self.assertEqual(len(claimed), 1)

    def test_model_infos_from_models_def(self):
        infos = provider.model_infos_from_models_def(
            {"krea2_turbo": {"architecture": "krea2_turbo", "path": "defaults/krea2_turbo.json"}, "bad": "not-a-dict"}
        )
        self.assertIn("krea2_turbo", infos)
        self.assertNotIn("bad", infos)
        self.assertEqual(infos["krea2_turbo"].architecture, "krea2_turbo")


class BuildScannedDefsTests(unittest.TestCase):
    def _build(self, entries, disk=(), initialiser=None, **kwargs):
        index = {"entries": {str(i): e for i, e in enumerate(entries)}}
        return provider.build_scanned_model_defs(index, _model_infos(), disk, initialiser=initialiser, **kwargs)

    def test_builds_available_checkpoint(self):
        defs = self._build([_entry("D:/models/krea2SATDirtyrealism_v4_fp8.safetensors", "krea2")])
        self.assertEqual(list(defs), ["scan_krea2SATDirtyrealism_v4_fp8"])
        model_def = defs["scan_krea2SATDirtyrealism_v4_fp8"]
        self.assertEqual(model_def["architecture"], "krea2_turbo")
        self.assertEqual(model_def["finetune_source_model"], "krea2_turbo")
        self.assertEqual(model_def[provider.SCAN_PATH_KEY], os.path.abspath("D:/models/krea2SATDirtyrealism_v4_fp8.safetensors"))
        self.assertEqual(model_def["URLs"], [os.path.abspath("D:/models/krea2SATDirtyrealism_v4_fp8.safetensors")])

    def test_skips_non_checkpoint_category(self):
        defs = self._build([_entry("D:/vae/wan_2.1_vae.safetensors", "vae_wan", categories=("vae",))])
        self.assertEqual(defs, {})

    def test_skips_unknown_signature(self):
        defs = self._build([_entry("D:/models/mystery.safetensors", None)])
        self.assertEqual(defs, {})

    def test_skips_unsupported_signature(self):
        # sd15 has no installed candidate -> unsupported -> not registered.
        defs = self._build([_entry("D:/models/v1-5-pruned-emaonly.safetensors", "sd15")])
        self.assertEqual(defs, {})

    def test_disk_id_wins(self):
        defs = self._build(
            [_entry("D:/models/krea2SATDirtyrealism_v4_fp8.safetensors", "krea2")],
            disk={"scan_krea2SATDirtyrealism_v4_fp8"},
        )
        self.assertEqual(defs, {})

    def test_promoted_rename_suppresses_scanned_entry(self):
        # A scanned checkpoint renamed+promoted to a disk finetune points its URLs
        # at the same local weight. The scan must not re-add the old entry.
        entry = _entry("D:/models/krea2SATDirtyrealism_v4_fp8.safetensors", "krea2")
        index = {"entries": {"0": entry}}
        claimed = provider.claimed_scan_paths(
            {"my_renamed_finetune": {"URLs": ["D:/models/krea2SATDirtyrealism_v4_fp8.safetensors"]}}
        )
        defs = provider.build_scanned_model_defs(
            index, _model_infos(), {"my_renamed_finetune"}, claimed_paths=claimed
        )
        self.assertEqual(defs, {})

    def test_duplicate_stems_suffixed(self):
        defs = self._build(
            [
                _entry("D:/a/model.safetensors", "krea2"),
                _entry("D:/b/model.safetensors", "krea2"),
            ]
        )
        self.assertEqual(sorted(defs), ["scan_model", "scan_model_2"])

    def test_deterministic_order_by_path(self):
        defs = self._build(
            [
                _entry("D:/b/zeta.safetensors", "krea2"),
                _entry("D:/a/alpha.safetensors", "krea2"),
            ]
        )
        self.assertEqual(list(defs), ["scan_alpha", "scan_zeta"])

    def test_initialiser_applied(self):
        def initialise(model_id, model_def):
            model_def = dict(model_def)
            model_def["path"] = provider.scanned_json_path(model_id)
            model_def["metadata"] = {provider.SCANNED_METADATA_KEY: True}
            return model_def

        defs = self._build([_entry("D:/models/alpha.safetensors", "z_image")], initialiser=initialise)
        self.assertEqual(defs["scan_alpha"]["path"], provider.scanned_json_path("scan_alpha"))
        self.assertTrue(provider.is_scanned_model_def(defs["scan_alpha"]))

    def test_real_examples(self):
        defs = self._build(
            [
                _entry("D:/Stability/models/checkpoints/realDream_krea2V1_nvfp4.safetensors", "krea2"),
                _entry("D:/Stability/models/checkpoints/bigLove_zia1_pruned_bf16.safetensors", "z_image"),
                _entry("D:/Stability/models/checkpoints/v1-5-pruned-emaonly.safetensors", "sd15"),
            ]
        )
        self.assertIn("scan_realDream_krea2V1_nvfp4", defs)
        self.assertIn("scan_bigLove_zia1_pruned_bf16", defs)
        self.assertNotIn("scan_v1-5-pruned-emaonly", defs)


class MergeTests(unittest.TestCase):
    def test_disk_entries_win_on_merge(self):
        models = {"krea2_turbo": {"architecture": "krea2_turbo"}}
        scanned = {"scan_x": {"architecture": "krea2_turbo"}, "krea2_turbo": {"architecture": "SHOULD_NOT_WIN"}}
        merged = provider.merge_scanned_models(models, scanned)
        self.assertEqual(merged["krea2_turbo"]["architecture"], "krea2_turbo")
        self.assertIn("scan_x", merged)


class PluginDiscoveryTests(unittest.TestCase):
    """M3 gap check: plugin-provided architectures must resolve when the plugin is live."""

    def test_model_infos_from_live_registry_includes_plugin_architecture(self):
        live = {
            "krea2_turbo": {"architecture": "krea2_turbo", "path": "defaults/krea2_turbo.json"},
            "stable_diffusion_1_4": {
                "architecture": "stable_diffusion_1_4",
                "path": "plugins/wan2gp-stable-diffusion-1-4/defaults/stable_diffusion_1_4.json",
            },
        }
        infos = provider.model_infos_from_models_def(live)
        self.assertEqual(infos["stable_diffusion_1_4"].architecture, "stable_diffusion_1_4")

    def test_sd15_resolves_only_once_a_plugin_supplies_the_architecture(self):
        base = _model_infos()
        scanned_index = {"entries": {"k": _entry("D:/c/v1-5-pruned-emaonly.safetensors", "sd15")}}
        # Without the SD plugin the checkpoint is unsupported -> not registered.
        self.assertEqual(provider.build_scanned_model_defs(scanned_index, base), {})
        # With the plugin architecture present in the live registry it registers.
        with_plugin = dict(base)
        with_plugin["stable_diffusion_1_4"] = amap.DefinitionInfo(
            "stable_diffusion_1_4", "stable_diffusion_1_4",
            "plugins/wan2gp-stable-diffusion-1-4/defaults/stable_diffusion_1_4.json",
        )
        defs = provider.build_scanned_model_defs(scanned_index, with_plugin)
        self.assertIn("scan_v1-5-pruned-emaonly", defs)
        self.assertEqual(defs["scan_v1-5-pruned-emaonly"]["architecture"], "stable_diffusion_1_4")


class StripScannedTests(unittest.TestCase):
    def test_removes_scanned_entries_and_keeps_disk(self):
        models = {
            "krea2_turbo": {"architecture": "krea2_turbo"},
            "scan_a": {"architecture": "krea2_turbo", provider.SCAN_PATH_KEY: "/x/a.safetensors"},
            "scan_b": {"architecture": "z_image", "metadata": {provider.SCANNED_METADATA_KEY: True}},
        }
        removed = provider.strip_scanned_models(models)
        self.assertEqual(removed, 2)
        self.assertEqual(set(models), {"krea2_turbo"})

    def test_strip_is_idempotent(self):
        models = {"scan_a": {provider.SCAN_PATH_KEY: "/x"}}
        self.assertEqual(provider.strip_scanned_models(models), 1)
        self.assertEqual(provider.strip_scanned_models(models), 0)

    def test_strip_tolerates_non_dict(self):
        self.assertEqual(provider.strip_scanned_models(None), 0)

    def test_repeated_injection_does_not_accumulate(self):
        # Mirrors refresh: on every cycle the registry is reset to disk defs, then
        # scanned models are stripped and rebuilt. The scan_* set must be stable.
        entries = [
            _entry("D:/models/realDream_krea2V1_nvfp4.safetensors", "krea2"),
            _entry("D:/models/intorealism_zitV10.safetensors", "z_image"),
        ]
        index = {"entries": {str(i): e for i, e in enumerate(entries)}}
        models = {"krea2_turbo": {"architecture": "krea2_turbo"}}
        snapshots = []
        for _ in range(3):
            provider.strip_scanned_models(models)
            models.update(provider.build_scanned_model_defs(index, _model_infos(), disk_model_types=set(models)))
            snapshots.append(sorted(mid for mid in models if mid.startswith(provider.SCANNED_ID_PREFIX)))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1], snapshots[2])
        self.assertEqual(len(snapshots[0]), 2)


if __name__ == "__main__":
    unittest.main()
