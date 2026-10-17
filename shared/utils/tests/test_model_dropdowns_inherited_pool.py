# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Level-3 variant inheritance tests.

A model that reuses a parent's checkpoint via string inheritance
(``"URLs": "krea2_turbo"``) must list that parent's variants (finetunes / scanned
checkpoints) in the level-3 picker, because it shares the same weights/architecture.
"""
import importlib.util
import os
import sys
import types
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.dirname(_TESTS_DIR)
_SHARED_DIR = os.path.dirname(_UTILS_DIR)


class _FakeDropdown:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _load_dropdowns():
    if "model_dropdowns" in sys.modules:
        return sys.modules["model_dropdowns"]
    if "gradio" not in sys.modules:
        stub = types.ModuleType("gradio")
        stub.Dropdown = _FakeDropdown
        stub.update = lambda *a, **k: None
        sys.modules["gradio"] = stub
    path = os.path.join(_SHARED_DIR, "model_dropdowns.py")
    spec = importlib.util.spec_from_file_location("model_dropdowns", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_dropdowns"] = module
    spec.loader.exec_module(module)
    return module


md = _load_dropdowns()


class FakeDeps:
    def __init__(self, defs, names, parents, families):
        self.families_infos = {"krea2": (1150, "Krea 2")}
        self._defs = defs
        self._names = names
        self._parents = parents
        self._families = families

    def get_model_def(self, model_type):
        return self._defs.get(model_type)

    def get_model_name(self, model_type):
        return self._names[model_type]

    def get_model_family(self, model_type, for_ui=True):
        return self._families[model_type]

    def get_parent_model_type(self, model_type):
        return self._parents[model_type]

    def get_base_model_type(self, model_type):
        return self._parents[model_type]


def _deps():
    defs = {
        "krea2_turbo": {"URLs": ["https://hf/Krea2Turbo_bf16.safetensors", "https://hf/Krea2Turbo_quanto.safetensors"]},
        "krea2_turbo_edit": {"URLs": "krea2_turbo"},
        "scan_my_krea": {"URLs": ["D:/models/my_krea.safetensors"], "path": "finetunes/scan_my_krea.json"},
    }
    names = {
        "krea2_turbo": "Krea 2 Turbo",
        "krea2_turbo_edit": "Krea 2 Turbo Identity Edit v1.2",
        "scan_my_krea": "my_krea",
    }
    parents = {"krea2_turbo": "krea2_turbo", "krea2_turbo_edit": "krea2_turbo_edit", "scan_my_krea": "krea2_turbo"}
    families = {key: "krea2" for key in defs}
    return FakeDeps(defs, names, parents, families)


class InheritedVariantPoolTests(unittest.TestCase):
    def test_owner_lists_scanned_variants(self):
        _families, _models, finetunes = md.get_sorted_dropdown(
            _deps(), ["krea2_turbo", "krea2_turbo_edit", "scan_my_krea"], "krea2", "krea2_turbo", three_levels=True
        )
        self.assertEqual([value for _label, value in finetunes], ["krea2_turbo", "scan_my_krea"])

    def test_inheriting_model_inherits_the_pool(self):
        # krea2_turbo_edit inherits krea2_turbo's weights -> its level-3 must show the
        # same scanned variant (plus its own entry), not collapse.
        _families, _models, finetunes = md.get_sorted_dropdown(
            _deps(), ["krea2_turbo", "krea2_turbo_edit", "scan_my_krea"], "krea2", "krea2_turbo_edit", three_levels=True
        )
        values = [value for _label, value in finetunes]
        self.assertIn("krea2_turbo_edit", values)
        self.assertIn("scan_my_krea", values)
        self.assertNotIn("krea2_turbo", values)  # the owner base entry itself must not leak in
        self.assertGreater(len(values), 1)  # -> dropdown 3 stays visible

    def test_non_inheriting_model_unchanged(self):
        _families, _models, finetunes = md.get_sorted_dropdown(
            _deps(), ["krea2_turbo_edit", "scan_my_krea"], "krea2", "scan_my_krea", three_levels=True
        )
        # A scanned variant is not itself an inheritor, so it shows only its own children.
        self.assertEqual([value for _label, value in finetunes], ["scan_my_krea"])


if __name__ == "__main__":
    unittest.main()
