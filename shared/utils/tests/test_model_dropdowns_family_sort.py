# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A-Z ordering test for the model-family dropdown.

Scope: the family dropdown (``get_sorted_dropdown`` -> first ``gr.Dropdown`` in
``generate_dropdown_model_list``) must be ordered alphabetically by its display
label, not by the numeric configured family-order key.

``shared/model_dropdowns.py`` imports gradio at module import time but the family
ordering path does not touch gradio, so a minimal stub lets us exercise it.
"""
import importlib.util
import os
import sys
import types
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.dirname(_TESTS_DIR)
_SHARED_DIR = os.path.dirname(_UTILS_DIR)


def _load_dropdowns():
    if "model_dropdowns" in sys.modules:
        return sys.modules["model_dropdowns"]
    if "gradio" not in sys.modules:
        stub = types.ModuleType("gradio")
        stub.Dropdown = object
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
    def __init__(self, families_infos, family_of, names, parents):
        self.families_infos = families_infos
        self._family_of = family_of
        self._names = names
        self._parents = parents

    def get_model_family(self, model_type, for_ui=True):
        return self._family_of[model_type]

    def get_model_name(self, model_type):
        return self._names[model_type]

    def get_parent_model_type(self, model_type):
        return self._parents[model_type]

    def get_base_model_type(self, model_type):
        return self._parents[model_type]


class FamilyDropdownSortingTests(unittest.TestCase):
    def _deps(self):
        # 'zeta' carries the LOWER configured order (1) yet must sort AFTER 'alpha'.
        families_infos = {"zeta": (1, "Zeta"), "alpha": (2, "Alpha")}
        family_of = {"alpha_base": "alpha", "zeta_base": "zeta"}
        names = {"alpha_base": "Alpha Base", "zeta_base": "Zeta Base"}
        parents = {"alpha_base": "alpha_base", "zeta_base": "zeta_base"}
        return FakeDeps(families_infos, family_of, names, parents)

    def test_family_dropdown_sorted_a_to_z_not_by_config_order(self):
        families, _models, _finetunes = md.get_sorted_dropdown(
            self._deps(), ["alpha_base", "zeta_base"], None, "alpha_base", three_levels=True
        )
        self.assertEqual(families, [("Alpha", "alpha"), ("Zeta", "zeta")])

    def test_case_insensitive_label_ordering(self):
        families_infos = {"b": (1, "beta"), "u": (2, "Upsilon"), "a": (3, "Alpha")}
        family_of = {"b_base": "b", "u_base": "u", "a_base": "a"}
        names = {"b_base": "B", "u_base": "U", "a_base": "A"}
        parents = {"b_base": "b_base", "u_base": "u_base", "a_base": "a_base"}
        deps = FakeDeps(families_infos, family_of, names, parents)
        families, _models, _finetunes = md.get_sorted_dropdown(
            deps, ["b_base", "u_base", "a_base"], None, "a_base", three_levels=True
        )
        self.assertEqual([label for label, _family in families], ["Alpha", "beta", "Upsilon"])


if __name__ == "__main__":
    unittest.main()
