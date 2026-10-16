# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Sorting regression tests for the model selector hierarchy.

``shared/model_dropdowns.py`` imports gradio at module import time but only uses
it inside ``generate_dropdown_model_list``. A minimal stub lets us exercise the
pure hierarchy/sorting logic without a full Gradio install.
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
    def __init__(self, families_infos, family_of, names, parents, paths=None):
        self.families_infos = families_infos
        self._family_of = family_of
        self._names = names
        self._parents = parents
        self._paths = paths or {}

    def get_model_family(self, model_type, for_ui=True):
        return self._family_of[model_type]

    def get_model_name(self, model_type):
        return self._names[model_type]

    def get_parent_model_type(self, model_type):
        return self._parents[model_type]

    def get_model_def(self, model_type):
        return {"path": self._paths.get(model_type, "")}


class HierarchySortingTests(unittest.TestCase):
    def test_level_three_finetunes_sorted_a_to_z_with_base_first(self):
        rows = [
            ("Base", "base", "base"),
            ("Base Zebra", "z", "base"),
            ("Base Apple", "a", "base"),
        ]
        _parents, children = md.create_models_hierarchy(rows)
        values = [value for _label, value in children["base"]]
        self.assertEqual(values[0], "base")  # base entry pinned first
        self.assertEqual(values[1:], ["a", "z"])  # finetunes A-Z

    def test_base_models_sorted_a_to_z(self):
        rows = [
            ("Zeta Base", "zb", "zb"),
            ("Alpha Base", "ab", "ab"),
        ]
        parents, _children = md.create_models_hierarchy(rows)
        self.assertEqual([name for name, _pid in parents], ["Alpha Base", "Zeta Base"])

    def test_orphan_group_children_sorted_a_to_z(self):
        rows = [
            ("Group", "grp", "grp"),
            ("Charlie", "c", "grp"),
            ("Alpha", "a", "grp"),
            ("Bravo", "b", "grp"),
        ]
        _parents, children = md.create_models_hierarchy(rows)
        values = [value for _label, value in children["grp"]]
        self.assertEqual(values[0], "grp")
        self.assertEqual(values[1:], ["a", "b", "c"])


class FamilySortingTests(unittest.TestCase):
    def _deps(self):
        # 'zeta' has the lower configured order (1) but must still sort after 'alpha'.
        families_infos = {"zeta": (1, "Zeta"), "alpha": (2, "Alpha")}
        family_of = {"alpha_base": "alpha", "alpha_base_zebra": "alpha", "alpha_base_apple": "alpha", "zeta_base": "zeta"}
        names = {
            "alpha_base": "Alpha Base",
            "alpha_base_zebra": "Alpha Base Zebra",
            "alpha_base_apple": "Alpha Base Apple",
            "zeta_base": "Zeta Base",
        }
        parents = {"alpha_base": "alpha_base", "alpha_base_zebra": "alpha_base", "alpha_base_apple": "alpha_base", "zeta_base": "zeta_base"}
        paths = {"alpha_base_zebra": os.path.join("finetunes", "z.json"), "alpha_base_apple": os.path.join("finetunes", "a.json")}
        return FakeDeps(families_infos, family_of, names, parents, paths)

    def test_families_sorted_a_to_z_not_by_config_order(self):
        tree = md.create_models_selector_hierarchy(self._deps(), ["alpha_base", "alpha_base_zebra", "alpha_base_apple", "zeta_base"])
        self.assertEqual([folder["name"] for folder in tree["folders"]], ["Alpha", "Zeta"])

    def test_family_level_three_items_sorted_a_to_z(self):
        tree = md.create_models_selector_hierarchy(self._deps(), ["alpha_base", "alpha_base_zebra", "alpha_base_apple", "zeta_base"])
        alpha_folder = tree["folders"][0]
        parent_folder = alpha_folder["folders"][0]
        items = parent_folder["items"]
        values = [item["value"] for item in items]
        self.assertEqual(values[0], "alpha_base")  # base pinned first
        self.assertEqual(values[1:], ["alpha_base_apple", "alpha_base_zebra"])

    def test_get_sorted_dropdown_families_sorted_a_to_z(self):
        deps = self._deps()
        families, _models, _finetunes = md.get_sorted_dropdown(
            deps, ["alpha_base", "zeta_base"], None, "alpha_base", three_levels=True
        )
        self.assertEqual(families, [("Alpha", "alpha"), ("Zeta", "zeta")])


class UnknownFamilyResilienceTests(unittest.TestCase):
    """A model with no resolvable family must not blank the selector.

    Regression: a hidden/malformed model def (no ``architecture``) makes
    ``get_model_family`` return the ``"unknown"`` sentinel, which is not a key in
    ``families_infos``; indexing it directly raised ``KeyError`` while building
    the dropdown and the whole selector disappeared.
    """

    def _deps(self):
        families_infos = {"alpha": (2, "Alpha")}
        family_of = {"alpha_base": "alpha", "bad_model": "unknown"}
        names = {"alpha_base": "Alpha Base", "bad_model": "Bad Model"}
        parents = {"alpha_base": "alpha_base", "bad_model": None}
        return FakeDeps(families_infos, family_of, names, parents)

    def test_unknown_family_model_is_excluded_without_raising(self):
        deps = self._deps()
        families, _models, finetunes = md.get_sorted_dropdown(
            deps, ["alpha_base", "bad_model"], "alpha", "alpha_base", three_levels=True
        )
        self.assertEqual(families, [("Alpha", "alpha")])
        self.assertEqual([value for _label, value in finetunes], ["alpha_base"])

    def test_current_model_with_unknown_family_does_not_raise(self):
        deps = self._deps()
        families, _models, finetunes = md.get_sorted_dropdown(
            deps, ["alpha_base", "bad_model"], "unknown", "bad_model", three_levels=True
        )
        self.assertEqual(families, [("Alpha", "alpha")])
        self.assertEqual(list(finetunes), [])


if __name__ == "__main__":
    unittest.main()
