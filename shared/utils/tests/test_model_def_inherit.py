import importlib.util
import os
import sys
import unittest


def _load(name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mdi = _load("model_def_inherit")


class SeedMissingFieldsTests(unittest.TestCase):
    def test_fills_missing_url_from_first_model_type(self):
        defs = {"scan_krea": {"URLs": ["D:/ckpts/realDream.safetensors"], "name": "realDream"}, "krea2_turbo": {"URLs": ["https://hf/Krea2Turbo.safetensors"]}}
        seeded = mdi.seed_missing_fields({}, ["scan_krea", "krea2_turbo"], defs.get)
        self.assertEqual(seeded["URLs"], ["D:/ckpts/realDream.safetensors"])
        self.assertEqual(seeded["name"], "realDream")

    def test_never_overwrites_existing_values(self):
        defs = {"scan_krea": {"URLs": ["D:/ckpts/realDream.safetensors"]}}
        seeded = mdi.seed_missing_fields({"URLs": ["keep-me"], "name": "mine"}, ["scan_krea"], defs.get)
        self.assertEqual(seeded["URLs"], ["keep-me"])
        self.assertEqual(seeded["name"], "mine")

    def test_falls_back_to_later_model_type(self):
        defs = {"scan_krea": {}, "krea2_turbo": {"URLs": ["https://hf/Krea2Turbo.safetensors"]}}
        seeded = mdi.seed_missing_fields({}, ["scan_krea", "krea2_turbo"], defs.get)
        self.assertEqual(seeded["URLs"], ["https://hf/Krea2Turbo.safetensors"])

    def test_missing_definitions_and_empty_values_ignored(self):
        defs = {"a": None, "b": {}, "c": {"URLs": []}}
        seeded = mdi.seed_missing_fields({}, ["a", "b", "c", None, ""], defs.get)
        self.assertEqual(seeded, {})

    def test_returns_new_dict_without_mutating_input(self):
        original = {"name": "x"}
        seeded = mdi.seed_missing_fields(original, ["m"], lambda _mt: {"URLs": ["u"]})
        self.assertEqual(original, {"name": "x"})
        self.assertEqual(seeded["URLs"], ["u"])
        self.assertIsNot(seeded, original)

    def test_none_model_values_handled(self):
        seeded = mdi.seed_missing_fields(None, ["m"], lambda _mt: {"name": "n"})
        self.assertEqual(seeded["name"], "n")

    def test_custom_fields_restrict_scope(self):
        defs = {"m": {"URLs": ["u"], "name": "n"}}
        seeded = mdi.seed_missing_fields({}, ["m"], defs.get, fields=("URLs",))
        self.assertEqual(seeded, {"URLs": ["u"]})


if __name__ == "__main__":
    unittest.main()
