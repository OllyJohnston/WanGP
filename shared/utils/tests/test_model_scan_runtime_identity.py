import importlib.util
import os
import sys
import types
import unittest


def _load(name):
    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(os.path.dirname(here), name + ".py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    from shared.utils import model_scan_runtime as rt
except Exception:
    rt = _load("model_scan_runtime")

SIBLINGS = ("model_scanner", "model_local_bind", "model_scan_config", "model_architecture_map", "files_locator")


class SiblingIdentityTests(unittest.TestCase):
    """W-2: each sibling must have a single identity registered under its
    canonical ``shared.utils.<name>`` key, so no caller ever sees a shadow copy."""

    def test_siblings_are_registered_under_canonical_name(self):
        for name in SIBLINGS:
            module = getattr(rt, name)
            self.assertIs(
                sys.modules.get("shared.utils." + name), module,
                f"{name} is not registered under its canonical package name",
            )

    def test_load_sibling_adopts_existing_short_module_without_shadowing(self):
        name = "model_scan_config"
        canonical = "shared.utils." + name
        sentinel = types.ModuleType(name)
        saved_canonical = sys.modules.pop(canonical, None)
        saved_short = sys.modules.get(name)
        sys.modules[name] = sentinel
        try:
            self.assertIs(rt._load_sibling(name), sentinel)
            self.assertIs(sys.modules[canonical], sentinel)
        finally:
            if saved_canonical is not None:
                sys.modules[canonical] = saved_canonical
            else:
                sys.modules.pop(canonical, None)
            if saved_short is not None:
                sys.modules[name] = saved_short
            else:
                sys.modules.pop(name, None)

    def test_import_does_not_swallow_unexpected_errors(self):
        original = rt.importlib.import_module

        def boom(_name):
            raise RuntimeError("boom")

        rt.importlib.import_module = boom
        try:
            with self.assertRaises(RuntimeError):
                rt._import("model_scan_config")
        finally:
            rt.importlib.import_module = original

    def test_import_falls_back_when_package_unavailable(self):
        original = rt.importlib.import_module

        def missing(_name):
            raise ModuleNotFoundError("no package")

        rt.importlib.import_module = missing
        try:
            module = rt._import("model_scan_config")
        finally:
            rt.importlib.import_module = original
        self.assertIs(module, sys.modules["shared.utils.model_scan_config"])


if __name__ == "__main__":
    unittest.main()
