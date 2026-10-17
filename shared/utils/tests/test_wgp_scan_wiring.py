"""Structural tests that ``wgp.py`` actually delegates its scan glue correctly.

``wgp.py`` cannot be imported here (torch/gradio), so these tests parse its source
with :mod:`ast` and assert the wiring contract: which integration functions each
hook calls, and with which injected dependencies. Behaviour itself is covered by
``test_model_scan_integration``.
"""

import ast
import os
import unittest


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


WGP_PATH = os.path.join(_repo_root(), "wgp.py")


def _dotted(node):
    """Return the dotted name for a Call/Attribute/Name node, else ``None``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class WgpScanWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WGP_PATH, "r", encoding="utf-8") as handle:
            cls.tree = ast.parse(handle.read(), filename=WGP_PATH)

    def func(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"wgp.py no longer defines top-level function {name!r}")

    def calls(self, node, dotted_name):
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and _dotted(call.func) == dotted_name
        ]

    @staticmethod
    def keywords(call):
        return {kw.arg: kw.value for kw in call.keywords}

    def name_of(self, node):
        return getattr(node, "id", None)

    def test_wgp_imports_the_integration_module(self):
        imported = set()
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        self.assertIn("model_scan_integration", imported)

    def test_inject_delegates_with_injected_dependencies(self):
        calls = self.calls(self.func("_inject_scanned_models"), "model_scan_integration.register_scanned_models")
        self.assertEqual(len(calls), 1)
        keywords = self.keywords(calls[0])
        self.assertEqual(self.name_of(keywords.get("initialiser")), "_initialise_scanned_model")
        self.assertEqual(self.name_of(keywords.get("runtime")), "model_scan_runtime")
        self.assertEqual(self.name_of(keywords.get("provider")), "model_scan_provider")

    def test_initialiser_delegates_with_handler_injection(self):
        calls = self.calls(self.func("_initialise_scanned_model"), "model_scan_integration.initialise_scanned_model")
        self.assertEqual(len(calls), 1)
        keywords = self.keywords(calls[0])
        self.assertEqual(self.name_of(keywords.get("provider")), "model_scan_provider")
        self.assertEqual(self.name_of(keywords.get("init_model_def")), "init_model_def")

    def test_try_scan_bind_delegates(self):
        calls = self.calls(self.func("_try_scan_bind"), "model_scan_integration.try_bind_local")
        self.assertEqual(len(calls), 1)

    def test_report_scan_status_delegates(self):
        calls = self.calls(self.func("_report_scan_status"), "model_scan_integration.report_scan_status")
        self.assertEqual(len(calls), 1)

    def test_refresh_model_defs_injects_scanned_models(self):
        self.assertTrue(self.calls(self.func("refresh_model_defs"), "_inject_scanned_models"))

    def test_download_models_binds_with_role_mapping(self):
        node = self.func("download_models")
        self.assertGreaterEqual(len(self.calls(node, "_try_scan_bind")), 2)
        self.assertTrue(self.calls(node, "model_scan_integration.role_for_file_type"))
        self.assertTrue(self.calls(node, "model_scan_integration.role_for_asset_prop"))

    def test_startup_configures_and_installs_the_runtime(self):
        self.assertTrue(self.calls(self.tree, "model_scan_runtime.configure"))
        self.assertTrue(self.calls(self.tree, "model_scan_runtime.install_locator_resolver"))

    def test_non_blocking_scan_is_owned_by_the_integration_module(self):
        # wgp.py must not join the scan thread itself; that policy lives in register_scanned_models.
        self.assertEqual(self.calls(self.func("_inject_scanned_models"), "model_scan_runtime.get_index"), [])


if __name__ == "__main__":
    unittest.main()
