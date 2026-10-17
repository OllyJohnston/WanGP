"""Unit tests for the wgp.py scan glue, extracted into a torch-free module.

``wgp.py`` cannot be imported without torch/gradio, so the integration logic
lives in ``model_scan_integration`` and is exercised here with fakes. The
remaining structural guarantees (that ``wgp.py`` actually delegates to this
module) are checked by ``test_wgp_scan_wiring.py``.
"""

import importlib.util
import os
import sys
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
    from shared.utils import model_scan_integration as integ
except Exception:
    integ = _load("model_scan_integration")

try:
    from shared.utils import model_scan_provider as provider
except Exception:
    provider = _load("model_scan_provider")


class FakeRuntime:
    """Minimal stand-in for ``model_scan_runtime`` recording the calls made."""

    ROLE_CHECKPOINT = "checkpoint"
    ROLE_TEXT_ENCODER = "text_encoder"
    ROLE_VAE = "vae"

    def __init__(self, enabled=True, bind=True, index=None, bind_result=None, raises=None):
        self._enabled = enabled
        self._bind = bind
        self._index = index
        self._bind_result = bind_result
        self._raises = raises
        self.get_index_calls = []
        self.bind_calls = []

    def is_enabled(self):
        return self._enabled

    def is_bind_enabled(self):
        return self._bind

    def get_index(self, force=False, wait=True):
        self.get_index_calls.append((force, wait))
        return self._index

    def attempt_bind(self, filename, role=None):
        self.bind_calls.append((filename, role))
        if isinstance(self._raises, Exception):
            raise self._raises
        return self._bind_result


class FakeBindResult(dict):
    def __init__(self, path=None, reason="no_match", candidates=()):
        super().__init__(path=path, reason=reason, candidates=list(candidates))
        self.path = path
        self.reason = reason
        self.candidates = list(candidates)


class FakeProvider:
    SCANNED_METADATA_KEY = "scanned"
    SCAN_PATH_KEY = "scan_path"

    def __init__(self):
        self.stripped = []
        self.injected = []

    def scanned_json_path(self, model_id):
        return os.path.join("finetunes", f"{model_id}.json")

    def strip_scanned_models(self, models_def):
        self.stripped.append(dict(models_def))
        return 0

    def inject_scanned_models(self, models_def, index, *, initialiser=None):
        self.injected.append((dict(models_def), index, initialiser))
        return ["scan_a"]


class RoleMappingTests(unittest.TestCase):
    def test_file_type_two_is_a_text_encoder(self):
        runtime = FakeRuntime()
        self.assertEqual(integ.role_for_file_type(2, runtime), runtime.ROLE_TEXT_ENCODER)

    def test_file_type_zero_and_one_are_checkpoints(self):
        runtime = FakeRuntime()
        self.assertEqual(integ.role_for_file_type(0, runtime), runtime.ROLE_CHECKPOINT)
        self.assertEqual(integ.role_for_file_type(1, runtime), runtime.ROLE_CHECKPOINT)

    def test_vae_prop_maps_to_vae_role(self):
        runtime = FakeRuntime()
        self.assertEqual(integ.role_for_asset_prop("VAE_URLs", runtime), runtime.ROLE_VAE)

    def test_other_props_are_unconstrained(self):
        runtime = FakeRuntime()
        self.assertIsNone(integ.role_for_asset_prop("preload_URLs", runtime))


class InitialiseScannedModelTests(unittest.TestCase):
    def test_synthetic_path_and_metadata_are_set(self):
        recorded = {}

        def init_model_def(model_id, model_def):
            recorded["model_id"] = model_id
            recorded["model_def"] = dict(model_def)
            return model_def

        model_def = {"architecture": "krea2_turbo", "scan_path": "D:\\ckpts\\a.safetensors"}
        result = integ.initialise_scanned_model(
            "scan_a", model_def, provider=provider, init_model_def=init_model_def
        )
        self.assertEqual(result["path"], provider.scanned_json_path("scan_a"))
        self.assertTrue(result["metadata"]["finetune"])
        self.assertTrue(result["metadata"][provider.SCANNED_METADATA_KEY])
        self.assertEqual(result["metadata"]["local_path"], "D:\\ckpts\\a.safetensors")
        # The handler initialiser must see the synthetic path, not the raw def.
        self.assertEqual(recorded["model_def"]["path"], provider.scanned_json_path("scan_a"))

    def test_input_model_def_is_not_mutated(self):
        model_def = {"scan_path": "D:\\ckpts\\a.safetensors"}
        integ.initialise_scanned_model(
            "scan_a", model_def, provider=provider, init_model_def=lambda mid, md: md
        )
        self.assertNotIn("path", model_def)
        self.assertNotIn("metadata", model_def)


class ReportScanStatusTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.printer = self.lines.append

    def test_timed_out_is_reported(self):
        integ.report_scan_status({"stats": {"timed_out": True}}, printer=self.printer)
        self.assertTrue(any("longer than expected" in line for line in self.lines))

    def test_warnings_are_listed_and_capped(self):
        warnings = [f"loop: D:\\p{i}" for i in range(25)]
        integ.report_scan_status({"stats": {"warnings": warnings}}, printer=self.printer)
        self.assertIn("25 path(s) skipped", self.lines[0])
        self.assertEqual(sum(1 for line in self.lines if line.startswith("  - ")), 11)
        self.assertTrue(self.lines[-1].endswith("and 15 more"))

    def test_clean_index_is_silent(self):
        integ.report_scan_status({"stats": {}}, printer=self.printer)
        self.assertEqual(self.lines, [])

    def test_non_dict_index_is_silent(self):
        integ.report_scan_status(None, printer=self.printer)
        self.assertEqual(self.lines, [])


class RegisterScannedModelsTests(unittest.TestCase):
    def test_disabled_strips_and_does_not_scan(self):
        runtime = FakeRuntime(enabled=False)
        fake_provider = FakeProvider()
        models_def = {"scan_a": {"metadata": {"scanned": True}}}
        result = integ.register_scanned_models(
            models_def, runtime=runtime, provider=fake_provider, initialiser=None, printer=lambda *_: None
        )
        self.assertEqual(result, [])
        self.assertEqual(len(fake_provider.stripped), 1)
        self.assertEqual(runtime.get_index_calls, [])
        self.assertEqual(fake_provider.injected, [])

    def test_enabled_scans_non_blocking_and_injects(self):
        runtime = FakeRuntime(enabled=True, index={"entries": {}, "stats": {}})
        fake_provider = FakeProvider()
        initialiser = lambda mid, md: md
        result = integ.register_scanned_models(
            {}, runtime=runtime, provider=fake_provider, initialiser=initialiser, printer=lambda *_: None
        )
        self.assertEqual(result, ["scan_a"])
        self.assertEqual(runtime.get_index_calls, [(False, False)])
        self.assertEqual(fake_provider.injected[0][2], initialiser)

    def test_errors_are_swallowed_and_reported(self):
        runtime = FakeRuntime(enabled=True, index=None)
        runtime.get_index = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        lines = []
        result = integ.register_scanned_models(
            {}, runtime=runtime, provider=FakeProvider(), initialiser=None, printer=lines.append
        )
        self.assertEqual(result, [])
        self.assertTrue(any("boom" in line for line in lines))

    def test_status_is_reported_when_an_index_exists(self):
        runtime = FakeRuntime(enabled=True, index={"entries": {}, "stats": {"warnings": ["loop: D:\\p"]}})
        lines = []
        integ.register_scanned_models(
            {}, runtime=runtime, provider=FakeProvider(), initialiser=None, printer=lines.append
        )
        self.assertTrue(any("D:\\p" in line for line in lines))


class TryBindLocalTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.printer = self.lines.append

    def test_blank_and_non_string_are_ignored(self):
        runtime = FakeRuntime()
        self.assertIsNone(integ.try_bind_local("", runtime=runtime, printer=self.printer))
        self.assertIsNone(integ.try_bind_local(None, runtime=runtime, printer=self.printer))
        self.assertEqual(runtime.bind_calls, [])

    def test_disabled_bind_returns_none(self):
        runtime = FakeRuntime(bind=False)
        self.assertIsNone(integ.try_bind_local("D:\\ckpts\\a.safetensors", runtime=runtime, printer=self.printer))
        self.assertEqual(runtime.bind_calls, [])

    def test_successful_bind_returns_path_and_logs(self):
        runtime = FakeRuntime(bind_result=FakeBindResult(path="D:\\local\\a.safetensors", reason="bound"))
        result = integ.try_bind_local(
            "https://hf.co/a.safetensors", role=runtime.ROLE_CHECKPOINT, runtime=runtime, printer=self.printer
        )
        self.assertEqual(result, "D:\\local\\a.safetensors")
        self.assertTrue(any("Reusing local file" in line for line in self.lines))
        self.assertEqual(runtime.bind_calls, [("https://hf.co/a.safetensors", runtime.ROLE_CHECKPOINT)])

    def test_quiet_reasons_are_not_logged(self):
        for reason in ("disabled", "no_index", "no_match"):
            self.lines = []
            runtime = FakeRuntime(bind_result=FakeBindResult(path=None, reason=reason))
            integ.try_bind_local("a.safetensors", runtime=runtime, printer=self.printer)
            self.assertEqual(self.lines, [], reason)

    def test_interesting_reason_is_logged_with_candidates(self):
        runtime = FakeRuntime(
            bind_result=FakeBindResult(path=None, reason="ambiguous", candidates=["D:\\a", "D:\\b"])
        )
        integ.try_bind_local("a.safetensors", runtime=runtime, printer=self.printer)
        self.assertEqual(len(self.lines), 1)
        self.assertIn("ambiguous", self.lines[0])
        self.assertIn(repr("D:\\a"), self.lines[0])

    def test_exceptions_are_swallowed_and_logged(self):
        runtime = FakeRuntime(raises=RuntimeError("kaboom"))
        self.assertIsNone(integ.try_bind_local("a.safetensors", runtime=runtime, printer=self.printer))
        self.assertTrue(any("kaboom" in line for line in self.lines))


if __name__ == "__main__":
    unittest.main()
