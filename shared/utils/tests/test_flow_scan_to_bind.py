"""End-to-end flow test: scan directory -> registry injection -> local bind -> no disk write.

Exercises the real scanner against a temporary ComfyUI-style layout using only
stdlib modules, so it runs without torch/gradio. It mirrors what
``wgp._inject_scanned_models`` does (scan -> ``provider.inject_scanned_models``
-> ``download_models`` binding) and asserts the four behaviours the feature was
built for: native registration, zero JSON written, download interception, and
clean promotion of a scanned model to a real finetune.
"""
import importlib.util
import json
import os
import struct
import sys
import tempfile
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
    from shared.utils import model_scan_provider as provider
    from shared.utils import model_scan_runtime as rt
except Exception:
    rt = _load("model_scan_runtime")
    provider = _load("model_scan_provider")


def write_safetensors(path, keys):
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    blob = json.dumps(header).encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)


def _fake_initialiser(model_id, model_def):
    """Mirror of ``wgp._initialise_scanned_model`` minus the handler-dependent ``init_model_def``."""
    model_def = dict(model_def)
    model_def["path"] = provider.scanned_json_path(model_id)
    metadata = model_def.setdefault("metadata", {})
    metadata["finetune"] = True
    metadata[provider.SCANNED_METADATA_KEY] = True
    metadata["local_path"] = model_def.get(provider.SCAN_PATH_KEY, "")
    return model_def


class ScanToBindFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.checkpoints = os.path.join(self.tmp, "models", "checkpoints")
        self.finetunes = os.path.join(self.tmp, "finetunes")
        os.makedirs(self.finetunes, exist_ok=True)
        self.target = os.path.join(self.checkpoints, "test_krea.safetensors")
        # ``blocks`` + ``qknorm`` are the Krea 2 header probes (model_signature.py).
        write_safetensors(self.target, ["blocks.0.attn.qknorm.weight"])
        rt.reset()
        rt.files_locator.clear_filename_resolvers()

    def tearDown(self):
        rt.reset()
        rt.files_locator.clear_filename_resolvers()
        self._tmp.cleanup()

    def _configure(self):
        rt.configure(
            {
                "model_scan_paths": {
                    "checkpoints": [self.checkpoints],
                    "text_encoders": [],
                    "vae": [],
                    "loras": [],
                },
                "model_scan_bind_local": True,
            },
            cache_path=os.path.join(self.tmp, "cache", "idx.json"),
        )

    def _installed_models_def(self):
        return {"krea2_turbo": {"architecture": "krea2_turbo", "path": "defaults/krea2_turbo.json"}}

    def test_scan_registers_model_under_family_without_writing_json(self):
        self._configure()
        index = rt.get_index(force=True, timeout=5)
        models_def = self._installed_models_def()

        added = provider.inject_scanned_models(models_def, index, initialiser=_fake_initialiser)

        self.assertEqual(added, ["scan_test_krea"])
        model_def = models_def["scan_test_krea"]
        self.assertEqual(model_def["architecture"], "krea2_turbo")
        self.assertEqual(model_def["finetune_source_model"], "krea2_turbo")
        self.assertTrue(model_def["metadata"]["finetune"])
        self.assertTrue(model_def["metadata"][provider.SCANNED_METADATA_KEY])

        # Nothing may be written to disk by the scan/injection.
        self.assertEqual(os.listdir(self.finetunes), [])
        self.assertFalse(os.path.exists(provider.scanned_json_path("scan_test_krea")))

    def test_download_request_binds_to_local_file(self):
        self._configure()
        rt.get_index(force=True, timeout=5)
        url = "https://huggingface.co/org/repo/resolve/main/test_krea.safetensors"

        result = rt.attempt_bind(url, role=provider.ROLE_CHECKPOINT)

        self.assertTrue(result, result.reason)
        self.assertEqual(os.path.abspath(result.path), os.path.abspath(self.target))

    def test_locator_resolver_binds_without_network(self):
        self._configure()
        rt.get_index(force=True, timeout=5)
        rt.install_locator_resolver()

        resolved = rt.files_locator.locate_file("test_krea.safetensors", error_if_none=False)

        self.assertEqual(os.path.abspath(resolved), os.path.abspath(self.target))

    def test_promoted_finetune_materialises_json_and_suppresses_scan_entry(self):
        self._configure()
        index = rt.get_index(force=True, timeout=5)
        models_def = self._installed_models_def()
        provider.inject_scanned_models(models_def, index, initialiser=_fake_initialiser)
        self.assertIn("scan_test_krea", models_def)

        # The editor promotes the scanned model to a real (here renamed) finetune:
        # it writes JSON and points URLs at the same local weight file.
        promoted_path = os.path.join(self.finetunes, "my_promoted.json")
        promoted_def = {
            "architecture": "krea2_turbo",
            "URLs": [os.path.abspath(self.target)],
            "metadata": {"finetune": True},
        }
        with open(promoted_path, "w", encoding="utf-8") as handle:
            json.dump({"model": promoted_def}, handle)
        models_def["my_promoted"] = promoted_def
        self.assertTrue(os.path.isfile(promoted_path))

        provider.inject_scanned_models(models_def, index, initialiser=_fake_initialiser)

        self.assertIn("my_promoted", models_def)
        self.assertNotIn("scan_test_krea", models_def)


if __name__ == "__main__":
    unittest.main()
