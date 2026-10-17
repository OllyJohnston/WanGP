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
    from shared.utils import model_scan_runtime as rt
except Exception:
    rt = _load("model_scan_runtime")

mb = sys.modules["model_local_bind"]
config = sys.modules["model_scan_config"]


def write_safetensors(path, keys):
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    blob = json.dumps(header).encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)


def server_config(checkpoints=None, text_encoders=None, vae=None, bind=False):
    return {
        "model_scan_paths": {
            "checkpoints": checkpoints or [],
            "text_encoders": text_encoders or [],
            "vae": vae or [],
            "loras": [],
        },
        "model_scan_bind_local": bind,
    }


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        rt.reset()
        rt.files_locator.clear_filename_resolvers()

    tearDown = setUp

    def test_disabled_by_default(self):
        rt.configure(server_config())
        self.assertFalse(rt.is_enabled())
        self.assertFalse(rt.is_bind_enabled())
        self.assertIsNone(rt.resolve_basename("A.safetensors"))
        self.assertEqual(rt.attempt_bind("A.safetensors").reason, mb.REASON_DISABLED)

    def test_paths_without_bind_are_inert(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "A.safetensors"), ["x.y"])
            rt.configure(server_config(checkpoints=[tmp]))
            self.assertTrue(rt.is_enabled())
            self.assertFalse(rt.is_bind_enabled())
            self.assertIsNone(rt.resolve_basename("A.safetensors"))

    def test_bind_resolves_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "A.safetensors")
            write_safetensors(target, ["x.y"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            self.assertTrue(rt.is_bind_enabled())
            self.assertEqual(rt.resolve_basename("A.safetensors"), target)

    def test_same_name_duplicates_collapse_to_single(self):
        # The scanner de-duplicates by filename, so two folders holding the same
        # name (e.g. junctions to one target) resolve to a single local file.
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "one", "A.safetensors"), ["x"])
            write_safetensors(os.path.join(tmp, "two", "A.safetensors"), ["x"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            resolved = rt.resolve_basename("A.safetensors")
            self.assertIsNotNone(resolved)
            self.assertEqual(os.path.basename(resolved), "A.safetensors")

    def test_index_is_cached_and_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "A.safetensors"), ["x"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            first = rt.get_index()
            self.assertIs(first, rt.get_index())
            self.assertIsNot(first, rt.get_index(force=True))

    def test_index_persists_to_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "A.safetensors"), ["x"])
            cache = os.path.join(tmp, "idx.json")
            rt.configure(server_config(checkpoints=[tmp], bind=True), cache_path=cache)
            rt.get_index()
            self.assertTrue(os.path.isfile(cache))

    def test_attempt_bind_checkpoint_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "wan.safetensors")
            write_safetensors(target, ["x.y"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            result = rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/wan.safetensors", role="checkpoint")
            self.assertEqual(result.path, target)

    def test_attempt_bind_text_encoder_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "clip_l.safetensors")
            write_safetensors(target, ["text_model.encoder.layers.0.x"])
            rt.configure(server_config(text_encoders=[tmp], bind=True))
            result = rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/clip_l.safetensors", role="text_encoder")
            self.assertTrue(result)
            self.assertEqual(result.path, target)

    def test_attempt_bind_wrong_category_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            # File lives in the checkpoints category but is requested as a text encoder,
            # so the category/role guard must reject it.
            write_safetensors(os.path.join(tmp, "t5xxl_fp16.safetensors"), ["encoder.block.0.x"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            result = rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/t5xxl_fp16.safetensors", role=rt.ROLE_TEXT_ENCODER)
            self.assertFalse(result)
            self.assertEqual(result.reason, mb.REASON_INCOMPATIBLE)

    def test_attempt_bind_uses_category_for_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "wan_2.1_vae.safetensors")
            write_safetensors(target, ["decoder.upsamples.0.x", "encoder.downsamples.0.x"])
            rt.configure(server_config(vae=[tmp], bind=True))
            result = rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/wan_2.1_vae.safetensors", role=rt.ROLE_VAE)
            self.assertEqual(result.path, target)

    def test_attempt_bind_encoder_in_wrong_category_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "clip_l.safetensors"), ["text_model.encoder.layers.0.x"])
            rt.configure(server_config(vae=[tmp], bind=True))  # encoder file placed under the vae folder
            result = rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/clip_l.safetensors", role=rt.ROLE_TEXT_ENCODER)
            self.assertFalse(result)
            self.assertEqual(result.reason, mb.REASON_INCOMPATIBLE)

    def test_expected_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "A.safetensors"), ["x"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            self.assertFalse(rt.attempt_bind("A.safetensors", expected_size=10 ** 12))

    def test_remote_size_used_from_hf_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "A.safetensors")
            write_safetensors(target, ["x"])
            size = os.path.getsize(target)
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            rt._hf_repo_size_cache["org/repo"] = {"A.safetensors": size}
            bound = rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/A.safetensors")
            self.assertEqual(bound.path, target)
            # A wrong remote size must refuse the bind.
            rt._hf_repo_size_cache["org/repo"] = {"A.safetensors": size + 1}
            self.assertFalse(rt.attempt_bind("https://huggingface.co/org/repo/resolve/main/A.safetensors"))

    def test_locator_resolver_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "A.safetensors")
            write_safetensors(target, ["x"])
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            rt.install_locator_resolver()
            self.assertEqual(rt.files_locator.locate_file("A.safetensors", error_if_none=False), target)
            rt.uninstall_locator_resolver()
            self.assertIsNone(rt.files_locator.locate_file("A.safetensors", error_if_none=False))

    def test_locator_resolver_inactive_when_bind_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "A.safetensors"), ["x"])
            rt.configure(server_config(checkpoints=[tmp], bind=False))
            rt.install_locator_resolver()
            self.assertIsNone(rt.files_locator.locate_file("A.safetensors", error_if_none=False))

    def test_resolve_basename_refuses_non_weight_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Legacy pickles are indexed by name but must never bind: binding would
            # route an untrusted local file into torch.load (arbitrary code execution).
            with open(os.path.join(tmp, "legacy.ckpt"), "wb") as handle:
                handle.write(b"\x80\x02not-a-real-pickle")
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            self.assertIsNone(rt.resolve_basename("legacy.ckpt"))

    def test_resolve_basename_size_mismatch_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "A.safetensors")
            write_safetensors(target, ["x"])
            size = os.path.getsize(target)
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            url = "https://huggingface.co/org/repo/resolve/main/A.safetensors"
            rt._hf_repo_size_cache["org/repo"] = {"A.safetensors": size}
            self.assertEqual(rt.resolve_basename(url), target)
            rt._hf_repo_size_cache["org/repo"] = {"A.safetensors": size + 1}
            self.assertIsNone(rt.resolve_basename(url))

    def test_locator_resolver_sees_url_for_size_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "A.safetensors")
            write_safetensors(target, ["x"])
            size = os.path.getsize(target)
            rt.configure(server_config(checkpoints=[tmp], bind=True))
            rt.install_locator_resolver()
            url = "https://huggingface.co/org/repo/resolve/main/A.safetensors"
            rt._hf_repo_size_cache["org/repo"] = {"A.safetensors": size + 1}
            self.assertIsNone(rt.files_locator.locate_file(url, error_if_none=False))
            rt._hf_repo_size_cache["org/repo"] = {"A.safetensors": size}
            self.assertEqual(rt.files_locator.locate_file(url, error_if_none=False), target)

    def test_scan_roots_excludes_loras(self):
        rt.configure(server_config(checkpoints=["D:/c"], text_encoders=["D:/t"]))
        self.assertEqual(rt.scan_roots(), {"checkpoints": ["D:/c"], "text_encoders": ["D:/t"]})

    def test_parse_hf_url(self):
        self.assertEqual(
            rt.parse_hf_url("https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/text_encoders/umt5.safetensors"),
            ("DeepBeepMeep/Wan2.1", "text_encoders/umt5.safetensors"),
        )
        self.assertEqual(rt.parse_hf_url("https://huggingface.co/x/y/blob/main/A.safetensors"), ("x/y", "A.safetensors"))
        self.assertEqual(rt.parse_hf_url("D:/local/A.safetensors"), (None, None))

    def test_bind_local_config_normalisation(self):
        self.assertTrue(config.normalize_bind_local("1"))
        self.assertTrue(config.normalize_bind_local("true"))
        self.assertFalse(config.normalize_bind_local("no"))
        self.assertFalse(config.bind_local_from_server({}))
        self.assertTrue(config.bind_local_from_server({"model_scan_bind_local": True}))


class BoundedScanTests(unittest.TestCase):
    def setUp(self):
        rt.reset()
        self._real_scan = rt.model_scanner.scan_roots

    def tearDown(self):
        rt.model_scanner.scan_roots = self._real_scan
        rt.reset()

    def test_timeout_returns_partial_and_marks_timed_out(self):
        slow_started = []

        def slow_scan(roots, **kwargs):
            slow_started.append(True)
            progress = kwargs.get("on_progress")
            if progress is not None:
                progress("D:/c/A.safetensors", {"path": "D:/c/A.safetensors", "categories": ["checkpoints"]})
            import time as _time
            _time.sleep(0.3)
            return {"version": 1, "entries": {}, "stats": {}}

        rt.model_scanner.scan_roots = slow_scan
        with tempfile.TemporaryDirectory() as tmp:
            rt.configure(server_config(checkpoints=[tmp]))
            index = rt.get_index(force=True, timeout=0.02)
        self.assertTrue(index.get("stats", {}).get("partial"))
        self.assertTrue(rt.status()["timed_out"])
        # Background thread keeps going and eventually publishes the full index.
        thread = rt._state["scan_thread"]
        if thread is not None:
            thread.join(2.0)
        self.assertFalse(rt.status()["scanning"])

    def test_wait_false_returns_immediately(self):
        rt.model_scanner.scan_roots = lambda roots, **kwargs: {"version": 1, "entries": {}, "stats": {}}
        with tempfile.TemporaryDirectory() as tmp:
            rt.configure(server_config(checkpoints=[tmp]))
            index = rt.get_index(force=True, wait=False)
        self.assertIn("entries", index)

    def test_status_html_disabled(self):
        rt.configure(server_config())
        self.assertIn("no scan folders configured", rt.status_html())

    def test_status_html_enabled_mentions_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_safetensors(os.path.join(tmp, "clip_l.safetensors"), ["text_model.encoder.layers.0.x"])
            rt.configure(server_config(text_encoders=[tmp], bind=True))
            rt.warm(timeout=5.0)
            html = rt.status_html()
        self.assertIn("Folder scan:", html)
        self.assertIn("local reuse ON", html)
        self.assertNotIn("no scan folders configured", html)


class StatusWarningHtmlTests(unittest.TestCase):
    """The config UI status line must list skipped paths (escaped), not just a count."""

    def setUp(self):
        rt.reset()
        rt.files_locator.clear_filename_resolvers()
        self._tmp = tempfile.TemporaryDirectory()
        rt.configure(server_config(checkpoints=[self._tmp.name], bind=True))

    def tearDown(self):
        rt.reset()
        rt.files_locator.clear_filename_resolvers()
        self._tmp.cleanup()

    def test_warning_paths_are_rendered(self):
        rt._state["warnings"] = ["PermissionError: D:\\blocked\\a.safetensors"]
        html = rt.status_html()
        self.assertIn("PermissionError: D:\\blocked\\a.safetensors", html)
        self.assertIn("<li", html)

    def test_warning_paths_are_html_escaped(self):
        rt._state["warnings"] = ["loop: D:\\x<script>alert(1)</script>.safetensors"]
        html = rt.status_html()
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_warning_list_is_capped_with_overflow_marker(self):
        overflow = rt.STATUS_WARNING_LIMIT + 15
        rt._state["warnings"] = [f"loop: D:\\p{i}" for i in range(overflow)]
        html = rt.status_html()
        self.assertEqual(html.count("<li"), rt.STATUS_WARNING_LIMIT + 1)
        self.assertIn(f"{overflow - rt.STATUS_WARNING_LIMIT} more", html)

    def test_no_warning_list_when_clean(self):
        rt.warm(timeout=5.0)
        html = rt.status_html()
        self.assertNotIn("<li", html)

    def test_index_stats_warnings_are_included(self):
        rt._state["index"] = {"version": 1, "entries": {}, "stats": {"files": 1, "warnings": ["max_depth: D:\\deep"]}}
        html = rt.status_html()
        self.assertIn("max_depth: D:\\deep", html)


class CachePathTests(unittest.TestCase):
    def test_default_cache_path_uses_ignored_cache_subdir(self):
        path = rt.default_cache_path(os.path.join("D:" + os.sep, "WanGP"))
        self.assertEqual(os.path.basename(path), rt.INDEX_CACHE_FILENAME)
        self.assertEqual(os.path.basename(os.path.dirname(path)), rt.INDEX_CACHE_DIR)
        self.assertEqual(rt.INDEX_CACHE_DIR, "cache")


if __name__ == "__main__":
    unittest.main()
