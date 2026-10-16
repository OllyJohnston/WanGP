# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Unit tests for the read-only model scanner.

Loaded directly from file when the heavy ``shared.utils`` package cannot be
imported (no torch); the module is standard-library only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

try:  # normal import in a full environment
    from shared.utils import model_scanner as sc
except ImportError:  # dependency-light test environment (no torch)
    _MODULE_PATH = Path(__file__).resolve().parents[1] / "model_scanner.py"
    _SPEC = importlib.util.spec_from_file_location("model_scanner", _MODULE_PATH)
    sc = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sc)


def write_safetensors(path, header, payload=b""):
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(payload)
    return path


SDXL_KEYS = {
    "model.diffusion_model.input_blocks.0.0.weight": {},
    "conditioner.embedders.0.x.weight": {},
    "conditioner.embedders.1.x.weight": {},
}


class HeaderReadTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_round_trip(self):
        path = write_safetensors(self.dir / "a.safetensors", {"t.0.weight": {}, "%__metadata__%": {}},
                                 payload=b"\x00\x01\x02")
        header = sc.read_safetensors_header(path)
        self.assertIn("t.0.weight", header)

    def test_short_prefix_rejected(self):
        path = self.dir / "tiny.safetensors"
        path.write_bytes(b"\x01\x02\x03")
        with self.assertRaises(sc.SafetensorsHeaderError) as ctx:
            sc.read_safetensors_header(path)
        self.assertEqual(str(ctx.exception), "short_prefix")

    def test_empty_header_rejected(self):
        path = self.dir / "empty.safetensors"
        path.write_bytes(struct.pack("<Q", 0))
        with self.assertRaises(sc.SafetensorsHeaderError) as ctx:
            sc.read_safetensors_header(path)
        self.assertEqual(str(ctx.exception), "empty_header")

    def test_header_too_large_rejected_before_allocation(self):
        # header_len exceeds the cap but the file is big enough to claim it.
        path = self.dir / "big.safetensors"
        path.write_bytes(struct.pack("<Q", 4096) + b"\x00" * 4096)
        with self.assertRaises(sc.SafetensorsHeaderError) as ctx:
            sc.read_safetensors_header(path, max_header_bytes=1024)
        self.assertEqual(str(ctx.exception), "header_too_large")

    def test_header_exceeds_file_rejected(self):
        path = self.dir / "trunc.safetensors"
        path.write_bytes(struct.pack("<Q", 10_000) + b"{}")
        with self.assertRaises(sc.SafetensorsHeaderError) as ctx:
            sc.read_safetensors_header(path)
        self.assertEqual(str(ctx.exception), "header_exceeds_file")

    def test_bad_json_rejected(self):
        path = self.dir / "bad.safetensors"
        blob = b"{not json"
        path.write_bytes(struct.pack("<Q", len(blob)) + blob)
        with self.assertRaises(sc.SafetensorsHeaderError) as ctx:
            sc.read_safetensors_header(path)
        self.assertEqual(str(ctx.exception), "bad_json")


class InspectTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_safetensors_classified(self):
        path = write_safetensors(self.dir / "xl.safetensors", SDXL_KEYS)
        entry = sc.inspect_file(path)
        self.assertTrue(entry["inspected"])
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["signature"], "sdxl")
        self.assertEqual(entry["format"], sc.FORMAT_SAFETENSORS)
        self.assertEqual(entry["tensor_count"], 3)

    def test_pickle_file_never_inspected(self):
        path = self.dir / "legacy.ckpt"
        path.write_bytes(b"\x80\x04not a real pickle but must never be parsed")
        calls = {"n": 0}
        real = sc.read_safetensors_header

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise AssertionError("header reader must not be called for pickle files")

        sc.read_safetensors_header = boom
        try:
            entry = sc.inspect_file(path)
        finally:
            sc.read_safetensors_header = real
        self.assertEqual(calls["n"], 0)
        self.assertFalse(entry["inspected"])
        self.assertEqual(entry["format"], sc.FORMAT_PICKLE)

    def test_corrupt_safetensors_keeps_entry_with_error(self):
        path = self.dir / "corrupt.safetensors"
        path.write_bytes(b"\x01\x02\x03")
        entry = sc.inspect_file(path)
        self.assertTrue(entry["inspected"])
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["error"], "short_prefix")


class ScanRootsTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        write_safetensors(self.dir / "xl.safetensors", SDXL_KEYS)
        write_safetensors(self.dir / "zit.safetensors", {
            "layers.0.x": {}, "noise_refiner.0.x": {}, "cap_embedder.0.x": {},
        })
        (self.dir / "notes.txt").write_text("ignore me")

    def tearDown(self):
        self._dir.cleanup()

    def test_scan_and_index(self):
        cache = os.path.join(self.dir, "index.json")
        index = sc.scan_roots([self.dir], cache_path=cache, now=1000.0)
        self.assertEqual(index["stats"]["files"], 2)
        self.assertEqual(index["stats"]["inspected"], 2)
        self.assertTrue(os.path.isfile(cache))
        signatures = {entry["signature"] for entry in index["entries"].values()}
        self.assertEqual(signatures, {"sdxl", "z_image"})

    def test_same_filename_across_roots_indexed_once(self):
        # Mirrors/junctions expose one file under several roots; the filename is
        # indexed a single time (deterministic first match).
        with tempfile.TemporaryDirectory() as other:
            write_safetensors(os.path.join(other, "xl.safetensors"), SDXL_KEYS)
            index = sc.scan_roots([self.dir, other], now=1000.0)
        self.assertEqual(index["stats"]["duplicates"], 1)
        self.assertEqual(len(index["entries"]), 2)  # xl.safetensors + zit.safetensors
        names = sorted(os.path.basename(entry["path"]) for entry in index["entries"].values())
        self.assertEqual(names, ["xl.safetensors", "zit.safetensors"])

    def test_unchanged_files_reuse_cache(self):
        cache = os.path.join(self.dir, "index.json")
        sc.scan_roots([self.dir], cache_path=cache, now=1000.0)

        calls = {"n": 0}
        real = sc.read_safetensors_header

        def counting(path, max_header_bytes=sc.DEFAULT_MAX_HEADER_BYTES):
            calls["n"] += 1
            return real(path, max_header_bytes)

        sc.read_safetensors_header = counting
        try:
            index = sc.scan_roots([self.dir], cache_path=cache, now=2000.0)
        finally:
            sc.read_safetensors_header = real
        self.assertEqual(calls["n"], 0)
        self.assertEqual(index["stats"]["reused"], 2)
        self.assertEqual(index["stats"]["inspected"], 0)

    def test_changed_file_is_reinspected(self):
        cache = os.path.join(self.dir, "index.json")
        sc.scan_roots([self.dir], cache_path=cache, now=1000.0)

        target = self.dir / "xl.safetensors"
        write_safetensors(target, SDXL_KEYS)
        stat = os.stat(target)
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        calls = {"n": 0}
        real = sc.read_safetensors_header

        def counting(path, max_header_bytes=sc.DEFAULT_MAX_HEADER_BYTES):
            calls["n"] += 1
            return real(path, max_header_bytes)

        sc.read_safetensors_header = counting
        try:
            index = sc.scan_roots([self.dir], cache_path=cache, now=3000.0)
        finally:
            sc.read_safetensors_header = real
        self.assertEqual(calls["n"], 1)
        self.assertEqual(index["stats"]["inspected"], 1)

    def test_category_mapping_tagged(self):
        index = sc.scan_roots({"checkpoints": [self.dir]}, now=1000.0)
        for entry in index["entries"].values():
            self.assertIn("checkpoints", entry["categories"])

    def test_cancel_stops_scan(self):
        index = sc.scan_roots([self.dir], should_cancel=lambda: True, now=1000.0)
        self.assertTrue(index["stats"]["cancelled"])
        self.assertEqual(index["stats"]["files"], 0)

    def test_missing_root_is_skipped(self):
        index = sc.scan_roots([str(self.dir / "does-not-exist")], now=1000.0)
        self.assertEqual(index["stats"]["files"], 0)


class IndexPersistenceTests(unittest.TestCase):
    def test_load_missing_returns_empty(self):
        index = sc.load_index(os.path.join(tempfile.gettempdir(), "definitely-absent-xyz.json"))
        self.assertEqual(index["entries"], {})

    def test_load_rejects_version_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"version": 999, "entries": {"a": {}}}, handle)
            self.assertEqual(sc.load_index(path)["entries"], {})

    def test_summary_counts(self):
        index = {"entries": {
            "a": {"signature": "sdxl", "role": "checkpoint", "error": None},
            "b": {"signature": "sdxl", "role": "checkpoint", "error": "bad"},
            "c": {"signature": None, "role": None, "error": None},
        }}
        summary = sc.index_summary(index)
        self.assertEqual(summary["files"], 3)
        self.assertEqual(summary["by_signature"]["sdxl"], 2)
        self.assertEqual(summary["errors"], 1)


class WalkHardeningTests(unittest.TestCase):
    def _write(self, path, keys=("a.b",)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_safetensors(path, {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys})

    def test_max_depth_stops_descent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(os.path.join(tmp, "d1", "d2", "d3", "deep.safetensors"))
            index = sc.scan_roots([tmp], max_depth=1, now=1.0)
            self.assertEqual(index["stats"]["files"], 0)
            self.assertTrue(any(w.startswith("max_depth") for w in index["stats"]["warnings"]))

    def test_deadline_bounds_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(os.path.join(tmp, "a.safetensors"))
            index = sc.scan_roots([tmp], deadline=0.0, now=1.0)
            self.assertTrue(index["stats"]["timed_out"])
            self.assertEqual(index["stats"]["files"], 0)

    def test_permission_error_skips_directory(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            self._write(os.path.join(tmp, "good", "a.safetensors"))
            bad = os.path.join(tmp, "bad")
            os.makedirs(bad)
            real_scandir = os.scandir

            def fake_scandir(path, *args, **kwargs):
                if os.path.abspath(path) == os.path.abspath(bad):
                    raise PermissionError(13, "denied")
                return real_scandir(path, *args, **kwargs)

            with mock.patch("os.scandir", side_effect=fake_scandir):
                index = sc.scan_roots([tmp], now=1.0)
            self.assertEqual(index["stats"]["files"], 1)
            self.assertTrue(any("PermissionError" in w for w in index["stats"]["warnings"]))

    def test_symlink_loop_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(os.path.join(tmp, "a.safetensors"))
            try:
                os.symlink(tmp, os.path.join(tmp, "loop"), target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks not permitted on this platform")
            index = sc.scan_roots([tmp], max_depth=10, now=1.0)
            self.assertEqual(index["stats"]["files"], 1)

    def test_loop_guard_detects_repeated_directory_identity(self):
        # Deterministic, privilege-free equivalent of a junction/symlink cycle:
        # every directory reports the same (st_dev, st_ino), so the walk must
        # stop after the first one instead of descending forever.
        from unittest import mock

        class FakeStat:
            st_dev = 1
            st_ino = 42
            st_size = 0
            st_mtime_ns = 0

        warnings = []
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "a", "b", "c"))
            with mock.patch("os.stat", return_value=FakeStat()):
                visited = list(sc.walk_files(tmp, max_depth=10, on_warning=lambda path, reason: warnings.append(reason)))
        self.assertEqual(len(visited), 1)
        self.assertIn("loop", warnings)


if __name__ == "__main__":
    unittest.main()
