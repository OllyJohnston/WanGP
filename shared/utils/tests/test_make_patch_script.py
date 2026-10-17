"""Integration tests for the WanGP Enhanced release-asset scripts.

These tests drive ``scripts/make-patch.ps1`` (Windows) or ``scripts/make-patch.sh``
(POSIX) end to end. They regenerate the patch against the recorded upstream base
commit and then prove, inside a throwaway clone, that

  * the patch contains no carriage returns (the historical corruption bug that
    made ``git apply`` fail on every hunk), and
  * ``git apply`` reproduces the fork exactly when replayed onto the base commit.

They skip cleanly when git, the base commit, or a suitable shell is unavailable,
so the fast stdlib-only suite stays portable.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPSTREAM_BASE_FILE = os.path.join(REPO_ROOT, "UPSTREAM_BASE")


def _load_module(name, *relative_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, *relative_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

GIT = shutil.which("git")
PWSH = shutil.which("pwsh") or shutil.which("powershell")
BASH = shutil.which("bash")


def _read_upstream_base():
    with open(UPSTREAM_BASE_FILE, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise AssertionError("UPSTREAM_BASE contains no commit")


def _read_version():
    with open(os.path.join(REPO_ROOT, "wgp.py"), encoding="utf-8") as fh:
        match = re.search(r'WanGP_version\s*=\s*"([^"]+)"', fh.read())
    if not match:
        raise AssertionError("WanGP_version not found in wgp.py")
    return match.group(1)


def _git(*args, cwd=REPO_ROOT):
    return subprocess.run(
        [GIT, *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def _git_bytes(*args, cwd=REPO_ROOT):
    return subprocess.run(
        [GIT, *args], cwd=cwd, capture_output=True, check=True
    ).stdout


def _run_script(out_dir):
    if PWSH:
        cmd = [
            PWSH, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", os.path.join(REPO_ROOT, "scripts", "make-patch.ps1"),
            "-OutDir", out_dir,
        ]
    elif BASH:
        cmd = [
            BASH, os.path.join(REPO_ROOT, "scripts", "make-patch.sh"),
            "--out-dir", out_dir,
        ]
    else:
        raise unittest.SkipTest("neither PowerShell nor bash is available")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"make-patch failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )


@unittest.skipUnless(GIT, "git is not available")
class MakePatchScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _read_upstream_base()
        if subprocess.run(
            [GIT, "cat-file", "-e", cls.base], cwd=REPO_ROOT
        ).returncode != 0:
            raise unittest.SkipTest(f"upstream base {cls.base} is not in this clone")
        cls.version = _read_version()
        cls.tmp = tempfile.mkdtemp(prefix="wangp-make-patch-")
        cls.out_dir = os.path.join(cls.tmp, "dist")
        _run_script(cls.out_dir)

        # Discover the artefacts instead of recomputing the (variable-length)
        # short SHA, so the test asserts the naming convention without duplicating
        # git's abbreviation logic.
        patches = sorted(glob.glob(os.path.join(cls.out_dir, "wangp-enhanced-*.patch")))
        overlays = sorted(
            glob.glob(os.path.join(cls.out_dir, "wangp-enhanced-*-overlay.zip"))
        )
        if len(patches) != 1 or len(overlays) != 1:
            raise AssertionError(
                f"expected exactly one patch and one overlay zip, got {patches} / {overlays}"
            )
        cls.patch_path = patches[0]
        cls.zip_path = overlays[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_asset_names_carry_version_and_base(self):
        patch_name = os.path.basename(self.patch_path)
        expected_prefix = f"wangp-enhanced-v{self.version}-vs-{self.base[:7]}"
        self.assertTrue(
            patch_name.startswith(expected_prefix),
            f"{patch_name!r} does not start with {expected_prefix!r}",
        )
        self.assertEqual(
            os.path.basename(self.zip_path),
            os.path.basename(self.patch_path)[: -len(".patch")] + "-overlay.zip",
        )

    def test_patch_uses_unix_line_endings(self):
        with open(self.patch_path, "rb") as fh:
            data = fh.read()
        self.assertTrue(data, "patch is empty")
        self.assertNotIn(
            b"\r", data, "patch contains CR bytes - git apply would fail on every hunk"
        )

    def test_overlay_zip_lists_changed_files_only(self):
        changed = set(
            _git(
                "diff", "--name-only", "--diff-filter=ACMR", self.base, "HEAD"
            ).split()
        )
        self.assertTrue(changed, "no changed files relative to base")
        with zipfile.ZipFile(self.zip_path) as zf:
            names = {n for n in zf.namelist() if not n.endswith("/")}
        self.assertEqual(names, changed)

    def test_patch_replays_fork_onto_base(self):
        clone = os.path.join(self.tmp, "clone")
        subprocess.run(
            [
                GIT, "clone", "--quiet", "--shared", "--no-checkout",
                "-c", "core.autocrlf=false", REPO_ROOT, clone,
            ],
            check=True, capture_output=True,
        )
        _git("checkout", "--quiet", "--detach", self.base, cwd=clone)
        _git("apply", "--check", self.patch_path, cwd=clone)
        _git("apply", self.patch_path, cwd=clone)
        for path in _git("diff", "--name-only", self.base, "HEAD").split():
            applied = open(os.path.join(clone, path), "rb").read()
            expected = _git_bytes("show", f"HEAD:{path}")
            self.assertEqual(applied, expected, f"mismatch after apply: {path}")


class OverlayZipHelperTests(unittest.TestCase):
    """The overlay archive must keep relative paths, not flatten to basenames."""

    def setUp(self):
        self.helper = _load_module("wangp_make_overlay_zip", "scripts", "make_overlay_zip.py")
        self.root = tempfile.mkdtemp(prefix="wangp-overlay-")
        os.makedirs(os.path.join(self.root, "shared", "utils", "tests"))
        for relative in ("top.py", "shared/utils/deep.py", "shared/utils/tests/t.py"):
            with open(os.path.join(self.root, *relative.split("/")), "w", encoding="utf-8") as fh:
                fh.write(f"# {relative}\n")
        self.zip_path = os.path.join(self.root, "out.zip")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_preserves_directory_structure(self):
        written = self.helper.build(
            self.zip_path,
            ["top.py", "shared/utils/deep.py", "shared/utils/tests/t.py"],
            root=self.root,
        )
        self.assertEqual(len(written), 3)
        with zipfile.ZipFile(self.zip_path) as archive:
            names = set(archive.namelist())
        self.assertEqual(
            names, {"top.py", "shared/utils/deep.py", "shared/utils/tests/t.py"}
        )

    def test_skips_missing_files_and_rejects_empty_input(self):
        self.helper.build(self.zip_path, ["top.py", "does/not/exist.py"], root=self.root)
        with zipfile.ZipFile(self.zip_path) as archive:
            self.assertEqual(set(archive.namelist()), {"top.py"})
        with self.assertRaises(ValueError):
            self.helper.build(self.zip_path, ["does/not/exist.py"], root=self.root)


if __name__ == "__main__":
    unittest.main()
