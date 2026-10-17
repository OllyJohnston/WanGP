import importlib.util
import os
import sys
import tempfile
import unittest


def _load(name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lb = _load("local_browse")

CHECKPOINT_EXTENSIONS = {".safetensors", ".sft", ".ckpt"}


class ListSubfoldersTests(unittest.TestCase):
    def test_lists_only_immediate_subfolders_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("zebra", "apple", "Mango"):
                os.makedirs(os.path.join(tmp, name))
            open(os.path.join(tmp, "file.safetensors"), "w").close()
            result = lb.list_subfolders(tmp)
            self.assertEqual([name for name, _ in result], ["apple", "Mango", "zebra"])
            self.assertTrue(all(os.path.isdir(path) for _, path in result))

    def test_missing_or_file_directory_returns_empty(self):
        self.assertEqual(lb.list_subfolders(""), [])
        self.assertEqual(lb.list_subfolders(os.path.join("no", "such", "dir")), [])
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "a.safetensors")
            open(target, "w").close()
            self.assertEqual(lb.list_subfolders(target), [])


class ListFilesTests(unittest.TestCase):
    def test_filters_by_extension_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("b.safetensors", "a.sft", "note.txt", "c.ckpt"):
                open(os.path.join(tmp, name), "w").close()
            os.makedirs(os.path.join(tmp, "subdir"))
            result = lb.list_files(tmp, CHECKPOINT_EXTENSIONS)
            self.assertEqual([name for name, _ in result], ["a.sft", "b.safetensors", "c.ckpt"])

    def test_no_extensions_returns_all_files_not_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.txt"), "w").close()
            open(os.path.join(tmp, "b.bin"), "w").close()
            os.makedirs(os.path.join(tmp, "folder"))
            result = lb.list_files(tmp, None)
            self.assertEqual([name for name, _ in result], ["a.txt", "b.bin"])

    def test_missing_directory_returns_empty(self):
        self.assertEqual(lb.list_files("", CHECKPOINT_EXTENSIONS), [])
        self.assertEqual(lb.list_files(os.path.join("no", "such", "dir"), CHECKPOINT_EXTENSIONS), [])


class NormalizeExtensionsTests(unittest.TestCase):
    def test_normalises_various_forms(self):
        self.assertEqual(lb.normalize_extensions([".SafeTensors", "sft", ""]), {".safetensors", ".sft"})
        self.assertEqual(lb.normalize_extensions(None), set())


class FolderOfValueTests(unittest.TestCase):
    def test_file_returns_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "clip_l.safetensors")
            open(target, "w").close()
            self.assertEqual(lb.folder_of_value(target), os.path.abspath(tmp))

    def test_directory_returns_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(lb.folder_of_value(tmp), os.path.abspath(tmp))

    def test_url_and_empty_return_none(self):
        self.assertIsNone(lb.folder_of_value("https://huggingface.co/org/repo/resolve/main/x.safetensors"))
        self.assertIsNone(lb.folder_of_value(""))
        self.assertIsNone(lb.folder_of_value(None))


if __name__ == "__main__":
    unittest.main()
