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
    from shared.utils import model_local_bind as mb
except Exception:
    mb = _load("model_local_bind")


def _entry(path, size=100, signature=None, role=None):
    return {"path": path, "size": size, "signature": signature, "role": role}


class BasenameTests(unittest.TestCase):
    def test_plain_name(self):
        self.assertEqual(mb.basename_of("umt5.safetensors"), "umt5.safetensors")

    def test_url(self):
        self.assertEqual(
            mb.basename_of("https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/text_encoders/umt5.safetensors"),
            "umt5.safetensors",
        )

    def test_url_with_query_and_alternate(self):
        self.assertEqual(mb.basename_of("https://h/globs/x/A.safetensors?download=true"), "A.safetensors")
        self.assertEqual(mb.basename_of("https://h/globs/x/A.safetensors|some/folder"), "A.safetensors")

    def test_windows_path(self):
        self.assertEqual(mb.basename_of(r"D:\models\vae\wan_2.1_vae.safetensors"), "wan_2.1_vae.safetensors")

    def test_empty(self):
        self.assertEqual(mb.basename_of(""), "")


class BuildMapTests(unittest.TestCase):
    def test_from_index_mapping(self):
        index = {"p": _entry(r"D:\m\A.safetensors"), "q": _entry(r"D:\m\b.safetensors")}
        mapping = mb.build_basename_map(index)
        self.assertEqual(set(mapping), {"a.safetensors", "b.safetensors"})

    def test_from_iterable(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors"), _entry("/n/A.safetensors")])
        self.assertEqual(len(mapping["a.safetensors"]), 2)


class FindBindTests(unittest.TestCase):
    def test_no_index(self):
        result = mb.find_local_bind("A.safetensors", None)
        self.assertFalse(result)
        self.assertEqual(result.reason, mb.REASON_NO_INDEX)

    def test_no_match(self):
        mapping = mb.build_basename_map([_entry("/m/B.safetensors")])
        result = mb.find_local_bind("A.safetensors", mapping)
        self.assertEqual(result.reason, mb.REASON_NO_MATCH)

    def test_single_match(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors")])
        result = mb.find_local_bind("https://h/x/A.safetensors", mapping)
        self.assertTrue(result)
        self.assertEqual(result.path, "/m/A.safetensors")
        self.assertEqual(result.reason, mb.REASON_BOUND)

    def test_case_insensitive(self):
        mapping = mb.build_basename_map([_entry("/m/Wan2.1_VAE.SafeTensors")])
        result = mb.find_local_bind("wan2.1_vae.safetensors", mapping)
        self.assertTrue(result)

    def test_ambiguous_never_binds(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors"), _entry("/n/A.safetensors")])
        result = mb.find_local_bind("A.safetensors", mapping)
        self.assertFalse(result)
        self.assertEqual(result.reason, mb.REASON_AMBIGUOUS)
        self.assertEqual(set(result.candidates), {"/m/A.safetensors", "/n/A.safetensors"})

    def test_prefers_safetensors_over_other_containers(self):
        mapping = mb.build_basename_map([_entry("/m/A.bin"), _entry("/n/A.safetensors")])
        result = mb.find_local_bind("A.safetensors", mapping)
        self.assertTrue(result)
        self.assertEqual(result.path, "/n/A.safetensors")

    def test_size_match_binds(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors", size=999)])
        result = mb.find_local_bind("A.safetensors", mapping, expected_size=999)
        self.assertTrue(result)

    def test_size_mismatch_rejects(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors", size=999)])
        result = mb.find_local_bind("A.safetensors", mapping, expected_size=1000)
        self.assertFalse(result)
        self.assertEqual(result.reason, mb.REASON_SIZE_MISMATCH)

    def test_guard_rejects_all(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors", signature="t5")])
        result = mb.find_local_bind("A.safetensors", mapping, role="text_encoder", guard=lambda r, e, role: False)
        self.assertFalse(result)
        self.assertEqual(result.reason, mb.REASON_INCOMPATIBLE)

    def test_guard_disambiguates(self):
        mapping = mb.build_basename_map(
            [_entry("/m/A.safetensors", signature="umt5"), _entry("/n/A.safetensors", signature="t5")]
        )
        result = mb.find_local_bind(
            "A.safetensors", mapping, role="text_encoder", guard=lambda r, e, role: e.get("signature") == "umt5"
        )
        self.assertTrue(result)
        self.assertEqual(result.path, "/m/A.safetensors")

    def test_allow_ambiguous_picks_deterministic_first(self):
        mapping = mb.build_basename_map([_entry("/z/A.safetensors"), _entry("/a/A.safetensors")])
        result = mb.find_local_bind("A.safetensors", mapping, allow_ambiguous=True)
        self.assertTrue(result)
        self.assertEqual(result.path, "/a/A.safetensors")

    def test_candidates_reported_for_logging(self):
        mapping = mb.build_basename_map([_entry("/m/A.safetensors")])
        result = mb.find_local_bind("A.safetensors", mapping)
        self.assertEqual(result.candidates, ("/m/A.safetensors",))


if __name__ == "__main__":
    unittest.main()
