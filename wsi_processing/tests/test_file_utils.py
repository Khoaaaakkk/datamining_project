from pathlib import Path
import tempfile
import unittest

from src.utils.file_utils import ensure_dir, ensure_parent, stem_without_double_suffix


class TestFileUtils(unittest.TestCase):
    def test_ensure_dir_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            d = ensure_dir(tmp_path / "a" / "b")
            self.assertTrue(d.exists())
            self.assertTrue(d.is_dir())

            parent = ensure_parent(tmp_path / "c" / "d" / "file.txt")
            self.assertTrue(parent.exists())
            self.assertTrue(parent.is_dir())

    def test_stem_without_double_suffix(self):
        self.assertEqual(stem_without_double_suffix("abc.svs"), "abc")
        self.assertEqual(stem_without_double_suffix("abc.tiff"), "abc")
        self.assertEqual(stem_without_double_suffix("x.y.z"), "x.y")


if __name__ == "__main__":
    unittest.main()
