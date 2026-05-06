import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(NUMPY_AVAILABLE and CV2_AVAILABLE, "numpy/cv2 not installed")
class TestQuality(unittest.TestCase):
    def test_valid_patch(self):
        import numpy as np
        from src.preprocessing.quality import is_valid_patch

        rng = np.random.default_rng(0)
        patch = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        self.assertTrue(is_valid_patch(patch, max_white_ratio=0.95, min_laplacian_var=1.0))

    def test_reject_white_patch(self):
        import numpy as np
        from src.preprocessing.quality import is_valid_patch

        patch = np.full((64, 64, 3), 255, dtype=np.uint8)
        self.assertFalse(is_valid_patch(patch, max_white_ratio=0.7, min_laplacian_var=1.0))


if __name__ == "__main__":
    unittest.main()
