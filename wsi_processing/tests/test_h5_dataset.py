import importlib.util
from pathlib import Path
import tempfile
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PANDAS_AVAILABLE and H5PY_AVAILABLE and NUMPY_AVAILABLE and TORCH_AVAILABLE, "pandas/h5py/numpy/torch not installed")
class TestH5BagDataset(unittest.TestCase):
    def test_load_single_h5(self):
        import h5py
        import numpy as np
        import pandas as pd

        from src.data_loader.h5_dataset import H5BagDataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            h5_dir = tmp_path / "h5"
            h5_dir.mkdir(parents=True, exist_ok=True)

            slide_id = "slide_001"
            h5_path = h5_dir / f"{slide_id}.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("features", data=np.random.randn(5, 16).astype(np.float32))
                f.create_dataset("coords", data=np.array([[0, 0], [16, 0], [0, 16], [16, 16], [32, 32]], dtype=np.int32))

            labels_csv = tmp_path / "labels.csv"
            pd.DataFrame({"slide_id": [slide_id], "label": [1]}).to_csv(labels_csv, index=False)

            ds = H5BagDataset(str(h5_dir), str(labels_csv), min_instances=1)
            self.assertEqual(len(ds), 1)
            sample = ds[0]
            self.assertEqual(sample["slide_id"], slide_id)
            self.assertEqual(tuple(sample["features"].shape), (5, 16))
            self.assertEqual(tuple(sample["coords"].shape), (5, 2))
            self.assertEqual(int(sample["label"].item()), 1)


if __name__ == "__main__":
    unittest.main()
