import numpy as np

from wsi_preprocess.preprocessing.patch_sampler import generate_patch_coords, mask_has_tissue


def test_mask_has_tissue_basic():
	mask = np.zeros((10, 10), dtype=np.uint8)
	mask[0:5, 0:5] = 255
	assert mask_has_tissue(mask, 0, 0, 4, threshold=0.2)
	assert not mask_has_tissue(mask, 6, 6, 4, threshold=0.2)


def test_generate_patch_coords_with_mask():
	mask = np.zeros((10, 10), dtype=np.uint8)
	mask[0:6, 0:6] = 255
	coords = generate_patch_coords(
		slide_dims=(16, 16),
		patch_size=4,
		step_size=4,
		mask=mask,
		downsample=1.0,
		tissue_threshold=0.2,
	)
	assert (0, 0) in coords
	assert (4, 4) in coords
	assert (12, 12) not in coords
