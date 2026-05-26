from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from wsi_preprocess.utils.file_utils import (
	ensure_dir,
	list_files,
	stem_without_double_suffix,
)


def read_yaml(path: str | Path) -> Dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def parse_args():
	parser = argparse.ArgumentParser(description="WSI preprocessing v2: extract Swin-L embeddings")
	parser.add_argument("--config", type=str, default="configs/default.yaml")
	parser.add_argument("--slide", type=str, default="", help="Single .svs path. If empty, process all files in raw_wsi dir")
	parser.add_argument("--output-dir", type=str, default="", help="Override output h5 directory")
	parser.add_argument("--save-report", type=str, default="", help="Path to save json processing report")
	parser.add_argument("--skip-existing", action="store_true", help="Skip slides that already have .h5 output")
	return parser.parse_args()


def main():
	args = parse_args()
	cfg = read_yaml(args.config)

	system_cfg = cfg.get("system", {})
	ops_dll = system_cfg.get("openslide_dll_dir")
	if ops_dll:
		__import__("os").environ.setdefault("OPENSLIDE_DLL_DIR", str(ops_dll))

	data_cfg = cfg.get("data", {})
	raw_dir = Path(data_cfg.get("raw_wsi_dir", "data/raw_wsi"))
	h5_dir = Path(args.output_dir or data_cfg.get("h5_features_dir", "data/h5_features"))
	masks_dir = Path(data_cfg.get("masks_dir", "data/masks"))
	output_cfg = cfg.get("output", {})
	default_report = output_cfg.get("report_path", "experiments/logs/preprocessing_report.json")

	ensure_dir(h5_dir)
	ensure_dir(masks_dir)

	if args.slide:
		slides = [Path(args.slide)]
	else:
		slides = list_files(raw_dir, [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn"])

	if len(slides) == 0:
		raise FileNotFoundError(f"No slide files found in {raw_dir}")

	from wsi_preprocess.preprocessing.embedding_pipeline import (
		WSIEmbeddingPipeline,
		save_report,
	)

	pipeline = WSIEmbeddingPipeline(cfg)
	report = []
	for slide_path in slides:
		slide_id = stem_without_double_suffix(slide_path)
		out_h5 = h5_dir / f"{slide_id}.h5"
		out_mask = masks_dir / f"{slide_id}.png"
		if out_h5.exists() and out_mask.exists():
			report.append(
				{
					"slide_path": str(slide_path),
					"status": "skipped_existing_outputs",
					"saved_h5": str(out_h5),
					"saved_mask": str(out_mask),
				}
			)
			continue
		if args.skip_existing and out_h5.exists():
			report.append(
				{
					"slide_path": str(slide_path),
					"status": "skipped_existing_h5",
					"saved_h5": str(out_h5),
					"saved_mask": str(out_mask),
				}
			)
			continue
		result = pipeline.process_slide(str(slide_path), str(out_h5), str(out_mask))
		result["status"] = "processed"
		report.append(result)
		print(f"[DONE] {slide_id}: {result['valid_patches']} patches")

	report_path = args.save_report or default_report
	save_report(report_path, report)
	print(f"Saved report -> {report_path}")


if __name__ == "__main__":
	main()
