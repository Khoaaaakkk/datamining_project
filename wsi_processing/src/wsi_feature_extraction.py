from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.preprocessing.extract_feature import WSIFeaturePipeline, read_yaml
from src.preprocessing.gdc_downloader import GDCSVSDownloader
from src.preprocessing.submitter_batch_controller import SubmitterBatchController
from src.preprocessing.wsi_cleanup import WSICleanupManager
from src.utils.file_utils import ensure_dir, stem_without_double_suffix


def _to_bool(value, default: bool = True) -> bool:
	if value is None:
		return default
	if isinstance(value, bool):
		return value
	return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class WSIFeatureExtractionPipeline:
	"""
	End-to-end pipeline:
	1) Download .svs from GDC by submitter_id batches
	2) Process slide to .h5 feature file
	3) Delete processed .svs to save disk space
	"""

	def __init__(self, config: Dict):
		self.cfg = config
		self.data_cfg = config.get("data", {})
		self.gdc_cfg = config.get("gdc_download", {})

		self.raw_wsi_dir = Path(self.data_cfg.get("raw_wsi_dir", "data/raw_wsi"))
		self.h5_dir = Path(self.data_cfg.get("h5_features_dir", "data/h5_features"))
		self.masks_dir = Path(self.data_cfg.get("masks_dir", "data/masks"))
		self.report_path = Path(self.gdc_cfg.get("report_path", "experiments/logs/gdc_download_report.json"))
		self.skip_existing_h5 = _to_bool(self.gdc_cfg.get("skip_existing_h5", True), True)
		self.delete_processed_svs = _to_bool(self.gdc_cfg.get("delete_processed_svs", True), True)

		ensure_dir(self.raw_wsi_dir)
		ensure_dir(self.h5_dir)
		ensure_dir(self.masks_dir)
		ensure_dir(self.report_path.parent)

		self.downloader = GDCSVSDownloader(config)
		self.batch_controller = SubmitterBatchController(config)
		self.feature_pipeline = WSIFeaturePipeline(config)
		self.cleanup_manager = WSICleanupManager(enabled=self.delete_processed_svs)

	def _process_slide(self, slide_path: Path) -> Dict:
		slide_id = stem_without_double_suffix(slide_path)
		out_h5 = self.h5_dir / f"{slide_id}.h5"
		out_mask = self.masks_dir / f"{slide_id}.png"

		if self.skip_existing_h5 and out_h5.exists():
			return {
				"slide_path": str(slide_path),
				"status": "skipped_existing_h5",
				"saved_h5": str(out_h5),
				"saved_mask": str(out_mask),
			}

		result = self.feature_pipeline.process_slide(str(slide_path), str(out_h5), str(out_mask))
		result["status"] = "processed"
		return result

	def run(self) -> List[Dict]:
		submitter_ids = self.batch_controller.resolve_submitter_ids()
		print(f"[INFO] Total submitter_ids: {len(submitter_ids)}")
		processed_submitter_prefixes = [
			h5_path.stem[:10]
			for h5_path in self.h5_dir.glob("*.h5")
			if len(h5_path.stem) >= 10
		]

		reports: List[Dict] = []
		batches = list(self.batch_controller.iter_batches(submitter_ids))
		for batch_index, batch_submitters in enumerate(
			tqdm(batches, total=len(batches), desc="Pipeline batches", unit="batch"),
			start=1,
		):
      
			remaining_batches = len(batches) - batch_index
			tqdm.write(f"[PROGRESS] Current batch: {batch_index} ====== Remaining {remaining_batches} batch(es) to process.")
			# Heuristic: skip batch if all submitter prefixes already appear in existing .h5 files.
			batch_submitter_prefixes = [str(submitter_id)[:10] for submitter_id in batch_submitters]
			if batch_submitter_prefixes and all(
				prefix in processed_submitter_prefixes for prefix in batch_submitter_prefixes
			):
				tqdm.write(
					f"[INFO] Batch {batch_index}: likely already processed by prefix match, skipping"
				)
				continue
			tqdm.write(f"[INFO] Batch {batch_index}: submitter_ids={len(batch_submitters)}")
			downloaded_slides = self.downloader.download_for_submitter_ids(batch_submitters, batch_index=batch_index)

			for slide_path in tqdm(
				downloaded_slides,
				total=len(downloaded_slides),
				desc=f"Batch {batch_index} slides",
				unit="slide",
				leave=False,
			):
				# print(f"[REPORT] {slide_path}")
				try:
					report = self._process_slide(slide_path)
					if report.get("status") in {"processed", "skipped_existing_h5"}:
						self.cleanup_manager.cleanup_slide(slide_path)
				except Exception as e:
					report = {
						"slide_path": str(slide_path),
						"status": "failed",
						"error": str(e),
					}
				reports.append(report)
				tqdm.write(f"[REPORT] {slide_path.name}: {report.get('status')}")
			self.cleanup_manager.cleanup_raw_wsi_subfolders(self.raw_wsi_dir)
			tqdm.write(f"[DONE] Batch {batch_index}: submitter_ids={len(batch_submitters)}")
			# print(f"[DONE] Batch {batch_index}: submitter_ids={len(batch_submitters)}")

   
		self.report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"[DONE] Saved report -> {self.report_path}")
		return reports


def parse_args():
	p = argparse.ArgumentParser(description="WSI pipeline: download .svs by submitter IDs -> extract .h5 -> cleanup .svs")
	p.add_argument("--config", type=str, default="configs/default.yaml")
	return p.parse_args()


def main():
	args = parse_args()
	cfg = read_yaml(args.config)
	pipeline = WSIFeatureExtractionPipeline(cfg)
	pipeline.run()


if __name__ == "__main__":
	main()
