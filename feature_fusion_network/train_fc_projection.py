from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional, Tuple

import h5py
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader


class H5PatchDataset(IterableDataset):
	def __init__(
		self,
		h5_dir: Path,
		dataset_name: str,
		max_files: Optional[int],
		max_patches_per_file: Optional[int],
		shuffle_files: bool,
	) -> None:
		super().__init__()
		self.h5_dir = h5_dir
		self.dataset_name = dataset_name
		self.max_files = max_files
		self.max_patches_per_file = max_patches_per_file
		self.shuffle_files = shuffle_files

	def _iter_files(self) -> Iterator[Path]:
		files = sorted(self.h5_dir.glob("*.h5"))
		if self.shuffle_files:
			files = list(files)
			torch.manual_seed(42)
			perm = torch.randperm(len(files)).tolist()
			files = [files[i] for i in perm]
		if self.max_files is not None:
			files = files[: self.max_files]
		for path in files:
			yield path

	def __iter__(self) -> Iterator[torch.Tensor]:
		for path in self._iter_files():
			with h5py.File(path, "r") as handle:
				if self.dataset_name not in handle:
					continue
				data = handle[self.dataset_name][()]
				if data.ndim != 2:
					continue
				if self.max_patches_per_file is not None:
					data = data[: self.max_patches_per_file]
				for row in data:
					yield torch.tensor(row, dtype=torch.float32)


class LinearAutoEncoder(nn.Module):
	def __init__(self, input_dim: int, latent_dim: int) -> None:
		super().__init__()
		self.encoder = nn.Linear(input_dim, latent_dim)
		self.decoder = nn.Linear(latent_dim, input_dim)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		z = self.encoder(x)
		recon = self.decoder(z)
		return z, recon



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train a linear 1536->512 projection for patch features."
	)
	parser.add_argument(
		"--h5-dir",
		type=Path,
		default=Path("data/h5_features"),
		help="Directory containing H5 feature files.",
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default="features",
		help="Dataset name inside each H5 file.",
	)
	parser.add_argument(
		"--input-dim",
		type=int,
		default=1536,
		help="Input patch feature dimension.",
	)
	parser.add_argument(
		"--latent-dim",
		type=int,
		default=512,
		help="Latent dimension for the projection.",
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=100,
		help="Maximum number of H5 files to sample.",
	)
	parser.add_argument(
		"--max-patches-per-file",
		type=int,
		default=512,
		help="Maximum patches to sample from each H5 file.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=256,
		help="Batch size for training.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=5,
		help="Number of training epochs.",
	)
	parser.add_argument(
		"--lr",
		type=float,
		default=1e-3,
		help="Learning rate.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cpu",
		help="Device to train on (cpu or cuda).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("outputs/fc_projection/fc_weights.pt"),
		help="Path to save encoder weights (state_dict).",
	)
	parser.add_argument(
		"--shuffle-files",
		action="store_true",
		help="Shuffle file order before sampling.",
	)
	return parser.parse_args()



def main() -> None:
	args = parse_args()
	device = torch.device(args.device)

	dataset = H5PatchDataset(
		h5_dir=args.h5_dir,
		dataset_name=args.dataset,
		max_files=args.max_files,
		max_patches_per_file=args.max_patches_per_file,
		shuffle_files=args.shuffle_files,
	)
	loader = DataLoader(dataset, batch_size=args.batch_size)

	model = LinearAutoEncoder(args.input_dim, args.latent_dim).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	loss_fn = nn.MSELoss()

	model.train()
	for epoch in range(1, args.epochs + 1):
		epoch_loss = 0.0
		batch_count = 0
		for batch in loader:
			batch = batch.to(device)
			optimizer.zero_grad()
			_, recon = model(batch)
			loss = loss_fn(recon, batch)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			batch_count += 1
		if batch_count == 0:
			raise RuntimeError("No training batches found. Check H5 files and dataset.")
		avg_loss = epoch_loss / batch_count
		print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")

	args.output.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.encoder.state_dict(), args.output)
	print(f"Saved encoder weights to: {args.output}")


if __name__ == "__main__":
	main()
