# Feature Fusion Network

Genomic-guided co-attention module for fusing genomic embeddings with patch-level WSI embeddings.

## GenomicGuidedCoAttention

### Inputs

- **Q** (Genomic embeddings): `(G, D)` or `(B, G, D)`
- **K** (Patch embeddings): `(N, D)` or `(B, N, D)`
- **V** (Patch embeddings): `(N, D)` or `(B, N, D)`

### Output

- **Output embeddings**: `(G, D)` or `(B, G, D)`
- **Attention weights**: `(G, N)` or `(B, G, N)`

## Example

```python
import torch
from feature_fusion_network import GenomicGuidedCoAttention

model = GenomicGuidedCoAttention(dim=512)

Q = torch.randn(6, 512)     # genomic embeddings
K = torch.randn(1000, 512)  # patch embeddings
V = torch.randn(1000, 512)

out, attn = model(Q, K, V)
print(out.shape, attn.shape)
```

## Runner script

Use the helper script to load genomic features from `data/csv/combined_genomic_features.pkl`
and patch features from `data/h5_features` (matched by the 12-character submitter ID).
Patch features are projected to 512 dimensions with a fully connected layer before
co-attention (use `--fc-weights` to load a trained projection).
If the pickle stores a tensor instead of a dictionary, the script will map rows to
submitter IDs using `data/reference/Final_Matched_Clinical.csv` (override with
`--submitter-ids-csv`).

```bash
python -m feature_fusion_network.run_co_attention --submitter-id TCGA-5T-A9QA --max-patches 512
```

The output is saved to `outputs/co_attention/<submitter_id>_co_attention.pt` and contains
the fused embeddings and attention weights.

## Train FC projection

The FC projection can be trained as a linear autoencoder using patch features:

```bash
python -m feature_fusion_network.train_fc_projection --max-files 50 --max-patches-per-file 256 --epochs 5 --device cpu
```

Then pass the saved weights to the co-attention runner:

```bash
python -m feature_fusion_network.run_co_attention --submitter-id TCGA-5T-A9QA --fc-weights outputs/fc_projection/fc_weights.pt
```

## Notes

- The module applies $\mathrm{softmax}(QK^T / \sqrt{d_k})$ along the patch dimension.
- Works with both batched and unbatched inputs.
