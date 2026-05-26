# Survival Prediction Template

This folder contains a minimal, runnable template for discrete survival risk prediction.

## Inputs

- `data/h5_feature/{submitter_id}.h5` (or prefix match)
- `data/outputs/co_attention/{submitter_id}.pt`
- `data/outputs/fusion/{submitter_id}.pt`

If your project uses `data/h5_features`, pass `--h5-dir data/h5_features`.

## Run inference

```powershell
python survival_prediction/main.py --manifest data/manifest/batch_0001.txt
```

Predictions are saved to `data/outputs/predictions/{submitter_id}_hazards.pt`.

## Evaluate

```powershell
python eval/evaluate.py --manifest data/manifest/batch_0001.txt
```

The evaluation script uses mock ground-truth data for template purposes.
