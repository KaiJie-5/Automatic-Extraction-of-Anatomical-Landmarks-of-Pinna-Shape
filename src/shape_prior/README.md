# PCA Shape Prior

This module builds and applies a PCA statistical shape prior for the 85 pinna landmarks.
It is inference-time post-processing only: it does not train the landmark model, change
the loss, or change the model architecture.

The prior is applied to the crop-local canonical normalized landmark prediction after the
landmark model and `LocalLandmarkRefiner`, and before:

```python
local_transform.denormalize_xyz(local_prediction)
```

## Generate A Fold Prior (Split train/val)

```powershell
python -m src.shape_prior.generate_prior `
  --mesh-dir data/mesh `
  --landmarks-dir data/landmarks `
  --folds-json artifacts/folds.json `
  --outer-fold 1 `
  --predictions-json artifacts/fold1_crop_calibration_predictions.json `
  --calibration-json artifacts/fold1_crop_calibration.json `
  --components 32 `
  --beta 1.0 `
  --output artifacts/fold1_pca_shape_prior.npz `
  --manifest artifacts/fold1_pca_shape_prior_manifest.json
```

## Evaluate A Fold Prior

```powershell
python -m src.shape_prior.evaluate_prior `
  --checkpoint-path runs/refinement_screen/k32/fold1_seed43/best_landmarks.pt `
  --prior-path artifacts/fold1_pca_shape_prior.npz `
  --mesh-dir data/mesh `
  --landmarks-dir data/landmarks `
  --folds-json artifacts/folds.json `
  --predictions-json artifacts/fold1_crop_calibration_predictions.json `
  --calibration-json artifacts/fold1_crop_calibration.json `
  --output artifacts/fold1_pca_shape_prior_eval.json
```

Expected fold-1 result:

```text
raw mean ~= 2.0588 mm
PCA mean ~= 1.9475 mm
improvement ~= 0.1113 mm
ears improved = 80/80
ears worsened = 0/80
```

## Generate The Final Prior (All training samples, no validation)

```powershell
python -m src.shape_prior.generate_prior `
  --mesh-dir data/mesh `
  --landmarks-dir data/landmarks `
  --folds-json artifacts/folds.json `
  --outer-fold final `
  --predictions-json artifacts/final_crop_calibration_predictions.json `
  --calibration-json artifacts/final_crop_calibration.json `
  --components 32 `
  --beta 1.0 `
  --output artifacts/final_pca_shape_prior.npz `
  --manifest artifacts/final_pca_shape_prior_manifest.json
```

## Embed The Final Prior

```powershell
python -m src.shape_prior.embed_prior `
  --checkpoint checkpoints/final_pipeline.pt `
  --prior artifacts/final_pca_shape_prior.npz `
  --output checkpoints/final_pipeline_pca.pt `
  --beta 1.0
```
