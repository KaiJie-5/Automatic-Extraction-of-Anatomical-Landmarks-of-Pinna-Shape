# Projection-Aware PCA Shape Prior

This optional post-processing experiment constrains the 85 crop-local canonical
landmarks to statistical ear shapes learned from labelled outer-training ears.
It does not retrain or alter the landmark backbone.

```text
outer-training landmark shapes -> PCA basis
held-out model output -> PCA blend -> exact triangle projection -> MD
```

The fold manifest is mandatory. It verifies the exact training IDs and SHA-256
hashes of the fold file, OOF centre predictions, crop calibration, and saved
prior. This prevents a prior from seeing held-out landmarks or being reused on
the wrong fold.

## Generate One Fold Prior

```bash
python train_pipeline.py generate-pca-prior \
  --mesh-dir data/mesh \
  --landmarks-dir data/landmarks \
  --folds-json artifacts/folds.json \
  --outer-fold 0 \
  --predictions-json artifacts/calibration_v2/fold0_crop_calibration_predictions.json \
  --calibration-json artifacts/calibration_v2/fold0_crop_calibration.json \
  --components 32 \
  --beta 1.0 \
  --output artifacts/pca/fold0/prior.npz \
  --manifest artifacts/pca/fold0/manifest.json
```

## Evaluate PCA Before Exact Projection

```bash
python train_pipeline.py evaluate-pca-prior \
  --checkpoint-path runs/pointnext_s/fold0_seed42/best_landmarks.pt \
  --prior-path artifacts/pca/fold0/prior.npz \
  --prior-manifest artifacts/pca/fold0/manifest.json \
  --mesh-dir data/mesh \
  --landmarks-dir data/landmarks \
  --folds-json artifacts/folds.json \
  --predictions-json artifacts/calibration_v2/fold0_crop_calibration_predictions.json \
  --calibration-json artifacts/calibration_v2/fold0_crop_calibration.json \
  --components 32 \
  --beta 0.5 \
  --run-seed 42 \
  --output runs/pca_projection/fold0_seed42.json
```

The report contains raw, PCA-only, projected-only, and PCA-then-projected MD;
ear-level median, p90, p95, and maximum errors; contour errors; and every
subject/ear result. Select components and beta only on the designated screening
fold, then lock both values for confirmation.

For confirmation, save all reports as
`runs/pca_projection/confirmation/foldN_seedS.json`, where `N` is 0--4 and `S`
is 42, 43, or 44. Enforce the registered promotion rule with:

```bash
python train_pipeline.py summarize-pca-prior \
  --report-root runs/pca_projection/confirmation \
  --seeds 42 43 44 \
  --output runs/pca_projection/confirmation_summary.json
```

## Final Prior and Checkpoint Embedding

Only after promotion, fit the same locked PCA configuration on all labelled
training subjects using the final OOF centre predictions and final calibration.
Embed the selected component count and beta into a copy of the final v2 bundle:

```bash
python -m src.shape_prior.embed_prior \
  --checkpoint checkpoints/final_pipeline.pt \
  --prior artifacts/pca/final/prior.npz \
  --components 32 \
  --beta 0.5 \
  --output checkpoints/final_pipeline_pca.pt
```

The estimator applies the embedded PCA blend before its configured exact surface
projection, matching fold evaluation order.

## Bilateral Mean/Asymmetry PCA Experiment

This separate, opt-in prior retains the proven independent-ear landmark model.
After canonical right-ear mirroring, it normalizes each predicted ear shape
independently and forms

```text
common morphology = (left + mirrored right) / 2
signed asymmetry  = (left - mirrored right) / 2
```

The common and asymmetry terms use independent PCA bases, component counts, and
blend strengths. The prior is fitted from complete subject pairs in the exact
outer-training fold. It never reads held-out annotations.

Generate a Fold-0 prior storing enough components for screening:

```bash
python train_pipeline.py generate-bilateral-pca-prior \
  --mesh-dir data/mesh \
  --landmarks-dir data/landmarks \
  --folds-json artifacts/folds.json \
  --outer-fold 0 \
  --predictions-json artifacts/calibration_v2/fold0_crop_calibration_predictions.json \
  --calibration-json artifacts/calibration_v2/fold0_crop_calibration.json \
  --common-components 64 \
  --asymmetry-components 32 \
  --output artifacts/bilateral_pca/fold0/prior.npz \
  --manifest artifacts/bilateral_pca/fold0/manifest.json
```

Broad grid screening deliberately skips expensive exact triangle projection.
`--reference-report` must be the matching independent-PCA report generated from
the exact same fold checkpoint:

```bash
python train_pipeline.py evaluate-bilateral-pca-prior \
  --checkpoint-path runs/pointnext_surface_heatmap_d256/fold0_seed42/best_landmarks.pt \
  --prior-path artifacts/bilateral_pca/fold0/prior.npz \
  --prior-manifest artifacts/bilateral_pca/fold0/manifest.json \
  --reference-report runs/pca_projection/heatmap_d256/fold0_seed42.json \
  --mesh-dir data/mesh \
  --landmarks-dir data/landmarks \
  --folds-json artifacts/folds.json \
  --predictions-json artifacts/calibration_v2/fold0_crop_calibration_predictions.json \
  --calibration-json artifacts/calibration_v2/fold0_crop_calibration.json \
  --common-components 16 32 64 \
  --asymmetry-components 4 8 16 \
  --common-betas 0.25 0.5 \
  --asymmetry-betas 0.5 0.75 1.0 \
  --skip-projection \
  --run-seed 42 \
  --output runs/bilateral_pca/screen/fold0_seed42.json \
  --device auto
```

After inspecting the screen, rerun only its leading settings without
`--skip-projection`. Lock one projected setting before the five-fold/three-seed
confirmation. Each confirmation report must contain exactly one setting, then
run:

```bash
python train_pipeline.py summarize-bilateral-pca-prior \
  --report-root runs/bilateral_pca/confirmation \
  --seeds 42 43 44 \
  --output runs/bilateral_pca/confirmation_summary.json
```

The bilateral summary compares against the already-promoted independent-PCA
result in each reference report. Promotion still requires a lower pooled
15-run MD and improvement on at least three of five folds.

Only after promotion, generate the prior with `--outer-fold final` and embed it
into a new checkpoint copy:

```bash
python -m src.shape_prior.embed_bilateral_prior \
  --checkpoint checkpoints/final_pipeline_pca.pt \
  --prior artifacts/bilateral_pca/final/prior.npz \
  --common-components 32 \
  --asymmetry-components 8 \
  --common-beta 0.5 \
  --asymmetry-beta 0.75 \
  --replace-independent-pca \
  --output checkpoints/final_pipeline_bilateral_pca.pt
```

The numerical setting above is illustrative; use only the setting locked by
the confirmation experiment.
