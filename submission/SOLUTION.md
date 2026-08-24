# Proposal-aligned 3D pinna landmark extractor

The submission implements one deterministic coarse-to-fine model. A PointNet++
locator predicts each ear centre from a broad side crop. A shared PointNeXt-S-style
landmark network then predicts the 85 ordered landmarks in a mirrored canonical ear
frame using four contour heads and a 32-neighbour local refinement module. The final
landmarks are reflected back into official head coordinates and projected to the
closest point on the selected triangle surface.

The checkpoint is a schema-version-2 bundle containing both model states, broad and
directional crop calibration, coordinate transforms, sampling seeds, and projection
configuration. No dataset paths or external calibration files are needed at
inference. Both ears are returned separately as finite `float32` arrays with shape
`(85, 3)`.

## Packaged runtime files

- `src/estimator.py`: official `LandmarkExtractor` interface.
- `src/calibration.py`, `canonical.py`, `geometry.py`, `preprocessing.py`:
  deterministic crop and coordinate processing.
- `src/pointnet2_model.py`, `pointnet2_utils.py`, `pointnext_model.py`,
  `proposal_models.py`: locator and landmark networks.
- `src/surface.py`: exact closest-point-on-triangle projection.
- `checkpoints/final_pipeline.pt`: final all-subject v2 checkpoint.
- `smoke_test.py`: shape, dtype, finiteness, loading, and repeatability check.
- `train_pipeline.py`, `configs/experiment_matrix.json`, and the additional
  training modules in `src/`: the auditable training and ablation pipeline.

The included KEMAR data is the official template sample and is used only for a
packaging smoke test. Its reported landmark distance is not an estimate of hidden-set
generalisation because KEMAR is part of final model training.
