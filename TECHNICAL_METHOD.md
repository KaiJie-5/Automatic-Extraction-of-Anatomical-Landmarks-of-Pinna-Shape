# Proposal-Aligned Pinna Landmark Method

The v2 system uses the official aligned head coordinates for localization and a
shared canonical ear frame for detailed landmark regression. Right ears and their
normals are reflected across Y before either model sees them. Every output is
transformed back to the original head frame.

## Leakage controls

All splits are subject-level. Each outer fold contains four inner folds used to
produce centre predictions for the outer training subjects. Landmark crops are
therefore generated from out-of-fold locator predictions, never ground-truth or
in-sample centres. The five outer validation prediction sets provide the 402 OOF
ear centres used for final calibration and final landmark training.

The primary directional crop combines the 99th percentile of annotated landmark
reach, the appropriate signed 99th-percentile locator error, and the smallest
0–5 mm safety margin reaching 99% complete-ear coverage. A separately calibrated
backup crop covers 100%. The backup activates only for invalid clipping/sampling
or geometry below the serialized first-percentile face-count/surface-area gate.

## Models and objectives

The locator regresses a three-value centre correction in millimetres from 16,384
broad-region surface points. The landmark model receives 16,384 tight-crop points
in predicted-centre-local coordinates and emits 85 ordered coordinates through
one head or four contour heads of 25, 30, 20, and 10 points.

The official mean Euclidean distance in millimetres is always the base loss.
Optional anchor, within-section spacing, and one-directional dense-surface losses
are enabled independently. Local KNN refinement, PointNeXt-S, MeshNet, and exact
surface projection are experiments and are not promoted without the registered
five-fold, three-seed rule in `configs/experiment_matrix.json`.

## Submission contract

`src.estimator.LandmarkExtractor` loads a schema-version-2 bundle by default and
returns two finite `float32` arrays of shape `(85, 3)`. Legacy full-head and fixed
ear-crop checkpoint dictionaries remain loadable. A final v2 bundle contains both
models, coordinate and crop calibration, deterministic sampling settings, and
postprocessing configuration.
