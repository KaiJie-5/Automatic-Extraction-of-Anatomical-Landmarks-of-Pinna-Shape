# Point Transformer V3 Research Experiment

This optional experiment changes only the landmark backbone. It reuses the
existing 16,384-point locator predictions, crop calibration, four contour heads,
spacing loss `0.01`, K=32 refinement, and exact surface projection. The proven
PointNeXt environment and competition submission are not modified.

## Separate environment

Run these commands on Iridis. Do not install the compiled packages into
`anthropometric_env`.

```bash
conda create -n anthropometric_ptv3_env python=3.10 -y
conda activate anthropometric_ptv3_env
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cuda-nvcc=11.8 -c nvidia -y
python -m pip install --upgrade pip ninja packaging
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation
python -m pip install -r requirements-ptv3.txt
python -m pip check
nvcc --version
```

The duplicate `torch-scatter` and `flash-attn` entries in the requirements file
act as version assertions after the wheel/build commands. The normal
`requirements.txt` remains the portable baseline environment.

## Required preflight

Submit the H200 gate before training either grid size:

```bash
sbatch submit_job_train_pointnet2.slurm ptv3-preflight \
  --num-points 16384 \
  --grid-size 0.01 0.02 \
  --seed 42 \
  --device cuda \
  --output artifacts/ptv3/preflight.json
```

The job exits with status 2 after writing the report if imports, BF16
forward/backward, finite gradients, output shape, or deterministic evaluation
fail. There is no non-Flash fallback.

The dependency-free adapter and CLI tests can be run separately:

```bash
python -m pytest -q tests/test_pointtransformerv3.py
```

These tests do not replace the H200 preflight.

## Fold 0 grid screen

Grid `0.01`:

```bash
sbatch submit_job_train_pointnet2.slurm fit-landmarks \
  --mesh-dir /iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape/data/mesh \
  --landmarks-dir /iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape/data/landmarks \
  --folds-json artifacts/folds.json \
  --outer-fold 0 \
  --predictions-json artifacts/calibration_v2/fold0_crop_calibration_predictions.json \
  --calibration-json artifacts/calibration_v2/fold0_crop_calibration.json \
  --output-dir runs/pointtransformerv3/grid_001/fold0_seed42 \
  --backbone pointtransformerv3 \
  --ptv3-grid-size 0.01 \
  --four-heads \
  --no-augment \
  --refinement-k 32 \
  --anchor-weight 0 \
  --spacing-weight 0.01 \
  --surface-weight 0 \
  --num-points 16384 \
  --seed 42 \
  --epochs 200 \
  --patience 30 \
  --workers 10 \
  --amp
```

Grid `0.02` uses the same command with these replacements:

```text
--ptv3-grid-size 0.02
--output-dir runs/pointtransformerv3/grid_002/fold0_seed42
```

## Projection evaluation

Evaluate each candidate with its matching checkpoint:

```bash
sbatch submit_job_train_pointnet2.slurm evaluate-projection \
  --checkpoint-path runs/pointtransformerv3/grid_001/fold0_seed42/best_landmarks.pt \
  --predictions-json artifacts/calibration_v2/fold0_crop_calibration_predictions.json \
  --calibration-json artifacts/calibration_v2/fold0_crop_calibration.json \
  --folds-json artifacts/folds.json \
  --mesh-dir /iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape/data/mesh \
  --landmarks-dir /iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape/data/landmarks \
  --run-seed 42 \
  --device cuda \
  --output runs/projection/pointtransformerv3/grid_001/fold0_seed42.json
```

Repeat with `grid_002`. Select by projected Fold-0 MD, using projection runtime
only if the MD values round equally to `0.001 mm`. Full five-fold, three-seed
confirmation is performed only for the selected grid size.
