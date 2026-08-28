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

# Torch 2.1 was built against NumPy 1.x and its cpp_extension still imports
# packaging through pkg_resources. Setuptools 70+ removed that compatibility
# export, so install these versions before building FlashAttention.
python -m pip install --upgrade pip
python -m pip install --force-reinstall \
  numpy==1.26.4 \
  setuptools==69.5.1 \
  wheel==0.43.0 \
  packaging==24.0 \
  psutil==5.9.8
python -m pip install ninja fsspec

python -m pip install -r requirements.txt
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install addict==2.4.0 timm==0.9.16 spconv-cu118==2.3.8

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation

# This is now a version assertion because the compiled dependencies are
# already installed. Do not use this as the first install command in a fresh
# environment.
python -m pip install --no-build-isolation -r requirements-ptv3.txt
python -m pip check
nvcc --version
```

Build FlashAttention on an allocated compute node rather than a login node. The
duplicate `torch-scatter` and `flash-attn` entries in the requirements file act
as version assertions after the wheel/build commands. The normal
`requirements.txt` remains the portable baseline environment.

Verify the environment before submitting the preflight:

```bash
python -c "import setuptools; print('setuptools:', setuptools.__version__); from pkg_resources import packaging; print('pkg_resources packaging:', packaging.__version__)"
python -c "import numpy, torch; print('NumPy:', numpy.__version__); print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -c "import addict, timm, spconv.pytorch, torch_scatter, flash_attn; print('All PTv3 dependencies imported successfully')"
```

The required core values are NumPy `1.26.4`, Torch `2.1.0`, CUDA `11.8`, and
setuptools `69.5.1`.

### FlashAttention installation failures

If FlashAttention reports
`ImportError: cannot import name 'packaging' from 'pkg_resources'`, setuptools
is too new. Restore the compatibility version and retry without build
isolation:

```bash
python -m pip install --force-reinstall setuptools==69.5.1 wheel==0.43.0 packaging==24.0 psutil==5.9.8
python -c "from pkg_resources import packaging; print(packaging.__version__)"
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation --no-cache-dir
```

If it reports `No module named 'torch'` while creating a temporary build
environment, the command omitted `--no-build-isolation`. If Torch warns that a
module compiled with NumPy 1.x cannot run with NumPy 2.x, restore
`numpy==1.26.4` before retrying. Garbled progress-bar characters such as
`â”` are terminal encoding only and are not installation errors.

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

The job exits with status 2 after writing the report if imports, cold-start FP16
forward/backward, finite gradients, output shape, native-spconv configuration,
or deterministic evaluation fail. PTv3 uses FP16 AMP with gradient scaling and
constructs every sparse convolution with `spconv.ConvAlgo.Native`. This avoids
the mixed-precision evaluation failure in spconv's implicit-GEMM
`ConvTunerSimple` path documented in
[spconv issue #563](https://github.com/traveller59/spconv/issues/563) and
[PTv3 issue #176](https://github.com/Pointcept/PointTransformerV3/issues/176).
PointNet++ and PointNeXt retain the normal BF16-on-H200 policy. Losses and
metrics remain FP32. There is no non-Flash fallback.

Do not work around the error by leaving the whole model in training mode during
validation: that also enables stochastic drop-path/order shuffling and changes
normalisation behaviour. The CUDA 12.8 source-build recipe in
[spconv issue #746](https://github.com/traveller59/spconv/issues/746#issuecomment-3155991737)
targets Blackwell compute capability 12.0; it is not the correct installation
for the H200 (compute capability 9.0) CUDA 11.8 environment used here.

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
