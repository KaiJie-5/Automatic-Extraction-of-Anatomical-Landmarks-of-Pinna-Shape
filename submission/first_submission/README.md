# 3D Pinna Landmark Extraction — Runtime Instructions

This submission predicts 85 ordered landmarks for the left ear and 85 ordered landmarks for the right ear from an aligned triangular 3D head mesh.

The evaluation entry point is:

```python
from src.estimator import LandmarkExtractor
```

The trained locator, PointNeXt landmark model, crop calibration, coordinate transforms, deterministic sampling settings, and surface-projection settings are bundled in:

```text
checkpoints/final_pipeline.pt
```

## 1. Verified environment

The submission was prepared and verified with the following environment:

| Component | Version |
|---|---:|
| Python | 3.10.20 |
| pip | 26.1.2 |
| PyTorch | 2.9.1+cu128 |
| CUDA used by PyTorch | 12.8 |
| cuDNN | 91002 |
| NumPy | 2.2.6 |
| trimesh | 4.12.2 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.9 |
| K3D | 2.17.0 |
| ipykernel | 7.2.0 |


## 2. Create the Conda environment

Open a terminal and change to the extracted submission directory:

```bash
cd /absolute/path/to/submission
```

Create and activate the Conda environment:

```bash
conda create --name anthropometric_env python=3.10.20 -y
conda activate anthropometric_env
```

Upgrade pip to the verified version:

```bash
python -m pip install --upgrade pip==26.1.2
```

Install the CUDA 12.8 PyTorch build:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify that the environment has no broken dependencies:

```bash
python -m pip check
```

The expected result is:

```text
No broken requirements found.
```

## 3. Verify Python and GPU access

Check the Python version:

```bash
python --version
```

Check PyTorch, CUDA, cuDNN, and the available GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Built CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('cuDNN:', torch.backends.cudnn.version()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

When running on a GPU node, `CUDA available` should normally be `True`.

Verify the principal imports:

```bash
python -c "import torch, numpy, trimesh, scipy; from src.estimator import LandmarkExtractor; print('Imports successful')"
```