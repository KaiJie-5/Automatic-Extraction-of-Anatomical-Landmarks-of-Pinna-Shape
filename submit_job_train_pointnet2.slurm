#!/bin/bash
#SBATCH --partition=i7_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00

set -eo pipefail
set +u

STAGE="${1:?Usage: sbatch submit_job_train_pointnet2.slurm STAGE [stage arguments...]}"
shift
PROJECT_DIR="/iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
RUN_LOG_DIR="${PROJECT_DIR}/slurm-logs"
mkdir -p "${RUN_LOG_DIR}"
cd "${PROJECT_DIR}"

# Preserve the active environment selected at submission time. Optional PTv3
# jobs install the CUDA 11.8 compiler in that Conda prefix and spconv may need
# the corresponding toolkit during cold-start kernel initialization.
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

exec > >(tee -a "${RUN_LOG_DIR}/${STAGE}_${SLURM_JOB_ID}.out") 2> >(tee -a "${RUN_LOG_DIR}/${STAGE}_${SLURM_JOB_ID}.err" >&2)

echo "stage=${STAGE}"
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "project=${PROJECT_DIR}"
echo "python=$(command -v python)"
echo "cuda_home=${CUDA_HOME:-unset}"
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

python train_pipeline.py "${STAGE}" "$@"
