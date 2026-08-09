#!/bin/bash
#SBATCH --partition=quad_h200
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

exec > >(tee -a "${RUN_LOG_DIR}/${STAGE}_${SLURM_JOB_ID}.out") 2> >(tee -a "${RUN_LOG_DIR}/${STAGE}_${SLURM_JOB_ID}.err" >&2)

echo "stage=${STAGE}"
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "project=${PROJECT_DIR}"
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

python train_pipeline.py "${STAGE}" "$@"
