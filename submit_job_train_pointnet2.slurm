#!/bin/bash
#SBATCH --job-name=pointnet2_pinna
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=pointnet2_train_%j.out
#SBATCH --error=pointnet2_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -euo pipefail

# =========================
# Easy hyperparameter block
# =========================
CONDA_ENV="anthropometric_env"
PROJECT_DIR="/iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
MESH_DIR="${PROJECT_DIR}/data/mesh"
LANDMARKS_DIR="${PROJECT_DIR}/data/landmarks"
RUN_ROOT="${PROJECT_DIR}/runs"
RUN_NAME="pointnet2_ssg_baseline"

SEED=42
VAL_RATIO=0.2
NUM_POINTS=16384
WORKERS=8

VARIANT="ssg"
HEAD_CHANNELS="512,256"
DROPOUT=0
SSG_NPOINTS="512,128"
SSG_RADII="0.2,0.4"
SSG_NSAMPLES="32,64"
SSG_MLPS="64,64,128;128,128,256;256,512,1024"

# These MSG settings are ignored unless VARIANT="msg".
MSG_NPOINTS="512,128"
MSG_RADII="0.1,0.2,0.4;0.2,0.4,0.8"
MSG_NSAMPLES="16,32,128;32,64,128"
MSG_MLPS="32,32,64|64,64,128|64,96,128;64,64,128|128,128,256|128,128,256"
MSG_GLOBAL_MLP="256,512,1024"

EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
OPTIMIZER="adamw"
LOSS="smooth_l1"
SMOOTH_L1_BETA=1.0
SCHEDULER="cosine"
STEP_SIZE=50
STEP_GAMMA=0.5
GRAD_CLIP_NORM=0.0
AMP=1

# =========================
# Run folder and environment
# =========================
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_NAME}_seed${SEED}_lr${LEARNING_RATE}_bs${BATCH_SIZE}_${SLURM_JOB_ID}_${TIMESTAMP}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

exec > >(tee -a "${RUN_DIR}/train.log") 2> >(tee -a "${RUN_DIR}/train.err" >&2)

echo "Run directory: ${RUN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

cat > "${RUN_DIR}/sbatch_config.txt" <<EOF
CONDA_ENV=${CONDA_ENV}
PROJECT_DIR=${PROJECT_DIR}
MESH_DIR=${MESH_DIR}
LANDMARKS_DIR=${LANDMARKS_DIR}
RUN_ROOT=${RUN_ROOT}
RUN_NAME=${RUN_NAME}
SEED=${SEED}
VAL_RATIO=${VAL_RATIO}
NUM_POINTS=${NUM_POINTS}
WORKERS=${WORKERS}
VARIANT=${VARIANT}
HEAD_CHANNELS=${HEAD_CHANNELS}
DROPOUT=${DROPOUT}
SSG_NPOINTS=${SSG_NPOINTS}
SSG_RADII=${SSG_RADII}
SSG_NSAMPLES=${SSG_NSAMPLES}
SSG_MLPS=${SSG_MLPS}
MSG_NPOINTS=${MSG_NPOINTS}
MSG_RADII=${MSG_RADII}
MSG_NSAMPLES=${MSG_NSAMPLES}
MSG_MLPS=${MSG_MLPS}
MSG_GLOBAL_MLP=${MSG_GLOBAL_MLP}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
LEARNING_RATE=${LEARNING_RATE}
WEIGHT_DECAY=${WEIGHT_DECAY}
OPTIMIZER=${OPTIMIZER}
LOSS=${LOSS}
SMOOTH_L1_BETA=${SMOOTH_L1_BETA}
SCHEDULER=${SCHEDULER}
STEP_SIZE=${STEP_SIZE}
STEP_GAMMA=${STEP_GAMMA}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM}
AMP=${AMP}
EOF

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

TRAIN_ARGS=(
    --mesh-dir "${MESH_DIR}"
    --landmarks-dir "${LANDMARKS_DIR}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --val-ratio "${VAL_RATIO}"
    --num-points "${NUM_POINTS}"
    --seed "${SEED}"
    --workers "${WORKERS}"
    --variant "${VARIANT}"
    --head-channels "${HEAD_CHANNELS}"
    --dropout "${DROPOUT}"
    --ssg-npoints "${SSG_NPOINTS}"
    --ssg-radii "${SSG_RADII}"
    --ssg-nsamples "${SSG_NSAMPLES}"
    --ssg-mlps "${SSG_MLPS}"
    --msg-npoints "${MSG_NPOINTS}"
    --msg-radii "${MSG_RADII}"
    --msg-nsamples "${MSG_NSAMPLES}"
    --msg-mlps "${MSG_MLPS}"
    --msg-global-mlp "${MSG_GLOBAL_MLP}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --optimizer "${OPTIMIZER}"
    --loss "${LOSS}"
    --smooth-l1-beta "${SMOOTH_L1_BETA}"
    --scheduler "${SCHEDULER}"
    --step-size "${STEP_SIZE}"
    --step-gamma "${STEP_GAMMA}"
    --grad-clip-norm "${GRAD_CLIP_NORM}"
    --device auto
)

if [[ "${AMP}" == "1" ]]; then
    TRAIN_ARGS+=(--amp)
fi

printf '%q ' python train_pointnet2.py "${TRAIN_ARGS[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
cat "${RUN_DIR}/command.txt"

python train_pointnet2.py "${TRAIN_ARGS[@]}"

echo "Training complete"
echo "Best checkpoint: ${CHECKPOINT_DIR}/best_model.pt"
date
