#!/bin/bash
#SBATCH --job-name=meshnet_pinna
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=meshnet_train_%j.out
#SBATCH --error=meshnet_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

# =========================
# Easy hyperparameter block
# =========================
CONDA_ENV="anthro_env"
PROJECT_DIR="/scratch/agxl1e22/huawei2/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
MESH_DIR="${PROJECT_DIR}/data/mesh"
LANDMARKS_DIR="${PROJECT_DIR}/data/landmarks"
CROPPED_DIR="${PROJECT_DIR}/data/cropped_mesh"
RUN_ROOT="${PROJECT_DIR}/runs"

SEED=42
VAL_RATIO=0.2
INPUT_MODE="precropped"
MAX_FACES=1024
MIRROR_RIGHT_EAR=0
WORKERS=8
FACE_CACHE=1
 
# MeshNet architecture (paper defaults)
NUM_KERNEL=64
SIGMA=0.2
AGGREGATION_METHOD="Concat"   # Concat/Max/Average
MASK_RATIO=0.0
HEAD_CHANNELS="512,256"
DROPOUT=0.2
 
# Augmentation (jitter on face centers, training split only)
AUGMENT=0
JITTER_SIGMA=0.002
JITTER_CLIP=0.01
 
EPOCHS=200
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
 
RUN_NAME="meshnet_${INPUT_MODE}_mf${MAX_FACES}_${AGGREGATION_METHOD}_${LOSS}"
 
# =========================
# Run folder and environment
# =========================
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_NAME}_seed${SEED}_lr${LEARNING_RATE}_bs${BATCH_SIZE}_${SLURM_JOB_ID}_${TIMESTAMP}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
FACE_CACHE_DIR="${PROJECT_DIR}/cache/meshnet_faces"
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
CROPPED_DIR=${CROPPED_DIR}
RUN_ROOT=${RUN_ROOT}
RUN_NAME=${RUN_NAME}
SEED=${SEED}
VAL_RATIO=${VAL_RATIO}
INPUT_MODE=${INPUT_MODE}
MAX_FACES=${MAX_FACES}
MIRROR_RIGHT_EAR=${MIRROR_RIGHT_EAR}
WORKERS=${WORKERS}
FACE_CACHE=${FACE_CACHE}
FACE_CACHE_DIR=${FACE_CACHE_DIR}
NUM_KERNEL=${NUM_KERNEL}
SIGMA=${SIGMA}
AGGREGATION_METHOD=${AGGREGATION_METHOD}
MASK_RATIO=${MASK_RATIO}
HEAD_CHANNELS=${HEAD_CHANNELS}
DROPOUT=${DROPOUT}
AUGMENT=${AUGMENT}
JITTER_SIGMA=${JITTER_SIGMA}
JITTER_CLIP=${JITTER_CLIP}
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
    --cropped-dir "${CROPPED_DIR}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --val-ratio "${VAL_RATIO}"
    --input-mode "${INPUT_MODE}"
    --max-faces "${MAX_FACES}"
    --seed "${SEED}"
    --workers "${WORKERS}"
    --num-kernel "${NUM_KERNEL}"
    --sigma "${SIGMA}"
    --aggregation-method "${AGGREGATION_METHOD}"
    --mask-ratio "${MASK_RATIO}"
    --head-channels "${HEAD_CHANNELS}"
    --dropout "${DROPOUT}"
    --jitter-sigma "${JITTER_SIGMA}"
    --jitter-clip "${JITTER_CLIP}"
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
 
if [[ "${MIRROR_RIGHT_EAR}" == "1" ]]; then
    TRAIN_ARGS+=(--mirror-right-ear)
else
    TRAIN_ARGS+=(--no-mirror-right-ear)
fi
 
if [[ "${AUGMENT}" == "1" ]]; then
    TRAIN_ARGS+=(--augment)
else
    TRAIN_ARGS+=(--no-augment)
fi
 
if [[ "${FACE_CACHE}" == "1" ]]; then
    mkdir -p "${FACE_CACHE_DIR}"
    TRAIN_ARGS+=(--face-cache-dir "${FACE_CACHE_DIR}")
fi
 
printf '%q ' python train_meshnet.py "${TRAIN_ARGS[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
cat "${RUN_DIR}/command.txt"
 
python train_meshnet.py "${TRAIN_ARGS[@]}"
 
echo "Training complete"
echo "Best checkpoint: ${CHECKPOINT_DIR}/best_model.pt"
date
 