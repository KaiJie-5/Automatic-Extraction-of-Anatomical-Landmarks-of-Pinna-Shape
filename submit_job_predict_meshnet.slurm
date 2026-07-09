#!/bin/bash
#SBATCH --job-name=meshnet_predict
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=meshnet_predict_%j.out
#SBATCH --error=meshnet_predict_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agxl1e22@soton.ac.uk

# =========================
# Easy prediction block
# =========================
CONDA_ENV="anthro_env"
PROJECT_DIR="/scratch/agxl1e22/huawei2/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
MESH_DIR="${PROJECT_DIR}/data/mesh"
LANDMARKS_DIR="${PROJECT_DIR}/data/landmarks"
CROPPED_DIR="${PROJECT_DIR}/data/cropped_mesh"

CHECKPOINT="${PROJECT_DIR}/runs/meshnet_precropped_mf1024_Concat_smooth_l1_seed42_lr1e-3_bs32_1139904_20260702_023059/checkpoints/best_model.pt"
FACE_CACHE_DIR="${PROJECT_DIR}/cache/meshnet_faces"
SPLIT="val"

# Optional: override max_faces (leave empty "" to use the checkpoint's value)
MAX_FACES=""

# Optional: comma-separated subject ids to override the split (leave empty "")
SUBJECT_IDS=""

# Where predictions are written
OUTPUT_DIR="${PROJECT_DIR}/predictions_meshnet/${SPLIT}"

# =========================
# Run folder and environment
# =========================
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${OUTPUT_DIR}/predict.log") 2> >(tee -a "${OUTPUT_DIR}/predict.err" >&2)

echo "Output directory: ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

cat > "${OUTPUT_DIR}/predict_config.txt" <<EOF
CONDA_ENV=${CONDA_ENV}
PROJECT_DIR=${PROJECT_DIR}
MESH_DIR=${MESH_DIR}
LANDMARKS_DIR=${LANDMARKS_DIR}
CROPPED_DIR=${CROPPED_DIR}
CHECKPOINT=${CHECKPOINT}
FACE_CACHE_DIR=${FACE_CACHE_DIR}
SPLIT=${SPLIT}
MAX_FACES=${MAX_FACES}
SUBJECT_IDS=${SUBJECT_IDS}
OUTPUT_DIR=${OUTPUT_DIR}
TIMESTAMP=${TIMESTAMP}
EOF

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    echo "Edit CHECKPOINT in this slurm file to point at your run's best_model.pt"
    exit 1
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

PREDICT_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --mesh-dir "${MESH_DIR}"
    --landmarks-dir "${LANDMARKS_DIR}"
    --cropped-dir "${CROPPED_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --split "${SPLIT}"
    --device auto
)

if [[ -n "${FACE_CACHE_DIR}" ]]; then
    PREDICT_ARGS+=(--face-cache-dir "${FACE_CACHE_DIR}")
fi

if [[ -n "${MAX_FACES}" ]]; then
    PREDICT_ARGS+=(--max-faces "${MAX_FACES}")
fi

if [[ -n "${SUBJECT_IDS}" ]]; then
    PREDICT_ARGS+=(--subject-ids "${SUBJECT_IDS}")
fi

printf '%q ' python predict_meshnet.py "${PREDICT_ARGS[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"
cat "${OUTPUT_DIR}/command.txt"

python predict_meshnet.py "${PREDICT_ARGS[@]}"

echo "Prediction complete"
echo "Predicted landmarks written to: ${OUTPUT_DIR}"
echo "Combined 170-point files: ${OUTPUT_DIR}/<subject>_full_pred_landmarks.csv"
echo "Per-ear metrics: ${OUTPUT_DIR}/prediction_metrics.csv"
date