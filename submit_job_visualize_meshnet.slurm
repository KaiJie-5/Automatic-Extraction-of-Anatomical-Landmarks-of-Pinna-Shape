#!/bin/bash
#SBATCH --job-name=meshnet_visualize
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=meshnet_visualize_%j.out
#SBATCH --error=meshnet_visualize_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agxl1e22@soton.ac.uk

# =========================
# Easy visualization block
# =========================
CONDA_ENV="anthro_env"
PROJECT_DIR="/scratch/agxl1e22/huawei2/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
MESH_DIR="${PROJECT_DIR}/data/mesh"
LANDMARKS_DIR="${PROJECT_DIR}/data/landmarks"
PRED_DIR="${PROJECT_DIR}/predictions_meshnet/val"
OUTPUT_DIR="${PROJECT_DIR}/landmark_overlays/val"

# Sphere size for the landmark markers:
#   POINT_RADIUS > 0  -> fixed radius in mesh units (mm)
#   POINT_RADIUS = 0  -> auto (RADIUS_FRACTION of each mesh's bounding-box diagonal)
POINT_RADIUS=0
RADIUS_FRACTION=0.004
SPHERE_SUBDIVISIONS=1

# NO_MESH=1 -> export only the colored landmark point clouds (skip merged mesh)
NO_MESH=0

# Optional: comma-separated subject ids to restrict to (leave empty "" for all)
SUBJECT_IDS=""

# =========================
# Environment
# =========================
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${OUTPUT_DIR}/visualize.log") 2> >(tee -a "${OUTPUT_DIR}/visualize.err" >&2)

echo "Output directory: ${OUTPUT_DIR}"
echo "Mesh dir: ${MESH_DIR}"
echo "Landmarks dir (GT): ${LANDMARKS_DIR}"
echo "Prediction dir: ${PRED_DIR}"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
date

cat > "${OUTPUT_DIR}/visualize_config.txt" <<EOF
CONDA_ENV=${CONDA_ENV}
PROJECT_DIR=${PROJECT_DIR}
MESH_DIR=${MESH_DIR}
LANDMARKS_DIR=${LANDMARKS_DIR}
PRED_DIR=${PRED_DIR}
OUTPUT_DIR=${OUTPUT_DIR}
POINT_RADIUS=${POINT_RADIUS}
RADIUS_FRACTION=${RADIUS_FRACTION}
SPHERE_SUBDIVISIONS=${SPHERE_SUBDIVISIONS}
NO_MESH=${NO_MESH}
SUBJECT_IDS=${SUBJECT_IDS}
TIMESTAMP=${TIMESTAMP}
EOF

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"

if [[ ! -d "${PRED_DIR}" ]]; then
    echo "ERROR: prediction dir not found: ${PRED_DIR}"
    echo "Run predict_meshnet.py first, or edit PRED_DIR in this slurm file."
    exit 1
fi

VIS_ARGS=(
    --mesh-dir "${MESH_DIR}"
    --landmarks-dir "${LANDMARKS_DIR}"
    --pred-dir "${PRED_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --point-radius "${POINT_RADIUS}"
    --radius-fraction "${RADIUS_FRACTION}"
    --sphere-subdivisions "${SPHERE_SUBDIVISIONS}"
)

if [[ "${NO_MESH}" == "1" ]]; then
    VIS_ARGS+=(--no-mesh)
fi

if [[ -n "${SUBJECT_IDS}" ]]; then
    VIS_ARGS+=(--subject-ids "${SUBJECT_IDS}")
fi

printf '%q ' python visualize_landmarks.py "${VIS_ARGS[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"
cat "${OUTPUT_DIR}/command.txt"

python visualize_landmarks.py "${VIS_ARGS[@]}"

echo "Visualization complete"
echo "Overlays written to: ${OUTPUT_DIR}/<subject>_overlay.ply (mesh gray, GT green, pred red)"
date