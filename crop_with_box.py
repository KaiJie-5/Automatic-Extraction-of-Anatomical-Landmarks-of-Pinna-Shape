"""Use the trained Box Regressor to crop meshes, validate coverage, and save PLYs."""

import os
import torch
import numpy as np
from pathlib import Path

from src.dataset import Dataset as MeshLandmarkDataset
from src.preprocessing import compute_mesh_normalization
from src.ear_crop import (
    sample_crop_point_features, 
    fit_crop_config_from_training_landmarks, 
    export_subject_crop_plys,
    compute_calibrated_asymmetric_extents,
    crop_box_from_center_extents,
    points_inside_box
)
from src.pointnet2_model import PointNet2BoxRegressor
from src.torch_dataset import split_subject_ids

def main():
    # --- Configuration ---
    mesh_dir = "data/mesh"
    landmarks_dir = "data/landmarks"
    model_path = "checkpoints_box/best_box_model.pt"
    output_dir = "predicted_crops"
    broad_margin = 0.40 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {model_path} onto {device}...")
    
    # Load the Model
    model = PointNet2BoxRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # Load the Dataset
    base_dataset = MeshLandmarkDataset(mesh_dir, landmarks_dir)
    train_ids, val_ids = split_subject_ids(base_dataset, 0.2, 0)
    
    # Get Current Broad Crop Config
    print("Re-calculating broad crop inputs...")
    broad_crop_config = fit_crop_config_from_training_landmarks(
        base_dataset, train_ids, margin=broad_margin
    )

    # Calibrate Asymmetric Safe Extents
    print("Calibrating asymmetric extents from training predictions...")
    safe_extents = compute_calibrated_asymmetric_extents(
        dataset=base_dataset,
        subject_ids=train_ids,
        broad_crop_config=broad_crop_config,
        box_model=model,
        ear_points=8192,
        device=device,
        margin=1.05,                 # Tight base margin for X and Z
        percentile=98.0,            # Cover the worst-case training prediction
        tta_runs=5,
        axis_multiplier=(1.0, 1.80, 1.0) # ONLY expand the Y-axis (index 1)
    )

    print(f"\nStarting inference and validation on {len(val_ids)} validation subjects...")
    
    total_ears = 0
    perfect_ears = 0

    # Process each validation subject
    for subject_id in val_ids:
        print(f"\nCropping {subject_id}...")
        
        idx = base_dataset.subject_ids.index(subject_id)
        mesh, left_lm, right_lm = base_dataset[idx]
        transform = compute_mesh_normalization(mesh)

        subject_tight_config = {}

        # Loop through both ears and their respective ground-truth landmarks
        for ear, landmarks in [("left", left_lm), ("right", right_lm)]:
            total_ears += 1
            ear_seed = idx * 2 + (0 if ear == "left" else 1)
            centers = []

            # Test-Time Augmentation (Predict 5 times)
            for k in range(5):
                broad_points = sample_crop_point_features(
                    mesh=mesh,
                    transform=transform,
                    crop_box=broad_crop_config[ear],
                    num_points=8192,
                    mirror_y=False,
                    seed=ear_seed + 1000 * k
                )

                point_tensor = torch.from_numpy(broad_points).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    pred_box = model(point_tensor).squeeze(0).cpu().numpy()
                
                centers.append(pred_box[:3])

            # Use the median prediction
            pred_center = np.median(np.stack(centers), axis=0)

            # Create the Asymmetric Tight Crop
            tight_box = crop_box_from_center_extents(
                center=pred_center,
                negative_extent=safe_extents[ear]["neg"],
                positive_extent=safe_extents[ear]["pos"],
                scale=1.0 
            )
            subject_tight_config[ear] = tight_box

            # VALIDATION: Check Ground-Truth Landmark Coverage
            landmarks_norm = transform.normalize_xyz(landmarks.astype(np.float32))
            inside = points_inside_box(landmarks_norm, tight_box)
            inside_count = int(inside.sum())

            if inside_count == 85:
                perfect_ears += 1
                print(f"  [{ear.upper()}] SUCCESS: 85/85 landmarks inside crop.")
            else:
                missing_ids = np.where(~inside)[0].tolist()
                print(f"  [{ear.upper()}] FAILED: {inside_count}/85 landmarks inside. Missing IDs: {missing_ids}")

        # Slice the 3D mesh and save the files
        export_subject_crop_plys(
            dataset=base_dataset,
            subject_ids=[subject_id], 
            crop_config=subject_tight_config,
            output_dir=output_dir,
            split_name="validation",
            ear_points=8192,
            oversample_factor=8,
            max_attempts=5
        )

    print(f"\n================ VALIDATION SUMMARY ================")
    print(f"Total Full Coverage: {perfect_ears}/{total_ears} ears ({perfect_ears/total_ears:.2%})")
    print(f"All tight crops have been saved to '{output_dir}/crops/validation'")

if __name__ == "__main__":
    main()