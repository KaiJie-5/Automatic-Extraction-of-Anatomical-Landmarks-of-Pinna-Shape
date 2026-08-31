
import numpy as np

def compute_mean_landmark_distance(predicted: np.ndarray, ground_truth) -> float:
    """
    predicted: (N, 3)
    ground truth:   (N, 3)
    """
    return np.linalg.norm(predicted - ground_truth, axis=1).mean()
