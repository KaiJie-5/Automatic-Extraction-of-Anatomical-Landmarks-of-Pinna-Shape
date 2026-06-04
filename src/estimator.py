from typing import Tuple
import numpy as np
from trimesh import Trimesh


class LandmarkExtractor:
    """Landmark extractor implementation."""

    def __init__(self):
        """This function needs to have default values for all arguments. These will be used when instantiating the class
        during the evaluation of your submission."""
        pass

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Method to extract left and right ear landmarks from a 3D mesh. Both output arrays need to be of size (85, 3),
        and need to contain the 85 landmark coordinates for the left and right ear in the correct order.
        This function will be called during the evaluation on the hidden test dataset.
        """
        raise NotImplementedError
