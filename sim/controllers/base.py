import numpy as np


class BaseController:
    """
    Base class for our controllers.
    """

    def compute_control(
        self, obs: np.ndarray, info: dict, target_state: dict
    ) -> np.ndarray:
        """
        state: dict or structured object with position, velocity, etc.
        target: desired reference
        returns: motor commands (np.array)
        """

    def reset(self):
        """Optional reset for integrators, etc."""
