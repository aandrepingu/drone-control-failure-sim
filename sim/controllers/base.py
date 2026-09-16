import numpy as np


class BaseController:
    """
    Base class for our controllers.
    """

    def compute_control(
        self,
        current_pos: np.ndarray,
        current_euler: np.ndarray,
        current_gyro: np.ndarray,
        target_pos: np.ndarray,
        target_yaw: float | None = None,
    ):
        """
        state: dict or structured object with position, velocity, etc.
        target: desired reference
        returns: motor commands (np.array)
        """
        raise NotImplementedError

    def reset(self):
        """Optional reset for integrators, etc."""
        raise NotImplementedError
