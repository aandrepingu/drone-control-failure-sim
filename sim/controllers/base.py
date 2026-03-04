
class BaseController:
    """
    Base class for our controllers.
    """
    def compute_control(self, state, target):
        """
        state: dict or structured object with position, velocity, etc.
        target: desired reference
        returns: motor commands (np.array)
        """
        raise NotImplementedError

    def reset(self):
        """Optional reset for integrators, etc."""
        raise NotImplementedError