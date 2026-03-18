from sim.failures.base import Failure

class ThrustLoss(Failure):
    """
    Scales down total thrust (battery sag, ESC, motor damage) to a single propeller
    alpha = 1.0 -> no loss
    alpha = 0.0 -> complete engine failure
    """
    def __init__(self, start_time, alpha,prop):
        super().__init__(start_time)
        self.alpha = alpha
        self.prop = prop
        if self.prop < 0 or self.prop > 3:
            raise ValueError(f"Propeller {self.prop} out of range 0-3")

    
    def apply(self, sim, t):
        """
        Apply thrust scaling to a given propeller
        
        :param self: Description
        :param sim: Description
        :param t: Description
        """
        if self.done or t < self.start_time:
            return
        sim.data.ctrl[self.prop] *= self.alpha
        self.done = True
        self.active = False
