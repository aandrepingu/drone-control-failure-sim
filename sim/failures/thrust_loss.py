from sim.failures.base import Failure

class ThrustLoss(Failure):
    """
    Scales down total thrust (battery sag, ESC, motor damage)
    alpha = 1.0 -> no loss
    alpha = 0.0 -> complete engine failure
    """
    def __init__(self, start_time, alpha):
        super().__init__(start_time)
        self.alpha = alpha
    
    def apply(self, sim, t):
        if t < self.start_time:
            return
        sim.data.ctrl[0] *= self.alpha