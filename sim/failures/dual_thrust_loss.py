from sim.failures.base import Failure

class ThrustLoss(Failure):
    def __init__(self, start_time, alpha_1, alpha_2, prop_1, prop_2):
        super().__init__(start_time)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        if self.prop_1 < 0 or self.prop_1 > 3:
            raise ValueError(f"Propeller {self.prop} out of range 0-3")
        if self.prop_2 < 0 or self.prop_2 > 3:
            raise ValueError(f"Propeller {self.prop} out of range 0-3")
        if self.prop_1 == self.prop_2:
            raise ValueError("Propellers must be unique")

    
    def apply(self, sim, t):
        if self.done or t < self.start_time:
            return
        sim.data.ctrl[self.prop_1] *= self.alpha_1
        sim.data.ctrl[self.prop_2] *= self.alpha_2
        self.done = True
        self.active = False