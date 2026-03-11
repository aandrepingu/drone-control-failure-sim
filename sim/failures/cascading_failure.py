from sim.failures.base import Failure

class ThrustLoss(Failure):
    def __init__(self, start_time1, start_time2, alpha_1, alpha_2, prop_1, prop_2):
        super().__init__(start_time1, start_time2)
        self.start_time1 = start_time1
        self.start_time2 = start_time2
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.prop_1 = prop_1
        self.prop_2 = prop_2
        if self.prop_1 < 0 or self.prop_1 > 3:
            raise ValueError(f"Propeller {self.prop} out of range 0-3")
        if self.prop_2 < 0 or self.prop_2 > 3:
            raise ValueError(f"Propeller {self.prop} out of range 0-3")
        if self.prop_1 == self.prop_2 or self.start_time1 == self.start_time2:
            raise ValueError("Propellers and times must be unique")

    
    def apply(self, sim, t1, t2):
        if self.done or (t1 < self.start_time1 and t2 < self.start_time2):
            return
        sim.data.ctrl[self.prop_1] *= self.alpha_1
        sim.data.ctrl[self.prop_2] *= self.alpha_2
        self.done = True
        self.active = False