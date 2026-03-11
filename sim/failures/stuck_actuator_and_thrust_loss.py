from sim.failures.base import Failure

class StuckActuatorAndThrustLoss(Failure):
    def __init__(self, loss_time, stuck_time, alpha, stuck_prop, index):
        super().__init__(loss_time, stuck_time)
        self.loss_time = loss_time
        self.stuck_time = stuck_time
        self.alpha = alpha
        self.stuck_prop = stuck_prop
        self.index = index
        self.stuck_value = None
        if self.prop_1 < 0 or self.prop_1 > 3:
            raise ValueError(f"Propeller {self.prop} out of range 0-3")

    def apply(self, sim, st, lt):
        if self.done or (st < self.stuck_time and lt < self.loss_time):
            return
        
        if self.stuck_value is None:
            self.stuck_value = sim.data.ctrl[self.index]
        
        sim.data.ctrl[self.index] = self.stuck_value
        sim.data.ctrl[self.prop] *= self.alpha
        self.done = True
        self.active = False