from sim.failures.base import Failure

class StuckActuator(Failure):
    """
    Actuator freezes at the value it had when the failure occurred.
    """
    def __init__(self, start_time, index):
        super().__init__(start_time)
        self.index = index
        self.stuck_value = None

    def apply(self, sim, t):
        if t < self.start_time:
            return
        
        if self.stuck_value is None:
            self.stuck_value = sim.data.ctrl[self.index]
        
        sim.data.ctrl[self.index] = self.stuck_value