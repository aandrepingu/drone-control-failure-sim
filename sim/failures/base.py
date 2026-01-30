class Failure:
    """
    Base class for all failures we will model e.g. propeller loss.
    Concrete failure types should inherit from this class and 
    implement the apply method to be injected into a simulation.
    """
    def __init__(self, start_time):
        self.start_time = start_time
        self.active = False
        self.done = False

    def apply(self, sim, t):
        raise NotImplementedError

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

class EngineFlameout(Failure):
    def __init__(self, start_time):
        super().__init__(start_time)

    def apply(self, sim, t):
        if t < self.start_time:
            return
        sim.data.ctrl[0] = 0.0

class AxisFailure(Failure):
    """
    axis:
    1 -> roll (rateX)
    2 -> pitch (rateY)
    3 -> yaw (rateZ)
    scale:
    0.0 -> dead
    0.2 -> degraded
    """
    def __init__(self, start_time, axis, scale=0.0):
        super().__init__(start_time)
        self.axis = axis
        self.scale = scale

    def apply(self, sim, t):
        if t < self.start_time:
            return
        sim.data.ctrl[self.axis] *= self.scale

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