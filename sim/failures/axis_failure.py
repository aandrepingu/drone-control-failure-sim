from sim.failures.base import Failure


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