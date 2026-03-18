from sim.failures.base import Failure


class EngineFlameout(Failure):
    def __init__(self, start_time):
        super().__init__(start_time)

    def apply(self, sim, t):
        if t < self.start_time:
            return
        sim.data.ctrl[0] = 0.0

