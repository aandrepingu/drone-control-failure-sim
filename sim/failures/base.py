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

