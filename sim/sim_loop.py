import mujoco

class Simulator:
    """
    Class which holds the simulated model, data, and failures we want to simulate,
    as well as time for future visualization and analysis purposes
    """
    def __init__(self, model, data, failures=None):
        """
        Initialize the Simulator object, capturing any failures
        if any are passed in. If not, a basic hover simulation with
        no failures is created by default.
        """
        self.model = model
        self.data = data
        self.failures = failures or []
        self.time = 0.0

    def step(self, dt=0.002):
        """
        Apply failures and step the simulation ahead.
        
        :param dt: time step length
        """
        for f in self.failures:
            f.apply(self, self.time)

        mujoco.mj_step(self.model, self.data)
        self.time += dt
