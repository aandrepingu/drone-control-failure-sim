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
        
        # set our drone to hover initially; this should be parameterized out later
        # when we get a proper "scenario" setup design
        hover_thrust = self.model.body_mass.sum() * 9.81 / 4
        self.data.ctrl[:] = hover_thrust

    def hover_pd(self):
        """
        Basic PD controller. 
        Note that ctrl is of the form [thrust1, thrust2, thrust3, thrust4, roll, pitch, yaw]

        where ctrl[0:4] are thrust sources on each propeller and ctrl[4:7] are actuators
        """
        wx, wy, wz = self.data.qvel[3:6]
        kp = 0.5
        kd = 0.05

        self.data.ctrl[4] = -kp * wx - kd * wx
        self.data.ctrl[5] = -kp * wy - kd * wy
        self.data.ctrl[6] = -kp * wz - kd * wz

    def step(self, dt=0.002):
        """
        Apply failures and step the simulation ahead.
        
        :param dt: time step length
        """
        for f in self.failures:
            f.apply(self, self.time)
        self.hover_pd()
        mujoco.mj_step(self.model, self.data)
        self.time += dt
