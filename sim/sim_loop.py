import mujoco
import numpy as np
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

    def mix(self, thrust_d, roll_d, pitch_d, yaw_d):
        """
        Quadcopter motor mixer that converts desired thrust, pitch,roll,and yaw
        to individual motor commands. This is the main mechanism by which a controller
        will affect the quadcopter's rotors. 
        
        :param thrust_d: Desired thrust
        :param roll_d: Desired roll
        :param pitch_d: Desired pitch
        :param yaw_d: Desired yaw
        """
        motor1_pwm = thrust_d - roll_d + pitch_d + yaw_d
        motor2_pwm = thrust_d + roll_d - pitch_d + yaw_d
        motor3_pwm = thrust_d + roll_d + pitch_d - yaw_d
        motor4_pwm = thrust_d - roll_d - pitch_d - yaw_d
        res= np.array([motor1_pwm,motor2_pwm,motor3_pwm,motor4_pwm])

        # clip motor forces to be in the range [0,7.0]. This can be moved outside the mix function if needed
        # Note that this is hardcoded to match the ctrlrange of our model's actuators.
        # This should be changed if a different model is used; TODO to parameterize the ctrlrange somehow
        np.clip(res, 0, 7.0)
        return res

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
