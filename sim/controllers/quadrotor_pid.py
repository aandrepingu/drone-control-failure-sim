import numpy as np
from .base import BaseController
from .pid import PID


class QuadrotorPIDController(BaseController):

    def __init__(self, mass, dt: float):
        
        self.dt = dt
        self.m = mass
        # self.L = params["arm_length"]
        # self.k_yaw = params["yaw_coeff"]
        self.g = 9.81

        # initialize individual PIDs for altitude, roll, pitch, yaw
        self.alt_pid = PID(6.0, 2.0, 3.0, dt, integral_limits=(-2,2), is_angle=False)
        self.roll_pid = PID(4.0, 0.0, 1.5, dt, integral_limits=(-2,2), is_angle=True)
        self.pitch_pid = PID(4.0, 0.0, 1.5, dt, integral_limits=(-2,2), is_angle=True)
        self.yaw_pid = PID(1.0, 0.0, 0.3, dt, integral_limits=(-2,2), is_angle=True)



    def reset(self):
        self.alt_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()

    def compute_control(self, state, target):
        """
        state:
            {
                "pos": np.array([x,y,z]),
                "euler": np.array([roll,pitch,yaw]),
            }

        target:
            {
                "z": float,
                "roll": float,
                "pitch": float,
                "yaw": float,
            }
        """

        z = state["pos"][2]
        roll, pitch, yaw = state["euler"]

        # get output from altitude pid
        thrust_correction = self.alt_pid.update(z,target["z"])
        # thrust = self.m * self.g / 4 + thrust_correction
        total_thrust = self.m * self.g + thrust_correction

        # get output from attitude pids 
        roll_cmd = self.roll_pid.update( roll, target["roll"])
        pitch_cmd= self.pitch_pid.update(pitch,target["pitch"])
        yaw_cmd = self.yaw_pid.update( yaw,target["yaw"])

        # apply motor mixing to these outputs
        return total_thrust, roll_cmd,pitch_cmd,yaw_cmd