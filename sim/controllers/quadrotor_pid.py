import numpy as np
from .base import BaseController
from .pid import PID


class QuadrotorPIDController(BaseController):

    def __init__(self, params: dict, dt: float):
        
        self.dt = dt
        self.m = params["mass"]
        self.L = params["arm_length"]
        self.k_yaw = params["yaw_coeff"]
        self.g = 9.81

        # initialize individual PIDs for altitude, roll, pitch, yaw
        self.alt_pid = PID(6.0, 2.0, 3.0, dt)
        self.roll_pid = PID(4.0, 0.0, 1.5, dt)
        self.pitch_pid = PID(4.0, 0.0, 1.5, dt)
        self.yaw_pid = PID(1.0, 0.0, 0.3, dt)



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
        thrust = self.m * self.g / 4 + thrust_correction

        # get output from attitude pids 
        roll_cmd = self.roll_pid.update( roll, target["roll"])
        pitch_cmd= self.pitch_pid.update(pitch,target["pitch"])
        yaw_cmd = self.yaw_pid.update( yaw,target["yaw"])

        # apply motor mixing to these outputs
        return thrust, roll_cmd,pitch_cmd,yaw_cmd