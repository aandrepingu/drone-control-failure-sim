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
        self.kv_xy = 0.8
        # initialize individual PIDs for altitude, roll, pitch, yaw
        self.x_pid = PID(1.5, 0.0, 0.0, dt, output_limits=(-2, 2))
        self.y_pid = PID(1.5, 0.0, 0.0, dt, output_limits=(-2, 2))
        self.alt_pid = PID(6.0, 2.0, 3.0, dt,integral_limits=(-2,2),output_limits=(-2,2),is_angle=False)
        self.roll_pid = PID(4.0, 0.0, 1.5, dt,integral_limits=(-2,2),output_limits=(-2,2),is_angle=True)
        self.pitch_pid = PID(4.0, 0.0, 1.5, dt,integral_limits=(-2,2),output_limits=(-2,2),is_angle=True)
        self.yaw_pid = PID(1.0, 0.0, 0.3, dt,integral_limits=(-2,2),output_limits=(-2,2),is_angle=True)



    def reset(self):
        self.alt_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.x_pid.reset()
        self.y_pid.reset()

    def compute_control(self, state, target):
        """
        state:
            {
                "pos": np.array([x,y,z]),
                "euler": np.array([roll,pitch,yaw]),
                "vel": np.array([vx,vy,vz]),
                "ang_vel": np.array([wx,wy,wz]),
                "accel" : np.array([ax,ay,az])
            }

        target:
            target = {
                "pos": np.array([x, y, z]),
                "yaw": float
            }

        """
        x, y, z = state["pos"]
        vx, vy, vz = state["vel"]
        
        x_target, y_target, z_target = target["pos"]
        
        roll, pitch, yaw = state["euler"]
        roll_rate, pitch_rate, yaw_rate = state["ang_vel"]

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

         # position error (world frame)
        ex = x_target - x
        ey = y_target - y

        # rotate error into body frame
        ex_body =  cos_yaw * ex + sin_yaw * ey
        ey_body = -sin_yaw * ex + cos_yaw * ey

        # now compute desired velocity in BODY frame
        vx_des = self.x_pid.update(0, ex_body)   # target = error
        vy_des = self.y_pid.update(0, ey_body)

        # world → body frame
        vx_body =  cos_yaw * vx + sin_yaw * vy
        vy_body = -sin_yaw * vx + cos_yaw * vy

        pitch_des = -vx_des - self.kv_xy * vx_body
        roll_des  =  vy_des - self.kv_xy * vy_body

        max_angle = 0.4  # ~23 degrees

        roll_des = np.clip(roll_des, -max_angle, max_angle)
        pitch_des = np.clip(pitch_des, -max_angle, max_angle)

        thrust_correction = self.alt_pid.update(z, z_target) - vz
        thrust = self.m * self.g / 4 + thrust_correction

       
        # get output from attitude pids, including angular velocities for derivative
        roll_cmd = self.roll_pid.update(
            measurement=roll,
            target=roll_des,
            derivative_override=roll_rate
        )

        pitch_cmd = self.pitch_pid.update(
            measurement=pitch,
            target=pitch_des,
            derivative_override=pitch_rate
        )
        yaw_cmd = self.yaw_pid.update( measurement=yaw,target=target["yaw"],derivative_override=yaw_rate)

        # apply motor mixing to these outputs
        return thrust, roll_cmd,pitch_cmd,yaw_cmd