import numpy as np
from .base import BaseController
from .pid import PID
from scipy.spatial.transform import Rotation



class QuadrotorPIDController(BaseController):

    def __init__(self, mass, dt: float):
        
        self.dt = dt
        self.m = mass
        self.g = 9.81
        self.base_thrust = self.m * self.g

        # Outer loop: position error → desired angle
        self.max_angle = 0.4  # radians

        self.x_pid = PID(0.67, 0.021, 2., dt, output_limits=(-self.max_angle,self.max_angle))
        self.y_pid = PID(0.67, 0.021, 2., dt, output_limits=(-self.max_angle,self.max_angle))
        self.z_pid = PID(10.0, 0.1, 5.0,  dt, output_limits=(-3,3),integral_limits=(-1,1))

        # Inner loop: attitude error → torques
        torque_limit = 1.0
        self.roll_pid  = PID(3.0, 0.0, 0.4, dt, output_limits=(-torque_limit, torque_limit), is_angle=True)
        self.pitch_pid = PID(3.0, 0.0, 0.4, dt, output_limits=(-torque_limit, torque_limit), is_angle=True)
        self.yaw_pid   = PID(0.04, 0.0, 0.3, dt, output_limits=(-torque_limit, torque_limit), is_angle=True)

        # Clamp desired angles to safe range


    def reset(self):
        for pid in [self.x_pid, self.y_pid, self.z_pid,
                    self.roll_pid, self.pitch_pid, self.yaw_pid]:
            pid.reset()

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
        pos        = state["pos"]
        vel        = state["vel"]
        euler      = state["euler"]   # [roll, pitch, yaw]
        ang_vel    = state["ang_vel"] # body frame

        roll, pitch, yaw = euler
        roll_rate, pitch_rate, yaw_rate = ang_vel

        x_target, y_target, z_target = target["pos"]


        # --- Altitude ---
        thrust_correction = self.z_pid.update(measurement=pos[2], target=z_target)
        tilt_compensation =  np.cos(roll) * np.cos(pitch)
        total_thrust = (self.base_thrust + thrust_correction) / (
            4 * tilt_compensation
        )

        err_x_world = x_target - pos[0]
        err_y_world = y_target - pos[1]
        yaw_target = np.arctan2(err_y_world, err_x_world)
        if np.linalg.norm([err_x_world,err_y_world]) < 0.5:
            yaw_target = yaw


        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Rotate error into drone body frame
        err_x_body =  err_x_world * cos_yaw + err_y_world * sin_yaw
        err_y_body = -err_x_world * sin_yaw + err_y_world * cos_yaw

        yaw_error = (yaw_target - yaw + np.pi) % (2 * np.pi) - np.pi

        pitch_des =  np.clip(
            self.x_pid.update(target=0.0, measurement=-err_x_body, #derivative_override=vel_body[0]
                              ),
            -self.max_angle, self.max_angle,
        )
        roll_des = -np.clip(
            self.y_pid.update(target=0.0, measurement=-err_y_body,#derivative_override=vel_body[1]
                              ),
            -self.max_angle, self.max_angle
        )


        roll_cmd  = self.roll_pid.update(measurement=roll,  target=roll_des,  #derivative_override=roll_rate
                                         )
        pitch_cmd = self.pitch_pid.update(measurement=pitch, target=pitch_des, #derivative_override=pitch_rate
                                          )
        # yaw_cmd   = self.yaw_pid.update(measurement=yaw_error, target=0.0)
        yaw_cmd   = self.yaw_pid.update(measurement=yaw, target=yaw_target)
        print(yaw_cmd,yaw_error)
        # print((pitch_des, roll_des,yaw_cmd))

        return total_thrust, roll_cmd, pitch_cmd, yaw_cmd