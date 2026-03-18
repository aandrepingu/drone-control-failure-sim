from dataclasses import dataclass
import numpy as np
import mujoco

@dataclass
class SimConfig:
    """
    Dataclass representing a starting configuration for the drone,
    consisting of starting position, velocity, and quaternion
    """
    position: np.ndarray # shape (3,)
    velocity: np.ndarray # shape (3,)
    quat: np.ndarray     # shape (4,) for orientation

    @classmethod
    def random(cls,
               pos_range,
               vel_range,
               tilt_range,
               yaw_range):
        """
        Generates a random SimConfig object based on ranges for position, velocity,
        tilt and yaw.
        
        :param pos_range: range for position [min,max]
        :param vel_range: range for velocity [min,max]
        :param tilt_range: range for tilt [min,max]
        :param yaw_range: range for yaw [min,max]
        """
        pos = np.random.uniform(*pos_range, size=3)
        vel = np.random.uniform(*vel_range, size=3)

        tilt_axis = np.random.randn(3)
        tilt_axis[2] = 0  # zero out yaw component so it is handled separately

        tilt_axis /= np.linalg.norm(tilt_axis)
        
        tilt_angle = np.random.uniform(*tilt_range)
        yaw_angle = np.random.uniform(*yaw_range)

        axis = np.array([0, 0, 1])
        
        q_tilt = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_tilt, tilt_axis, tilt_angle)

        q_yaw=np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_yaw, axis, yaw_angle)

        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat,q_yaw, q_tilt)

        return cls(pos, vel, quat)