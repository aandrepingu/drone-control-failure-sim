import mujoco
import numpy as np
from core.sim_module import SimModule
from core.state import ActuatorState, DynamicsState
from scipy.spatial.transform import Rotation as R


class MujocoDynamicsModule(SimModule):
    """
    Module that handles mujoco dynamics and publishes the model state to the shared
    data object
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dynamics_state: DynamicsState,
        actuator_state: ActuatorState,
    ):
        super().__init__(period=2, offset=0)
        self.model = model
        self.data = data
        self.dynamics_state = dynamics_state
        self.actuator_state = actuator_state

    def HandleDispatch(self, current_time: int):
        """
        Apply calculated motor commands from the control module to the model,
        then publish dynamics data to the shared state objects.
        """
        # apply motor commands and advance model
        self.data.ctrl[:] = self.actuator_state.motor_thrusts
        mujoco.mj_step(self.model, self.data)

        # publish ground truth data to DynamicsState object
        self.dynamics_state.position[:] = self.data.qpos[0:3]

        # extract quaternion (mujoco format: [w,x,y,z])
        quat_wxyz = self.data.qpos[3:7]
        self.dynamics_state.quaternion[:] = quat_wxyz

        # convert quaternion to Euler angles [roll, pitch, yaw] in radians
        # scipy expects [x, y, z, w] format
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        self.dynamics_state.euler_angles[:] = R.from_quat(quat_xyzw).as_euler("xyz")

        # velocities and accelerations
        self.dynamics_state.linear_velocity[:] = self.data.qvel[0:3]
        self.dynamics_state.angular_velocity[:] = self.data.qvel[3:6]
        self.dynamics_state.acceleration[:] = self.data.qacc[0:3]
