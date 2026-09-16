import mujoco
import numpy as np

from sim.controllers.base import BaseController
from sim.core.sim_module import SimModule
from sim.core.state import ActuatorState, ControlTargets, SensorData


class ControlModule(SimModule):
    """
    Module that applies control actions to the drone. Uses a predefined controller
    that can either be RL-based or a classical controller like PID.
    """

    def __init__(
        self,
        controller: BaseController,
        sensor_data: SensorData,
        control_targets: ControlTargets,
        actuator_state: ActuatorState,
    ):
        super().__init__(period=2, offset=0)
        self.controller = controller
        self.sensor_data = sensor_data
        self.control_targets = control_targets
        self.actuator_state = actuator_state

        # motor mixer matrix: multiply this by
        # [thrust, roll, pitch, yaw] column vector to get
        # motor outputs for motor 0,1,2,3
        self.mixer_matrix = np.array(
            [
                [1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0],
            ],
            dtype=float,
        )

    def HandleDispatch(self, current_time: int):
        """
        Compute control actions and update the actuator state with desired actions
        """

        thrust, roll_cmd, pitch_cmd, yaw_cmd = self.controller.compute_control(
            current_pos=self.sensor_data.gps_position,
            current_euler=self.sensor_data.estimated_attitude,
            current_gyro=self.sensor_data.imu_gyro,
            target_pos=self.control_targets.target_pos,
        )

        self.control_targets.desired_thrust = thrust
        self.control_targets.desired_torques[:] = [roll_cmd, pitch_cmd, yaw_cmd]

        command_vector = np.array([thrust, roll_cmd, pitch_cmd, yaw_cmd])
        raw_motor_commands = self.mixer_matrix @ command_vector

        self.actuator_state.motor_thrusts[:] = np.clip(raw_motor_commands, 0.0, 7.0)

    def _reset_state(self):
        self.controller.reset()
