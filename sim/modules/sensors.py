import mujoco
import numpy as np

from sim.core.sim_module import SimModule
from sim.core.state import DynamicsState, SensorData


class SensorModule(SimModule):
    """
    Module that applies noise to dynamics measurements to create realistic state measurements.
    """

    def __init__(self, dynamics_state: DynamicsState, sensor_data: SensorData):
        super().__init__(period=2, offset=0)
        self.dynamics_state = dynamics_state
        self.sensor_data = sensor_data

    def HandleDispatch(self, current_time: int):
        """
        Apply noise to dynamics state measurements.

        Not implemented yet; for now dynamics data is copied over
        """
        # IMU measurements (Specific force / Angular rates)
        self.sensor_data.imu_accel[:] = self.dynamics_state.acceleration
        self.sensor_data.imu_gyro[:] = self.dynamics_state.angular_velocity

        # Barometer altitude (Z-axis position)
        self.sensor_data.baro_altitude = float(self.dynamics_state.position[2])

        # GPS position and velocity
        self.sensor_data.gps_position[:] = self.dynamics_state.position
        self.sensor_data.gps_velocity[:] = self.dynamics_state.linear_velocity

        # attitude (euler angles)
        self.sensor_data.estimated_attitude[:] = self.dynamics_state.euler_angles
