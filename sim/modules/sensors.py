import mujoco
import numpy as np

from sim.core.sim_module import SimModule
from sim.core.state import DynamicsState, SensorData


class SensorModule(SimModule):
    """
    Module that applies AWGN to dynamics measurements to create realistic state measurements.
    """

    def __init__(self, dynamics_state: DynamicsState, sensor_data: SensorData):
        super().__init__(period=2, offset=0)
        self.dynamics_state = dynamics_state
        self.sensor_data = sensor_data
        self.seed = 0
        self.rng = np.random.RandomState(self.seed)

        # Base noise parameters (Standard Deviations)
        self.imu_accel_noise = 0.1
        self.imu_gyro_noise = 0.01
        self.baro_noise = 0.05
        self.gps_pos_noise = 0.05
        self.gps_vel_noise = 0.1

        # State for autocorrelated attitude error (faking an EKF output)
        self.attitude_error = np.zeros(3)

    def apply_seed(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Reset autocorrelated state to ensure deterministic runs
        self.attitude_error = np.zeros(3)

    def HandleDispatch(self, current_time: int):
        """
        Applies AWGN to ground-truth dynamics.
        """
        self.sensor_data.imu_accel[:] = (
            self.dynamics_state.acceleration
            + self.rng.normal(0, self.imu_accel_noise, 3)
        )

        self.sensor_data.imu_gyro[:] = (
            self.dynamics_state.angular_velocity
            + self.rng.normal(0, self.imu_gyro_noise, 3)
        )

        self.sensor_data.baro_altitude = float(
            self.dynamics_state.position[2] + self.rng.normal(0, self.baro_noise)
        )

        self.sensor_data.gps_position[:] = (
            self.dynamics_state.position + self.rng.normal(0, self.gps_pos_noise, 3)
        )
        self.sensor_data.gps_velocity[:] = (
            self.dynamics_state.linear_velocity
            + self.rng.normal(0, self.gps_vel_noise, 3)
        )

        # apply low-pass filter to the noise to simulate the smooth but slightly
        # inaccurate tracking of a real attitude estimator
        noise_input = self.rng.normal(0, 0.05, 3)
        self.attitude_error = 0.95 * self.attitude_error + 0.05 * noise_input
        self.sensor_data.estimated_attitude[:] = (
            self.dynamics_state.euler_angles + self.attitude_error
        )
