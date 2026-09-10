from dataclasses import dataclass, field

import numpy as np


@dataclass
class DynamicsState:
    """
    Contains ground truth MuJoCo dynamics data
    """

    # -- Primary dynamics state --
    # Position vector [x,y,z]
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Linear velocity vector [vx, vy, vz]
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Quaternion [w, x, y, z]
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )

    # Angular velocity [p, q, r]
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Unsure if we need acceleration at all but leaving this here
    acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # [ax, ay, az]

    # -- Derived/helper state variables --
    # Euler angles [roll,pitch,yaw]
    euler_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class SensorData:
    """
    Contains simulated sensor data (with noise/drift/biases) that gets fed
    into control modules.
    """

    # IMU acceleration [ax, ay, az] (m/s^2)
    imu_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # IMU body rates [p, q, r] (rad/s)
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Barometer altitude (m)
    baro_altitude: float = 0.0

    # GPS position [x, y, z] (m)
    gps_position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # GPS velocity [vx, vy, vz] (m/s)
    gps_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Motor RPM
    motor_rpm: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class ControlTargets:
    """Target setpoints, set and used by the control module(s)."""

    # Desired [x, y, z]
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Desired [vx, vy, vz]
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Desired [roll, pitch, yaw]
    target_attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Collective thrust command
    desired_thrust: float = 0.0

    # Desired body torques [tau_x, tau_y, tau_z]
    desired_torques: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ActuatorState:
    """Raw motor output commands and calculated rotor forces."""

    # Normalized commands [0.0, 1.0] from mixer
    motor_commands: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Output force per rotor
    motor_thrusts: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # motor_pwm: np.ndarray = field(default_factory=lambda: np.zeros(4))      # Raw signal sent to ESCs


@dataclass
class FaultStatus:
    """Health flags and damage multipliers used for failure injection."""

    # [1.0 = healthy, 0.0 = total loss]
    actuator_effectiveness: np.ndarray = field(default_factory=lambda: np.ones(4))

    # True if any failure is active
    fault_active: bool = False

    # e.g., ["thrust_loss_m2"]
    active_fault_ids: list[str] = field(default_factory=list)


@dataclass
class RLContext:
    """Step feedback and metadata required by Gymnasium environments."""

    # Current step reward
    reward: float = 0.0

    # Episode ended due to crash/goal state
    terminated: bool = False 

    # Episode ended due to time limit 
    truncated: bool = False  

    # Steps elapsed in current episode
    step_count: int = 0 


@dataclass
class StateBoard:
    dynamics_state: DynamicsState = field(default_factory=DynamicsState)
    sensor_data: SensorData = field(default_factory=SensorData)
    control_targets: ControlTargets = field(default_factory=ControlTargets)
    actuator_state: ActuatorState = field(default_factory=ActuatorState)
    fault_status: FaultStatus = field(default_factory=FaultStatus)
    rl_context: RLContext = field(default_factory=RLContext)
