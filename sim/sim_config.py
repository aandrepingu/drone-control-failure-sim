from dataclasses import dataclass, field
import numpy as np
import mujoco


@dataclass
class SimConfig:
    """
    Dataclass representing a starting configuration for the drone,
    consisting of starting position, velocity, orientation quaternion, and wind force.
    """

    position: np.ndarray  # shape (3,)
    velocity: np.ndarray  # shape (3,)
    quat: np.ndarray  # shape (4,) for orientation [w, x, y, z]
    wind_force: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )  # shape (3,) [fx, fy, fz]

    @classmethod
    def random(
        cls,
        pos_range=(-0.5, 0.5),
        vel_range=(0.5, 2.0),
        tilt_range=(-1, 1),
        yaw_range=(-1, 1),
        rng: np.random.Generator | None = None,
        max_xy_wind: float = 0.8,
        max_z_wind: float = 0.2,
        wind_force: np.ndarray | None = None,
    ):
        """
        Generates a random SimConfig object based on ranges for position, velocity,
        tilt, yaw, and wind force.

        :param pos_range: range for position [min, max]
        :param vel_range: range for velocity [min, max]
        :param tilt_range: range for tilt [min, max]
        :param yaw_range: range for yaw [min, max]
        :param rng: NumPy random Generator
        :param max_xy_wind: max horizontal wind force magnitude in Newtons (±xy)
        :param max_z_wind: max vertical wind force magnitude in Newtons (±z)
        :param wind_force: optional manual override for wind force [fx, fy, fz]
        """
        if rng is None:
            rng = np.random.default_rng()

        pos = rng.uniform(*pos_range, size=3)
        vel = rng.uniform(*vel_range, size=3)

        tilt_axis = rng.standard_normal(3)
        tilt_axis[2] = 0  # zero out yaw component so it is handled separately

        norm = np.linalg.norm(tilt_axis)
        if norm > 1e-6:
            tilt_axis /= norm
        else:
            tilt_axis = np.array([1.0, 0.0, 0.0])

        tilt_angle = rng.uniform(*tilt_range)
        yaw_angle = rng.uniform(*yaw_range)

        axis = np.array([0.0, 0.0, 1.0])

        q_tilt = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_tilt, tilt_axis, tilt_angle)

        q_yaw = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_yaw, axis, yaw_angle)

        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q_yaw, q_tilt)

        if wind_force is not None:
            wind = np.array(wind_force, dtype=float)
        else:
            fx = rng.uniform(-max_xy_wind, max_xy_wind)
            fy = rng.uniform(-max_xy_wind, max_xy_wind)
            fz = rng.uniform(-max_z_wind, max_z_wind)
            wind = np.array([fx, fy, fz], dtype=float)

        return cls(position=pos, velocity=vel, quat=quat, wind_force=wind)

    @classmethod
    def sample(
        cls,
        rng: np.random.Generator,
        options: dict | None = None,
        pos_range=(-0.5, 0.5),
        vel_range=(-1.0, 1.0),
        tilt_range=(-1, 1),
        yaw_range=(-1, 1),
        max_xy_wind: float = 0.8,
        max_z_wind: float = 0.2,
    ) -> "SimConfig":
        """
        Sample an episode SimConfig using rng and optional overrides from options dictionary.
        """
        options = options or {}
        pos_r = options.get("pos_range", pos_range)
        vel_r = options.get("vel_range", vel_range)
        tilt_r = options.get("tilt_range", tilt_range)
        yaw_r = options.get("yaw_range", yaw_range)
        max_xy = options.get("max_xy_wind", max_xy_wind)
        max_z = options.get("max_z_wind", max_z_wind)
        wind_f = options.get("wind_force", None)

        config = cls.random(
            pos_range=pos_r,
            vel_range=vel_r,
            tilt_range=tilt_r,
            yaw_range=yaw_r,
            rng=rng,
            max_xy_wind=max_xy,
            max_z_wind=max_z,
            wind_force=wind_f,
        )

        if "position" in options:
            config.position = np.array(options["position"], dtype=float)
        if "velocity" in options:
            config.velocity = np.array(options["velocity"], dtype=float)
        if "quat" in options:
            config.quat = np.array(options["quat"], dtype=float)

        return config
