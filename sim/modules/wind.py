import numpy as np

from sim.core.sim_module import SimModule
from sim.core.state import DynamicsState, WindState


class WindModule(SimModule):
    """
    Module that generates environmental wind forces and publishes them
    to the shared WindState on the StateBoard.
    """

    def __init__(
        self,
        wind_state: WindState,
        dynamics_state: DynamicsState,
        max_xy_force: float = 0.8,  # Max horizontal wind force in Newtons
        max_z_force: float = 0.2,  # Max vertical updraft/downdraft in Newtons
        period: int = 2,
        offset: int = 0,
    ):
        super().__init__(period=period, offset=offset)
        self.wind_state = wind_state
        self.dynamics_state = dynamics_state

        self.max_xy_force = max_xy_force
        self.max_z_force = max_z_force

        self.seed = 0
        self.rng = np.random.RandomState(self.seed)
        self.episode_wind_force = np.zeros(3, dtype=float)

    def apply_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Sample a new constant wind vector for the episode
        fx = self.rng.uniform(-self.max_xy_force, self.max_xy_force)
        fy = self.rng.uniform(-self.max_xy_force, self.max_xy_force)
        fz = self.rng.uniform(-self.max_z_force, self.max_z_force)

        self.reset(wind_force=np.array([fx, fy, fz], dtype=float))

    def set_constant_wind(self, force_vector: np.ndarray):
        """Allows manual override via env.reset(options={'wind_force': [fx, fy, fz]})"""
        self.reset(wind_force=force_vector)

    def HandleDispatch(self, current_time: int) -> None:
        self.wind_state.force[:] = self.episode_wind_force

    def reset(self, wind_force: np.ndarray | None = None):
        """
        Resets module timing and applies the episode wind force.

        :param wind_force: 3D force vector [fx, fy, fz] in Newtons.
        """
        super().reset()
        if wind_force is not None:
            self.episode_wind_force[:] = np.asarray(wind_force, dtype=float)
            print(f"Wind vector: {self.episode_wind_force}")
        self.wind_state.force[:] = self.episode_wind_force
        self.wind_state.torque[:] = 0.0
