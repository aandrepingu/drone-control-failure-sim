import os
from typing import List, Tuple

import numpy as np

from sim.controllers.base import BaseController
from sim.controllers.quadrotor_pid import QuadrotorPIDController


class TileCoder:
    """
    Multi-grid offset tilings feature extractor for 6D continuous planar quadrotor state:
    s_planar = [e_x, e_z, v_x, v_z, theta, q]
    """

    def __init__(
        self,
        low: Tuple[float, ...] = (-1.5, -1.5, -2.5, -2.5, -0.8, -4.0),
        high: Tuple[float, ...] = (1.5, 1.5, 2.5, 2.5, 0.8, 4.0),
        bins_per_dim: Tuple[int, ...] = (6, 6, 6, 6, 8, 8),
        num_tilings: int = 16,
    ):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.bins_per_dim = np.array(bins_per_dim, dtype=np.int32)
        self.num_tilings = num_tilings
        self.tile_widths = (self.high - self.low) / self.bins_per_dim

        # Asymmetric displacement multipliers across the 6 dimensions
        self.displacements = np.array([1, 3, 5, 7, 9, 11], dtype=np.int32)

        # Precompute tiling offsets: offset[t, d]
        self.offsets = np.zeros((num_tilings, len(self.low)), dtype=np.float32)
        for t in range(num_tilings):
            for d in range(len(self.low)):
                self.offsets[t, d] = (
                    ((t * self.displacements[d]) % num_tilings)
                    / float(num_tilings)
                    * self.tile_widths[d]
                )

        self.t_indices = np.arange(num_tilings, dtype=np.int32)

    def get_tiles(self, state_6d: np.ndarray) -> List[Tuple]:
        """
        Clips state_6d to [low, high], applies each tiling's offset,
        computes integer bin indices in [0, bins_per_dim[d] - 1],
        and returns the 16 index tuples (t, b0, b1, b2, b3, b4, b5).
        """
        s = np.clip(state_6d, self.low, self.high)
        coords = np.floor((s - self.low + self.offsets) / self.tile_widths).astype(
            np.int32
        )
        coords = np.clip(coords, 0, self.bins_per_dim - 1)
        return [
            (
                t,
                int(coords[t, 0]),
                int(coords[t, 1]),
                int(coords[t, 2]),
                int(coords[t, 3]),
                int(coords[t, 4]),
                int(coords[t, 5]),
            )
            for t in range(self.num_tilings)
        ]

    def get_tile_indices(self, state_6d: np.ndarray) -> Tuple:
        """
        Fast tensor indexing tuple for numpy: (t_indices, b0, b1, b2, b3, b4, b5).
        """
        s = np.clip(state_6d, self.low, self.high)
        coords = np.floor((s - self.low + self.offsets) / self.tile_widths).astype(
            np.int32
        )
        coords = np.clip(coords, 0, self.bins_per_dim - 1)
        return (
            self.t_indices,
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            coords[:, 3],
            coords[:, 4],
            coords[:, 5],
        )


class PlanarTileQController(BaseController):
    """
    Linear Semi-Gradient Q-Learning controller with Tile Coding for 6D planar quadrotor control.
    Uses hybrid controller architecture: internal QuadrotorPIDController stabilizes out-of-plane axes
    (y=0, roll=0, yaw=0) while the tile-coded Q-learning policy commands vertical thrust offset and pitch torque.
    """

    def __init__(
        self,
        mass: float,
        dt: float = 0.02,
        num_tilings: int = 16,
        alpha: float = 0.15,
        gamma: float = 0.98,
        epsilon: float = 0.2,
        training: bool = True,
    ):
        self.mass = mass
        self.dt = dt
        self.num_tilings = num_tilings
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

        # Out-of-plane PID stabilizer (roll, yaw, y-pos)
        self.helper_pid = QuadrotorPIDController(mass=mass, dt=dt)

        # 25-action discrete grid (5x5): collective thrust offset and differential pitch command
        delta_thrust_levels = np.array([-0.45, -0.15, 0.0, 0.15, 0.45], dtype=np.float32)
        pitch_cmd_levels = np.array([-0.25, -0.08, 0.0, 0.08, 0.25], dtype=np.float32)
        self.action_table = np.array(
            [[dt_val, pc_val] for dt_val in delta_thrust_levels for pc_val in pitch_cmd_levels],
            dtype=np.float32,
        )
        self.num_actions = len(self.action_table)

        # Feature extractor
        self.tile_coder = TileCoder(num_tilings=num_tilings)

        # Weights table: (num_tilings, b0, b1, b2, b3, b4, b5, num_actions)
        self.weights = np.zeros(
            (num_tilings, *self.tile_coder.bins_per_dim, self.num_actions),
            dtype=np.float32,
        )

        # Motor mixer matrix
        self.mixer_matrix = np.array(
            [
                [1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0],
            ],
            dtype=float,
        )

        # History for TD learning
        self.last_state = None
        self.last_action_idx = None

    def reset(self):
        """Reset internal PID controllers and step history."""
        self.helper_pid.reset()
        self.last_state = None
        self.last_action_idx = None

    def extract_planar_state(self, obs: np.ndarray, target_state: dict) -> np.ndarray:
        """
        Extracts 6D planar state [e_x, e_z, v_x, v_z, theta, q] from 12D observation vector.
        """
        pos = obs[0:3]
        vel = obs[3:6]
        euler = obs[6:9]
        gyro = obs[9:12]

        target_pos = target_state.get("pos", np.zeros(3))
        ex = pos[0] - target_pos[0]
        ez = pos[2] - target_pos[2]
        vx = vel[0]
        vz = vel[2]
        theta = euler[1]  # pitch angle
        q = gyro[1]       # pitch rate

        return np.array([ex, ez, vx, vz, theta, q], dtype=np.float32)

    def get_q_values(self, state_6d: np.ndarray) -> np.ndarray:
        """
        Computes Q(s, a) = sum_{t=1}^{num_tilings} w[t, tile_t(s), a] for all actions.
        """
        tile_idx = self.tile_coder.get_tile_indices(state_6d)
        return self.weights[tile_idx].sum(axis=0)

    def select_action_index(
        self, state_6d: np.ndarray, training: bool | None = None
    ) -> int:
        """
        Epsilon-greedy action selection.
        """
        if training is None:
            training = self.training

        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))

        q_values = self.get_q_values(state_6d)
        return int(np.argmax(q_values))

    def compute_control(
        self, obs: np.ndarray, info: dict, target_state: dict
    ) -> np.ndarray:
        """
        Compute hybrid control action: Q-learning policy drives collective thrust and pitch torque;
        QuadrotorPIDController stabilizes out-of-plane axes (y, roll, yaw).
        """
        if obs is None or len(obs) == 0:
            return np.zeros(4)

        state_6d = self.extract_planar_state(obs, target_state)
        action_idx = self.select_action_index(state_6d, training=self.training)

        self.last_state = state_6d
        self.last_action_idx = action_idx

        delta_thrust, pitch_cmd = self.action_table[action_idx]

        # Base hover thrust per motor: (m * g) / 4
        hover_thrust = (self.mass * 9.81) / 4.0
        total_thrust = hover_thrust + delta_thrust

        # Out-of-plane PID stabilization
        pos = obs[0:3]
        vel = obs[3:6]
        euler = obs[6:9]
        gyro = obs[9:12]

        target_pos = target_state.get("pos", np.zeros(3))
        target_yaw = target_state.get("yaw", 0.0)
        if target_yaw is None:
            target_yaw = 0.0

        err_x_world = target_pos[0] - pos[0]
        err_y_world = target_pos[1] - pos[1]
        yaw = euler[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Rotate lateral error into drone body frame
        err_y_body = -err_x_world * sin_yaw + err_y_world * cos_yaw
        vel_y_body = -vel[0] * sin_yaw + vel[1] * cos_yaw

        roll_des = -np.clip(
            self.helper_pid.y_pid.update(
                target=0.0, measurement=-err_y_body, derivative_override=vel_y_body
            ),
            -self.helper_pid.max_angle,
            self.helper_pid.max_angle,
        )
        roll_cmd = self.helper_pid.roll_pid.update(
            measurement=euler[0], target=roll_des, derivative_override=gyro[0]
        )
        yaw_cmd = self.helper_pid.yaw_pid.update(
            measurement=euler[2], target=target_yaw, derivative_override=gyro[2]
        )

        command_vector = np.array(
            [total_thrust, roll_cmd, pitch_cmd, yaw_cmd], dtype=float
        )
        motor_thrusts = self.mixer_matrix @ command_vector
        return np.clip(motor_thrusts, 0.0, 7.0)

    def update(
        self,
        next_obs: np.ndarray,
        reward: float,
        terminated: bool,
        target_state: dict,
    ) -> float:
        """
        Semi-gradient Q-learning weight update:
        delta = R + gamma * (1 - terminated) * max_{a'} Q(S', a') - Q(S, A)
        w[tile_t(S), A] += (alpha / num_tilings) * delta
        """
        if self.last_state is None or self.last_action_idx is None:
            return 0.0

        last_tile_idx = self.tile_coder.get_tile_indices(self.last_state)
        q_current = float(
            self.weights[last_tile_idx + (self.last_action_idx,)].sum()
        )

        if terminated:
            td_target = reward
        else:
            next_state_6d = self.extract_planar_state(next_obs, target_state)
            next_tile_idx = self.tile_coder.get_tile_indices(next_state_6d)
            q_next = self.weights[next_tile_idx].sum(axis=0)
            td_target = reward + self.gamma * float(np.max(q_next))

        td_error = td_target - q_current
        step_size = (self.alpha / float(self.num_tilings)) * td_error

        self.weights[last_tile_idx + (self.last_action_idx,)] += step_size
        return float(abs(td_error))

    def save(self, filepath: str):
        """Save weights table to compressed .npz archive."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(
            filepath,
            weights=self.weights,
            action_table=self.action_table,
        )

    def load(self, filepath: str):
        """Load weights table from .npz archive."""
        data = np.load(filepath)
        self.weights = data["weights"]
        if "action_table" in data:
            self.action_table = data["action_table"]
            self.num_actions = len(self.action_table)
