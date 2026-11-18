import unittest
import numpy as np
from sim.controllers.tile_q_controller import TileCoder, PlanarTileQController
from gym_env.drone_env import DroneEnv, load_model


class TestTileCoding(unittest.TestCase):
    def setUp(self):
        self.tile_coder = TileCoder()

    def test_tile_coder_bounds_and_structure(self):
        """Verify get_tiles returns 16 valid index tuples and indices are within bounds."""
        state = np.zeros(6, dtype=np.float32)
        tiles = self.tile_coder.get_tiles(state)
        self.assertEqual(len(tiles), 16)

        bins_per_dim = (6, 6, 6, 6, 8, 8)
        for tile in tiles:
            self.assertEqual(len(tile), 7)
            t = tile[0]
            self.assertTrue(0 <= t < 16)
            for d in range(6):
                self.assertTrue(
                    0 <= tile[d + 1] < bins_per_dim[d],
                    f"Tile index {tile[d + 1]} out of bounds for dim {d} (max {bins_per_dim[d]})",
                )

    def test_tile_coder_generalization(self):
        """Verify nearby states share active tiles while distant states do not."""
        s0 = np.zeros(6, dtype=np.float32)
        s_near = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s_far = np.array([1.2, 1.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        tiles_s0 = self.tile_coder.get_tiles(s0)
        tiles_near = self.tile_coder.get_tiles(s_near)
        tiles_far = self.tile_coder.get_tiles(s_far)

        matching_near = sum(1 for t1, t2 in zip(tiles_s0, tiles_near) if t1 == t2)
        matching_far = sum(1 for t1, t2 in zip(tiles_s0, tiles_far) if t1 == t2)

        self.assertGreaterEqual(matching_near, 8)
        self.assertLess(matching_far, 4)

    def test_planar_tile_q_controller_actions(self):
        """Verify action table and control computation bounds."""
        controller = PlanarTileQController(mass=0.4, dt=0.02)
        self.assertEqual(len(controller.action_table), 25)
        self.assertEqual(controller.weights.shape, (16, 6, 6, 6, 6, 8, 8, 25))

        obs = np.zeros(12, dtype=np.float32)
        target_state = {"pos": np.zeros(3), "yaw": 0.0}
        action = controller.compute_control(obs, {}, target_state)
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= 0.0) and np.all(action <= 7.0))

    def test_weight_update(self):
        """Verify semi-gradient TD updates correctly adjust weights."""
        controller = PlanarTileQController(mass=0.4, dt=0.02, alpha=0.15, gamma=0.98)
        obs = np.zeros(12, dtype=np.float32)
        target_state = {"pos": np.zeros(3), "yaw": 0.0}

        controller.compute_control(obs, {}, target_state)
        initial_q = controller.get_q_values(np.zeros(6))[controller.last_action_idx]

        reward = 2.0
        td_error = controller.update(obs, reward, False, target_state)
        self.assertGreater(td_error, 0.0)

        updated_q = controller.get_q_values(np.zeros(6))[controller.last_action_idx]
        self.assertGreater(updated_q, initial_q)


if __name__ == "__main__":
    unittest.main()
