import mujoco
import numpy as np
from sim.sim_config import SimConfig
from scipy.spatial.transform import Rotation
from sim.controllers.base import BaseController

class Simulator:
    """
    Class which holds the simulated model, data, and failures we want to simulate,
    as well as time for future visualization and analysis purposes
    """
    def __init__(self, model, data, sim_config, controller:BaseController=None, failures=None):
        """
        Initialize the Simulator object, capturing any failures
        if any are passed in. If not, a basic hover simulation with
        no failures is created by default.
        """
        self.model = model
        self.data = data
        self.failures = failures or []
        self.controller = controller 
        self.time = 0.0
        self.config = sim_config
        self.apply_config(self.config)
        self.prev_vel = np.zeros(3)

        self.target = {
            'pos' : sim_config.position,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0
        }

        # set our drone to hover initially; this should be parameterized out later
        # when we get a proper "scenario" setup design

    def apply_config(self, sim_config):
        """
        Apply a SimConfig to the current simulation.
        
        :param sim_config: sim config object
        """
        self.data.qpos[0:3] = sim_config.position
        self.data.qpos[3:7] = sim_config.quat
        self.data.qvel[0:3] = sim_config.velocity
        self.data.qvel[3:6] = 0.0
        mujoco.mj_forward(self.model,self.data)
        
    def get_state(self,dt):
        """
        Extract drone state from mujoco.
        """
        pos = self.data.qpos[0:3]

        quat = self.data.qpos[3:7]
        quat_xyzw = np.array([quat[1],quat[2],quat[3],quat[0]])
        r = Rotation.from_quat(quat_xyzw, scalar_first=False)
        euler = r.as_euler("xyz")

        vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]

        acc = ( vel - self.prev_vel ) / dt
        self.prev_vel = vel.copy()

        """
        15-dimensional vector of state, consisting of displacement, velocity,
        angular velocity, angular displacement, and acceleration, each of which
        is a 3d vector.
        """
        return {
            "pos": pos,
            "vel": vel,
            "euler": euler,
            "ang_vel": ang_vel,
            "accel" : acc
        }


    def reset(self):
        self.apply_config(self.config)


    def mix(self, thrust_d, roll_d, pitch_d, yaw_d):
        """
        Quadcopter motor mixer that converts desired thrust, pitch,roll,and yaw
        to individual motor commands. This is the main mechanism by which a controller
        will affect the quadcopter's rotors. 
        
        :param thrust_d: Desired thrust
        :param roll_d: Desired roll
        :param pitch_d: Desired pitch
        :param yaw_d: Desired yaw
        """
        motor1_pwm = thrust_d + roll_d - pitch_d + yaw_d
        motor2_pwm = thrust_d - roll_d - pitch_d - yaw_d
        motor3_pwm =  thrust_d - roll_d + pitch_d + yaw_d
        motor4_pwm = thrust_d + roll_d + pitch_d - yaw_d

        res= np.array([motor1_pwm,motor2_pwm,motor3_pwm,motor4_pwm])

        # clip motor forces to be in the range [0,7.0]. This can be moved outside the mix function if needed
        # Note that this is hardcoded to match the ctrlrange of our model's actuators.
        # This should be changed if a different model is used; TODO to parameterize the ctrlrange somehow
        return np.clip(res, 0, 7.0)
       


    def step(self, dt=0.002):
        """
        Apply failures and step the simulation ahead.
        
        :param dt: time step length
        """
        state = self.get_state(dt)
        
        if self.controller is not None:
            thrust, roll, pitch, yaw = self.controller.compute_control(
                state,
                self.target
            )

            motor_cmds = self.mix(thrust, roll, pitch, yaw)

            self.data.ctrl[:] = motor_cmds

        for f in self.failures:
            f.apply(self, self.time)
        mujoco.mj_step(self.model, self.data)
        self.time += dt
