import numpy as np

def get_state(model, data):
    """
    Returns a map representing the state of the model in 
    terms of position, quaternion, linear and angular velocity.
    
    :param model: Mujoco MjModel object
    :param data: Mujoco MjData object
    """
    return {
        "pos": data.qpos[:3].copy(),
        "quat": data.qpos[3:7].copy(),
        "lin_vel": data.qvel[:3].copy(),
        "ang_vel": data.qvel[3:6].copy(),
    }
