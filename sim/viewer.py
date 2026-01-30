import mujoco.viewer

def launch_viewer(model, data):
    return mujoco.viewer.launch_passive(model, data)
