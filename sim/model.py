import mujoco
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

def load_model(xml_name="quadrotor.xml"):
    """
    Load model from xml file and extract the mujoco MjData object.
    
    :param xml_name: name of the xml model you want to load
    """
    xml_path = ASSETS / xml_name
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data
