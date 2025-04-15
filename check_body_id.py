import mujoco

# Path to your scene XML file
xml_path = 'g1_description/scene_mjx_alt.xml'

# Load the MuJoCo model
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"Error loading model from {xml_path}:")
    print(e)
    exit()

# Index to check
body_index_to_check = 15

# Check if the index is valid
if body_index_to_check < 0 or body_index_to_check >= model.nbody:
    print(f"Error: Body index {body_index_to_check} is out of range (0 to {model.nbody - 1}).")
else:
    # Get the name of the body at the specified index
    body_name = model.body(body_index_to_check).name
    print(f"Body at index {body_index_to_check} is named: '{body_name}'")
