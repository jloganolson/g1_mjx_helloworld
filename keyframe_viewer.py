# temp.py
import mujoco
import mujoco.viewer
import os
import time # Import time for pausing

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the XML file
xml_path = os.path.join(script_dir, "g1_description", "scene_mjx_alt.xml")

# Check if the XML file exists
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found at: {xml_path}")

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Find the keyframe ID for 'default_pose'
key_name = 'home'
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)

if key_id == -1:
    print(f"Warning: Keyframe '{key_name}' not found in the model.")
else:
    print(f"Setting model to keyframe '{key_name}' (ID: {key_id})")
    # Set the simulation state to the keyframe
    data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = model.key_qvel[key_id] # Also set velocity if defined
    data.time = model.key_time[key_id]    # Set time if defined
    # Recompute derived quantities
    mujoco.mj_forward(model, data)

# Create and launch the viewer
# Use launch_passive for non-simulating viewing
viewer = mujoco.viewer.launch_passive(model, data)

print(f"Viewer launched with '{key_name}' pose.")
print("Close the viewer window to exit the script.")

# Keep the script running while the viewer is open
try:
    while viewer.is_running():
        time.sleep(0.1) # Pause briefly to avoid busy-waiting
except KeyboardInterrupt:
    print("Script interrupted.")
finally:
    if viewer and viewer.is_running():
        viewer.close()

print("Viewer closed.")