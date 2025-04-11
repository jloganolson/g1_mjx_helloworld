# temp.py
import mujoco
import mujoco.viewer
import os

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the XML file
xml_path = os.path.join(script_dir, "g1_description", "g1_mjx_alt.xml")

# Check if the XML file exists
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found at: {xml_path}")

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Create and run the viewer
viewer = mujoco.viewer.launch(model, data)
viewer.render() # Initial render

print(f"Viewer launched. Look for the site 'imu_in_torso' (a small sphere).")
print("Close the viewer window to exit the script.")

# Keep the viewer running until closed
while viewer.is_running():
    step_start = data.time
    while data.time - step_start < 1/60.0: # Simulate at roughly 60 Hz
        mujoco.mj_step(model, data)
    viewer.render()

viewer.close()