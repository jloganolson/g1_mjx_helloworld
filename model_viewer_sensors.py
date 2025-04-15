import mujoco
import mujoco.viewer
import os
import time # Import time for periodic printing

# --- Configuration ---
PRINT_INTERVAL_SECONDS = 0.5 # How often to print sensor data (in seconds)
# Set to None or "" to print all sensors, otherwise prints sensors containing this text (case-insensitive)
# You can set this back to "accel" if you only want the accelerometer
TARGET_SENSOR_SUBSTRING = None # "accel"
# --------------------

# Get the absolute path to the directory containing this script
# NOTE: If running interactively (like in a notebook or IDE),
# __file__ might not be defined. You might need to hardcode the path
# or use a different method to find the XML file.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the XML file
    # Make sure 'g1_description/scene_mjx_alt.xml' exists relative to this script
    xml_path = os.path.join(script_dir, "g1_description", "scene_mjx_alt.xml")
    print(f"Attempting to load XML from: {xml_path}") # Debug print for path
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environments)
    # Adjust this path as needed!
    xml_path = "g1_description/scene_mjx_alt.xml"
    print(f"Warning: __file__ not defined. Assuming XML path relative to current dir: {xml_path}")


# Check if the XML file exists
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found at: {xml_path}")

# Load the model
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("Model loaded successfully.") # Debug print
except Exception as e:
    print(f"Error loading model from {xml_path}: {e}")
    exit(1)

# --- DIAGNOSTIC: Print sensor info before starting ---
# (Keeping this block for confirmation, but it shouldn't change)
print(f"\n--- Sensor Initialization Check ---")
print(f"Number of sensors found in model: {model.nsensor}")
if model.nsensor > 0:
    print("Available sensors:")
    for i in range(model.nsensor):
        sensor_name_ptr = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_name = sensor_name_ptr if sensor_name_ptr else f"Unnamed Sensor {i}"
        sensor_type = model.sensor_type[i]
        sensor_objtype = model.sensor_objtype[i]
        sensor_objid = model.sensor_objid[i]
        obj_name_ptr = mujoco.mj_id2name(model, sensor_objtype, sensor_objid)
        obj_name = obj_name_ptr if obj_name_ptr else "N/A"
        print(f"  - Index: {i}, Name: '{sensor_name}', Type: {mujoco.mjtSensor(sensor_type).name}, Attached Object: '{obj_name}' (Type: {mujoco.mjtObj(sensor_objtype).name}, ID: {sensor_objid})")
else:
    print("No sensors are defined in the loaded model.")
print("---------------------------------\n")
# --- End Diagnostic ---


# --- Launch the PASSIVE Viewer ---
viewer = None # Initialize viewer to None
try:
    print("Launching MuJoCo PASSIVE viewer...")
    # Use launch_passive to allow the Python loop to run
    viewer = mujoco.viewer.launch_passive(model, data)
    print("Passive viewer launched. Starting simulation loop...")
    # Keep the viewer running and print sensor data periodically
    last_print_time = time.time()

    while viewer.is_running():
        step_start = time.time() # Use real time for loop timing

        # Step the simulation
        try:
            mujoco.mj_step(model, data)
        except Exception as e:
            print(f"Error during mj_step: {e}", flush=True)
            break # Exit loop on step error

        # --- Periodic Sensor Data Printing ---
        current_time = time.time()
        if current_time - last_print_time >= PRINT_INTERVAL_SECONDS:
            # Added flush=True to all prints within this block
            print(f"\n--- Printing Sensor Data Check (Sim Time: {data.time:.3f}s) ---", flush=True)
            if model.nsensor > 0:
                print("  Sensor Values:", flush=True)
                sensor_found_in_loop = False
                for i in range(model.nsensor):
                    sensor_name_ptr = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                    sensor_name = sensor_name_ptr if sensor_name_ptr else f"Unnamed Sensor {i}"

                    should_print = (
                        TARGET_SENSOR_SUBSTRING is None or
                        TARGET_SENSOR_SUBSTRING == "" or
                        TARGET_SENSOR_SUBSTRING.lower() in sensor_name.lower()
                    )

                    if should_print:
                        sensor_found_in_loop = True
                        sensor_adr = model.sensor_adr[i]
                        sensor_dim = model.sensor_dim[i]
                        sensor_values = data.sensordata[sensor_adr : sensor_adr + sensor_dim]
                        values_str = ", ".join([f"{val:.4f}" for val in sensor_values])
                        print(f"    {sensor_name} (ID: {i}, Dim: {sensor_dim}): [{values_str}]", flush=True)

                if TARGET_SENSOR_SUBSTRING and not sensor_found_in_loop:
                     print(f"    (Info: No sensors found containing '{TARGET_SENSOR_SUBSTRING}' during this interval.)", flush=True)

            else:
                print("  (Info: No sensors defined in model to print values for.)", flush=True)
            print("----------------------------------------------------------", flush=True)
            last_print_time = current_time
        # ------------------------------------

        # Sync the viewer state with the simulation data
        # This is crucial when using launch_passive
        try:
             viewer.sync()
        except Exception as e:
            # Handle cases where the viewer might close unexpectedly
            print(f"Error during viewer.sync(): {e}", flush=True)
            if not viewer.is_running():
                 print("Viewer closed during sync.", flush=True)
                 break # Exit loop if viewer is no longer running

        # Optional: Regulate loop speed to simulation timestep
        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)


except Exception as e:
    print(f"An error occurred during simulation or viewer operation: {e}", flush=True)

finally:
    # Ensure the viewer is closed properly
    if viewer and viewer.is_running(): # Check if viewer was successfully created and is running
        print("Closing passive viewer.", flush=True)
        viewer.close()
    elif viewer:
        print("Passive viewer was already closed.", flush=True)
    else:
        print("Viewer object was not created.", flush=True)


print("Script finished.", flush=True)
