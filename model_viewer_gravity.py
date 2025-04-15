import mujoco
import mujoco.viewer
import os
import time
import numpy as np
import matplotlib.pyplot as plt # Import plotting library
import jax
import jax.numpy as jp # Add JAX imports

# --- Configuration ---
LOGGING_INTERVAL_SECONDS = 0.05 # Log data more frequently for better plot resolution
MAX_SIM_TIME = 10.0 # Run simulation for a maximum of 10 seconds
# Focus on the accelerometer
TARGET_SENSOR_SUBSTRING = "accelerometer"
# --------------------

# --- Data Logging Setup ---
time_log = []
accel_log = []
qpos2_log = [] # Add list to log qpos[2]
orientation_log = [] # Add list for orientation reward
# -------------------------

# --- Get XML Path ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "g1_description", "scene_mjx_alt.xml")
    print(f"Attempting to load XML from: {xml_path}")
except NameError:
    xml_path = "g1_description/scene_mjx_alt.xml"
    print(f"Warning: __file__ not defined. Assuming XML path: {xml_path}")

if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found at: {xml_path}")

# --- Load Model ---
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from {xml_path}: {e}")
    exit(1)

# --- Get IMU site ID (using correct name) ---
imu_site_name = "imu_in_pelvis" # Corrected name
imu_site_id = -1
try:
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site_name)
    if imu_site_id == -1:
        print(f"Warning: Site '{imu_site_name}' not found in the model. Orientation reward cannot be calculated.")
except ValueError:
    print(f"Error finding site '{imu_site_name}'. Orientation reward cannot be calculated.")
# -----------------------------------------------

# --- Compute initial state sensors and print values ---
try:
    # Compute forward kinematics and sensors for the initial state
    mujoco.mj_forward(model, data)
    print("Initial mj_forward() computed.")

    # Get initial qpos[2] (often root Z height)
    initial_qpos2 = data.qpos[2] if len(data.qpos) > 2 else "N/A (qpos too short)"

    # Find and get initial accelerometer Z value
    initial_accel_z = "N/A"
    accel_sensor_name = "N/A"
    if model.nsensor > 0:
        for i in range(model.nsensor):
            sensor_name_ptr = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_name = sensor_name_ptr if sensor_name_ptr else f"Unnamed Sensor {i}"

            # Check if this sensor matches the target substring
            is_target_sensor = (
                TARGET_SENSOR_SUBSTRING is not None and
                TARGET_SENSOR_SUBSTRING != "" and
                TARGET_SENSOR_SUBSTRING.lower() in sensor_name.lower()
            )

            if is_target_sensor:
                accel_sensor_name = sensor_name # Store the name found
                sensor_adr = model.sensor_adr[i]
                sensor_dim = model.sensor_dim[i]
                if sensor_dim == 3:
                    sensor_values = data.sensordata[sensor_adr : sensor_adr + sensor_dim]
                    initial_accel_z = sensor_values[2] # Get Z component (index 2)
                else:
                    initial_accel_z = f"N/A (Sensor '{sensor_name}' dim is {sensor_dim}, expected 3)"
                break # Assume only one accelerometer sensor matching the name
        if initial_accel_z == "N/A" and accel_sensor_name == "N/A":
             initial_accel_z = f"N/A (Sensor containing '{TARGET_SENSOR_SUBSTRING}' not found)"

    else:
        initial_accel_z = "N/A (No sensors in model)"

    # Print the initial values
    print(f"\n--- Initial State Values ---")
    # Format differently depending on whether it's a number or N/A string
    if isinstance(initial_qpos2, (int, float)):
         print(f"Initial qpos[2] (Root Z?): {initial_qpos2:.4f}")
    else:
         print(f"Initial qpos[2] (Root Z?): {initial_qpos2}")

    if isinstance(initial_accel_z, (int, float)):
         print(f"Initial '{accel_sensor_name}' Z: {initial_accel_z:.4f}")
    else:
         print(f"Initial Accelerometer Z    : {initial_accel_z}")
    print("--------------------------\n")

except Exception as e:
    print(f"Error during initial state computation: {e}")
    # Decide if you want to exit or continue if initial values fail
    # exit(1)

# --- Launch the PASSIVE Viewer ---
viewer = None
try:
    print("Launching MuJoCo PASSIVE viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)
    print(f"Passive viewer launched. Simulating for up to {MAX_SIM_TIME} seconds...")
    print("Observe the robot falling. Data will be plotted after simulation ends.")

    last_log_time = time.time() # Use real time for logging interval trigger

    # Main simulation loop
    while viewer.is_running() and data.time < MAX_SIM_TIME:
        step_start = time.time()

        # Step the simulation
        try:
            mujoco.mj_step(model, data)
        except Exception as e:
            print(f"Error during mj_step: {e}", flush=True)
            break

        # --- Periodic Data Logging ---
        current_time = time.time()
        if current_time - last_log_time >= LOGGING_INTERVAL_SECONDS:
            # Log time and qpos[2]
            current_sim_time = data.time
            current_qpos2 = data.qpos[2] if len(data.qpos) > 2 else np.nan # Log NaN if qpos is too short

            # --- Calculate and log orientation reward ---
            current_orientation_reward = np.nan # Default to NaN
            if imu_site_id != -1:
                try:
                    # Get the site's rotation matrix (orientation)
                    site_xmat = data.site_xmat[imu_site_id].reshape(3, 3)
                    # Transform world Z-axis [0, 0, 1] into the site's local frame to get the 'up' vector
                    imu_up_vec = site_xmat @ jp.array([0.0, 0.0, 1.0])
                    world_up_vec = jp.array([0.0, 0.0, 1.0])
                    # Calculate reward (using the formula structure from balance.py)
                    cos_dist = jp.dot(imu_up_vec, world_up_vec)
                    normalized = 0.5 * cos_dist + 0.5
                    current_orientation_reward = jp.square(normalized)
                except Exception as e_reward:
                    print(f"Error calculating orientation reward at time {current_sim_time:.2f}: {e_reward}")
            # -------------------------------------------

            # Find and log the accelerometer data
            accel_value_found = False
            if model.nsensor > 0:
                for i in range(model.nsensor):
                    sensor_name_ptr = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                    sensor_name = sensor_name_ptr if sensor_name_ptr else f"Unnamed Sensor {i}"

                    # Filter for the target sensor
                    should_log_sensor = (
                        TARGET_SENSOR_SUBSTRING is not None and
                        TARGET_SENSOR_SUBSTRING != "" and
                        TARGET_SENSOR_SUBSTRING.lower() in sensor_name.lower()
                    )

                    if should_log_sensor:
                        sensor_adr = model.sensor_adr[i]
                        sensor_dim = model.sensor_dim[i]
                        # Ensure dimension is 3 for accelerometer
                        if sensor_dim == 3:
                            sensor_values = data.sensordata[sensor_adr : sensor_adr + sensor_dim]
                            # Append all data together for this time step
                            time_log.append(current_sim_time)
                            qpos2_log.append(current_qpos2)
                            accel_log.append(sensor_values.copy())
                            orientation_log.append(float(current_orientation_reward)) # Log orientation reward
                            accel_value_found = True
                        break # Assume only one accelerometer sensor matching the name

            # If accelerometer wasn't found/logged, maybe still log time/qpos? Or skip?
            # Current logic only logs if accel is found. Adjust if needed.

            last_log_time = current_time
        # ------------------------------------

        # Sync the viewer
        try:
             viewer.sync()
        except Exception as e:
            print(f"Error during viewer.sync(): {e}", flush=True)
            if not viewer.is_running():
                 print("Viewer closed during sync.", flush=True)
                 break

except Exception as e:
    print(f"An error occurred: {e}", flush=True)

finally:
    # Clean up viewer
    if viewer and viewer.is_running():
        print("Simulation time limit reached or viewer closed. Closing passive viewer.", flush=True)
        viewer.close()
    elif viewer:
        print("Passive viewer was already closed.", flush=True)
    else:
        print("Viewer object was not created.", flush=True)

# --- Plotting the Data ---
print("Preparing plot...", flush=True)
# Check if all necessary data was logged
if time_log and accel_log and qpos2_log and orientation_log:
    accel_data = np.array(accel_log) # Convert list of arrays to a 2D NumPy array
    qpos2_data = np.array(qpos2_log) # Convert qpos2 log to NumPy array
    orientation_data = np.array(orientation_log) # Convert orientation log to NumPy array

    # Ensure data shapes are valid
    if accel_data.shape[0] == len(time_log) and \
       qpos2_data.shape[0] == len(time_log) and \
       orientation_data.shape[0] == len(time_log) and \
       accel_data.shape[1] == 3:

        # Create subplots: 3 rows, 1 column, sharing the X axis
        fig, ax = plt.subplots(3, 1, figsize=(12, 15), sharex=True) # Changed to 3 rows, adjusted figsize

        # --- Plot Accelerometer on first subplot (ax[0]) ---
        ax[0].plot(time_log, accel_data[:, 0], label='Accel X')
        ax[0].plot(time_log, accel_data[:, 1], label='Accel Y')
        ax[0].plot(time_log, accel_data[:, 2], label='Accel Z')
        ax[0].set_ylabel("Acceleration (m/s^2)") # Adjust unit if needed
        ax[0].set_title("Accelerometer Data During Simulation")
        ax[0].legend()
        ax[0].grid(True)

        # --- Plot qpos[2] on second subplot (ax[1]) ---
        ax[1].plot(time_log, qpos2_data, label='qpos[2] (Root Z?)', color='purple') # Added color
        ax[1].set_ylabel("Position (m)") # Adjust unit if needed
        ax[1].set_title("Root Height (qpos[2]) During Simulation")
        ax[1].legend()
        ax[1].grid(True)

        # --- Plot Orientation Reward on third subplot (ax[2]) ---
        ax[2].plot(time_log, orientation_data, label='Orientation Reward', color='green')
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Reward Value")
        ax[2].set_title("Orientation Reward During Simulation")
        ax[2].legend()
        ax[2].grid(True)
        ax[2].set_ylim([-0.1, 1.1]) # Set y-axis limits for reward (typically 0 to 1)

        # --- Final Plot Adjustments ---
        plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
        print("Displaying plot. Close the plot window to exit the script.")
        plt.show() # Display the plot (blocks until closed)
    else:
        print(f"Error: Logged data shape mismatch or accelerometer data not 3D. Cannot plot.", flush=True)
        print(f"Time points: {len(time_log)}, Accel shape: {accel_data.shape}, Qpos2 points: {len(qpos2_log)}, Orient points: {len(orientation_log)}", flush=True)

else:
    # Updated print message
    print("No data logged for one or more variables (time, accel, qpos2, orientation). Skipping plot.", flush=True)

print("Script finished.", flush=True)
