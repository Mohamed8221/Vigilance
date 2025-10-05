# data_stream_simulator.py (Modified for specific simulation logic)
import paho.mqtt.client as mqtt
import pandas as pd
import time
import json
import os
import random
import logging
import argparse
import signal
import sys

# --- Configuration ---
BROKER_ADDRESS = "localhost"
PORT = 1883
BASE_DATA_PATH = os.path.join("data", "edge_iiot_set")
NORMAL_TRAFFIC_PATH = os.path.join(BASE_DATA_PATH, "normal_traffic")
ATTACK_TRAFFIC_PATH = os.path.join(BASE_DATA_PATH, "attack_traffic")
REGISTRATION_TOPIC = "iot/registration/hello"
DISCOVERY_TIMEOUT_SECONDS = 45
STREAM_SPEED_SECONDS = 0.05

# --- MODIFIED: Set normal phase duration to 3 minutes (180 seconds)
NORMAL_PHASE_DURATION_SECONDS = 180
ATTACK_PHASE_DURATION_SECONDS = 120  # Duration for the concurrent attack phase

SAMPLE_SIZE_PER_FILE = 2000  # Increased sample size for longer simulation

# --- Global State ---
active_devices = set()
client = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def handle_shutdown(signum, frame):
    """Handles SIGINT/SIGTERM for graceful shutdown."""
    logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
    if client:
        client.loop_stop()
        client.disconnect()
        logging.info("Disconnected from MQTT broker.")
    exit(0)


def on_registration(client, userdata, msg):
    """Callback for when a device registers itself."""
    global active_devices
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        device_id = payload.get("device_id")
        if device_id:
            logging.info(f"Discovery: Detected device '{device_id}'")
            active_devices.add(device_id)
    except Exception as e:
        logging.error(f"Error processing registration message: {e}")


def get_csv_files(path):
    """Recursively finds all CSV files."""
    if not os.path.exists(path):
        logging.error(f"Directory not found: {path}")
        return []
    csv_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    logging.info(f"Found {len(csv_files)} CSV files in {path}")
    return csv_files


def connect_mqtt():
    """Connects to the MQTT broker."""
    global client
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="data_stream_simulator")
        client.connect(BROKER_ADDRESS, PORT)
        logging.info("Successfully connected to MQTT broker.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        return None


def load_file_sample(file_path, sample_size=2000):
    """Load a small sample from a single file for speed."""
    try:
        df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
        # Use replace=True to handle small files gracefully
        if len(df) < sample_size:
            return df.sample(n=sample_size, random_state=42, replace=True).reset_index(drop=True)
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    except Exception as e:
        logging.warning(f"Could not load or sample file {file_path}: {e}")
        return pd.DataFrame()


def main():
    global client, active_devices

    parser = argparse.ArgumentParser(description="Federated Learning Data & Attack Simulator")
    parser.add_argument("--num-devices", type=int, required=True, help="Expected number of devices to discover.")
    args = parser.parse_args()
    NUM_DEVICES = args.num_devices

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    normal_files = get_csv_files(NORMAL_TRAFFIC_PATH)
    attack_files = get_csv_files(ATTACK_TRAFFIC_PATH)
    if not normal_files or not attack_files:
        logging.error("No valid CSV files found in data directories. Exiting.")
        return

    client = connect_mqtt()
    if not client: return

    # --- Phase 0: Device Discovery ---
    logging.info("--- Starting Phase 0: Device Discovery ---")
    client.subscribe(REGISTRATION_TOPIC)
    client.on_message = on_registration
    client.loop_start()

    start_discovery_time = time.time()
    logging.info(f"Waiting for {NUM_DEVICES} devices to register...")
    while len(active_devices) < NUM_DEVICES and (time.time() - start_discovery_time) < DISCOVERY_TIMEOUT_SECONDS:
        time.sleep(1)

    if len(active_devices) < NUM_DEVICES:
        logging.error(f"Discovery timeout! Only found {len(active_devices)}/{NUM_DEVICES} devices. Exiting.")
        client.loop_stop();
        return

    logging.info(f"--- Discovery Complete: Found all {len(active_devices)} devices ---")
    client.unsubscribe(REGISTRATION_TOPIC)
    device_ids = list(active_devices)

    # --- Data Assignment and Loading ---
    logging.info("Assigning and loading data samples for each device...")

    # --- NEW LOGIC START: Assign a unique random normal file to each device ---
    if NUM_DEVICES > len(normal_files):
        logging.error(
            f"Error: Number of devices ({NUM_DEVICES}) is greater than the number of available normal traffic files ({len(normal_files)}).")
        sys.exit(1)

    # Select a random, unique subset of normal files, one for each device
    selected_normal_files = random.sample(normal_files, NUM_DEVICES)
    device_normal_files = dict(zip(device_ids, selected_normal_files))
    # --- NEW LOGIC END ---

    device_data_samples = {}
    for device_id, file_path in device_normal_files.items():
        logging.info(f"Device '{device_id}' assigned normal file: {os.path.basename(file_path)}")
        df = load_file_sample(file_path, SAMPLE_SIZE_PER_FILE)
        if not df.empty:
            device_data_samples[device_id] = df

    # --- Victim and Attack File Selection (Logic remains the same) ---
    victim_id = random.choice(device_ids)
    attack_file = random.choice(attack_files)
    logging.warning(f"Selected VICTIM for targeted attack: '{victim_id}'")
    logging.info(f"  -> Attack file: {os.path.basename(attack_file)}")
    attack_df_sample = load_file_sample(attack_file, SAMPLE_SIZE_PER_FILE)

    if not device_data_samples or attack_df_sample.empty:
        logging.error("Could not load sufficient data samples. Exiting.");
        return

    logging.info("Data loading complete. Starting simulation...")

    try:
        device_row_indices = {device_id: 0 for device_id in device_ids}
        attack_row_index = 0

        # --- Phase 1: Normal Operation (3 minutes) ---
        logging.info(f"--- Starting Normal Operation Phase ({NORMAL_PHASE_DURATION_SECONDS} seconds) ---")
        phase_start_time = time.time()
        while time.time() - phase_start_time < NORMAL_PHASE_DURATION_SECONDS:
            for device_id in device_ids:
                data_topic = f"iot/device/data/{device_id}"
                device_df = device_data_samples.get(device_id)
                if device_df is None or device_df.empty: continue

                row_idx = device_row_indices[device_id]
                row = device_df.iloc[row_idx].to_dict()
                row['device_id'] = device_id
                client.publish(data_topic, json.dumps(row))

                # Cycle through the data for each device
                device_row_indices[device_id] = (row_idx + 1) % len(device_df)
            time.sleep(STREAM_SPEED_SECONDS)

        # --- Phase 2: Concurrent Attack ---
        logging.info(f"--- Starting Concurrent Attack Phase ({ATTACK_PHASE_DURATION_SECONDS} seconds) ---")
        phase_start_time = time.time()
        while time.time() - phase_start_time < ATTACK_PHASE_DURATION_SECONDS:
            # This loop correctly implements the concurrent attack logic
            for device_id in device_ids:
                data_topic = f"iot/device/data/{device_id}"

                # If it's the victim device, send attack data
                if device_id == victim_id:
                    row = attack_df_sample.iloc[attack_row_index].to_dict()
                    attack_row_index = (attack_row_index + 1) % len(attack_df_sample)
                # For all other devices, continue sending their normal data
                else:
                    device_df = device_data_samples.get(device_id)
                    if device_df is None or device_df.empty: continue
                    row_idx = device_row_indices[device_id]
                    row = device_df.iloc[row_idx].to_dict()
                    device_row_indices[device_id] = (row_idx + 1) % len(device_df)

                row['device_id'] = device_id
                client.publish(data_topic, json.dumps(row))

            time.sleep(STREAM_SPEED_SECONDS)

        logging.info("--- Simulation Complete ---")

    except KeyboardInterrupt:
        logging.info("Simulator stopped by user.")
    finally:
        if client and client.is_connected():
            client.loop_stop()
            client.disconnect()
            logging.info("Disconnected from MQTT broker.")


if __name__ == "__main__":
    main()