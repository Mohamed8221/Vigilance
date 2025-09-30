# federated_learning_loop.py
import paho.mqtt.client as mqtt
import numpy as np
import pandas as pd
import json
import time
import logging
import argparse
import os
from sklearn.preprocessing import StandardScaler
from train_local_model import train_local_model

# --- Configuration ---
BROKER_ADDRESS = "localhost"
PORT = 1883
UPDATES_TOPIC = "iot/model/updates"
GLOBAL_MODEL_TOPIC = "iot/model/global"
REGISTRATION_TOPIC = "iot/registration/hello"
TRAINING_THRESHOLD = 100

# --- Global State ---
global_weights = None
device_data_buffer = []


def setup_logging(device_id):
    """Configures logging to include the device ID."""
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - [DEVICE: {device_id}] - %(levelname)s - %(message)s')


def on_message(client, userdata, msg):
    """Callback to handle incoming data and global model updates."""
    global device_data_buffer, global_weights
    if msg.topic == userdata['data_topic']:
        try:
            device_data_buffer.append(json.loads(msg.payload.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logging.warning("Received a malformed data packet. Ignoring.")
    elif msg.topic == GLOBAL_MODEL_TOPIC:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            global_weights = [np.array(w, dtype=np.float32) for w in payload.get("global_weights", [])]
            logging.info("Successfully received and updated global model from server.")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logging.warning("Received a malformed global model update. Ignoring.")


def preprocess_data(data_buffer, feature_list):
    """Preprocesses a buffer of data to prepare it for training."""
    if not data_buffer:
        return None, None, 0

    df = pd.DataFrame(data_buffer)
    if 'Attack_label' not in df.columns:
        logging.error("'Attack_label' column is missing from the data buffer.")
        return None, None, 0

    y = pd.to_numeric(df['Attack_label'], errors='coerce').fillna(0).astype(int)
    X = df.reindex(columns=feature_list, fill_value=0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, len(X_scaled)


def publish_model_update(client, weights, device_id, accuracy, sample_size):
    """Publishes the local model update to the server."""
    payload = {"device_id": device_id, "weights": [w.tolist() for w in weights],
               "accuracy": accuracy, "sample_size": sample_size}
    client.publish(UPDATES_TOPIC, json.dumps(payload))
    logging.info(f"Published model update (Accuracy: {accuracy:.2%})")


def main():
    """Main function to initialize and run the client loop."""
    global device_data_buffer, global_weights

    parser = argparse.ArgumentParser(description="Vigilance Federated Learning Client")
    parser.add_argument("--id", type=str, required=True, help="Unique ID for this device")
    args = parser.parse_args()
    device_id = args.id
    setup_logging(device_id)

    DATA_TOPIC = f"iot/device/data/{device_id}"

    try:
        with open('feature_list.json', 'r') as f:
            master_feature_list = json.load(f)
    except FileNotFoundError:
        logging.error("FATAL: feature_list.json not found. Run prepare_global_test_set.py first.");
        return

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=device_id)
    client.user_data_set({"device_id": device_id, "data_topic": DATA_TOPIC})

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(DATA_TOPIC)
            client.subscribe(GLOBAL_MODEL_TOPIC)
            client.publish(REGISTRATION_TOPIC, json.dumps({"device_id": device_id}))
            logging.info("Successfully connected and subscribed.")
        else:
            logging.error(f"Failed to connect, return code {reason_code}")

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_ADDRESS, PORT)
        client.loop_start()
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}");
        return

    try:
        while True:
            if len(device_data_buffer) >= TRAINING_THRESHOLD:
                logging.info(f"Buffer full ({len(device_data_buffer)} samples). Starting training.")
                data_to_process = list(device_data_buffer)
                device_data_buffer = []

                x_data, y_data, sample_size = preprocess_data(data_to_process, master_feature_list)

                if x_data is not None and sample_size > 0:
                    weights, accuracy = train_local_model(x_data, y_data, global_weights=global_weights)
                    if weights is not None:
                        publish_model_update(client, weights, device_id, accuracy, sample_size)
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Device loop stopped by user.")
    finally:
        if client.is_connected():
            client.loop_stop()
            client.disconnect()
        logging.info("Device disconnected.")


if __name__ == "__main__":
    main()