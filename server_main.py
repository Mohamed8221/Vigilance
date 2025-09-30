# server_main.py (FINAL CORRECTED & REFACTORED VERSION)
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
import logging
import argparse
import signal
import threading
import os
from datetime import datetime

# --- Environment & Imports ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model_aggregator import federated_averaging
from vulnerability_assessment import assess_vulnerabilities
from local_model_trainer import create_model

# --- Configuration ---
BROKER_ADDRESS = "localhost"
PORT = 1883
UPDATES_TOPIC = "iot/model/updates"
GLOBAL_MODEL_TOPIC = "iot/model/global"
VULNERABILITY_REPORT_PATH = "vulnerability_report.json"
GLOBAL_WEIGHTS_PATH = os.path.join("model_state", "global_weights.npy")
MIN_LOCAL_ACCURACY = 0.40
NORM_THRESHOLD_FACTOR = 2.0
ADAPTIVE_BASELINE_ALPHA = 0.1

# --- Global State ---
client_updates = []
global_weights = None
client = None
round_counter = 0  # The global variable
file_lock = threading.Lock()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SERVER] - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('server.log'), logging.StreamHandler()])


# --- Core Functions ---

def handle_shutdown(signum, frame):
    """Handles graceful server shutdown."""
    logging.info("Shutdown signal received. Disconnecting...")
    if client and client.is_connected():
        client.loop_stop()
        client.disconnect()
    logging.info("Server shutdown complete.")
    exit(0)


def on_message(client, userdata, msg):
    """Callback to handle incoming MQTT messages."""
    global client_updates
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        if "device_id" in payload and "weights" in payload:
            update = {
                "device_id": payload["device_id"],
                "weights": [np.array(w, dtype=np.float32) for w in payload["weights"]],
                "local_accuracy": payload.get("accuracy", 0.0),
                "sample_size": payload.get("sample_size", 0)
            }
            client_updates.append(update)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logging.warning("Received a malformed message. Ignoring.")


def evaluate_global_model(weights, x_test, y_test):
    """Evaluate the global model's accuracy on the test set."""
    if not weights or x_test is None:
        return 0.0
    try:
        model = create_model(x_test.shape[1])
        model.set_weights(weights)
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return accuracy
    except Exception:
        logging.error("Error during global model evaluation.", exc_info=True)
        return 0.0


def main():
    """Main function to initialize and run the server."""
    # --- THIS IS THE FIX ---
    # `round_counter` is added to the global statement.
    global client_updates, global_weights, round_counter, client

    parser = argparse.ArgumentParser(description="Vigilance Federated Learning Server")
    parser.add_argument("--threshold", type=int, default=3, help="Updates per round.")
    args = parser.parse_args()

    logging.info("=" * 70 + "\nVigilance Server Starting Up...")

    baseline_mean, baseline_std = None, None
    x_test, y_test = (np.load('global_test_X.npy'), np.load('global_test_y.npy')) if os.path.exists(
        'global_test_X.npy') else (None, None)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Federated_Server")
    client.on_message = on_message
    try:
        client.connect(BROKER_ADDRESS, PORT)
        client.subscribe(UPDATES_TOPIC)
        client.loop_start()
        logging.info(f"Connected to MQTT broker, listening on: {UPDATES_TOPIC}")
    except Exception as e:
        logging.error(f"MQTT connection failed: {e}. Please ensure the broker is running.")
        return

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while True:
        if len(client_updates) < args.threshold:
            time.sleep(5)
            continue

        round_counter += 1  # This will now work correctly
        logging.info(f"--- Starting Aggregation Round {round_counter} ---")
        updates_for_round = client_updates[:args.threshold]
        client_updates = client_updates[args.threshold:]

        # Filter and process updates...
        # (The rest of the logic remains the same)
        accepted_updates, device_reports = [], []
        update_norms = [np.linalg.norm([np.linalg.norm(w) for w in u['weights']]) for u in updates_for_round]
        norms_mean, norms_std = np.mean(update_norms), np.std(update_norms)

        for i, update in enumerate(updates_for_round):
            report = {"device_id": update['device_id'], "local_accuracy": update['local_accuracy'],
                      "update_norm": float(update_norms[i])}

            if update['local_accuracy'] < MIN_LOCAL_ACCURACY:
                report['status'] = f"rejected_low_accuracy ({update['local_accuracy']:.1%})"
            elif norms_std > 0 and update_norms[i] > (norms_mean + (NORM_THRESHOLD_FACTOR * norms_std)):
                report['status'] = f"rejected_high_norm ({update_norms[i]:.2f})"
            else:
                report['status'] = "accepted"
                accepted_updates.append(update)
            device_reports.append(report)

        if accepted_updates:
            global_weights = federated_averaging([u['weights'] for u in accepted_updates],
                                                 [u['sample_size'] for u in accepted_updates])
            global_accuracy = evaluate_global_model(global_weights, x_test, y_test)
            vulnerabilities = assess_vulnerabilities(global_weights, baseline_mean, baseline_std)

            current_mean, current_std = vulnerabilities.get('current_mean'), vulnerabilities.get('current_std')
            if baseline_mean is None:
                baseline_mean, baseline_std = current_mean, current_std
            elif not any(a['status'] == 'anomaly' for a in vulnerabilities.get('anomalies', [])):
                baseline_mean = (ADAPTIVE_BASELINE_ALPHA * current_mean) + (1 - ADAPTIVE_BASELINE_ALPHA) * baseline_mean
                baseline_std = (ADAPTIVE_BASELINE_ALPHA * current_std) + (1 - ADAPTIVE_BASELINE_ALPHA) * baseline_std

            report_data = {**vulnerabilities, "global_accuracy": global_accuracy, "device_reports": device_reports}
            with open(VULNERABILITY_REPORT_PATH, "w") as f:
                json.dump(report_data, f, indent=4)
            client.publish(GLOBAL_MODEL_TOPIC, json.dumps({"global_weights": [w.tolist() for w in global_weights]}))
            logging.info(f"Round {round_counter} complete. New model published.")
        else:
            logging.error("No updates accepted in this round.")


if __name__ == "__main__":
    main()