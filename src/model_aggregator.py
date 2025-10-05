#model_aggregator.py
import numpy as np
import logging

def federated_averaging(weight_updates, sample_sizes=None):
    """
    Performs Federated Averaging on weight updates in a robust manner.
    Supports both simple and weighted averaging.

    Args:
        weight_updates (list of list): List of model weights from devices.
        sample_sizes (list of int, optional): Sizes of local data per device for weighted averaging.

    Returns:
        list: Aggregated global weights, or an empty list if inputs are invalid.
    """
    if not weight_updates:
        logging.warning("No weight updates provided for aggregation.")
        return []

    num_devices = len(weight_updates)
    if num_devices == 0:
        return []

    averaged_weights = []

    # Iterate through each layer (e.g., weights of layer 1, biases of layer 1, etc.)
    # zip(*weight_updates) groups the first layer from all devices, then the second, and so on.
    for layer_weights_tuple in zip(*weight_updates):
        try:
            stacked_weights = np.stack(layer_weights_tuple, axis=0)
        except ValueError as e:
            logging.error(f"Inconsistent shapes in a layer across devices. Cannot aggregate. Error: {e}")
            return weight_updates[0]

        if sample_sizes is None:
            averaged_layer = np.mean(stacked_weights, axis=0)
        else:
            total_size = sum(sample_sizes)
            if total_size == 0:
                logging.warning("Total sample size is 0. Falling back to simple averaging for this layer.")
                averaged_layer = np.mean(stacked_weights, axis=0)
            else:
                weights_for_avg = np.array(sample_sizes).reshape(-1, *([1] * (stacked_weights.ndim - 1)))
                averaged_layer = np.sum(stacked_weights * weights_for_avg, axis=0) / total_size

        averaged_weights.append(averaged_layer)

    if sample_sizes:
        logging.info(f"Weighted averaging completed for {num_devices} devices.")
    else:
        logging.info(f"Simple averaging completed for {num_devices} devices.")

    return averaged_weights

def flair_aggregation(weight_updates, sample_sizes=None):
    """
    FLAIR Aggregation: Federated Learning with Adaptive Reputation.
    Assigns reputation scores based on gradient behavior and weights contributions accordingly.

    Args:
        weight_updates (list of list): List of model weights from devices.
        sample_sizes (list of int, optional): Sizes of local data per device for weighted averaging.

    Returns:
        list: Aggregated global weights.
    """
    if not weight_updates:
        logging.warning("No weight updates provided for aggregation.")
        return []

    num_devices = len(weight_updates)
    if num_devices == 0:
        return []

    # Calculate reputation scores based on gradient behavior (simplified)
    reputation_scores = np.ones(num_devices)  # Default reputation = 1
    norms = [np.linalg.norm([np.linalg.norm(w) for w in update]) for update in weight_updates]
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    for i, norm in enumerate(norms):
        if norm > mean_norm + 2 * std_norm:
            reputation_scores[i] = 0.5  # Lower reputation for outliers

    # Normalize reputation scores
    reputation_scores /= np.sum(reputation_scores)

    # Aggregate with weighted average based on reputation
    averaged_weights = []
    for layer_weights_tuple in zip(*weight_updates):
        try:
            stacked_weights = np.stack(layer_weights_tuple, axis=0)
        except ValueError as e:
            logging.error(f"Inconsistent shapes in a layer across devices. Cannot aggregate. Error: {e}")
            return weight_updates[0]

        weights_for_avg = np.array(reputation_scores).reshape(-1, *([1] * (stacked_weights.ndim - 1)))
        averaged_layer = np.sum(stacked_weights * weights_for_avg, axis=0)

        averaged_weights.append(averaged_layer)

    logging.info(f"FLAIR aggregation completed for {num_devices} devices.")
    return averaged_weights