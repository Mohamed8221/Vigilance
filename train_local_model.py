# train_local_model.py
import tensorflow as tf
import logging


def create_model(input_dim, initial_weights=None, dropout_rate=0.2):
    """Builds and compiles a simple neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if initial_weights:
        try:
            model.set_weights(initial_weights)
        except ValueError as e:
            logging.warning(f"Could not set weights, likely due to shape mismatch: {e}. Starting with fresh model.")
    return model


def train_local_model(x_data, y_data, global_weights=None, epochs=10, batch_size=32):
    """Trains a model on local data and returns weights and validation accuracy."""
    if x_data.shape[0] < 10:
        logging.warning("Not enough data to train with a validation split. Skipping.")
        return global_weights, 0.0

    try:
        model = create_model(x_data.shape[1], initial_weights=global_weights)

        logging.info("Starting local training...")
        history = model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

        # Safely get the final validation accuracy
        final_val_accuracy = history.history.get('val_accuracy', [0.0])[-1]
        logging.info(f"Local training finished. Validation Accuracy: {final_val_accuracy:.2%}")

        return model.get_weights(), final_val_accuracy
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}", exc_info=True)
        return global_weights, 0.0