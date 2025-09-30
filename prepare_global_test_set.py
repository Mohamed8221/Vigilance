# prepare_global_test_set.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prepare_test_set.log'),
        logging.StreamHandler()
    ]
)

RAW_DATA_PATH = os.path.join(
    "data", "edge_iiot_set", "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv"
)
FEATURE_LIST_PATH = "feature_list.json"
TEST_X_PATH = "global_test_X.npy"
TEST_Y_PATH = "global_test_y.npy"


def validate_dataset(df):
    """Validate dataset structure and content."""
    required_columns = ['Attack_label']

    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Required column '{col}' not found in dataset.")
            logging.error(f"Available columns: {df.columns.tolist()[:20]}")
            return False

    logging.info(f"Dataset validation passed. Shape: {df.shape}")
    return True


def main():
    logging.info("=" * 70)
    logging.info("Preparing Compatible Global Test Set & Feature List")
    logging.info("=" * 70)

    # Step 1: Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"FATAL: Raw data file not found at: {RAW_DATA_PATH}")
        logging.error("")
        logging.error("Please ensure Edge-IIoTset dataset is downloaded and placed at:")
        logging.error(f"  {RAW_DATA_PATH}")
        logging.error("")
        logging.error("Download from: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset")
        return

    # Step 2: Load dataset
    try:
        logging.info(f"Loading raw data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        logging.info(f"Successfully loaded dataset - Shape: {df.shape}")
        logging.info(f"Columns: {len(df.columns)}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.error("Please check file permissions and CSV format.")
        return

    # Step 3: Validate dataset
    if not validate_dataset(df):
        return

    # Step 4: Extract target and features
    if 'Attack_label' not in df.columns:
        logging.error("FATAL: Target column 'Attack_label' not found.")
        return

    y = df['Attack_label']

    # Drop target and attack type columns
    columns_to_drop = ['Attack_label', 'Attack_type']
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_drops, inplace=True, errors='ignore')
    logging.info(f"Dropped {len(existing_drops)} label columns: {existing_drops}")

    # Step 5: Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[numeric_cols].copy()
    logging.info(f"Selected {len(numeric_cols)} numeric feature columns")

    # Step 6: Force all columns to numeric and handle errors
    logging.info("Converting all features to numeric type...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Fill NaN values resulting from coercion
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values after conversion. Filling with 0.")
        X.fillna(0, inplace=True)
    else:
        logging.info("No NaN values found after conversion.")

    # Step 7: Save feature list
    feature_list = X.columns.tolist()

    try:
        with open(FEATURE_LIST_PATH, 'w') as f:
            json.dump(feature_list, f, indent=2)
        logging.info(f"Master feature list saved to {FEATURE_LIST_PATH}")
        logging.info(f"Total features: {len(feature_list)}")
    except Exception as e:
        logging.error(f"Failed to save feature list: {e}")
        return

    # Step 8: Create train/test split
    logging.info("Creating train/test split (80/20)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        logging.info(f"Train set: {X_train.shape[0]} samples")
        logging.info(f"Test set: {X_test.shape[0]} samples")

        # Show class distribution
        train_dist = y_train.value_counts().to_dict()
        test_dist = y_test.value_counts().to_dict()
        logging.info(f"Train class distribution: {train_dist}")
        logging.info(f"Test class distribution: {test_dist}")

    except Exception as e:
        logging.error(f"Failed to create train/test split: {e}")
        return

    # Step 9: Scale test set
    logging.info("Scaling test set features...")
    try:
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        logging.info(f"Test set scaled successfully. Shape: {X_test_scaled.shape}")
    except Exception as e:
        logging.error(f"Failed to scale test set: {e}")
        return

    # Step 10: Save test set
    try:
        np.save(TEST_X_PATH, X_test_scaled)
        np.save(TEST_Y_PATH, y_test.values)
        logging.info(f"Test features saved to {TEST_X_PATH}")
        logging.info(f"Test labels saved to {TEST_Y_PATH}")
    except Exception as e:
        logging.error(f"Failed to save test set: {e}")
        return

    # Step 11: Validation
    logging.info("")
    logging.info("=" * 70)
    logging.info("VALIDATION CHECK")
    logging.info("=" * 70)

    validation_passed = True

    # Check feature list
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, 'r') as f:
            loaded_features = json.load(f)
            if len(loaded_features) == len(feature_list):
                logging.info(f"✓ Feature list verified: {len(loaded_features)} features")
            else:
                logging.error(f"✗ Feature list mismatch!")
                validation_passed = False
    else:
        logging.error(f"✗ Feature list file not found!")
        validation_passed = False

    # Check test set files
    if os.path.exists(TEST_X_PATH):
        loaded_X = np.load(TEST_X_PATH)
        if loaded_X.shape == X_test_scaled.shape:
            logging.info(f"✓ Test X verified: {loaded_X.shape}")
        else:
            logging.error(f"✗ Test X shape mismatch!")
            validation_passed = False
    else:
        logging.error(f"✗ Test X file not found!")
        validation_passed = False

    if os.path.exists(TEST_Y_PATH):
        loaded_y = np.load(TEST_Y_PATH)
        if loaded_y.shape == y_test.values.shape:
            logging.info(f"✓ Test y verified: {loaded_y.shape}")
        else:
            logging.error(f"✗ Test y shape mismatch!")
            validation_passed = False
    else:
        logging.error(f"✗ Test y file not found!")
        validation_passed = False

    # Final status
    logging.info("=" * 70)
    if validation_passed:
        logging.info("SUCCESS: Global test set preparation completed successfully!")
        logging.info("")
        logging.info("You can now run:")
        logging.info("  1. python server_main.py --threshold 2")
        logging.info("  2. python federated_learning_loop.py --id device_1")
        logging.info("  3. python data_stream_simulator.py --num-devices 2")
    else:
        logging.error("FAILED: Some validation checks failed. Please review errors above.")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()