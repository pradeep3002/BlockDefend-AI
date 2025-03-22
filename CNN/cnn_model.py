import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pymongo import MongoClient
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data():
    # Load data from MongoDB
    client = MongoClient('localhost', 27017)
    db = client['network_analysis']
    collection = db['processed_data']
    
    # Get all documents
    cursor = collection.find()
    data = list(cursor)
    
    if not data:
        raise ValueError("No data found in MongoDB collection")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} records from MongoDB")
    
    # Verify is_malicious exists
    if 'is_malicious' not in df.columns:
        raise ValueError("The 'is_malicious' column is missing in the data")
    
    # Get feature columns (those starting with 'feature_')
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if not feature_cols:
        raise ValueError("No feature columns found in the data")
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    # Prepare features and labels
    X = df[feature_cols].values
    y = df['is_malicious'].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Check for any invalid values
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Dataset contains NaN or infinite values")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    return X, y

def build_model(input_shape):
    model = tf.keras.models.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Dense layers instead of Conv1D for this type of data
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    try:
        # Prepare the data
        X, y = prepare_data()
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build and compile the model
        model = build_model(X_train.shape[1])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
        # Add early stopping and model checkpoint
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate the model
        test_results = model.evaluate(X_test, y_test, verbose=1)
        metrics = dict(zip(model.metrics_names, test_results))
        
        logger.info("Test Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save the model
        model.save('intrusion_detection_model.keras')
        logger.info("Model saved successfully")
        
        # Make some predictions to verify
        predictions = model.predict(X_test[:5])
        logger.info("\nSample Predictions vs Actual:")
        for pred, actual in zip(predictions, y_test[:5]):
            logger.info(f"Predicted: {pred[0]:.4f}, Actual: {actual}")
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        raise

if __name__ == "__main__":
    main()