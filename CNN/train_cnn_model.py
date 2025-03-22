import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt

# ✅ GPU Optimization: Enable Memory Growth
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU Error: {e}")

# ✅ Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CNNModelTraining")

# ✅ Load the dataset
logger.info("📂 Loading dataset...")
df = pd.read_csv("network_data.csv")

# ✅ Use a subset of data for faster training (adjustable)
df = df.sample(n=100000, random_state=42)  # Change `n=100000` to use more data

# ✅ Separate features and labels
X = df.drop(columns=["is_malicious", "original_source_ip", "original_destination_ip"]).values  # Drop IPs
y = df["is_malicious"].values  # Labels

# ✅ Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ Reshape for CNN input (1D convolution over features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ✅ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build CNN Model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification (malicious vs. safe)
])

# ✅ Compile model with GPU acceleration
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Train the model with optimized batch size & GPU support
logger.info("🚀 Training CNN model...")
history = model.fit(
    X_train, y_train,
    epochs=10,  # ✅ Reduced for quick testing (increase if needed)
    batch_size=1024,  # ✅ Larger batch size for efficiency
    validation_data=(X_test, y_test)
)

# ✅ Plot Training & Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# ✅ Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

# ✅ Save the trained model
model.save("network_model.h5")
logger.info("✅ Model training complete. Saved as network_model.h5")