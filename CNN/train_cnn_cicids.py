import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ✅ Suppress TF Logs (Speeds Up Training)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ Enable Mixed Precision for Faster Training on Apple M1/M2
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ✅ Check GPU Availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# ✅ Load Preprocessed Data
X = np.load("data/X_data.npy")
y = np.load("data/y_labels.npy")

# ✅ Split Data (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Reshape for Conv1D Input
X_train = X_train.reshape((-1, X_train.shape[1], 1))
X_val = X_val.reshape((-1, X_val.shape[1], 1))

# ✅ Compute Class Weights (Handles Data Imbalance)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# ✅ Define the CNN Model
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(256, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(1, activation='sigmoid')  # Binary classification (0 = Normal, 1 = Attack)
])

# ✅ Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # ✅ Fixed Learning Rate (No Schedule)
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ Add Callbacks (Early Stopping & Reduce LR on Plateau)
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# ✅ Train Model with Faster Settings
history = model.fit(
    X_train, y_train,
    epochs=10,  # ✅ Reduced from 20 to 10
    batch_size=512,  # ✅ Increased from 64 to 512
    validation_data=(X_val, y_val),
    class_weight=class_weights,  # Handle imbalanced data
    callbacks=callbacks_list,
    steps_per_epoch=5000  # ✅ Limits training steps to speed up each epoch
)

# ✅ Save Model
model.save("models/cnn_cicids_trained.h5")

# ✅ Plot Training History
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy on CIC-IDS 2017")
plt.legend()
plt.savefig("accuracy_plot.png")

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss on CIC-IDS 2017")
plt.legend()
plt.savefig("loss_plot.png")
print("✅ Training Complete! Model saved as cnn_cicids_trained.h5")