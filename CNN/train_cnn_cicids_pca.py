import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ✅ Enable GPU Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ✅ Load Preprocessed PCA Data
X = np.load("data/X_pca.npy")
y = np.load("data/y_pca.npy")

# ✅ Split Data (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Compute Class Weights for Imbalance Handling
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# ✅ Define the Improved CNN Model
model = keras.Sequential([
    layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(256, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.4),

    layers.Conv1D(512, kernel_size=3, activation='relu'),  # Added deeper layer
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# ✅ Compile Model with Learning Rate Decay
initial_lr = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=1000, decay_rate=0.9, staircase=True
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks (Early Stopping + Reduce LR on Plateau)
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# ✅ Train the Model
history = model.fit(
    X_train, y_train,
    epochs=15,  # 🔥 Lowered for faster testing
    batch_size=128,  # 🔥 Increased for efficiency
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks_list
)

# ✅ Save the Model
model.save("models/cnn_cicids_pca_v2.h5")

# ✅ Plot Training Results
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Improved Model Accuracy on CIC-IDS 2017")
plt.legend()
plt.savefig("accuracy_plot_v2.png")

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Improved Model Loss on CIC-IDS 2017")
plt.legend()
plt.savefig("loss_plot_v2.png")

print("✅ Model Training Complete! Saved as cnn_cicids_pca_v2.h5")