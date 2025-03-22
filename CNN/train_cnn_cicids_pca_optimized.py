import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ✅ Enable Mixed Precision for Faster Training on Apple M1/M2
set_global_policy('mixed_float16')

# ✅ Load Preprocessed Data
X = np.load("data/X_pca.npy")
y = np.load("data/y_pca.npy")

# ✅ Ensure Correct Label Format
y = y.astype("float32")  # Prevent precision issues

# ✅ Split Data (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Compute Class Weights (Handles Data Imbalance)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))

# ✅ Define the Optimized CNN Model
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(128, kernel_size=1, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(256, kernel_size=1, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(1, activation='sigmoid')
])

# ✅ Compile Model with Fixed Precision Loss & Optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# ✅ Add Callbacks for Efficient Training
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# ✅ Train Model with Optimized Parameters
history = model.fit(
    X_train, y_train,
    epochs=5,  # Reduced for testing, increase if needed
    batch_size=128,  # Optimized batch size
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks_list
)

# ✅ Save the Trained Model
model.save("models/cnn_cicids_pca_optimized.h5")

# ✅ Plot Training History
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Model Accuracy (PCA Features)")
plt.legend()
plt.savefig("accuracy_pca_optimized.png")

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("CNN Model Loss (PCA Features)")
plt.legend()
plt.savefig("loss_pca_optimized.png")

print("✅ Training Complete! Model saved as cnn_cicids_pca_optimized.h5")
print("✅ Training Results saved as accuracy_pca_optimized.png & loss_pca_optimized.png")