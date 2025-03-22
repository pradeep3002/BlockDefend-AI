from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ✅ Load Model
model_path = "models/cnn_cicids_pca_test.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)

# ✅ Load Test Data
data_x_path = "data/X_pca.npy"
data_y_path = "data/y_labels.npy"
if not os.path.exists(data_x_path) or not os.path.exists(data_y_path):
    raise FileNotFoundError("Dataset files not found. Check paths and filenames.")

X = np.load(data_x_path)
y = np.load(data_y_path)

# ✅ Ensure X and y have the same number of samples
if X.shape[0] != y.shape[0]:
    print(f"⚠️ Mismatched dataset sizes: X = {X.shape}, y = {y.shape}. Trimming y to match X.")
    y = y[:X.shape[0]]

# ✅ Split into Test Set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"📊 Model Test Accuracy: {accuracy * 100:.2f}%")
print(f"📊 Model Test Loss: {loss:.4f}")