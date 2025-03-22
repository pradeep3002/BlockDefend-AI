from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# ✅ Load Model
model = load_model("models/cnn_cicids_pca_test.h5")
print("✅ Model Loaded!")

# ✅ Load Test Data
X_test = np.load("data/X_pca.npy")
y_test = np.load("data/y_pca.npy")

# ✅ Find a Malicious Sample
malicious_indices = np.where(y_test == 1)[0]  # Find indexes where label = 1
if len(malicious_indices) == 0:
    print("⚠️ No malicious samples found in test set!")
    exit()

sample_index = malicious_indices[0]  # Take the first malicious sample
malicious_sample = X_test[sample_index].reshape(1, -1)  # Reshape for model

# ✅ Predict on Malicious Sample
prediction = model.predict(malicious_sample)
probability = float(prediction[0][0])

# ✅ Display Result
print(f"🔍 Prediction Probability: {probability:.6f}")
if probability > 0.5:
    print("🚨 ALERT! Model detected MALICIOUS traffic!")
else:
    print("❌ Model classified this as SAFE, but it should be MALICIOUS!")
