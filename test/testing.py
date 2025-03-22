import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ✅ Load Trained Model
model = tf.keras.models.load_model("models/cnn_cicids_pca_test.h5")

# ✅ Load Scaler
scaler = StandardScaler()
scaler.mean_ = np.load("data/scaler_mean.npy")
scaler.scale_ = np.load("data/scaler_scale.npy")

# ✅ Example Malicious Data (Replace with Real Malicious Sample)
malicious_sample = np.array([
    80,  # Destination Port
    5000000,  # Flow Duration
    10,  # Total Fwd Packets
    12,  # Total Backward Packets
    8000,  # Total Length of Fwd Packets
    5000,  # Total Length of Bwd Packets
    3000,  # Fwd Packet Length Max
    0,  # Fwd Packet Length Min
    750,  # Fwd Packet Length Mean
    1000,  # Fwd Packet Length Std
    3000,  # Bwd Packet Length Max
    0,  # Bwd Packet Length Min
    600,  # Bwd Packet Length Mean
    1200,  # Bwd Packet Length Std
    1000000,  # Flow Bytes/s
    50,  # Flow Packets/s
    120000,  # Flow IAT Mean
    50000,  # Flow IAT Std
    300000,  # Flow IAT Max
    20000,  # Flow IAT Min
    90000,  # Fwd IAT Total
    60000,  # Fwd IAT Mean
    50000,  # Fwd IAT Std
    200000,  # Fwd IAT Max
    10000,  # Fwd IAT Min
    70000,  # Bwd IAT Total
    50000  # Bwd IAT Mean
]).reshape(1, -1)

# ✅ Scale Data
malicious_sample = scaler.transform(malicious_sample)

# ✅ Reshape for CNN Model
malicious_sample = malicious_sample.reshape(1, 27, 1)

# ✅ Run Prediction
prediction = model.predict(malicious_sample)
is_malicious = int(prediction[0][0] > 0.5)

print(f"🔍 Prediction Probability: {prediction[0][0]:.6f}")
print(f"🔹 Final Classification: {'MALICIOUS 🚨' if is_malicious else 'SAFE ✅'}")
