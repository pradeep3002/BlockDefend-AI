import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

start_time = time.time()

# ✅ Load preprocessed dataset
X = np.load("data/X_data.npy")
y = np.load("data/y_labels.npy")

# ✅ Reshape from (samples, features, 1) → (samples, features)
X = X.reshape(X.shape[0], X.shape[1])

# ✅ Convert to DataFrame for easier analysis
df = pd.DataFrame(X)
print(f"✅ Loaded dataset with shape: {df.shape}")

# ✅ Compute Mutual Information (MI) with Parallel Processing
print("🔍 Computing Mutual Information...")
mi_scores = mutual_info_classif(df, y, discrete_features=False, n_neighbors=5, random_state=42)  # More neighbors for stability
mi_scores_df = pd.DataFrame({"Feature": range(len(mi_scores)), "MI Score": mi_scores})
mi_scores_df = mi_scores_df.sort_values(by="MI Score", ascending=False)

print("✅ MI Computation Done!")

# ✅ Select Top 60 Features (instead of 30)
top_n = 60  
selected_features = mi_scores_df["Feature"].values[:top_n]
X_selected = df.iloc[:, selected_features]

# ✅ Standardize the selected features
print("🔄 Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ✅ Apply PCA (Retain 98% variance instead of fixed components)
print("🧠 Applying PCA...")
pca = PCA(n_components=0.98, svd_solver='auto', random_state=42)  
X_pca = pca.fit_transform(X_scaled)

# ✅ Save Explained Variance for Analysis
np.save("data/pca_explained_variance.npy", pca.explained_variance_ratio_)

# ✅ Plot Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")
plt.axhline(y=0.98, color='r', linestyle='--', label="98% Variance Retained")
plt.legend()
plt.grid()
plt.savefig("pca_variance.png")
plt.show()

# ✅ Save processed dataset
np.save("data/X_pca.npy", X_pca)
np.save("data/y_pca.npy", y)

end_time = time.time()
print(f"✅ Feature Selection & PCA Complete! Data saved as X_pca.npy & y_pca.npy.")
print(f"📊 Total Variance Retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")
print(f"⏳ Time Taken: {end_time - start_time:.2f} seconds")