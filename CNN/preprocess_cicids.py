import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# ✅ Initialize Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Path to CICIDS 2017 dataset
DATA_FOLDER = "data/cicids_2017"

# ✅ List all CSV files in the folder
file_paths = [os.path.join(DATA_FOLDER, file) for file in os.listdir(DATA_FOLDER) if file.endswith(".csv")]

# ✅ Load & Merge All CSV Files
df_list = []
for file in file_paths:
    logger.info(f"📂 Loading {file}...")
    df_list.append(pd.read_csv(file, encoding="ISO-8859-1", low_memory=False))

df = pd.concat(df_list, ignore_index=True)
logger.info(f"✅ Loaded {len(df):,} rows from CICIDS 2017 dataset.")

# ✅ Take only 100,000 rows for quick testing
df = df.sample(n=100000, random_state=42)  
logger.info(f"✅ Sampled {len(df):,} rows for testing.")

# ✅ Remove spaces from column names
df.columns = df.columns.str.strip()

# ✅ Fix special characters in attack labels
df["Label"] = df["Label"].str.replace("ï¿½", "-", regex=True)

# ✅ Drop Unnecessary Columns
columns_to_drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# ✅ Convert 'Label' into Binary Classification
attack_labels = ["DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator",
                 "SSH-Patator", "DoS slowloris", "DoS Slowhttptest", "Bot",
                 "Web Attack - Brute Force", "Web Attack - XSS",
                 "Infiltration", "Web Attack - Sql Injection", "Heartbleed"]

df["is_malicious"] = df["Label"].apply(lambda x: 1 if x in attack_labels else 0)  # ✅ Rename to `is_malicious`
df.drop(columns=["Label"], inplace=True)  # ✅ Remove old column

logger.info("✅ Attack labels converted to binary classification (1 = Attack, 0 = Normal).")

# ✅ Convert all non-numeric columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# ✅ Handle Missing & Infinite Values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ✅ Feature Selection using Mutual Information (MI)
logger.info("🔍 Computing Mutual Information...")
X = df.drop(columns=["is_malicious"]).values
y = df["is_malicious"].values
mi_scores = mutual_info_classif(X, y, discrete_features=False, n_neighbors=3, random_state=42)

# ✅ Select Top 27 Features (Ensure it Matches CNN Model)
top_n = 27
selected_features = np.argsort(mi_scores)[-top_n:]
X_selected = X[:, selected_features]
logger.info(f"✅ Selected top {len(selected_features)} features.")

# ✅ Standardization (Scaling)
logger.info("🔄 Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ✅ Save Scaler Mean & Scale for Future Use
np.save("data/scaler_mean.npy", scaler.mean_)
np.save("data/scaler_scale.npy", scaler.scale_)

# ✅ Apply PCA (Reduce Dimensions to Match Training)
logger.info("🧠 Applying PCA...")
pca = PCA(n_components=27, svd_solver='auto', random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ✅ Save Processed Data
df_processed = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, 28)])
df_processed["is_malicious"] = y  # ✅ Include `is_malicious`
df_processed.to_csv("data/processed_cicids.csv", index=False)

logger.info("✅ Data saved: processed_cicids.csv")

# ✅ Save PCA Explained Variance
np.save("data/pca_explained_variance.npy", pca.explained_variance_ratio_)
logger.info(f"📊 PCA Explained Variance: {sum(pca.explained_variance_ratio_) * 100:.2f}%")