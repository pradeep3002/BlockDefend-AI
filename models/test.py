import pandas as pd

df = pd.read_csv("data/processed_cicids.csv")  # Ensure this file exists
print(df.columns)  # Check available columns