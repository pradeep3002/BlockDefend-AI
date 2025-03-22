import pandas as pd

df = pd.read_csv("data/processed_cicids.csv")
print(df[" Label"].value_counts())