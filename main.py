import pandas as pd

df = pd.read_csv("melbourne/melb_data.csv")

print(df.describe())
print(df.corr(numeric_only=True))
