import pandas as pd

df = pd.read_csv("melbourne/melb_data.csv")

# select target
y = df["Price"]
