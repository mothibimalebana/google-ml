import pandas as pd

df = pd.read_csv("home-data-for-ml-course/train.csv", index_col=0)

print(df.describe())
print(df.corr(numeric_only=True))
