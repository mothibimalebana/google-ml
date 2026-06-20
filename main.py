import pandas as pd

df_part_1 = pd.read_csv("winemag/winemag-data_first150k.csv")
df_part_2 = pd.read_csv("winemag/winemag-data-130k-v2.csv")

df = pd.concat([df_part_1, df_part_2])

print(df)