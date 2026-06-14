import numpy as np
import pandas as pd


# machine learning
import keras
import ml_edu.experiment
import ml_edu.results

# data visualization
import plotly.express as px

# load data
df = pd.read_csv("chicago_taxi_train.csv")

# update dataframe
training_df = df.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

# data exploration
print(training_df.describe(include='all'))

# correlation matrix
print(training_df.corr(numeric_only=True))