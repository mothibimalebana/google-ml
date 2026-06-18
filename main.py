import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("melbourne/melb_data.csv")
df = df.dropna(axis=0)

# select target
y = df['Price']

# select features
features = [
    'Rooms',
    'Distance', 
    'Postcode', 
    'Bedroom2', 
    'Bathroom', 
    'Car', 
    'Landsize', 
    'BuildingArea', 
    'YearBuilt', 
    'Lattitude', 
    'Longtitude', 
    'Propertycount'
    ]

X = df[features]

# split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)
# define a model
model = DecisionTreeRegressor(random_state=0)
model.fit(train_X, train_y)

predicted_y = model.predict(val_X.head())
print("estimated: \n", predicted_y)
print("\nactual: \n", val_y.head())

print("\nmae: \n", mean_absolute_error(val_y.head(), predicted_y)),