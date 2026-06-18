import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

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
model = LinearRegression()
model.fit(X, y)

print("estimated: ", model.predict(X.head()))
print("actual: ", y.head())

print("error: ", model.predict(X.head()) - y.head())