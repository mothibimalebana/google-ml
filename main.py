import pandas as pd

df = pd.read_csv("melbourne/melb_data.csv")

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
print(X.head())
