import pandas as pd
from sklearn.model_selection import train_test_split

melbourne = pd.read_csv("melbourne/melb_data.csv")

# select target
y = melbourne['Price']

# find features
X = melbourne.drop('Price', axis=1) # drop price (target)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, train_size=0.8)


cols_with_missing_value = [col for col in X.columns if X[col].isnull().any()] # find list of columns w/ a missisng value
X = X.drop(cols_with_missing_value, axis=1) # drop columns w/ a missing value
X = X.select_dtypes(exclude=["object", "str"]) # for simplicity only take numberical
