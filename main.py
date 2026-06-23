import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# read csv as a dataframe
melbourne = pd.read_csv("melbourne/melb_data.csv")

# select target from dataframe
y = melbourne['Price']

# select features
X = melbourne.drop(['Price'], axis=1)     


# strategy 1: drop columns                                          
cols_with_missing_value = [col for col in X.columns if X[col].isnull().any()]       
reduced_X = X.drop(cols_with_missing_value, axis=1)                                         # drop columns with missing values
reduced_X = reduced_X.select_dtypes(exclude=["object", "str"])

# split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# score different methods used on dataset
def score_dataset(X_train, X_test, y_test, y_train):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    melbourne_model = model.fit(X_train, y_train)
    y_pred = melbourne_model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

print("drop columns: ", score_dataset(X_train, X_test, y_test, y_train))