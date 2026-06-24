import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
melbourne = pd.read_csv("melbourne/melb_data.csv")

# seperate the target from predictors
y = melbourne['Price']
X = melbourne.drop(['Price'], axis=1)
X.copy()

# divide data into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# drop columns with missing values (simplest approach)
cols_with_missing_value = [col for col in X.columns if X[col].isnull().any()]
X_train = X_train.drop(cols_with_missing_value, axis=1)
X_test = X_test.drop(cols_with_missing_value, axis=1)

# select columns with relatively low cardinality
cols_with_low_cardinality = [col for col in X_train.columns if X_train[col].nunique() < 10 and (X_train[col].dtype == "str" or X_train[col].dtype == "object")]

# select numberical columns
cols_with_numbers = [col for col in X_train.columns if (X_train[col].dtype == "int64" or X_train[col].dtype == "float64")]

# keep selected columns only
my_cols = cols_with_low_cardinality + cols_with_numbers
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()

print(X_train.head())