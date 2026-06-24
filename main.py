import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

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

# (approach 1) get a list of categorical variables
categoricals_features = [col for col in X_train.columns if X_train[col].dtype == "str"]

# (approach 2) make a copy of original data
ordinal_X_train = X_train.copy()
ordinal_X_test = X_test.copy()
ordinal_encoder = OrdinalEncoder()
ordinal_X_train[categoricals_features] = ordinal_encoder.fit_transform(X_train[categoricals_features])
ordinal_X_test[categoricals_features] = ordinal_encoder.transform(X_test[categoricals_features]) 

# (approach 3) apply one-hot encoder to each column w/ categorical data
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categoricals_features]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[categoricals_features]))
OH_cols_train.index = X_train[categoricals_features].index
OH_cols_test.index = X_test[categoricals_features].index
print(OH_cols_train)



# define function to measure quality of each approach
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

drop_X_train = X_train.drop(categoricals_features, axis=1)
drop_X_test = X_test.drop(categoricals_features, axis=1)

# print("MAE from approach 1 (Drop categorical variables): ", score_dataset(drop_X_train, drop_X_test, y_train, y_test))
# print("Mae from approach 2 (Ordinal encoding): ", score_dataset(ordinal_X_train, ordinal_X_test, y_train, y_test))