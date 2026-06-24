import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# read csv as dataframe
melbourne = pd.read_csv("melbourne/melb_data.csv")

# select target and predictors
y = melbourne['Price']
X = melbourne.drop(['Price'], axis=1)

# split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, train_size=0.8)

# seperate numerical columns from categorical columns
numerical_cols = [col for col in X_train.columns if X[col].dtype in ["float64", "int64"]]
categorical_cols = [col for col in X_train.columns if (X[col].dtype in ["str", "object"] and X[col].nunique() < 10)]
full_cols = numerical_cols + categorical_cols
X_train = X_train[full_cols]
X_test = X_test[full_cols]

# define preprocessing steps
numerical_transformer = SimpleImputer(strategy="constant")

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])
