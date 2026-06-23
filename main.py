import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

credit_card_data = pd.read_csv("AER_credit_card_data.csv")

# select target
y = credit_card_data['income']

# select features
X = credit_card_data.drop(['income'], axis=1)
X = credit_card_data.select_dtypes(exclude=["str", "object"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)

# impute missing values
imp_mean = SimpleImputer()
imputed_X_train = pd.DataFrame(imp_mean.fit_transform(X_train, y_train))
imputed_X_test = pd.DataFrame(imp_mean.transform(X_test))
imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns

model = RandomForestRegressor(10, random_state=0)
credit_card_data_model = model.fit(imputed_X_train, y_train)
y_predicted = model.predict(imputed_X_test)
credit_card_mae = mean_absolute_error(y_test, y_predicted)
print(credit_card_mae*10000)
