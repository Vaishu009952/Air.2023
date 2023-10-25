import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

v=pd.read_csv("./madrid_2018.csv")
print(v)
v.dropna(inplace=True)
v.drop_duplicates(inplace=True)
v=v[['NO_2','SO_2','PM10']]
print(v)
X = v[['NO_2', 'SO_2']]
y = v['PM10']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
