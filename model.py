import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("dataset.csv")

X = data[['feature']]
y = data['target']

X_train, X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state - 42)

model = DecisionTreeRegressor()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("R2 Score:", r2_score(y_test,predictions))
