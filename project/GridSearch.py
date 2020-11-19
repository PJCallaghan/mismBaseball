import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

df = pd.read_csv("../data/salaries-and-fan-graph-stats.csv")
df.drop(["Name", "Team"], axis=1, inplace=True)
df = pd.get_dummies(drop_first=True, data=df)
y = df["Salary"]
x = df.drop(["Salary"], axis=1)
model = SVR()
param_grid = {'kernel': ('linear', 'rbf', 'sigmoid'), 'C': [1, 5, 10], 'degree': [3, 8],
              'coef0': [0.01, 10, 0.5], 'gamma': [1, 0.1, 0.001]}
grid = GridSearchCV(model, param_grid, verbose=3, scoring="neg_mean_squared_error")
grid.fit(x, y)
