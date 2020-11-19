import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# %% Part 1
df = pd.read_csv("../data/salaries-and-fan-graph-stats.csv")
df.drop(["Name", "Team"], axis=1, inplace=True)
df = pd.get_dummies(drop_first=True, data=df)
y = df["Salary"]
x = df.drop(["Salary"], axis=1)
bestFeatures = SelectKBest(score_func=f_regression, k=6)
best_x = bestFeatures.fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(best_x, y, test_size=0.3, random_state=1)

# %% Plot?
# plt.scatter(df["Years"], df["Hits"])
# plt.xlabel("Years")
# plt.ylabel("Hits")
# plt.show()

# %% Decision tree to predict salary
dtr = DecisionTreeRegressor(max_depth=3)
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)

# %% How did it do?
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# %% Random forest instead
rfr = RandomForestRegressor(n_estimators=500, random_state=1)
rfr.fit(x_train, y_train)
y_pred_forest = rfr.predict(x_test)
rfr_mse = mean_squared_error(y_test, y_pred_forest)
rfr_rmse = rfr_mse ** 0.5

# %% Linear Model
model = LinearRegression()
model.fit(x_train, y_train)
linear_y_pred = model.predict(x_test)
l_mse = mean_squared_error(y_test, linear_y_pred)
l_rmse = l_mse ** 0.5

# %% Suport Vector
# {'C': 10, 'coef0': 0.01, 'degree': 3, 'gamma': 1, 'kernel': 'linear'}
sv = SVR(C=10, gamma=1, kernel="linear")
sv.fit(x_train, y_train)
sv_y_pred = sv.predict(x_test)
sv_f1 = mean_squared_error(y_test, sv_y_pred)


# %% Cross Validation
cv = 10
lr_score = cross_val_score(model, x, y, scoring="r2", cv=cv)
dt_score = cross_val_score(dtr, x, y, scoring="r2", cv=cv)
rfr_score = cross_val_score(rfr, x, y, scoring="r2", cv=cv)
sv_score = cross_val_score(sv, x, y, scoring="r2", cv=cv)

# %% Print the scores
print(f"Mean Squared Error:\nDTR: {rmse}\nRFR: {rfr_rmse}\nLinear Regression:{l_rmse}\nSVR: {sv_f1}")
print(
    f"Cross validation Scores: LR {lr_score.mean()}, DT: {dt_score.mean()}, RFR: {rfr_score.mean()}, SV: {sv_score.mean()}")
