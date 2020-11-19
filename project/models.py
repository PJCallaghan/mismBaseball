import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# %% Part 1
df = pd.read_csv("../data/salaries-and-fan-graph-stats.csv")
df.drop(["Name"], axis=1, inplace=True)
df = pd.get_dummies(drop_first=True, data=df)
y = df["Salary"]
x = df.drop(["Salary"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# %% Plot?
# plt.scatter(df["Years"], df["Hits"])
# plt.xlabel("Years")
# plt.ylabel("Hits")
# plt.show()

# %% Decision tree to predict salary
dtr = DecisionTreeRegressor()
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

# %% Logistic Regression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
lr_y_pred = lr.predict(x_test)
lr_f1 = f1_score(y_test, lr_y_pred, average="weighted")
print(f"Lgistic Regression f1 score is: {lr_f1}")

# %% Decision tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_y_pred = dt.predict(x_test)
dt_f1 = f1_score(y_test, dt_y_pred, average="weighted")
print(f"Decision Tree f1 score is: {dt_f1}")

# %% Random Forest
rfr = RandomForestClassifier(n_estimators=500)
rfr.fit(x_train, y_train)
rfr_y_pred = rfr.predict(x_test)
rfr_f1 = f1_score(y_test, rfr_y_pred, average="weighted")
print(f"Random Forest f1 score is: {rfr_f1}")

# %% Suport Vector
sv = SVC(C=10, gamma=0.0001, kernel="linear")
sv.fit(x_train, y_train)
sv_y_pred = sv.predict(x_test)
sv_f1 = f1_score(y_test, sv_y_pred, average="weighted")
print(f"Random Forest f1 score is: {sv_f1}")

# %% Cross Validation
cv = 10
lr_score = cross_val_score(lr, x, y, scoring="f1_weighted", cv=cv)
dt_score = cross_val_score(dt, x, y, scoring="f1_weighted", cv=cv)
rfr_score = cross_val_score(rfr, x, y, scoring="f1_weighted", cv=cv)
sv_score = cross_val_score(sv, x, y, scoring="f1_weighted", cv=cv)

# %% Print the scores
print(f"Cross validation Scores: LR {lr_score.mean()}, DT: {dt_score.mean()}, RFR: {rfr_score.mean()}, SV: {sv_score.mean()}")